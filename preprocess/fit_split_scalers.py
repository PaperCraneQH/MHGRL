"""Fit scalers on the processed training cache for each fold.

Use the PDB codes in the exact ``train_*.pt`` files consumed by training, not
all rows in the split CSVs. Some rows may fail preprocessing.

Write one scaler per folded split and one shared scaler for the fixed temporal
holdout. Scalers and adjacent manifests record the training codes and hashes.
Use ``--verify-only`` to check existing artifacts without refitting.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FOLDED_SPLITS = ("base", "random", "scaffold", "seq_identity")
SUPPORTED_SPLITS = FOLDED_SPLITS + ("holdout",)


def _normalise_code(value: object) -> str:
    return str(value).strip().lower()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_codes(codes: Iterable[str]) -> str:
    payload = "".join(f"{code}\n" for code in sorted(set(codes))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _csv_codes(path: Path) -> List[str]:
    df = pd.read_csv(path)
    column = "PDB_code" if "PDB_code" in df.columns else str(df.columns[0])
    return [_normalise_code(value) for value in df[column].tolist()]


def _pt_codes(path: Path) -> List[str]:
    # Import lazily so ``--help`` remains usable even outside the training env.
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError(
            "PyTorch is required to read the processed training caches. "
            "Run this script in the same environment as training."
        ) from exc

    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, (tuple, list)) or not obj:
        raise ValueError(f"unexpected PyG cache format: {path}")
    data = obj[0]
    if not hasattr(data, "pdb_code"):
        raise ValueError(f"processed cache has no pdb_code field: {path}")
    codes = [_normalise_code(value) for value in data.pdb_code]
    del obj, data
    gc.collect()
    return codes


def _paths_for(split: str, fold: int, processed_root: Path,
               split_root: Path) -> Dict[str, object]:
    if split == "holdout":
        return {
            "train_pt": processed_root / "holdout" / "train_holdout.pt",
            "train_csv": split_root / "holdout" / "train.csv",
            "eval_pt": [
                processed_root / "holdout" / "valid_holdout.pt",
                processed_root / "holdout" / "test_holdout.pt",
            ],
            "eval_csv": [
                split_root / "holdout" / "valid_holdout_2018.csv",
                split_root / "holdout" / "test_holdout_2019.csv",
            ],
        }

    eval_pt = [processed_root / split / f"valid_{split}_{fold}.pt"]
    eval_csv = [split_root / split / f"valid_{split}_{fold}.csv"]
    if split != "base":
        eval_pt.append(processed_root / split / f"test_{split}_{fold}.pt")
        eval_csv.append(split_root / split / f"test_{split}_{fold}.csv")
    else:
        eval_pt.extend([
            processed_root / "external" / "CASF-2016.pt",
            processed_root / "external" / "CSAR-HiQ.pt",
        ])
        eval_csv.extend([
            split_root.parent / "CASF-2016.csv",
            split_root.parent / "CSAR-HiQ_dedup.csv",
        ])
    return {
        "train_pt": processed_root / split / f"train_{split}_{fold}.pt",
        "train_csv": split_root / split / f"train_{split}_{fold}.csv",
        "eval_pt": eval_pt,
        "eval_csv": eval_csv,
    }


def _artifact_paths(out_root: Path, split: str, fold: int) -> Tuple[Path, Path]:
    out_dir = out_root / split
    stem = (
        "residue_N60_scaler_holdout_fixed"
        if split == "holdout"
        else f"residue_N60_scaler_{split}_fold{fold}"
    )
    return out_dir / f"{stem}.npz", out_dir / f"{stem}.manifest.json"


def _require_unique(codes: Sequence[str], label: str) -> None:
    if not codes:
        raise ValueError(f"{label}: no codes")
    n_unique = len(set(codes))
    if n_unique != len(codes):
        raise ValueError(f"{label}: {len(codes) - n_unique} duplicate PDB codes")


def _load_protocol_codes(paths: Dict[str, object]) -> Tuple[List[str], List[str], dict]:
    train_pt = Path(paths["train_pt"])
    train_csv = Path(paths["train_csv"])
    eval_pts = [Path(path) for path in paths["eval_pt"]]
    eval_csvs = [Path(path) for path in paths["eval_csv"]]
    for path in [train_pt, train_csv, *eval_pts, *eval_csvs]:
        if not path.is_file():
            raise FileNotFoundError(path)

    train_codes = _pt_codes(train_pt)
    declared_train_codes = _csv_codes(train_csv)
    _require_unique(train_codes, str(train_pt))
    _require_unique(declared_train_codes, str(train_csv))
    if not set(train_codes).issubset(set(declared_train_codes)):
        extra = sorted(set(train_codes) - set(declared_train_codes))[:10]
        raise ValueError(f"{train_pt}: codes absent from declared train CSV: {extra}")

    eval_codes: List[str] = []
    eval_rows = []
    for pt_path, csv_path in zip(eval_pts, eval_csvs):
        actual = _pt_codes(pt_path)
        declared = _csv_codes(csv_path)
        _require_unique(actual, str(pt_path))
        _require_unique(declared, str(csv_path))
        if not set(actual).issubset(set(declared)):
            extra = sorted(set(actual) - set(declared))[:10]
            raise ValueError(f"{pt_path}: codes absent from declared eval CSV: {extra}")
        eval_codes.extend(actual)
        eval_rows.append({
            "processed_path": str(pt_path.resolve()),
            "processed_sha256": _sha256_file(pt_path),
            "actual_rows": len(actual),
            "actual_code_sha256": _sha256_codes(actual),
            "declared_csv": str(csv_path.resolve()),
            "declared_csv_sha256": _sha256_file(csv_path),
            "declared_rows": len(declared),
            "dropped_before_cache": len(declared) - len(actual),
        })

    overlap = sorted(set(train_codes) & set(eval_codes))
    if overlap:
        raise ValueError(
            f"train/evaluation leakage detected ({len(overlap)} codes), "
            f"examples: {overlap[:10]}"
        )
    protocol = {
        "train_processed_path": str(train_pt.resolve()),
        "train_processed_sha256": _sha256_file(train_pt),
        "train_declared_csv": str(train_csv.resolve()),
        "train_declared_csv_sha256": _sha256_file(train_csv),
        "train_declared_rows": len(declared_train_codes),
        "train_actual_rows": len(train_codes),
        "train_dropped_before_cache": len(declared_train_codes) - len(train_codes),
        "evaluation_sets": eval_rows,
        "train_eval_overlap": 0,
    }
    return train_codes, eval_codes, protocol


def _read_scalar(value: np.ndarray) -> object:
    item = np.asarray(value).reshape(-1)[0]
    return item.item() if hasattr(item, "item") else item


def _verify_artifact(out_path: Path, expected_codes: Sequence[str],
                     expected_features: int, expected_features_sha256: str,
                     eval_codes: Sequence[str]) -> dict:
    if not out_path.is_file():
        raise FileNotFoundError(out_path)
    with np.load(out_path, allow_pickle=False) as saved:
        required = {"mean", "std", "n_train", "n_features", "train_codes",
                    "train_codes_sha256", "features_sha256"}
        missing = required - set(saved.files)
        if missing:
            raise ValueError(f"{out_path}: missing fields {sorted(missing)}")
        mean = saved["mean"]
        std = saved["std"]
        artifact_codes = [_normalise_code(code) for code in saved["train_codes"]]
        stored_code_hash = str(_read_scalar(saved["train_codes_sha256"]))
        stored_feature_hash = str(_read_scalar(saved["features_sha256"]))
        n_train = int(_read_scalar(saved["n_train"]))
        n_features = int(_read_scalar(saved["n_features"]))

    expected_set = set(expected_codes)
    artifact_set = set(artifact_codes)
    if artifact_set != expected_set:
        raise ValueError(
            f"{out_path}: stored train-code population differs from current cache "
            f"(missing={len(expected_set - artifact_set)}, extra={len(artifact_set - expected_set)})"
        )
    if n_train != len(expected_codes) or n_features != expected_features:
        raise ValueError(
            f"{out_path}: metadata mismatch n_train={n_train}, n_features={n_features}"
        )
    if mean.shape != (expected_features,) or std.shape != (expected_features,):
        raise ValueError(f"{out_path}: invalid mean/std shapes {mean.shape}/{std.shape}")
    if stored_code_hash != _sha256_codes(expected_codes):
        raise ValueError(f"{out_path}: train_codes_sha256 mismatch")
    if stored_feature_hash != expected_features_sha256:
        raise ValueError(f"{out_path}: source feature SHA-256 mismatch")
    leaked = artifact_set & set(eval_codes)
    if leaked:
        raise ValueError(f"{out_path}: {len(leaked)} eval codes stored in scaler")
    return {
        "artifact_sha256": _sha256_file(out_path),
        "n_train": n_train,
        "n_features": n_features,
        "n_constant": int(np.count_nonzero(std < 1e-6)),
        "train_eval_overlap": 0,
    }


def _fit_one(features: np.ndarray, feature_index: Dict[str, int],
             features_path: Path, feature_sha256: str, split: str, fold: int,
             processed_root: Path, split_root: Path, out_root: Path,
             overwrite: bool, verify_only: bool) -> dict:
    paths = _paths_for(split, fold, processed_root, split_root)
    train_codes, eval_codes, protocol = _load_protocol_codes(paths)
    missing_features = sorted(set(train_codes) - set(feature_index))
    if missing_features:
        raise ValueError(
            f"{split} fold {fold}: {len(missing_features)} train codes have no residue features; "
            f"examples: {missing_features[:10]}"
        )

    out_path, manifest_path = _artifact_paths(out_root, split, fold)
    if verify_only:
        verification = _verify_artifact(
            out_path, train_codes, features.shape[1], feature_sha256, eval_codes)
        status = "verified"
    else:
        if out_path.exists() and not overwrite:
            raise FileExistsError(
                f"{out_path} exists; pass --overwrite or use --verify-only"
            )
        indices = np.fromiter(
            (feature_index[code] for code in train_codes),
            dtype=np.int64,
            count=len(train_codes),
        )
        # Match the float32 statistics from 2_fit_scaler.py.
        subset = features[indices]
        mean = subset.mean(axis=0).astype(np.float32)
        std = subset.std(axis=0).astype(np.float32)
        del subset, indices
        gc.collect()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            mean=mean,
            std=std,
            n_train=np.int64(len(train_codes)),
            n_features=np.int64(features.shape[1]),
            train_codes=np.asarray(sorted(train_codes), dtype="U16"),
            train_codes_sha256=np.asarray(_sha256_codes(train_codes)),
            features_sha256=np.asarray(feature_sha256),
            split=np.asarray(split),
            fold=np.int64(-1 if split == "holdout" else fold),
            fit_source=np.asarray("processed_train_pdb_codes"),
        )
        verification = _verify_artifact(
            out_path, train_codes, features.shape[1], feature_sha256, eval_codes)
        status = "written"

    manifest = {
        "schema_version": 1,
        "split": split,
        "fold": "fixed" if split == "holdout" else fold,
        "holdout_fixed_partition": split == "holdout",
        "scaler_path": str(out_path.resolve()),
        "features_path": str(features_path.resolve()),
        "features_sha256": feature_sha256,
        "train_code_sha256": _sha256_codes(train_codes),
        "fit_population": "pdb_code values from the exact processed train .pt cache",
        "protocol": protocol,
        "verification": verification,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if not verify_only:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    elif manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("train_code_sha256") != manifest["train_code_sha256"]:
            raise ValueError(f"{manifest_path}: train code hash is stale")
        if prior.get("features_sha256") != feature_sha256:
            raise ValueError(f"{manifest_path}: source feature hash is stale")
        if prior.get("protocol") != protocol:
            raise ValueError(f"{manifest_path}: split/cache provenance is stale")
        prior_artifact_hash = (prior.get("verification") or {}).get("artifact_sha256")
        if prior_artifact_hash != verification["artifact_sha256"]:
            raise ValueError(f"{manifest_path}: scaler artifact hash is stale")
    elif verify_only:
        raise FileNotFoundError(manifest_path)

    return {
        "split": split,
        "fold": "fixed" if split == "holdout" else fold,
        "status": status,
        "n_train_csv": protocol["train_declared_rows"],
        "n_train_used": len(train_codes),
        "n_train_dropped_before_cache": protocol["train_dropped_before_cache"],
        "n_eval_actual": sum(row["actual_rows"] for row in protocol["evaluation_sets"]),
        "train_eval_overlap": 0,
        "n_features": features.shape[1],
        "n_constant": verification["n_constant"],
        "train_code_sha256": manifest["train_code_sha256"],
        "scaler_sha256": verification["artifact_sha256"],
        "scaler_path": str(out_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--features", type=Path,
        default=ROOT / "features_residue" / "residue_N60.npz")
    parser.add_argument(
        "--processed-root", type=Path,
        default=ROOT / "data_processed_esm")
    parser.add_argument(
        "--split-root", type=Path,
        default=ROOT / "data" / "splits" / "cold_start")
    parser.add_argument(
        "--out-root", type=Path,
        default=ROOT / "features_residue" / "scalers" / "train_only")
    parser.add_argument(
        "--split", nargs="+", default=["all"],
        choices=("all",) + SUPPORTED_SPLITS,
        help="one or more split names; 'all' fits every supported split")
    parser.add_argument(
        "--fold", nargs="+", type=int, default=list(range(5)),
        help="fold indices for folded splits (default: 0 1 2 3 4)")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace existing scaler artifacts")
    parser.add_argument(
        "--verify-only", action="store_true",
        help="validate existing scalers and manifests without fitting")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    splits = list(SUPPORTED_SPLITS) if "all" in args.split else list(dict.fromkeys(args.split))
    folds = sorted(set(args.fold))
    if not folds or any(fold not in range(5) for fold in folds):
        raise ValueError(f"--fold must be a non-empty subset of 0..4, got {folds}")
    if args.verify_only and args.overwrite:
        raise ValueError("--verify-only and --overwrite are mutually exclusive")

    features_path = args.features.resolve()
    print(f"Loading residue features: {features_path}", flush=True)
    feature_sha256 = _sha256_file(features_path)
    with np.load(features_path, allow_pickle=True) as source:
        feature_codes = [_normalise_code(value) for value in source["codes"]]
        _require_unique(feature_codes, str(features_path))
        features = np.ascontiguousarray(source["feats"].astype(np.float32, copy=False))
    feature_index = {code: idx for idx, code in enumerate(feature_codes)}
    print(
        f"Feature store: rows={len(feature_codes)}, features={features.shape[1]}, "
        f"sha256={feature_sha256[:12]}...",
        flush=True,
    )

    records = []
    for split in splits:
        split_folds = [0] if split == "holdout" else folds
        for fold in split_folds:
            record = _fit_one(
                features=features,
                feature_index=feature_index,
                features_path=features_path,
                feature_sha256=feature_sha256,
                split=split,
                fold=fold,
                processed_root=args.processed_root.resolve(),
                split_root=args.split_root.resolve(),
                out_root=args.out_root.resolve(),
                overwrite=args.overwrite,
                verify_only=args.verify_only,
            )
            records.append(record)
            print(
                f"[{record['status']}] {split} fold={record['fold']}: "
                f"train={record['n_train_used']} "
                f"(CSV={record['n_train_csv']}, dropped={record['n_train_dropped_before_cache']}), "
                f"eval_overlap=0, const={record['n_constant']} -> {record['scaler_path']}",
                flush=True,
            )

    index_path = args.out_root.resolve() / "scaler_index.csv"
    current = pd.DataFrame.from_records(records)
    if args.verify_only:
        print(f"Verified {len(current)} scaler artifacts; index not modified.", flush=True)
    else:
        # Preserve unselected rows when regenerating only a subset/fold.
        if index_path.is_file():
            prior = pd.read_csv(index_path, dtype={"fold": str})
            keys = {(str(row.split), str(row.fold)) for row in current.itertuples()}
            keep = [
                (str(row.split), str(row.fold)) not in keys
                for row in prior.itertuples()
            ]
            current = pd.concat([prior.loc[keep], current], ignore_index=True)
        current["_fold_order"] = current["fold"].map(
            lambda value: 5 if str(value) == "fixed" else int(value))
        current = current.sort_values(["split", "_fold_order"]).drop(columns="_fold_order")
        index_path.parent.mkdir(parents=True, exist_ok=True)
        current.to_csv(index_path, index=False)
        print(f"Index written: {index_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
