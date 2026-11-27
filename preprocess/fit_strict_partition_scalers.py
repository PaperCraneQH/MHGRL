"""Fit scalers on the processed training data for each strict shell partition.

Artifact names encode the partition family and shell count from store metadata.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess import fit_split_scalers as protocol  # noqa: E402


def _scalar(source: np.lib.npyio.NpzFile, key: str):
    value = np.asarray(source[key]).reshape(-1)[0]
    return value.item() if hasattr(value, "item") else value


def _load_feature_store(path: Path) -> dict:
    sha256 = protocol._sha256_file(path)
    with np.load(path, allow_pickle=False) as source:
        required = {
            "codes", "feats", "n_shells", "n_pairs", "partition_family",
            "shell_lower_edges_angstrom", "shell_upper_edges_angstrom",
            "shell_centers_angstrom", "shell_width_angstrom",
            "max_radius_angstrom", "feature_layout",
        }
        missing = required.difference(source.files)
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        codes = [protocol._normalise_code(value) for value in source["codes"]]
        protocol._require_unique(codes, str(path))
        feats = np.ascontiguousarray(source["feats"].astype(np.float32, copy=False))
        n_shells = int(_scalar(source, "n_shells"))
        n_pairs = int(_scalar(source, "n_pairs"))
        family = str(_scalar(source, "partition_family"))
        layout = str(_scalar(source, "feature_layout"))
        lower = source["shell_lower_edges_angstrom"].astype(np.float32)
        upper = source["shell_upper_edges_angstrom"].astype(np.float32)
        centers = source["shell_centers_angstrom"].astype(np.float32)
        width = float(_scalar(source, "shell_width_angstrom"))
        max_radius = float(_scalar(source, "max_radius_angstrom"))
    if feats.shape != (len(codes), n_shells * n_pairs):
        raise ValueError(
            f"{path}: feats shape {feats.shape} != "
            f"({len(codes)}, {n_shells * n_pairs})")
    if family != "strict_equal_width_0_30A" or layout != "shell_major":
        raise ValueError(
            f"{path}: expected strict_equal_width_0_30A/shell_major, got "
            f"{family}/{layout}")
    return {
        "path": path,
        "sha256": sha256,
        "codes": codes,
        "index": {code: idx for idx, code in enumerate(codes)},
        "feats": feats,
        "n_shells": n_shells,
        "n_pairs": n_pairs,
        "family": family,
        "layout": layout,
        "lower": lower,
        "upper": upper,
        "centers": centers,
        "width": width,
        "max_radius": max_radius,
    }


def fit_mean_std_chunked(
    features: np.ndarray,
    indices: Sequence[int],
    *,
    chunk_size: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute population mean/std without materialising a full train subset."""
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        raise ValueError("cannot fit a scaler on an empty training population")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    total = np.zeros(features.shape[1], dtype=np.float64)
    total_sq = np.zeros(features.shape[1], dtype=np.float64)
    for start in range(0, idx.size, int(chunk_size)):
        chunk = features[idx[start:start + int(chunk_size)]].astype(
            np.float64, copy=False)
        total += chunk.sum(axis=0, dtype=np.float64)
        total_sq += np.square(chunk).sum(axis=0, dtype=np.float64)
    mean64 = total / float(idx.size)
    variance = np.maximum(total_sq / float(idx.size) - mean64 * mean64, 0.0)
    return mean64.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _artifact_paths(out_root: Path, store: dict, split: str, fold: int):
    n_shells = store["n_shells"]
    out_dir = out_root / f"N{n_shells}" / split
    suffix = "fixed" if split == "holdout" else f"fold{fold}"
    stem = f"residue_N{n_shells}_strict_0_30A_scaler_{split}_{suffix}"
    return out_dir / f"{stem}.npz", out_dir / f"{stem}.manifest.json"


def _verify(path: Path, store: dict, train_codes: Sequence[str]) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as source:
        required = {
            "mean", "std", "n_train", "n_features", "train_codes",
            "train_codes_sha256", "features_sha256", "n_shells", "n_pairs",
            "shell_width_angstrom", "max_radius_angstrom",
        }
        missing = required.difference(source.files)
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        mean = source["mean"]
        std = source["std"]
        codes = [protocol._normalise_code(value) for value in source["train_codes"]]
        n_train = int(_scalar(source, "n_train"))
        n_features = int(_scalar(source, "n_features"))
        feature_hash = str(_scalar(source, "features_sha256"))
        code_hash = str(_scalar(source, "train_codes_sha256"))
        n_shells = int(_scalar(source, "n_shells"))
    expected_features = store["feats"].shape[1]
    if mean.shape != (expected_features,) or std.shape != (expected_features,):
        raise ValueError(f"{path}: invalid mean/std shape")
    if n_train != len(train_codes) or n_features != expected_features:
        raise ValueError(f"{path}: n_train/n_features mismatch")
    if set(codes) != set(train_codes):
        raise ValueError(f"{path}: train population mismatch")
    if feature_hash != store["sha256"]:
        raise ValueError(f"{path}: source feature hash mismatch")
    if code_hash != protocol._sha256_codes(train_codes):
        raise ValueError(f"{path}: train code hash mismatch")
    if n_shells != store["n_shells"]:
        raise ValueError(f"{path}: n_shells mismatch")
    return {
        "scaler_sha256": protocol._sha256_file(path),
        "n_train": n_train,
        "n_features": n_features,
        "n_constant": int(np.count_nonzero(std < 1e-6)),
    }


def fit_all(args: argparse.Namespace) -> List[dict]:
    stores = [_load_feature_store(path.resolve()) for path in args.features]
    n_values = [store["n_shells"] for store in stores]
    if len(n_values) != len(set(n_values)):
        raise ValueError(f"duplicate n_shells feature stores: {n_values}")
    reference_codes = stores[0]["codes"]
    for store in stores[1:]:
        if store["codes"] != reference_codes:
            raise ValueError(
                f"{store['path']}: code order/population differs from "
                f"{stores[0]['path']}")

    splits = (
        list(protocol.SUPPORTED_SPLITS)
        if "all" in args.split else list(dict.fromkeys(args.split))
    )
    folds = sorted(set(args.fold))
    records: List[dict] = []
    for split in splits:
        split_folds = [0] if split == "holdout" else folds
        for fold in split_folds:
            paths = protocol._paths_for(
                split, fold, args.processed_root.resolve(), args.split_root.resolve())
            train_codes, eval_codes, protocol_record = protocol._load_protocol_codes(paths)
            leaked = set(train_codes).intersection(eval_codes)
            if leaked:
                raise ValueError(f"{split}/{fold}: train/eval leakage")
            for store in stores:
                missing = sorted(set(train_codes).difference(store["index"]))
                if missing:
                    raise ValueError(
                        f"{split}/{fold}/N{store['n_shells']}: "
                        f"{len(missing)} missing train codes; examples={missing[:10]}")
                out_path, manifest_path = _artifact_paths(
                    args.out_root.resolve(), store, split, fold)
                if args.verify_only:
                    verification = _verify(out_path, store, train_codes)
                    status = "verified"
                else:
                    if out_path.exists() and not args.overwrite:
                        raise FileExistsError(
                            f"{out_path} exists; use --overwrite or --verify-only")
                    indices = [store["index"][code] for code in train_codes]
                    mean, std = fit_mean_std_chunked(
                        store["feats"], indices, chunk_size=args.chunk_size)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        out_path,
                        mean=mean,
                        std=std,
                        n_train=np.asarray(len(train_codes), dtype=np.int64),
                        n_features=np.asarray(store["feats"].shape[1], dtype=np.int64),
                        train_codes=np.asarray(sorted(train_codes), dtype="U16"),
                        train_codes_sha256=np.asarray(
                            protocol._sha256_codes(train_codes)),
                        features_sha256=np.asarray(store["sha256"]),
                        features_path=np.asarray(str(store["path"].resolve())),
                        split=np.asarray(split),
                        fold=np.asarray(-1 if split == "holdout" else fold),
                        n_shells=np.asarray(store["n_shells"], dtype=np.int64),
                        n_pairs=np.asarray(store["n_pairs"], dtype=np.int64),
                        shell_lower_edges_angstrom=store["lower"],
                        shell_upper_edges_angstrom=store["upper"],
                        shell_centers_angstrom=store["centers"],
                        shell_width_angstrom=np.asarray(
                            store["width"], dtype=np.float32),
                        max_radius_angstrom=np.asarray(
                            store["max_radius"], dtype=np.float32),
                        feature_layout=np.asarray(store["layout"]),
                        partition_family=np.asarray(store["family"]),
                        fit_source=np.asarray("processed_train_pdb_codes"),
                    )
                    verification = _verify(out_path, store, train_codes)
                    manifest = {
                        "schema_version": 1,
                        "split": split,
                        "fold": "fixed" if split == "holdout" else fold,
                        "n_shells": store["n_shells"],
                        "shell_width_angstrom": store["width"],
                        "max_radius_angstrom": store["max_radius"],
                        "feature_path": str(store["path"].resolve()),
                        "feature_sha256": store["sha256"],
                        "scaler_path": str(out_path.resolve()),
                        "fit_population": (
                            "pdb_code values from the exact processed train .pt cache"),
                        "protocol": protocol_record,
                        "verification": verification,
                    }
                    manifest_path.write_text(
                        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
                    status = "written"
                    del mean, std, indices
                    gc.collect()
                record = {
                    "split": split,
                    "fold": "fixed" if split == "holdout" else fold,
                    "n_shells": store["n_shells"],
                    "shell_width_angstrom": store["width"],
                    "status": status,
                    "n_train": len(train_codes),
                    "n_features": store["feats"].shape[1],
                    "n_constant": verification["n_constant"],
                    "feature_path": str(store["path"].resolve()),
                    "scaler_path": str(out_path.resolve()),
                    "scaler_sha256": verification["scaler_sha256"],
                }
                records.append(record)
                print(
                    f"[{status}] {split}/{record['fold']} "
                    f"N={store['n_shells']} train={len(train_codes)} -> {out_path}",
                    flush=True)

    if not args.verify_only:
        index_path = args.out_root.resolve() / "strict_partition_scaler_index.csv"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame.from_records(records).to_csv(index_path, index=False)
        print(f"Index written: {index_path}", flush=True)
    return records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_dir = ROOT / "features_residue" / "strict_0_30A"
    parser.add_argument(
        "--features", type=Path, nargs="+",
        default=[
            default_dir / f"residue_N{n}_strict_0_30A.npz"
            for n in (60, 30, 20, 10)
        ])
    parser.add_argument(
        "--processed-root", type=Path, default=ROOT / "data_processed_esm")
    parser.add_argument(
        "--split-root", type=Path,
        default=ROOT / "data" / "splits" / "cold_start")
    parser.add_argument(
        "--out-root", type=Path,
        default=default_dir / "scalers" / "train_only")
    parser.add_argument(
        "--split", nargs="+", default=["all"],
        choices=("all",) + protocol.SUPPORTED_SPLITS)
    parser.add_argument("--fold", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.verify_only and args.overwrite:
        raise ValueError("--verify-only and --overwrite are mutually exclusive")
    if not args.fold or any(fold not in range(5) for fold in args.fold):
        raise ValueError(f"--fold must be a non-empty subset of 0..4, got {args.fold}")
    fit_all(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
