"""Build equal-width shell features over 0–30 Å.

Reuse the converted structure paths in ``staging/inputs.dat``. Coarser shells
sum adjacent 0.5 Å shells so contact totals stay unchanged across partitions.
Each NPZ records its shell boundaries.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = ROOT / "features_residue" / "staging" / "inputs.dat"
N_PAIRS = 168
FINE_SHELLS = 60
MAX_RADIUS_ANGSTROM = 30.0


def _load_legacy_builder():
    """Load the canonical residue/ligand parsers without copying their logic."""
    path = Path(__file__).with_name("1_build_shell_features.py")
    spec = importlib.util.spec_from_file_location("_shell_feature_builder_v1", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load feature builder from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_LEGACY = _load_legacy_builder()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_shell_boundaries(
    n_shells: int,
    max_radius_angstrom: float = MAX_RADIUS_ANGSTROM,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lower edges, upper edges and centers for equal-width shells."""
    n_shells = int(n_shells)
    max_radius_angstrom = float(max_radius_angstrom)
    if n_shells <= 0:
        raise ValueError(f"n_shells must be positive, got {n_shells}")
    if not np.isfinite(max_radius_angstrom) or max_radius_angstrom <= 0:
        raise ValueError(
            "max_radius_angstrom must be finite and positive, got "
            f"{max_radius_angstrom}")
    edges = np.linspace(
        0.0, max_radius_angstrom, n_shells + 1, dtype=np.float64)
    return edges[:-1], edges[1:], (edges[:-1] + edges[1:]) * 0.5


def strict_onion_bin(
    pair_idx: np.ndarray,
    distances_nm: np.ndarray,
    *,
    n_shells: int = FINE_SHELLS,
    max_radius_angstrom: float = MAX_RADIUS_ANGSTROM,
) -> np.ndarray:
    """Bin contacts into ``[0,u0], (u0,u1], ...`` in shell-major order.

    ``distances_nm`` follows the canonical parser, which converts PDB Angstrom
    coordinates to nanometres.  Values beyond the common maximum radius are
    excluded rather than clipped into the final shell.
    """
    pair_idx = np.asarray(pair_idx, dtype=np.int64).reshape(-1)
    distances_nm = np.asarray(distances_nm, dtype=np.float64).reshape(-1)
    if pair_idx.shape != distances_nm.shape:
        raise ValueError(
            f"pair_idx/distances shape mismatch: {pair_idx.shape} vs "
            f"{distances_nm.shape}")
    if pair_idx.size and (pair_idx.min() < 0 or pair_idx.max() >= N_PAIRS):
        raise ValueError(f"pair indices must be in [0,{N_PAIRS - 1}]")
    if np.any(~np.isfinite(distances_nm)) or np.any(distances_nm < 0):
        raise ValueError("distances must be finite and non-negative")

    _, upper_a, _ = strict_shell_boundaries(
        n_shells, max_radius_angstrom=max_radius_angstrom)
    upper_nm = upper_a * 0.1
    shell_idx = np.searchsorted(upper_nm, distances_nm, side="left")
    keep = shell_idx < int(n_shells)
    flat_idx = shell_idx[keep] * N_PAIRS + pair_idx[keep]
    return np.bincount(
        flat_idx, minlength=int(n_shells) * N_PAIRS).astype(np.float32)


def aggregate_shell_major(
    fine_flat: np.ndarray,
    *,
    source_shells: int = FINE_SHELLS,
    target_shells: int,
) -> np.ndarray:
    """Exactly aggregate adjacent shells in a shell-major feature matrix."""
    source_shells = int(source_shells)
    target_shells = int(target_shells)
    if target_shells <= 0 or source_shells % target_shells:
        raise ValueError(
            f"target_shells={target_shells} must divide "
            f"source_shells={source_shells}")
    array = np.asarray(fine_flat)
    if array.shape[-1] != source_shells * N_PAIRS:
        raise ValueError(
            f"last dimension must be {source_shells * N_PAIRS}, got "
            f"{array.shape[-1]}")
    leading = array.shape[:-1]
    group = source_shells // target_shells
    table = array.reshape(*leading, target_shells, group, N_PAIRS)
    return table.sum(axis=-2, dtype=np.float32).reshape(
        *leading, target_shells * N_PAIRS)


def _worker(line: str):
    parts = line.split()
    if len(parts) < 3:
        return None
    code, rec_fp, lig_fp = parts[:3]
    try:
        residues, residue_xyz = _LEGACY.parse_protein(rec_fp)
        ligand_elements, ligand_xyz = _LEGACY.parse_ligand(lig_fp)
        if not residues or not ligand_elements:
            return code, None
        pairs, distances_nm = _LEGACY.res_atom_min_dist(
            residues, residue_xyz, ligand_elements, ligand_xyz)
        pair_idx = np.fromiter(
            (_LEGACY.KEY_INDEX[pair] for pair in pairs),
            dtype=np.int64,
            count=len(pairs),
        )
        return code, strict_onion_bin(pair_idx, distances_nm)
    except Exception as exc:  # noqa: BLE001 - preserve per-code audit trail
        return code, f"ERR:{type(exc).__name__}:{exc}"


def _metadata(n_shells: int, source_inputs_sha256: str) -> Dict[str, np.ndarray]:
    lower, upper, centers = strict_shell_boundaries(n_shells)
    width = MAX_RADIUS_ANGSTROM / float(n_shells)
    return {
        "n_shells": np.asarray(n_shells, dtype=np.int64),
        "n_pairs": np.asarray(N_PAIRS, dtype=np.int64),
        "shell_lower_edges_angstrom": lower.astype(np.float32),
        "shell_upper_edges_angstrom": upper.astype(np.float32),
        "shell_centers_angstrom": centers.astype(np.float32),
        "shell_width_angstrom": np.asarray(width, dtype=np.float32),
        "max_radius_angstrom": np.asarray(
            MAX_RADIUS_ANGSTROM, dtype=np.float32),
        "binning_convention": np.asarray(
            "[0, upper_0], then (lower_i, upper_i]"),
        "feature_layout": np.asarray("shell_major"),
        "partition_family": np.asarray("strict_equal_width_0_30A"),
        "fine_source_n_shells": np.asarray(FINE_SHELLS, dtype=np.int64),
        "aggregation_factor": np.asarray(
            FINE_SHELLS // int(n_shells), dtype=np.int64),
        "source_inputs_sha256": np.asarray(source_inputs_sha256),
    }


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    try:
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _read_lines(path: Path, limit: int) -> List[str]:
    with path.open(encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if limit > 0:
        lines = lines[:limit]
    if not lines:
        raise ValueError(f"no valid rows in {path}")
    return lines


def _validate_partitions(partitions: Iterable[int]) -> List[int]:
    values = list(dict.fromkeys(int(value) for value in partitions))
    if FINE_SHELLS not in values:
        values.insert(0, FINE_SHELLS)
    bad = [value for value in values if value <= 0 or FINE_SHELLS % value]
    if bad:
        raise ValueError(
            f"all partitions must divide N={FINE_SHELLS}, got {bad}")
    return values


def verify_outputs(out_dir: Path, partitions: Sequence[int]) -> dict:
    """Validate metadata and exact count conservation among saved stores."""
    stores = {}
    codes_ref = None
    for n_shells in partitions:
        path = out_dir / f"residue_N{n_shells}_strict_0_30A.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as source:
            required = {
                "codes", "feats", "n_shells", "n_pairs",
                "shell_lower_edges_angstrom", "shell_upper_edges_angstrom",
                "shell_centers_angstrom", "shell_width_angstrom",
                "max_radius_angstrom", "feature_layout", "partition_family",
            }
            missing = required.difference(source.files)
            if missing:
                raise ValueError(f"{path}: missing metadata {sorted(missing)}")
            codes = source["codes"].astype(str)
            feats = source["feats"].astype(np.float32, copy=False)
            if feats.shape != (len(codes), int(n_shells) * N_PAIRS):
                raise ValueError(f"{path}: inconsistent shape {feats.shape}")
            lower, upper, centers = strict_shell_boundaries(n_shells)
            if not np.allclose(source["shell_lower_edges_angstrom"], lower):
                raise ValueError(f"{path}: lower boundaries mismatch")
            if not np.allclose(source["shell_upper_edges_angstrom"], upper):
                raise ValueError(f"{path}: upper boundaries mismatch")
            if not np.allclose(source["shell_centers_angstrom"], centers):
                raise ValueError(f"{path}: centers mismatch")
            if str(np.asarray(source["feature_layout"]).item()) != "shell_major":
                raise ValueError(f"{path}: feature_layout is not shell_major")
        if codes_ref is None:
            codes_ref = codes
        elif not np.array_equal(codes, codes_ref):
            raise ValueError(f"{path}: code order differs from N={partitions[0]}")
        stores[int(n_shells)] = feats

    fine = stores[FINE_SHELLS]
    for n_shells, coarse in stores.items():
        expected = aggregate_shell_major(
            fine, source_shells=FINE_SHELLS, target_shells=n_shells)
        if not np.array_equal(coarse, expected):
            mismatch = int(np.count_nonzero(coarse != expected))
            raise ValueError(
                f"N={n_shells}: {mismatch} cells violate exact aggregation")
    return {
        "rows": int(fine.shape[0]),
        "partitions": list(map(int, partitions)),
        "fine_total_contacts": int(fine.sum(dtype=np.float64)),
        "count_conservation": "exact",
    }


def build(
    *,
    inputs: Path,
    out_dir: Path,
    partitions: Sequence[int],
    nproc: int,
    limit: int,
    overwrite: bool,
) -> dict:
    partitions = _validate_partitions(partitions)
    lines = _read_lines(inputs, limit)
    paths = {
        n: out_dir / f"residue_N{n}_strict_0_30A.npz"
        for n in partitions
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "output(s) already exist; use --overwrite: " + ", ".join(existing))

    source_hash = _sha256_file(inputs)
    codes: List[str] = []
    rows: List[np.ndarray] = []
    failures: List[str] = []
    nproc = max(1, int(nproc))
    print(
        f"[strict-shells] rows={len(lines)} nproc={nproc} "
        f"partitions={partitions}", flush=True)
    if nproc == 1:
        iterator = map(_worker, lines)
        pool = None
    else:
        # Fork inherits the dynamically loaded structure parser.
        pool = mp.get_context("fork").Pool(nproc)
        iterator = pool.imap(_worker, lines, chunksize=8)
    try:
        for done, result in enumerate(iterator, start=1):
            if result is None:
                failures.append(f"line-{done}\tERR:malformed input row")
                continue
            code, payload = result
            if isinstance(payload, np.ndarray):
                codes.append(str(code).lower())
                rows.append(payload)
            else:
                failures.append(f"{code}\t{payload}")
            if done % 1000 == 0:
                print(
                    f"[strict-shells] {done}/{len(lines)} ok={len(rows)} "
                    f"fail={len(failures)}", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    if not rows:
        raise RuntimeError("no features were generated")

    fine = np.stack(rows, axis=0).astype(np.float32, copy=False)
    code_array = np.asarray(codes, dtype="U16")
    if len(set(codes)) != len(codes):
        raise ValueError("duplicate PDB codes in generated rows")
    for n_shells in partitions:
        feats = (
            fine if n_shells == FINE_SHELLS
            else aggregate_shell_major(
                fine, source_shells=FINE_SHELLS, target_shells=n_shells)
        )
        _atomic_save_npz(
            paths[n_shells], codes=code_array, feats=feats,
            **_metadata(n_shells, source_hash))
        print(
            f"[strict-shells] wrote {paths[n_shells]} shape={feats.shape}",
            flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    failure_path = out_dir / "generation_failures.tsv"
    failure_path.write_text(
        "\n".join(failures) + ("\n" if failures else ""), encoding="utf-8")
    verification = verify_outputs(out_dir, partitions)
    manifest = {
        "schema_version": 1,
        "partition_family": "strict_equal_width_0_30A",
        "source_inputs": str(inputs.resolve()),
        "source_inputs_sha256": source_hash,
        "requested_rows": len(lines),
        "generated_rows": len(rows),
        "failed_rows": len(failures),
        "n_pairs": N_PAIRS,
        "fine_n_shells": FINE_SHELLS,
        "max_radius_angstrom": MAX_RADIUS_ANGSTROM,
        "partitions": [
            {
                "n_shells": n,
                "width_angstrom": MAX_RADIUS_ANGSTROM / n,
                "path": str(paths[n].resolve()),
                "sha256": _sha256_file(paths[n]),
            }
            for n in partitions
        ],
        "verification": verification,
    }
    manifest_path = out_dir / "strict_shell_partitions.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"[strict-shells] verified: {verification}", flush=True)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "features_residue" / "strict_0_30A")
    parser.add_argument(
        "--partitions", nargs="+", type=int, default=[60, 30, 20, 10])
    parser.add_argument(
        "--nproc", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    partitions = _validate_partitions(args.partitions)
    if args.verify_only:
        report = verify_outputs(args.out_dir.resolve(), partitions)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    build(
        inputs=args.inputs.resolve(),
        out_dir=args.out_dir.resolve(),
        partitions=partitions,
        nproc=args.nproc,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
