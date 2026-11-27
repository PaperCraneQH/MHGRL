"""Per-sample prediction tables and repeat-level hard-sample aggregation."""
from __future__ import annotations

import csv
import fcntl
import math
import os
from collections import defaultdict
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


CORE_FIELDS = (
    "test_set",
    "repeat",
    "sample_index",
    "pdb_code",
    "ground_truth",
    "prediction",
    "residual",
    "abs_error",
    "squared_error",
)


def _as_list(values: Iterable) -> List:
    if hasattr(values, "tolist"):
        values = values.tolist()
    return list(values)


def write_prediction_table(
    path: str,
    test_set: str,
    repeat: int,
    pdb_codes: Sequence[str],
    ground_truth: Iterable[float],
    predictions: Iterable[float],
    extra_columns: Optional[Mapping[str, Sequence]] = None,
) -> str:
    """Write one row per test sample, preserving DataLoader order and PDB ID."""
    truth = [float(x) for x in _as_list(ground_truth)]
    pred = [float(x) for x in _as_list(predictions)]
    codes = [str(x or "").lower().strip() for x in pdb_codes]
    n = len(truth)
    if len(pred) != n or len(codes) != n:
        raise ValueError(
            f"prediction table length mismatch: truth={n}, pred={len(pred)}, "
            f"pdb_codes={len(codes)}"
        )

    extras: Dict[str, List] = {}
    for key, values in (extra_columns or {}).items():
        vals = _as_list(values)
        if len(vals) != n:
            raise ValueError(f"extra column {key!r} has {len(vals)} rows; expected {n}")
        extras[str(key)] = vals

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = list(CORE_FIELDS) + list(extras)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, (code, y, yhat) in enumerate(zip(codes, truth, pred)):
            residual = yhat - y
            row = {
                "test_set": test_set,
                "repeat": int(repeat),
                "sample_index": i,
                "pdb_code": code or f"sample_{i:04d}",
                "ground_truth": f"{y:.8f}",
                "prediction": f"{yhat:.8f}",
                "residual": f"{residual:.8f}",
                "abs_error": f"{abs(residual):.8f}",
                "squared_error": f"{residual * residual:.8f}",
            }
            for key, values in extras.items():
                row[key] = values[i]
            writer.writerow(row)
    return path


def aggregate_prediction_tables(paths: Sequence[str], output_path: str) -> List[dict]:
    """Aggregate matched samples across repeats and rank persistent failures.

    ``persistent_hard`` is true when either:
      * |error| >= 1 pK in at least 80% of available repeats and MAE >= 1 pK; or
      * |error| >= 2 pK in at least 60% of available repeats.

    The thresholds are deliberately explicit and are diagnostics, not a statistical test.
    """
    grouped: Dict[tuple, List[dict]] = defaultdict(list)
    extra_fields: List[str] = []
    for path in paths:
        with open(path, encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for field in reader.fieldnames or []:
                if field not in CORE_FIELDS and field not in extra_fields:
                    extra_fields.append(field)
            for row in reader:
                key = (row.get("test_set", ""), row.get("pdb_code", ""))
                grouped[key].append(row)

    rows: List[dict] = []
    for (test_set, pdb_code), items in grouped.items():
        truth = [float(x["ground_truth"]) for x in items]
        pred = [float(x["prediction"]) for x in items]
        residual = [p - y for p, y in zip(pred, truth)]
        abs_error = [abs(x) for x in residual]
        n = len(items)
        n_ge_1 = sum(x >= 1.0 for x in abs_error)
        n_ge_2 = sum(x >= 2.0 for x in abs_error)
        frac_ge_1 = n_ge_1 / n
        frac_ge_2 = n_ge_2 / n
        mae = mean(abs_error)
        persistent = (mae >= 1.0 and frac_ge_1 >= 0.8) or frac_ge_2 >= 0.6
        row = {
            "test_set": test_set,
            "pdb_code": pdb_code,
            "ground_truth": mean(truth),
            "ground_truth_std": pstdev(truth) if n > 1 else 0.0,
            "n_repeats": n,
            "mean_prediction": mean(pred),
            "prediction_std": pstdev(pred) if n > 1 else 0.0,
            "mean_residual": mean(residual),
            "residual_std": pstdev(residual) if n > 1 else 0.0,
            "mean_abs_error": mae,
            "abs_error_std": pstdev(abs_error) if n > 1 else 0.0,
            "rmse": math.sqrt(mean([x * x for x in residual])),
            "mean_squared_error": mean([x * x for x in residual]),
            "squared_error_std": (
                pstdev([x * x for x in residual]) if n > 1 else 0.0),
            "max_abs_error": max(abs_error),
            "abs_error_ge_1_count": n_ge_1,
            "abs_error_ge_2_count": n_ge_2,
            "bad_fraction_ge_1": frac_ge_1,
            "bad_fraction_ge_2": frac_ge_2,
            "error_direction": "over" if mean(residual) > 0 else "under",
            "persistent_hard": persistent,
        }
        first = items[0]
        for field in extra_fields:
            values = [item.get(field, "") for item in items]
            # Average directional predictions across repeats as well.
            if field in (
                    "prediction_to_shell", "prediction_from_shell",
                    "direction_gap"):
                numeric = [float(value) for value in values if value != ""]
                row[field] = mean(numeric) if numeric else ""
                row[f"{field}_std"] = (
                    pstdev(numeric) if len(numeric) > 1 else 0.0)
            else:
                row[field] = first.get(field, "")
        rows.append(row)

    rows.sort(key=lambda x: (-float(x["mean_abs_error"]), x["pdb_code"]))
    for rank, row in enumerate(rows, 1):
        row["error_rank"] = rank

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fields = [
        "error_rank",
        "test_set",
        "pdb_code",
        "ground_truth",
        "ground_truth_std",
        "n_repeats",
        "mean_prediction",
        "prediction_std",
        "mean_residual",
        "residual_std",
        "mean_abs_error",
        "abs_error_std",
        "rmse",
        "mean_squared_error",
        "squared_error_std",
        "max_abs_error",
        "abs_error_ge_1_count",
        "abs_error_ge_2_count",
        "bad_fraction_ge_1",
        "bad_fraction_ge_2",
        "error_direction",
        "persistent_hard",
    ]
    for field in extra_fields:
        fields.append(field)
        if field in (
                "prediction_to_shell", "prediction_from_shell",
                "direction_gap"):
            fields.append(f"{field}_std")
    lock_path = output_path + ".lock"
    with open(lock_path, "a", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        try:
            temporary = output_path + ".tmp"
            with open(temporary, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            os.replace(temporary, output_path)
        finally:
            fcntl.flock(lock_handle, fcntl.LOCK_UN)
    return rows
