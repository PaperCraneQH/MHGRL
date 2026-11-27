#!/usr/bin/env python3
"""Build an HDF5 cache of pocket Cβ + ligand heavy-atom coordinates.

  /<pdb>/prot_xyz   float32 [P,3]   P <= 64
  /<pdb>/prot_type  int16   [P]     OnionNet 21-way residue id
  /<pdb>/lig_xyz    float32 [L,3]   L <= 48
  /<pdb>/lig_type   int16   [L]     H,C,O,N,P,S,Hal,DU
  /<pdb>/lig_bond   int16   [E,2]
  attrs: n_prot, n_lig, n_bond
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

THREE_TO_IDX = {
    "GLY": 0, "ALA": 1, "VAL": 2, "LEU": 3, "ILE": 4, "PRO": 5, "PHE": 6,
    "TYR": 7, "TRP": 8, "SER": 9, "THR": 10, "CYS": 11, "MET": 12, "ASN": 13,
    "GLN": 14, "ASP": 15, "GLU": 16, "LYS": 17, "ARG": 18, "HIS": 19,
    "MSE": 12, "SEC": 11, "PYL": 17, "HYP": 5,
}
OTH = 20
MAX_PROT = 64
MAX_LIG = 48
ELE_TO_IDX = {
    "H": 0, "C": 1, "O": 2, "N": 3, "P": 4, "S": 5,
    "F": 6, "CL": 6, "BR": 6, "I": 6,
}


def wanted_codes(splits_dir: Path) -> List[str]:
    codes: Dict[str, None] = {}
    for path in splits_dir.rglob("*.csv"):
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or "PDB_code" not in reader.fieldnames:
                    continue
                for row in reader:
                    codes.setdefault(row["PDB_code"].lower().strip(), None)
        except Exception:  # noqa: BLE001
            continue
    return list(codes)


def index_complexes(roots: List[str]) -> Dict[str, Tuple[str, str]]:
    """code -> (pocket_pdb, ligand_path)."""
    pockets: Dict[str, str] = {}
    ligands: Dict[str, str] = {}
    for root in roots:
        if not os.path.isdir(root):
            continue
        for path in glob.glob(f"{root}/**/*pocket.pdb", recursive=True):
            name = Path(path).name.lower()
            code = name.split("_")[0]
            # Prefer official *_pocket.pdb over *_valid_pocket.pdb.
            if code in pockets and "valid_pocket" not in pockets[code]:
                continue
            pockets[code] = path
        for ext in ("sdf", "mol2", "pdb"):
            for path in glob.glob(f"{root}/**/*_ligand.{ext}", recursive=True):
                name = Path(path).name.lower()
                if "opt" in name:
                    continue
                code = name.split("_")[0]
                ligands.setdefault(code, path)
    out = {}
    for code, pocket in pockets.items():
        lig = ligands.get(code)
        if lig is None:
            folder = Path(pocket).parent
            for ext in ("sdf", "mol2", "pdb"):
                cand = folder / f"{code}_ligand.{ext}"
                if cand.is_file():
                    lig = str(cand)
                    break
        if lig is not None:
            out[code] = (pocket, lig)
    return out


def parse_cb(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (xyz [P,3], types [P]) for unique pocket residues."""
    residues: Dict[tuple, dict] = {}
    with open(path, errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            resname = line[17:20].strip().upper()
            if resname in ("HOH", "WAT", "DOD"):
                continue
            atom = line[12:16].strip().upper()
            key = (line[21], line[22:26].strip(), line[26])
            try:
                xyz = np.array([
                    float(line[30:38]), float(line[38:46]), float(line[46:54])
                ], dtype=np.float32)
            except ValueError:
                continue
            rec = residues.setdefault(key, {
                "res": resname, "ca": None, "cb": None})
            if atom == "CA":
                rec["ca"] = xyz
            elif atom == "CB":
                rec["cb"] = xyz
    xyzs, types = [], []
    for rec in residues.values():
        pos = rec["cb"] if rec["cb"] is not None else rec["ca"]
        if pos is None:
            continue
        xyzs.append(pos)
        types.append(THREE_TO_IDX.get(rec["res"], OTH))
    if not xyzs:
        return np.zeros((0, 3), np.float32), np.zeros((0,), np.int16)
    return np.stack(xyzs, 0), np.asarray(types, dtype=np.int16)


def parse_ligand(path: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    from rdkit import Chem
    mol = None
    ext = Path(path).suffix.lower()
    if ext == ".sdf":
        mol = Chem.MolFromMolFile(path, removeHs=True, sanitize=False)
    elif ext == ".mol2":
        for sanitize in (True, False):
            mol = Chem.MolFromMol2File(path, removeHs=True, sanitize=sanitize)
            if mol is not None:
                break
    elif ext == ".pdb":
        mol = Chem.MolFromPDBFile(path, removeHs=True, sanitize=False)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    conf = mol.GetConformer() if mol.GetNumConformers() else None
    if conf is None:
        return None
    xyz, types = [], []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        xyz.append([pos.x, pos.y, pos.z])
        types.append(ELE_TO_IDX.get(atom.GetSymbol().upper(), 7))
    bonds = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
    return (
        np.asarray(xyz, dtype=np.float32),
        np.asarray(types, dtype=np.int16),
        np.asarray(bonds, dtype=np.int16).reshape(-1, 2),
    )


def trim(prot_xyz, prot_type, lig_xyz, lig_type, lig_bond):
    if lig_xyz.shape[0] > MAX_LIG:
        lig_xyz = lig_xyz[:MAX_LIG]
        lig_type = lig_type[:MAX_LIG]
        keep = (lig_bond[:, 0] < MAX_LIG) & (lig_bond[:, 1] < MAX_LIG)
        lig_bond = lig_bond[keep]
    if prot_xyz.shape[0] > MAX_PROT:
        center = lig_xyz.mean(axis=0, keepdims=True)
        order = np.linalg.norm(prot_xyz - center, axis=1).argsort()[:MAX_PROT]
        order.sort()
        prot_xyz = prot_xyz[order]
        prot_type = prot_type[order]
    return prot_xyz, prot_type, lig_xyz, lig_type, lig_bond


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(
        ROOT / "data_processed_esm" / "coarse_interface" / "cb_heavy.h5"))
    ap.add_argument("--structure-roots", nargs="+", required=True,
                    help="Directories containing the raw complex structures")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.overwrite:
        print(f"{out} exists; pass --overwrite to rebuild")
        return 1

    codes = wanted_codes(ROOT / "data" / "splits")
    index = index_complexes(args.structure_roots)
    print(f"split codes={len(codes)}  complexes with pocket+ligand={len(index)}")
    missing = [c for c in codes if c not in index]
    print(f"missing structures: {len(missing)}")

    n_ok = n_fail = 0
    tmp = out.with_suffix(".tmp.h5")
    if tmp.exists():
        tmp.unlink()
    with h5py.File(tmp, "w") as h5:
        for i, code in enumerate(codes):
            paths = index.get(code)
            if paths is None:
                n_fail += 1
                continue
            try:
                prot_xyz, prot_type = parse_cb(paths[0])
                lig = parse_ligand(paths[1])
                if lig is None or prot_xyz.shape[0] == 0 or lig[0].shape[0] == 0:
                    n_fail += 1
                    continue
                lig_xyz, lig_type, lig_bond = lig
                prot_xyz, prot_type, lig_xyz, lig_type, lig_bond = trim(
                    prot_xyz, prot_type, lig_xyz, lig_type, lig_bond)
                g = h5.create_group(code)
                g.create_dataset("prot_xyz", data=prot_xyz)
                g.create_dataset("prot_type", data=prot_type)
                g.create_dataset("lig_xyz", data=lig_xyz)
                g.create_dataset("lig_type", data=lig_type)
                g.create_dataset("lig_bond", data=lig_bond)
                g.attrs["n_prot"] = int(prot_xyz.shape[0])
                g.attrs["n_lig"] = int(lig_xyz.shape[0])
                g.attrs["n_bond"] = int(lig_bond.shape[0])
                n_ok += 1
            except Exception as exc:  # noqa: BLE001
                n_fail += 1
                if n_fail <= 8:
                    print(f"  fail {code}: {exc!r}")
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(codes)} ok={n_ok} fail={n_fail}", flush=True)
        h5.attrs["num_codes"] = n_ok
        h5.attrs["max_prot"] = MAX_PROT
        h5.attrs["max_lig"] = MAX_LIG
    tmp.replace(out)
    print(f"wrote {out}  ok={n_ok} fail={n_fail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
