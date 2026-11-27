"""In-memory HDF5 cache of pocket Cβ + ligand heavy-atom coordinates."""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch

_IN_MEM: Dict[str, dict] = {}


def _load_all(path: str) -> dict:
    store = {}
    with h5py.File(path, "r") as f:
        for code in f.keys():
            g = f[code]
            store[code] = {
                "prot_xyz": g["prot_xyz"][...],
                "prot_type": g["prot_type"][...],
                "lig_xyz": g["lig_xyz"][...],
                "lig_type": g["lig_type"][...],
                "lig_bond": g["lig_bond"][...],
            }
    return store


class CoarseInterfaceCache:
    def __init__(self, h5_path: str, max_prot: int = 64, max_lig: int = 48,
                 in_memory: bool = True):
        if not os.path.isfile(h5_path):
            raise FileNotFoundError(h5_path)
        self.h5_path = os.path.abspath(h5_path)
        self.max_prot = int(max_prot)
        self.max_lig = int(max_lig)
        if in_memory:
            if self.h5_path not in _IN_MEM:
                _IN_MEM[self.h5_path] = _load_all(self.h5_path)
                n = len(_IN_MEM[self.h5_path])
                print(f"[CoarseInterface] loaded {n} complexes from "
                      f"{os.path.basename(h5_path)}", flush=True)
            self._mem = _IN_MEM[self.h5_path]
        else:
            self._mem = None
            self._h5 = h5py.File(self.h5_path, "r")

    def _get(self, code: str) -> Optional[dict]:
        key = str(code or "").lower().strip()
        if self._mem is not None:
            return self._mem.get(key)
        if key not in self._h5:
            return None
        g = self._h5[key]
        return {
            "prot_xyz": g["prot_xyz"][...],
            "prot_type": g["prot_type"][...],
            "lig_xyz": g["lig_xyz"][...],
            "lig_type": g["lig_type"][...],
            "lig_bond": g["lig_bond"][...],
        }

    def lookup_batch(self, pdb_codes: List[str], device=None) -> dict:
        B = len(pdb_codes)
        P, L = self.max_prot, self.max_lig
        prot_xyz = np.zeros((B, P, 3), np.float32)
        prot_type = np.zeros((B, P), np.int64)
        prot_mask = np.zeros((B, P), np.bool_)
        lig_xyz = np.zeros((B, L, 3), np.float32)
        lig_type = np.zeros((B, L), np.int64)
        lig_mask = np.zeros((B, L), np.bool_)
        lig_bond = np.zeros((B, L * 3, 2), np.int64)
        bond_mask = np.zeros((B, L * 3), np.bool_)
        for i, code in enumerate(pdb_codes):
            rec = self._get(code)
            if rec is None:
                continue
            p = min(int(rec["prot_xyz"].shape[0]), P)
            q = min(int(rec["lig_xyz"].shape[0]), L)
            prot_xyz[i, :p] = rec["prot_xyz"][:p]
            prot_type[i, :p] = rec["prot_type"][:p]
            prot_mask[i, :p] = True
            lig_xyz[i, :q] = rec["lig_xyz"][:q]
            lig_type[i, :q] = rec["lig_type"][:q]
            lig_mask[i, :q] = True
            bonds = rec["lig_bond"]
            keep = (bonds[:, 0] < q) & (bonds[:, 1] < q)
            bonds = bonds[keep][: L * 3]
            e = int(bonds.shape[0])
            if e:
                lig_bond[i, :e] = bonds
                bond_mask[i, :e] = True

        def to_t(arr, dtype):
            t = torch.from_numpy(arr).to(dtype)
            return t.to(device) if device is not None else t

        return {
            "prot_xyz": to_t(prot_xyz, torch.float32),
            "prot_type": to_t(prot_type, torch.long),
            "prot_mask": to_t(prot_mask, torch.bool),
            "lig_xyz": to_t(lig_xyz, torch.float32),
            "lig_type": to_t(lig_type, torch.long),
            "lig_mask": to_t(lig_mask, torch.bool),
            "lig_bond": to_t(lig_bond, torch.long),
            "bond_mask": to_t(bond_mask, torch.bool),
        }
