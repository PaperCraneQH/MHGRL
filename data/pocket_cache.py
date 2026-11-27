"""Read per-residue pocket embeddings from HDF5 ``/<pdb_code>/emb`` datasets.

Keys are lowercase PDB codes, not protein sequence hashes.
"""
from __future__ import annotations

import os
import threading
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

_IN_MEM: Dict[str, Dict[str, np.ndarray]] = {}


def _load_all(path: str) -> Dict[str, np.ndarray]:
    import time
    t0 = time.time()
    store: Dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for code in f.keys():
            store[code] = f[code]["emb"][...]
    mb = sum(a.nbytes for a in store.values()) / (1024 ** 2)
    print(f"[PocketCache][in_memory] loaded {len(store):,} pockets "
          f"= {mb:.1f} MB fp16 from {os.path.basename(path)} "
          f"in {time.time() - t0:.1f}s", flush=True)
    return store


class PocketResidueCache:
    """Per-residue pocket embeddings for a batch of PDB codes."""

    def __init__(self, h5_path: str, dtype: torch.dtype = torch.float32,
                 in_memory: bool = True, missing_policy: str = "zero"):
        if not os.path.isfile(h5_path):
            raise FileNotFoundError(f"pocket cache not found: {h5_path}")
        self.h5_path = os.path.abspath(h5_path)
        self._dtype = dtype
        self.in_memory = bool(in_memory)
        self.missing_policy = missing_policy
        with h5py.File(self.h5_path, "r") as f:
            self.model_name = str(f.attrs.get("model_name", ""))
            self.hidden = int(f.attrs.get("hidden", 0))
            self.max_res = int(f.attrs.get("max_res", 0))
            self.num_codes = int(f.attrs.get("num_codes", len(f.keys())))
        if self.in_memory:
            if self.h5_path not in _IN_MEM:
                _IN_MEM[self.h5_path] = _load_all(self.h5_path)
            self._mem = _IN_MEM[self.h5_path]
        self._local = threading.local()
        self.n_lookups = 0
        self.n_missing = 0

    def _handle(self) -> h5py.File:
        h = getattr(self._local, "handle", None)
        pid = getattr(self._local, "pid", None)
        if h is None or pid != os.getpid():
            h = h5py.File(self.h5_path, "r")
            self._local.handle = h
            self._local.pid = os.getpid()
        return h

    def __contains__(self, code: str) -> bool:
        key = str(code or "").lower().strip()
        if self.in_memory:
            return key in self._mem
        return key in self._handle()

    def __len__(self) -> int:
        return len(self._mem) if self.in_memory else len(self._handle().keys())

    def _get_one(self, code: str) -> Optional[np.ndarray]:
        key = str(code or "").lower().strip()
        self.n_lookups += 1
        if self.in_memory:
            arr = self._mem.get(key)
        else:
            f = self._handle()
            arr = f[key]["emb"][...] if key in f else None
        if arr is None:
            self.n_missing += 1
            if self.missing_policy == "error":
                raise KeyError(f"pocket not in cache: {code!r}")
        return arr

    def lookup_batch(
        self, pdb_codes: List[str],
        device: Optional[torch.device] = None,
        max_res: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(emb [B, R, H], mask [B, R])`` padded to the batch maximum."""
        arrays = [self._get_one(c) for c in pdb_codes]
        lengths = [0 if a is None else int(a.shape[0]) for a in arrays]
        cap = int(max_res) if max_res else (self.max_res or max(lengths + [1]))
        R = max(min(max(lengths + [1]), cap), 1)
        B = len(arrays)
        emb = np.zeros((B, R, self.hidden), dtype=np.float32)
        mask = np.zeros((B, R), dtype=bool)
        for i, arr in enumerate(arrays):
            if arr is None or arr.shape[0] == 0:
                # Unmask one zero-valued slot to avoid NaNs in attention.
                mask[i, 0] = True
                continue
            n = min(int(arr.shape[0]), R)
            emb[i, :n] = arr[:n].astype(np.float32, copy=False)
            mask[i, :n] = True
        t = torch.from_numpy(emb).to(self._dtype)
        m = torch.from_numpy(mask)
        if device is not None:
            t = t.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)
        return t, m

    def __repr__(self) -> str:
        return (f"PocketResidueCache(path={os.path.basename(self.h5_path)}, "
                f"hidden={self.hidden}, max_res={self.max_res}, "
                f"n_codes={self.num_codes})")
