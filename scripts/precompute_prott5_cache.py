#!/usr/bin/env python3
"""Precompute mean-pooled ProtT5 protein features in HDF5 format."""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = "Rostlab/prot_t5_xl_half_uniref50-enc"


def seq_key(seq: str) -> str:
    return hashlib.sha1(seq.encode("utf-8")).hexdigest()[:16]


def collect_sequences(data_root: Path, splits: List[str]) -> List[str]:
    from data import PDBbindESMDataset
    seen: Dict[str, None] = {}
    for split_dir in splits:
        d = data_root / split_dir
        if not d.is_dir():
            continue
        for pt in sorted(d.glob("*.pt")):
            name = f"{split_dir}/{pt.stem}"
            try:
                store = PDBbindESMDataset(root=str(data_root), dataset=name)._data
            except Exception as exc:  # noqa: BLE001
                print(f"  [skip] {name}: {exc!r}")
                continue
            seqs = getattr(store, "seq", None)
            if seqs is None:
                print(f"  [skip] {name}: no .seq attribute")
                continue
            for s in seqs:
                seen.setdefault(str(s), None)
            print(f"  {name:34s} cumulative unique = {len(seen)}")
    return list(seen)


def build(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer, T5EncoderModel

    data_root = ROOT / args.data_root
    out_path = Path(args.out) if args.out else (
        data_root / "esm_cache"
        / f"prott5_u50_mlen{args.max_len}_{args.pool}pool.h5")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.overwrite:
        print(f"{out_path} exists; pass --overwrite to rebuild")
        return 1

    print("collecting sequences ...")
    seqs = collect_sequences(data_root, args.splits)
    lengths = np.array([len(s.replace(" ", "")) for s in seqs])
    print(f"unique sequences: {len(seqs)}  "
          f"median={np.median(lengths):.0f} max={lengths.max()}  "
          f"> max_len({args.max_len}): {(lengths > args.max_len).sum()}")

    os.environ.setdefault("HF_HOME", str(ROOT / args.hf_cache_dir))
    hf_cache = str(ROOT / args.hf_cache_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"loading {args.model} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=hf_cache, use_fast=False)
    model = T5EncoderModel.from_pretrained(args.model, cache_dir=hf_cache)
    model = model.to(device).eval().half()
    hidden = int(model.config.d_model)
    for p in model.parameters():
        p.requires_grad = False
    print(f"hidden={hidden}")

    # Encode the longest sequences first so memory failures surface early.
    order = np.argsort(-lengths)
    budget = int(args.batch_tokens)

    written = 0
    t0 = time.time()
    with h5py.File(out_path, "w") as f:
        f.attrs["model_name"] = args.model
        f.attrs["hidden"] = hidden
        f.attrs["pool"] = args.pool
        i = 0
        while i < len(order):
            first = int(order[i])
            longest = min(int(lengths[first]) + 1, args.max_len)
            n_max = max(int(budget // max(longest, 1)), 1)
            idx = [int(x) for x in order[i:i + n_max]]
            i += len(idx)
            batch = [seqs[j] for j in idx]
            spaced = [" ".join(list(s.replace(" ", ""))) or "A" for s in batch]
            tok = tokenizer(spaced, padding=True, truncation=True,
                            max_length=args.max_len, return_tensors="pt",
                            return_special_tokens_mask=args.pool == "residue_mean")
            ids = tok["input_ids"].to(device)
            mask = tok["attention_mask"].to(device)
            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=mask)
                h = out.last_hidden_state.float()
            if args.pool == "residue_mean":
                special = tok["special_tokens_mask"].to(device)
                pool_mask = mask.bool() & ~special.bool()
            else:
                pool_mask = mask
            m = pool_mask.float().unsqueeze(-1)
            if args.pool in ("mean", "residue_mean"):
                pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
            elif args.pool == "max":
                pooled = h.masked_fill(
                    ~mask.bool().unsqueeze(-1),
                    torch.finfo(h.dtype).min).max(1).values
            else:
                pooled = h[:, 0]
            arr = pooled.cpu().numpy().astype(np.float16)
            for row, seq in enumerate(batch):
                key = seq_key(seq)
                if key in f:
                    continue
                g = f.create_group(key)
                g.create_dataset("emb", data=arr[row], dtype="f2")
                written += 1
            if written and (written % 500 < len(batch)):
                done = written / len(seqs)
                eta = (time.time() - t0) * (1 - done) / max(done, 1e-6)
                print(f"  {written}/{len(seqs)}  tokens={ids.shape[1]}  "
                      f"elapsed={time.time() - t0:.0f}s  eta={eta:.0f}s",
                      flush=True)
            del ids, mask, out, h, pooled
            if device.type == "cuda":
                torch.cuda.empty_cache()
        f.attrs["num_smiles"] = written

    print(f"wrote {written} sequences to {out_path} "
          f"in {time.time() - t0:.0f}s "
          f"({out_path.stat().st_size / 1024**2:.1f} MB)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, required=True)
    ap.add_argument("--pool", default="mean", choices=["mean", "residue_mean", "max", "cls"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-root", default="data_processed_esm")
    ap.add_argument("--hf-cache-dir", default="hf_cache")
    ap.add_argument("--splits", nargs="*", default=[
        "base", "external", "holdout", "random", "scaffold", "seq_identity"])
    ap.add_argument("--batch-tokens", type=int, default=8192,
                    help="padded token budget per batch")
    ap.add_argument("--out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    return build(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
