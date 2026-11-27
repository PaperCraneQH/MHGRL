"""Precompute pooled ChemBERTa ligand features in HDF5 format."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import PDBbindESMDataset
from data.chemberta_cache import smiles_hash


def collect_smiles(data_root: Path) -> list[str]:
    unique = {}
    for split in ("base", "random", "scaffold", "seq_identity", "holdout", "external"):
        for path in sorted((data_root / split).glob("*.pt")):
            dataset = PDBbindESMDataset(root=str(data_root), dataset=f"{split}/{path.stem}")
            for value in dataset._data.smiles:
                unique.setdefault(value, None)
    return list(unique)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="DeepChem/ChemBERTa-77M-MTR")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data_processed_esm")
    parser.add_argument("--hf-cache-dir", type=Path, default=ROOT / "hf_cache")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    output = args.out or (
        args.data_root / "chemberta_cache" / "chemberta77M_mtr_mlen512_clspool.h5"
    )
    if output.exists() and not args.overwrite:
        parser.error(f"{output} already exists; use --overwrite to rebuild it")
    smiles = collect_smiles(args.data_root)
    if not smiles:
        parser.error(f"No ligand records found in {args.data_root}; build the datasets first.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.h5")
    if temporary.exists():
        parser.error(f"Remove the incomplete file before retrying: {temporary}")

    from transformers import AutoModel, AutoTokenizer

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.hf_cache_dir)
    model = AutoModel.from_pretrained(args.model, cache_dir=args.hf_cache_dir).to(device).eval()
    with h5py.File(temporary, "x") as handle, torch.inference_mode():
        for start in range(0, len(smiles), args.batch_size):
            batch = smiles[start:start + args.batch_size]
            encoded = tokenizer([s or "C" for s in batch], truncation=True, padding=True,
                                max_length=512, return_tensors="pt")
            output_vectors = model(
                input_ids=encoded.input_ids.to(device),
                attention_mask=encoded.attention_mask.to(device),
            ).last_hidden_state[:, 0].float().cpu().numpy()
            for ligand, vector in zip(batch, output_vectors):
                handle.create_group(smiles_hash(ligand)).create_dataset(
                    "emb", data=vector.astype(np.float16))
        handle.attrs.update(
            format="pooled_fp16_v1",
            hidden=model.config.hidden_size,
            max_len_input=512,
            model_name=args.model,
            num_smiles=len(smiles),
            pool="cls",
        )
    temporary.replace(output)
    print(f"Wrote {len(smiles)} ligand vectors to {output}")


if __name__ == "__main__":
    main()
