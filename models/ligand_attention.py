"""Optional cross-attention from a protein or shell query to ligand tokens.

Queries have shape [B, d] or [B, Q, d]. The token cache is keyed by PDB
code. Shell queries give gated residual updates to shell nodes. Protein
queries condition the pooled ligand node.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from .pocket_attention import PocketCrossAttention


class LigandTokenCrossAttention(PocketCrossAttention):
    """Masked cross-attention over ligand tokens using protein or shell queries."""

    _QUERY_SOURCES = ("protein", "shell")

    def __init__(self, node_dim: int, token_hidden: int = 384,
                 n_heads: int = 4, dropout: float = 0.1,
                 mode: str = "residual", query_source: str = "shell",
                 max_tokens: int = 128):
        super().__init__(
            node_dim=node_dim, pocket_hidden=token_hidden, n_heads=n_heads,
            dropout=dropout, mode=mode, query_source=query_source,
            max_res=max_tokens)
        self.description = (
            f"LigandTokenCrossAttention dim={node_dim} heads={n_heads} "
            f"mode={mode} query={query_source} max_tokens={max_tokens}")


def build_ligand_attention(node_dim: int, cfg: Optional[Dict]
                           ) -> Optional[LigandTokenCrossAttention]:
    """Build from ``fusion.ligand_attention``; None when absent/off."""
    cfg = dict(cfg or {})
    if not bool(cfg.get("enabled", False)):
        return None
    return LigandTokenCrossAttention(
        node_dim=node_dim,
        token_hidden=int(cfg.get("token_hidden", 384)),
        n_heads=int(cfg.get("n_heads", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
        mode=str(cfg.get("mode", "residual")),
        query_source=str(cfg.get("query_source", "shell")),
        max_tokens=int(cfg.get("max_tokens", 128)),
    )


def open_ligand_token_cache(cfg: Optional[Dict]):
    """Open the ligand token cache declared in ``fusion.ligand_attention``."""
    cfg = dict(cfg or {})
    if not bool(cfg.get("enabled", False)):
        return None
    path = cfg.get("cache_path")
    if not path:
        raise ValueError("ligand_attention.enabled requires cache_path")
    from pathlib import Path

    from data.pocket_cache import PocketResidueCache
    root = Path(__file__).resolve().parents[1]
    full = path if Path(path).is_absolute() else str(root / path)
    return PocketResidueCache(
        full, dtype=torch.float32,
        in_memory=bool(cfg.get("cache_in_memory", True)),
        missing_policy=str(cfg.get("missing_policy", "zero")))
