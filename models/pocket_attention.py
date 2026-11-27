"""Optional cross-attention over binding-pocket residue embeddings.

The residue cache is keyed by PDB code. Ligand queries condition the pooled
protein node. Shell queries give gated residual updates to shell nodes.
The mean-only variant pools pocket residues without attention.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PocketCrossAttention(nn.Module):
    """Multi-head cross-attention from queries onto pocket residue embeddings."""

    _MODES = ("replace", "residual", "mean_only")
    _QUERY_SOURCES = ("ligand", "shell")

    def __init__(self, node_dim: int, pocket_hidden: int = 1024,
                 n_heads: int = 4, dropout: float = 0.1,
                 mode: str = "replace", query_source: str = "shell",
                 max_res: int = 128):
        super().__init__()
        if mode not in self._MODES:
            raise ValueError(
                f"pocket_attention.mode must be one of {self._MODES}, got {mode!r}")
        if query_source not in self._QUERY_SOURCES:
            raise ValueError(
                f"pocket_attention.query_source must be one of "
                f"{self._QUERY_SOURCES}, got {query_source!r}")
        if node_dim % n_heads != 0:
            raise ValueError(
                f"node_dim={node_dim} must be divisible by n_heads={n_heads}")

        self.node_dim = int(node_dim)
        self.n_heads = int(n_heads)
        self.head_dim = self.node_dim // self.n_heads
        self.mode = mode
        self.query_source = query_source
        self.max_res = int(max_res)

        self.res_proj = nn.Linear(pocket_hidden, self.node_dim)
        self.res_norm = nn.LayerNorm(self.node_dim)
        self.q_proj = nn.Linear(self.node_dim, self.node_dim)
        self.k_proj = nn.Linear(self.node_dim, self.node_dim)
        self.v_proj = nn.Linear(self.node_dim, self.node_dim)
        self.out_proj = nn.Linear(self.node_dim, self.node_dim)
        self.out_norm = nn.LayerNorm(self.node_dim)
        self.dropout = nn.Dropout(float(dropout))
        # Initialize residual-mode gates near zero.
        self.use_gate = mode == "residual"
        self.gate = nn.Parameter(torch.tensor(-2.0))
        self.last_attn_entropy: Optional[float] = None
        self.description = (
            f"PocketCrossAttention dim={node_dim} heads={n_heads} "
            f"mode={mode} query={query_source} max_res={max_res}")

    def pocket_mean(self, pocket: torch.Tensor,
                    pocket_mask: torch.Tensor) -> torch.Tensor:
        """Mask-aware mean over pocket residues, projected to node width."""
        h = self.res_norm(self.res_proj(pocket))
        m = pocket_mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

    def forward(self, query: torch.Tensor, pocket: torch.Tensor,
                pocket_mask: torch.Tensor) -> torch.Tensor:
        """Attend from ``query`` onto pocket residues.

        ``query`` is ``[B, d]`` or ``[B, Q, d]``; the output matches its shape.
        """
        squeeze = query.dim() == 2
        q_in = query.unsqueeze(1) if squeeze else query
        B, Q, _ = q_in.shape

        h = self.res_norm(self.res_proj(pocket))            # [B, R, d]
        if self.mode == "mean_only":
            m = pocket_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
            out = pooled.unsqueeze(1).expand(B, Q, self.node_dim)
            self.last_attn_entropy = 1.0
            return out.squeeze(1) if squeeze else out

        R = h.size(1)
        q = self.q_proj(q_in).view(B, Q, self.n_heads, self.head_dim)
        k = self.k_proj(h).view(B, R, self.n_heads, self.head_dim)
        v = self.v_proj(h).view(B, R, self.n_heads, self.head_dim)
        q = q.transpose(1, 2)                                # [B, Hd, Q, hd]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        logits = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        pad = ~pocket_mask.view(B, 1, 1, R)
        logits = logits.masked_fill(pad, torch.finfo(logits.dtype).min)
        attn = logits.softmax(dim=-1)
        with torch.no_grad():
            # Use the normalized weights before dropout to compute entropy.
            valid = pocket_mask.sum(dim=1).to(attn.dtype)
            p = attn.clamp_min(1e-12)
            ent = -(p * p.log()).sum(-1)                     # [B, Hd, Q]
            # Normalized entropy is undefined for a single valid key.
            usable = valid > 1
            if bool(usable.any()):
                norm = torch.log(valid.clamp(min=2.0)).view(B, 1, 1)
                ratio = (ent / norm)[usable.view(B, 1, 1).expand_as(ent)]
                self.last_attn_entropy = float(ratio.mean())
            else:
                self.last_attn_entropy = float('nan')
        attn = self.dropout(attn)
        ctx = (attn @ v).transpose(1, 2).reshape(B, Q, self.node_dim)
        ctx = self.out_norm(self.out_proj(ctx))

        out = ctx * torch.sigmoid(self.gate) if self.use_gate else ctx
        return out.squeeze(1) if squeeze else out


class DilutionGate(nn.Module):
    """Per-sample mixing of the whole-chain and pocket protein representations.

    The gate interpolates between the pooled chain and pocket embeddings::

        prot = (1 - g) * prot_global + g * prot_pocket
        g    = sigmoid(MLP([log L, log n_pocket, pocket_share]))

    ``pocket_share = n_pocket / min(L, share_max_len)`` and the two residue
    counts can be computed at inference time without affinity labels.

    Modes
    -----
    ``learned``  A per-complex gate using the residue-count features.
    ``scalar``   A learned scalar gate shared by all complexes.
    ``fixed``    A length-based switch, ``g = 1{L > threshold}``.
    """

    _MODES = ("learned", "scalar", "fixed")

    def __init__(self, mode: str = "learned", hidden: int = 32,
                 threshold: float = 512.0, share_max_len: int = 512,
                 init_bias: float = 0.0):
        super().__init__()
        if mode not in self._MODES:
            raise ValueError(
                f"pocket gate mode must be one of {self._MODES}, got {mode!r}")
        self.mode = mode
        self.threshold = float(threshold)
        self.share_max_len = int(share_max_len)
        if mode == "learned":
            self.net = nn.Sequential(
                nn.Linear(3, int(hidden)), nn.GELU(),
                nn.Linear(int(hidden), 1))
            with torch.no_grad():
                self.net[-1].bias.fill_(float(init_bias))
        elif mode == "scalar":
            self.logit = nn.Parameter(torch.tensor(float(init_bias)))
        self.last_gate_mean: Optional[float] = None
        self.last_gate_share_corr: Optional[float] = None
        self.description = (
            f"DilutionGate mode={mode} threshold={threshold:g} "
            f"share_max_len={share_max_len}")

    def features(self, lengths: torch.Tensor, n_pocket: torch.Tensor
                 ) -> torch.Tensor:
        eff = torch.minimum(lengths, torch.full_like(
            lengths, float(self.share_max_len))).clamp(min=1.0)
        share = (n_pocket / eff).clamp(0.0, 1.0)
        return torch.stack([
            torch.log1p(lengths) / 8.0,      # ~O(1) for 100-5000 residues
            torch.log1p(n_pocket) / 5.0,
            share,
        ], dim=-1)

    def forward(self, lengths: torch.Tensor, n_pocket: torch.Tensor
                ) -> torch.Tensor:
        """Return the mixing weight ``g`` of shape ``[B, 1]``."""
        feats = self.features(lengths, n_pocket)
        if self.mode == "fixed":
            g = (lengths > self.threshold).to(feats.dtype).unsqueeze(-1)
        elif self.mode == "scalar":
            g = torch.sigmoid(self.logit).expand(lengths.size(0), 1)
        else:
            g = torch.sigmoid(self.net(feats))
        with torch.no_grad():
            share = feats[:, 2]
            gg = g.view(-1)
            self.last_gate_mean = float(gg.mean())
            if gg.numel() > 2 and float(gg.std()) > 1e-6 \
                    and float(share.std()) > 1e-6:
                gc = gg - gg.mean()
                sc = share - share.mean()
                self.last_gate_share_corr = float(
                    (gc * sc).mean() / (gg.std(unbiased=False)
                                        * share.std(unbiased=False)))
            else:
                self.last_gate_share_corr = float('nan')
        return g


def build_dilution_gate(cfg: Optional[Dict]) -> Optional[DilutionGate]:
    """Build from ``fusion.pocket_attention.gate``; None when absent/off."""
    cfg = dict(cfg or {})
    gate_cfg = dict(cfg.get("gate") or {})
    if not bool(gate_cfg.get("enabled", False)):
        return None
    return DilutionGate(
        mode=str(gate_cfg.get("mode", "learned")),
        hidden=int(gate_cfg.get("hidden", 32)),
        threshold=float(gate_cfg.get("threshold", 512)),
        share_max_len=int(gate_cfg.get("share_max_len", 512)),
        init_bias=float(gate_cfg.get("init_bias", 0.0)),
    )


def build_pocket_attention(node_dim: int, cfg: Optional[Dict]
                           ) -> Optional[PocketCrossAttention]:
    """Build from the ``fusion.pocket_attention`` sub-dict; None when off."""
    cfg = dict(cfg or {})
    if not bool(cfg.get("enabled", False)):
        return None
    return PocketCrossAttention(
        node_dim=node_dim,
        pocket_hidden=int(cfg.get("pocket_hidden", 1024)),
        n_heads=int(cfg.get("n_heads", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
        mode=str(cfg.get("mode", "replace")),
        query_source=str(cfg.get("query_source", "shell")),
        max_res=int(cfg.get("max_res", 128)),
    )


class PocketNodePool(nn.Module):
    """Project pocket residues and collapse them to one graph node.

    This is not cross-attention.  The failed R12–15 path replaced or gated the
    protein node; here the pooled pocket is a *fourth* node type that flatten
    can keep or ignore by position.
    """

    def __init__(self, node_dim: int, pocket_hidden: int = 1024):
        super().__init__()
        self.res_proj = nn.Linear(int(pocket_hidden), int(node_dim))
        self.res_norm = nn.LayerNorm(int(node_dim))
        self.last_n_res = 0.0
        self.last_prot_l2 = 0.0

    def forward(self, pocket: torch.Tensor, pocket_mask: torch.Tensor,
                prot: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.res_norm(self.res_proj(pocket))
        m = pocket_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        with torch.no_grad():
            self.last_n_res = float(pocket_mask.sum(dim=1).to(h.dtype).mean())
            if prot is not None:
                self.last_prot_l2 = float((pooled - prot).norm(dim=-1).mean())
        return pooled


def open_pocket_cache(cfg: Optional[Dict]):
    """Open the pocket cache declared in ``fusion.pocket_attention`` or ``pocket_node``."""
    cfg = dict(cfg or {})
    if not bool(cfg.get("enabled", False)):
        return None
    path = cfg.get("cache_path")
    if not path:
        raise ValueError("pocket_attention.enabled requires cache_path")
    from pathlib import Path

    from data.pocket_cache import PocketResidueCache
    root = Path(__file__).resolve().parents[1]
    full = path if Path(path).is_absolute() else str(root / path)
    return PocketResidueCache(
        full, dtype=torch.float32,
        in_memory=bool(cfg.get("cache_in_memory", True)),
        missing_policy=str(cfg.get("missing_policy", "zero")))
