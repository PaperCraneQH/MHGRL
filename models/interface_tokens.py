"""Encode the pocket–ligand interface as one protein token and one ligand token."""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multiscale_3d import N_RBF, GeoNodeEmbed, ligand_centroid, rbf


class _BipartiteAttn(nn.Module):
    """Distance-biased attention from query nodes to key nodes."""

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f'dim {dim} not divisible by n_heads {n_heads}')
        self.n_heads = int(n_heads)
        self.head_dim = int(dim) // int(n_heads)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(float(dropout))
        self.radius_width = 1.5
        self.radius_center = 2.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        q_xyz: torch.Tensor,
        k_xyz: torch.Tensor,
        q_mask: torch.Tensor,
        k_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, Nq, D = query.shape
        Nk = key.size(1)
        H, d = self.n_heads, self.head_dim
        q = self.q(query).view(B, Nq, H, d).transpose(1, 2)
        k = self.k(key).view(B, Nk, H, d).transpose(1, 2)
        v = self.v(key).view(B, Nk, H, d).transpose(1, 2)
        attn = (q @ k.transpose(-1, -2)) / math.sqrt(d)
        dist = torch.cdist(q_xyz.to(query.dtype), k_xyz.to(query.dtype))
        bias = -((dist - self.radius_center) / self.radius_width) ** 2
        attn = attn + bias.unsqueeze(1)
        attn = attn.masked_fill(~k_mask.unsqueeze(1).unsqueeze(2), -1e9)
        attn = F.softmax(attn, dim=-1)
        valid_q = q_mask.any(dim=1).to(query.dtype).view(B, 1, 1, 1)
        valid_k = k_mask.any(dim=1).to(query.dtype).view(B, 1, 1, 1)
        attn = torch.nan_to_num(attn, nan=0.0) * valid_q * valid_k
        ctx = (attn @ v).transpose(1, 2).reshape(B, Nq, D)
        out = query + self.out(self.drop(ctx))
        return out * q_mask.unsqueeze(-1).to(query.dtype)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask.unsqueeze(-1).to(x.dtype)
    return (x * w).sum(1) / w.sum(1).clamp(min=1.0)


class InterfaceTokenGraph(nn.Module):
    """Cβ + ligand atoms, independent GAT-style attention, two 192-d tokens.

    ``residual`` / ``cross``: last Linear is zero-init, so the 62-node flatten
    starts unchanged. ``replace`` writes the pooled 3D tokens into
    the protein/ligand slots immediately.
    """

    def __init__(
        self,
        dim: int = 192,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        mode: str = 'residual',
    ):
        super().__init__()
        mode = str(mode or 'residual').lower()
        if mode not in ('residual', 'replace', 'cross', 'concat', 'concat_one'):
            raise ValueError(
                'interface token mode must be residual|replace|cross|'
                f'concat|concat_one, got {mode!r}')
        self.mode = mode
        self.embed = GeoNodeEmbed(dim)
        self.rbf_p = nn.Linear(N_RBF, dim)
        self.rbf_l = nn.Linear(N_RBF, dim)
        self.layers_pl = nn.ModuleList([
            _BipartiteAttn(dim, n_heads, dropout) for _ in range(int(n_layers))])
        self.layers_lp = nn.ModuleList([
            _BipartiteAttn(dim, n_heads, dropout) for _ in range(int(n_layers))])
        self.cross_p = None
        self.cross_l = None
        if mode == 'cross':
            self.cross_p = _BipartiteAttn(dim, n_heads, dropout)
            self.cross_l = _BipartiteAttn(dim, n_heads, dropout)
        self.out_p = nn.Linear(dim, dim)
        self.out_l = nn.Linear(dim, dim)
        if mode in ('residual', 'cross'):
            nn.init.zeros_(self.out_p.weight)
            nn.init.zeros_(self.out_p.bias)
            nn.init.zeros_(self.out_l.weight)
            nn.init.zeros_(self.out_l.bias)
        self.last_abs = 0.0
        self.last_n_cb = 0.0
        self.last_n_atom = 0.0

    def _encode(
        self, batch: Dict[str, torch.Tensor], dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cb, atom = self.embed(batch, dtype)
        prot_xyz = batch['prot_xyz'].to(dtype)
        lig_xyz = batch['lig_xyz'].to(dtype)
        prot_mask = batch['prot_mask']
        lig_mask = batch['lig_mask']
        cent = ligand_centroid(lig_xyz, lig_mask)
        rad_c = (prot_xyz - cent.unsqueeze(1)).norm(dim=-1)
        rad_a = (lig_xyz - cent.unsqueeze(1)).norm(dim=-1)
        cb = cb + self.rbf_p(rbf(rad_c, n_bins=N_RBF, d_max=30.0))
        atom = atom + self.rbf_l(rbf(rad_a, n_bins=N_RBF, d_max=30.0))
        cb = cb * prot_mask.unsqueeze(-1).to(dtype)
        atom = atom * lig_mask.unsqueeze(-1).to(dtype)
        for pl, lp in zip(self.layers_pl, self.layers_lp):
            cb = pl(cb, atom, prot_xyz, lig_xyz, prot_mask, lig_mask)
            atom = lp(atom, cb, lig_xyz, prot_xyz, lig_mask, prot_mask)
        return cb, atom, cent

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        dtype: torch.dtype,
        prot_h: Optional[torch.Tensor] = None,
        lig_h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cb, atom, cent = self._encode(batch, dtype)
        prot_mask = batch['prot_mask']
        lig_mask = batch['lig_mask']
        if self.mode == 'cross':
            if prot_h is None or lig_h is None:
                raise ValueError('cross mode requires prot_h and lig_h')
            ones = torch.ones(
                prot_h.size(0), 1, dtype=torch.bool, device=prot_h.device)
            delta_p = self.cross_p(
                prot_h.unsqueeze(1), cb, cent.unsqueeze(1),
                batch['prot_xyz'].to(dtype), ones, prot_mask)
            delta_l = self.cross_l(
                lig_h.unsqueeze(1), atom, cent.unsqueeze(1),
                batch['lig_xyz'].to(dtype), ones, lig_mask)
            prot_tok = self.out_p(delta_p.squeeze(1))
            lig_tok = self.out_l(delta_l.squeeze(1))
        else:
            prot_tok = self.out_p(_masked_mean(cb, prot_mask))
            lig_tok = self.out_l(_masked_mean(atom, lig_mask))
        with torch.no_grad():
            self.last_abs = float(
                (prot_tok.detach().abs().mean() + lig_tok.detach().abs().mean())
                * 0.5)
            self.last_n_cb = float(prot_mask.to(dtype).sum(1).mean())
            self.last_n_atom = float(lig_mask.to(dtype).sum(1).mean())
        return prot_tok, lig_tok
