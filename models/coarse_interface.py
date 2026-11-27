"""Encode pocket Cβ and ligand heavy-atom contacts as shell updates."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

N_RES = 21
N_ELE = 8
N_RBF = 16
PP_CUTOFF = 8.0
PL_CUTOFF = 5.0


def _rbf(dist: torch.Tensor, n_bins: int = N_RBF, d_max: float = 8.0) -> torch.Tensor:
    centers = torch.linspace(0.0, d_max, n_bins, device=dist.device, dtype=dist.dtype)
    width = d_max / n_bins
    return torch.exp(-((dist.unsqueeze(-1) - centers) / width) ** 2)


def _radial_bin(dist: torch.Tensor, n_shells: int = 60) -> torch.Tensor:
    # Shell upper bounds 1.0, 1.5, ..., 30.5 Å.
    idx = torch.floor((dist - 1.0) / 0.5).long()
    return idx.clamp(0, n_shells - 1)


class CoarseInterfaceEncoder(nn.Module):
    def __init__(self, node_dim: int = 192, n_layers: int = 2,
                 n_shells: int = 60, extra_nodes: int = 0,
                 extra_cb: int = 16, extra_lig: int = 8,
                 edge_readout: bool = False, dropout: float = 0.1):
        super().__init__()
        self.dim = int(node_dim)
        self.n_layers = int(n_layers)
        self.n_shells = int(n_shells)
        self.extra_cb = int(extra_cb)
        self.extra_lig = int(extra_lig)
        self.extra_nodes = int(extra_nodes)
        if self.extra_nodes > 0:
            self.extra_nodes = self.extra_cb + self.extra_lig
        self.edge_readout = bool(edge_readout)
        self.prot_emb = nn.Embedding(N_RES, self.dim)
        self.lig_emb = nn.Embedding(N_ELE, self.dim)
        self.rel_emb = nn.Embedding(3, self.dim)  # PP, LL, PL
        hidden = self.dim
        self.msg = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.dim * 2 + N_RBF + self.dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.dim),
            )
            for _ in range(self.n_layers)
        ])
        self.upd = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(self.dim), nn.Dropout(dropout))
            for _ in range(self.n_layers)
        ])
        self.to_shell = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.to_shell.weight)
        nn.init.zeros_(self.to_shell.bias)
        if self.edge_readout:
            self.edge_head = nn.Sequential(
                nn.Linear(self.dim * 2 + N_RBF, self.dim),
                nn.GELU(),
                nn.Linear(self.dim, 1),
            )
            nn.init.zeros_(self.edge_head[-1].weight)
            nn.init.zeros_(self.edge_head[-1].bias)
        self.last_shell_abs = 0.0
        self.last_edge_abs = 0.0
        self.last_n_pl = 0.0
        self.last_n_prot = 0.0
        self.last_n_lig = 0.0

    def _update(self, h: torch.Tensor, dist: torch.Tensor,
                pair: torch.Tensor, rel: torch.Tensor, layer: int) -> torch.Tensor:
        B, N, D = h.shape
        if not bool(pair.any()):
            return h
        bidx, src, dst = pair.nonzero(as_tuple=True)
        msg = self.msg[layer](torch.cat([
            h[bidx, src], h[bidx, dst],
            _rbf(dist[bidx, src, dst]),
            self.rel_emb(rel[bidx, src, dst]),
        ], dim=-1))
        dest = bidx * N + src
        agg = h.new_zeros(B * N, D)
        cnt = h.new_zeros(B * N, 1)
        agg.index_add_(0, dest, msg)
        cnt.index_add_(0, dest, torch.ones(dest.size(0), 1, device=h.device, dtype=h.dtype))
        agg = (agg / cnt.clamp(min=1.0)).view(B, N, D)
        return self.upd[layer](h + agg)

    def forward(
        self, batch: Dict[str, torch.Tensor], shell: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        prot_xyz = batch["prot_xyz"].to(device=shell.device, dtype=shell.dtype)
        lig_xyz = batch["lig_xyz"].to(device=shell.device, dtype=shell.dtype)
        prot_mask = batch["prot_mask"].to(device=shell.device)
        lig_mask = batch["lig_mask"].to(device=shell.device)
        B, P, _ = prot_xyz.shape
        L = lig_xyz.size(1)
        N = P + L
        xyz = torch.cat([prot_xyz, lig_xyz], dim=1)
        mask = torch.cat([prot_mask, lig_mask], dim=1)
        is_prot = torch.zeros(B, N, dtype=torch.bool, device=shell.device)
        is_prot[:, :P] = True
        h = torch.zeros(B, N, self.dim, device=shell.device, dtype=shell.dtype)
        h[:, :P] = self.prot_emb(batch["prot_type"].to(shell.device).clamp(0, N_RES - 1))
        h[:, P:] = self.lig_emb(batch["lig_type"].to(shell.device).clamp(0, N_ELE - 1))
        h = h * mask.unsqueeze(-1).to(h.dtype)

        dist = torch.cdist(xyz, xyz)
        both = mask.unsqueeze(2) & mask.unsqueeze(1)
        eye = torch.eye(N, dtype=torch.bool, device=shell.device).unsqueeze(0)
        pp = is_prot.unsqueeze(2) & is_prot.unsqueeze(1) & (dist < PP_CUTOFF)
        pl = (is_prot.unsqueeze(2) ^ is_prot.unsqueeze(1)) & (dist < PL_CUTOFF)
        ll = (~is_prot.unsqueeze(2)) & (~is_prot.unsqueeze(1))
        ll_bond = torch.zeros(B, N, N, dtype=torch.bool, device=shell.device)
        src = batch["lig_bond"].to(shell.device)[..., 0] + P
        dst = batch["lig_bond"].to(shell.device)[..., 1] + P
        valid_b = batch["bond_mask"].to(shell.device)
        b_flat = torch.arange(B, device=shell.device).unsqueeze(1).expand_as(src)
        if bool(valid_b.any()):
            ll_bond[b_flat[valid_b], src[valid_b], dst[valid_b]] = True
            ll_bond[b_flat[valid_b], dst[valid_b], src[valid_b]] = True
        pair = both & ~eye & (pp | pl | (ll & ll_bond))
        rel = torch.zeros(B, N, N, dtype=torch.long, device=shell.device)
        rel = torch.where(ll & ll_bond, torch.ones_like(rel), rel)
        rel = torch.where(pl, torch.full_like(rel, 2), rel)

        for li in range(self.n_layers):
            h = self._update(h, dist, pair, rel, li)

        lig_w = lig_mask.unsqueeze(-1).to(lig_xyz.dtype)
        lig_cent = (lig_xyz * lig_w).sum(1) / lig_w.sum(1).clamp(min=1.0)
        rad = (xyz - lig_cent.unsqueeze(1)).norm(dim=-1)
        bins = _radial_bin(rad, self.n_shells).masked_fill(~mask, 0)
        shell_sum = torch.zeros(
            B, self.n_shells, self.dim, device=shell.device, dtype=h.dtype)
        counts = torch.zeros(B, self.n_shells, 1, device=shell.device, dtype=h.dtype)
        shell_sum.scatter_add_(
            1, bins.unsqueeze(-1).expand(-1, -1, self.dim),
            h * mask.unsqueeze(-1).to(h.dtype))
        counts.scatter_add_(1, bins.unsqueeze(-1), mask.unsqueeze(-1).to(h.dtype))
        residual = self.to_shell(shell_sum / counts.clamp(min=1.0))
        with torch.no_grad():
            self.last_shell_abs = float(residual.detach().abs().mean())
            self.last_n_prot = float(prot_mask.to(h.dtype).sum(1).mean())
            self.last_n_lig = float(lig_mask.to(h.dtype).sum(1).mean())
        shell_out = shell + residual

        extra = None
        if self.extra_nodes > 0:
            n_cb = min(self.extra_cb, P)
            n_la = min(self.extra_lig, L)
            prot_rad = rad[:, :P].masked_fill(~prot_mask, 1e6)
            lig_rad = rad[:, P:].masked_fill(~lig_mask, 1e6)
            _, p_idx = prot_rad.topk(n_cb, dim=1, largest=False)
            _, l_idx = lig_rad.topk(n_la, dim=1, largest=False)
            extra_p = torch.gather(
                h[:, :P], 1, p_idx.unsqueeze(-1).expand(-1, -1, self.dim))
            extra_l = torch.gather(
                h[:, P:], 1, l_idx.unsqueeze(-1).expand(-1, -1, self.dim))
            extra = torch.cat([extra_p, extra_l], dim=1)
            if extra.size(1) < self.extra_nodes:
                extra = torch.cat([
                    extra,
                    extra.new_zeros(B, self.extra_nodes - extra.size(1), self.dim),
                ], dim=1)
            extra = extra[:, :self.extra_nodes]

        edge_pred = None
        if self.edge_readout:
            pl_mask = pair & pl
            if bool(pl_mask.any()):
                bidx, src_i, dst_j = pl_mask.nonzero(as_tuple=True)
                logits = self.edge_head(torch.cat([
                    h[bidx, src_i], h[bidx, dst_j],
                    _rbf(dist[bidx, src_i, dst_j]),
                ], dim=-1)).squeeze(-1)
                acc = h.new_zeros(B)
                cnt = h.new_zeros(B)
                acc.index_add_(0, bidx, logits)
                cnt.index_add_(0, bidx, torch.ones_like(logits))
                edge_pred = (acc / cnt.clamp(min=1.0)).view(B, 1)
            else:
                edge_pred = h.new_zeros(B, 1)
            with torch.no_grad():
                self.last_edge_abs = float(edge_pred.detach().abs().mean())
                self.last_n_pl = float(pl_mask.to(h.dtype).sum(dim=(1, 2)).mean())
        return shell_out, extra, edge_pred
