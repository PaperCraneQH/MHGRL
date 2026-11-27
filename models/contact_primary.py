"""Cβ + ligand-atom contact graph as the affinity computational graph.

The 62-node onion flatten is not the readout.  Edges carry instance
distances.  Optional onion 168-d columns are node features only.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .coarse_interface import N_ELE, N_RES, N_RBF, PL_CUTOFF, PP_CUTOFF, _rbf, _radial_bin


class ContactPrimaryGraph(nn.Module):
    def __init__(
        self,
        dim: int = 192,
        n_layers: int = 3,
        dropout: float = 0.1,
        onion_feat: bool = False,
        readout: str = 'pool',
        n_cb_read: int = 32,
        n_atom_read: int = 16,
        use_lm: bool = True,
    ):
        super().__init__()
        readout = str(readout or 'pool').lower()
        if readout not in ('pool', 'ordered'):
            raise ValueError(f'contact readout must be pool|ordered, got {readout!r}')
        self.dim = int(dim)
        self.n_layers = int(n_layers)
        self.onion_feat = bool(onion_feat)
        self.readout = readout
        self.n_cb_read = int(n_cb_read)
        self.n_atom_read = int(n_atom_read)
        self.use_lm = bool(use_lm)
        self.prot_emb = nn.Embedding(N_RES, self.dim)
        self.lig_emb = nn.Embedding(N_ELE, self.dim)
        self.rel_emb = nn.Embedding(5, self.dim)  # PP, LL, PL, P-LM, L-LM
        self.onion_proj = nn.Linear(168, self.dim) if self.onion_feat else None
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
        if self.readout == 'pool':
            self.head = nn.Sequential(
                nn.Linear(self.dim, 384),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(384, 1),
            )
        else:
            in_dim = (self.n_cb_read + self.n_atom_read) * self.dim
            self.head = nn.Sequential(
                nn.Linear(in_dim, 1536),
                nn.GELU(),
                nn.LayerNorm(1536),
                nn.Dropout(dropout),
                nn.Linear(1536, 384),
                nn.GELU(),
                nn.LayerNorm(384),
                nn.Dropout(dropout),
                nn.Linear(384, 1),
            )
        self.last_abs = 0.0
        self.last_n_prot = 0.0
        self.last_n_lig = 0.0
        self.last_n_pl = 0.0

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
        self,
        batch: Dict[str, torch.Tensor],
        dtype: torch.dtype,
        freq_raw: Optional[torch.Tensor] = None,
        prot: Optional[torch.Tensor] = None,
        lig: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = batch['prot_xyz'].device
        prot_xyz = batch['prot_xyz'].to(device=device, dtype=dtype)
        lig_xyz = batch['lig_xyz'].to(device=device, dtype=dtype)
        prot_mask = batch['prot_mask'].to(device=device)
        lig_mask = batch['lig_mask'].to(device=device)
        B, P, _ = prot_xyz.shape
        L = lig_xyz.size(1)
        extra = 2 if (self.use_lm and prot is not None and lig is not None) else 0
        N = P + L + extra
        xyz = F.pad(torch.cat([prot_xyz, lig_xyz], dim=1), (0, 0, 0, extra))
        mask = F.pad(torch.cat([prot_mask, lig_mask], dim=1), (0, extra), value=True)
        is_prot = torch.zeros(B, N, dtype=torch.bool, device=device)
        is_prot[:, :P] = True
        is_lig = torch.zeros(B, N, dtype=torch.bool, device=device)
        is_lig[:, P:P + L] = True
        h = torch.zeros(B, N, self.dim, device=device, dtype=dtype)
        h[:, :P] = self.prot_emb(batch['prot_type'].to(device).clamp(0, N_RES - 1))
        h[:, P:P + L] = self.lig_emb(batch['lig_type'].to(device).clamp(0, N_ELE - 1))
        if extra:
            h[:, P + L] = prot.to(dtype)
            h[:, P + L + 1] = lig.to(dtype)
            xyz[:, P + L] = (
                lig_xyz * lig_mask.unsqueeze(-1).to(dtype)
            ).sum(1) / lig_mask.unsqueeze(-1).to(dtype).sum(1).clamp(min=1.0)
            xyz[:, P + L + 1] = xyz[:, P + L]

        if self.onion_proj is not None:
            if freq_raw is None:
                raise ValueError('onion_feat requires freq_raw [B,168,60]')
            table = freq_raw.to(device=device, dtype=dtype)
            if table.dim() != 3 or table.size(1) != 168:
                raise ValueError(f'freq_raw must be [B,168,60], got {tuple(table.shape)}')
            lig_cent = xyz[:, P + L] if extra else (
                lig_xyz * lig_mask.unsqueeze(-1).to(dtype)
            ).sum(1) / lig_mask.unsqueeze(-1).to(dtype).sum(1).clamp(min=1.0)
            rad = (xyz - lig_cent.unsqueeze(1)).norm(dim=-1)
            bins = _radial_bin(rad, table.size(-1))
            gathered = table.gather(
                2, bins.unsqueeze(1).expand(-1, table.size(1), -1))
            h = h + self.onion_proj(gathered.transpose(1, 2))

        h = h * mask.unsqueeze(-1).to(dtype)
        dist = torch.cdist(xyz, xyz)
        both = mask.unsqueeze(2) & mask.unsqueeze(1)
        eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
        pp = is_prot.unsqueeze(2) & is_prot.unsqueeze(1) & (dist < PP_CUTOFF)
        pl = ((is_prot.unsqueeze(2) & is_lig.unsqueeze(1))
              | (is_lig.unsqueeze(2) & is_prot.unsqueeze(1))) & (dist < PL_CUTOFF)
        ll = is_lig.unsqueeze(2) & is_lig.unsqueeze(1)
        ll_bond = torch.zeros(B, N, N, dtype=torch.bool, device=device)
        src = batch['lig_bond'].to(device)[..., 0] + P
        dst = batch['lig_bond'].to(device)[..., 1] + P
        valid_b = batch['bond_mask'].to(device)
        b_flat = torch.arange(B, device=device).unsqueeze(1).expand_as(src)
        if bool(valid_b.any()):
            ll_bond[b_flat[valid_b], src[valid_b], dst[valid_b]] = True
            ll_bond[b_flat[valid_b], dst[valid_b], src[valid_b]] = True
        pair = both & ~eye & (pp | pl | (ll & ll_bond))
        rel = torch.zeros(B, N, N, dtype=torch.long, device=device)
        rel = torch.where(ll & ll_bond, torch.ones_like(rel), rel)
        rel = torch.where(pl, torch.full_like(rel, 2), rel)
        if extra:
            p_lm = torch.zeros(B, N, dtype=torch.bool, device=device)
            l_lm = torch.zeros(B, N, dtype=torch.bool, device=device)
            p_lm[:, P + L] = True
            l_lm[:, P + L + 1] = True
            lm_p = ((p_lm.unsqueeze(2) & is_prot.unsqueeze(1))
                    | (is_prot.unsqueeze(2) & p_lm.unsqueeze(1)))
            lm_l = ((l_lm.unsqueeze(2) & is_lig.unsqueeze(1))
                    | (is_lig.unsqueeze(2) & l_lm.unsqueeze(1)))
            pair = pair | (both & ~eye & (lm_p | lm_l))
            rel = torch.where(lm_p, torch.full_like(rel, 3), rel)
            rel = torch.where(lm_l, torch.full_like(rel, 4), rel)

        for li in range(self.n_layers):
            h = self._update(h, dist, pair, rel, li)

        struct = h[:, :P + L]
        struct_mask = mask[:, :P + L]
        if self.readout == 'pool':
            feat = (struct * struct_mask.unsqueeze(-1).to(dtype)).sum(1)
            feat = feat / struct_mask.unsqueeze(-1).to(dtype).sum(1).clamp(min=1.0)
        else:
            prot_rad = dist[:, :P, P:P + L].masked_fill(
                ~lig_mask.unsqueeze(1), 1.0e9).min(dim=-1).values
            lig_rad = dist[:, P:P + L, :P].masked_fill(
                ~prot_mask.unsqueeze(1), 1.0e9).min(dim=-1).values
            prot_rad = prot_rad.masked_fill(~prot_mask, 1.0e9)
            lig_rad = lig_rad.masked_fill(~lig_mask, 1.0e9)
            _, p_idx = prot_rad.topk(min(self.n_cb_read, P), dim=1, largest=False)
            _, l_idx = lig_rad.topk(min(self.n_atom_read, L), dim=1, largest=False)
            extra_p = torch.gather(
                h[:, :P], 1, p_idx.unsqueeze(-1).expand(-1, -1, self.dim))
            extra_l = torch.gather(
                h[:, P:P + L], 1, l_idx.unsqueeze(-1).expand(-1, -1, self.dim))
            feat = torch.cat([extra_p, extra_l], dim=1).reshape(B, -1)
        pred = self.head(feat)
        with torch.no_grad():
            self.last_abs = float(pred.detach().abs().mean())
            self.last_n_prot = float(prot_mask.to(dtype).sum(1).mean())
            self.last_n_lig = float(lig_mask.to(dtype).sum(1).mean())
            self.last_n_pl = float((pair & pl).to(dtype).sum(dim=(1, 2)).mean())
        return pred
