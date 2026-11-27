"""Map the 60 ordered slots onto nearest pocket residues instead of radial shells."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multiscale_3d import N_ELE, N_RES, PL_CUTOFF, radial_bin

COMPACT_DIM = N_RES + 3 * N_ELE + 1  # 21 + 24 + 1 = 46


def rank_residues(
        batch: Dict[str, torch.Tensor],
        n_slots: int = 60) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Nearest Cβ first. Returns ``index, slot_mask, min_dist`` each ``[B, S]``."""
    prot_xyz = batch['prot_xyz']
    lig_xyz = batch['lig_xyz']
    prot_mask = batch['prot_mask']
    lig_mask = batch['lig_mask']
    dist = torch.cdist(prot_xyz, lig_xyz)
    pair_ok = prot_mask.unsqueeze(2) & lig_mask.unsqueeze(1)
    dist = dist.masked_fill(~pair_ok, 1.0e9)
    min_d = dist.min(dim=-1).values.masked_fill(~prot_mask, 1.0e9)
    order = min_d.argsort(dim=-1)
    idx = order[:, :n_slots]
    chosen = min_d.gather(1, idx)
    return idx, chosen < 1.0e8, chosen


def _gather_xyz(xyz: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return xyz.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))


def residue_type_ele_table(
        batch: Dict[str, torch.Tensor],
        n_slots: int = 60,
        cutoff: float = PL_CUTOFF) -> torch.Tensor:
    """``[B, 168, S]``. Column j is the j-th nearest residue's type×element counts inside cutoff."""
    idx, slot_mask, _ = rank_residues(batch, n_slots)
    B = idx.size(0)
    device = idx.device
    dtype = batch['prot_xyz'].dtype
    prot_xyz = _gather_xyz(batch['prot_xyz'], idx)
    ptype = batch['prot_type'].gather(1, idx).clamp(0, N_RES - 1)
    lig_xyz = batch['lig_xyz']
    lig_type = batch['lig_type'].clamp(0, N_ELE - 1)
    lig_mask = batch['lig_mask']
    dist = torch.cdist(prot_xyz, lig_xyz)
    near = (dist < float(cutoff)) & slot_mask.unsqueeze(-1) & lig_mask.unsqueeze(1)
    lig_oh = F.one_hot(lig_type, N_ELE).to(dtype)
    ele = (lig_oh.unsqueeze(1) * near.unsqueeze(-1).to(dtype)).sum(2)
    table = prot_xyz.new_zeros(B, N_RES * N_ELE, n_slots)
    b = torch.arange(B, device=device).view(B, 1, 1).expand(B, n_slots, N_ELE)
    s = torch.arange(n_slots, device=device).view(1, n_slots, 1).expand(B, n_slots, N_ELE)
    e = torch.arange(N_ELE, device=device).view(1, 1, N_ELE).expand(B, n_slots, N_ELE)
    table[b, ptype.unsqueeze(-1) * N_ELE + e, s] = ele
    return table


def residue_radial_hist(
        batch: Dict[str, torch.Tensor],
        n_slots: int = 60,
        n_bins: int = 60) -> torch.Tensor:
    """``[B, S, 8, n_bins]`` pair-distance histogram of each ranked residue."""
    idx, slot_mask, _ = rank_residues(batch, n_slots)
    prot_xyz = _gather_xyz(batch['prot_xyz'], idx)
    lig_xyz = batch['lig_xyz']
    lig_type = batch['lig_type'].clamp(0, N_ELE - 1)
    lig_mask = batch['lig_mask']
    dist = torch.cdist(prot_xyz, lig_xyz)
    valid = slot_mask.unsqueeze(-1) & lig_mask.unsqueeze(1)
    hist = prot_xyz.new_zeros(idx.size(0), n_slots, N_ELE, n_bins)
    if not bool(valid.any()):
        return hist
    bins = radial_bin(dist, n_bins)
    b, s, k = valid.nonzero(as_tuple=True)
    hist.index_put_(
        (b, s, lig_type[b, k], bins[b, s, k]),
        torch.ones(b.size(0), device=prot_xyz.device, dtype=prot_xyz.dtype),
        accumulate=True)
    return hist


def residue_compact(
        batch: Dict[str, torch.Tensor],
        n_slots: int = 60,
        cutoff: float = PL_CUTOFF) -> torch.Tensor:
    """``[B, S, 46]``: type + 5Å element counts / min / mean + neighbor count."""
    idx, slot_mask, _ = rank_residues(batch, n_slots)
    prot_xyz = _gather_xyz(batch['prot_xyz'], idx)
    ptype = batch['prot_type'].gather(1, idx).clamp(0, N_RES - 1)
    lig_xyz = batch['lig_xyz']
    lig_type = batch['lig_type'].clamp(0, N_ELE - 1)
    lig_mask = batch['lig_mask']
    dist = torch.cdist(prot_xyz, lig_xyz)
    near = (dist < float(cutoff)) & slot_mask.unsqueeze(-1) & lig_mask.unsqueeze(1)
    dtype = prot_xyz.dtype
    lig_oh = F.one_hot(lig_type, N_ELE).to(dtype)
    ele = (lig_oh.unsqueeze(1) * near.unsqueeze(-1).to(dtype)).sum(2)
    # per-element min / mean among near atoms
    ele_dist = dist.unsqueeze(-1).expand(-1, -1, -1, N_ELE)
    ele_ok = near.unsqueeze(-1) & (lig_type.unsqueeze(1).unsqueeze(-1) ==
                                   torch.arange(N_ELE, device=dist.device).view(1, 1, 1, N_ELE))
    ele_big = ele_dist.masked_fill(~ele_ok, 1.0e9)
    ele_min = ele_big.min(dim=2).values.clamp(max=cutoff)
    ele_min = ele_min.masked_fill(ele_min >= 1.0e8, 0.0)
    ele_sum = (ele_dist * ele_ok.to(dtype)).sum(2)
    ele_mean = ele_sum / ele.clamp(min=1.0)
    ele_mean = ele_mean.masked_fill(ele <= 0, 0.0)
    n_near = near.to(dtype).sum(-1, keepdim=True)
    type_oh = F.one_hot(ptype, N_RES).to(dtype)
    feat = torch.cat([type_oh, ele, ele_min, ele_mean, n_near], dim=-1)
    return feat * slot_mask.unsqueeze(-1).to(dtype)


class ResidueSlotEncoder(nn.Module):
    """Project ranked-residue features to 60×192. ``table`` reuses the onion projector."""

    def __init__(self, hidden_dim: int = 192, n_slots: int = 60,
                 n_bins: int = 60, mode: str = 'hist'):
        super().__init__()
        mode = str(mode or 'hist').lower()
        if mode not in ('hist', 'compact', 'table'):
            raise ValueError(f'residue slot mode must be hist|compact|table, got {mode!r}')
        self.mode = mode
        self.n_slots = int(n_slots)
        self.n_bins = int(n_bins)
        self.hidden_dim = int(hidden_dim)
        self.last_n_res = 0.0
        self.last_abs = 0.0
        if mode == 'hist':
            self.proj = nn.Sequential(
                nn.Linear(N_ELE * int(n_bins), int(hidden_dim)),
                nn.GELU(),
                nn.LayerNorm(int(hidden_dim)),
            )
        elif mode == 'compact':
            self.proj = nn.Sequential(
                nn.Linear(COMPACT_DIM, int(hidden_dim)),
                nn.GELU(),
                nn.LayerNorm(int(hidden_dim)),
            )
        else:
            self.proj = None

    def encode_nodes(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.mode == 'hist':
            hist = residue_radial_hist(batch, self.n_slots, self.n_bins)
            x = hist.reshape(hist.size(0), self.n_slots, -1)
            h = self.proj(x)
        elif self.mode == 'compact':
            h = self.proj(residue_compact(batch, self.n_slots))
        else:
            raise RuntimeError('table mode uses the onion node_proj, not encode_nodes')
        with torch.no_grad():
            self.last_abs = float(h.detach().abs().mean())
            self.last_n_res = float(rank_residues(batch, self.n_slots)[1].to(h.dtype).sum(1).mean())
        return h

    def table(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        table = residue_type_ele_table(batch, self.n_slots)
        with torch.no_grad():
            self.last_abs = float(table.detach().abs().mean())
            self.last_n_res = float(rank_residues(batch, self.n_slots)[1].sum(1).float().mean())
        return table
