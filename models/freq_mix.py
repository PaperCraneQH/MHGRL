"""Mix a 3D pair-distance 168×60 histogram into the onion frequency table."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .multiscale_3d import N_ELE, N_RES, radial_bin


def pair_distance_hist(batch: Dict[str, torch.Tensor],
                       n_shells: int = 60) -> torch.Tensor:
    """Classic onion: residue-type × ligand-element counts by pair distance.

    Returns ``[B, 168, S]``.  Padded Cβ / atoms are ignored.
    """
    prot_xyz = batch['prot_xyz']
    lig_xyz = batch['lig_xyz']
    prot_mask = batch['prot_mask']
    lig_mask = batch['lig_mask']
    prot_type = batch['prot_type'].clamp(0, N_RES - 1)
    lig_type = batch['lig_type'].clamp(0, N_ELE - 1)
    B = prot_xyz.size(0)
    dtype = prot_xyz.dtype
    hist = prot_xyz.new_zeros(B, N_RES * N_ELE, n_shells)
    valid = prot_mask.unsqueeze(2) & lig_mask.unsqueeze(1)
    if not bool(valid.any()):
        return hist
    dist = torch.cdist(prot_xyz, lig_xyz)
    bins = radial_bin(dist, n_shells)
    pair = prot_type.unsqueeze(2) * N_ELE + lig_type.unsqueeze(1)
    b, i, j = valid.nonzero(as_tuple=True)
    hist.index_put_(
        (b, pair[b, i, j], bins[b, i, j]),
        torch.ones(b.size(0), device=prot_xyz.device, dtype=dtype),
        accumulate=True)
    return hist


def match_mass(onion: torch.Tensor, three: torch.Tensor) -> torch.Tensor:
    """Scale ``three`` so each sample has the same total count as ``onion``."""
    onion_sum = onion.flatten(1).sum(1).clamp(min=1.0)
    three_sum = three.flatten(1).sum(1).clamp(min=1.0)
    return three * (onion_sum / three_sum).view(-1, 1, 1)


class FreqMix(nn.Module):
    """``mix`` / ``replace`` change the table before projection; ``edges`` does not."""

    def __init__(self, mode: str = 'mix', gate_init: float = -4.0):
        super().__init__()
        mode = str(mode or 'mix').lower()
        if mode not in ('mix', 'replace', 'edges'):
            raise ValueError(
                f'freq mix mode must be mix|replace|edges, got {mode!r}')
        self.mode = mode
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.last_gate = 0.0
        self.last_abs = 0.0
        self.last_n_pl = 0.0

    def forward(self, onion: torch.Tensor,
                three: torch.Tensor) -> torch.Tensor:
        three = match_mass(onion.to(three.dtype), three)
        if self.mode == 'replace':
            mixed = three
            gate = 1.0
        else:
            gate = torch.sigmoid(self.gate)
            mixed = (1.0 - gate) * onion.to(three.dtype) + gate * three
        with torch.no_grad():
            self.last_gate = float(gate)
            self.last_abs = float(mixed.detach().abs().mean())
            self.last_n_pl = float((three.detach().sum(1) > 0).to(three.dtype).sum(1).mean())
        return mixed
