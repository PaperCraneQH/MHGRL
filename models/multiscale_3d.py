"""Build an optional graph extension with pocket Cβ and ligand-atom nodes.

These nodes are updated along with the shell nodes. Distances are computed
from coordinates for each complex.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

N_RES = 21
N_ELE = 8
N_RBF = 16
PP_CUTOFF = 8.0
PL_CUTOFF = 5.0


def radial_bin(dist: torch.Tensor, n_shells: int = 60) -> torch.Tensor:
    idx = torch.floor((dist - 1.0) / 0.5).long()
    return idx.clamp(0, n_shells - 1)


def rbf(dist: torch.Tensor, n_bins: int = N_RBF, d_max: float = 8.0) -> torch.Tensor:
    centers = torch.linspace(0.0, d_max, n_bins, device=dist.device, dtype=dist.dtype)
    width = d_max / n_bins
    return torch.exp(-((dist.unsqueeze(-1) - centers) / width) ** 2)


class GeoNodeEmbed(nn.Module):
    def __init__(self, dim: int = 192):
        super().__init__()
        self.prot_emb = nn.Embedding(N_RES, dim)
        self.lig_emb = nn.Embedding(N_ELE, dim)

    def forward(self, batch: Dict[str, torch.Tensor],
                dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        prot = self.prot_emb(batch["prot_type"].clamp(0, N_RES - 1)).to(dtype)
        lig = self.lig_emb(batch["lig_type"].clamp(0, N_ELE - 1)).to(dtype)
        prot = prot * batch["prot_mask"].unsqueeze(-1).to(dtype)
        lig = lig * batch["lig_mask"].unsqueeze(-1).to(dtype)
        return prot, lig


def ligand_centroid(lig_xyz: torch.Tensor, lig_mask: torch.Tensor) -> torch.Tensor:
    w = lig_mask.unsqueeze(-1).to(lig_xyz.dtype)
    return (lig_xyz * w).sum(1) / w.sum(1).clamp(min=1.0)


def build_geo_edges(
    batch: Dict[str, torch.Tensor],
    *,
    n_shells: int,
    n_single: int,
    cb_start: int,
    atom_start: int,
    prot_idx: int,
    lig_idx: int,
    hier: bool,
    intra: bool,
    inter: bool,
    lm_broadcast: bool,
    rel_cb2s: int, rel_s2cb: int, rel_a2s: int, rel_s2a: int,
    rel_cb2cb: int, rel_a2a: int, rel_cb2a: int, rel_a2cb: int,
    rel_p2cb: int, rel_cb2p: int, rel_l2a: int, rel_a2l: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Return batch-offset src, dst, grp, rel plus a small stats dict."""
    device = batch["prot_xyz"].device
    prot_xyz = batch["prot_xyz"]
    lig_xyz = batch["lig_xyz"]
    prot_mask = batch["prot_mask"]
    lig_mask = batch["lig_mask"]
    B, P, _ = prot_xyz.shape
    L = lig_xyz.size(1)
    base = (torch.arange(B, device=device) * n_single).unsqueeze(1)
    srcs, dsts, grps, rels = [], [], [], []
    stats = {"n_cb": 0.0, "n_atom": 0.0, "n_hier": 0.0, "n_pl": 0.0}

    def _push(s, d, rel, mask):
        if mask is not None:
            s, d = s[mask], d[mask]
        if s.numel() == 0:
            return
        srcs.append(s)
        dsts.append(d)
        grps.append(torch.zeros_like(s))
        rels.append(torch.full_like(s, rel))

    stats["n_cb"] = float(prot_mask.to(prot_xyz.dtype).sum(1).mean())
    stats["n_atom"] = float(lig_mask.to(lig_xyz.dtype).sum(1).mean())
    cent = ligand_centroid(lig_xyz, lig_mask)

    if hier:
        rad_c = (prot_xyz - cent.unsqueeze(1)).norm(dim=-1)
        rad_a = (lig_xyz - cent.unsqueeze(1)).norm(dim=-1)
        bin_c = radial_bin(rad_c, n_shells)
        bin_a = radial_bin(rad_a, n_shells)
        cb_i = torch.arange(P, device=device).unsqueeze(0).expand(B, P)
        at_i = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        b_c = torch.arange(B, device=device).unsqueeze(1).expand_as(cb_i)
        b_a = torch.arange(B, device=device).unsqueeze(1).expand_as(at_i)
        s_cb = (base[b_c, 0] + cb_start + cb_i)[prot_mask]
        d_sh = (base[b_c, 0] + bin_c)[prot_mask]
        s_at = (base[b_a, 0] + atom_start + at_i)[lig_mask]
        d_sha = (base[b_a, 0] + bin_a)[lig_mask]
        _push(s_cb, d_sh, rel_cb2s, None)
        _push(d_sh, s_cb, rel_s2cb, None)
        _push(s_at, d_sha, rel_a2s, None)
        _push(d_sha, s_at, rel_s2a, None)
        stats["n_hier"] = float(
            prot_mask.to(prot_xyz.dtype).sum(1).mean()
            + lig_mask.to(lig_xyz.dtype).sum(1).mean())

    if intra:
        dist_pp = torch.cdist(prot_xyz, prot_xyz)
        both = prot_mask.unsqueeze(2) & prot_mask.unsqueeze(1)
        eye = torch.eye(P, dtype=torch.bool, device=device).unsqueeze(0)
        pp = both & ~eye & (dist_pp < PP_CUTOFF)
        if bool(pp.any()):
            b, i, j = pp.nonzero(as_tuple=True)
            s = base[b, 0] + cb_start + i
            d = base[b, 0] + cb_start + j
            _push(s, d, rel_cb2cb, None)
        src_b = batch["lig_bond"][..., 0]
        dst_b = batch["lig_bond"][..., 1]
        bm = batch["bond_mask"]
        if bool(bm.any()):
            bb = torch.arange(B, device=device).unsqueeze(1).expand_as(src_b)
            s = base[bb[bm], 0] + atom_start + src_b[bm]
            d = base[bb[bm], 0] + atom_start + dst_b[bm]
            _push(s, d, rel_a2a, None)
            _push(d, s, rel_a2a, None)

    if inter:
        dist_pl = torch.cdist(prot_xyz, lig_xyz)
        pl = prot_mask.unsqueeze(2) & lig_mask.unsqueeze(1) & (dist_pl < PL_CUTOFF)
        if bool(pl.any()):
            b, i, j = pl.nonzero(as_tuple=True)
            s = base[b, 0] + cb_start + i
            d = base[b, 0] + atom_start + j
            _push(s, d, rel_cb2a, None)
            _push(d, s, rel_a2cb, None)
            stats["n_pl"] = float(pl.to(prot_xyz.dtype).sum(dim=(1, 2)).mean())

    if lm_broadcast:
        cb_i = torch.arange(P, device=device).unsqueeze(0).expand(B, P)
        at_i = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        b_c = torch.arange(B, device=device).unsqueeze(1).expand_as(cb_i)
        b_a = torch.arange(B, device=device).unsqueeze(1).expand_as(at_i)
        p_node = (base[b_c, 0] + prot_idx)[prot_mask]
        c_node = (base[b_c, 0] + cb_start + cb_i)[prot_mask]
        l_node = (base[b_a, 0] + lig_idx)[lig_mask]
        a_node = (base[b_a, 0] + atom_start + at_i)[lig_mask]
        _push(p_node, c_node, rel_p2cb, None)
        _push(c_node, p_node, rel_cb2p, None)
        _push(l_node, a_node, rel_l2a, None)
        _push(a_node, l_node, rel_a2l, None)

    if not srcs:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty, empty, empty, stats
    return (
        torch.cat(srcs), torch.cat(dsts), torch.cat(grps), torch.cat(rels), stats)
