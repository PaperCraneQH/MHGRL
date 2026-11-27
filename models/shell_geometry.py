"""Instance-level geometry written into the 60 ordered shells.

Coordinates never enter the 62-node GAT as extra neighbors.  Each radial
slot gets real distances (scalars, RBF histograms, or a distance-biased
cross-attention over Cβ / heavy atoms).  Output projections are zero-init.
"""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multiscale_3d import (
    N_ELE,
    N_RBF,
    N_RES,
    PL_CUTOFF,
    GeoNodeEmbed,
    ligand_centroid,
    radial_bin,
    rbf,
)

N_FEAT = 8
N_RICH = 8 + N_RBF + N_RBF + N_RBF  # stats + Cβ radius + atom radius + contact
N_TYPE = N_RES + N_ELE
N_PAIR_TYPE = N_RES * N_ELE
N_RICH_PAIR = N_RICH + N_TYPE
N_RICH_PAIRTYPE = N_RICH + N_PAIR_TYPE
N_RBF_PAIRTYPE = N_RBF + N_PAIR_TYPE


def _shell_stats(batch: Dict[str, torch.Tensor], n_shells: int,
                 dtype: torch.dtype) -> torch.Tensor:
    """Return [B, S, 8] real-valued descriptors.  Empty shells stay 0."""
    prot_xyz = batch["prot_xyz"].to(dtype)
    lig_xyz = batch["lig_xyz"].to(dtype)
    prot_mask = batch["prot_mask"]
    lig_mask = batch["lig_mask"]
    B = prot_xyz.size(0)
    cent = ligand_centroid(lig_xyz, lig_mask)
    rad_c = (prot_xyz - cent.unsqueeze(1)).norm(dim=-1)
    rad_a = (lig_xyz - cent.unsqueeze(1)).norm(dim=-1)
    bin_c = radial_bin(rad_c, n_shells)
    bin_a = radial_bin(rad_a, n_shells)
    feats = prot_xyz.new_zeros(B, n_shells, N_FEAT)

    def _fill(bins, mask, values, col_count, col_mean, col_min):
        safe = values.masked_fill(~mask, 0.0)
        ones = mask.to(dtype)
        count = prot_xyz.new_zeros(B, n_shells)
        total = prot_xyz.new_zeros(B, n_shells)
        mins = prot_xyz.new_zeros(B, n_shells)
        count.scatter_add_(1, bins, ones)
        total.scatter_add_(1, bins, safe)
        for s in range(n_shells):
            sel = mask & bins.eq(s)
            if bool(sel.any()):
                mins[:, s] = values.masked_fill(~sel, 1e6).min(dim=1).values
        feats[:, :, col_count] = count
        feats[:, :, col_mean] = total / count.clamp(min=1.0)
        feats[:, :, col_min] = mins.masked_fill(mins > 1e5, 0.0)

    _fill(bin_c, prot_mask, rad_c, 0, 1, 2)
    _fill(bin_a, lig_mask, rad_a, 3, 4, 5)

    dist_pl = torch.cdist(prot_xyz, lig_xyz)
    pl = prot_mask.unsqueeze(2) & lig_mask.unsqueeze(1) & (dist_pl < PL_CUTOFF)
    if bool(pl.any()):
        b, i, j = pl.nonzero(as_tuple=True)
        d = dist_pl[b, i, j]
        s = bin_c[b, i]
        n_pl = prot_xyz.new_zeros(B, n_shells)
        sum_pl = prot_xyz.new_zeros(B, n_shells)
        n_pl.index_put_((b, s), torch.ones_like(d), accumulate=True)
        sum_pl.index_put_((b, s), d, accumulate=True)
        feats[:, :, 6] = n_pl
        feats[:, :, 7] = sum_pl / n_pl.clamp(min=1.0)
    return feats


def _rich_stats(batch: Dict[str, torch.Tensor], n_shells: int,
                dtype: torch.dtype) -> torch.Tensor:
    """8 scalars plus radius / contact RBF.  Empty shells stay 0, including RBF."""
    stats = _shell_stats(batch, n_shells, dtype)
    occupied_c = stats[:, :, 0:1] > 0
    occupied_a = stats[:, :, 3:4] > 0
    rbf_c = rbf(stats[:, :, 1], n_bins=N_RBF, d_max=30.0) * occupied_c.to(dtype)
    rbf_a = rbf(stats[:, :, 4], n_bins=N_RBF, d_max=30.0) * occupied_a.to(dtype)

    prot_xyz = batch["prot_xyz"].to(dtype)
    lig_xyz = batch["lig_xyz"].to(dtype)
    prot_mask = batch["prot_mask"]
    lig_mask = batch["lig_mask"]
    B = prot_xyz.size(0)
    cent = ligand_centroid(lig_xyz, lig_mask)
    rad_c = (prot_xyz - cent.unsqueeze(1)).norm(dim=-1)
    bin_c = radial_bin(rad_c, n_shells)
    hist = prot_xyz.new_zeros(B, n_shells, N_RBF)
    dist_pl = torch.cdist(prot_xyz, lig_xyz)
    pl = prot_mask.unsqueeze(2) & lig_mask.unsqueeze(1) & (dist_pl < PL_CUTOFF)
    if bool(pl.any()):
        b, i, j = pl.nonzero(as_tuple=True)
        d = dist_pl[b, i, j]
        s = bin_c[b, i]
        hist.index_put_((b, s), rbf(d, n_bins=N_RBF, d_max=PL_CUTOFF),
                        accumulate=True)
        cnt = prot_xyz.new_zeros(B, n_shells, 1)
        cnt.index_put_(
            (b, s),
            torch.ones(d.size(0), 1, device=d.device, dtype=dtype),
            accumulate=True)
        hist = hist / cnt.clamp(min=1.0)
    return torch.cat([stats, rbf_c, rbf_a, hist], dim=-1)


def _contact_hist(batch: Dict[str, torch.Tensor], n_shells: int,
                  dtype: torch.dtype) -> torch.Tensor:
    """Per-shell 5 Å contact-distance RBF.  Empty shells stay 0."""
    prot_xyz = batch["prot_xyz"].to(dtype)
    lig_xyz = batch["lig_xyz"].to(dtype)
    prot_mask = batch["prot_mask"]
    lig_mask = batch["lig_mask"]
    B = prot_xyz.size(0)
    cent = ligand_centroid(lig_xyz, lig_mask)
    rad_c = (prot_xyz - cent.unsqueeze(1)).norm(dim=-1)
    bin_c = radial_bin(rad_c, n_shells)
    hist = prot_xyz.new_zeros(B, n_shells, N_RBF)
    dist_pl = torch.cdist(prot_xyz, lig_xyz)
    pl = prot_mask.unsqueeze(2) & lig_mask.unsqueeze(1) & (dist_pl < PL_CUTOFF)
    if bool(pl.any()):
        b, i, j = pl.nonzero(as_tuple=True)
        d = dist_pl[b, i, j]
        s = bin_c[b, i]
        hist.index_put_((b, s), rbf(d, n_bins=N_RBF, d_max=PL_CUTOFF),
                        accumulate=True)
        cnt = prot_xyz.new_zeros(B, n_shells, 1)
        cnt.index_put_(
            (b, s),
            torch.ones(d.size(0), 1, device=d.device, dtype=dtype),
            accumulate=True)
        hist = hist / cnt.clamp(min=1.0)
    return hist


def _contact_types(batch: Dict[str, torch.Tensor], n_shells: int,
                   dtype: torch.dtype) -> torch.Tensor:
    """Mean residue / element occupancy of 5 Å contacts, aligned to shells."""
    prot_xyz = batch["prot_xyz"].to(dtype)
    lig_xyz = batch["lig_xyz"].to(dtype)
    prot_mask = batch["prot_mask"]
    lig_mask = batch["lig_mask"]
    prot_type = batch["prot_type"].clamp(0, N_RES - 1)
    lig_type = batch["lig_type"].clamp(0, N_ELE - 1)
    B = prot_xyz.size(0)
    cent = ligand_centroid(lig_xyz, lig_mask)
    rad_c = (prot_xyz - cent.unsqueeze(1)).norm(dim=-1)
    bin_c = radial_bin(rad_c, n_shells)
    occ_r = prot_xyz.new_zeros(B, n_shells, N_RES)
    occ_e = prot_xyz.new_zeros(B, n_shells, N_ELE)
    dist_pl = torch.cdist(prot_xyz, lig_xyz)
    pl = prot_mask.unsqueeze(2) & lig_mask.unsqueeze(1) & (dist_pl < PL_CUTOFF)
    if bool(pl.any()):
        b, i, j = pl.nonzero(as_tuple=True)
        s = bin_c[b, i]
        occ_r.index_put_(
            (b, s), F.one_hot(prot_type[b, i], N_RES).to(dtype), accumulate=True)
        occ_e.index_put_(
            (b, s), F.one_hot(lig_type[b, j], N_ELE).to(dtype), accumulate=True)
        cnt = prot_xyz.new_zeros(B, n_shells, 1)
        cnt.index_put_(
            (b, s),
            torch.ones(s.size(0), 1, device=s.device, dtype=dtype),
            accumulate=True)
        occ_r = occ_r / cnt.clamp(min=1.0)
        occ_e = occ_e / cnt.clamp(min=1.0)
    return torch.cat([occ_r, occ_e], dim=-1)


def _contact_pair_hist(batch: Dict[str, torch.Tensor], n_shells: int,
                       dtype: torch.dtype) -> torch.Tensor:
    """5 Å residue-type × ligand-element histogram per shell.  Empty shells stay 0."""
    prot_xyz = batch["prot_xyz"].to(dtype)
    lig_xyz = batch["lig_xyz"].to(dtype)
    prot_mask = batch["prot_mask"]
    lig_mask = batch["lig_mask"]
    prot_type = batch["prot_type"].clamp(0, N_RES - 1)
    lig_type = batch["lig_type"].clamp(0, N_ELE - 1)
    B = prot_xyz.size(0)
    cent = ligand_centroid(lig_xyz, lig_mask)
    rad_c = (prot_xyz - cent.unsqueeze(1)).norm(dim=-1)
    bin_c = radial_bin(rad_c, n_shells)
    hist = prot_xyz.new_zeros(B, n_shells, N_PAIR_TYPE)
    dist_pl = torch.cdist(prot_xyz, lig_xyz)
    pl = prot_mask.unsqueeze(2) & lig_mask.unsqueeze(1) & (dist_pl < PL_CUTOFF)
    if bool(pl.any()):
        b, i, j = pl.nonzero(as_tuple=True)
        s = bin_c[b, i]
        t = prot_type[b, i] * N_ELE + lig_type[b, j]
        hist.index_put_(
            (b, s), F.one_hot(t, N_PAIR_TYPE).to(dtype), accumulate=True)
        cnt = prot_xyz.new_zeros(B, n_shells, 1)
        cnt.index_put_(
            (b, s),
            torch.ones(s.size(0), 1, device=s.device, dtype=dtype),
            accumulate=True)
        hist = hist / cnt.clamp(min=1.0)
    return hist


def _contact_types_weighted(batch: Dict[str, torch.Tensor], n_shells: int,
                            dtype: torch.dtype) -> torch.Tensor:
    """Kernel-weight residue and element occupancy for contacts within 5 Å."""
    prot_xyz = batch["prot_xyz"].to(dtype)
    lig_xyz = batch["lig_xyz"].to(dtype)
    prot_mask = batch["prot_mask"]
    lig_mask = batch["lig_mask"]
    prot_type = batch["prot_type"].clamp(0, N_RES - 1)
    lig_type = batch["lig_type"].clamp(0, N_ELE - 1)
    B = prot_xyz.size(0)
    cent = ligand_centroid(lig_xyz, lig_mask)
    rad_c = (prot_xyz - cent.unsqueeze(1)).norm(dim=-1)
    bin_c = radial_bin(rad_c, n_shells)
    occ_r = prot_xyz.new_zeros(B, n_shells, N_RES)
    occ_e = prot_xyz.new_zeros(B, n_shells, N_ELE)
    dist_pl = torch.cdist(prot_xyz, lig_xyz)
    pl = prot_mask.unsqueeze(2) & lig_mask.unsqueeze(1) & (dist_pl < PL_CUTOFF)
    if bool(pl.any()):
        b, i, j = pl.nonzero(as_tuple=True)
        d = dist_pl[b, i, j]
        s = bin_c[b, i]
        w = (1.0 - d / float(PL_CUTOFF)).clamp(min=0.0)
        occ_r.index_put_(
            (b, s), F.one_hot(prot_type[b, i], N_RES).to(dtype) * w.unsqueeze(-1),
            accumulate=True)
        occ_e.index_put_(
            (b, s), F.one_hot(lig_type[b, j], N_ELE).to(dtype) * w.unsqueeze(-1),
            accumulate=True)
        cnt = prot_xyz.new_zeros(B, n_shells, 1)
        cnt.index_put_((b, s), w.unsqueeze(-1), accumulate=True)
        occ_r = occ_r / cnt.clamp(min=1.0)
        occ_e = occ_e / cnt.clamp(min=1.0)
    return torch.cat([occ_r, occ_e], dim=-1)


class ShellInstanceGeometry(nn.Module):
    def __init__(self, node_dim: int = 192, n_shells: int = 60,
                 mode: str = "stats"):
        super().__init__()
        mode = str(mode or "stats").lower()
        if mode not in ("stats", "rich"):
            raise ValueError(f"shell geometry mode must be stats|rich, got {mode!r}")
        self.mode = mode
        self.n_shells = int(n_shells)
        in_dim = N_RICH if mode == "rich" else N_FEAT
        self.proj = nn.Linear(in_dim, int(node_dim))
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.last_abs = 0.0
        self.last_n_cb = 0.0
        self.last_n_atom = 0.0
        self.last_n_pl = 0.0

    def forward(self, batch: Dict[str, torch.Tensor],
                shell: torch.Tensor) -> torch.Tensor:
        if self.mode == "rich":
            feats = _rich_stats(batch, self.n_shells, shell.dtype)
        else:
            feats = _shell_stats(batch, self.n_shells, shell.dtype)
        residual = self.proj(feats)
        with torch.no_grad():
            self.last_abs = float(residual.detach().abs().mean())
            self.last_n_cb = float(batch["prot_mask"].to(shell.dtype).sum(1).mean())
            self.last_n_atom = float(batch["lig_mask"].to(shell.dtype).sum(1).mean())
            self.last_n_pl = float(feats[:, :, 6].sum(1).mean())
        return shell + residual


class ShellTo3DAttention(nn.Module):
    """60 ordered shells retrieve Cβ / ligand atoms.  Not in the main GAT."""

    def __init__(self, node_dim: int = 192, n_shells: int = 60,
                 n_heads: int = 4, dropout: float = 0.1,
                 radius_width: float = 0.5):
        super().__init__()
        dim = int(node_dim)
        if dim % int(n_heads) != 0:
            raise ValueError(f"node_dim {dim} not divisible by n_heads {n_heads}")
        self.n_shells = int(n_shells)
        self.n_heads = int(n_heads)
        self.head_dim = dim // self.n_heads
        self.radius_width = float(radius_width)
        self.geo = GeoNodeEmbed(dim)
        self.rbf_proj = nn.Linear(N_RBF, dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(float(dropout))
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.last_abs = 0.0
        self.last_n_cb = 0.0
        self.last_n_atom = 0.0
        self.last_entropy = 0.0

    def forward(self, batch: Dict[str, torch.Tensor],
                shell: torch.Tensor) -> torch.Tensor:
        dtype = shell.dtype
        cb, atom = self.geo(batch, dtype)
        nodes = torch.cat([cb, atom], dim=1)
        mask = torch.cat([batch["prot_mask"], batch["lig_mask"]], dim=1)
        xyz = torch.cat(
            [batch["prot_xyz"].to(dtype), batch["lig_xyz"].to(dtype)], dim=1)
        cent = ligand_centroid(batch["lig_xyz"].to(dtype), batch["lig_mask"])
        rad = (xyz - cent.unsqueeze(1)).norm(dim=-1)
        nodes = nodes + self.rbf_proj(rbf(rad, n_bins=N_RBF, d_max=30.0))
        nodes = nodes * mask.unsqueeze(-1).to(dtype)

        B, S, D = shell.shape
        H, d = self.n_heads, self.head_dim
        N = nodes.size(1)
        centers = (
            1.0 + 0.5 * torch.arange(S, device=shell.device, dtype=dtype) + 0.25)
        bias = -((rad.unsqueeze(1) - centers.view(1, S, 1)) / self.radius_width) ** 2
        bias = bias.masked_fill(~mask.unsqueeze(1), -1e9)

        q = self.q(shell).view(B, S, H, d).transpose(1, 2)
        k = self.k(nodes).view(B, N, H, d).transpose(1, 2)
        v = self.v(nodes).view(B, N, H, d).transpose(1, 2)
        attn = (q @ k.transpose(-1, -2)) / math.sqrt(d) + bias.unsqueeze(1)
        attn = F.softmax(attn, dim=-1)
        valid = mask.any(dim=1).to(dtype).view(B, 1, 1, 1)
        attn = torch.nan_to_num(attn, nan=0.0) * valid
        ctx = (attn @ v).transpose(1, 2).reshape(B, S, D)
        residual = self.out(self.drop(ctx))
        residual = residual * valid.view(B, 1, 1)
        with torch.no_grad():
            self.last_abs = float(residual.detach().abs().mean())
            self.last_n_cb = float(batch["prot_mask"].to(dtype).sum(1).mean())
            self.last_n_atom = float(batch["lig_mask"].to(dtype).sum(1).mean())
            mass = attn.detach().clamp_min(1e-8)
            self.last_entropy = float((-(mass * mass.log()).sum(-1)).mean())
        return shell + residual


class ShellGeoChannels(nn.Module):
    """Instance geometry aligned to the 60 shells, not added to shell_embed."""

    def __init__(self, mode: str = "contact", n_shells: int = 60):
        super().__init__()
        mode = str(mode or "contact").lower()
        if mode not in ("contact", "rich", "rich_pair", "pair_type",
                        "rich_pairtype", "rbf_pairtype",
                        "rich_pair_w", "rich_pair_proj", "pair_proj85"):
            raise ValueError(
                f"geo channel mode must be contact|rich|rich_pair|"
                f"pair_type|rich_pairtype|rbf_pairtype|"
                f"rich_pair_w|rich_pair_proj|pair_proj85, got {mode!r}")
        self.mode = mode
        self.n_shells = int(n_shells)
        self.out_dim = {
            "contact": N_RBF,
            "rich": N_RICH,
            "rich_pair": N_RICH_PAIR,
            "pair_type": N_PAIR_TYPE,
            "rich_pairtype": N_RICH_PAIRTYPE,
            "rbf_pairtype": N_RBF_PAIRTYPE,
            "rich_pair_w": N_RICH_PAIR,
            "rich_pair_proj": N_RICH_PAIR,
            "pair_proj85": N_RICH_PAIR,
        }[mode]
        self.pair_proj = None
        if mode == "rich_pair_proj":
            self.pair_proj = nn.Linear(N_PAIR_TYPE, N_TYPE)
            nn.init.zeros_(self.pair_proj.weight)
            nn.init.zeros_(self.pair_proj.bias)
        elif mode == "pair_proj85":
            self.pair_proj = nn.Linear(N_RICH + N_PAIR_TYPE, N_RICH_PAIR)
            nn.init.zeros_(self.pair_proj.weight)
            nn.init.zeros_(self.pair_proj.bias)
            with torch.no_grad():
                self.pair_proj.weight[:N_RICH, :N_RICH] = torch.eye(N_RICH)
        self.last_abs = 0.0
        self.last_n_pl = 0.0

    def forward(self, batch: Dict[str, torch.Tensor],
                dtype: torch.dtype) -> torch.Tensor:
        if self.mode == "rich_pair":
            feats = torch.cat([
                _rich_stats(batch, self.n_shells, dtype),
                _contact_types(batch, self.n_shells, dtype),
            ], dim=-1)
        elif self.mode == "rich_pair_w":
            feats = torch.cat([
                _rich_stats(batch, self.n_shells, dtype),
                _contact_types_weighted(batch, self.n_shells, dtype),
            ], dim=-1)
        elif self.mode == "rich_pair_proj":
            rich = _rich_stats(batch, self.n_shells, dtype)
            hist = _contact_pair_hist(batch, self.n_shells, dtype)
            feats = torch.cat([rich, self.pair_proj(hist)], dim=-1)
        elif self.mode == "pair_proj85":
            rich = _rich_stats(batch, self.n_shells, dtype)
            raw = torch.cat([
                rich, _contact_pair_hist(batch, self.n_shells, dtype),
            ], dim=-1)
            feats = self.pair_proj(raw)
        elif self.mode == "rich_pairtype":
            feats = torch.cat([
                _rich_stats(batch, self.n_shells, dtype),
                _contact_pair_hist(batch, self.n_shells, dtype),
            ], dim=-1)
        elif self.mode == "rbf_pairtype":
            feats = torch.cat([
                _contact_hist(batch, self.n_shells, dtype),
                _contact_pair_hist(batch, self.n_shells, dtype),
            ], dim=-1)
        elif self.mode == "pair_type":
            feats = _contact_pair_hist(batch, self.n_shells, dtype)
        elif self.mode == "rich":
            feats = _rich_stats(batch, self.n_shells, dtype)
        else:
            feats = _contact_hist(batch, self.n_shells, dtype)
        with torch.no_grad():
            self.last_abs = float(feats.detach().abs().mean())
            if self.mode in ("rich", "rich_pair", "rich_pairtype",
                             "rich_pair_w", "rich_pair_proj"):
                self.last_n_pl = float(feats[:, :, 6].sum(1).mean())
            elif self.mode == "pair_proj85":
                self.last_n_pl = float(rich[:, :, 6].sum(1).mean())
            else:
                self.last_n_pl = float(
                    (feats.detach().abs().sum(-1) > 0).to(feats.dtype).sum(1).mean())
        return feats


class ShellGeoHead(nn.Module):
    """Small MLP over ordered geo channels.  Last layer starts at zero."""

    def __init__(self, in_dim: int, hidden: int = 384, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.last_abs = 0.0

    def forward(self, geo: torch.Tensor) -> torch.Tensor:
        pred = self.net(geo.reshape(geo.size(0), -1))
        with torch.no_grad():
            self.last_abs = float(pred.detach().abs().mean())
        return pred


class ShellGeoFilm(nn.Module):
    """Post-GAT FiLM on the 60 shells.  Zero-init so start is identity."""

    def __init__(self, geo_dim: int, node_dim: int = 192, use_shift: bool = False):
        super().__init__()
        self.scale = nn.Linear(int(geo_dim), int(node_dim))
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)
        self.shift = None
        if bool(use_shift):
            self.shift = nn.Linear(int(geo_dim), int(node_dim))
            nn.init.zeros_(self.shift.weight)
            nn.init.zeros_(self.shift.bias)
        self.last_abs = 0.0

    def forward(self, shell: torch.Tensor, geo: torch.Tensor) -> torch.Tensor:
        gamma = self.scale(geo)
        out = shell * (1.0 + gamma)
        if self.shift is not None:
            out = out + self.shift(geo)
        with torch.no_grad():
            self.last_abs = float(gamma.detach().abs().mean())
        return out
