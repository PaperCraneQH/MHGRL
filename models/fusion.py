"""Fusion heads for ligand + protein (+ optional structure) representations."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mean(h: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return h.mean(dim=1)
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def _masked_max(h: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is not None:
        neg = torch.finfo(h.dtype).min
        h = h.masked_fill(~mask.bool().unsqueeze(-1), neg)
    return h.max(dim=1).values


def _to_vec(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    return _masked_mean(x, mask) if x.dim() == 3 else x


def _tri_mlp_head(in_dim: int, hidden: int, mid: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, mid),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(mid, 1),
    )


class ConcatMLPFusion(nn.Module):
    def __init__(self, d_lig: int = 128, d_prot: int = 128,
                 hidden: int = 1024, mid: int = 256, dropout: float = 0.2):
        super().__init__()
        self.head = _tri_mlp_head(d_lig + d_prot, hidden, mid, dropout)

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        prot = _to_vec(prot, prot_mask)
        lig = _to_vec(lig, lig_mask)
        return self.head(torch.cat([lig, prot], dim=1))


class BilinearInteractionFusion(nn.Module):
    def __init__(self, d_lig: int = 128, d_prot: int = 128,
                 hidden: int = 1024, mid: int = 256, dropout: float = 0.2):
        super().__init__()
        assert d_lig == d_prot
        self.head = _tri_mlp_head(4 * d_lig, hidden, mid, dropout)

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        prot = _to_vec(prot, prot_mask)
        lig = _to_vec(lig, lig_mask)
        x = torch.cat([lig, prot, lig * prot, (lig - prot).abs()], dim=1)
        return self.head(x)


class CrossAttnGatedFusion(nn.Module):
    def __init__(self, d_lig: int = 128, d_prot: int = 128,
                 n_heads: int = 4, dropout: float = 0.1,
                 use_gate: bool = True):
        super().__init__()
        assert d_lig == d_prot
        d = d_lig
        self.use_gate = bool(use_gate)
        self.attn_lig = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.attn_prot = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        if self.use_gate:
            self.gate = nn.Linear(2 * d, d)
            head_in = d
        else:
            self.gate = None
            head_in = 2 * d
        self.head = nn.Sequential(
            nn.Linear(head_in, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 1)
        )

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert prot.dim() == 3
        lig = _to_vec(lig, lig_mask)
        lig_q = lig.unsqueeze(1)
        key_pad = ~prot_mask.bool() if prot_mask is not None else None
        lig_upd, _ = self.attn_lig(lig_q, prot, prot, key_padding_mask=key_pad)
        if prot_mask is not None:
            m = prot_mask.unsqueeze(-1).float()
            prot_q = (prot * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1.0)
        else:
            prot_q = prot.mean(1, keepdim=True)
        prot_upd, _ = self.attn_prot(prot_q, lig_q, lig_q)
        lig_upd = lig_upd.squeeze(1)
        prot_upd = prot_upd.squeeze(1)
        if self.use_gate:
            g = torch.sigmoid(self.gate(torch.cat([lig_upd, prot_upd], dim=-1)))
            fused = g * lig_upd + (1 - g) * prot_upd
        else:
            fused = torch.cat([lig_upd, prot_upd], dim=-1)
        return self.head(fused)


class LCBCrossAttnFusion(nn.Module):
    def __init__(self, d_lig: int = 128, d_prot: int = 128,
                 n_heads: int = 4, branch_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        assert d_lig == d_prot
        d = d_lig
        self.attn_p = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.attn_l = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.br_p = nn.Linear(d, branch_dim)
        self.br_l = nn.Linear(d, branch_dim)
        self.br_pp = nn.Linear(d, branch_dim)
        self.br_pl = nn.Linear(d, branch_dim)
        self.norm = nn.LayerNorm(4 * branch_dim)
        self.head = nn.Sequential(
            nn.Linear(4 * branch_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert lig.dim() == 3 and prot.dim() == 3
        kp_lig = (~lig_mask.bool()) if lig_mask is not None else None
        kp_prot = (~prot_mask.bool()) if prot_mask is not None else None
        sp_attn, _ = self.attn_p(prot, lig, lig, key_padding_mask=kp_lig)
        sl_attn, _ = self.attn_l(lig, prot, prot, key_padding_mask=kp_prot)
        a_p = F.relu(self.br_p(_masked_max(sp_attn, prot_mask)))
        a_l = F.relu(self.br_l(_masked_max(sl_attn, lig_mask)))
        sp_p = F.relu(self.br_pp(_masked_mean(prot, prot_mask)))
        sp_l = F.relu(self.br_pl(_masked_mean(lig, lig_mask)))
        x = self.norm(torch.cat([a_p, a_l, sp_p, sp_l], dim=1))
        return self.head(x)


class LCBCrossAttnFusion3(nn.Module):
    def __init__(self, d_lig: int = 128, d_prot: int = 128, d_plitext: int = 128,
                 n_heads: int = 4, branch_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        assert d_lig == d_prot
        d = d_lig
        self.attn_p = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.attn_l = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.br_p = nn.Linear(d, branch_dim)
        self.br_l = nn.Linear(d, branch_dim)
        self.br_pp = nn.Linear(d, branch_dim)
        self.br_pl = nn.Linear(d, branch_dim)
        self.br_struct = nn.Linear(d_plitext, branch_dim)
        self.norm = nn.LayerNorm(5 * branch_dim)
        self.head = nn.Sequential(
            nn.Linear(5 * branch_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                plitext: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert lig.dim() == 3 and prot.dim() == 3
        kp_lig = (~lig_mask.bool()) if lig_mask is not None else None
        kp_prot = (~prot_mask.bool()) if prot_mask is not None else None
        sp_attn, _ = self.attn_p(prot, lig, lig, key_padding_mask=kp_lig)
        sl_attn, _ = self.attn_l(lig, prot, prot, key_padding_mask=kp_prot)
        a_p = F.relu(self.br_p(_masked_max(sp_attn, prot_mask)))
        a_l = F.relu(self.br_l(_masked_max(sl_attn, lig_mask)))
        sp_p = F.relu(self.br_pp(_masked_mean(prot, prot_mask)))
        sp_l = F.relu(self.br_pl(_masked_mean(lig, lig_mask)))
        z_s = F.relu(self.br_struct(plitext))
        x = self.norm(torch.cat([a_p, a_l, sp_p, sp_l, z_s], dim=1))
        return self.head(x)


class ConcatMLPFusion3(nn.Module):
    def __init__(self, d_lig: int = 128, d_prot: int = 128, d_plitext: int = 128,
                 hidden: int = 1024, mid: int = 256, dropout: float = 0.2):
        super().__init__()
        self.head = _tri_mlp_head(d_lig + d_prot + d_plitext, hidden, mid, dropout)

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                plitext: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        prot = _to_vec(prot, prot_mask)
        lig = _to_vec(lig, lig_mask)
        return self.head(torch.cat([lig, prot, plitext], dim=1))


class StructOnlyFusion(nn.Module):
    """Structure-only head (component ablation: w/o dual language models).

    Ignores the ligand and protein language-model views and regresses affinity
    from the shell-graph vector alone, isolating the contribution of the PLMs.
    """

    def __init__(self, d_lig: int = 128, d_prot: int = 128, d_plitext: int = 512,
                 hidden: int = 1024, mid: int = 256, dropout: float = 0.2):
        super().__init__()
        self.head = _tri_mlp_head(d_plitext, hidden, mid, dropout)

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                plitext: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.head(plitext)


class BilinearTriFusion(nn.Module):
    """Regress from bilinear and difference features across modalities."""

    def __init__(self, d_lig: int = 128, d_prot: int = 128, d_plitext: int = 128,
                 hidden: int = 1536, mid: int = 640, dropout: float = 0.4):
        super().__init__()
        assert d_lig == d_prot == d_plitext
        d = d_lig
        self.head = _tri_mlp_head(9 * d, hidden, mid, dropout)

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                plitext: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        lig = _to_vec(lig, lig_mask)
        prot = _to_vec(prot, prot_mask)
        s = plitext
        x = torch.cat([
            lig, prot, s,
            lig * prot, lig * s, prot * s,
            (lig - prot).abs(), (lig - s).abs(), (prot - s).abs(),
        ], dim=1)
        return self.head(x)


class GatedTriFusion(nn.Module):
    """Fuse modality features with a learned softmax gate."""

    def __init__(self, d_lig: int = 128, d_prot: int = 128, d_plitext: int = 128,
                 hidden: int = 1536, mid: int = 640, dropout: float = 0.4):
        super().__init__()
        assert d_lig == d_prot == d_plitext
        d = d_lig
        self.gate = nn.Linear(3 * d, 3)
        self.head = _tri_mlp_head(d, hidden, mid, dropout)

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                plitext: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        lig = _to_vec(lig, lig_mask)
        prot = _to_vec(prot, prot_mask)
        s = plitext
        w = F.softmax(self.gate(torch.cat([lig, prot, s], dim=1)), dim=-1)
        fused = w[:, 0:1] * lig + w[:, 1:2] * prot + w[:, 2:3] * s
        return self.head(fused)


class LMXAttnStructFusion(nn.Module):
    """Concatenate cross-attended protein and ligand features with structure features."""

    def __init__(self, d_lig: int = 128, d_prot: int = 128, d_plitext: int = 128,
                 n_heads: int = 4, hidden: int = 1536, mid: int = 640, dropout: float = 0.4):
        super().__init__()
        assert d_lig == d_prot
        d = d_lig
        self.attn_p = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.attn_l = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.head = _tri_mlp_head(2 * d + d_plitext, hidden, mid, dropout)

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                plitext: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert lig.dim() == 3 and prot.dim() == 3
        kp_lig = (~lig_mask.bool()) if lig_mask is not None else None
        kp_prot = (~prot_mask.bool()) if prot_mask is not None else None
        sp_attn, _ = self.attn_p(prot, lig, lig, key_padding_mask=kp_lig)
        sl_attn, _ = self.attn_l(lig, prot, prot, key_padding_mask=kp_prot)
        h_p = _masked_max(sp_attn, prot_mask)
        h_l = _masked_max(sl_attn, lig_mask)
        return self.head(torch.cat([h_p, h_l, plitext], dim=1))


class CosineTriFusion(nn.Module):
    """Predict from mean cosine similarity in a shared latent space."""

    def __init__(self, d_lig: int = 128, d_prot: int = 128, d_plitext: int = 128,
                 proj_dim: int = 256, n_proj_layers: int = 2, dropout: float = 0.1,
                 init_scale: float = 3.0, init_bias: float = 6.5):
        super().__init__()
        self.lig_proj = CosineSimilarityFusion._make_proj(d_lig, proj_dim, n_proj_layers, dropout)
        self.prot_proj = CosineSimilarityFusion._make_proj(d_prot, proj_dim, n_proj_layers, dropout)
        self.struct_proj = CosineSimilarityFusion._make_proj(d_plitext, proj_dim, n_proj_layers, dropout)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.bias = nn.Parameter(torch.tensor(float(init_bias)))

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                plitext: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        lig = _to_vec(lig, lig_mask)
        prot = _to_vec(prot, prot_mask)
        z_l = self.lig_proj(lig)
        z_p = self.prot_proj(prot)
        z_s = self.struct_proj(plitext)
        cos_lp = F.cosine_similarity(z_l, z_p, dim=-1)
        cos_ls = F.cosine_similarity(z_l, z_s, dim=-1)
        cos_ps = F.cosine_similarity(z_p, z_s, dim=-1)
        cos_mean = (cos_lp + cos_ls + cos_ps) / 3.0
        return (self.scale * cos_mean + self.bias).unsqueeze(-1)


class HierarchicalFusion(nn.Module):
    """Combine a bilinear protein–ligand representation with structure features."""

    def __init__(self, d_lig: int = 128, d_prot: int = 128, d_plitext: int = 128,
                 hidden: int = 1536, mid: int = 640, dropout: float = 0.4):
        super().__init__()
        assert d_lig == d_prot
        d = d_lig
        self.lp_proj = nn.Sequential(
            nn.Linear(4 * d, d),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = _tri_mlp_head(d + d_plitext, hidden, mid, dropout)

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                plitext: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        lig = _to_vec(lig, lig_mask)
        prot = _to_vec(prot, prot_mask)
        h_lp = self.lp_proj(torch.cat([lig, prot, lig * prot, (lig - prot).abs()], dim=1))
        return self.head(torch.cat([h_lp, plitext], dim=1))


class CosineSimilarityFusion(nn.Module):
    def __init__(self, d_lig: int = 128, d_prot: int = 128,
                 proj_dim: int = 256, n_proj_layers: int = 2,
                 dropout: float = 0.1,
                 init_scale: float = 3.0, init_bias: float = 6.5):
        super().__init__()
        self.lig_proj = self._make_proj(d_lig, proj_dim, n_proj_layers, dropout)
        self.prot_proj = self._make_proj(d_prot, proj_dim, n_proj_layers, dropout)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.bias = nn.Parameter(torch.tensor(float(init_bias)))

    @staticmethod
    def _make_proj(d_in: int, proj_dim: int, n_layers: int, dropout: float) -> nn.Module:
        layers = []
        d = d_in
        for i in range(max(1, n_layers)):
            last = (i == n_layers - 1)
            layers.append(nn.Linear(d, proj_dim))
            if not last:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            d = proj_dim
        return nn.Sequential(*layers)

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        prot = _to_vec(prot, prot_mask)
        lig = _to_vec(lig, lig_mask)
        z_l = self.lig_proj(lig)
        z_p = self.prot_proj(prot)
        cos = F.cosine_similarity(z_l, z_p, dim=-1)
        return (self.scale * cos + self.bias).unsqueeze(-1)


class WideStructTriFusion(nn.Module):
    """Fuse low-order protein–ligand interactions with a wider structure vector.

    The pooled molecular features have equal width. The structure feature
    enters the prediction head without that width constraint.
    """

    def __init__(self, d_lig: int = 128, d_prot: int = 128, d_plitext: int = 512,
                 hidden: int = 1536, mid: int = 640, dropout: float = 0.3):
        super().__init__()
        assert d_lig == d_prot, 'WideStructTriFusion requires d_lig == d_prot'
        self.head = _tri_mlp_head(4 * d_lig + d_plitext, hidden, mid, dropout)

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                plitext: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        lig = _to_vec(lig, lig_mask)
        prot = _to_vec(prot, prot_mask)
        lm = torch.cat([lig, prot, lig * prot, (lig - prot).abs()], dim=1)
        return self.head(torch.cat([lm, plitext], dim=1))


class RawWideConcatFusion(nn.Module):
    """Concatenate native-width ligand, protein, and structure embeddings.

    Unlike WideStructTriFusion, this head concatenates the pooled vectors
    without projection or explicit protein–ligand interaction terms.
    """

    def __init__(self, d_lig: int = 384, d_prot: int = 320, d_plitext: int = 512,
                 hidden: int = 1536, mid: int = 640, dropout: float = 0.3):
        super().__init__()
        in_dim = d_lig + d_prot + d_plitext
        self.head = _tri_mlp_head(in_dim, hidden, mid, dropout)

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                plitext: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        lig = _to_vec(lig, lig_mask)
        prot = _to_vec(prot, prot_mask)
        return self.head(torch.cat([lig, prot, plitext], dim=1))


class RawWideXAttnFusion(nn.Module):
    """Fuse molecular and structural features via residual cross-attention.

    First exchange information between protein and ligand representations.
    Then exchange information between the molecular and structural branches.
    """

    def __init__(self, d_lig: int = 384, d_prot: int = 320, d_plitext: int = 512,
                 d_attn: int = 256, n_heads: int = 4,
                 attn_dropout: float = 0.1, dropout: float = 0.3,
                 hidden: int = 1536, mid: int = 640,
                 skip_lp_xattn: bool = False):
        super().__init__()
        assert d_attn % n_heads == 0, f'd_attn={d_attn} must be divisible by n_heads={n_heads}'
        self.d_attn = int(d_attn)
        self.skip_lp_xattn = bool(skip_lp_xattn)

        self.lig_proj = nn.Linear(d_lig, d_attn)
        self.prot_proj = nn.Linear(d_prot, d_attn)
        self.struct_proj = nn.Linear(d_plitext, d_attn)
        self.ln_lig = nn.LayerNorm(d_attn)
        self.ln_prot = nn.LayerNorm(d_attn)
        self.ln_struct = nn.LayerNorm(d_attn)

        ad = float(attn_dropout)
        if not self.skip_lp_xattn:
            self.lp_l2p = nn.MultiheadAttention(d_attn, n_heads, dropout=ad, batch_first=True)
            self.lp_p2l = nn.MultiheadAttention(d_attn, n_heads, dropout=ad, batch_first=True)
            self.ln_lp_l = nn.LayerNorm(d_attn)
            self.ln_lp_p = nn.LayerNorm(d_attn)
        self.s2lm = nn.MultiheadAttention(d_attn, n_heads, dropout=ad, batch_first=True)
        self.l2s = nn.MultiheadAttention(d_attn, n_heads, dropout=ad, batch_first=True)
        self.p2s = nn.MultiheadAttention(d_attn, n_heads, dropout=ad, batch_first=True)

        self.drop = nn.Dropout(ad)
        self.ln_s2lm = nn.LayerNorm(d_attn)
        self.ln_l2s = nn.LayerNorm(d_attn)
        self.ln_p2s = nn.LayerNorm(d_attn)

        self.head = _tri_mlp_head(3 * d_attn, hidden, mid, dropout)

    @staticmethod
    def _xattn_residual(q: torch.Tensor, mem: torch.Tensor,
                        attn: nn.MultiheadAttention,
                        ln: nn.LayerNorm, drop: nn.Dropout) -> torch.Tensor:
        """q [B,d], mem [B,L,d] → LN(q + Dropout(Attn(q, mem)))."""
        out, _ = attn(q.unsqueeze(1), mem, mem)
        return ln(q + drop(out.squeeze(1)))

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                plitext: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        lig = self.ln_lig(self.lig_proj(_to_vec(lig, lig_mask)))
        prot = self.ln_prot(self.prot_proj(_to_vec(prot, prot_mask)))
        struct = self.ln_struct(self.struct_proj(plitext))

        if self.skip_lp_xattn:
            lig_x, prot_x = lig, prot
        else:
            lig_t = lig.unsqueeze(1)
            prot_t = prot.unsqueeze(1)
            lig_x = self._xattn_residual(lig, prot_t, self.lp_l2p, self.ln_lp_l, self.drop)
            prot_x = self._xattn_residual(prot, lig_t, self.lp_p2l, self.ln_lp_p, self.drop)

        # Stage-B: struct <- [lig, prot]
        lm_mem = torch.stack([lig_x, prot_x], dim=1)
        struct_x = self._xattn_residual(struct, lm_mem, self.s2lm, self.ln_s2lm, self.drop)
        struct_kv = struct_x.unsqueeze(1)

        # Stage-B: lig/prot <- struct (separate queries to avoid shared-KV bias)
        lig_out = self._xattn_residual(lig_x, struct_kv, self.l2s, self.ln_l2s, self.drop)
        prot_out = self._xattn_residual(prot_x, struct_kv, self.p2s, self.ln_p2s, self.drop)

        return self.head(torch.cat([lig_out, prot_out, struct_x], dim=1))


class SymWideXAttnFusion(nn.Module):
    """Fuse three modalities symmetrically.

    Each modality attends to a two-token memory containing the other two.
    The head concatenates the three updated vectors.
    """

    def __init__(self, d_lig: int = 384, d_prot: int = 320, d_plitext: int = 512,
                 d_attn: int = 384, n_heads: int = 6,
                 attn_dropout: float = 0.1, dropout: float = 0.3,
                 hidden: int = 1536, mid: int = 640):
        super().__init__()
        assert d_attn % n_heads == 0, f'd_attn={d_attn} must be divisible by n_heads={n_heads}'
        self.d_attn = int(d_attn)

        self.lig_proj = nn.Linear(d_lig, d_attn)
        self.prot_proj = nn.Linear(d_prot, d_attn)
        self.struct_proj = nn.Linear(d_plitext, d_attn)
        self.ln_lig = nn.LayerNorm(d_attn)
        self.ln_prot = nn.LayerNorm(d_attn)
        self.ln_struct = nn.LayerNorm(d_attn)

        ad = float(attn_dropout)
        self.attn_lig = nn.MultiheadAttention(d_attn, n_heads, dropout=ad, batch_first=True)
        self.attn_prot = nn.MultiheadAttention(d_attn, n_heads, dropout=ad, batch_first=True)
        self.attn_struct = nn.MultiheadAttention(d_attn, n_heads, dropout=ad, batch_first=True)
        self.ln_out_lig = nn.LayerNorm(d_attn)
        self.ln_out_prot = nn.LayerNorm(d_attn)
        self.ln_out_struct = nn.LayerNorm(d_attn)
        self.drop = nn.Dropout(ad)

        self.head = _tri_mlp_head(3 * d_attn, hidden, mid, dropout)

    def _xattn(self, q: torch.Tensor, mem: torch.Tensor,
               attn: nn.MultiheadAttention, ln: nn.LayerNorm) -> torch.Tensor:
        """q [B,d], mem [B,L,d] → LN(q + Dropout(Attn(q, mem)))."""
        out, _ = attn(q.unsqueeze(1), mem, mem)
        return ln(q + self.drop(out.squeeze(1)))

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                plitext: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        z_l = self.ln_lig(self.lig_proj(_to_vec(lig, lig_mask)))
        z_p = self.ln_prot(self.prot_proj(_to_vec(prot, prot_mask)))
        z_c = self.ln_struct(self.struct_proj(plitext))

        # Each modality attends concat of the other two (2-token memory)
        mem_l = torch.stack([z_p, z_c], dim=1)   # lig <- [prot; struct]
        mem_p = torch.stack([z_l, z_c], dim=1)   # prot <- [lig; struct]
        mem_c = torch.stack([z_l, z_p], dim=1)   # struct <- [lig; prot]

        z_l2 = self._xattn(z_l, mem_l, self.attn_lig, self.ln_out_lig)
        z_p2 = self._xattn(z_p, mem_p, self.attn_prot, self.ln_out_prot)
        z_c2 = self._xattn(z_c, mem_c, self.attn_struct, self.ln_out_struct)

        return self.head(torch.cat([z_l2, z_p2, z_c2], dim=1))


class GatedResidualTriFusion(nn.Module):
    """Mix molecular, structure, and joint predictions with a per-sample gate.

    Training also returns the two separate branch outputs for auxiliary losses.
    """

    def __init__(self, d_lig: int = 128, d_prot: int = 128, d_plitext: int = 512,
                 hidden: int = 1536, mid: int = 640, dropout: float = 0.3,
                 modality_dropout: float = 0.0, gate_hidden: int = 128,
                 init_joint_bias: float = 0.0):
        super().__init__()
        assert d_lig == d_prot, 'GatedResidualTriFusion requires d_lig == d_prot'
        self.modality_dropout = float(modality_dropout)
        lm_dim = 4 * d_lig
        self.lm_head = _tri_mlp_head(lm_dim, hidden, mid, dropout)
        self.st_head = _tri_mlp_head(d_plitext, hidden, mid, dropout)
        self.fuse_head = _tri_mlp_head(lm_dim + d_plitext, hidden, mid, dropout)
        self.gate = nn.Sequential(
            nn.Linear(2 * d_lig + d_plitext, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 3),
        )
        if init_joint_bias != 0.0:
            with torch.no_grad():
                self.gate[-1].bias.copy_(
                    torch.tensor([0.0, 0.0, float(init_joint_bias)]))

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                plitext: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None):
        lig = _to_vec(lig, lig_mask)
        prot = _to_vec(prot, prot_mask)
        lm_feat = torch.cat([lig, prot, lig * prot, (lig - prot).abs()], dim=1)

        y_lm = self.lm_head(lm_feat)
        y_st = self.st_head(plitext)
        y_fuse = self.fuse_head(torch.cat([lm_feat, plitext], dim=1))

        logits = self.gate(torch.cat([lig, prot, plitext], dim=1))   # [B, 3]
        if self.training and self.modality_dropout > 0.0:
            B = logits.size(0)
            r = torch.rand(B, device=logits.device)
            neg = torch.finfo(logits.dtype).min
            drop_lm = r < (self.modality_dropout / 2.0)               # mask LM expert
            drop_st = (r >= (self.modality_dropout / 2.0)) & (r < self.modality_dropout)
            logits = logits.clone()
            logits[drop_lm, 0] = neg
            logits[drop_st, 1] = neg
        w = torch.softmax(logits, dim=-1)                            # [B, 3]
        y = w[:, 0:1] * y_lm + w[:, 1:2] * y_st + w[:, 2:3] * y_fuse

        if self.training:
            return y, [y_lm, y_st]
        return y


class HeteroShellGraphFusion(nn.Module):
    """Use shell nodes as queries over ligand tokens and protein residues.

    Each cross-attention branch contributes a gated residual. The prediction
    head takes the flattened shell representations. The optional randomize_lm
    control replaces the language-model embeddings with same-shape noise.
    """

    def __init__(self, d_lig: int = 128, d_prot: int = 128, node_dim: int = 128,
                 n_shells: int = 60, n_layers: int = 1, n_heads: int = 4,
                 dropout: float = 0.2, use_ffn: bool = True, ffn_mult: int = 2,
                 head_hidden: int = 512, head_mid: int = 256,
                 gate_init: float = -4.0, randomize_lm: bool = False,
                 disable_lig: bool = False, disable_prot: bool = False):
        super().__init__()
        assert d_lig == node_dim and d_prot == node_dim, \
            f'HeteroShellGraphFusion requires d_lig==d_prot==node_dim, got {d_lig},{d_prot},{node_dim}'
        if disable_lig and disable_prot:
            raise ValueError('Cannot disable both disable_lig and disable_prot (keep at least one LM injection path)')
        self.n_shells = int(n_shells)
        self.node_dim = int(node_dim)
        self.n_layers = int(n_layers)
        self.randomize_lm = bool(randomize_lm)
        self.use_ffn = bool(use_ffn)
        # Ablation: ligand-only or protein-only injection.
        self.disable_lig = bool(disable_lig)
        self.disable_prot = bool(disable_prot)

        if not self.disable_lig:
            self.ln_lig_q = nn.ModuleList([nn.LayerNorm(node_dim) for _ in range(n_layers)])
            self.attn_lig = nn.ModuleList([
                nn.MultiheadAttention(node_dim, n_heads, dropout=dropout, batch_first=True)
                for _ in range(n_layers)])
            self.gate_lig = nn.Parameter(torch.full((n_layers,), float(gate_init)))
        if not self.disable_prot:
            self.ln_prot_q = nn.ModuleList([nn.LayerNorm(node_dim) for _ in range(n_layers)])
            self.attn_prot = nn.ModuleList([
                nn.MultiheadAttention(node_dim, n_heads, dropout=dropout, batch_first=True)
                for _ in range(n_layers)])
            self.gate_prot = nn.Parameter(torch.full((n_layers,), float(gate_init)))
        if self.use_ffn:
            self.ffn_ln = nn.ModuleList([nn.LayerNorm(node_dim) for _ in range(n_layers)])
            self.ffn = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(node_dim, ffn_mult * node_dim), nn.GELU(),
                    nn.Dropout(dropout), nn.Linear(ffn_mult * node_dim, node_dim),
                ) for _ in range(n_layers)])
            self.gate_ffn = nn.Parameter(torch.full((n_layers,), float(gate_init)))
        self.head = _tri_mlp_head(self.n_shells * node_dim, head_hidden, head_mid, dropout)

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                plitext: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert lig.dim() == 3 and prot.dim() == 3 and plitext.dim() == 3, \
            'F08 requires sequence-level lig/prot (per_token/per_residue) and shell nodes (return_nodes=True)'
        h = plitext                                          # [B, n_shells, d] shell nodes
        if self.randomize_lm:  # Random-embedding control.
            lig = torch.randn_like(lig)
            prot = torch.randn_like(prot)
        kp_lig = (~lig_mask.bool()) if lig_mask is not None else None
        kp_prot = (~prot_mask.bool()) if prot_mask is not None else None
        for i in range(self.n_layers):
            if not self.disable_lig:
                a_lig, _ = self.attn_lig[i](self.ln_lig_q[i](h), lig, lig, key_padding_mask=kp_lig)
                h = h + torch.sigmoid(self.gate_lig[i]) * a_lig
            if not self.disable_prot:
                a_prot, _ = self.attn_prot[i](self.ln_prot_q[i](h), prot, prot, key_padding_mask=kp_prot)
                h = h + torch.sigmoid(self.gate_prot[i]) * a_prot
            if self.use_ffn:
                h = h + torch.sigmoid(self.gate_ffn[i]) * self.ffn[i](self.ffn_ln[i](h))
        flat = h.reshape(h.size(0), -1)
        return self.head(flat)


def _fusion_common_kwargs(cfg: dict) -> dict:
    return dict(
        hidden=int(cfg.get('hidden', 1024)),
        mid=int(cfg.get('mid', 256)),
        dropout=float(cfg.get('dropout', 0.2)),
    )


def build_fusion(cfg: dict, d_lig: int = 128, d_prot: int = 128,
                 d_plitext: int = 0) -> nn.Module:
    f_type = cfg.get('type', 'concat_mlp')
    kw = _fusion_common_kwargs(cfg)

    if f_type == 'concat_mlp':
        if d_plitext > 0:
            return ConcatMLPFusion3(d_lig=d_lig, d_prot=d_prot, d_plitext=d_plitext, **kw)
        return ConcatMLPFusion(d_lig=d_lig, d_prot=d_prot, **kw)

    if f_type in ('struct_only', 'struct_mlp'):
        assert d_plitext > 0, 'struct_only requires structure tower (plitext.enabled=true)'
        return StructOnlyFusion(d_plitext=d_plitext, **kw)

    if f_type == 'bilinear':
        if d_plitext > 0:
            raise NotImplementedError('Use bilinear_tri for three-branch bilinear fusion.')
        return BilinearInteractionFusion(d_lig=d_lig, d_prot=d_prot, **kw)

    if f_type == 'bilinear_tri':
        assert d_plitext > 0
        return BilinearTriFusion(d_lig=d_lig, d_prot=d_prot, d_plitext=d_plitext, **kw)

    if f_type == 'gated_tri':
        assert d_plitext > 0
        return GatedTriFusion(d_lig=d_lig, d_prot=d_prot, d_plitext=d_plitext, **kw)

    if f_type in ('wide_struct_tri', 'wide_tri'):
        assert d_plitext > 0
        return WideStructTriFusion(d_lig=d_lig, d_prot=d_prot, d_plitext=d_plitext, **kw)

    if f_type in ('raw_wide_concat', 'e1_raw', 'raw_concat_tri'):
        assert d_plitext > 0
        return RawWideConcatFusion(
            d_lig=d_lig, d_prot=d_prot, d_plitext=d_plitext, **kw)

    if f_type in ('raw_wide_xattn', 'e1_rawxattn', 'raw_xattn_tri'):
        assert d_plitext > 0
        return RawWideXAttnFusion(
            d_lig=d_lig, d_prot=d_prot, d_plitext=d_plitext,
            d_attn=int(cfg.get('d_attn', 256)),
            n_heads=int(cfg.get('n_heads', 4)),
            attn_dropout=float(cfg.get('attn_dropout', 0.1)),
            skip_lp_xattn=bool(cfg.get('skip_lp_xattn', False)),
            **kw)

    if f_type in ('sym_wide_xattn', 'sym_xattn_tri', 'symxattn'):
        assert d_plitext > 0
        return SymWideXAttnFusion(
            d_lig=d_lig, d_prot=d_prot, d_plitext=d_plitext,
            d_attn=int(cfg.get('d_attn', 384)),
            n_heads=int(cfg.get('n_heads', 6)),
            attn_dropout=float(cfg.get('attn_dropout', 0.1)),
            **kw)

    if f_type in ('gated_residual_tri', 'gated_res_tri', 'moe_tri'):
        assert d_plitext > 0
        return GatedResidualTriFusion(
            d_lig=d_lig, d_prot=d_prot, d_plitext=d_plitext,
            modality_dropout=float(cfg.get('modality_dropout', 0.0)),
            gate_hidden=int(cfg.get('gate_hidden', 128)),
            init_joint_bias=float(cfg.get('init_joint_bias', 0.0)),
            **kw)

    if f_type in ('hetero_node_graph_pooled', 'heteroshell62'):
        assert d_plitext > 0, (
            'hetero_node_graph_pooled requires plitext.enabled=true and '
            'plitext.return_graph_inputs=true')
        from .hetero_node_graph import HeteroNodeGraphPooledFusion
        gate_init = cfg.get('gate_init', None)
        return HeteroNodeGraphPooledFusion(
            d_lig=d_lig,
            d_prot=d_prot,
            node_dim=d_plitext,
            n_shells=int(cfg.get('n_shells', 60)),
            n_layers=int(cfg.get('n_layers', 3)),
            n_heads=int(cfg.get('n_heads', 4)),
            dropout=float(cfg.get('dropout', 0.2)),
            use_ffn=bool(cfg.get('use_ffn', True)),
            ffn_mult=int(cfg.get('ffn_mult', 2)),
            graph_update_mode=str(cfg.get('graph_update_mode', 'legacy_parallel')),
            gnn_type=str(cfg.get('gnn_type', 'GAT_GCN')),
            norm_style=str(cfg.get('norm_style', 'pre')),
            head_hidden=int(cfg.get('head_hidden', 512)),
            head_mid=int(cfg.get('head_mid', 256)),
            k_shell=int(cfg.get('k_shell', 2)),
            shell_edge_mode=str(cfg.get(
                'shell_edge_mode', 'physical_chain')),
            shell_edge_layer_modes=cfg.get('shell_edge_layer_modes'),
            similarity_topk_k=(
                None if cfg.get('similarity_topk_k') is None
                else int(cfg['similarity_topk_k'])),
            similarity_topk_update=str(cfg.get(
                'similarity_topk_update', 'per_layer')),
            similarity_topk_scope=str(cfg.get(
                'similarity_topk_scope', 'shell_only')),
            shell_upper_edges_angstrom=cfg.get(
                'shell_upper_edges_angstrom'),
            connect_prot_lig=bool(cfg.get('connect_prot_lig', False)),
            lm_shell_edge_mode=str(cfg.get('lm_shell_edge_mode', 'bidirectional')),
            lm_shell_topk=int(cfg.get('lm_shell_topk', -1)),
            lm_shell_topk_scope=str(cfg.get(
                'lm_shell_topk_scope', 'prot')),
            readout_mode=str(cfg.get('readout_mode', 'all_nodes')),
            gate_init=(None if gate_init is None else float(gate_init)),
            feedback_alpha_init=(
                None if cfg.get('feedback_alpha_init') is None
                else float(cfg['feedback_alpha_init'])
            ),
            shell_proj_dim=int(cfg.get('shell_proj_dim', 512)),
            e1_head_hidden=int(cfg.get('e1_head_hidden', 1536)),
            e1_head_mid=int(cfg.get('e1_head_mid', 640)),
            e1_dropout=float(cfg.get('e1_dropout', 0.3)),
            sym_readout_dim=int(cfg.get('sym_readout_dim', 384)),
            sym_readout_heads=int(cfg.get('sym_readout_heads', 6)),
            sym_attn_dropout=float(cfg.get('sym_attn_dropout', 0.1)),
            lm_node_norm=str(cfg.get('lm_node_norm', 'layernorm')),
            edge_mode=str(cfg.get('edge_mode', 'topology_only')),
            edge_attr_mode=str(cfg.get('edge_attr_mode', 'full')),
            edge_attr_features=str(cfg.get(
                'edge_attr_features', 'basic')),
            edge_embed_dim=int(cfg.get('edge_embed_dim', 32)),
            edge_flatten_pool=bool(cfg.get('edge_flatten_pool', False)),
            edge_flatten_scope=str(cfg.get('edge_flatten_scope', 'all')),
            edge_flatten_attn=bool(cfg.get('edge_flatten_attn', False)),
            edge_attr_eps=float(cfg.get('edge_attr_eps', 1e-8)),
            add_self_loops=bool(cfg.get('add_self_loops', True)),
            smooth_mlp_dims=list(cfg.get(
                'smooth_mlp_dims', [3072, 1536, 768, 384, 128])),
            smooth_mlp_dropout=float(cfg.get('smooth_mlp_dropout', 0.2)),
            edge_pair_dim=int(cfg.get('edge_pair_dim', 192)),
            dual_query_heads=int(cfg.get('dual_query_heads', 4)),
            dual_query_hidden=int(cfg.get('dual_query_hidden', 512)),
            dual_query_mid=int(cfg.get('dual_query_mid', 128)),
            dual_query_dropout=float(cfg.get('dual_query_dropout', 0.2)),
            pool_mode=str(cfg.get('pool_mode', 'mean')),
            pool_hidden=int(cfg.get('pool_hidden', 384)),
            pool_mid=int(cfg.get('pool_mid', 128)),
            pool_dropout=float(cfg.get('pool_dropout', 0.3)),
            set2set_steps=int(cfg.get('set2set_steps', 3)),
            radial_pool_bins=int(cfg.get('radial_pool_bins', 10)),
            lowrank_rank=int(cfg.get('lowrank_rank', 512)),
            lowrank_hidden=int(cfg.get('lowrank_hidden', 1536)),
            lowrank_latent=int(cfg.get('lowrank_latent', 384)),
            lowrank_dropout=float(cfg.get('lowrank_dropout', 0.3)),
            lowrank_use_pl_interaction=bool(cfg.get(
                'lowrank_use_pl_interaction', False)),
            lowrank_use_radial_branch=bool(cfg.get(
                'lowrank_use_radial_branch', False)),
            lowrank_radial_bins=int(cfg.get('lowrank_radial_bins', 10)),
            lowrank_branch_gate_init=float(cfg.get(
                'lowrank_branch_gate_init', -2.0)),
            full_residual_hidden=int(cfg.get('full_residual_hidden', 1536)),
            full_residual_latent=int(cfg.get('full_residual_latent', 384)),
            full_residual_dropout=float(cfg.get(
                'full_residual_dropout', 0.3)),
            full_residual_use_pl=bool(cfg.get(
                'full_residual_use_pl', False)),
            full_residual_use_radial=bool(cfg.get(
                'full_residual_use_radial', False)),
            full_residual_radial_bins=int(cfg.get(
                'full_residual_radial_bins', 10)),
            full_residual_gate_init=float(cfg.get(
                'full_residual_gate_init', -3.0)),
            layer_mix_mode=str(cfg.get('layer_mix_mode', 'none')),
            layer_mix_gate_init=float(cfg.get(
                'layer_mix_gate_init', -3.0)),
            relation_calibration=bool(cfg.get(
                'relation_calibration', False)),
            edge_score_path=bool(cfg.get('edge_score_path', True)),
            edge_value_path=bool(cfg.get('edge_value_path', True)),
            edge_value_film=bool(cfg.get('edge_value_film', False)),
            layerscale_init=(
                None if cfg.get('layerscale_init') is None
                else float(cfg['layerscale_init'])),
            attention_grouping=str(cfg.get(
                'attention_grouping', 'destination')),
            relation_fusion=str(cfg.get('relation_fusion', 'sum')),
            relation_adapter_rank=int(cfg.get(
                'relation_adapter_rank', 0)),
            relation_adapter_paths=str(cfg.get(
                'relation_adapter_paths', 'kv')),
            relation_adapter_alpha=float(cfg.get(
                'relation_adapter_alpha', 1.0)),
            qk_norm=bool(cfg.get('qk_norm', False)),
            qk_temperature=str(cfg.get('qk_temperature', 'fixed')),
            spatial_bias=str(cfg.get('spatial_bias', 'none')),
            init_residual=str(cfg.get('init_residual', 'none')),
            extra_modulate=str(cfg.get('extra_modulate', 'none')),
            edge_flatten_skip=str(cfg.get('edge_flatten_skip', 'none')),
            post_graph_glu=str(cfg.get('post_graph_glu', 'none')),
            relation_schedule=str(cfg.get('relation_schedule', 'all')),
            edge_dropout=float(cfg.get('edge_dropout', 0.0)),
            edge_dropout_scope=str(cfg.get(
                'edge_dropout_scope', 'cross')),
            type_adapter_dim=int(cfg.get('type_adapter_dim', 0)),
            type_adapter_scope=str(cfg.get(
                'type_adapter_scope', 'all')),
            post_graph_radial_mode=str(cfg.get(
                'post_graph_radial_mode', 'none')),
            post_graph_radial_gate_init=float(cfg.get(
                'post_graph_radial_gate_init', -3.0)),
            post_graph_radial_order=str(cfg.get(
                'post_graph_radial_order', 'radial')),
            use_type_embedding=bool(cfg.get('use_type_embedding', True)),
            readout_node_dropout=float(cfg.get(
                'readout_node_dropout', 0.0)),
            readout_feature_dropout=float(cfg.get(
                'readout_feature_dropout', 0.0)),
            use_ehign_aux=bool(cfg.get('use_ehign_aux', False)),
            output_affine=bool(cfg.get('output_affine', False)),
            output_affine_scale_init=float(cfg.get(
                'output_affine_scale_init', 1.0)),
            randomize_lm=bool(cfg.get('randomize_lm', False)),
            disable_lig=bool(cfg.get('disable_lig', False)),
            disable_prot=bool(cfg.get('disable_prot', False)),
            feature_dir_cfg=(cfg.get('feature_dir') or None),
            pocket_attention_cfg=(cfg.get('pocket_attention') or None),
            ligand_attention_cfg=(cfg.get('ligand_attention') or None),
            edge_state_update=bool(cfg.get('edge_state_update', False)),
            edge_flatten_update=bool(cfg.get('edge_flatten_update', False)),
            edge_flatten_update_scope=str(
                cfg.get('edge_flatten_update_scope', 'all')),
            edge_flatten_update_extra=str(
                cfg.get('edge_flatten_update_extra', 'none')),
            pl_interact=str(cfg.get('pl_interact', 'none')),
            jk_flatten=str(cfg.get('jk_flatten', 'none')),
            jk_flatten_dim=int(cfg.get('jk_flatten_dim', 32)),
            diff_flatten=str(cfg.get('diff_flatten', 'none')),
            diff_flatten_dim=int(cfg.get('diff_flatten_dim', 32)),
            diff_flatten_steps=int(cfg.get('diff_flatten_steps', 10)),
            diff_flatten_alpha=float(cfg.get('diff_flatten_alpha', 0.1)),
            edge_update_hidden=int(cfg.get('edge_update_hidden', 64)),
            pair_readout=str(cfg.get('pair_readout', 'none')),
            pair_readout_scope=str(cfg.get('pair_readout_scope', 'cross')),
            pair_readout_hidden=int(cfg.get('pair_readout_hidden', 64)),
            pair_to_node=bool(cfg.get('pair_to_node', False)),
            pair_gate_init=float(cfg.get('pair_gate_init', -3.0)),
            pair_dense_layers=int(cfg.get('pair_dense_layers', 2)),
            pair_dense_dim=int(cfg.get('pair_dense_dim', 32)),
            pair_dense_hidden=int(cfg.get('pair_dense_hidden', 64)),
            pair_flatten_mode=str(cfg.get('pair_flatten_mode') or 'off'),
            pair_flatten_hidden=int(cfg.get('pair_flatten_hidden', 256)),
            add_pocket_node=bool(cfg.get('add_pocket_node', False)),
            pocket_node_cfg=(cfg.get('pocket_node') or None),
            signed_shell_residual=bool(cfg.get('signed_shell_residual', False)),
            coarse_interface_cfg=(cfg.get('coarse_interface') or None),
            multiscale_3d_cfg=(cfg.get('multiscale_3d') or None),
            shell_geometry_cfg=(cfg.get('shell_geometry') or None),
            shell_to_3d_cfg=(cfg.get('shell_to_3d') or None),
            shell_geo_readout_cfg=(cfg.get('shell_geo_readout') or None),
            interface_tokens_cfg=(cfg.get('interface_tokens') or None),
            side_tower_cfg=(cfg.get('side_tower') or None),
            contact_primary_cfg=(cfg.get('contact_primary') or None),
        )

    if f_type in ('hetero_shell_graph', 'f08'):
        assert d_plitext > 0, 'hetero_shell_graph requires struct tower (plitext.enabled=true, return_nodes=true)'
        return HeteroShellGraphFusion(
            d_lig=d_lig, d_prot=d_prot, node_dim=d_plitext,
            n_shells=int(cfg.get('n_shells', 60)),
            n_layers=int(cfg.get('xattn_layers', 1)),
            n_heads=int(cfg.get('n_heads', 4)),
            dropout=float(cfg.get('dropout', 0.2)),
            use_ffn=bool(cfg.get('use_ffn', True)),
            ffn_mult=int(cfg.get('ffn_mult', 2)),
            head_hidden=int(cfg.get('head_hidden', 512)),
            head_mid=int(cfg.get('head_mid', 256)),
            gate_init=float(cfg.get('gate_init', -4.0)),
            randomize_lm=bool(cfg.get('randomize_lm', False)),
            disable_lig=bool(cfg.get('disable_lig', False)),
            disable_prot=bool(cfg.get('disable_prot', False)),
        )

    if f_type == 'lm_xattn_struct':
        assert d_plitext > 0
        return LMXAttnStructFusion(
            d_lig=d_lig, d_prot=d_prot, d_plitext=d_plitext,
            n_heads=int(cfg.get('n_heads', 4)), **kw)

    if f_type == 'cosine_tri':
        assert d_plitext > 0
        return CosineTriFusion(
            d_lig=d_lig, d_prot=d_prot, d_plitext=d_plitext,
            proj_dim=int(cfg.get('proj_dim', 256)),
            n_proj_layers=int(cfg.get('n_proj_layers', 2)),
            dropout=float(cfg.get('dropout', 0.1)),
            init_scale=float(cfg.get('init_scale', 3.0)),
            init_bias=float(cfg.get('init_bias', 6.5)),
        )

    if f_type == 'hierarchical':
        assert d_plitext > 0
        return HierarchicalFusion(d_lig=d_lig, d_prot=d_prot, d_plitext=d_plitext, **kw)

    if f_type == 'cross_attn':
        if d_plitext > 0:
            raise NotImplementedError('cross_attn with structure branch is not supported.')
        return CrossAttnGatedFusion(
            d_lig=d_lig, d_prot=d_prot,
            n_heads=int(cfg.get('n_heads', 4)),
            dropout=float(cfg.get('dropout', 0.1)),
            use_gate=bool(cfg.get('use_gate', True)),
        )

    if f_type in ('lcb_cross_attn', 'lcb'):
        if d_plitext > 0:
            return LCBCrossAttnFusion3(
                d_lig=d_lig, d_prot=d_prot, d_plitext=d_plitext,
                n_heads=int(cfg.get('n_heads', 4)),
                branch_dim=int(cfg.get('branch_dim', 256)),
                dropout=float(cfg.get('dropout', 0.2)),
            )
        return LCBCrossAttnFusion(
            d_lig=d_lig, d_prot=d_prot,
            n_heads=int(cfg.get('n_heads', 4)),
            branch_dim=int(cfg.get('branch_dim', 256)),
            dropout=float(cfg.get('dropout', 0.2)),
        )

    if f_type in ('cosine', 'cosine_sim', 'balm'):
        if d_plitext > 0:
            raise NotImplementedError('Use cosine_tri for three-branch cosine fusion.')
        return CosineSimilarityFusion(
            d_lig=d_lig, d_prot=d_prot,
            proj_dim=int(cfg.get('proj_dim', 256)),
            n_proj_layers=int(cfg.get('n_proj_layers', 2)),
            dropout=float(cfg.get('dropout', 0.1)),
            init_scale=float(cfg.get('init_scale', 3.0)),
            init_bias=float(cfg.get('init_bias', 6.5)),
        )

    raise ValueError(f'Unknown fusion type: {f_type}')
