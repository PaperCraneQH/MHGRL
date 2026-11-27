"""Heterogeneous graph fusion for shell and molecular representations.

HeteroNodeGraphPooledFusion uses a pooled protein node, a pooled ligand node,
and a fixed number of shell nodes. HeteroNodeGraphFusion is a variable-size
alternative that adds residue and ligand-token nodes and reads out shell nodes.
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _validated_shell_upper_edges(
        n_shells: int, values=None) -> torch.Tensor:
    """Return the physical upper radius of each shell."""
    n_shells = int(n_shells)
    if n_shells <= 0:
        raise ValueError(f'n_shells must be positive, got {n_shells}')
    if values is None:
        return 1.0 + 0.5 * torch.arange(n_shells, dtype=torch.float32)
    raw = torch.as_tensor(values, dtype=torch.float32)
    if raw.ndim != 1 or raw.numel() != n_shells:
        raise ValueError(
            'shell_upper_edges_angstrom must be a one-dimensional sequence '
            f'of length {n_shells}, got shape={tuple(raw.shape)}')
    if not torch.isfinite(raw).all():
        raise ValueError('shell_upper_edges_angstrom must be finite')
    if not bool((raw > 0).all()):
        raise ValueError('shell_upper_edges_angstrom must be strictly positive')
    if raw.numel() > 1 and not bool((raw[1:] > raw[:-1]).all()):
        raise ValueError('shell_upper_edges_angstrom must be strictly increasing')
    return raw.clone()

_LM_NODE_NORM_MODES = (
    'none', 'layernorm', 'batchnorm', 'l2', 'rmsnorm', 'zscore',
)


class _L2NormDim(nn.Module):
    """Normalize each node vector to unit L2 norm."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1)


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class _ZScoreInstance(nn.Module):
    """Standardize each node vector independently without stored statistics."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        return (x - mu) / std.clamp(min=self.eps)


def build_lm_node_norm(mode: str, dim: int) -> nn.Module:
    """Build the normalization for projected protein and ligand nodes."""
    m = str(mode or 'none').lower().strip()
    if m in ('none', '', 'null', 'identity', 'off'):
        return nn.Identity()
    if m == 'layernorm':
        return nn.LayerNorm(dim)
    if m in ('batchnorm', 'bn'):
        return nn.BatchNorm1d(dim)
    if m in ('l2', 'l2norm', 'unit'):
        return _L2NormDim()
    if m == 'rmsnorm':
        return _RMSNorm(dim)
    if m in ('zscore', 'z-score', 'zscore_instance', 'instance_zscore'):
        return _ZScoreInstance()
    raise ValueError(
        f'lm_node_norm 须为 {_LM_NODE_NORM_MODES}, got {mode!r}')


def _segment_sum(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Sum src along its first axis within each index group."""
    out = src.new_zeros((n, *src.shape[1:]))
    out.index_add_(0, index, src)
    return out


def _segment_softmax(score: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Apply softmax within each destination group. score: [E, H]; index: [E]."""
    idx2 = index.unsqueeze(-1).expand_as(score)
    # Subtract each group's maximum before exponentiation.
    mx = score.new_full((n, score.size(1)), float('-inf'))
    mx.scatter_reduce_(0, idx2, score, reduce='amax', include_self=True)
    score = score - mx[index]
    e = score.exp()
    denom = _segment_sum(e, index, n)
    return e / (denom[index] + 1e-16)


# One-letter amino acid codes in OnionNet's 21-category order.
_AA1_TO_TYPE = {
    'G': 0, 'A': 1, 'V': 2, 'L': 3, 'I': 4, 'P': 5, 'F': 6, 'Y': 7, 'W': 8, 'S': 9,
    'T': 10, 'C': 11, 'M': 12, 'N': 13, 'Q': 14, 'D': 15, 'E': 16, 'K': 17, 'R': 18, 'H': 19,
}
_OTH_TYPE = 20
N_RES_TYPES = 21
N_LIG_ELES = 8

# Process-local cache of CPU edge tensors, keyed by topology signature and PDB ID.
_TOPO_CACHE: dict = {}

ET_SS = 0
ET_PP = 1      # Protein backbone.
ET_LL = 2
ET_P2S = 3     # Gated protein-to-shell edge with contact-frequency bias.
ET_S2P = 4
ET_L2S = 5     # Gated ligand-to-shell edge.
ET_S2L = 6
ET_SELF = 7
N_EDGE_TYPES = 8


def seq_to_res_types(seq: str, n: int, device: torch.device) -> torch.Tensor:
    """Map the first n residues to OnionNet indices, using 20 for unknowns."""
    seq = seq or ''
    ids = [_AA1_TO_TYPE.get(seq[i].upper(), _OTH_TYPE) if i < len(seq) else _OTH_TYPE
           for i in range(n)]
    return torch.tensor(ids, dtype=torch.long, device=device)


class HeteroGraphTXBlock(nn.Module):
    """Combine sparse multi-head attention with gated neighbor means.

    Inputs use a flattened batch of disjoint graphs. Both branches aggregate
    incoming messages before residual normalization and the optional FFN.
    """

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.2,
                 use_ffn: bool = True, ffn_mult: int = 2,
                 edge_dim: int = 0):
        super().__init__()
        assert dim % n_heads == 0, f'dim={dim} 必须被 n_heads={n_heads} 整除'
        self.h = n_heads
        self.hd = dim // n_heads
        self.scale = self.hd ** -0.5
        self.attn_ln = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.gnn_ln = nn.LayerNorm(dim)
        self.gnn_lin = nn.Linear(dim, dim)
        self.edge_dim = int(edge_dim)
        if self.edge_dim > 0:
            self.edge_score = nn.Linear(self.edge_dim, n_heads, bias=False)
            self.edge_value = nn.Linear(self.edge_dim, dim, bias=False)
            self.edge_gnn = nn.Linear(self.edge_dim, dim, bias=False)
        self.use_ffn = bool(use_ffn)
        if self.use_ffn:
            self.ffn_ln = nn.LayerNorm(dim)
            self.ffn = nn.Sequential(
                nn.Linear(dim, ffn_mult * dim), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(ffn_mult * dim, dim),
            )

    def forward(self, H: torch.Tensor, edge_index: torch.Tensor,
                edge_gate: torch.Tensor, edge_bias: torch.Tensor,
                edge_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        # H: [N, dim]; edge_index: [2, E], in source/destination order.
        # edge_gate and edge_bias: [E]; the bias reflects contact frequency.
        N = H.size(0)
        src, dst = edge_index[0], edge_index[1]

        hn = self.attn_ln(H)
        q = self.q(hn).view(N, self.h, self.hd)
        k = self.k(hn).view(N, self.h, self.hd)
        v = self.v(hn).view(N, self.h, self.hd)
        score = (q[dst] * k[src]).sum(-1) * self.scale + edge_bias.unsqueeze(-1)
        if self.edge_dim > 0:
            assert edge_embed is not None and edge_embed.size(-1) == self.edge_dim
            score = score + self.edge_score(edge_embed)
        alpha = _segment_softmax(score, dst, N)                   # Normalize incoming edges at each destination.
        alpha = alpha * edge_gate.unsqueeze(-1)
        alpha = self.attn_drop(alpha)
        v_src = v[src]
        if self.edge_dim > 0:
            v_src = v_src + self.edge_value(edge_embed).view(-1, self.h, self.hd)
        msg = alpha.unsqueeze(-1) * v_src
        out = _segment_sum(msg, dst, N).reshape(N, -1)
        out_attn = self.o(out)

        # Mean over gated neighbor projections.
        g = self.gnn_lin(self.gnn_ln(H))
        g_src = g[src]
        if self.edge_dim > 0:
            g_src = g_src + self.edge_gnn(edge_embed)
        m = g_src * edge_gate.unsqueeze(-1)
        deg = _segment_sum(edge_gate.new_ones(edge_gate.size(0)), dst, N).clamp(min=1.0)
        out_gnn = _segment_sum(m, dst, N) / deg.unsqueeze(-1)

        H = H + out_attn + out_gnn
        if self.use_ffn:
            H = H + self.ffn(self.ffn_ln(H))
        return H


class ShellFMHeteroGraphAttention(nn.Module):
    """Sparse ShellFM graph operator with one message-passing scheme per layer."""

    SCHEMES = ('GAT', 'GATV2', 'GCN', 'GIN')

    def __init__(self, dim: int, n_heads: int, scheme: str,
                 dropout: float = 0.1, edge_dim: int = 0,
                 relation_calibration: bool = False,
                 n_relations: int = 8,
                 edge_score_path: bool = True,
                 edge_value_path: bool = True,
                 edge_value_film: bool = False,
                 attention_grouping: str = 'destination',
                 relation_fusion: str = 'sum',
                 relation_adapter_rank: int = 0,
                 relation_adapter_paths: str = 'kv',
                 relation_adapter_alpha: float = 1.0,
                 qk_norm: bool = False,
                 qk_temperature: str = 'fixed',
                 spatial_bias: str = 'none'):
        super().__init__()
        scheme = str(scheme).upper()
        assert scheme in self.SCHEMES, \
            f'scheme must be one of {self.SCHEMES}, got {scheme!r}'
        assert dim % n_heads == 0, \
            f'dim={dim} must be divisible by n_heads={n_heads}'
        self.scheme = scheme
        self.h = int(n_heads)
        self.hd = dim // self.h
        self.scale = self.hd ** -0.5
        self.edge_dim = int(edge_dim)
        self.relation_calibration = bool(relation_calibration)
        self.n_relations = int(n_relations)
        self.edge_score_path = bool(edge_score_path)
        self.edge_value_path = bool(edge_value_path)
        self.edge_value_film = bool(edge_value_film)
        self.attention_grouping = str(attention_grouping).lower()
        self.relation_fusion = str(relation_fusion).lower()
        self.relation_adapter_rank = int(relation_adapter_rank)
        self.relation_adapter_paths = str(relation_adapter_paths).lower()
        self.relation_adapter_alpha = float(relation_adapter_alpha)
        self.qk_norm = bool(qk_norm)
        self.qk_temperature = str(qk_temperature).lower()
        self.spatial_bias = str(spatial_bias).lower()
        if self.spatial_bias not in ('none', 'graphormer', 'alibi', 'ss'):
            raise ValueError(
                'spatial_bias must be none|graphormer|alibi|ss, got '
                f'{spatial_bias!r}')
        if self.qk_temperature not in ('fixed', 'learned'):
            raise ValueError('qk_temperature must be fixed|learned')
        if self.qk_norm and scheme not in ('GAT', 'GATV2'):
            raise ValueError('QK normalization is defined only for GAT/GATV2')
        if not self.qk_norm and self.qk_temperature != 'fixed':
            raise ValueError('learned QK temperature requires qk_norm=true')
        if self.relation_adapter_rank < 0:
            raise ValueError('relation_adapter_rank must be non-negative')
        if self.relation_adapter_paths not in ('k', 'v', 'kv'):
            raise ValueError('relation_adapter_paths must be k|v|kv')
        if self.relation_adapter_rank > 0 and scheme not in ('GAT', 'GATV2'):
            raise ValueError('relation adapters are defined only for GAT/GATV2')
        if self.attention_grouping not in (
                'destination', 'destination_relation'):
            raise ValueError(
                'attention_grouping must be destination|destination_relation')
        if self.relation_fusion not in ('sum', 'mean', 'learned'):
            raise ValueError('relation_fusion must be sum|mean|learned')
        if (self.attention_grouping != 'destination_relation'
                and self.relation_fusion != 'sum'):
            raise ValueError(
                'relation_fusion is used only with destination_relation grouping')
        if self.attention_grouping == 'destination_relation' and scheme not in ('GAT', 'GATV2'):
            raise ValueError(
                'destination_relation grouping is defined only for GAT/GATV2')
        if self.edge_value_film and not self.edge_value_path:
            raise ValueError('edge_value_film requires edge_value_path')
        if self.relation_calibration and self.scheme not in ('GAT', 'GATV2'):
            raise ValueError(
                'relation calibration is defined only for GAT/GATV2 attention')
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(dropout)
        if scheme in ('GAT', 'GATV2'):
            self.q_proj = nn.Linear(dim, dim)
            self.k_proj = nn.Linear(dim, dim)
            self.attn_dropout = nn.Dropout(dropout)
            if self.qk_norm and self.qk_temperature == 'learned':
                self.qk_log_scale = nn.Parameter(torch.full(
                    (self.h,), math.log(math.sqrt(self.hd))))
            if scheme == 'GATV2':
                self.gatv2_attn = nn.Parameter(torch.empty(self.h, self.hd))
                nn.init.xavier_uniform_(self.gatv2_attn)
        self.spatial_proj = None
        self.alibi_slope = None
        if self.spatial_bias in ('graphormer', 'ss'):
            self.spatial_proj = nn.Linear(1, self.h, bias=False)
            nn.init.zeros_(self.spatial_proj.weight)
        elif self.spatial_bias == 'alibi':
            self.alibi_slope = nn.Parameter(torch.zeros(self.h))
        if scheme == 'GIN':
            self.eps = nn.Parameter(torch.zeros(1))
        if self.edge_dim > 0 and self.edge_score_path:
            # Edge embeddings contribute to both scores and values.
            self.edge_score = nn.Linear(self.edge_dim, self.h, bias=False)
        if self.edge_dim > 0 and self.edge_value_path:
            self.edge_value = nn.Linear(self.edge_dim, dim, bias=False)
            if self.edge_value_film:
                self.edge_value_scale = nn.Linear(
                    self.edge_dim, dim, bias=False)
                nn.init.zeros_(self.edge_value_scale.weight)
        if self.relation_calibration:
            # Start with zero relation score bias and unit relation value scale.
            self.relation_score_bias = nn.Parameter(torch.zeros(
                self.n_relations, self.h))
            self.relation_value_log_scale = nn.Parameter(torch.zeros(
                self.n_relations, self.h))
        if (self.attention_grouping == 'destination_relation'
                and self.relation_fusion == 'learned'):
            # Normalize within relation groups, then fuse the groups present at each destination.
            self.relation_fusion_logits = nn.Parameter(torch.zeros(
                self.n_relations, self.h))
        if self.relation_adapter_rank > 0:
            rank = self.relation_adapter_rank
            self.relation_adapter_scale = self.relation_adapter_alpha / rank
            with torch.random.fork_rng(devices=[]):
                if 'k' in self.relation_adapter_paths:
                    self.rel_k_down = nn.Parameter(torch.empty(
                        self.n_relations, self.h, self.hd, rank))
                    self.rel_k_up = nn.Parameter(torch.zeros(
                        self.n_relations, self.h, rank, self.hd))
                    nn.init.xavier_uniform_(self.rel_k_down)
                if 'v' in self.relation_adapter_paths:
                    self.rel_v_down = nn.Parameter(torch.empty(
                        self.n_relations, self.h, self.hd, rank))
                    self.rel_v_up = nn.Parameter(torch.zeros(
                        self.n_relations, self.h, rank, self.hd))
                    nn.init.xavier_uniform_(self.rel_v_down)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), self.h, self.hd)

    def _relation_adapter(self, x: torch.Tensor, rel: torch.Tensor,
                          down: torch.Tensor,
                          up: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(x)
        for relation_id in range(self.n_relations):
            mask = rel.eq(relation_id)
            if not bool(mask.any()):
                continue
            latent = torch.einsum(
                'ehd,hdr->ehr', x[mask], down[relation_id])
            output[mask] = torch.einsum(
                'ehr,hrd->ehd', latent, up[relation_id])
        return self.relation_adapter_scale * output

    def _spatial_score(self, dist: torch.Tensor,
                       rel: Optional[torch.Tensor]) -> torch.Tensor:
        """Zero-init radial bias added to QK scores. dist is [E,1]."""
        if self.spatial_bias == 'alibi':
            return -self.alibi_slope.unsqueeze(0) * dist
        bias = self.spatial_proj(dist)
        if self.spatial_bias == 'ss':
            if rel is None:
                raise ValueError('ss spatial bias requires relation ids')
            bias = bias * rel.eq(0).to(dtype=bias.dtype).unsqueeze(-1)
        return bias

    def forward(self, H: torch.Tensor, edge_index: torch.Tensor,
                edge_embed: Optional[torch.Tensor] = None,
                rel: Optional[torch.Tensor] = None,
                spatial_dist: Optional[torch.Tensor] = None) -> torch.Tensor:
        # H [N,D], edge_index [2,E], message direction src -> dst.
        N = H.size(0)
        src, dst = edge_index[0], edge_index[1]
        V = self._split_heads(self.v_proj(H))
        if self.edge_dim > 0 and self.edge_value_path:
            assert edge_embed is not None and edge_embed.size(-1) == self.edge_dim
            edge_value = self.edge_value(edge_embed).view(-1, self.h, self.hd)
            edge_scale = (
                torch.tanh(self.edge_value_scale(edge_embed)).view(
                    -1, self.h, self.hd)
                if self.edge_value_film else None)
        else:
            edge_value = None
            edge_scale = None

        if self.scheme in ('GAT', 'GATV2'):
            Q = self._split_heads(self.q_proj(H))
            K = self._split_heads(self.k_proj(H))
            if self.qk_norm:
                Q = F.normalize(Q, p=2, dim=-1)
            k_src = K[src]
            if self.relation_adapter_rank > 0:
                if rel is None:
                    raise ValueError(
                        'relation adapters require relation ids')
                if 'k' in self.relation_adapter_paths:
                    k_src = k_src + self._relation_adapter(
                        k_src, rel, self.rel_k_down, self.rel_k_up)
            if self.qk_norm:
                k_src = F.normalize(k_src, p=2, dim=-1)
            if self.qk_norm:
                qk_scale = (
                    self.qk_log_scale.clamp(max=math.log(100.0)).exp()
                    if self.qk_temperature == 'learned'
                    else math.sqrt(self.hd))
            else:
                qk_scale = self.scale
            if self.scheme == 'GATV2':
                mid = F.leaky_relu(Q[dst] + k_src, negative_slope=0.2)
                score = (self.gatv2_attn * mid).sum(-1)
                if self.qk_norm:
                    score = score * qk_scale
            elif self.qk_norm:
                score = (Q[dst] * k_src).sum(-1) * qk_scale
            else:
                score = (Q[dst] * k_src).sum(-1) * self.scale
            if self.edge_dim > 0 and self.edge_score_path:
                assert edge_embed is not None
                score = score + self.edge_score(edge_embed)
            if self.spatial_bias != 'none':
                if spatial_dist is None:
                    raise ValueError('spatial_bias requires spatial_dist')
                score = score + self._spatial_score(
                    spatial_dist.to(dtype=score.dtype), rel)
            if self.relation_calibration:
                if rel is None:
                    raise ValueError(
                        'relation ids are required for relation calibration')
                score = score + self.relation_score_bias[rel]
            node_value = V[src]
            if (self.relation_adapter_rank > 0
                    and 'v' in self.relation_adapter_paths):
                node_value = node_value + self._relation_adapter(
                    node_value, rel, self.rel_v_down, self.rel_v_up)
            value = (
                node_value if edge_value is None
                else node_value + edge_value)
            if edge_scale is not None:
                value = (1.0 + edge_scale) * node_value + edge_value
            if self.relation_calibration:
                value_scale = torch.exp(
                    self.relation_value_log_scale[rel].clamp(-2.0, 2.0))
                value = value * value_scale.unsqueeze(-1)
            if self.attention_grouping == 'destination':
                alpha = self.attn_dropout(
                    _segment_softmax(score, dst, N))
                out = _segment_sum(alpha.unsqueeze(-1) * value, dst, N)
            else:
                if rel is None:
                    raise ValueError(
                        'relation ids are required for relation-wise attention')
                group = dst * self.n_relations + rel
                alpha = self.attn_dropout(_segment_softmax(
                    score, group, N * self.n_relations))
                grouped = _segment_sum(
                    alpha.unsqueeze(-1) * value,
                    group, N * self.n_relations,
                ).view(N, self.n_relations, self.h, self.hd)
                present = _segment_sum(
                    value.new_ones((value.size(0), 1)),
                    group, N * self.n_relations,
                ).view(N, self.n_relations).gt(0)
                if self.relation_fusion == 'sum':
                    out = grouped.sum(dim=1)
                elif self.relation_fusion == 'mean':
                    denom = present.sum(dim=1).clamp(min=1).to(
                        grouped.dtype).view(N, 1, 1)
                    out = grouped.sum(dim=1) / denom
                else:
                    logits = self.relation_fusion_logits.unsqueeze(0).expand(
                        N, -1, -1)
                    logits = logits.masked_fill(
                        ~present.unsqueeze(-1), float('-inf'))
                    weights = torch.softmax(logits, dim=1)
                    out = (weights.unsqueeze(-1) * grouped).sum(dim=1)
        elif self.scheme == 'GCN':
            # Symmetric Kipf normalization for bidirectional edges.
            one = H.new_ones(src.numel())
            deg_src = _segment_sum(one, src, N).clamp(min=1.0)
            deg_dst = _segment_sum(one, dst, N).clamp(min=1.0)
            norm = deg_src[src].rsqrt() * deg_dst[dst].rsqrt()
            value = V[src] if edge_value is None else V[src] + edge_value
            if edge_scale is not None:
                value = (1.0 + edge_scale) * V[src] + edge_value
            out = _segment_sum(norm.view(-1, 1, 1) * value, dst, N)
        else:
            is_self = src.eq(dst)
            value = V[src] if edge_value is None else V[src] + edge_value
            if edge_scale is not None:
                value = (1.0 + edge_scale) * V[src] + edge_value
            neigh = _segment_sum(value[~is_self], dst[~is_self], N)
            out = (1.0 + self.eps) * V + neigh

        out = out.reshape(N, -1)
        return self.proj_dropout(self.out_proj(out))


class TypedBottleneckAdapter(nn.Module):
    """Small zero-output FFN adapters for shell/protein/ligand node types."""

    def __init__(self, dim: int, bottleneck: int,
                 scope: str = 'all', n_types: int = 3):
        super().__init__()
        self.scope = str(scope).lower()
        if self.scope not in ('all', 'shell', 'lm', 'shared'):
            raise ValueError('type_adapter_scope must be all|shell|lm|shared')
        self.n_types = int(n_types)
        self.active_types = {
            'all': tuple(range(self.n_types)),
            'shell': (0,),
            'lm': (1, 2),
            # Shared routing preserves the parameter count of typed routing.
            'shared': tuple(range(self.n_types)),
        }[self.scope]
        # Keep the initialization of later layers unchanged when adding a zero-output adapter.
        with torch.random.fork_rng(devices=[]):
            self.adapters = nn.ModuleDict()
            for type_id in self.active_types:
                module = nn.Sequential(
                    nn.Linear(dim, int(bottleneck)), nn.GELU(),
                    nn.Linear(int(bottleneck), dim),
                )
                nn.init.zeros_(module[-1].weight)
                nn.init.zeros_(module[-1].bias)
                self.adapters[str(type_id)] = module

    def forward(self, x: torch.Tensor, ntype: torch.Tensor) -> torch.Tensor:
        if self.scope == 'shared':
            return sum(
                self.adapters[str(type_id)](x)
                for type_id in self.active_types) / len(self.active_types)
        output = torch.zeros_like(x)
        for type_id in self.active_types:
            mask = ntype.eq(type_id).unsqueeze(-1).to(dtype=x.dtype)
            output = output + self.adapters[str(type_id)](x) * mask
        return output


class ShellFMHeteroGraphTransformerBlock(nn.Module):
    """Graph attention and feed-forward updates with pre- or post-normalization."""

    def __init__(self, dim: int, n_heads: int, scheme: str,
                 dropout: float = 0.1, use_ffn: bool = True,
                 ffn_mult: int = 2, edge_dim: int = 0,
                 norm_style: str = 'pre',
                 relation_calibration: bool = False,
                 n_relations: int = 8,
                 edge_score_path: bool = True,
                 edge_value_path: bool = True,
                 edge_value_film: bool = False,
                 layerscale_init: Optional[float] = None,
                 attention_grouping: str = 'destination',
                 relation_fusion: str = 'sum',
                 relation_adapter_rank: int = 0,
                 relation_adapter_paths: str = 'kv',
                 relation_adapter_alpha: float = 1.0,
                 qk_norm: bool = False,
                 qk_temperature: str = 'fixed',
                 spatial_bias: str = 'none',
                 type_adapter_dim: int = 0,
                 type_adapter_scope: str = 'all'):
        super().__init__()
        self.norm_style = str(norm_style).lower()
        if self.norm_style not in ('pre', 'post'):
            raise ValueError(f'norm_style must be pre|post, got {norm_style!r}')
        self.attn_ln = nn.LayerNorm(dim)
        self.attn = ShellFMHeteroGraphAttention(
            dim, n_heads, scheme, dropout, edge_dim=edge_dim,
            relation_calibration=relation_calibration,
            n_relations=n_relations,
            edge_score_path=edge_score_path,
            edge_value_path=edge_value_path,
            edge_value_film=edge_value_film,
            attention_grouping=attention_grouping,
            relation_fusion=relation_fusion,
            relation_adapter_rank=relation_adapter_rank,
            relation_adapter_paths=relation_adapter_paths,
            relation_adapter_alpha=relation_adapter_alpha,
            qk_norm=qk_norm,
            qk_temperature=qk_temperature,
            spatial_bias=spatial_bias)
        self.use_layerscale = layerscale_init is not None
        if self.use_layerscale:
            self.attn_scale = nn.Parameter(torch.full(
                (dim,), float(layerscale_init)))
        self.use_ffn = bool(use_ffn)
        self.type_adapter_dim = int(type_adapter_dim)
        if self.type_adapter_dim > 0 and not self.use_ffn:
            raise ValueError('type adapters require use_ffn=true')
        if self.use_ffn:
            self.ffn_ln = nn.LayerNorm(dim)
            self.ffn = nn.Sequential(
                nn.Linear(dim, ffn_mult * dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_mult * dim, dim),
                nn.Dropout(dropout),
            )
            if self.use_layerscale:
                self.ffn_scale = nn.Parameter(torch.full(
                    (dim,), float(layerscale_init)))
            self.type_adapter = (
                TypedBottleneckAdapter(
                    dim, self.type_adapter_dim, scope=type_adapter_scope)
                if self.type_adapter_dim > 0 else None)

    def forward(self, H: torch.Tensor, edge_index: torch.Tensor,
                edge_embed: Optional[torch.Tensor] = None,
                rel: Optional[torch.Tensor] = None,
                ntype: Optional[torch.Tensor] = None,
                spatial_dist: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.norm_style == 'pre':
            attn_out = self.attn(
                self.attn_ln(H), edge_index, edge_embed, rel=rel,
                spatial_dist=spatial_dist)
            if self.use_layerscale:
                attn_out = self.attn_scale * attn_out
            H = H + attn_out
            if self.use_ffn:
                ffn_input = self.ffn_ln(H)
                ffn_out = self.ffn(ffn_input)
                if self.type_adapter is not None:
                    if ntype is None:
                        raise ValueError('type adapters require node type ids')
                    ffn_out = ffn_out + self.type_adapter(ffn_input, ntype)
                if self.use_layerscale:
                    ffn_out = self.ffn_scale * ffn_out
                H = H + ffn_out
        else:
            attn_out = self.attn(
                H, edge_index, edge_embed, rel=rel, spatial_dist=spatial_dist)
            if self.use_layerscale:
                attn_out = self.attn_scale * attn_out
            H = self.attn_ln(H + attn_out)
            if self.use_ffn:
                ffn_out = self.ffn(H)
                if self.type_adapter is not None:
                    if ntype is None:
                        raise ValueError('type adapters require node type ids')
                    ffn_out = ffn_out + self.type_adapter(H, ntype)
                if self.use_layerscale:
                    ffn_out = self.ffn_scale * ffn_out
                H = self.ffn_ln(H + ffn_out)
        return H


class HeterogeneousGraphTransformerAttention(nn.Module):
    """Sparse attention with node-type projections and relation-specific key/value transforms."""

    def __init__(self, dim: int, n_heads: int, n_node_types: int,
                 n_relations: int, dropout: float = 0.1,
                 edge_dim: int = 0):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(
                f'dim={dim} must be divisible by n_heads={n_heads}')
        self.dim = int(dim)
        self.h = int(n_heads)
        self.hd = self.dim // self.h
        self.scale = self.hd ** -0.5
        self.n_node_types = int(n_node_types)
        self.n_relations = int(n_relations)
        self.edge_dim = int(edge_dim)

        def typed_linears():
            return nn.ModuleList([
                nn.Linear(self.dim, self.dim)
                for _ in range(self.n_node_types)
            ])

        self.q_proj = typed_linears()
        self.k_proj = typed_linears()
        self.v_proj = typed_linears()
        self.out_proj = typed_linears()
        self.rel_k = nn.Parameter(torch.empty(
            self.n_relations, self.h, self.hd, self.hd))
        self.rel_v = nn.Parameter(torch.empty(
            self.n_relations, self.h, self.hd, self.hd))
        self.rel_prior = nn.Parameter(torch.ones(
            self.n_relations, self.h))
        nn.init.xavier_uniform_(self.rel_k)
        nn.init.xavier_uniform_(self.rel_v)
        self.attn_dropout = nn.Dropout(float(dropout))
        self.proj_dropout = nn.Dropout(float(dropout))
        if self.edge_dim > 0:
            self.edge_score = nn.Linear(
                self.edge_dim, self.h, bias=False)
            self.edge_value = nn.Linear(
                self.edge_dim, self.dim, bias=False)

    @staticmethod
    def _typed_project(layers: nn.ModuleList, H: torch.Tensor,
                       ntype: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(H)
        for type_id, layer in enumerate(layers):
            mask = ntype.eq(type_id).unsqueeze(-1).to(dtype=H.dtype)
            out = out + layer(H) * mask
        return out

    def forward(self, H: torch.Tensor, edge_index: torch.Tensor,
                rel: torch.Tensor, ntype: torch.Tensor,
                edge_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        N = H.size(0)
        src, dst = edge_index[0], edge_index[1]
        Q = self._typed_project(
            self.q_proj, H, ntype).view(N, self.h, self.hd)
        K = self._typed_project(
            self.k_proj, H, ntype).view(N, self.h, self.hd)
        V = self._typed_project(
            self.v_proj, H, ntype).view(N, self.h, self.hd)

        # Group by relation to avoid materializing a matrix of shape [H, hd, hd] for each edge.
        k_src, v_src = K[src], V[src]
        rel_k = torch.zeros_like(k_src)
        rel_v = torch.zeros_like(v_src)
        for relation_id in range(self.n_relations):
            mask = rel.eq(relation_id)
            if not bool(mask.any()):
                continue
            rel_k[mask] = torch.einsum(
                'ehd,hdf->ehf', k_src[mask], self.rel_k[relation_id])
            rel_v[mask] = torch.einsum(
                'ehd,hdf->ehf', v_src[mask], self.rel_v[relation_id])
        score = (
            Q[dst] * rel_k).sum(dim=-1) * self.scale
        score = score * self.rel_prior[rel]
        if self.edge_dim > 0:
            if edge_embed is None or edge_embed.size(-1) != self.edge_dim:
                raise ValueError('HGT attention requires matching edge embeddings')
            score = score + self.edge_score(edge_embed)
            rel_v = rel_v + self.edge_value(edge_embed).view(
                -1, self.h, self.hd)
        alpha = self.attn_dropout(_segment_softmax(score, dst, N))
        out = _segment_sum(
            alpha.unsqueeze(-1) * rel_v, dst, N).reshape(N, self.dim)
        out = self._typed_project(self.out_proj, out, ntype)
        return self.proj_dropout(out)


class HeterogeneousGraphTransformerBlock(nn.Module):
    """HGT update with pre- or post-normalization."""

    def __init__(self, dim: int, n_heads: int, n_node_types: int,
                 n_relations: int, dropout: float = 0.1,
                 use_ffn: bool = True, ffn_mult: int = 2,
                 edge_dim: int = 0, norm_style: str = 'post'):
        super().__init__()
        self.norm_style = str(norm_style).lower()
        if self.norm_style not in ('pre', 'post'):
            raise ValueError(f'norm_style must be pre|post, got {norm_style!r}')
        self.attn_ln = nn.LayerNorm(dim)
        self.attn = HeterogeneousGraphTransformerAttention(
            dim=dim, n_heads=n_heads, n_node_types=n_node_types,
            n_relations=n_relations, dropout=dropout, edge_dim=edge_dim)
        self.use_ffn = bool(use_ffn)
        if self.use_ffn:
            self.ffn_ln = nn.LayerNorm(dim)
            self.ffn = nn.Sequential(
                nn.Linear(dim, ffn_mult * dim), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_mult * dim, dim), nn.Dropout(dropout),
            )

    def forward(self, H: torch.Tensor, edge_index: torch.Tensor,
                rel: torch.Tensor, ntype: torch.Tensor,
                edge_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.norm_style == 'pre':
            H = H + self.attn(
                self.attn_ln(H), edge_index, rel, ntype, edge_embed)
            if self.use_ffn:
                H = H + self.ffn(self.ffn_ln(H))
        else:
            H = self.attn_ln(
                H + self.attn(H, edge_index, rel, ntype, edge_embed))
            if self.use_ffn:
                H = self.ffn_ln(H + self.ffn(H))
        return H


class ShellFMHeteroGPSBlock(nn.Module):
    """Combine local relation attention with global attention within each complex."""

    def __init__(self, dim: int, n_heads: int, scheme: str,
                 dropout: float = 0.1, use_ffn: bool = True,
                 ffn_mult: int = 2, edge_dim: int = 0,
                 norm_style: str = 'pre'):
        super().__init__()
        self.norm_style = str(norm_style).lower()
        if self.norm_style not in ('pre', 'post'):
            raise ValueError(f'norm_style must be pre|post, got {norm_style!r}')
        self.local_ln = nn.LayerNorm(dim)
        self.global_ln = nn.LayerNorm(dim)
        self.local_attn = ShellFMHeteroGraphAttention(
            dim, n_heads, scheme, dropout, edge_dim=edge_dim)
        self.global_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True)
        self.branch_dropout = nn.Dropout(dropout)
        self.merge_ln = nn.LayerNorm(dim)
        self.use_ffn = bool(use_ffn)
        if self.use_ffn:
            self.ffn_ln = nn.LayerNorm(dim)
            self.ffn = nn.Sequential(
                nn.Linear(dim, ffn_mult * dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_mult * dim, dim),
                nn.Dropout(dropout),
            )

    def _global(self, H: torch.Tensor, B: int, n_nodes: int,
                apply_norm: bool) -> torch.Tensor:
        nodes = H.reshape(B, n_nodes, -1)
        if apply_norm:
            nodes = self.global_ln(nodes)
        out, _ = self.global_attn(nodes, nodes, nodes, need_weights=False)
        return out.reshape_as(H)

    def forward(self, H: torch.Tensor, edge_index: torch.Tensor,
                edge_embed: Optional[torch.Tensor], B: int,
                n_nodes: int) -> torch.Tensor:
        if H.size(0) != B * n_nodes:
            raise ValueError(
                f'GPS node layout mismatch: H={tuple(H.shape)}, B={B}, N={n_nodes}')
        if self.norm_style == 'pre':
            local = self.local_attn(
                self.local_ln(H), edge_index, edge_embed)
            global_out = self._global(H, B, n_nodes, apply_norm=True)
            H = H + self.branch_dropout(local + global_out)
            if self.use_ffn:
                H = H + self.ffn(self.ffn_ln(H))
        else:
            local = self.local_attn(H, edge_index, edge_embed)
            global_out = self._global(H, B, n_nodes, apply_norm=False)
            H = self.merge_ln(H + self.branch_dropout(local + global_out))
            if self.use_ffn:
                H = self.ffn_ln(H + self.ffn(H))
        return H


class PostGraphRadialResidual(nn.Module):
    """Add a zero-initialized radial convolution without changing shell order."""

    MODES = ('k3', 'k5', 'dilated3', 'multiscale')

    def __init__(self, dim: int, mode: str, n_shells: int,
                 gate_init: float = -3.0, order: str = 'radial'):
        super().__init__()
        self.mode = str(mode).lower()
        if self.mode not in self.MODES:
            raise ValueError(
                f'post-graph radial mode must be one of {self.MODES}')
        self.order = str(order).lower()
        if self.order not in ('radial', 'permuted'):
            raise ValueError('post_graph_radial_order must be radial|permuted')
        if self.order == 'permuted':
            generator = torch.Generator(device='cpu')
            generator.manual_seed(20260723)
            permutation = torch.randperm(int(n_shells), generator=generator)
            inverse = torch.argsort(permutation)
        else:
            permutation = torch.arange(int(n_shells))
            inverse = permutation
        self.register_buffer('permutation', permutation, persistent=False)
        self.register_buffer('inverse_permutation', inverse, persistent=False)
        specifications = {
            'k3': [(3, 1)],
            'k5': [(5, 1)],
            'dilated3': [(3, 2)],
            'multiscale': [(3, 1), (5, 1), (3, 2)],
        }[self.mode]
        self.branches = nn.ModuleList([
            nn.Conv1d(
                dim, dim, kernel_size=kernel, dilation=dilation,
                padding=dilation * (kernel - 1) // 2,
                groups=dim,
            )
            for kernel, dilation in specifications
        ])
        self.mix = nn.Conv1d(dim * len(specifications), dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        # The zero-initialized mixer learns before the depthwise radial filters.
        nn.init.zeros_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)

    def forward(self, shell: torch.Tensor) -> torch.Tensor:
        # Convolve along the shell axis of [B, S, D].
        ordered = shell[:, self.permutation]
        x = ordered.transpose(1, 2)
        branch = [F.gelu(conv(x)) for conv in self.branches]
        residual = self.mix(torch.cat(branch, dim=1)).transpose(1, 2)
        residual = self.norm(residual)
        residual = residual[:, self.inverse_permutation]
        return shell + torch.sigmoid(self.gate) * residual


class ExtraColumnModulate(nn.Module):
    """Modulate the extra readout features while leaving node states unchanged.

    se_h: extra *= 1 + tanh(Linear_0(H)).
    film_h: extra = extra * (1 + gamma) + beta, with [gamma, beta] = Linear_0(H).
    phi: extra = extra + MLP_0(extra).
    """

    def __init__(self, shell_dim: int, extra_dim: int, mode: str):
        super().__init__()
        self.mode = str(mode).lower()
        if self.mode not in ('se_h', 'film_h', 'phi'):
            raise ValueError(
                'extra_modulate mode must be se_h|film_h|phi, got '
                f'{mode!r}')
        self.proj = None
        self.mlp = None
        if self.mode == 'se_h':
            self.proj = nn.Linear(int(shell_dim), int(extra_dim))
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        elif self.mode == 'film_h':
            self.proj = nn.Linear(int(shell_dim), int(extra_dim) * 2)
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        else:
            hidden = max(int(extra_dim), 64)
            self.mlp = nn.Sequential(
                nn.Linear(int(extra_dim), hidden),
                nn.GELU(),
                nn.Linear(hidden, int(extra_dim)),
            )
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, shell_h: torch.Tensor,
                extra: torch.Tensor) -> torch.Tensor:
        if self.mode == 'se_h':
            return extra * (1.0 + torch.tanh(self.proj(shell_h)))
        if self.mode == 'film_h':
            gamma, beta = self.proj(shell_h).chunk(2, dim=-1)
            return extra * (1.0 + gamma) + beta
        return extra + self.mlp(extra)


class PostGraphGLU(nn.Module):
    """Gated FFN before flattening. The zero-initialized output starts as a no-op."""

    def __init__(self, dim: int, variant: str, mult: int = 2):
        super().__init__()
        self.variant = str(variant).lower()
        if self.variant not in ('swiglu', 'geglu'):
            raise ValueError(
                'post_graph_glu must be swiglu|geglu, got '
                f'{variant!r}')
        hidden = int(mult) * int(dim)
        self.w_in = nn.Linear(int(dim), hidden)
        self.w_gate = nn.Linear(int(dim), hidden)
        self.w_out = nn.Linear(hidden, int(dim))
        nn.init.zeros_(self.w_out.weight)
        nn.init.zeros_(self.w_out.bias)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        value = self.w_in(H)
        gate = self.w_gate(H)
        if self.variant == 'swiglu':
            hidden = F.silu(value) * gate
        else:
            hidden = F.gelu(value) * gate
        return H + self.w_out(hidden)


class ProgressiveAllNodeMLP(nn.Module):
    """Flatten all node states, then reduce width before regression."""

    def __init__(self, in_dim: int, hidden_dims: List[int],
                 dropout: float = 0.2):
        super().__init__()
        dims = [int(in_dim)] + [int(x) for x in hidden_dims]
        if not hidden_dims or any(x <= 0 for x in dims):
            raise ValueError(f'invalid progressive MLP dims: {dims}')
        layers: List[nn.Module] = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            if d_out >= d_in:
                raise ValueError(
                    f'progressive MLP must decrease monotonically, got {dims}')
            layers.extend([
                nn.Linear(d_in, d_out),
                nn.GELU(),
                nn.LayerNorm(d_out),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor,
                residual_384: Optional[torch.Tensor] = None,
                hidden0_add: Optional[torch.Tensor] = None) -> torch.Tensor:
        mods = list(self.net)
        h = mods[0](x)
        if hidden0_add is not None:
            h = h + hidden0_add
        for mod in mods[1:-1]:
            h = mod(h)
        if residual_384 is not None:
            h = h + residual_384
        return mods[-1](h)


class OrderedLowRankReadout(nn.Module):
    """Preserve node order and factor the first dense layer into two linear maps.

    Optional protein/ligand and radial features add gated latent residuals.
    """

    def __init__(self, node_dim: int, n_nodes: int, n_shells: int,
                 rank: int = 512, hidden_dim: int = 1536,
                 latent_dim: int = 384, dropout: float = 0.3,
                 use_pl_interaction: bool = False,
                 use_radial_branch: bool = False,
                 radial_bins: int = 10, branch_gate_init: float = -2.0):
        super().__init__()
        self.node_dim = int(node_dim)
        self.n_nodes = int(n_nodes)
        self.n_shells = int(n_shells)
        self.rank = int(rank)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.use_pl_interaction = bool(use_pl_interaction)
        self.use_radial_branch = bool(use_radial_branch)
        self.radial_bins = int(radial_bins)
        if min(self.rank, self.hidden_dim, self.latent_dim) <= 0:
            raise ValueError('ordered low-rank dimensions must be positive')
        if self.n_nodes != self.n_shells + 2:
            raise ValueError(
                'ordered low-rank readout requires shell+protein+ligand layout')
        if self.use_radial_branch and (
                self.radial_bins <= 0
                or self.n_shells % self.radial_bins != 0):
            raise ValueError(
                f'n_shells={self.n_shells} must divide radial_bins={self.radial_bins}')

        flat_dim = self.n_nodes * self.node_dim
        self.flat_factor_in = nn.Linear(flat_dim, self.rank, bias=False)
        self.flat_factor_out = nn.Linear(
            self.rank, self.hidden_dim, bias=True)
        self.flat_to_latent = nn.Sequential(
            nn.GELU(), nn.LayerNorm(self.hidden_dim),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.latent_dim),
            nn.GELU(), nn.LayerNorm(self.latent_dim),
            nn.Dropout(float(dropout)),
        )

        if self.use_pl_interaction:
            self.pl_branch = nn.Sequential(
                nn.Linear(4 * self.node_dim, self.latent_dim),
                nn.GELU(), nn.LayerNorm(self.latent_dim),
                nn.Dropout(float(dropout)),
            )
            self.pl_gate = nn.Parameter(torch.tensor(float(branch_gate_init)))
        if self.use_radial_branch:
            radial_dim = 2 * self.radial_bins * self.node_dim
            self.radial_branch = nn.Sequential(
                nn.Linear(radial_dim, self.latent_dim),
                nn.GELU(), nn.LayerNorm(self.latent_dim),
                nn.Dropout(float(dropout)),
            )
            self.radial_gate = nn.Parameter(
                torch.tensor(float(branch_gate_init)))
        self.output = nn.Linear(self.latent_dim, 1)

    def forward(self, nodes: torch.Tensor, prot_idx: int,
                lig_idx: int) -> torch.Tensor:
        if nodes.dim() != 3 or nodes.shape[1:] != (
                self.n_nodes, self.node_dim):
            raise ValueError(
                f'expected [B,{self.n_nodes},{self.node_dim}], '
                f'got {tuple(nodes.shape)}')
        B = nodes.size(0)
        latent = self.flat_to_latent(
            self.flat_factor_out(
                self.flat_factor_in(nodes.reshape(B, -1))))
        if self.use_pl_interaction:
            prot_h, lig_h = nodes[:, prot_idx], nodes[:, lig_idx]
            pl_feat = torch.cat([
                prot_h, lig_h, prot_h * lig_h,
                (prot_h - lig_h).abs(),
            ], dim=-1)
            latent = latent + torch.sigmoid(self.pl_gate) * self.pl_branch(pl_feat)
        if self.use_radial_branch:
            bin_width = self.n_shells // self.radial_bins
            shell_bins = nodes[:, :self.n_shells].reshape(
                B, self.radial_bins, bin_width, self.node_dim)
            radial = torch.cat([
                shell_bins.mean(dim=2), shell_bins.max(dim=2).values,
            ], dim=-1).reshape(B, -1)
            latent = latent + torch.sigmoid(
                self.radial_gate) * self.radial_branch(radial)
        return self.output(latent)


class FullOrderedResidualReadout(nn.Module):
    """Read ordered node states with optional gated residual features."""

    def __init__(self, node_dim: int, n_nodes: int, n_shells: int,
                 hidden_dim: int = 1536, latent_dim: int = 384,
                 dropout: float = 0.3, use_pl_interaction: bool = False,
                 use_radial_branch: bool = False, radial_bins: int = 10,
                 branch_gate_init: float = -3.0):
        super().__init__()
        self.node_dim = int(node_dim)
        self.n_nodes = int(n_nodes)
        self.n_shells = int(n_shells)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.use_pl_interaction = bool(use_pl_interaction)
        self.use_radial_branch = bool(use_radial_branch)
        self.radial_bins = int(radial_bins)
        if self.n_nodes != self.n_shells + 2:
            raise ValueError(
                'full ordered residual readout requires 60 shell + P + L')
        if self.use_radial_branch and (
                self.radial_bins <= 0
                or self.n_shells % self.radial_bins != 0):
            raise ValueError(
                f'n_shells={self.n_shells} must divide radial_bins='
                f'{self.radial_bins}')

        flat_dim = self.n_nodes * self.node_dim
        self.main_to_latent = nn.Sequential(
            nn.Linear(flat_dim, self.hidden_dim),
            nn.GELU(), nn.LayerNorm(self.hidden_dim),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.latent_dim),
            nn.GELU(), nn.LayerNorm(self.latent_dim),
            nn.Dropout(float(dropout)),
        )
        if self.use_pl_interaction:
            self.pl_branch = nn.Sequential(
                nn.Linear(4 * self.node_dim, self.latent_dim),
                nn.GELU(), nn.LayerNorm(self.latent_dim),
                nn.Dropout(float(dropout)),
            )
            self.pl_gate = nn.Parameter(torch.tensor(float(branch_gate_init)))
        if self.use_radial_branch:
            radial_dim = 2 * self.radial_bins * self.node_dim
            self.radial_branch = nn.Sequential(
                nn.Linear(radial_dim, self.latent_dim),
                nn.GELU(), nn.LayerNorm(self.latent_dim),
                nn.Dropout(float(dropout)),
            )
            self.radial_gate = nn.Parameter(
                torch.tensor(float(branch_gate_init)))
        self.output = nn.Linear(self.latent_dim, 1)

    def forward(self, nodes: torch.Tensor, prot_idx: int,
                lig_idx: int) -> torch.Tensor:
        if nodes.dim() != 3 or nodes.shape[1:] != (
                self.n_nodes, self.node_dim):
            raise ValueError(
                f'expected [B,{self.n_nodes},{self.node_dim}], '
                f'got {tuple(nodes.shape)}')
        B = nodes.size(0)
        latent = self.main_to_latent(nodes.reshape(B, -1))
        if self.use_pl_interaction:
            prot_h, lig_h = nodes[:, prot_idx], nodes[:, lig_idx]
            pl_feat = torch.cat([
                prot_h, lig_h, prot_h * lig_h,
                (prot_h - lig_h).abs(),
            ], dim=-1)
            latent = latent + torch.sigmoid(
                self.pl_gate) * self.pl_branch(pl_feat)
        if self.use_radial_branch:
            bin_width = self.n_shells // self.radial_bins
            shell_bins = nodes[:, :self.n_shells].reshape(
                B, self.radial_bins, bin_width, self.node_dim)
            radial = torch.cat([
                shell_bins.mean(dim=2), shell_bins.max(dim=2).values,
            ], dim=-1).reshape(B, -1)
            latent = latent + torch.sigmoid(
                self.radial_gate) * self.radial_branch(radial)
        return self.output(latent)


class DualQueryRadialReadout(nn.Module):
    """Summarize radial shell states with protein, ligand, and affinity queries.

    Readout input: [p, l, p*l, |p-l|, ctx_p, ctx_l, ctx_aff], shape [B, 7D].
    """

    def __init__(self, node_dim: int, n_shells: int = 60,
                 n_heads: int = 4, hidden_dim: int = 512,
                 mid_dim: int = 128, dropout: float = 0.2,
                 shell_upper_edges_angstrom=None):
        super().__init__()
        if node_dim % n_heads != 0:
            raise ValueError(
                f'node_dim={node_dim} must be divisible by n_heads={n_heads}')
        if hidden_dim <= 0 or mid_dim <= 0:
            raise ValueError('dual-query readout hidden dimensions must be positive')
        self.node_dim = int(node_dim)
        self.n_shells = int(n_shells)
        radii = _validated_shell_upper_edges(
            self.n_shells, shell_upper_edges_angstrom)
        self.register_buffer(
            'normalized_radius', (radii / radii[-1]).view(1, self.n_shells, 1),
            persistent=True)
        self.radius_encoder = nn.Sequential(
            nn.Linear(1, self.node_dim), nn.SiLU(),
            nn.Linear(self.node_dim, self.node_dim),
        )
        self.shell_ln = nn.LayerNorm(self.node_dim)
        self.query_ln = nn.LayerNorm(self.node_dim)
        self.cross_attention = nn.MultiheadAttention(
            self.node_dim, int(n_heads), dropout=float(dropout),
            batch_first=True)
        self.context_ln = nn.LayerNorm(self.node_dim)
        self.affinity_query = nn.Parameter(torch.empty(1, 1, self.node_dim))
        nn.init.normal_(self.affinity_query, mean=0.0, std=0.02)

        in_dim = 7 * self.node_dim
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(mid_dim)),
            nn.GELU(),
            nn.LayerNorm(int(mid_dim)),
            nn.Dropout(float(dropout)),
            nn.Linear(int(mid_dim), 1),
        )

    def forward(self, shell_h: torch.Tensor, prot_h: torch.Tensor,
                lig_h: torch.Tensor) -> torch.Tensor:
        if shell_h.dim() != 3 or shell_h.size(1) != self.n_shells:
            raise ValueError(
                f'expected shell_h [B,{self.n_shells},D], got {tuple(shell_h.shape)}')
        B = shell_h.size(0)
        radius = self.radius_encoder(
            self.normalized_radius.to(dtype=shell_h.dtype))
        shell_kv = self.shell_ln(shell_h + radius)
        task_query = self.affinity_query.expand(B, -1, -1)
        queries = torch.cat([
            prot_h.unsqueeze(1), lig_h.unsqueeze(1), task_query,
        ], dim=1)
        query_norm = self.query_ln(queries)
        context, _ = self.cross_attention(
            query_norm, shell_kv, shell_kv, need_weights=False)
        context = self.context_ln(queries + context)
        prot_context, lig_context, affinity_context = context.unbind(dim=1)
        fused = torch.cat([
            prot_h,
            lig_h,
            prot_h * lig_h,
            (prot_h - lig_h).abs(),
            prot_context,
            lig_context,
            affinity_context,
        ], dim=-1)
        return self.head(fused)


class AllNodePoolingReadout(nn.Module):
    """Pool all node states. Meanmax and Set2Set return [B, 2D]; other pools return [B, D]."""

    MODES = ('mean', 'sum', 'max', 'meanmax', 'attention', 'set2set')

    def __init__(self, node_dim: int, mode: str,
                 hidden_dim: int = 384, mid_dim: int = 128,
                 dropout: float = 0.3, set2set_steps: int = 3):
        super().__init__()
        self.node_dim = int(node_dim)
        self.mode = str(mode).lower()
        if self.mode not in self.MODES:
            raise ValueError(
                f'pool mode must be one of {self.MODES}, got {mode!r}')
        if hidden_dim <= 0 or mid_dim <= 0:
            raise ValueError('pool readout hidden dimensions must be positive')
        self.set2set_steps = int(set2set_steps)
        if self.set2set_steps <= 0:
            raise ValueError('set2set_steps must be positive')
        if self.mode == 'attention':
            gate_hidden = max(32, self.node_dim // 2)
            self.gate = nn.Sequential(
                nn.Linear(self.node_dim, gate_hidden),
                nn.Tanh(),
                nn.Linear(gate_hidden, 1, bias=False),
            )
        elif self.mode == 'set2set':
            self.set2set_lstm = nn.LSTM(
                input_size=2 * self.node_dim,
                hidden_size=self.node_dim,
                num_layers=1,
                batch_first=True,
            )

        pooled_dim = (
            2 * self.node_dim
            if self.mode in ('meanmax', 'set2set') else self.node_dim)
        self.head = nn.Sequential(
            nn.Linear(pooled_dim, int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(mid_dim)),
            nn.GELU(),
            nn.LayerNorm(int(mid_dim)),
            nn.Dropout(float(dropout)),
            nn.Linear(int(mid_dim), 1),
        )

    def _set2set(self, nodes: torch.Tensor) -> torch.Tensor:
        B, _, D = nodes.shape
        q_star = nodes.new_zeros(B, 2 * D)
        hidden = (
            nodes.new_zeros(1, B, D),
            nodes.new_zeros(1, B, D),
        )
        for _ in range(self.set2set_steps):
            q, hidden = self.set2set_lstm(q_star.unsqueeze(1), hidden)
            q = q.squeeze(1)
            score = (nodes * q.unsqueeze(1)).sum(dim=-1)
            alpha = torch.softmax(score, dim=1)
            read = (alpha.unsqueeze(-1) * nodes).sum(dim=1)
            q_star = torch.cat([q, read], dim=-1)
        return q_star

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        if nodes.dim() != 3 or nodes.size(-1) != self.node_dim:
            raise ValueError(
                f'expected nodes [B,N,{self.node_dim}], got {tuple(nodes.shape)}')
        if self.mode == 'mean':
            pooled = nodes.mean(dim=1)
        elif self.mode == 'sum':
            pooled = nodes.sum(dim=1)
        elif self.mode == 'max':
            pooled = nodes.max(dim=1).values
        elif self.mode == 'meanmax':
            pooled = torch.cat([
                nodes.mean(dim=1), nodes.max(dim=1).values,
            ], dim=-1)
        elif self.mode == 'attention':
            weight = torch.softmax(self.gate(nodes).squeeze(-1), dim=1)
            pooled = (weight.unsqueeze(-1) * nodes).sum(dim=1)
        else:
            pooled = self._set2set(nodes)
        return self.head(pooled)


class HeteroTypePoolingReadout(nn.Module):
    """Preserve shell/protein/ligand roles during compact graph readout."""

    def __init__(self, node_dim: int, hidden_dim: int = 384,
                 mid_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.node_dim = int(node_dim)
        in_dim = 4 * self.node_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dim)),
            nn.GELU(), nn.LayerNorm(int(hidden_dim)),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(mid_dim)),
            nn.GELU(), nn.LayerNorm(int(mid_dim)),
            nn.Dropout(float(dropout)),
            nn.Linear(int(mid_dim), 1),
        )

    def forward(self, shell_h: torch.Tensor, prot_h: torch.Tensor,
                lig_h: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([
            shell_h.mean(dim=1), shell_h.max(dim=1).values,
            prot_h, lig_h,
        ], dim=-1)
        return self.head(feat)


class RadialBinPoolingReadout(nn.Module):
    """Pool contiguous shell bins while retaining their physical radius order."""

    def __init__(self, node_dim: int, n_shells: int = 60, n_bins: int = 10,
                 hidden_dim: int = 512, mid_dim: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        self.node_dim = int(node_dim)
        self.n_shells = int(n_shells)
        self.n_bins = int(n_bins)
        if self.n_bins <= 0 or self.n_shells % self.n_bins != 0:
            raise ValueError(
                f'n_shells={self.n_shells} must be divisible by n_bins={self.n_bins}')
        self.bin_width = self.n_shells // self.n_bins
        in_dim = (2 * self.n_bins + 2) * self.node_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dim)),
            nn.GELU(), nn.LayerNorm(int(hidden_dim)),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(mid_dim)),
            nn.GELU(), nn.LayerNorm(int(mid_dim)),
            nn.Dropout(float(dropout)),
            nn.Linear(int(mid_dim), 1),
        )

    def forward(self, shell_h: torch.Tensor, prot_h: torch.Tensor,
                lig_h: torch.Tensor) -> torch.Tensor:
        if shell_h.dim() != 3 or shell_h.size(1) != self.n_shells:
            raise ValueError(
                f'expected shell_h [B,{self.n_shells},D], got {tuple(shell_h.shape)}')
        B = shell_h.size(0)
        bins = shell_h.reshape(
            B, self.n_bins, self.bin_width, self.node_dim)
        radial = torch.cat([
            bins.mean(dim=2), bins.max(dim=2).values,
        ], dim=-1).reshape(B, -1)
        return self.head(torch.cat([radial, prot_h, lig_h], dim=-1))


class EHIGNDirectionalEdgeReadout(nn.Module):
    """Score edges by a source, destination, and relation triple product.

    The readout sums protein/ligand-to-shell and shell-to-protein/ligand scores separately.
    """

    def __init__(self, node_dim: int, edge_dim: int, pair_dim: int = 192):
        super().__init__()
        if edge_dim <= 0:
            raise ValueError('EHIGN edge readout requires relation/edge embeddings')
        self.pair_dim = int(pair_dim)

        def make_direction():
            modules = nn.ModuleDict({
                'src': nn.Linear(node_dim, self.pair_dim),
                'dst': nn.Linear(node_dim, self.pair_dim),
                'edge': nn.Linear(edge_dim, self.pair_dim),
                # Use a small initialization and no bias for the unnormalized edge sum.
                'score': nn.Linear(self.pair_dim, 1, bias=False),
            })
            nn.init.normal_(modules['score'].weight, mean=0.0, std=1e-3)
            return modules

        self.to_shell = make_direction()
        self.from_shell = make_direction()

    @staticmethod
    def _score_direction(modules: nn.ModuleDict, H: torch.Tensor,
                         src: torch.Tensor, dst: torch.Tensor,
                         edge_embed: torch.Tensor, edge_batch: torch.Tensor,
                         mask: torch.Tensor, B: int) -> torch.Tensor:
        s, d, e, b = src[mask], dst[mask], edge_embed[mask], edge_batch[mask]
        pair = modules['src'](H[s]) * modules['dst'](H[d]) * modules['edge'](e)
        edge_score = modules['score'](pair)
        return _segment_sum(edge_score, b, B)

    def forward(self, H: torch.Tensor, edge_index: torch.Tensor,
                edge_embed: torch.Tensor, edge_batch: torch.Tensor,
                to_shell_mask: torch.Tensor, from_shell_mask: torch.Tensor,
                B: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src, dst = edge_index[0], edge_index[1]
        y_to_shell = self._score_direction(
            self.to_shell, H, src, dst, edge_embed, edge_batch,
            to_shell_mask, B)
        y_from_shell = self._score_direction(
            self.from_shell, H, src, dst, edge_embed, edge_batch,
            from_shell_mask, B)
        y = 0.5 * (y_to_shell + y_from_shell)
        return y, y_to_shell, y_from_shell


class HeteroNodeGraphFusion(nn.Module):
    """Fuse shell, residue, and ligand-token nodes, then read out shell nodes."""

    needs_graph_inputs = True

    def __init__(self, d_lig: int = 128, d_prot: int = 128, node_dim: int = 128,
                 n_shells: int = 60, n_pairs: int = 168,
                 n_layers: int = 3, n_heads: int = 4, dropout: float = 0.2,
                 use_ffn: bool = True, ffn_mult: int = 2,
                 head_hidden: int = 512, head_mid: int = 256,
                 k_shell: int = 2, k_seq: int = 1,
                 prot_shell_topk: int = -1, lig_full: bool = True,
                 use_freq_bias: bool = True, gate_init: float = -4.0,
                 randomize_lm: bool = False,
                 disable_lig: bool = False, disable_prot: bool = False):
        super().__init__()
        assert d_lig == node_dim and d_prot == node_dim, \
            f'要求 d_lig==d_prot==node_dim, got {d_lig},{d_prot},{node_dim}'
        self.dim = int(node_dim)
        self.n_shells = int(n_shells)
        self.n_pairs = int(n_pairs)
        self.n_layers = int(n_layers)
        self.k_shell = int(k_shell)
        self.k_seq = int(k_seq)
        self.prot_shell_topk = int(prot_shell_topk)
        self.lig_full = bool(lig_full)
        self.use_freq_bias = bool(use_freq_bias)
        self.randomize_lm = bool(randomize_lm)
        self.disable_lig = bool(disable_lig)
        self.disable_prot = bool(disable_prot)

        self.type_emb = nn.Embedding(3, self.dim)
        nn.init.normal_(self.type_emb.weight, std=0.02)

        self.blocks = nn.ModuleList([
            HeteroGraphTXBlock(self.dim, n_heads, dropout, use_ffn, ffn_mult)
            for _ in range(self.n_layers)
        ])

        # A negative initial logit makes the injection gate close to zero.
        self.gate_prot = nn.Parameter(torch.tensor(float(gate_init)))
        self.gate_lig = nn.Parameter(torch.tensor(float(gate_init)))
        self.freq_alpha = nn.Parameter(torch.tensor(1.0))

        # Cache the bidirectional shell-chain template on CPU without self-loops.
        self._ss_cpu = self._chain_template(self.n_shells, self.k_shell)

        from .fusion import _tri_mlp_head
        self.head = _tri_mlp_head(self.n_shells * self.dim, head_hidden, head_mid, dropout)

    def _topo_sig(self) -> str:
        """Return topology options for the process-local graph cache key."""
        return (f's{self.n_shells}_ks{self.k_shell}_kq{self.k_seq}_tk{self.prot_shell_topk}'
                f'_lf{int(self.lig_full)}_fb{int(self.use_freq_bias)}'
                f'_dl{int(self.disable_lig)}_dp{int(self.disable_prot)}')

    @staticmethod
    def _chain_template(n: int, k: int) -> torch.Tensor:
        """Return bidirectional shell-chain edges [2, E] where 1 <= |i - j| <= k."""
        src, dst = [], []
        for i in range(n):
            for d in range(1, k + 1):
                if i - d >= 0:
                    src.append(i); dst.append(i - d)
                if i + d < n:
                    src.append(i); dst.append(i + d)
        if not src:
            return torch.zeros(2, 0, dtype=torch.long)
        return torch.tensor([src, dst], dtype=torch.long)

    def _build_local_edges(self, np_b: int, nl_b: int, seq: str,
                           w_prot_b: torch.Tensor) -> dict:
        """Build cacheable CPU edges with shell, protein, then ligand node indices.

        Return (src int32, dst int32, relation int8, log_weight fp16, n_nodes).
        """
        S = self.n_shells
        cpu = torch.device('cpu')
        src: List[torch.Tensor] = []
        dst: List[torch.Tensor] = []
        et: List[torch.Tensor] = []
        lw: List[torch.Tensor] = []

        def add(s, d, etype, logw=None):
            src.append(s.to(torch.int32)); dst.append(d.to(torch.int32))
            et.append(torch.full((s.numel(),), etype, dtype=torch.int8))
            lw.append(logw if logw is not None else torch.zeros(s.numel()))

        p0, l0 = S, S + np_b
        ss = self._ss_cpu
        add(ss[0], ss[1], ET_SS)
        # Connect adjacent residues along the protein backbone.
        if np_b > 1 and self.k_seq > 0:
            ps, pd = [], []
            for d in range(1, self.k_seq + 1):
                idx = torch.arange(np_b - d)
                ps.append(idx); pd.append(idx + d)
                ps.append(idx + d); pd.append(idx)
            ps = torch.cat(ps); pd = torch.cat(pd)
            add(ps + p0, pd + p0, ET_PP)
        # Connect all ligand tokens.
        if nl_b > 1 and self.lig_full:
            ii, jj = torch.meshgrid(torch.arange(nl_b), torch.arange(nl_b), indexing='ij')
            mm = ii != jj
            add(ii[mm] + l0, jj[mm] + l0, ET_LL)
        # Add contact-frequency bias to protein-to-shell edges only.
        if np_b > 0:
            rtypes = seq_to_res_types(seq, np_b, cpu)
            w_p = w_prot_b[rtypes]
            if self.prot_shell_topk > 0 and self.prot_shell_topk < S:
                topv, topi = torch.topk(w_p, self.prot_shell_topk, dim=1)
                rows = torch.arange(np_b).unsqueeze(1).expand_as(topi)
                p_loc = rows.reshape(-1); s_loc = topi.reshape(-1)
                lw_p = torch.log1p(topv.clamp(min=0.0)).reshape(-1)
            else:
                rows = torch.arange(np_b).unsqueeze(1).expand(np_b, S)
                cols = torch.arange(S).unsqueeze(0).expand(np_b, S)
                p_loc = rows.reshape(-1); s_loc = cols.reshape(-1)
                lw_p = torch.log1p(w_p.clamp(min=0.0)).reshape(-1)
            bias = lw_p if self.use_freq_bias else torch.zeros_like(lw_p)
            add(p_loc + p0, s_loc, ET_P2S, bias)
            add(s_loc, p_loc + p0, ET_S2P)
        if nl_b > 0:
            rows = torch.arange(nl_b).unsqueeze(1).expand(nl_b, S)
            cols = torch.arange(S).unsqueeze(0).expand(nl_b, S)
            l_loc = rows.reshape(-1); s_loc = cols.reshape(-1)
            add(l_loc + l0, s_loc, ET_L2S)
            add(s_loc, l_loc + l0, ET_S2L)
        n_b = S + np_b + nl_b
        sl = torch.arange(n_b)
        add(sl, sl, ET_SELF)

        return {
            'src': torch.cat(src), 'dst': torch.cat(dst),
            'etype': torch.cat(et), 'logw': torch.cat(lw).to(torch.float16),
            'n': n_b,
        }

    def _build_graph(self, lig, prot, shell_embed, freq_raw, seqs,
                     prot_mask, lig_mask, pdb_codes=None):
        """Batch disjoint graphs while reusing the cached CPU edge lists."""
        device = shell_embed.device
        B = shell_embed.size(0)
        S = self.n_shells
        sig = self._topo_sig()

        node_feats: List[torch.Tensor] = []
        node_types: List[torch.Tensor] = []
        src_list: List[torch.Tensor] = []
        dst_list: List[torch.Tensor] = []
        et_list: List[torch.Tensor] = []
        lw_list: List[torch.Tensor] = []
        shell_idx = torch.empty(B, S, dtype=torch.long)              # Moves to the device after assembly.

        w_prot_cpu = None       # Compute only on a cache miss.
        offset = 0
        for b in range(B):
            np_b = int(prot_mask[b].sum().item()) if prot_mask is not None else prot.size(1)
            nl_b = int(lig_mask[b].sum().item()) if lig_mask is not None else lig.size(1)
            np_b = max(0, min(np_b, prot.size(1)))
            nl_b = max(0, min(nl_b, lig.size(1)))
            if self.disable_prot:
                np_b = 0
            if self.disable_lig:
                nl_b = 0

            feats = [shell_embed[b]]
            types = [torch.zeros(S, dtype=torch.long, device=device)]
            if np_b > 0:
                pf = prot[b, :np_b]
                if self.randomize_lm:
                    pf = torch.randn_like(pf)
                feats.append(pf)
                types.append(torch.ones(np_b, dtype=torch.long, device=device))
            if nl_b > 0:
                lf = lig[b, :nl_b]
                if self.randomize_lm:
                    lf = torch.randn_like(lf)
                feats.append(lf)
                types.append(torch.full((nl_b,), 2, dtype=torch.long, device=device))
            node_feats.append(torch.cat(feats, dim=0))
            node_types.append(torch.cat(types, dim=0))

            key = f'{sig}|{pdb_codes[b]}' if pdb_codes is not None else None
            loc = _TOPO_CACHE.get(key) if key is not None else None
            if loc is None or loc['n'] != S + np_b + nl_b:
                if w_prot_cpu is None:
                    w_prot_cpu = freq_raw.view(B, N_RES_TYPES, N_LIG_ELES, S).sum(dim=2) \
                        .detach().to('cpu')                          # [B, 21, S]
                loc = self._build_local_edges(np_b, nl_b, seqs[b], w_prot_cpu[b])
                if key is not None:
                    _TOPO_CACHE[key] = loc

            src_list.append(loc['src'].to(torch.int64) + offset)
            dst_list.append(loc['dst'].to(torch.int64) + offset)
            et_list.append(loc['etype'])
            lw_list.append(loc['logw'])
            shell_idx[b] = torch.arange(offset, offset + S)
            offset += S + np_b + nl_b

        H0 = torch.cat(node_feats, dim=0)
        ntype = torch.cat(node_types, dim=0)
        H0 = H0 + self.type_emb(ntype)
        edge_index = torch.stack([torch.cat(src_list), torch.cat(dst_list)], dim=0) \
            .to(device)
        edge_type = torch.cat(et_list).to(device).long()
        edge_logw = torch.cat(lw_list).to(device).float()
        shell_idx = shell_idx.to(device)
        return H0, edge_index, edge_type, edge_logw, shell_idx

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                shell_embed: torch.Tensor, freq_raw: torch.Tensor,
                seqs: List[str],
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None,
                pdb_codes: Optional[List[str]] = None) -> torch.Tensor:
        assert lig.dim() == 3 and prot.dim() == 3 and shell_embed.dim() == 3, \
            'F08-node 需要 lig/prot 序列级 (per_token/per_residue) 与 shell 预block 嵌入'
        H, edge_index, edge_type, edge_logw, shell_idx = self._build_graph(
            lig, prot, shell_embed, freq_raw, seqs, prot_mask, lig_mask, pdb_codes)

        # Only injection edges are gated.
        gate_by_type = torch.ones(N_EDGE_TYPES, device=H.device)
        gate_by_type = gate_by_type.clone()
        gate_by_type[ET_P2S] = torch.sigmoid(self.gate_prot)
        gate_by_type[ET_L2S] = torch.sigmoid(self.gate_lig)
        edge_gate = gate_by_type[edge_type]
        edge_bias = self.freq_alpha * edge_logw

        for blk in self.blocks:
            H = blk(H, edge_index, edge_gate, edge_bias)

        # Gather shell nodes in radial order before flattening.
        B = shell_embed.size(0)
        shell_h = H[shell_idx.reshape(-1)].reshape(B, self.n_shells * self.dim)
        return self.head(shell_h)


class HeteroNodeGraphPooledFusion(nn.Module):
    """Fuse shell nodes with one pooled protein and one pooled ligand node per complex."""

    needs_graph_inputs = True

    _EDGE_MODES = ('topology_only', 'relation_only', 'relation_attr')
    _EDGE_ATTR_MODES = ('full', 'position_only', 'contact_only')

    # Keep relation IDs stable across topology variants.
    REL_SS = 0
    REL_P2S = 1
    REL_S2P = 2
    REL_L2S = 3
    REL_S2L = 4
    REL_SELF = 5
    REL_P2L = 6
    REL_L2P = 7
    REL_K2P = 8
    REL_P2K = 9
    REL_CB2S = 8
    REL_S2CB = 9
    REL_A2S = 10
    REL_S2A = 11
    REL_CB2CB = 12
    REL_A2A = 13
    REL_CB2A = 14
    REL_A2CB = 15
    REL_P2CB = 16
    REL_CB2P = 17
    REL_L2A = 18
    REL_A2L = 19
    N_RELATIONS = 8
    N_RELATIONS_POCKET = 10
    N_RELATIONS_MULTISCALE = 20
    _READOUT_MODES = (
        'shell_only', 'all_nodes', 'shell_concat_raw_lm',
        'shell_bilinear_lm', 'shell_e1_head', 'compact_sym_xattn',
        'all_nodes_smooth_mlp', 'ehign_bidirectional_edge_sum',
        'dual_query_radial', 'all_nodes_pool', 'hetero_type_pool',
        'radial_bin_pool', 'ordered_lowrank', 'full_ordered_residual',
    )

    def __init__(self, d_lig: int = 128, d_prot: int = 128, node_dim: int = 128,
                 n_shells: int = 60, n_layers: int = 3, n_heads: int = 4,
                 dropout: float = 0.2, use_ffn: bool = True, ffn_mult: int = 2,
                 graph_update_mode: str = 'legacy_parallel',
                 gnn_type: str = 'GAT_GCN',
                 norm_style: str = 'pre',
                 head_hidden: int = 512, head_mid: int = 256,
                 k_shell: int = 2, connect_prot_lig: bool = False,
                 lm_shell_edge_mode: str = 'bidirectional',
                 lm_shell_topk: int = -1,
                 lm_shell_topk_scope: str = 'prot',
                 readout_mode: str = 'shell_only',
                 gate_init: Optional[float] = None,
                 feedback_alpha_init: Optional[float] = None,
                 shell_proj_dim: int = 512,
                 e1_head_hidden: int = 1536, e1_head_mid: int = 640,
                 e1_dropout: float = 0.3,
                 sym_readout_dim: int = 384,
                 sym_readout_heads: int = 6,
                 sym_attn_dropout: float = 0.1,
                 lm_node_norm: str = 'none',
                 edge_mode: str = 'topology_only',
                 edge_attr_mode: str = 'full',
                 edge_attr_features: str = 'basic',
                 edge_embed_dim: int = 32,
                 edge_flatten_pool: bool = False,
                 edge_flatten_scope: str = 'all',
                 edge_flatten_attn: bool = False,
                 edge_attr_eps: float = 1e-8,
                 add_self_loops: bool = True,
                 smooth_mlp_dims: Optional[List[int]] = None,
                 smooth_mlp_dropout: float = 0.2,
                 edge_pair_dim: int = 192,
                 dual_query_heads: int = 4,
                 dual_query_hidden: int = 512,
                 dual_query_mid: int = 128,
                 dual_query_dropout: float = 0.2,
                 pool_mode: str = 'mean',
                 pool_hidden: int = 384,
                 pool_mid: int = 128,
                 pool_dropout: float = 0.3,
                 set2set_steps: int = 3,
                 radial_pool_bins: int = 10,
                 lowrank_rank: int = 512,
                 lowrank_hidden: int = 1536,
                 lowrank_latent: int = 384,
                 lowrank_dropout: float = 0.3,
                 lowrank_use_pl_interaction: bool = False,
                 lowrank_use_radial_branch: bool = False,
                 lowrank_radial_bins: int = 10,
                 lowrank_branch_gate_init: float = -2.0,
                 full_residual_hidden: int = 1536,
                 full_residual_latent: int = 384,
                 full_residual_dropout: float = 0.3,
                 full_residual_use_pl: bool = False,
                 full_residual_use_radial: bool = False,
                 full_residual_radial_bins: int = 10,
                 full_residual_gate_init: float = -3.0,
                 layer_mix_mode: str = 'none',
                 layer_mix_gate_init: float = -3.0,
                 relation_calibration: bool = False,
                 edge_score_path: bool = True,
                 edge_value_path: bool = True,
                 edge_value_film: bool = False,
                 layerscale_init: Optional[float] = None,
                 attention_grouping: str = 'destination',
                 relation_fusion: str = 'sum',
                 relation_adapter_rank: int = 0,
                 relation_adapter_paths: str = 'kv',
                 relation_adapter_alpha: float = 1.0,
                 qk_norm: bool = False,
                 qk_temperature: str = 'fixed',
                 spatial_bias: str = 'none',
                 init_residual: str = 'none',
                 extra_modulate: str = 'none',
                 edge_flatten_skip: str = 'none',
                 post_graph_glu: str = 'none',
                 edge_flatten_branch: str = 'none',
                 relation_schedule: str = 'all',
                 edge_dropout: float = 0.0,
                 edge_dropout_scope: str = 'cross',
                 type_adapter_dim: int = 0,
                 type_adapter_scope: str = 'all',
                 post_graph_radial_mode: str = 'none',
                 post_graph_radial_gate_init: float = -3.0,
                 post_graph_radial_order: str = 'radial',
                 use_type_embedding: bool = True,
                 readout_node_dropout: float = 0.0,
                 readout_feature_dropout: float = 0.0,
                 use_ehign_aux: bool = False,
                 output_affine: bool = False,
                 output_affine_scale_init: float = 1.0,
                 randomize_lm: bool = False,
                 disable_lig: bool = False, disable_prot: bool = False,
                 feature_dir_cfg: Optional[Dict] = None,
                 pocket_attention_cfg: Optional[Dict] = None,
                 ligand_attention_cfg: Optional[Dict] = None,
                 edge_state_update: bool = False,
                 edge_flatten_update: bool = False,
                 edge_flatten_update_scope: str = 'all',
                 edge_flatten_update_extra: str = 'none',
                 pl_interact: str = 'none',
                 jk_flatten: str = 'none',
                 jk_flatten_dim: int = 32,
                 diff_flatten: str = 'none',
                 diff_flatten_dim: int = 32,
                 diff_flatten_steps: int = 10,
                 diff_flatten_alpha: float = 0.1,
                 edge_update_hidden: int = 64,
                 pair_readout: str = 'none',
                 pair_readout_scope: str = 'cross',
                 pair_readout_hidden: int = 64,
                 pair_to_node: bool = False,
                 pair_gate_init: float = -3.0,
                 pair_dense_layers: int = 2,
                 pair_dense_dim: int = 32,
                 pair_dense_hidden: int = 64,
                 pair_flatten_mode: str = 'off',
                 pair_flatten_hidden: int = 256,
                 add_pocket_node: bool = False,
                 pocket_node_cfg: Optional[Dict] = None,
                 signed_shell_residual: bool = False,
                 coarse_interface_cfg: Optional[Dict] = None,
                 multiscale_3d_cfg: Optional[Dict] = None,
                 shell_geometry_cfg: Optional[Dict] = None,
                 shell_to_3d_cfg: Optional[Dict] = None,
                 shell_geo_readout_cfg: Optional[Dict] = None,
                 interface_tokens_cfg: Optional[Dict] = None,
                 side_tower_cfg: Optional[Dict] = None,
                 contact_primary_cfg: Optional[Dict] = None,
                 shell_edge_mode: str = 'physical_chain',
                 shell_edge_layer_modes: Optional[List[str]] = None,
                 similarity_topk_k: Optional[int] = None,
                 similarity_topk_update: str = 'per_layer',
                 shell_upper_edges_angstrom=None,
                 similarity_topk_scope: str = 'shell_only'):
        super().__init__()
        self.feature_dir_cfg = dict(feature_dir_cfg or {})
        self.pocket_attention_cfg = dict(pocket_attention_cfg or {})
        self.ligand_attention_cfg = dict(ligand_attention_cfg or {})
        self.pocket_node_cfg = dict(pocket_node_cfg or {})
        self.coarse_interface_cfg = dict(coarse_interface_cfg or {})
        self.multiscale_cfg = dict(multiscale_3d_cfg or {})
        self.shell_geometry_cfg = dict(shell_geometry_cfg or {})
        self.shell_to_3d_cfg = dict(shell_to_3d_cfg or {})
        self.shell_geo_readout_cfg = dict(shell_geo_readout_cfg or {})
        self.interface_tokens_cfg = dict(interface_tokens_cfg or {})
        self.side_tower_cfg = dict(side_tower_cfg or {})
        self.contact_primary_cfg = dict(contact_primary_cfg or {})
        self.multiscale_3d = bool(self.multiscale_cfg.get('enabled', False))
        self.max_cb = int(self.multiscale_cfg.get('max_cb', 64))
        self.max_atom = int(self.multiscale_cfg.get('max_atom', 48))
        self.geo_hier = bool(self.multiscale_cfg.get('hier_edges', True))
        self.geo_intra = bool(self.multiscale_cfg.get('intra_edges', False))
        self.geo_inter = bool(self.multiscale_cfg.get('inter_edges', False))
        self.geo_lm = bool(self.multiscale_cfg.get('lm_broadcast', False))
        self.add_pocket_node = bool(add_pocket_node) or bool(
            self.pocket_node_cfg.get('enabled', False))
        self.connect_pocket_lig = bool(
            self.pocket_node_cfg.get('connect_pocket_lig', False))
        self.connect_pocket_prot = bool(
            self.pocket_node_cfg.get('connect_pocket_prot', False))
        if self.multiscale_3d and self.add_pocket_node:
            raise ValueError('multiscale_3d cannot be combined with add_pocket_node')
        if self.multiscale_3d:
            self.n_graph_relations = self.N_RELATIONS_MULTISCALE
            self.n_node_types = 5
        else:
            self.n_graph_relations = (
                self.N_RELATIONS_POCKET if self.connect_pocket_prot
                else self.N_RELATIONS)
            self.n_node_types = 4 if self.add_pocket_node else 3
        self.edge_state_update = bool(edge_state_update)
        self.edge_flatten_update = bool(edge_flatten_update)
        self.edge_flatten_update_scope = str(
            edge_flatten_update_scope).lower()
        if self.edge_flatten_update_scope not in ('all', 'cross'):
            raise ValueError(
                'edge_flatten_update_scope must be all|cross, got '
                f'{edge_flatten_update_scope!r}')
        self.edge_flatten_update_extra = str(
            edge_flatten_update_extra).lower()
        if self.edge_flatten_update_extra not in (
                'none', 'radial', 'rbf', 'geo'):
            raise ValueError(
                'edge_flatten_update_extra must be none|radial|rbf|geo, got '
                f'{edge_flatten_update_extra!r}')
        self.pl_interact = str(pl_interact).lower()
        if self.pl_interact not in (
                'none', 'hadamard', 'hadamard32', 'bilinear32'):
            raise ValueError(
                'pl_interact must be none|hadamard|hadamard32|bilinear32, got '
                f'{pl_interact!r}')
        self.jk_flatten = str(jk_flatten).lower()
        if self.jk_flatten not in ('none', 'first', 'max', 'concat'):
            raise ValueError(
                'jk_flatten must be none|first|max|concat, got '
                f'{jk_flatten!r}')
        self.jk_flatten_width = int(jk_flatten_dim)
        self.diff_flatten = str(diff_flatten).lower()
        if self.diff_flatten not in (
                'none', 'appnp_mean', 'appnp_lig', 'appnp_h0'):
            raise ValueError(
                'diff_flatten must be none|appnp_mean|appnp_lig|appnp_h0, got '
                f'{diff_flatten!r}')
        self.diff_flatten_width = int(diff_flatten_dim)
        self.diff_flatten_steps = int(diff_flatten_steps)
        self.diff_flatten_alpha = float(diff_flatten_alpha)
        self.last_diff_col_abs = 0.0
        self.pl_interact_dim = {
            'none': 0,
            'hadamard': int(node_dim),
            'hadamard32': 32,
            'bilinear32': 32,
        }[self.pl_interact]
        self.last_pl_col_abs = 0.0
        self.pair_readout = str(pair_readout or 'none').lower()
        self.pair_readout_scope = str(pair_readout_scope or 'cross').lower()
        self.pair_to_node_enabled = bool(pair_to_node)
        self.pair_dense_layers = int(pair_dense_layers)
        self.pair_dense_dim = int(pair_dense_dim)
        self.pair_dense_hidden = int(pair_dense_hidden)
        mode = str(pair_flatten_mode or 'off').lower()
        if mode in ('false', '0', 'none'):
            mode = 'off'
        self.pair_flatten_mode = mode
        self.pair_flatten_hidden = int(pair_flatten_hidden)
        if self.pair_readout not in ('none', 'residual', 'primary'):
            raise ValueError(
                f'pair_readout must be none|residual|primary, got {pair_readout!r}')
        if self.pair_readout_scope not in ('cross', 'cross_pl', 'all'):
            raise ValueError(
                f'pair_readout_scope must be cross|cross_pl|all, '
                f'got {pair_readout_scope!r}')
        if self.pair_flatten_mode not in ('off', 'gated_shrink'):
            raise ValueError(
                f'pair_flatten_mode must be off|gated_shrink, '
                f'got {pair_flatten_mode!r}')
        if (self.edge_state_update or self.pair_readout != 'none'
                or self.pair_to_node_enabled) and str(edge_mode) == 'topology_only':
            raise ValueError(
                'pair/edge-state modules require relation_only or relation_attr')
        assert d_lig == node_dim and d_prot == node_dim, \
            f'要求 d_lig==d_prot==node_dim, got {d_lig},{d_prot},{node_dim}'
        _edge_modes = ('bidirectional', 'inject_only', 'feedback_only')
        assert lm_shell_edge_mode in _edge_modes, \
            f'lm_shell_edge_mode 须为 {_edge_modes}, got {lm_shell_edge_mode!r}'
        self.dim = int(node_dim)
        self.n_shells = int(n_shells)
        shell_upper = _validated_shell_upper_edges(
            self.n_shells, shell_upper_edges_angstrom)
        self.register_buffer(
            'shell_upper_edges_angstrom', shell_upper, persistent=False)
        self.n_layers = int(n_layers)
        self.graph_update_mode = str(graph_update_mode).lower()
        if self.graph_update_mode not in (
                'legacy_parallel', 'shellfm', 'shellgps', 'hgt'):
            raise ValueError(
                'graph_update_mode must be legacy_parallel|shellfm|shellgps|hgt, '
                f'got {graph_update_mode!r}')
        self.gnn_type = str(gnn_type).upper()
        if self.gnn_type not in ('GAT', 'GATV2', 'GCN', 'GIN', 'GAT_GCN'):
            raise ValueError(f'unsupported gnn_type={gnn_type!r}')
        self.norm_style = str(norm_style).lower()
        if self.norm_style not in ('pre', 'post'):
            raise ValueError(f'norm_style must be pre|post, got {norm_style!r}')
        self.k_shell = int(k_shell)
        self.shell_edge_mode = str(shell_edge_mode).strip().lower()
        if self.shell_edge_mode not in (
                'physical_chain', 'similarity_topk', 'attention_topk'):
            raise ValueError(
                'shell_edge_mode must be physical_chain|similarity_topk|'
                'attention_topk, got '
                f'{shell_edge_mode!r}')
        # Attention Top-k ranks shells with the layer's query/key scores.
        self._global_topk_mode = (
            self.shell_edge_mode
            if self.shell_edge_mode in ('similarity_topk', 'attention_topk')
            else None)

        # Keep the physical radius independent of the similarity neighbor count.
        self._shell_edge_layer_modes_explicit = (
            shell_edge_layer_modes is not None)
        if shell_edge_layer_modes is None:
            self.shell_edge_layer_modes = (
                self.shell_edge_mode,) * self.n_layers
        else:
            if (isinstance(shell_edge_layer_modes, (str, bytes))
                    or not isinstance(shell_edge_layer_modes, (list, tuple))):
                raise ValueError(
                    'shell_edge_layer_modes must be a list/tuple with exactly '
                    'one mode per graph layer')
            if len(shell_edge_layer_modes) != self.n_layers:
                raise ValueError(
                    'shell_edge_layer_modes length must equal n_layers, got '
                    f'{len(shell_edge_layer_modes)} != {self.n_layers}')
            layer_modes = tuple(
                str(mode).strip().lower()
                for mode in shell_edge_layer_modes)
            invalid_modes = sorted(set(layer_modes).difference(
                ('physical_chain', 'similarity_topk', 'attention_topk')))
            if invalid_modes:
                raise ValueError(
                    'shell_edge_layer_modes entries must be '
                    'physical_chain|similarity_topk|attention_topk, got '
                    f'{invalid_modes!r}')
            if self.shell_edge_mode != 'physical_chain':
                raise ValueError(
                    'explicit shell_edge_layer_modes requires '
                    'shell_edge_mode=physical_chain; use the legacy global '
                    'shell_edge_mode=similarity_topk for all-Top-k')
            later_topk = [
                layer for layer, mode in enumerate(layer_modes)
                if layer > 0 and mode in ('similarity_topk', 'attention_topk')
            ]
            if later_topk:
                raise ValueError(
                    'explicit shell_edge_layer_modes permits a Top-k topology '
                    'only in layer 0, got later layers '
                    f'{later_topk!r}')
            self.shell_edge_layer_modes = layer_modes

        has_similarity_topology = (
            self.shell_edge_mode in ('similarity_topk', 'attention_topk')
            if not self._shell_edge_layer_modes_explicit
            else any(
                mode in ('similarity_topk', 'attention_topk')
                for mode in self.shell_edge_layer_modes))
        if similarity_topk_k is not None and not has_similarity_topology:
            raise ValueError(
                'similarity_topk_k requires an active similarity_topk or '
                'attention_topk layer')
        self.similarity_topk_k = int(
            self.k_shell if similarity_topk_k is None
            else similarity_topk_k)
        self.similarity_topk_update = str(
            similarity_topk_update).strip().lower()
        if self.similarity_topk_update not in (
                'per_layer', 'first_layer_reuse'):
            raise ValueError(
                'similarity_topk_update must be '
                'per_layer|first_layer_reuse, got '
                f'{similarity_topk_update!r}')
        if self.similarity_topk_update == 'first_layer_reuse':
            if self._shell_edge_layer_modes_explicit:
                raise ValueError(
                    'similarity_topk_update=first_layer_reuse conflicts '
                    'with explicit shell_edge_layer_modes')
            if self.shell_edge_mode not in (
                    'similarity_topk', 'attention_topk'):
                raise ValueError(
                    'similarity_topk_update=first_layer_reuse requires '
                    'shell_edge_mode=similarity_topk|attention_topk')
        self._layerwise_shell_topology = (
            self._shell_edge_layer_modes_explicit
            and any(mode in ('similarity_topk', 'attention_topk')
                    for mode in self.shell_edge_layer_modes))
        self.similarity_topk_scope = str(similarity_topk_scope).strip().lower()
        if self.similarity_topk_scope not in ('shell_only', 'all_relations'):
            raise ValueError(
                'similarity_topk_scope must be shell_only|all_relations, got '
                f'{self.similarity_topk_scope!r}')
        if self.similarity_topk_scope == 'all_relations':
            if self.shell_edge_mode != 'similarity_topk' or self._shell_edge_layer_modes_explicit:
                raise ValueError(
                    'similarity_topk_scope=all_relations requires a global '
                    'shell_edge_mode=similarity_topk')
            if connect_prot_lig or self.add_pocket_node or self.multiscale_3d:
                raise ValueError(
                    'all_relations supports shell, protein and ligand nodes '
                    'only, without direct protein-ligand edges')
            if bool(self.contact_primary_cfg.get('enabled', False)):
                raise ValueError('all_relations cannot bypass the shell graph')
            if int(lm_shell_topk) != -1:
                raise ValueError(
                    'all_relations uses similarity_topk_k for every relation; '
                    'leave the contact-frequency lm_shell_topk at -1')
        if has_similarity_topology:
            if (self.similarity_topk_scope == 'all_relations'
                    and self.similarity_topk_k < 1):
                raise ValueError('all_relations requires similarity_topk_k >= 1')
            if (self.similarity_topk_scope == 'shell_only'
                    and self.similarity_topk_k not in (3, 5, 7, 9)):
                raise ValueError(
                    'similarity_topk/attention_topk requires '
                    'similarity_topk_k to be one of '
                    f'3, 5, 7, 9, got {self.similarity_topk_k}')
            if self.similarity_topk_k >= self.n_shells:
                raise ValueError(
                    'similarity_topk/attention_topk requires '
                    'similarity_topk_k < n_shells, '
                    f'got {self.similarity_topk_k} >= {self.n_shells}')
            if self.n_layers < 1:
                raise ValueError(
                    'similarity_topk/attention_topk requires at least one '
                    'graph layer')
            if self.edge_state_update:
                raise ValueError(
                    'similarity_topk/attention_topk is incompatible with '
                    'edge_state_update: edge identities change between '
                    'layers, so carrying edge state across layers is not '
                    'well-defined')
        self.connect_prot_lig = bool(connect_prot_lig)
        self.lm_shell_edge_mode = str(lm_shell_edge_mode)
        self.lm_shell_topk = int(lm_shell_topk)
        assert readout_mode in self._READOUT_MODES, \
            f'readout_mode 须为 {self._READOUT_MODES}, got {readout_mode!r}'
        self.readout_mode = str(readout_mode)
        self.lm_node_norm_mode = str(lm_node_norm or 'none').lower()
        assert edge_mode in self._EDGE_MODES, \
            f'edge_mode 须为 {self._EDGE_MODES}, got {edge_mode!r}'
        assert edge_attr_mode in self._EDGE_ATTR_MODES, \
            f'edge_attr_mode 须为 {self._EDGE_ATTR_MODES}, got {edge_attr_mode!r}'
        self.edge_mode = str(edge_mode)
        self.edge_attr_mode = str(edge_attr_mode)
        self.edge_attr_features = str(edge_attr_features).lower()
        edge_attr_dims = {
            'basic': 3,
            'basic_padded': 11,
            'radial_delta': 5,
            'shell_contact': 6,
            'chemical_entropy': 6,
            'extended': 11,
        }
        if self.edge_attr_features not in edge_attr_dims:
            raise ValueError(
                'edge_attr_features must be basic|radial_delta|shell_contact|'
                'chemical_entropy|extended|basic_padded')
        self.edge_attr_input_dim = edge_attr_dims[self.edge_attr_features]
        self.edge_embed_dim = int(edge_embed_dim)
        self.edge_flatten_pool = bool(edge_flatten_pool)
        self.edge_flatten_scope = str(edge_flatten_scope).lower()
        if self.edge_flatten_scope not in ('all', 'cross'):
            raise ValueError(
                'edge_flatten_scope must be all|cross, got '
                f'{edge_flatten_scope!r}')
        self.edge_flatten_attn = bool(edge_flatten_attn)
        self.edge_flatten_dim = (
            self.edge_embed_dim if self.edge_flatten_pool else 0)
        self.jk_flatten_dim = (
            int(self.jk_flatten_width) if self.jk_flatten != 'none' else 0)
        self.jk_proj = None
        if self.jk_flatten == 'first' or self.jk_flatten == 'max':
            self.jk_proj = nn.Linear(self.dim, self.jk_flatten_dim)
        elif self.jk_flatten == 'concat':
            self.jk_proj = nn.Linear(
                self.dim * max(int(self.n_layers), 1), self.jk_flatten_dim)
        self.diff_flatten_dim = (
            int(self.diff_flatten_width) if self.diff_flatten != 'none' else 0)
        self.diff_proj = (
            nn.Linear(self.dim, self.diff_flatten_dim)
            if self.diff_flatten_dim > 0 else None)
        if self.edge_flatten_pool and self.edge_flatten_attn:
            self.edge_flatten_score = nn.Linear(self.edge_embed_dim, 1)
            nn.init.zeros_(self.edge_flatten_score.weight)
            nn.init.zeros_(self.edge_flatten_score.bias)
        else:
            self.edge_flatten_score = None
        self.last_edge_pool_abs = 0.0
        self.edge_attr_eps = float(edge_attr_eps)
        self.add_self_loops = bool(add_self_loops)
        self.layer_mix_mode = str(layer_mix_mode).lower()
        if self.layer_mix_mode not in ('none', 'global', 'typewise'):
            raise ValueError(
                'layer_mix_mode must be none|global|typewise, got '
                f'{layer_mix_mode!r}')
        self.relation_calibration = bool(relation_calibration)
        self.edge_score_path = bool(edge_score_path)
        self.edge_value_path = bool(edge_value_path)
        self.edge_value_film = bool(edge_value_film)
        self.layerscale_init = (
            None if layerscale_init is None else float(layerscale_init))
        self.attention_grouping = str(attention_grouping).lower()
        self.relation_fusion = str(relation_fusion).lower()
        self.relation_adapter_rank = int(relation_adapter_rank)
        self.relation_adapter_paths = str(relation_adapter_paths).lower()
        self.relation_adapter_alpha = float(relation_adapter_alpha)
        self.qk_norm = bool(qk_norm)
        self.qk_temperature = str(qk_temperature).lower()
        self.spatial_bias = str(spatial_bias).lower()
        if self.spatial_bias not in ('none', 'graphormer', 'alibi', 'ss'):
            raise ValueError(
                'spatial_bias must be none|graphormer|alibi|ss, got '
                f'{spatial_bias!r}')
        self.last_spatial_abs = 0.0
        self.init_residual = str(init_residual).lower()
        if self.init_residual not in ('none', 'last', 'every', 'shells'):
            raise ValueError(
                'init_residual must be none|last|every|shells, got '
                f'{init_residual!r}')
        self.init_residual_alpha = (
            nn.Parameter(torch.zeros(()))
            if self.init_residual != 'none' else None)
        self.extra_modulate_mode = str(extra_modulate).lower()
        if self.extra_modulate_mode not in ('none', 'se_h', 'film_h', 'phi'):
            raise ValueError(
                'extra_modulate must be none|se_h|film_h|phi, got '
                f'{extra_modulate!r}')
        self.extra_modulate = None
        self.edge_flatten_skip = str(edge_flatten_skip).lower()
        if self.edge_flatten_skip not in ('none', 'e0', 'densenet', 'gate'):
            raise ValueError(
                'edge_flatten_skip must be none|e0|densenet|gate, got '
                f'{edge_flatten_skip!r}')
        self.edge_flatten_skip_alpha = None
        self.edge_flatten_skip_gate = None
        self.edge_flatten_skip_proj = None
        self.relation_schedule = str(relation_schedule).lower()
        if self.relation_schedule not in (
                'all', 'inject_ss_feedback', 'inject_all_feedback',
                'alternating'):
            raise ValueError(
                'relation_schedule must be all|inject_ss_feedback|'
                'inject_all_feedback|alternating')
        self.edge_dropout = float(edge_dropout)
        self.edge_dropout_scope = str(edge_dropout_scope).lower()
        if not 0.0 <= self.edge_dropout < 1.0:
            raise ValueError('edge_dropout must be in [0,1)')
        if self.edge_dropout_scope not in ('cross', 'shell', 'all_nonself'):
            raise ValueError(
                'edge_dropout_scope must be cross|shell|all_nonself')
        self.post_graph_radial_mode = str(post_graph_radial_mode).lower()
        self.post_graph_radial_order = str(post_graph_radial_order).lower()
        if self.post_graph_radial_order not in ('radial', 'permuted'):
            raise ValueError(
                'post_graph_radial_order must be radial|permuted')
        if (self.post_graph_radial_mode == 'none'
                and self.post_graph_radial_order != 'radial'):
            raise ValueError(
                'permuted radial order requires a post-graph radial mode')
        if self.post_graph_radial_mode not in (
                'none', *PostGraphRadialResidual.MODES):
            raise ValueError(
                'post_graph_radial_mode must be none|k3|k5|dilated3|multiscale')
        self.type_adapter_dim = int(type_adapter_dim)
        self.type_adapter_scope = str(type_adapter_scope).lower()
        if self.type_adapter_dim < 0:
            raise ValueError('type_adapter_dim must be non-negative')
        if self.type_adapter_scope not in ('all', 'shell', 'lm', 'shared'):
            raise ValueError('type_adapter_scope must be all|shell|lm|shared')
        if self.type_adapter_dim > 0 and self.graph_update_mode != 'shellfm':
            raise ValueError('type adapters currently require shellfm mode')
        self.use_type_embedding = bool(use_type_embedding)
        self.readout_node_dropout = float(readout_node_dropout)
        self.readout_feature_dropout = float(readout_feature_dropout)
        if not 0.0 <= self.readout_node_dropout < 1.0:
            raise ValueError('readout_node_dropout must be in [0,1)')
        if not 0.0 <= self.readout_feature_dropout < 1.0:
            raise ValueError('readout_feature_dropout must be in [0,1)')
        if self.layer_mix_mode != 'none' and self.n_layers < 1:
            raise ValueError('layer mixing requires at least one graph layer')
        if self.relation_calibration and not (
                self.graph_update_mode == 'shellfm'
                and self.gnn_type == 'GAT'):
            raise ValueError(
                'relation_calibration requires shellfm with GAT layers')
        self.use_ehign_aux = bool(use_ehign_aux)
        self.randomize_lm = bool(randomize_lm)
        self.disable_lig = bool(disable_lig)
        self.disable_prot = bool(disable_prot)
        self.output_affine = bool(output_affine)
        if self.output_affine:
            self.pred_scale = nn.Parameter(
                torch.tensor(float(output_affine_scale_init)))
            self.pred_bias = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_parameter('pred_scale', None)
            self.register_parameter('pred_bias', None)
        if self.graph_update_mode in ('shellfm', 'shellgps', 'hgt'):
            if gate_init is not None or feedback_alpha_init is not None:
                raise ValueError(
                    'ShellFM/ShellGPS/HGT mode does not use legacy edge gates or '
                    'feedback scaling')
        # Normalize the graph inputs without changing the projected vectors used by raw-LM readouts.
        self.lig_node_norm = build_lm_node_norm(self.lm_node_norm_mode, self.dim)
        self.prot_node_norm = build_lm_node_norm(self.lm_node_norm_mode, self.dim)
        from .pocket_attention import (
            build_dilution_gate, build_pocket_attention)
        self.pocket_attn = build_pocket_attention(
            self.dim, self.pocket_attention_cfg)
        self.pocket_gate = build_dilution_gate(self.pocket_attention_cfg)
        self._pocket_cache = None
        from .ligand_attention import build_ligand_attention
        self.ligand_attn = build_ligand_attention(
            self.dim, self.ligand_attention_cfg)
        self._ligand_cache = None
        self._dynamic_prot_shell = (
            not self.disable_prot
            and 0 < self.lm_shell_topk < self.n_shells
        )
        # prot_lig prunes both semantic nodes using the same contact-based shell ranking.
        self.lm_shell_topk_scope = str(lm_shell_topk_scope).strip().lower()
        if self.lm_shell_topk_scope not in ('prot', 'prot_lig'):
            raise ValueError(
                'lm_shell_topk_scope must be prot|prot_lig, got '
                f'{lm_shell_topk_scope!r}')
        if self.lm_shell_topk_scope == 'prot_lig' and not (
                0 < self.lm_shell_topk < self.n_shells):
            raise ValueError(
                'lm_shell_topk_scope=prot_lig requires '
                '0 < lm_shell_topk < n_shells, got '
                f'{self.lm_shell_topk}')
        self._dynamic_lig_shell = (
            not self.disable_lig
            and self.lm_shell_topk_scope == 'prot_lig'
            and 0 < self.lm_shell_topk < self.n_shells
        )

        if self.use_type_embedding:
            self.type_emb = nn.Embedding(self.n_node_types, self.dim)
            nn.init.normal_(self.type_emb.weight, std=0.02)
        else:
            self.type_emb = None
        self.signed_shell_residual = bool(signed_shell_residual)
        self.signed_shell = None
        if self.signed_shell_residual:
            from .shell_graph_layers import SignedShellResidual
            self.signed_shell = SignedShellResidual(self.dim)
        self.coarse_interface = None
        self._coarse_cache = None
        self.interface_extra_nodes = 0
        self.interface_concat_nodes = 0
        self.last_iface_col_abs = 0.0
        self.interface_edge_readout = False
        if bool(self.coarse_interface_cfg.get('enabled', False)):
            from .coarse_interface import CoarseInterfaceEncoder
            extra = int(self.coarse_interface_cfg.get('extra_nodes', 0))
            extra_cb = int(self.coarse_interface_cfg.get('extra_cb', 16))
            extra_lig = int(self.coarse_interface_cfg.get('extra_lig', 8))
            self.interface_edge_readout = bool(
                self.coarse_interface_cfg.get('edge_readout', False))
            self.coarse_interface = CoarseInterfaceEncoder(
                node_dim=self.dim,
                n_layers=int(self.coarse_interface_cfg.get('n_layers', 2)),
                n_shells=self.n_shells,
                extra_nodes=extra,
                extra_cb=extra_cb,
                extra_lig=extra_lig,
                edge_readout=self.interface_edge_readout,
                dropout=float(self.coarse_interface_cfg.get('dropout', 0.1)))
            self.interface_extra_nodes = int(self.coarse_interface.extra_nodes)
        self.shell_geometry = None
        if bool(self.shell_geometry_cfg.get('enabled', False)):
            from .shell_geometry import ShellInstanceGeometry
            self.shell_geometry = ShellInstanceGeometry(
                node_dim=self.dim, n_shells=self.n_shells,
                mode=str(self.shell_geometry_cfg.get('mode', 'stats')))
        self.shell_to_3d = None
        if bool(self.shell_to_3d_cfg.get('enabled', False)):
            from .shell_geometry import ShellTo3DAttention
            self.shell_to_3d = ShellTo3DAttention(
                node_dim=self.dim, n_shells=self.n_shells,
                n_heads=int(self.shell_to_3d_cfg.get('n_heads', 4)),
                dropout=float(self.shell_to_3d_cfg.get('dropout', 0.1)),
                radius_width=float(self.shell_to_3d_cfg.get('radius_width', 0.5)))
        self.shell_geo_readout = None
        self.geo_edge_proj = None
        self.geo_head = None
        self.geo_film = None
        self.geo_concat_flatten = False
        self.geo_concat_dim = 0
        self.geo_tie_mode = 'none'
        self.geo_stem = None
        self.geo_residual_proj = None
        self.geo_factor_A = None
        self.geo_factor_B = None
        self.last_geo_col_abs = 0.0
        self.last_geo_edge_abs = 0.0
        if bool(self.shell_geo_readout_cfg.get('enabled', False)):
            from .shell_geometry import ShellGeoChannels, ShellGeoHead, ShellGeoFilm
            self.shell_geo_readout = ShellGeoChannels(
                mode=str(self.shell_geo_readout_cfg.get('mode', 'contact')),
                n_shells=self.n_shells)
            self.geo_concat_flatten = bool(
                self.shell_geo_readout_cfg.get('concat_flatten', True))
            self.geo_concat_dim = int(self.shell_geo_readout.out_dim)
            tie = str(self.shell_geo_readout_cfg.get('tie_mode', 'none') or 'none').lower()
            if tie in ('false', '0', 'off', ''):
                tie = 'none'
            if tie not in ('none', 'factor', 'residual', 'stem'):
                raise ValueError(
                    f"shell_geo_readout.tie_mode must be none|factor|residual|stem, got {tie!r}")
            self.geo_tie_mode = tie
            extra = int(self.shell_geo_readout.out_dim)
            if tie == 'factor':
                self.geo_concat_flatten = False
                self.geo_concat_dim = 0
                rank = int(self.shell_geo_readout_cfg.get('factor_rank', 16))
                hid = int((smooth_mlp_dims or [1536, 384])[0])
                if rank <= 0:
                    raise ValueError(f'factor_rank must be > 0, got {rank}')
                self.geo_factor_A = nn.Parameter(
                    torch.zeros(self.n_shells, hid, rank))
                self.geo_factor_B = nn.Parameter(torch.zeros(rank, extra))
            elif tie == 'residual':
                self.geo_concat_flatten = False
                self.geo_concat_dim = 0
                self.geo_residual_proj = nn.Linear(extra, self.dim)
                nn.init.zeros_(self.geo_residual_proj.weight)
                nn.init.zeros_(self.geo_residual_proj.bias)
            elif tie == 'stem':
                stem_dim = int(self.shell_geo_readout_cfg.get('stem_dim', 32))
                if stem_dim <= 0:
                    raise ValueError(f'stem_dim must be > 0, got {stem_dim}')
                self.geo_stem = nn.Linear(extra, stem_dim)
                nn.init.zeros_(self.geo_stem.weight)
                nn.init.zeros_(self.geo_stem.bias)
                self.geo_concat_flatten = True
                self.geo_concat_dim = stem_dim
            if bool(self.shell_geo_readout_cfg.get('edge_inject', False)):
                self.geo_edge_proj = nn.Linear(
                    self.shell_geo_readout.out_dim, self.edge_embed_dim)
                nn.init.zeros_(self.geo_edge_proj.weight)
                nn.init.zeros_(self.geo_edge_proj.bias)
            if bool(self.shell_geo_readout_cfg.get('dedicated_head', False)):
                self.geo_head = ShellGeoHead(
                    in_dim=self.n_shells * self.shell_geo_readout.out_dim,
                    hidden=int(self.shell_geo_readout_cfg.get('head_hidden', 384)),
                    dropout=float(self.shell_geo_readout_cfg.get('head_dropout', 0.2)))
            if bool(self.shell_geo_readout_cfg.get('film', False)):
                self.geo_film = ShellGeoFilm(
                    geo_dim=self.shell_geo_readout.out_dim,
                    node_dim=self.dim,
                    use_shift=bool(self.shell_geo_readout_cfg.get('film_shift', False)))
        self.contact_primary = None
        if bool(self.contact_primary_cfg.get('enabled', False)):
            from .contact_primary import ContactPrimaryGraph
            self.contact_primary = ContactPrimaryGraph(
                dim=self.dim,
                n_layers=int(self.contact_primary_cfg.get('n_layers', 3)),
                dropout=float(self.contact_primary_cfg.get('dropout', 0.1)),
                onion_feat=bool(self.contact_primary_cfg.get('onion_feat', False)),
                readout=str(self.contact_primary_cfg.get('readout', 'pool')),
                n_cb_read=int(self.contact_primary_cfg.get('n_cb_read', 32)),
                n_atom_read=int(self.contact_primary_cfg.get('n_atom_read', 16)),
                use_lm=bool(self.contact_primary_cfg.get('use_lm', True)))
        self.interface_tokens = None
        if bool(self.interface_tokens_cfg.get('enabled', False)):
            from .interface_tokens import InterfaceTokenGraph
            self.interface_tokens = InterfaceTokenGraph(
                dim=self.dim,
                n_heads=int(self.interface_tokens_cfg.get('n_heads', 4)),
                n_layers=int(self.interface_tokens_cfg.get('n_layers', 2)),
                dropout=float(self.interface_tokens_cfg.get('dropout', 0.1)),
                mode=str(self.interface_tokens_cfg.get('mode', 'residual')))
            tok_mode = self.interface_tokens.mode
            if tok_mode == 'concat':
                self.interface_concat_nodes = 2
            elif tok_mode == 'concat_one':
                self.interface_concat_nodes = 1
        self.side_tower = None
        if bool(self.side_tower_cfg.get('enabled', False)):
            from .side_shell_tower import SideShellTower
            self.side_tower = SideShellTower(
                mode=str(self.side_tower_cfg.get('mode', 'rich_pair')),
                n_shells=self.n_shells,
                token_dim=int(self.side_tower_cfg.get('token_dim', self.dim)),
                per_shell=int(self.side_tower_cfg.get('per_shell', 16)),
                hidden=int(self.side_tower_cfg.get('hidden', 384)),
                n_layers=int(self.side_tower_cfg.get('n_layers', 0)),
                n_heads=int(self.side_tower_cfg.get('n_heads', 4)),
                dropout=float(self.side_tower_cfg.get('dropout', 0.1)))
        self.geo_embed = None
        self._coarse_cache = getattr(self, '_coarse_cache', None)
        if self.multiscale_3d:
            from .multiscale_3d import GeoNodeEmbed
            self.geo_embed = GeoNodeEmbed(self.dim)
            self.last_n_cb = 0.0
            self.last_n_atom = 0.0
            self.last_n_hier = 0.0
            self.last_n_pl = 0.0
        self.pocket_pool = None
        if self.add_pocket_node:
            from .pocket_attention import PocketNodePool
            self.pocket_pool = PocketNodePool(
                self.dim,
                pocket_hidden=int(self.pocket_node_cfg.get(
                    'pocket_hidden', 1024)))

        if self.graph_update_mode in ('shellfm', 'shellgps'):
            self.schemes = [
                ('GAT' if li % 2 == 0 else 'GCN')
                if self.gnn_type == 'GAT_GCN' else self.gnn_type
                for li in range(self.n_layers)
            ]
            block_edge_dim = (
                self.edge_embed_dim if self.edge_mode != 'topology_only' else 0)
            if self.graph_update_mode == 'shellfm':
                self.blocks = nn.ModuleList([
                    ShellFMHeteroGraphTransformerBlock(
                        self.dim, n_heads, scheme, dropout, use_ffn,
                        ffn_mult, edge_dim=block_edge_dim,
                        norm_style=self.norm_style,
                        relation_calibration=self.relation_calibration,
                        n_relations=self.n_graph_relations,
                        edge_score_path=self.edge_score_path,
                        edge_value_path=self.edge_value_path,
                        edge_value_film=self.edge_value_film,
                        layerscale_init=self.layerscale_init,
                        attention_grouping=self.attention_grouping,
                        relation_fusion=self.relation_fusion,
                        relation_adapter_rank=self.relation_adapter_rank,
                        relation_adapter_paths=self.relation_adapter_paths,
                        relation_adapter_alpha=self.relation_adapter_alpha,
                        qk_norm=self.qk_norm,
                        qk_temperature=self.qk_temperature,
                        spatial_bias=self.spatial_bias,
                        type_adapter_dim=self.type_adapter_dim,
                        type_adapter_scope=self.type_adapter_scope)
                    for scheme in self.schemes
                ])
            else:
                self.blocks = nn.ModuleList([
                    ShellFMHeteroGPSBlock(
                        self.dim, n_heads, scheme, dropout, use_ffn,
                        ffn_mult, edge_dim=block_edge_dim,
                        norm_style=self.norm_style)
                    for scheme in self.schemes
                ])
        elif self.graph_update_mode == 'hgt':
            self.schemes = ['HGT'] * self.n_layers
            block_edge_dim = (
                self.edge_embed_dim
                if self.edge_mode != 'topology_only' else 0)
            self.blocks = nn.ModuleList([
                HeterogeneousGraphTransformerBlock(
                    dim=self.dim, n_heads=n_heads,                     n_node_types=self.n_node_types,
                    n_relations=self.n_graph_relations, dropout=dropout,
                    use_ffn=use_ffn, ffn_mult=ffn_mult,
                    edge_dim=block_edge_dim, norm_style=self.norm_style)
                for _ in range(self.n_layers)
            ])
        else:
            self.schemes = ['legacy_parallel'] * self.n_layers
            block_edge_dim = (
                self.edge_embed_dim
                if self.edge_mode != 'topology_only' else 0)
            self.blocks = nn.ModuleList([
                HeteroGraphTXBlock(
                    self.dim, n_heads, dropout, use_ffn, ffn_mult,
                    edge_dim=block_edge_dim,
                )
                for _ in range(self.n_layers)
            ])

        # Attention Top-k requires the sparse GAT block's query/key projections.
        if 'attention_topk' in self.shell_edge_layer_modes:
            if self.graph_update_mode != 'shellfm':
                raise ValueError(
                    'shell_edge_mode=attention_topk requires '
                    'graph_update_mode=shellfm (got '
                    f'{self.graph_update_mode!r})')
            unsupported = [
                layer for layer, mode in enumerate(self.shell_edge_layer_modes)
                if mode == 'attention_topk' and self.schemes[layer] != 'GAT'
            ]
            if unsupported:
                raise ValueError(
                    'shell_edge_mode=attention_topk reuses the GAT attention '
                    'scores and requires every Top-k layer to use gnn_type=GAT; '
                    f'unsupported layers {unsupported} with schemes '
                    f'{self.schemes!r}')
            if self.attention_grouping != 'destination':
                raise ValueError(
                    'shell_edge_mode=attention_topk requires '
                    'attention_grouping=destination')
            if self.qk_temperature == 'learned' and not self.qk_norm:
                raise ValueError(
                    'learned QK temperature requires qk_norm=true')

        # Mix earlier layer states uniformly at initialization.
        if self.layer_mix_mode == 'global':
            self.layer_mix_logits = nn.Parameter(torch.zeros(self.n_layers))
            self.layer_mix_gate = nn.Parameter(
                torch.tensor(float(layer_mix_gate_init)))
        elif self.layer_mix_mode == 'typewise':
            self.layer_mix_logits = nn.Parameter(torch.zeros(
                self.n_node_types, self.n_layers))
            self.layer_mix_gate = nn.Parameter(torch.full(
                (self.n_node_types,), float(layer_mix_gate_init)))
        if self.edge_mode != 'topology_only':
            self.edge_rel_emb = nn.Embedding(
                self.n_graph_relations, self.edge_embed_dim)
            nn.init.normal_(self.edge_rel_emb.weight, std=0.02)
            if self.multiscale_3d:
                with torch.no_grad():
                    self.edge_rel_emb.weight[self.N_RELATIONS:].zero_()
        if self.edge_mode == 'relation_attr':
            self.edge_attr_encoder = nn.Sequential(
                nn.Linear(
                    self.edge_attr_input_dim, self.edge_embed_dim,
                    bias=False),
                nn.GELU(),
                nn.Linear(self.edge_embed_dim, self.edge_embed_dim, bias=False),
            )

        from .pair_state import (
            EdgeStateUpdate, build_edge_updaters, build_pair_residual,
            build_pair_to_node)
        pair_cfg = {
            'pair_readout': self.pair_readout,
            'pair_readout_hidden': int(pair_readout_hidden),
            'pair_to_node': self.pair_to_node_enabled,
            'pair_gate_init': float(pair_gate_init),
            'pair_dense_layers': self.pair_dense_layers,
            'pair_dense_dim': self.pair_dense_dim,
            'pair_dense_hidden': self.pair_dense_hidden,
        }
        self.edge_updaters = (
            build_edge_updaters(
                self.n_layers, self.dim, self.edge_embed_dim,
                int(edge_update_hidden))
            if self.edge_state_update else None)
        extra_dim = 0
        if self.edge_flatten_update_extra == 'radial':
            extra_dim = 1
        elif self.edge_flatten_update_extra == 'rbf':
            extra_dim = 8
        elif self.edge_flatten_update_extra == 'geo':
            extra_dim = int(self.geo_concat_dim)
            if extra_dim <= 0:
                raise ValueError(
                    'edge_flatten_update_extra=geo requires shell_geo_readout')
        self.edge_flatten_extra_dim = extra_dim
        if extra_dim == 8:
            self.register_buffer(
                'edge_rbf_centers', torch.linspace(0.0, 1.0, 8))
            self.edge_rbf_width = 0.15
        else:
            self.edge_rbf_centers = None
            self.edge_rbf_width = 0.15
        self.edge_flatten_updater = (
            EdgeStateUpdate(
                self.dim, self.edge_embed_dim, int(edge_update_hidden),
                extra_dim=extra_dim)
            if self.edge_flatten_update else None)
        if self.edge_flatten_skip != 'none':
            if not self.edge_flatten_pool:
                raise ValueError(
                    'edge_flatten_skip requires edge_flatten_pool')
            ed = int(self.edge_embed_dim)
            if self.edge_flatten_skip == 'e0':
                self.edge_flatten_skip_alpha = nn.Parameter(torch.zeros(()))
            elif self.edge_flatten_skip == 'gate':
                self.edge_flatten_skip_gate = nn.Parameter(
                    torch.tensor(-4.0))
            else:
                self.edge_flatten_skip_proj = nn.Linear(ed * 2, ed)
                with torch.no_grad():
                    self.edge_flatten_skip_proj.weight.zero_()
                    self.edge_flatten_skip_proj.weight[:, :ed] = torch.eye(ed)
                    self.edge_flatten_skip_proj.bias.zero_()
        self.post_graph_glu_mode = str(post_graph_glu).lower()
        if self.post_graph_glu_mode not in ('none', 'swiglu', 'geglu'):
            raise ValueError(
                'post_graph_glu must be none|swiglu|geglu, got '
                f'{post_graph_glu!r}')
        self.post_graph_glu = (
            PostGraphGLU(self.dim, self.post_graph_glu_mode)
            if self.post_graph_glu_mode != 'none' else None)
        self.edge_flatten_branch = str(edge_flatten_branch).lower()
        if self.edge_flatten_branch not in ('none', 'cross', 'ss', 'incoming'):
            raise ValueError(
                'edge_flatten_branch must be none|cross|ss|incoming, got '
                f'{edge_flatten_branch!r}')
        self.edge_flatten_branch_proj = None
        if self.edge_flatten_branch != 'none':
            if not self.edge_flatten_pool:
                raise ValueError(
                    'edge_flatten_branch requires edge_flatten_pool')
            ed = int(self.edge_embed_dim)
            self.edge_flatten_branch_proj = nn.Linear(ed * 2, ed)
            with torch.no_grad():
                self.edge_flatten_branch_proj.weight.zero_()
                self.edge_flatten_branch_proj.weight[:, :ed] = torch.eye(ed)
                self.edge_flatten_branch_proj.bias.zero_()
        self.pl_hadamard_proj = None
        self.pl_bilin_p = None
        self.pl_bilin_l = None
        if self.pl_interact == 'hadamard32':
            self.pl_hadamard_proj = nn.Linear(self.dim, 32)
        elif self.pl_interact == 'bilinear32':
            self.pl_bilin_p = nn.Linear(self.dim, 32)
            self.pl_bilin_l = nn.Linear(self.dim, 32)
        self.pair_residual = build_pair_residual(pair_cfg, self.edge_embed_dim)
        self.pair_to_node = build_pair_to_node(
            pair_cfg, self.dim, self.edge_embed_dim)

        # No injection gating when gate_init is None.
        self.use_gate = gate_init is not None
        if self.use_gate:
            self.gate_prot = nn.Parameter(torch.tensor(float(gate_init)))
            self.gate_lig = nn.Parameter(torch.tensor(float(gate_init)))

        # The optional feedback gate scales only shell-to-semantic edges.
        self.use_feedback_scale = feedback_alpha_init is not None
        if self.use_feedback_scale:
            alpha = float(feedback_alpha_init)
            if alpha <= 0.0:
                logit = -10.0
            elif alpha >= 1.0:
                logit = 10.0
            else:
                logit = float(torch.log(torch.tensor(alpha / (1.0 - alpha))))
            self.feedback_logit = nn.Parameter(torch.tensor(logit))

        src, dst, grp, rel = self._build_single_template()
        self.register_buffer('_tpl_src', src, persistent=False)
        self.register_buffer('_tpl_dst', dst, persistent=False)
        self.register_buffer('_tpl_grp', grp, persistent=False)
        self.register_buffer('_tpl_rel', rel, persistent=False)

        from .pair_state import build_dense_pair, cross_pair_block_mask
        self.pair_primary = build_dense_pair(
            pair_cfg, self.dim, self.edge_embed_dim, self._n_single)
        if self.pair_primary is not None:
            self.register_buffer(
                '_pair_block_mask',
                cross_pair_block_mask(
                    self._n_single, self.n_shells,
                    self._p_idx, self._l_idx,
                    self.pair_readout_scope, src.device),
                persistent=False)

        from .fusion import _tri_mlp_head
        S, d = self.n_shells, self.dim
        self._readout_n = (
            self._flatten_n
            + int(self.interface_extra_nodes)
            + int(self.interface_concat_nodes))
        if self.readout_mode == 'shell_only':
            in_dim = S * d
            self.head = _tri_mlp_head(in_dim, head_hidden, head_mid, dropout)
        elif self.readout_mode == 'all_nodes':
            in_dim = self._n_single * d
            self.head = _tri_mlp_head(in_dim, head_hidden, head_mid, dropout)
        elif self.readout_mode == 'shell_concat_raw_lm':
            in_dim = S * d + 2 * d
            self.head = _tri_mlp_head(in_dim, head_hidden, head_mid, dropout)
        elif self.readout_mode == 'shell_bilinear_lm':
            in_dim = S * d + d
            self.head = _tri_mlp_head(in_dim, head_hidden, head_mid, dropout)
        elif self.readout_mode == 'shell_e1_head':
            self.shell_proj = nn.Linear(S * d, int(shell_proj_dim))
            in_dim = int(shell_proj_dim) + 4 * d
            self.head = _tri_mlp_head(in_dim, int(e1_head_hidden), int(e1_head_mid), float(e1_dropout))
        elif self.readout_mode == 'compact_sym_xattn':
            from .fusion import SymWideXAttnFusion
            self.sym_readout = SymWideXAttnFusion(
                d_lig=d, d_prot=d, d_plitext=S * d,
                d_attn=int(sym_readout_dim), n_heads=int(sym_readout_heads),
                attn_dropout=float(sym_attn_dropout), dropout=float(e1_dropout),
                hidden=int(e1_head_hidden), mid=int(e1_head_mid),
            )
        elif self.readout_mode == 'all_nodes_smooth_mlp':
            if self.pair_readout == 'primary' and self.pair_flatten_mode == 'off':
                self.smooth_head = None
                self.flatten_gate = None
            elif (self.pair_readout == 'primary'
                  and self.pair_flatten_mode == 'gated_shrink'):
                hidden = int(self.pair_flatten_hidden)
                self.smooth_head = nn.Sequential(
                    nn.Linear(self._readout_n * d, hidden),
                    nn.GELU(),
                    nn.Dropout(float(smooth_mlp_dropout)),
                    nn.Linear(hidden, 1),
                )
                nn.init.zeros_(self.smooth_head[-1].weight)
                nn.init.zeros_(self.smooth_head[-1].bias)
                self.flatten_gate = nn.Parameter(
                    torch.tensor(float(pair_gate_init)))
            else:
                dims = list(smooth_mlp_dims or [3072, 1536, 768, 384, 128])
                readout_in = self._readout_n * d
                extra = (int(self.geo_concat_dim) + int(self.edge_flatten_dim)
                         + int(self.jk_flatten_dim))
                if extra > 0:
                    readout_in = (
                        self.n_shells * (d + extra)
                        + (self._flatten_n - self.n_shells) * d
                        + int(self.interface_concat_nodes) * d)
                readout_in += int(self.pl_interact_dim) + int(self.diff_flatten_dim)
                if bool(self.feature_dir_cfg.get('enabled', False)):
                    from .feature_dir import FeatureDirReadout
                    self.smooth_head = FeatureDirReadout(
                        readout_in, dims,
                        dropout=float(smooth_mlp_dropout))
                else:
                    self.smooth_head = ProgressiveAllNodeMLP(
                        readout_in, dims,
                        dropout=float(smooth_mlp_dropout))
                self.flatten_gate = None
                if self.interface_extra_nodes > 0 or self.interface_concat_nodes > 0:
                    first = getattr(self.smooth_head, 'net', self.smooth_head)[0]
                    with torch.no_grad():
                        first.weight[:, -int(
                            self.interface_extra_nodes
                            + self.interface_concat_nodes) * d:] = 0.0
                extra = (int(self.geo_concat_dim) + int(self.edge_flatten_dim)
                         + int(self.jk_flatten_dim))
                if self.extra_modulate_mode != 'none':
                    if extra <= 0:
                        raise ValueError(
                            'extra_modulate requires flatten extra columns')
                    self.extra_modulate = ExtraColumnModulate(
                        self.dim, extra, self.extra_modulate_mode)
                if extra > 0:
                    first = getattr(self.smooth_head, 'net', self.smooth_head)[0]
                    with torch.no_grad():
                        for s in range(self.n_shells):
                            start = s * (d + extra) + d
                            end = (s + 1) * (d + extra)
                            first.weight[:, start:end] = 0.0
                tail = int(self.pl_interact_dim) + int(self.diff_flatten_dim)
                if tail > 0:
                    first = getattr(self.smooth_head, 'net', self.smooth_head)[0]
                    with torch.no_grad():
                        first.weight[:, -tail:] = 0.0
            self.post_graph_radial = (
                None if self.post_graph_radial_mode == 'none'
                else PostGraphRadialResidual(
                    d, self.post_graph_radial_mode, n_shells=S,
                    gate_init=float(post_graph_radial_gate_init),
                    order=self.post_graph_radial_order))
        elif self.readout_mode == 'dual_query_radial':
            if self._p_idx is None or self._l_idx is None:
                raise ValueError(
                    'dual_query_radial requires both protein and ligand nodes')
            self.dual_query_readout = DualQueryRadialReadout(
                node_dim=d,
                n_shells=S,
                n_heads=int(dual_query_heads),
                hidden_dim=int(dual_query_hidden),
                mid_dim=int(dual_query_mid),
                dropout=float(dual_query_dropout),
                shell_upper_edges_angstrom=(
                    self.shell_upper_edges_angstrom),
            )
        elif self.readout_mode == 'all_nodes_pool':
            self.all_nodes_pool_readout = AllNodePoolingReadout(
                node_dim=d,
                mode=str(pool_mode),
                hidden_dim=int(pool_hidden),
                mid_dim=int(pool_mid),
                dropout=float(pool_dropout),
                set2set_steps=int(set2set_steps),
            )
        elif self.readout_mode == 'hetero_type_pool':
            if self._p_idx is None or self._l_idx is None:
                raise ValueError(
                    'hetero_type_pool requires both protein and ligand nodes')
            self.hetero_type_readout = HeteroTypePoolingReadout(
                node_dim=d,
                hidden_dim=int(pool_hidden),
                mid_dim=int(pool_mid),
                dropout=float(pool_dropout),
            )
        elif self.readout_mode == 'radial_bin_pool':
            if self._p_idx is None or self._l_idx is None:
                raise ValueError(
                    'radial_bin_pool requires both protein and ligand nodes')
            self.radial_bin_readout = RadialBinPoolingReadout(
                node_dim=d,
                n_shells=S,
                n_bins=int(radial_pool_bins),
                hidden_dim=int(pool_hidden),
                mid_dim=int(pool_mid),
                dropout=float(pool_dropout),
            )
        elif self.readout_mode == 'ordered_lowrank':
            if self._p_idx is None or self._l_idx is None:
                raise ValueError(
                    'ordered_lowrank requires both protein and ligand nodes')
            self.ordered_lowrank_readout = OrderedLowRankReadout(
                node_dim=d,
                n_nodes=self._n_single,
                n_shells=S,
                rank=int(lowrank_rank),
                hidden_dim=int(lowrank_hidden),
                latent_dim=int(lowrank_latent),
                dropout=float(lowrank_dropout),
                use_pl_interaction=bool(lowrank_use_pl_interaction),
                use_radial_branch=bool(lowrank_use_radial_branch),
                radial_bins=int(lowrank_radial_bins),
                branch_gate_init=float(lowrank_branch_gate_init),
            )
        elif self.readout_mode == 'full_ordered_residual':
            if self._p_idx is None or self._l_idx is None:
                raise ValueError(
                    'full_ordered_residual requires protein and ligand nodes')
            if not full_residual_use_pl and not full_residual_use_radial:
                raise ValueError(
                    'full_ordered_residual requires at least one residual branch')
            self.full_ordered_residual_readout = FullOrderedResidualReadout(
                node_dim=d,
                n_nodes=self._n_single,
                n_shells=S,
                hidden_dim=int(full_residual_hidden),
                latent_dim=int(full_residual_latent),
                dropout=float(full_residual_dropout),
                use_pl_interaction=bool(full_residual_use_pl),
                use_radial_branch=bool(full_residual_use_radial),
                radial_bins=int(full_residual_radial_bins),
                branch_gate_init=float(full_residual_gate_init),
            )
        else:
            if self.edge_mode == 'topology_only':
                raise ValueError(
                    'ehign_bidirectional_edge_sum requires relation_only or '
                    'relation_attr so every edge has a learned feature')
            if self.lm_shell_edge_mode != 'bidirectional':
                raise ValueError(
                    'ehign_bidirectional_edge_sum requires bidirectional LM-Shell edges')
            self.edge_readout = EHIGNDirectionalEdgeReadout(
                node_dim=d, edge_dim=self.edge_embed_dim,
                pair_dim=int(edge_pair_dim))

        if self.use_ehign_aux:
            if self.readout_mode != 'ordered_lowrank':
                raise ValueError(
                    'use_ehign_aux is supported only with ordered_lowrank')
            if self.edge_mode == 'topology_only':
                raise ValueError(
                    'use_ehign_aux requires relation or attribute edge features')
            if self.lm_shell_edge_mode != 'bidirectional':
                raise ValueError(
                    'use_ehign_aux requires bidirectional LM-Shell edges')
            self.edge_aux_readout = EHIGNDirectionalEdgeReadout(
                node_dim=d, edge_dim=self.edge_embed_dim,
                pair_dim=int(edge_pair_dim))

        if (self.post_graph_radial_mode != 'none'
                and self.readout_mode != 'all_nodes_smooth_mlp'):
            raise ValueError(
                'post_graph_radial_mode is defined only for '
                'all_nodes_smooth_mlp')

    def _node_layout(self) -> Tuple[int, Optional[int], Optional[int], Optional[int]]:
        """Return the node count and protein, ligand, and pocket indices.

        Shells come first. Optional C-beta and atom nodes follow and are excluded from the flatten readout.
        """
        S = self.n_shells
        idx = S
        prot_idx = None
        lig_idx = None
        pocket_idx = None
        if not self.disable_prot:
            prot_idx = idx; idx += 1
        if not self.disable_lig:
            lig_idx = idx; idx += 1
        if self.add_pocket_node:
            pocket_idx = idx; idx += 1
        self._flatten_n = idx
        self._cb_start = None
        self._atom_start = None
        if self.multiscale_3d:
            self._cb_start = idx
            idx += self.max_cb
            self._atom_start = idx
            idx += self.max_atom
        return idx, prot_idx, lig_idx, pocket_idx

    def _build_single_template(self):
        S = self.n_shells
        n_single, p_idx, l_idx, k_idx = self._node_layout()
        src: List[int] = []
        dst: List[int] = []
        grp: List[int] = []
        rel: List[int] = []

        def add(s: int, d: int, g: int, r: int) -> None:
            src.append(s); dst.append(d); grp.append(g); rel.append(r)

        # The dynamic mode inserts shell-to-shell edges during the forward pass.
        if (self.shell_edge_mode == 'physical_chain'
                or self._layerwise_shell_topology):
            for i in range(S):
                for d in range(1, self.k_shell + 1):
                    if i - d >= 0:
                        add(i, i - d, 0, self.REL_SS)
                    if i + d < S:
                        add(i, i + d, 0, self.REL_SS)
        # Groups: 0 = regular; 1 and 2 = protein and ligand injection; 3 and 4 = feedback.
        if (p_idx is not None and not self._dynamic_prot_shell
                and self.similarity_topk_scope == 'shell_only'):
            for s in range(S):
                if self.lm_shell_edge_mode in ('bidirectional', 'inject_only'):
                    add(p_idx, s, 1, self.REL_P2S)
                if self.lm_shell_edge_mode in ('bidirectional', 'feedback_only'):
                    add(s, p_idx, 3, self.REL_S2P)
        if (l_idx is not None and not self._dynamic_lig_shell
                and self.similarity_topk_scope == 'shell_only'):
            for s in range(S):
                if self.lm_shell_edge_mode in ('bidirectional', 'inject_only'):
                    add(l_idx, s, 2, self.REL_L2S)
                if self.lm_shell_edge_mode in ('bidirectional', 'feedback_only'):
                    add(s, l_idx, 4, self.REL_S2L)
        if self.connect_prot_lig and p_idx is not None and l_idx is not None:
            add(p_idx, l_idx, 0, self.REL_P2L)
            add(l_idx, p_idx, 0, self.REL_L2P)
        if k_idx is not None:
            for s in range(S):
                if self.lm_shell_edge_mode in ('bidirectional', 'inject_only'):
                    add(k_idx, s, 1, self.REL_P2S)
                if self.lm_shell_edge_mode in ('bidirectional', 'feedback_only'):
                    add(s, k_idx, 3, self.REL_S2P)
            if self.connect_pocket_lig and l_idx is not None:
                add(k_idx, l_idx, 0, self.REL_P2L)
                add(l_idx, k_idx, 0, self.REL_L2P)
            if self.connect_pocket_prot and p_idx is not None:
                add(k_idx, p_idx, 0, self.REL_K2P)
                add(p_idx, k_idx, 0, self.REL_P2K)
        if self.add_self_loops:
            for i in range(n_single):
                add(i, i, 0, self.REL_SELF)

        self._n_single = n_single
        self._p_idx = p_idx
        self._l_idx = l_idx
        self._k_idx = k_idx
        return (torch.tensor(src, dtype=torch.long),
                torch.tensor(dst, dtype=torch.long),
                torch.tensor(grp, dtype=torch.long),
                torch.tensor(rel, dtype=torch.long))

    def _similarity_shell_edges(
            self, H: torch.Tensor, B: int, N1: int,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select k non-self source shells per target by cosine similarity.

        Gradients pass through selected messages, not the discrete indices.
        """
        S = self.n_shells
        if H.dim() != 2 or H.size(0) != int(B) * int(N1):
            raise ValueError(
                'similarity shell topology expects H=[B*N1,D], got '
                f'{tuple(H.shape)} for B={B}, N1={N1}')
        with torch.no_grad():
            # Disable autocast to rank cosine scores in float32.
            with torch.autocast(device_type=H.device.type, enabled=False):
                shell = H.detach().reshape(B, N1, self.dim)[:, :S]
                unit = F.normalize(shell.float(), p=2, dim=-1, eps=1e-12)
                similarity = torch.matmul(unit, unit.transpose(1, 2))
                diagonal = torch.eye(
                    S, dtype=torch.bool,
                    device=similarity.device).unsqueeze(0)
                similarity.masked_fill_(diagonal, float('-inf'))
                # Stable sorting breaks ties by the smaller source index.
                ranked_sources = torch.argsort(
                    similarity, dim=-1, descending=True, stable=True)
                top_sources = ranked_sources[:, :, :self.similarity_topk_k]

            offsets = (
                torch.arange(B, device=H.device, dtype=torch.long)
                .view(B, 1, 1) * int(N1))
            targets = (
                torch.arange(S, device=H.device, dtype=torch.long)
                .view(1, S, 1).expand(B, S, self.similarity_topk_k))
            src = (top_sources.to(dtype=torch.long) + offsets).reshape(-1)
            dst = (targets + offsets).reshape(-1)
        return src, dst

    def _attention_shell_edges(
            self, H: torch.Tensor, layer: int, B: int, N1: int,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select k non-self source shells per target by mean query/key score.

        Use the layer's input normalization, projections, and scale. Edge,
        relation, and spatial terms are applied by attention after selection.
        """
        S = self.n_shells
        if H.dim() != 2 or H.size(0) != int(B) * int(N1):
            raise ValueError(
                'attention shell topology expects H=[B*N1,D], got '
                f'{tuple(H.shape)} for B={B}, N1={N1}')
        if not 0 <= int(layer) < len(self.blocks):
            raise ValueError(
                f'attention shell topology layer out of range: {layer}')
        blk = self.blocks[int(layer)]
        attn = getattr(blk, 'attn', None)
        if not isinstance(attn, ShellFMHeteroGraphAttention) \
                or attn.scheme != 'GAT':
            raise ValueError(
                'attention shell topology requires a ShellFM GAT block, got '
                f'{type(blk).__name__}/{getattr(attn, "scheme", None)!r}')
        with torch.no_grad():
            # Compute topology scores in float32.
            with torch.autocast(device_type=H.device.type, enabled=False):
                h = H.detach().reshape(B, N1, self.dim)[:, :S].float()
                if blk.norm_style == 'pre':
                    h = blk.attn_ln(h)
                q = attn.q_proj(h).view(B, S, attn.h, attn.hd)
                k = attn.k_proj(h).view(B, S, attn.h, attn.hd)
                if attn.qk_norm:
                    q = F.normalize(q, p=2, dim=-1, eps=1e-12)
                    k = F.normalize(k, p=2, dim=-1, eps=1e-12)
                    if attn.qk_temperature == 'learned':
                        scale = attn.qk_log_scale.clamp(
                            max=math.log(100.0)).exp()
                    else:
                        scale = math.sqrt(attn.hd)
                else:
                    scale = attn.scale
                # scores[b, h, dst, src] = <Q_dst, K_src> * scale_h
                scores = torch.einsum('bqhd,bkhd->bhqk', q, k)
                if torch.is_tensor(scale):
                    scores = scores * scale.view(1, -1, 1, 1)
                else:
                    scores = scores * float(scale)
                scores = scores.mean(dim=1)                      # [B, S, S]
                diagonal = torch.eye(
                    S, dtype=torch.bool,
                    device=scores.device).unsqueeze(0)
                scores = scores.masked_fill(diagonal, float('-inf'))
                # Stable sorting breaks ties by the smaller source index.
                ranked_sources = torch.argsort(
                    scores, dim=-1, descending=True, stable=True)
                top_sources = ranked_sources[:, :, :self.similarity_topk_k]
            offsets = (
                torch.arange(B, device=H.device, dtype=torch.long)
                .view(B, 1, 1) * int(N1))
            targets = (
                torch.arange(S, device=H.device, dtype=torch.long)
                .view(1, S, 1).expand(B, S, self.similarity_topk_k))
            src = (top_sources.to(dtype=torch.long) + offsets).reshape(-1)
            dst = (targets + offsets).reshape(-1)
        return src, dst

    def _topk_shell_edges(
            self, H: torch.Tensor, layer: int, B: int, N1: int,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dispatch the per-layer dynamic shell Top-k edge construction."""
        mode = self.shell_edge_layer_modes[int(layer)]
        if mode == 'similarity_topk':
            return self._similarity_shell_edges(H, B, N1)
        if mode == 'attention_topk':
            return self._attention_shell_edges(H, layer, B, N1)
        raise ValueError(
            f'no dynamic shell topology for layer {layer} mode {mode!r}')

    def _prepend_topk_shell_edges(
            self, H: torch.Tensor, layer: int, B: int, N1: int,
            static_src: torch.Tensor, static_dst: torch.Tensor,
            static_grp: torch.Tensor, static_rel: torch.Tensor,
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Replace the dynamic relations while retaining the fixed template."""
        if self.similarity_topk_scope == 'all_relations':
            return self._prepend_similarity_relation_edges(
                H, layer, B, N1, static_src, static_dst, static_grp, static_rel)
        ss_src, ss_dst = self._topk_shell_edges(H, layer, B, N1)
        ss_grp = torch.zeros_like(ss_src)
        ss_rel = torch.full_like(ss_src, self.REL_SS)
        return (
            torch.cat([ss_src, static_src]),
            torch.cat([ss_dst, static_dst]),
            torch.cat([ss_grp, static_grp]),
            torch.cat([ss_rel, static_rel]),
        )

    def _prepend_similarity_relation_edges(
            self, H: torch.Tensor, layer: int, B: int, N1: int,
            static_src: torch.Tensor, static_dst: torch.Tensor,
            static_grp: torch.Tensor, static_rel: torch.Tensor,
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rank neighbors independently for each query over its eligible shells."""
        block = self.blocks[layer]
        ln = block.local_ln if self.graph_update_mode == 'shellgps' else block.attn_ln
        S = self.n_shells
        semantic = [
            (index, to_group, to_rel, from_group, from_rel)
            for index, to_group, to_rel, from_group, from_rel in (
                (self._p_idx, 1, self.REL_P2S, 3, self.REL_S2P),
                (self._l_idx, 2, self.REL_L2S, 4, self.REL_S2L),
            )
            if index is not None
        ]
        inject = self.lm_shell_edge_mode in ('bidirectional', 'inject_only')
        feedback = self.lm_shell_edge_mode in ('bidirectional', 'feedback_only')
        with torch.no_grad(), torch.autocast(device_type=H.device.type, enabled=False):
            normalized = F.layer_norm(
                H.detach().float(), ln.normalized_shape,
                ln.weight.float(), ln.bias.float(), ln.eps)
            ss_src, ss_dst = self._similarity_shell_edges(normalized, B, N1)
            sources = [ss_src, static_src]
            targets = [ss_dst, static_dst]
            groups = [torch.zeros_like(ss_src), static_grp]
            relations = [torch.full_like(ss_src, self.REL_SS), static_rel]
            if semantic:
                indices = [item[0] for item in semantic]
                nodes = normalized.reshape(B, N1, self.dim)
                shell = F.normalize(nodes[:, :S], p=2, dim=-1, eps=1e-12)
                query = F.normalize(nodes[:, indices], p=2, dim=-1, eps=1e-12)
                scores = query @ shell.transpose(1, 2)
                top = torch.argsort(scores, dim=-1, descending=True, stable=True)
                top = top[:, :, :self.similarity_topk_k]
                offsets = torch.arange(B, device=H.device).unsqueeze(1) * N1
                for i, (index, to_group, to_rel, from_group, from_rel) in enumerate(semantic):
                    shell_ids = (top[:, i] + offsets).reshape(-1)
                    semantic_ids = (offsets + index).expand_as(top[:, i]).reshape(-1)
                    if inject:
                        sources.append(semantic_ids)
                        targets.append(shell_ids)
                        groups.append(torch.full_like(shell_ids, to_group))
                        relations.append(torch.full_like(shell_ids, to_rel))
                    if feedback:
                        sources.append(shell_ids)
                        targets.append(semantic_ids)
                        groups.append(torch.full_like(shell_ids, from_group))
                        relations.append(torch.full_like(shell_ids, from_rel))
        return tuple(torch.cat(items) for items in (sources, targets, groups, relations))

    def _prepend_similarity_shell_edges(
            self, H: torch.Tensor, B: int, N1: int,
            static_src: torch.Tensor, static_dst: torch.Tensor,
            static_grp: torch.Tensor, static_rel: torch.Tensor,
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepend dynamic shell-to-shell edges while retaining the other edges."""
        ss_src, ss_dst = self._similarity_shell_edges(H, B, N1)
        ss_grp = torch.zeros_like(ss_src)
        ss_rel = torch.full_like(ss_src, self.REL_SS)
        return (
            torch.cat([ss_src, static_src]),
            torch.cat([ss_dst, static_dst]),
            torch.cat([ss_grp, static_grp]),
            torch.cat([ss_rel, static_rel]),
        )

    def _append_prot_shell_edges(self, freq_raw: torch.Tensor, B: int, N1: int,
                                 device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rank shells by w(s) = sum_p freq(p, s) when pruning semantic edges.

        With prot_lig scope, both semantic nodes use the same ranked shells.
        """
        assert self._p_idx is not None or self._l_idx is not None
        S = self.n_shells
        k = min(self.lm_shell_topk, S)
        w_shell = freq_raw.sum(dim=1)
        src_l: List[int] = []
        dst_l: List[int] = []
        grp_l: List[int] = []
        rel_l: List[int] = []
        inject = self.lm_shell_edge_mode in ('bidirectional', 'inject_only')
        feedback = self.lm_shell_edge_mode in ('bidirectional', 'feedback_only')
        for b in range(B):
            _, topi = torch.topk(w_shell[b], k)
            if self._dynamic_prot_shell:
                assert self._p_idx is not None
                p_off = b * N1 + self._p_idx
                for s in topi.tolist():
                    s_off = b * N1 + int(s)
                    if inject:
                        src_l.append(p_off); dst_l.append(s_off); grp_l.append(1); rel_l.append(self.REL_P2S)
                    if feedback:
                        src_l.append(s_off); dst_l.append(p_off); grp_l.append(3); rel_l.append(self.REL_S2P)
            if self._dynamic_lig_shell:
                assert self._l_idx is not None
                l_off = b * N1 + self._l_idx
                for s in topi.tolist():
                    s_off = b * N1 + int(s)
                    if inject:
                        src_l.append(l_off); dst_l.append(s_off); grp_l.append(2); rel_l.append(self.REL_L2S)
                    if feedback:
                        src_l.append(s_off); dst_l.append(l_off); grp_l.append(4); rel_l.append(self.REL_S2L)
        if not src_l:
            z = torch.zeros(0, dtype=torch.long, device=device)
            return z, z, z, z
        return (torch.tensor(src_l, dtype=torch.long, device=device),
                torch.tensor(dst_l, dtype=torch.long, device=device),
                torch.tensor(grp_l, dtype=torch.long, device=device),
                torch.tensor(rel_l, dtype=torch.long, device=device))

    def _edge_gates(self, grp: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Return the injection and feedback scale for each edge."""
        if self.use_gate:
            g1 = torch.sigmoid(self.gate_prot)
            g2 = torch.sigmoid(self.gate_lig)
        else:
            g1 = g2 = torch.ones((), device=device)
        if self.use_feedback_scale:
            gf = torch.sigmoid(self.feedback_logit)
        else:
            gf = torch.ones((), device=device)
        gate_by_grp = torch.stack([
            torch.ones((), device=device),
            g1, g2, gf, gf,
        ])
        return gate_by_grp[grp]

    def _raw_edge_attributes(self, src: torch.Tensor, dst: torch.Tensor,
                             rel: torch.Tensor, freq_raw: torch.Tensor,
                             B: int, N1: int) -> torch.Tensor:
        """Build numeric attributes for each directed edge separately from its relation embedding."""
        device = src.device
        dtype = freq_raw.dtype if freq_raw.is_floating_point() else torch.float32
        attr = torch.zeros(src.numel(), 3, device=device, dtype=dtype)
        local_src = src.remainder(N1)
        local_dst = dst.remainder(N1)
        edge_batch = dst.div(N1, rounding_mode='floor')

        # Normalize positions by the outer shell radius.
        shell_upper = self.shell_upper_edges_angstrom.to(
            device=device, dtype=dtype)
        shell_u = shell_upper / shell_upper[-1]
        ss = rel.eq(self.REL_SS)
        if ss.any() and self.edge_attr_mode != 'contact_only':
            attr[ss, 0] = shell_u[local_src[ss]]
            attr[ss, 1] = shell_u[local_dst[ss]]

        inject = rel.eq(self.REL_P2S) | rel.eq(self.REL_L2S)
        feedback = rel.eq(self.REL_S2P) | rel.eq(self.REL_S2L)
        cross = inject | feedback
        if cross.any():
            # Identify the shell endpoint for each direction of cross edge.
            shell_idx = torch.where(
                inject[cross], local_dst[cross], local_src[cross])
            if self.edge_attr_mode != 'contact_only':
                attr[cross, 0] = shell_u[shell_idx]
            if self.edge_attr_mode != 'position_only':
                c_shell = freq_raw.sum(dim=1).to(dtype=dtype)
                total = c_shell.sum(dim=1, keepdim=True)
                q_shell = c_shell / (total + self.edge_attr_eps)
                b = edge_batch[cross]
                c = c_shell[b, shell_idx]
                attr[cross, 1] = torch.log1p(c.clamp_min(0.0))
                attr[cross, 2] = q_shell[b, shell_idx]
        if self.edge_attr_features == 'basic':
            return attr
        if self.edge_attr_features == 'basic_padded':
            return torch.cat([
                attr,
                torch.zeros(
                    attr.size(0), 8, device=attr.device, dtype=attr.dtype),
            ], dim=1)

        extras = []
        src_shell = local_src.lt(self.n_shells)
        dst_shell = local_dst.lt(self.n_shells)
        c_shell = freq_raw.sum(dim=1).to(dtype=dtype)
        q_shell = c_shell / (
            c_shell.sum(dim=1, keepdim=True) + self.edge_attr_eps)

        if self.edge_attr_features in ('radial_delta', 'extended'):
            radial = torch.zeros(src.numel(), 2, device=device, dtype=dtype)
            ss = rel.eq(self.REL_SS)
            signed = (
                local_dst[ss].to(dtype) - local_src[ss].to(dtype)
            ) / max(1, self.n_shells - 1)
            radial[ss, 0] = signed
            radial[ss, 1] = signed.abs()
            extras.append(radial)

        if self.edge_attr_features in ('shell_contact', 'extended'):
            contact = torch.zeros(src.numel(), 3, device=device, dtype=dtype)
            if src_shell.any():
                b = edge_batch[src_shell]
                s = local_src[src_shell]
                contact[src_shell, 0] = torch.log1p(
                    c_shell[b, s].clamp_min(0.0))
            if dst_shell.any():
                b = edge_batch[dst_shell]
                s = local_dst[dst_shell]
                contact[dst_shell, 1] = torch.log1p(
                    c_shell[b, s].clamp_min(0.0))
            both = src_shell & dst_shell
            if both.any():
                b = edge_batch[both]
                contact[both, 2] = (
                    q_shell[b, local_dst[both]]
                    - q_shell[b, local_src[both]])
            extras.append(contact)

        if self.edge_attr_features in ('chemical_entropy', 'extended'):
            # Compute per-shell entropy for residue, element, and residue-element counts.
            counts = freq_raw.view(
                B, N_RES_TYPES, N_LIG_ELES, self.n_shells).to(dtype=dtype)

            def normalized_entropy(x: torch.Tensor, classes: int) -> torch.Tensor:
                prob = x / (x.sum(dim=1, keepdim=True) + self.edge_attr_eps)
                entropy = -(prob * torch.log(
                    prob.clamp_min(self.edge_attr_eps))).sum(dim=1)
                return entropy / math.log(classes)

            prot_entropy = normalized_entropy(counts.sum(dim=2), N_RES_TYPES)
            lig_entropy = normalized_entropy(counts.sum(dim=1), N_LIG_ELES)
            pair_entropy = normalized_entropy(
                counts.view(B, N_RES_TYPES * N_LIG_ELES, self.n_shells),
                N_RES_TYPES * N_LIG_ELES)
            chemistry = torch.zeros(
                src.numel(), 3, device=device, dtype=dtype)
            has_shell = src_shell | dst_shell
            if has_shell.any():
                b = edge_batch[has_shell]
                # Use the unique shell endpoint on cross edges and the destination otherwise.
                s = torch.where(
                    dst_shell[has_shell], local_dst[has_shell],
                    local_src[has_shell])
                chemistry[has_shell, 0] = prot_entropy[b, s]
                chemistry[has_shell, 1] = lig_entropy[b, s]
                chemistry[has_shell, 2] = pair_entropy[b, s]
            extras.append(chemistry)

        output = torch.cat([attr] + extras, dim=1)
        if output.size(1) != self.edge_attr_input_dim:
            raise RuntimeError(
                f'edge attribute width {output.size(1)} != '
                f'{self.edge_attr_input_dim}')
        return output

    def _cross_pair_mask(self, rel: torch.Tensor) -> torch.Tensor:
        if self.pair_readout_scope == 'all':
            return torch.ones_like(rel, dtype=torch.bool)
        return (
            rel.eq(self.REL_P2S) | rel.eq(self.REL_S2P)
            | rel.eq(self.REL_L2S) | rel.eq(self.REL_S2L))

    def _add_pair_residual(self, pred: torch.Tensor,
                           edge_embed: Optional[torch.Tensor],
                           rel: torch.Tensor, dst: torch.Tensor,
                           B: int, N1: int) -> torch.Tensor:
        if self.pair_residual is None or edge_embed is None:
            return pred
        edge_batch = dst.div(N1, rounding_mode='floor')
        return pred + self.pair_residual(
            edge_embed, edge_batch, self._cross_pair_mask(rel), B)

    def _edge_embeddings(self, rel: torch.Tensor, src: torch.Tensor,
                         dst: torch.Tensor, freq_raw: torch.Tensor,
                         B: int, N1: int) -> Optional[torch.Tensor]:
        if self.edge_mode == 'topology_only':
            return None
        edge_embed = self.edge_rel_emb(rel)
        if self.edge_mode == 'relation_attr':
            raw_attr = self._raw_edge_attributes(src, dst, rel, freq_raw, B, N1)
            edge_embed = edge_embed + self.edge_attr_encoder(raw_attr.to(edge_embed.dtype))
        return edge_embed

    def _inject_geo_edges(self, edge_embed: torch.Tensor, rel: torch.Tensor,
                          src: torch.Tensor, dst: torch.Tensor,
                          geo: torch.Tensor, n_single: int) -> torch.Tensor:
        """Add per-shell geometry to existing P/L ↔ shell edges only."""
        S = self.n_shells
        batch = src.div(n_single, rounding_mode='floor')
        local_src = src - batch * n_single
        local_dst = dst - batch * n_single
        cross_to = rel.eq(self.REL_P2S) | rel.eq(self.REL_L2S)
        cross_from = rel.eq(self.REL_S2P) | rel.eq(self.REL_S2L)
        cross = cross_to | cross_from
        shell_idx = torch.where(cross_to, local_dst, local_src).clamp(0, S - 1)
        add = self.geo_edge_proj(geo[batch, shell_idx])
        add = add * cross.unsqueeze(-1).to(add.dtype)
        with torch.no_grad():
            self.last_geo_edge_abs = float(add.detach().abs().mean())
        return edge_embed + add

    def _edge_radial_dist(self, src: torch.Tensor, dst: torch.Tensor,
                          n_single: int) -> torch.Tensor:
        """Onion-centered |r_src - r_dst|; P/L sit at the ligand origin."""
        S = self.n_shells
        batch = src.div(n_single, rounding_mode='floor')
        local_src = src - batch * n_single
        local_dst = dst - batch * n_single
        r_src = torch.where(
            local_src < S,
            (local_src.to(dtype=torch.float32) + 1.0) / float(S),
            torch.zeros_like(local_src, dtype=torch.float32))
        r_dst = torch.where(
            local_dst < S,
            (local_dst.to(dtype=torch.float32) + 1.0) / float(S),
            torch.zeros_like(local_dst, dtype=torch.float32))
        return (r_src - r_dst).abs().unsqueeze(-1)

    def _edge_flatten_update_extra(
            self, src: torch.Tensor, dst: torch.Tensor,
            n_single: int, batch_size: int,
            geo_channels: Optional[torch.Tensor],
            dtype: torch.dtype) -> Optional[torch.Tensor]:
        mode = self.edge_flatten_update_extra
        if mode == 'none' or self.edge_flatten_extra_dim <= 0:
            return None
        if mode in ('radial', 'rbf'):
            dist = self._edge_radial_dist(src, dst, n_single).to(dtype=dtype)
            if mode == 'radial':
                return dist
            centers = self.edge_rbf_centers.to(device=dist.device, dtype=dtype)
            width = dist.new_tensor(float(self.edge_rbf_width))
            return torch.exp(-((dist - centers) ** 2) / (width * width))
        batch = src.div(n_single, rounding_mode='floor')
        local_src = src - batch * n_single
        local_dst = dst - batch * n_single
        S = self.n_shells
        shell = torch.where(local_dst < S, local_dst, local_src)
        touch = (local_src < S) | (local_dst < S)
        if geo_channels is None:
            return src.new_zeros(
                src.size(0), self.edge_flatten_extra_dim, dtype=dtype)
        extra = geo_channels[batch.clamp(max=int(batch_size) - 1),
                             shell.clamp(0, S - 1)]
        return extra * touch.unsqueeze(-1).to(extra.dtype)

    def _jk_shell_feature(self, layer_states, B: int, N1: int):
        """JK-Net extras from H0..H(L-1); last layer already sits in flatten."""
        shallow = layer_states[:-1]
        shells = [h.view(B, N1, self.dim)[:, :self.n_shells] for h in shallow]
        if self.jk_flatten == 'first':
            feat = shells[0]
        elif self.jk_flatten == 'max':
            feat = torch.stack(shells, dim=0).max(dim=0).values
        elif self.jk_flatten == 'concat':
            feat = torch.cat(shells, dim=-1)
        else:
            raise RuntimeError(f'unexpected jk_flatten {self.jk_flatten!r}')
        return self.jk_proj(feat)

    def _appnp_propagate(self, seed, src, dst):
        """APPNP: Z = (1-α) Â Z + α seed, Â = random-walk with self-loops."""
        n = seed.size(0)
        loop = torch.arange(n, device=seed.device)
        s = torch.cat([src, loop])
        d = torch.cat([dst, loop])
        deg = seed.new_zeros(n)
        deg.scatter_add_(0, d, seed.new_ones(d.size(0)))
        deg = deg.clamp(min=1.0).unsqueeze(-1)
        alpha = seed.new_tensor(self.diff_flatten_alpha)
        z = seed
        dim = seed.size(-1)
        for _ in range(self.diff_flatten_steps):
            agg = seed.new_zeros(seed.shape)
            agg.scatter_add_(0, d.unsqueeze(-1).expand(-1, dim), z[s])
            z = (1.0 - alpha) * (agg / deg) + alpha * seed
        return z

    def _diff_flatten_feature(self, H, src, dst, B, N1, h0):
        seed = h0 if self.diff_flatten == 'appnp_h0' else H
        z = self._appnp_propagate(seed, src, dst).view(B, N1, self.dim)
        if self.diff_flatten == 'appnp_lig':
            if self._l_idx is None:
                raise ValueError('appnp_lig requires a ligand node')
            feat = z[:, self._l_idx]
        else:
            feat = z.mean(dim=1)
        return self.diff_proj(feat)

    def _pl_interact_feature(self, prot: torch.Tensor,
                             lig: torch.Tensor) -> torch.Tensor:
        """Protein–ligand interaction vector concatenated onto flatten."""
        if self.pl_interact == 'hadamard':
            return prot * lig
        if self.pl_interact == 'hadamard32':
            return self.pl_hadamard_proj(prot * lig)
        if self.pl_interact == 'bilinear32':
            return self.pl_bilin_p(prot) * self.pl_bilin_l(lig)
        raise RuntimeError(f'unexpected pl_interact {self.pl_interact!r}')

    def _edge_flatten_incident(
            self, src: torch.Tensor, dst: torch.Tensor,
            rel: Optional[torch.Tensor], n_single: int,
            scope: Optional[str] = None,
            direction: str = 'both',
            ) -> tuple:
        """Incident (batch, shell) masks; optionally keep only P/L ↔ shell."""
        S = self.n_shells
        batch = src.div(n_single, rounding_mode='floor')
        local_src = src - batch * n_single
        local_dst = dst - batch * n_single
        src_shell = local_src < S
        dst_shell = local_dst < S
        scope = str(scope or self.edge_flatten_scope).lower()
        if scope == 'cross':
            if rel is None:
                raise ValueError(
                    'edge_flatten_scope=cross requires relation ids')
            cross = (
                rel.eq(self.REL_P2S) | rel.eq(self.REL_S2P)
                | rel.eq(self.REL_L2S) | rel.eq(self.REL_S2L))
            src_shell = src_shell & cross
            dst_shell = dst_shell & cross
        elif scope == 'ss':
            if rel is None:
                raise ValueError('scope=ss requires relation ids')
            ss = rel.eq(self.REL_SS)
            src_shell = src_shell & ss
            dst_shell = dst_shell & ss
        if direction == 'incoming':
            src_shell = src_shell & False
        elif direction == 'outgoing':
            dst_shell = dst_shell & False
        elif direction != 'both':
            raise ValueError(
                'direction must be both|incoming|outgoing, got '
                f'{direction!r}')
        return batch, local_src, local_dst, src_shell, dst_shell

    def _pool_skip_shell_edges(
            self, edge_embed, edge_embed_e0, src, dst, n_single,
            batch_size, rel):
        """Pool updated edge embeddings, optionally mixing in the initial embeddings."""
        pooled = self._pool_shell_edges(
            edge_embed, src, dst, n_single, batch_size, rel)
        if self.edge_flatten_skip == 'none' or edge_embed_e0 is None:
            return self._mix_edge_branch(
                pooled, edge_embed, src, dst, n_single, batch_size, rel)
        pooled0 = self._pool_shell_edges(
            edge_embed_e0, src, dst, n_single, batch_size, rel)
        if self.edge_flatten_skip == 'e0':
            pooled = pooled + self.edge_flatten_skip_alpha * pooled0
        elif self.edge_flatten_skip == 'gate':
            mix = torch.sigmoid(self.edge_flatten_skip_gate)
            pooled = (1.0 - mix) * pooled + mix * pooled0
        else:
            pooled = self.edge_flatten_skip_proj(
                torch.cat([pooled, pooled0], dim=-1))
        return self._mix_edge_branch(
            pooled, edge_embed, src, dst, n_single, batch_size, rel)

    def _mix_edge_branch(
            self, pooled, edge_embed, src, dst, n_single, batch_size, rel):
        if self.edge_flatten_branch == 'none':
            return pooled
        if self.edge_flatten_branch == 'incoming':
            extra = self._pool_shell_edges(
                edge_embed, src, dst, n_single, batch_size, rel,
                direction='incoming')
        else:
            extra = self._pool_shell_edges(
                edge_embed, src, dst, n_single, batch_size, rel,
                scope=self.edge_flatten_branch)
        return self.edge_flatten_branch_proj(
            torch.cat([pooled, extra], dim=-1))

    def _pool_shell_edges(self, edge_embed: torch.Tensor, src: torch.Tensor,
                          dst: torch.Tensor, n_single: int,
                          batch_size: int,
                          rel: Optional[torch.Tensor] = None,
                          scope: Optional[str] = None,
                          direction: str = 'both') -> torch.Tensor:
        """Pool incident edge embeddings per shell.

        With cross scope, keep only protein/ligand-shell edges. A zero attention score starts as mean pooling.
        """
        S = self.n_shells
        dim = edge_embed.size(-1)
        B = int(batch_size)
        batch, local_src, local_dst, src_shell, dst_shell = (
            self._edge_flatten_incident(
                src, dst, rel, n_single, scope=scope, direction=direction))
        parts_b, parts_s, parts_e = [], [], []
        if bool(src_shell.any()):
            parts_b.append(batch[src_shell])
            parts_s.append(local_src[src_shell])
            parts_e.append(edge_embed[src_shell])
        if bool(dst_shell.any()):
            parts_b.append(batch[dst_shell])
            parts_s.append(local_dst[dst_shell])
            parts_e.append(edge_embed[dst_shell])
        pooled = edge_embed.new_zeros(B, S, dim)
        if not parts_b:
            self.last_edge_pool_abs = 0.0
            return pooled
        b = torch.cat(parts_b)
        s = torch.cat(parts_s)
        e = torch.cat(parts_e)
        if self.edge_flatten_attn and self.edge_flatten_score is not None:
            scores = self.edge_flatten_score(e).squeeze(-1)
            key = (b * S + s).long()
            nkey = B * S
            max_sc = scores.new_full((nkey,), float('-inf'))
            max_sc.scatter_reduce_(
                0, key, scores, reduce='amax', include_self=True)
            shifted = (scores - max_sc[key]).exp()
            den = scores.new_zeros(nkey)
            den.scatter_add_(0, key, shifted)
            w = shifted / den[key].clamp(min=1e-12)
            pooled.index_put_((b, s), e * w.unsqueeze(-1), accumulate=True)
        else:
            count = edge_embed.new_zeros(B, S, 1)
            pooled.index_put_((b, s), e, accumulate=True)
            count.index_put_(
                (b, s),
                torch.ones(s.size(0), 1, device=s.device,
                           dtype=edge_embed.dtype),
                accumulate=True)
            pooled = pooled / count.clamp(min=1.0)
        with torch.no_grad():
            self.last_edge_pool_abs = float(pooled.detach().abs().mean())
        return pooled

    def _layer_edge_mask(self, rel: torch.Tensor, layer: int) -> torch.Tensor:
        """Return the relation mask for one graph layer."""
        if self.relation_schedule == 'all' or self.n_layers <= 1:
            mask = torch.ones_like(rel, dtype=torch.bool)
        else:
            structural = (
                rel.eq(self.REL_SS) | rel.eq(self.REL_SELF)
                | rel.eq(self.REL_P2L) | rel.eq(self.REL_L2P)
                | rel.eq(self.REL_K2P) | rel.eq(self.REL_P2K))
            if self.multiscale_3d:
                structural = structural | rel.ge(self.N_RELATIONS)
            inject = rel.eq(self.REL_P2S) | rel.eq(self.REL_L2S)
            feedback = rel.eq(self.REL_S2P) | rel.eq(self.REL_S2L)
            if self.relation_schedule == 'inject_ss_feedback':
                if layer == 0:
                    mask = structural | inject
                elif layer == self.n_layers - 1:
                    mask = structural | feedback
                else:
                    mask = structural
            elif self.relation_schedule == 'inject_all_feedback':
                if layer == 0:
                    mask = structural | inject
                elif layer == self.n_layers - 1:
                    mask = structural | feedback
                else:
                    mask = structural | inject | feedback
            else:
                mask = structural | (inject if layer % 2 == 0 else feedback)

        if self.training and self.edge_dropout > 0.0:
            if self.edge_dropout_scope == 'cross':
                eligible = (
                    rel.eq(self.REL_P2S) | rel.eq(self.REL_S2P)
                    | rel.eq(self.REL_L2S) | rel.eq(self.REL_S2L))
            elif self.edge_dropout_scope == 'shell':
                eligible = rel.eq(self.REL_SS)
            else:
                eligible = ~rel.eq(self.REL_SELF)
            keep = torch.rand(rel.size(0), device=rel.device).ge(
                self.edge_dropout)
            mask = mask & (~eligible | keep)
        return mask

    def pocket_cache(self):
        """Lazily open the pocket cache (shared in-memory store across models)."""
        if self._pocket_cache is None:
            from .pocket_attention import open_pocket_cache
            cfg = (
                self.pocket_node_cfg if self.add_pocket_node
                else self.pocket_attention_cfg)
            opened = dict(cfg)
            opened['enabled'] = True
            self._pocket_cache = open_pocket_cache(opened)
            if self._pocket_cache is None:
                raise RuntimeError(
                    'pocket cache is required but could not be opened; '
                    'check fusion.pocket_node.cache_path or '
                    'fusion.pocket_attention.cache_path')
        return self._pocket_cache

    def ligand_cache(self):
        """Lazily open the ligand token cache (shared in-memory store)."""
        if self._ligand_cache is None:
            from .ligand_attention import open_ligand_token_cache
            self._ligand_cache = open_ligand_token_cache(
                self.ligand_attention_cfg)
            if self._ligand_cache is None:
                raise RuntimeError(
                    'ligand_attention is enabled but the cache could not be '
                    'opened; check fusion.ligand_attention.cache_path')
        return self._ligand_cache

    def coarse_cache(self):
        """Lazily open the Cβ + heavy-atom coordinate cache."""
        if self._coarse_cache is None:
            from data.coarse_interface_cache import CoarseInterfaceCache
            cfg = (
                self.contact_primary_cfg if self.contact_primary is not None
                else self.multiscale_cfg if self.multiscale_3d
                else self.coarse_interface_cfg if self.coarse_interface is not None
                else self.shell_to_3d_cfg if self.shell_to_3d is not None
                else self.shell_geo_readout_cfg if self.shell_geo_readout is not None
                else self.interface_tokens_cfg if self.interface_tokens is not None
                else self.side_tower_cfg if self.side_tower is not None
                else self.shell_geometry_cfg)
            path = cfg.get(
                'cache_path',
                'data_processed_esm/coarse_interface/cb_heavy.h5')
            if not os.path.isabs(path):
                root = os.path.abspath(os.path.join(
                    os.path.dirname(__file__), os.pardir))
                path = os.path.join(root, path)
            self._coarse_cache = CoarseInterfaceCache(
                path,
                max_prot=int(cfg.get('max_cb', cfg.get('max_prot', 64))),
                max_lig=int(cfg.get('max_atom', cfg.get('max_lig', 48))),
                in_memory=bool(cfg.get('cache_in_memory', True)))
        return self._coarse_cache

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                shell_embed: torch.Tensor, freq_raw: torch.Tensor,
                seqs: List[str],
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None,
                pdb_codes: Optional[List[str]] = None) -> torch.Tensor:
        assert shell_embed.dim() == 3, 'F09-node 需要 shell 预block 嵌入 [B,60,d]'
        # Masked mean pooling reduces sequence inputs [B, L, d] to one node per molecule.
        from .fusion import _to_vec
        lig = _to_vec(lig, lig_mask)
        prot = _to_vec(prot, prot_mask)
        assert lig.dim() == 2 and prot.dim() == 2, \
            f'F09-node 池化后应为 [B,d], got lig={tuple(lig.shape)} prot={tuple(prot.shape)}'
        if self.contact_primary is not None:
            if pdb_codes is None:
                raise ValueError('contact_primary requires pdb_codes')
            iface = self.coarse_cache().lookup_batch(
                list(pdb_codes), device=shell_embed.device)
            return self.contact_primary(
                iface, dtype=shell_embed.dtype, freq_raw=freq_raw,
                prot=prot, lig=lig)
        lig_raw, prot_raw = lig, prot
        device = shell_embed.device
        B = shell_embed.size(0)
        S = self.n_shells
        N1 = self._n_single

        if self.randomize_lm:
            prot = torch.randn_like(prot)
            lig = torch.randn_like(lig)
            lig_raw, prot_raw = lig, prot

        if self.pocket_attn is not None and pdb_codes is not None:
            pocket, pmask = self.pocket_cache().lookup_batch(
                list(pdb_codes), device=device,
                max_res=self.pocket_attn.max_res)
            pocket = pocket.to(shell_embed.dtype)
            if self.pocket_attn.query_source == 'shell':
                # Retrieve pocket context for each radial band without replacing the protein node.
                shell_embed = shell_embed + self.pocket_attn(
                    shell_embed, pocket, pmask)
            else:
                prot_pocket = self.pocket_attn(lig, pocket, pmask)
                if self.pocket_gate is not None:
                    lengths = torch.tensor(
                        [len(str(s or '').replace(' ', '')) for s in seqs],
                        dtype=prot.dtype, device=prot.device)
                    n_pocket = pmask.sum(dim=1).to(prot.dtype)
                    g = self.pocket_gate(lengths, n_pocket).to(prot.dtype)
                    prot = (1.0 - g) * prot + g * prot_pocket
                elif self.pocket_attn.mode == 'residual':
                    prot = prot + prot_pocket
                else:
                    prot = prot_pocket

        if self.ligand_attn is not None and pdb_codes is not None:
            tokens, tmask = self.ligand_cache().lookup_batch(
                list(pdb_codes), device=device,
                max_res=self.ligand_attn.max_res)
            tokens = tokens.to(shell_embed.dtype)
            if self.ligand_attn.query_source == 'shell':
                shell_embed = shell_embed + self.ligand_attn(
                    shell_embed, tokens, tmask)
            elif self.ligand_attn.mode == 'residual':
                lig = lig + self.ligand_attn(prot, tokens, tmask)
            else:
                lig = self.ligand_attn(prot, tokens, tmask)

        # Keep the unnormalized projections for readout modes that use them.
        if self.signed_shell is not None:
            shell_embed = self.signed_shell(shell_embed, freq_raw)

        interface_extra = None
        interface_edge_pred = None
        geo_channels = None
        iface = None
        if (self.coarse_interface is not None
                or self.shell_geometry is not None
                or self.shell_to_3d is not None
                or self.shell_geo_readout is not None
                or self.interface_tokens is not None
                or self.side_tower is not None):
            if pdb_codes is None:
                raise ValueError(
                    'shell geometry / shell_to_3d / geo readout / '
                    'interface_tokens / side_tower / '
                    'coarse_interface requires pdb_codes')
            iface = self.coarse_cache().lookup_batch(
                list(pdb_codes), device=device)
            if self.shell_geometry is not None:
                shell_embed = self.shell_geometry(iface, shell_embed)
            if self.shell_to_3d is not None:
                shell_embed = self.shell_to_3d(iface, shell_embed)
            if self.shell_geo_readout is not None:
                geo_channels = self.shell_geo_readout(iface, shell_embed.dtype)
            if self.coarse_interface is not None:
                shell_embed, interface_extra, interface_edge_pred = (
                    self.coarse_interface(iface, shell_embed))

        lig_graph = self.lig_node_norm(lig)
        prot_graph = self.prot_node_norm(prot)
        pocket_graph = None
        if self.add_pocket_node:
            if pdb_codes is None:
                raise ValueError('add_pocket_node requires pdb_codes')
            pocket, pmask = self.pocket_cache().lookup_batch(
                list(pdb_codes), device=device,
                max_res=int(self.pocket_node_cfg.get('max_res', 128)))
            pocket = pocket.to(shell_embed.dtype)
            pocket_graph = self.pocket_pool(pocket, pmask, prot=prot_graph)
            pocket_graph = self.prot_node_norm(pocket_graph)

        # Flatten the batch's node blocks in shell, protein, ligand, and optional pocket order.
        feats = [shell_embed]
        types = [torch.zeros(S, dtype=torch.long, device=device)]
        if self._p_idx is not None:
            feats.append(prot_graph.unsqueeze(1))
            types.append(torch.ones(1, dtype=torch.long, device=device))
        if self._l_idx is not None:
            feats.append(lig_graph.unsqueeze(1))
            types.append(torch.full((1,), 2, dtype=torch.long, device=device))
        if self._k_idx is not None:
            assert pocket_graph is not None
            feats.append(pocket_graph.unsqueeze(1))
            types.append(torch.full((1,), 3, dtype=torch.long, device=device))
        geo_batch = None
        if self.multiscale_3d:
            if pdb_codes is None:
                raise ValueError('multiscale_3d requires pdb_codes')
            geo_batch = self.coarse_cache().lookup_batch(
                list(pdb_codes), device=device)
            cb_h, atom_h = self.geo_embed(geo_batch, shell_embed.dtype)
            feats.append(cb_h)
            feats.append(atom_h)
            types.append(torch.full(
                (self.max_cb,), 3, dtype=torch.long, device=device))
            types.append(torch.full(
                (self.max_atom,), 4, dtype=torch.long, device=device))
        H = torch.cat(feats, dim=1).reshape(B * N1, self.dim)
        ntype = torch.cat(types, dim=0).repeat(B)
        if self.type_emb is not None:
            H = H + self.type_emb(ntype)
        h0_for_diff = H if self.diff_flatten == 'appnp_h0' else None
        h0_skip = H if self.init_residual != 'none' else None

        base = (torch.arange(B, device=device) * N1).unsqueeze(1)
        src = (self._tpl_src.unsqueeze(0) + base).reshape(-1)
        dst = (self._tpl_dst.unsqueeze(0) + base).reshape(-1)
        grp = self._tpl_grp.repeat(B)
        rel = self._tpl_rel.repeat(B)
        if self._dynamic_prot_shell or self._dynamic_lig_shell:
            ds, dd, dg, dr = self._append_prot_shell_edges(freq_raw, B, N1, device)
            src = torch.cat([src, ds])
            dst = torch.cat([dst, dd])
            grp = torch.cat([grp, dg])
            rel = torch.cat([rel, dr])
        if self.multiscale_3d:
            from .multiscale_3d import build_geo_edges
            gs, gd, gg, gr, gstat = build_geo_edges(
                geo_batch,
                n_shells=S, n_single=N1,
                cb_start=int(self._cb_start), atom_start=int(self._atom_start),
                prot_idx=int(self._p_idx if self._p_idx is not None else 0),
                lig_idx=int(self._l_idx if self._l_idx is not None else 0),
                hier=self.geo_hier, intra=self.geo_intra,
                inter=self.geo_inter, lm_broadcast=self.geo_lm,
                rel_cb2s=self.REL_CB2S, rel_s2cb=self.REL_S2CB,
                rel_a2s=self.REL_A2S, rel_s2a=self.REL_S2A,
                rel_cb2cb=self.REL_CB2CB, rel_a2a=self.REL_A2A,
                rel_cb2a=self.REL_CB2A, rel_a2cb=self.REL_A2CB,
                rel_p2cb=self.REL_P2CB, rel_cb2p=self.REL_CB2P,
                rel_l2a=self.REL_L2A, rel_a2l=self.REL_A2L)
            self.last_n_cb = gstat["n_cb"]
            self.last_n_atom = gstat["n_atom"]
            self.last_n_hier = gstat["n_hier"]
            self.last_n_pl = gstat["n_pl"]
            if gs.numel() > 0:
                src = torch.cat([src, gs])
                dst = torch.cat([dst, gd])
                grp = torch.cat([grp, gg])
                rel = torch.cat([rel, gr])
        # Retain a fixed template so a layer cannot accumulate dynamic edges.
        static_topology = (
            (src, dst, grp, rel)
            if (self._global_topk_mode is not None
                and not self._shell_edge_layer_modes_explicit) else None)
        reused_similarity_topology = None
        if (static_topology is not None
                and self.similarity_topk_update == 'first_layer_reuse'):
            reused_similarity_topology = (
                self._prepend_topk_shell_edges(
                    H, 0, B, N1, *static_topology))
        # A hybrid schedule replaces only shell edges in its dynamic layer.
        physical_topology = None
        non_ss_topology = None
        if self._layerwise_shell_topology:
            physical_topology = (src, dst, grp, rel)
            keep_non_ss = ~rel.eq(self.REL_SS)
            non_ss_topology = tuple(
                item[keep_non_ss] for item in physical_topology)
        edge_index = torch.stack([src, dst], dim=0)
        edge_gate = self._edge_gates(grp, device)
        edge_bias = torch.zeros(edge_index.size(1), device=device)
        edge_embed = None
        edge_embed_e0 = None
        if static_topology is None and not self._layerwise_shell_topology:
            edge_embed = self._edge_embeddings(
                rel, src, dst, freq_raw, B, N1)
            edge_embed_e0 = (
                edge_embed if self.edge_flatten_skip != 'none' else None)
            if (self.geo_edge_proj is not None
                    and geo_channels is not None
                    and edge_embed is not None):
                edge_embed = self._inject_geo_edges(
                    edge_embed, rel, src, dst, geo_channels, N1)

        layer_states = (
            [H] if (self.layer_mix_mode != 'none' or self.jk_flatten_dim > 0)
            else None)
        spatial_dist = None
        if (static_topology is None
                and not self._layerwise_shell_topology
                and self.spatial_bias != 'none'):
            spatial_dist = self._edge_radial_dist(src, dst, N1)
        for layer, blk in enumerate(self.blocks):
            layer_topology = None
            if self._layerwise_shell_topology:
                if self.shell_edge_layer_modes[layer] in (
                        'similarity_topk', 'attention_topk'):
                    assert non_ss_topology is not None
                    layer_topology = self._prepend_topk_shell_edges(
                        H, layer, B, N1, *non_ss_topology)
                else:
                    assert physical_topology is not None
                    layer_topology = physical_topology
            elif static_topology is not None:
                if reused_similarity_topology is not None:
                    layer_topology = reused_similarity_topology
                else:
                    layer_topology = self._prepend_topk_shell_edges(
                        H, layer, B, N1, *static_topology)
            if layer_topology is not None:
                src, dst, grp, rel = layer_topology
                edge_index = torch.stack([src, dst], dim=0)
                edge_gate = self._edge_gates(grp, device)
                edge_bias = torch.zeros(edge_index.size(1), device=device)
                edge_embed = self._edge_embeddings(
                    rel, src, dst, freq_raw, B, N1)
                edge_embed_e0 = (
                    edge_embed if self.edge_flatten_skip != 'none' else None)
                if (self.geo_edge_proj is not None
                        and geo_channels is not None
                        and edge_embed is not None):
                    edge_embed = self._inject_geo_edges(
                        edge_embed, rel, src, dst, geo_channels, N1)
                spatial_dist = None
                if self.spatial_bias != 'none':
                    spatial_dist = self._edge_radial_dist(src, dst, N1)
            layer_mask = self._layer_edge_mask(rel, layer)
            layer_edge_index = edge_index[:, layer_mask]
            layer_rel = rel[layer_mask]
            layer_edge_embed = (
                edge_embed[layer_mask] if edge_embed is not None else None)
            layer_edge_gate = edge_gate[layer_mask]
            layer_edge_bias = edge_bias[layer_mask]
            layer_spatial = None
            if self.spatial_bias != 'none':
                if spatial_dist is None:
                    raise RuntimeError('spatial_bias requires precomputed dist')
                layer_spatial = spatial_dist[layer_mask]
            if self.graph_update_mode == 'shellfm':
                H = blk(
                    H, layer_edge_index, layer_edge_embed, rel=layer_rel,
                    ntype=ntype, spatial_dist=layer_spatial)
            elif self.graph_update_mode == 'shellgps':
                H = blk(
                    H, layer_edge_index, layer_edge_embed, B, N1)
            elif self.graph_update_mode == 'hgt':
                H = blk(
                    H, layer_edge_index, layer_rel, ntype,
                    layer_edge_embed)
            else:
                H = blk(
                    H, layer_edge_index, layer_edge_gate, layer_edge_bias,
                    layer_edge_embed)
            if self.init_residual == 'every' and h0_skip is not None:
                H = H + self.init_residual_alpha * h0_skip
            if layer_states is not None:
                layer_states.append(H)
            if self.edge_updaters is not None and edge_embed is not None:
                edge_embed = self.edge_updaters[layer](H, src, dst, edge_embed)

        if self.layer_mix_mode != 'none' and layer_states is not None:
            # Add a gated mixture of earlier states to the final graph state.
            shallow = torch.stack(layer_states[:-1], dim=0)
            if self.layer_mix_mode == 'global':
                weights = torch.softmax(self.layer_mix_logits, dim=0)
                residual = (weights.view(-1, 1, 1) * shallow).sum(dim=0)
                H = H + torch.sigmoid(self.layer_mix_gate) * residual
            else:
                weights = torch.softmax(self.layer_mix_logits, dim=1)
                residual_by_type = torch.stack([
                    (weights[t].view(-1, 1, 1) * shallow).sum(dim=0)
                    for t in range(3)
                ], dim=0)
                node_ids = torch.arange(H.size(0), device=device)
                residual = residual_by_type[ntype, node_ids]
                gates = torch.sigmoid(self.layer_mix_gate)[ntype].unsqueeze(-1)
                H = H + gates * residual

        if self.pair_to_node is not None and edge_embed is not None:
            H = self.pair_to_node(H, edge_embed, dst)
        if (self.edge_flatten_updater is not None
                and edge_embed is not None):
            extra = self._edge_flatten_update_extra(
                src, dst, N1, B, geo_channels, edge_embed.dtype)
            updated = self.edge_flatten_updater(
                H, src, dst, edge_embed, extra)
            if self.edge_flatten_update_scope == 'cross':
                cross = (
                    rel.eq(self.REL_P2S) | rel.eq(self.REL_S2P)
                    | rel.eq(self.REL_L2S) | rel.eq(self.REL_S2L)
                ).unsqueeze(-1).to(updated.dtype)
                edge_embed = edge_embed * (1.0 - cross) + updated * cross
            else:
                edge_embed = updated

        if self.init_residual in ('last', 'shells') and h0_skip is not None:
            add = self.init_residual_alpha * h0_skip
            if self.init_residual == 'shells':
                out = H.clone()
                view = out.view(B, N1, self.dim)
                view[:, :self.n_shells] = (
                    view[:, :self.n_shells]
                    + add.view(B, N1, self.dim)[:, :self.n_shells])
                H = out
            else:
                H = H + add

        if self.post_graph_glu is not None:
            H = self.post_graph_glu(H)

        batch_off = torch.arange(B, device=device) * N1
        if self.readout_mode == 'ehign_bidirectional_edge_sum':
            assert edge_embed is not None
            edge_batch = dst.div(N1, rounding_mode='floor')
            to_shell = rel.eq(self.REL_P2S) | rel.eq(self.REL_L2S)
            from_shell = rel.eq(self.REL_S2P) | rel.eq(self.REL_S2L)
            pred, pred_to_shell, pred_from_shell = self.edge_readout(
                H, edge_index, edge_embed, edge_batch,
                to_shell, from_shell, B)
            return {
                'prediction': pred,
                'directional_predictions': [pred_to_shell, pred_from_shell],
                'loss_mode': 'ehign_bidirectional',
            }

        if self.readout_mode == 'all_nodes_smooth_mlp':
            if self.pair_primary is not None:
                pred = self.pair_primary(
                    H, edge_embed, src, dst, self._pair_block_mask, B)
                if self.smooth_head is not None:
                    all_idx = batch_off.unsqueeze(1) + torch.arange(
                        N1, device=device).unsqueeze(0)
                    all_h = H[all_idx.reshape(-1)].reshape(B, N1, self.dim)
                    all_flat = all_h.reshape(B, N1 * self.dim)
                    residual = self.smooth_head(all_flat)
                    if self.flatten_gate is not None:
                        residual = torch.sigmoid(self.flatten_gate) * residual
                    pred = pred + residual
                if self.output_affine:
                    pred = self.pred_scale * pred + self.pred_bias
                return pred
            flatten_n = int(getattr(self, '_flatten_n', N1))
            all_idx = batch_off.unsqueeze(1) + torch.arange(
                flatten_n, device=device).unsqueeze(0)
            all_h = H[all_idx.reshape(-1)].reshape(B, flatten_n, self.dim)
            if self.post_graph_radial is not None:
                shell_h = self.post_graph_radial(all_h[:, :S])
                all_h = torch.cat([shell_h, all_h[:, S:]], dim=1)
            if self.training and self.readout_node_dropout > 0.0:
                keep = torch.empty(
                    B, flatten_n, 1, device=all_h.device,
                    dtype=all_h.dtype).bernoulli_(
                        1.0 - self.readout_node_dropout)
                all_h = all_h * keep / (1.0 - self.readout_node_dropout)
            if self.readout_feature_dropout > 0.0:
                all_h = F.dropout(
                    all_h, p=self.readout_feature_dropout,
                    training=self.training)
            if interface_extra is not None:
                all_h = torch.cat([all_h, interface_extra], dim=1)
            if self.geo_film is not None and geo_channels is not None:
                filmed = self.geo_film(all_h[:, :S], geo_channels)
                all_h = torch.cat([filmed, all_h[:, S:]], dim=1)
            if self.interface_tokens is not None:
                if iface is None:
                    if pdb_codes is None:
                        raise ValueError('interface_tokens requires pdb_codes')
                    iface = self.coarse_cache().lookup_batch(
                        list(pdb_codes), device=device)
                if self._p_idx is None or self._l_idx is None:
                    raise ValueError(
                        'interface_tokens requires protein and ligand nodes')
                p_tok, l_tok = self.interface_tokens(
                    iface, all_h.dtype,
                    prot_h=all_h[:, self._p_idx],
                    lig_h=all_h[:, self._l_idx])
                tok_mode = self.interface_tokens.mode
                if tok_mode == 'concat':
                    all_h = torch.cat(
                        [all_h, p_tok.unsqueeze(1), l_tok.unsqueeze(1)], dim=1)
                elif tok_mode == 'concat_one':
                    all_h = torch.cat(
                        [all_h, ((p_tok + l_tok) * 0.5).unsqueeze(1)], dim=1)
                else:
                    all_h = all_h.clone()
                    if tok_mode == 'replace':
                        all_h[:, self._p_idx] = p_tok
                        all_h[:, self._l_idx] = l_tok
                    else:
                        all_h[:, self._p_idx] = all_h[:, self._p_idx] + p_tok
                        all_h[:, self._l_idx] = all_h[:, self._l_idx] + l_tok
            if (self.geo_residual_proj is not None
                    and geo_channels is not None):
                add = self.geo_residual_proj(geo_channels)
                all_h = all_h.clone()
                all_h[:, :S] = all_h[:, :S] + add
                with torch.no_grad():
                    self.last_geo_col_abs = float(
                        self.geo_residual_proj.weight.detach().abs().mean())
            geo_for_cat = geo_channels
            if self.geo_stem is not None and geo_channels is not None:
                geo_for_cat = self.geo_stem(geo_channels)
            hidden0_add = None
            if (self.geo_factor_A is not None
                    and geo_channels is not None):
                code = geo_channels @ self.geo_factor_B.to(
                    geo_channels.dtype).t()
                hidden0_add = torch.einsum(
                    'sor,bsr->bo',
                    self.geo_factor_A.to(geo_channels.dtype), code)
                with torch.no_grad():
                    recon = torch.einsum(
                        'sor,re->soe',
                        self.geo_factor_A.detach(),
                        self.geo_factor_B.detach())
                    self.last_geo_col_abs = float(recon.abs().mean())
            extras = []
            if geo_for_cat is not None and self.geo_concat_flatten:
                extras.append(geo_for_cat)
            if (self.edge_flatten_pool and edge_embed is not None):
                extras.append(self._pool_skip_shell_edges(
                    edge_embed, edge_embed_e0, src, dst, N1, B, rel))
            if self.jk_flatten_dim > 0:
                if layer_states is None:
                    raise RuntimeError('jk_flatten requires layer_states')
                extras.append(self._jk_shell_feature(layer_states, B, N1))
            if extras:
                extra_cat = torch.cat(extras, dim=-1)
                if self.extra_modulate is not None:
                    extra_cat = self.extra_modulate(all_h[:, :S], extra_cat)
                extra = extra_cat.size(-1)
                shell_cat = torch.cat([all_h[:, :S], extra_cat], dim=-1)
                rest = all_h[:, S:]
                all_flat = torch.cat([
                    shell_cat.reshape(B, S * (self.dim + extra)),
                    rest.reshape(B, rest.size(1) * self.dim),
                ], dim=-1)
                first = getattr(self.smooth_head, 'net', self.smooth_head)[0]
                with torch.no_grad():
                    cols = [
                        first.weight[:, s * (self.dim + extra) + self.dim:
                                     (s + 1) * (self.dim + extra)]
                        for s in range(S)
                    ]
                    self.last_geo_col_abs = float(
                        torch.cat(cols, dim=-1).abs().mean())
            else:
                all_flat = all_h.reshape(B, all_h.size(1) * self.dim)
            if self.interface_concat_nodes > 0:
                first = getattr(self.smooth_head, 'net', self.smooth_head)[0]
                n_extra = int(self.interface_concat_nodes) * self.dim
                with torch.no_grad():
                    self.last_iface_col_abs = float(
                        first.weight[:, -n_extra:].abs().mean())
            side_h = None
            if self.side_tower is not None:
                if iface is None:
                    if pdb_codes is None:
                        raise ValueError('side_tower requires pdb_codes')
                    iface = self.coarse_cache().lookup_batch(
                        list(pdb_codes), device=device)
                raw_side = self.side_tower(iface, all_flat.dtype)
                side_h = torch.tanh(raw_side)
            if self.pl_interact_dim > 0:
                if self._p_idx is None or self._l_idx is None:
                    raise ValueError('pl_interact requires protein and ligand nodes')
                z = self._pl_interact_feature(
                    all_h[:, self._p_idx], all_h[:, self._l_idx])
                all_flat = torch.cat([all_flat, z], dim=-1)
                first = getattr(self.smooth_head, 'net', self.smooth_head)[0]
                with torch.no_grad():
                    self.last_pl_col_abs = float(
                        first.weight[:, -int(self.pl_interact_dim):].abs().mean())
            if self.diff_flatten_dim > 0:
                z = self._diff_flatten_feature(
                    H, src, dst, B, N1, h0_for_diff)
                all_flat = torch.cat([all_flat, z], dim=-1)
                first = getattr(self.smooth_head, 'net', self.smooth_head)[0]
                with torch.no_grad():
                    self.last_diff_col_abs = float(
                        first.weight[:, -int(self.diff_flatten_dim):].abs().mean())
            if hidden0_add is None:
                pred = self.smooth_head(all_flat, residual_384=side_h)
            else:
                pred = self.smooth_head(
                    all_flat, residual_384=side_h, hidden0_add=hidden0_add)
            if self.geo_head is not None and geo_channels is not None:
                pred = pred + self.geo_head(geo_channels)
            pred = self._add_pair_residual(pred, edge_embed, rel, dst, B, N1)
            if interface_edge_pred is not None:
                pred = pred + interface_edge_pred
            if self.output_affine:
                pred = self.pred_scale * pred + self.pred_bias
            return pred

        if self.readout_mode == 'ordered_lowrank':
            all_idx = (
                batch_off.unsqueeze(1)
                + torch.arange(N1, device=device).unsqueeze(0))
            all_h = H[all_idx.reshape(-1)].reshape(B, N1, self.dim)
            pred = self.ordered_lowrank_readout(
                all_h, self._p_idx, self._l_idx)
            if not self.use_ehign_aux:
                return pred
            assert edge_embed is not None
            edge_batch = dst.div(N1, rounding_mode='floor')
            to_shell = rel.eq(self.REL_P2S) | rel.eq(self.REL_L2S)
            from_shell = rel.eq(self.REL_S2P) | rel.eq(self.REL_S2L)
            _, pred_to_shell, pred_from_shell = self.edge_aux_readout(
                H, edge_index, edge_embed, edge_batch,
                to_shell, from_shell, B)
            return {
                'prediction': pred,
                'directional_predictions': [
                    pred_to_shell, pred_from_shell],
                'loss_mode': 'main_with_ehign_aux',
            }

        if self.readout_mode == 'full_ordered_residual':
            all_idx = (
                batch_off.unsqueeze(1)
                + torch.arange(N1, device=device).unsqueeze(0))
            all_h = H[all_idx.reshape(-1)].reshape(B, N1, self.dim)
            return self.full_ordered_residual_readout(
                all_h, self._p_idx, self._l_idx)

        if self.readout_mode == 'dual_query_radial':
            all_idx = batch_off.unsqueeze(1) + torch.arange(N1, device=device).unsqueeze(0)
            all_h = H[all_idx.reshape(-1)].reshape(B, N1, self.dim)
            return self.dual_query_readout(
                all_h[:, :S], all_h[:, self._p_idx], all_h[:, self._l_idx])

        if self.readout_mode in (
                'all_nodes_pool', 'hetero_type_pool', 'radial_bin_pool'):
            all_idx = (
                batch_off.unsqueeze(1)
                + torch.arange(N1, device=device).unsqueeze(0))
            all_h = H[all_idx.reshape(-1)].reshape(B, N1, self.dim)
            if self.readout_mode == 'all_nodes_pool':
                return self.all_nodes_pool_readout(all_h)
            if self.readout_mode == 'hetero_type_pool':
                return self.hetero_type_readout(
                    all_h[:, :S], all_h[:, self._p_idx], all_h[:, self._l_idx])
            return self.radial_bin_readout(
                all_h[:, :S], all_h[:, self._p_idx], all_h[:, self._l_idx])

        shell_idx = batch_off.unsqueeze(1) + torch.arange(S, device=device).unsqueeze(0)
        shell_flat = H[shell_idx.reshape(-1)].reshape(B, S * self.dim)

        if self.readout_mode == 'shell_only':
            feat = shell_flat
        elif self.readout_mode == 'all_nodes':
            all_idx = batch_off.unsqueeze(1) + torch.arange(N1, device=device).unsqueeze(0)
            feat = H[all_idx.reshape(-1)].reshape(B, N1 * self.dim)
        elif self.readout_mode == 'shell_concat_raw_lm':
            feat = torch.cat([shell_flat, lig_raw, prot_raw], dim=1)
        elif self.readout_mode == 'shell_bilinear_lm':
            feat = torch.cat([shell_flat, lig_raw * prot_raw], dim=1)
        elif self.readout_mode == 'shell_e1_head':
            shell_p = self.shell_proj(shell_flat)
            lm = torch.cat([
                lig_raw, prot_raw, lig_raw * prot_raw, (lig_raw - prot_raw).abs(),
            ], dim=1)
            feat = torch.cat([shell_p, lm], dim=1)
        else:
            return self.sym_readout(lig_raw, prot_raw, shell_flat)
        return self.head(feat)
