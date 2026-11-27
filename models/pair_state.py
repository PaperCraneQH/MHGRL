"""Optional edge-state updates, node feedback, and affinity residuals.

Residual outputs start at zero so the update initially preserves the input.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _segment_sum(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = src.new_zeros((n, *src.shape[1:]))
    out.index_add_(0, index, src)
    return out


class EdgeStateUpdate(nn.Module):
    """IGN / Interformer-style edge rewrite: e ← e + MLP([e, h_s, h_d, h_s⊙h_d]).

    The last linear is zeros, so the first forward is an identity on e.
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 64,
                 extra_dim: int = 0):
        super().__init__()
        self.extra_dim = int(extra_dim)
        in_dim = int(edge_dim) + 3 * int(node_dim) + self.extra_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), int(edge_dim)),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, H: torch.Tensor, src: torch.Tensor, dst: torch.Tensor,
                edge_embed: torch.Tensor,
                extra: Optional[torch.Tensor] = None) -> torch.Tensor:
        h_s, h_d = H[src], H[dst]
        parts = [edge_embed, h_s, h_d, h_s * h_d]
        if self.extra_dim > 0:
            if extra is None:
                extra = edge_embed.new_zeros(edge_embed.size(0), self.extra_dim)
            parts.append(extra)
        delta = self.net(torch.cat(parts, dim=-1))
        return edge_embed + delta


class PairToNodeFeedback(nn.Module):
    """AF3-style pair → single: mean incoming updated edges, add back to H."""

    def __init__(self, node_dim: int, edge_dim: int, gate_init: float = -3.0):
        super().__init__()
        self.proj = nn.Linear(int(edge_dim), int(node_dim), bias=False)
        nn.init.zeros_(self.proj.weight)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, H: torch.Tensor, edge_embed: torch.Tensor,
                dst: torch.Tensor) -> torch.Tensor:
        n = H.size(0)
        msg = self.proj(edge_embed)
        agg = _segment_sum(msg, dst, n)
        deg = _segment_sum(
            edge_embed.new_ones(dst.numel()), dst, n).clamp(min=1.0)
        return H + torch.sigmoid(self.gate) * (agg / deg.unsqueeze(-1))


class CrossPairResidual(nn.Module):
    """Boltz-2-style affinity residual from masked cross edges.

    Pools only the edges in ``mask`` (protein↔shell and ligand↔shell by
    default), then a small MLP produces a scalar added to the node-flatten
    prediction.  ``gate`` starts at -3 so the residual is ~0.05× at init.
    """

    def __init__(self, edge_dim: int, hidden: int = 64,
                 gate_init: float = -3.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(int(edge_dim), int(hidden)),
            nn.GELU(),
            nn.LayerNorm(int(hidden)),
        )
        self.out = nn.Linear(int(hidden), 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, edge_embed: torch.Tensor, edge_batch: torch.Tensor,
                mask: torch.Tensor, batch_size: int) -> torch.Tensor:
        w = mask.to(dtype=edge_embed.dtype).unsqueeze(-1)
        feat = self.proj(edge_embed) * w
        pooled = _segment_sum(feat, edge_batch, batch_size)
        denom = _segment_sum(w.squeeze(-1), edge_batch, batch_size).clamp(min=1.0)
        pooled = pooled / denom.unsqueeze(-1)
        residual = torch.sigmoid(self.gate) * self.out(pooled)
        self.last_abs_mean = float(residual.detach().abs().mean().item())
        return residual


def build_edge_updaters(n_layers: int, node_dim: int, edge_dim: int,
                        hidden: int) -> nn.ModuleList:
    return nn.ModuleList([
        EdgeStateUpdate(node_dim, edge_dim, hidden)
        for _ in range(int(n_layers))
    ])


def build_pair_residual(cfg: Optional[dict], edge_dim: int
                        ) -> Optional[CrossPairResidual]:
    cfg = dict(cfg or {})
    if str(cfg.get('pair_readout', 'none')).lower() != 'residual':
        return None
    return CrossPairResidual(
        edge_dim,
        hidden=int(cfg.get('pair_readout_hidden', 64)),
        gate_init=float(cfg.get('pair_gate_init', -3.0)),
    )


def build_pair_to_node(cfg: Optional[dict], node_dim: int, edge_dim: int
                       ) -> Optional[PairToNodeFeedback]:
    cfg = dict(cfg or {})
    if not bool(cfg.get('pair_to_node', False)):
        return None
    return PairToNodeFeedback(
        node_dim, edge_dim,
        gate_init=float(cfg.get('pair_gate_init', -3.0)),
    )


class DensePairUpdate(nn.Module):
    """One Pairformer-style layer on a dense z[B, N, N, C].

    Last linear is zeros, so the first forward leaves the scattered edge
    seed unchanged and only later steps write into pairs that had no edge.
    """

    def __init__(self, node_dim: int, pair_dim: int, hidden: int = 64):
        super().__init__()
        in_dim = int(pair_dim) + 3 * int(node_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), int(pair_dim)),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        h_i = H.unsqueeze(2).expand(-1, -1, H.size(1), -1)
        h_j = H.unsqueeze(1).expand(-1, H.size(1), -1, -1)
        delta = self.net(torch.cat([z, h_i, h_j, h_i * h_j], dim=-1))
        return z + delta


class DensePairReadout(nn.Module):
    """Affinity from a masked mean of the dense pair tensor.

    This is the primary head (not a gated residual).  ``last_abs_mean`` is
    the mechanism probe: it must leave ~0 if the pair path is in use.
    """

    def __init__(self, pair_dim: int, hidden: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(int(pair_dim), int(hidden)),
            nn.GELU(),
            nn.LayerNorm(int(hidden)),
        )
        self.out = nn.Linear(int(hidden), 1)

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        w = mask.to(dtype=z.dtype).view(1, mask.size(0), mask.size(1), 1)
        feat = self.proj(z) * w
        denom = w.sum(dim=(1, 2)).clamp(min=1.0)
        pooled = feat.sum(dim=(1, 2)) / denom
        pred = self.out(pooled)
        self.last_abs_mean = float(pred.detach().abs().mean().item())
        return pred


class DensePairPrimary(nn.Module):
    """Seed z from sparse edges, update densely, read affinity from cross blocks."""

    def __init__(self, node_dim: int, pair_dim: int, n_nodes: int,
                 n_layers: int = 2, hidden: int = 64,
                 readout_hidden: int = 64):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.pair_dim = int(pair_dim)
        self.updates = nn.ModuleList([
            DensePairUpdate(node_dim, pair_dim, hidden)
            for _ in range(int(n_layers))
        ])
        self.readout = DensePairReadout(pair_dim, readout_hidden)

    def seed(self, edge_embed: Optional[torch.Tensor], src: torch.Tensor,
             dst: torch.Tensor, batch_size: int) -> torch.Tensor:
        n, c = self.n_nodes, self.pair_dim
        device = src.device
        dtype = (edge_embed.dtype if edge_embed is not None
                 else torch.float32)
        z = torch.zeros(batch_size, n, n, c, device=device, dtype=dtype)
        if edge_embed is None or src.numel() == 0:
            return z
        if int(edge_embed.size(-1)) != c:
            raise ValueError(
                f'edge_embed dim {edge_embed.size(-1)} != pair_dim {c}')
        local_src = src.remainder(n)
        local_dst = dst.remainder(n)
        edge_batch = src.div(n, rounding_mode='floor')
        z[edge_batch, local_src, local_dst] = edge_embed
        return z

    def forward(self, H: torch.Tensor, edge_embed: Optional[torch.Tensor],
                src: torch.Tensor, dst: torch.Tensor, mask: torch.Tensor,
                batch_size: int) -> torch.Tensor:
        nodes = H.view(batch_size, self.n_nodes, H.size(-1))
        z = self.seed(edge_embed, src, dst, batch_size)
        for update in self.updates:
            z = update(z, nodes)
        pred = self.readout(z, mask)
        self.last_abs_mean = self.readout.last_abs_mean
        return pred


def cross_pair_block_mask(n_nodes: int, n_shells: int,
                          p_idx: Optional[int], l_idx: Optional[int],
                          scope: str, device) -> torch.Tensor:
    """Boolean [N, N] mask of the pairs that enter the primary readout."""
    mask = torch.zeros(n_nodes, n_nodes, dtype=torch.bool, device=device)
    if scope == 'all':
        mask.fill_(True)
        mask.fill_diagonal_(False)
        return mask
    if p_idx is not None:
        mask[p_idx, :n_shells] = True
        mask[:n_shells, p_idx] = True
    if l_idx is not None:
        mask[l_idx, :n_shells] = True
        mask[:n_shells, l_idx] = True
    if scope == 'cross_pl' and p_idx is not None and l_idx is not None:
        mask[p_idx, l_idx] = True
        mask[l_idx, p_idx] = True
    mask.fill_diagonal_(False)
    return mask


def build_dense_pair(cfg: Optional[dict], node_dim: int, pair_dim: int,
                     n_nodes: int) -> Optional[DensePairPrimary]:
    cfg = dict(cfg or {})
    if str(cfg.get('pair_readout', 'none')).lower() != 'primary':
        return None
    return DensePairPrimary(
        node_dim=node_dim,
        pair_dim=int(cfg.get('pair_dense_dim', pair_dim)),
        n_nodes=n_nodes,
        n_layers=int(cfg.get('pair_dense_layers', 2)),
        hidden=int(cfg.get('pair_dense_hidden', 64)),
        readout_hidden=int(cfg.get('pair_readout_hidden', 64)),
    )
