"""Shared building blocks for the Shell-Graph Graph-Transformer backbone."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .shell_graph_model import _build_chain_k_adjacency


# OnionNet order: GLY ALA VAL LEU ILE PRO PHE TYR TRP SER THR CYS MET
# ASN GLN ASP GLU LYS ARG HIS OTH.  Ligand: H C O N P S Hal DU.
_HYDRO_RES = (0, 1, 2, 3, 4, 5, 6, 8, 12)
_POLAR_RES = (7, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19)
_HYDRO_LIG = (0, 1, 6)
_POLAR_LIG = (2, 3, 4, 5)


def signed_contact_channels(pair: torch.Tensor) -> torch.Tensor:
    """Turn a 21×8 contact table into 4 signed shell channels.

    ``pair`` is ``[..., 21, 8]``.  Returns ``[..., 4]``:

    0 favorable (hydro–hydro + polar–polar)
    1 unfavorable (hydro–polar + polar–hydro)
    2 polar residue × carbon (crude desolvation)
    3 favorable − unfavorable
    """
    hydro_res = pair[..., _HYDRO_RES, :].sum(dim=-2)
    polar_res = pair[..., _POLAR_RES, :].sum(dim=-2)
    fav = (hydro_res[..., _HYDRO_LIG].sum(dim=-1)
           + polar_res[..., _POLAR_LIG].sum(dim=-1))
    unfav = (hydro_res[..., _POLAR_LIG].sum(dim=-1)
             + polar_res[..., _HYDRO_LIG].sum(dim=-1))
    polar_carbon = pair[..., _POLAR_RES, 1].sum(dim=-1)
    return torch.stack([fav, unfav, polar_carbon, fav - unfav], dim=-1)


class LSTMNodeEmbed(nn.Module):
    """Bi-LSTM node embedding: [B, N, in_dim] -> [B, N, hidden_dim]."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % 2 == 0, 'hidden_dim must be even for bi-LSTM'
        self.num_layers = int(num_layers)
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim // 2,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=(dropout if self.num_layers > 1 else 0.0),
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        h = self.ln(h)
        return self.dropout(h)


class IndependentLinearNodeEmbed(nn.Module):
    """Build each shell independently; graph layers perform all shell mixing."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MarginalLSTMNodeEmbed(nn.Module):
    """Use residue and element marginals, then retain the shell Bi-LSTM."""

    def __init__(self, hidden_dim: int, num_layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.node_embed = LSTMNodeEmbed(
            21 + 8, hidden_dim, num_layers=num_layers, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pair = x.reshape(*x.shape[:-1], 21, 8)
        residue = pair.sum(dim=-1) / (8.0 ** 0.5)
        element = pair.sum(dim=-2) / (21.0 ** 0.5)
        return self.node_embed(torch.cat([residue, element], dim=-1))


class SignedShellResidual(nn.Module):
    """Add a zero-init residual built from raw signed contact counts."""

    def __init__(self, node_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, int(node_dim)),
            nn.GELU(),
            nn.Linear(int(node_dim), int(node_dim)),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.last_abs_mean = 0.0

    def forward(self, shell: torch.Tensor, freq_raw: torch.Tensor) -> torch.Tensor:
        # freq_raw [B, 168, 60] -> [B, 60, 21, 8]
        pair = freq_raw.transpose(1, 2).reshape(
            freq_raw.size(0), freq_raw.size(2), 21, 8)
        signed = signed_contact_channels(pair)
        channels = torch.stack([
            torch.log1p(signed[..., 0].clamp_min(0.0)),
            torch.log1p(signed[..., 1].clamp_min(0.0)),
            torch.log1p(signed[..., 2].clamp_min(0.0)),
            signed[..., 3] / (signed[..., 0] + signed[..., 1] + 1.0),
        ], dim=-1)
        delta = self.mlp(channels.to(shell.dtype))
        with torch.no_grad():
            self.last_abs_mean = float(delta.abs().mean())
        return shell + delta


class SignedLSTMNodeEmbed(nn.Module):
    """Keep the 168 pair channels and append 4 signed contact summaries."""

    def __init__(self, hidden_dim: int, num_layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.node_embed = LSTMNodeEmbed(
            168 + 4, hidden_dim, num_layers=num_layers, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pair = x.reshape(*x.shape[:-1], 21, 8)
        signed = signed_contact_channels(pair)
        return self.node_embed(torch.cat([x, signed], dim=-1))


class FactorizedPairLSTMNodeEmbed(nn.Module):
    """Factor the 21x8 contact table into chemical pair basis channels.

    Each basis is an elementwise product of a residue embedding and an element
    embedding.  A Bi-LSTM then models the ordered shell sequence exactly as in
    the baseline, isolating the contact-channel representation as the changed
    factor.
    """

    def __init__(self, rank: int, hidden_dim: int, num_layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        if int(rank) <= 0:
            raise ValueError('factorized pair rank must be positive')
        self.rank = int(rank)
        self.residue_embedding = nn.Parameter(torch.empty(21, self.rank))
        self.element_embedding = nn.Parameter(torch.empty(8, self.rank))
        nn.init.xavier_uniform_(self.residue_embedding)
        nn.init.xavier_uniform_(self.element_embedding)
        self.node_embed = LSTMNodeEmbed(
            self.rank, hidden_dim, num_layers=num_layers, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pair = x.reshape(*x.shape[:-1], 21, 8)
        basis = (
            self.residue_embedding[:, None, :]
            * self.element_embedding[None, :, :])
        encoded = torch.einsum('...ae,aer->...r', pair, basis)
        encoded = encoded / (168.0 ** 0.5)
        return self.node_embed(encoded)


class PositionalEncoding(nn.Module):
    """Positional / structural encoding along the shell axis (length N)."""

    def __init__(self, mode: str, n_shells: int, hidden_dim: int,
                 k_neighbors: int, k_rw: int = 8):
        super().__init__()
        assert mode in ('none', 'sinusoidal', 'learnable', 'rwse'), \
            f'unknown PE mode: {mode}'
        self.mode = mode
        self.n_shells = int(n_shells)
        self.hidden_dim = int(hidden_dim)

        if mode == 'sinusoidal':
            pe = torch.zeros(n_shells, hidden_dim)
            position = torch.arange(0, n_shells, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, hidden_dim, 2).float()
                * (-math.log(10000.0) / hidden_dim)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe, persistent=False)

        elif mode == 'learnable':
            self.embed = nn.Embedding(n_shells, hidden_dim)
            nn.init.normal_(self.embed.weight, std=0.02)

        elif mode == 'rwse':
            adj = _build_chain_k_adjacency(n_shells, k_neighbors)
            A = adj['A_self'].float()
            deg = A.sum(dim=-1).clamp(min=1.0)
            P = A / deg.unsqueeze(-1)
            P_t = P.clone()
            rwse_diags = []
            for _ in range(k_rw):
                rwse_diags.append(torch.diagonal(P_t, 0))
                P_t = P_t @ P
            rwse = torch.stack(rwse_diags, dim=-1)
            self.register_buffer('rwse_features', rwse, persistent=False)
            self.proj = nn.Linear(k_rw, hidden_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.mode == 'none':
            return h
        if self.mode == 'sinusoidal':
            return h + self.pe.unsqueeze(0)
        if self.mode == 'learnable':
            idx = torch.arange(self.n_shells, device=h.device)
            return h + self.embed(idx).unsqueeze(0)
        if self.mode == 'rwse':
            pe = self.proj(self.rwse_features)
            return h + pe.unsqueeze(0)
        return h


class GraphPooling(nn.Module):
    """Graph-level readout: [B, N, D] -> [B, out_dim]."""

    def __init__(self, mode: str, n_shells: int, hidden_dim: int):
        super().__init__()
        assert mode in ('flatten', 'mean', 'sum', 'max', 'maxmean', 'attn'), \
            f'unknown pooling mode: {mode}'
        self.mode = mode
        self.n_shells = int(n_shells)
        self.hidden_dim = int(hidden_dim)

        if mode == 'flatten':
            self.out_dim = self.n_shells * self.hidden_dim
        elif mode in ('mean', 'sum', 'max'):
            self.out_dim = self.hidden_dim
        elif mode == 'maxmean':
            self.out_dim = 2 * self.hidden_dim
        elif mode == 'attn':
            self.out_dim = self.hidden_dim
            self.q_proj = nn.Linear(hidden_dim, hidden_dim)
            self.attn_query = nn.Parameter(torch.randn(hidden_dim) * 0.02)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.mode == 'flatten':
            return h.reshape(h.size(0), -1)
        if self.mode == 'mean':
            return h.mean(dim=1)
        if self.mode == 'sum':
            return h.sum(dim=1)
        if self.mode == 'max':
            return h.max(dim=1).values
        if self.mode == 'maxmean':
            return torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)
        if self.mode == 'attn':
            scores = (self.q_proj(h) * self.attn_query).sum(dim=-1)
            weights = F.softmax(scores, dim=-1).unsqueeze(-1)
            return (weights * h).sum(dim=1)
        raise RuntimeError(f'unsupported pooling mode {self.mode}')
