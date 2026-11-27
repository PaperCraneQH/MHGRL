"""Project ordered shell geometry into a residual for the regression head."""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from .shell_geometry import ShellGeoChannels


def _sinusoidal_pe(n: int, dim: int) -> torch.Tensor:
    pos = torch.arange(n).unsqueeze(1).float()
    div = torch.exp(
        torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros(n, dim)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].size(1)])
    return pe.unsqueeze(0)


class SideShellTower(nn.Module):
    """Project per-shell geometry to 384-d.  Last linear starts at zero."""

    def __init__(
        self,
        mode: str = 'rich_pair',
        n_shells: int = 60,
        token_dim: int = 192,
        per_shell: int = 16,
        hidden: int = 384,
        n_layers: int = 0,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.channels = ShellGeoChannels(mode=mode, n_shells=n_shells)
        self.n_shells = int(n_shells)
        self.in_proj = nn.Linear(self.channels.out_dim, int(token_dim))
        self.register_buffer(
            'pe', _sinusoidal_pe(int(n_shells), int(token_dim)), persistent=False)
        self.encoder = None
        if int(n_layers) > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=int(token_dim), nhead=int(n_heads),
                dim_feedforward=int(token_dim) * 2,
                dropout=float(dropout), batch_first=True,
                activation='gelu')
            self.encoder = nn.TransformerEncoder(layer, num_layers=int(n_layers))
        self.down = nn.Linear(int(token_dim), int(per_shell))
        self.to_hidden = nn.Linear(int(n_shells) * int(per_shell), int(hidden))
        nn.init.zeros_(self.to_hidden.weight)
        nn.init.zeros_(self.to_hidden.bias)
        self.last_abs = 0.0
        self.last_n_pl = 0.0

    def forward(self, batch: Dict[str, torch.Tensor],
                dtype: torch.dtype) -> torch.Tensor:
        geo = self.channels(batch, dtype)
        tok = self.in_proj(geo) + self.pe.to(device=geo.device, dtype=dtype)
        if self.encoder is not None:
            tok = self.encoder(tok)
        compact = self.down(tok).reshape(tok.size(0), -1)
        hidden = self.to_hidden(compact)
        with torch.no_grad():
            self.last_abs = float(hidden.detach().abs().mean())
            self.last_n_pl = float(getattr(self.channels, 'last_n_pl', 0.0))
        return hidden
