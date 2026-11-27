"""Shell-node initialization and dynamic graph encoding for MHGRL."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class PaperShellEncoder(nn.Module):
    """Encode standardized shell features in radial order with a BiLSTM."""

    def __init__(self, dim: int = 256, n_features: int = 168) -> None:
        super().__init__()
        if dim < 2 or dim % 2 or n_features < 1:
            raise ValueError("dim must be positive and even, and n_features must be positive")
        self.n_features = n_features
        self.lstm = nn.LSTM(n_features, dim // 2, batch_first=True, bidirectional=True)

    def forward(self, shell_features: torch.Tensor) -> torch.Tensor:
        if shell_features.ndim != 3 or shell_features.shape[-1] != self.n_features:
            raise ValueError(f"Expected shell features of shape [B, N, {self.n_features}]")
        states, _ = self.lstm(shell_features)
        return states


class PaperGraphLayer(nn.Module):
    """Select shell neighbors and aggregate their values with multi-head attention."""

    def __init__(self, dim: int, n_heads: int, n_shells: int, k: int,
                 dropout: float, ffn_mult: int) -> None:
        super().__init__()
        if not 1 <= k < n_shells:
            raise ValueError("k must be positive and smaller than n_shells")
        if n_heads < 1 or dim < 1 or dim % n_heads or ffn_mult < 1:
            raise ValueError("dim must be divisible by n_heads and ffn_mult must be positive")
        self.n_shells = n_shells
        self.k = k
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.attn_norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_mult * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * dim, dim),
            nn.Dropout(dropout),
        )

    @torch.no_grad()
    def select_neighbors(self, normalized: torch.Tensor, *,
                         predefined: bool = False) -> torch.Tensor:
        """Return shell-key indices for each query, ordered by cosine similarity."""
        with torch.autocast(device_type=normalized.device.type, enabled=False):
            if normalized.dtype in (torch.float16, torch.bfloat16):
                normalized = normalized.float()
            unit = F.normalize(normalized, dim=-1)
            scores = unit @ unit[:, :self.n_shells].transpose(1, 2)
            shell_idx = torch.arange(self.n_shells, device=normalized.device)
            scores[:, shell_idx, shell_idx] = -torch.inf
            if predefined:
                distance = (shell_idx[:, None] - shell_idx).abs()
                scores[:, :self.n_shells] = -distance
            return scores.argsort(dim=-1, descending=True, stable=True)[..., :self.k]

    @torch.no_grad()
    def build_topology(self, states: torch.Tensor, *, predefined: bool = False) -> torch.Tensor:
        return self.select_neighbors(self.attn_norm(states), predefined=predefined)

    def forward(self, states: torch.Tensor, neighbors: torch.Tensor | None = None) -> torch.Tensor:
        batch, n_nodes, dim = states.shape
        normalized = self.attn_norm(states)
        if neighbors is None:
            neighbors = self.select_neighbors(normalized)
        q = self.q_proj(normalized).reshape(batch, n_nodes, self.n_heads, self.head_dim)
        shells = normalized[:, :self.n_shells]
        k = self.k_proj(shells).reshape(batch, self.n_shells, self.n_heads, self.head_dim)
        v = self.v_proj(shells).reshape(batch, self.n_shells, self.n_heads, self.head_dim)
        batch_idx = torch.arange(batch, device=states.device)[:, None, None]
        key = k[batch_idx, neighbors]
        value = v[batch_idx, neighbors]
        scores = (q.unsqueeze(2) * key).sum(dim=-1) * self.scale
        weights = scores.softmax(dim=2)
        merged = (weights.unsqueeze(-1) * value).sum(dim=2).reshape(batch, n_nodes, dim)
        states = states + self.attn_dropout(self.out_proj(merged))
        return states + self.ffn(self.ffn_norm(states))


class PaperMHGRL(nn.Module):
    """Fuse shell, protein, and ligand nodes and predict binding affinity."""

    def __init__(self, dim: int = 256, n_shells: int = 60, n_layers: int = 3,
                 n_heads: int = 4, k: int = 5, dropout: float = 0.1,
                 ffn_mult: int = 2, lig_dim: int = 384, prot_dim: int = 1024,
                 head_dims: tuple[int, ...] = (1536, 384),
                 topology_mode: str = "dynamic") -> None:
        super().__init__()
        if n_layers < 1 or lig_dim < 1 or prot_dim < 1:
            raise ValueError("Layer count and embedding dimensions must be positive")
        if not head_dims or any(width < 1 for width in head_dims):
            raise ValueError("The prediction head must have positive hidden dimensions")
        if topology_mode not in ("dynamic", "predefined", "first_layer"):
            raise ValueError("topology_mode must be dynamic, predefined, or first_layer")
        self.dim = dim
        self.n_shells = n_shells
        self.k = k
        self.topology_mode = topology_mode
        self.ligand_proj = nn.Linear(lig_dim, dim)
        self.protein_proj = nn.Linear(prot_dim, dim)
        self.layers = nn.ModuleList(
            PaperGraphLayer(dim, n_heads, n_shells, k, dropout, ffn_mult)
            for _ in range(n_layers)
        )
        widths = ((n_shells + 2) * dim, *head_dims)
        head = []
        for in_dim, out_dim in zip(widths, widths[1:]):
            head.extend((nn.Linear(in_dim, out_dim), nn.GELU()))
        self.head = nn.Sequential(*head, nn.Linear(widths[-1], 1))

    def initial_adjacency(self) -> torch.Tensor:
        """Return the physical MMHG before the first layer rebuilds its edges."""
        n = self.n_shells
        idx = torch.arange(n, device=self.ligand_proj.weight.device)
        distance = (idx[:, None] - idx).abs()
        distance.fill_diagonal_(n)
        nearest = distance.argsort(dim=-1, stable=True)[:, :self.k]
        adjacency = torch.zeros((n + 2, n + 2), device=idx.device, dtype=torch.bool)
        adjacency[idx[:, None], nearest] = True
        adjacency[:n, n:] = True
        adjacency[n:, :n] = True
        return adjacency

    def encode(self, shell_states: torch.Tensor, ligand_embedding: torch.Tensor,
               protein_embedding: torch.Tensor) -> torch.Tensor:
        """Return final node states in shell, protein, ligand order."""
        if shell_states.ndim != 3 or shell_states.shape[1:] != (self.n_shells, self.dim):
            raise ValueError(f"Expected shell states of shape [B, {self.n_shells}, {self.dim}]")
        batch = shell_states.shape[0]
        if ligand_embedding.shape != (batch, self.ligand_proj.in_features):
            raise ValueError("Ligand embeddings do not match the model input shape")
        if protein_embedding.shape != (batch, self.protein_proj.in_features):
            raise ValueError("Protein embeddings do not match the model input shape")
        protein = self.protein_proj(protein_embedding).unsqueeze(1)
        ligand = self.ligand_proj(ligand_embedding).unsqueeze(1)
        states = torch.cat((shell_states, protein, ligand), dim=1)
        neighbors = None
        if self.topology_mode != "dynamic":
            neighbors = self.layers[0].build_topology(
                states, predefined=self.topology_mode == "predefined")
        for layer in self.layers:
            states = layer(states, neighbors)
        return states

    def forward(self, shell_states: torch.Tensor, ligand_embedding: torch.Tensor,
                protein_embedding: torch.Tensor) -> torch.Tensor:
        states = self.encode(shell_states, ligand_embedding, protein_embedding)
        return self.head(states.flatten(start_dim=1)).squeeze(-1)
