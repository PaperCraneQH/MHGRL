"""Structure tower: residue N60 features -> ShellGraphGTModel -> projection."""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .shell_graph_gt_model import ShellGraphGTModel


class _ResidueFeatStore:
    def __init__(self, features_path: str, scaler_path: Optional[str] = None,
                 missing_policy: str = 'zero'):
        if not os.path.isfile(features_path):
            raise FileNotFoundError(f'_ResidueFeatStore: not found {features_path}')
        z = np.load(features_path, allow_pickle=True)
        codes = z['codes']
        self.features = np.ascontiguousarray(z['feats'].astype(np.float32))
        self.n_features = int(self.features.shape[1])
        self.pdb_to_idx: Dict[str, int] = {
            (str(c).lower().strip()): i for i, c in enumerate(codes)
        }
        self.missing_policy = missing_policy
        self._zero_vec = np.zeros(self.n_features, dtype=np.float32)
        self.n_lookups = 0
        self.n_missing = 0

        if scaler_path and os.path.isfile(scaler_path):
            s = np.load(scaler_path)
            mean = s['mean'].astype(np.float32)
            std = s['std'].astype(np.float32)
        else:
            mean = self.features.mean(axis=0)
            std = self.features.std(axis=0)
            print(f'[ShellGraphStructEncoder][WARN] scaler missing ({scaler_path}); '
                  f'using global stats.', flush=True)
        std = np.where(std < 1e-6, 1.0, std)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def lookup_batch(self, pdb_codes: List[str]) -> np.ndarray:
        out = np.empty((len(pdb_codes), self.n_features), dtype=np.float32)
        for b, pdb in enumerate(pdb_codes):
            self.n_lookups += 1
            key = (pdb or '').lower().strip()
            idx = self.pdb_to_idx.get(key, None)
            if idx is None:
                self.n_missing += 1
                if self.missing_policy == 'zero':
                    out[b] = self._zero_vec
                else:
                    raise KeyError(f'PDB {pdb!r} not in residue_N60 store')
            else:
                out[b] = self.features[idx]
        return out


class ShellGraphStructEncoder(nn.Module):
    """Structure tower backed by the Graph-Transformer shell-graph encoder."""

    TRAIN_MODES = ('e2e', 'freeze_backbone', 'pretrained_init')
    FEATURE_LAYOUTS = ('legacy_pair_major', 'shell_major')

    def __init__(self,
                 features_path: str,
                 scaler_path: Optional[str] = None,
                 out_dim: int = 128,
                 n_pairs: int = 168,
                 n_shells: int = 60,
                 gnn_type: str = 'GAT_GCN',
                 hidden_dim: int = 128,
                 n_layers: int = 3,
                 k_neighbors: int = 2,
                 n_heads: int = 4,
                 lstm_num_layers: int = 1,
                 use_ffn: bool = True,
                 pos_encoding: str = 'none',
                 node_embed_type: str = 'lstm',
                 pair_factor_rank: int = 32,
                 pooling: str = 'flatten',
                 dropout: float = 0.1,
                 proj_dropout: float = 0.2,
                 missing_policy: str = 'zero',
                 train_mode: str = 'e2e',
                 tf_ffn_mult: int = 2,
                 tf_dropout: float = -1.0,
                 return_graph_inputs: bool = False,
                 graph_input_stage: str = 'pre_block',
                 freq_mix_cfg: Optional[Dict] = None,
                 residue_slots_cfg: Optional[Dict] = None,
                 feature_layout: str = 'legacy_pair_major',
                 **kwargs):
        super().__init__()
        self.n_pairs = int(n_pairs)
        self.n_shells = int(n_shells)
        self.feature_layout = str(feature_layout).lower()
        if self.feature_layout not in self.FEATURE_LAYOUTS:
            raise ValueError(
                f'feature_layout must be one of {self.FEATURE_LAYOUTS}, '
                f'got {feature_layout!r}')
        self.freq_mix_cfg = dict(freq_mix_cfg or {})
        self.residue_slots_cfg = dict(residue_slots_cfg or {})
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        # Graph fusion consumes shell nodes and raw contact frequencies.
        self.return_graph_inputs = bool(return_graph_inputs)
        self.graph_input_stage = str(graph_input_stage).lower()
        if self.graph_input_stage not in ('pre_block', 'post_block'):
            raise ValueError(
                'graph_input_stage must be pre_block|post_block, '
                f'got {graph_input_stage!r}')
        self.train_mode = str(train_mode).lower()
        if self.train_mode not in self.TRAIN_MODES:
            raise ValueError(f'train_mode must be one of {self.TRAIN_MODES}, got {train_mode!r}')

        self.store = _ResidueFeatStore(features_path, scaler_path=scaler_path,
                                       missing_policy=missing_policy)
        self.register_buffer('feat_mean', torch.from_numpy(self.store.mean), persistent=False)
        self.register_buffer('feat_std', torch.from_numpy(self.store.std), persistent=False)

        self.backbone = ShellGraphGTModel(
            in_shape=(self.n_pairs, self.n_shells, 1),
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            k_neighbors=k_neighbors,
            gnn_type=gnn_type,
            n_heads=n_heads,
            dropout=dropout,
            lstm_num_layers=lstm_num_layers,
            use_ffn=use_ffn,
            ffn_mult=tf_ffn_mult,
            attn_dropout=tf_dropout,
            pos_encoding=pos_encoding,
            node_embed_type=node_embed_type,
            pair_factor_rank=pair_factor_rank,
            pooling=pooling,
        )
        pool_out = self.backbone.pool.out_dim
        self.backbone.classifier = nn.Identity()
        self.proj = nn.Sequential(
            nn.Linear(pool_out, out_dim),
            nn.ReLU(),
            nn.Dropout(proj_dropout),
        )
        self.apply_train_mode()
        self.freq_mix = None
        self.residue_slots = None
        self._coarse_cache = None
        if bool(self.freq_mix_cfg.get('enabled', False)):
            from .freq_mix import FreqMix
            self.freq_mix = FreqMix(
                mode=str(self.freq_mix_cfg.get('mode', 'mix')),
                gate_init=float(self.freq_mix_cfg.get('gate_init', -4.0)))
        if bool(self.residue_slots_cfg.get('enabled', False)):
            from .residue_slots import ResidueSlotEncoder
            self.residue_slots = ResidueSlotEncoder(
                hidden_dim=int(self.hidden_dim),
                n_slots=self.n_shells,
                n_bins=int(self.residue_slots_cfg.get('n_bins', self.n_shells)),
                mode=str(self.residue_slots_cfg.get('mode', 'hist')))

    def _stored_flat_to_pair_shell(self, flat: torch.Tensor) -> torch.Tensor:
        """Convert flattened NPZ rows to the backbone's ``[B, pair, shell]``.

        Historical experiments interpreted the flat vector directly as
        pair-major.  The feature generator actually writes shell-major rows,
        so new experiments explicitly transpose that stored layout.  Keeping
        the legacy branch as the default preserves old checkpoints/results.
        """
        if self.feature_layout == 'legacy_pair_major':
            return flat.view(flat.size(0), self.n_pairs, self.n_shells)
        return flat.view(flat.size(0), self.n_shells, self.n_pairs) \
            .transpose(1, 2).contiguous()

    def _pair_shell_to_stored_flat(self, table: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`_stored_flat_to_pair_shell` for mixed features."""
        if self.feature_layout == 'legacy_pair_major':
            return table.reshape(table.size(0), -1)
        return table.transpose(1, 2).contiguous().reshape(table.size(0), -1)

    def apply_train_mode(self) -> None:
        if self.train_mode == 'e2e':
            for p in self.backbone.parameters():
                p.requires_grad = True
        else:
            for p in self.backbone.parameters():
                p.requires_grad = False
        # The output projection is bypassed when fusion consumes node tensors.
        for p in self.proj.parameters():
            p.requires_grad = not self.return_graph_inputs

    def load_pretrained_backbone(self, ckpt_path: str) -> int:
        if not ckpt_path or not os.path.isfile(ckpt_path):
            print(f'[ShellGraphStructEncoder][WARN] pretrained ckpt not found: {ckpt_path}',
                  flush=True)
            return 0
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        src = ckpt.get('state_dict', ckpt)
        mapped = {}
        for k, v in src.items():
            if k.startswith('classifier.'):
                continue
            mapped[f'backbone.{k}'] = v
        missing, unexpected = self.load_state_dict(mapped, strict=False)
        n_loaded = len(mapped) - len([m for m in missing if m.startswith('backbone.')])
        print(f'[ShellGraphStructEncoder] loaded pretrained backbone from {ckpt_path} '
              f'({n_loaded} tensors)', flush=True)
        if unexpected:
            print(f'  unexpected keys (truncated): {unexpected[:5]}', flush=True)
        self.apply_train_mode()
        return n_loaded

    def coarse_cache(self):
        if self._coarse_cache is None:
            from data.coarse_interface_cache import CoarseInterfaceCache
            src = self.residue_slots_cfg if self.residue_slots_cfg.get('cache_path') else self.freq_mix_cfg
            path = src.get(
                'cache_path',
                'data_processed_esm/coarse_interface/cb_heavy.h5')
            if not os.path.isabs(path):
                root = os.path.abspath(os.path.join(
                    os.path.dirname(__file__), os.pardir))
                path = os.path.join(root, path)
            self._coarse_cache = CoarseInterfaceCache(
                path,
                max_prot=int(src.get('max_cb', 64)),
                max_lig=int(src.get('max_atom', 48)),
                in_memory=bool(src.get('cache_in_memory', True)))
        return self._coarse_cache

    def forward(self, pdb_codes: List[str],
                device: Optional[torch.device] = None) -> torch.Tensor:
        if device is None:
            device = next(self.proj.parameters()).device
        x = self.store.lookup_batch(pdb_codes)
        x_raw = torch.from_numpy(x).to(device)
        onion = self._stored_flat_to_pair_shell(x_raw)
        freq_raw = onion
        if self.freq_mix is not None:
            from .freq_mix import pair_distance_hist
            iface = self.coarse_cache().lookup_batch(
                list(pdb_codes), device=device)
            three = pair_distance_hist(iface, self.n_shells).to(x_raw.dtype)
            mixed = self.freq_mix(onion, three)
            freq_raw = mixed
            if self.freq_mix.mode != 'edges':
                # The scaler is stored in the NPZ's original flattened order.
                x_raw = self._pair_shell_to_stored_flat(mixed)
        if self.residue_slots is not None:
            iface = self.coarse_cache().lookup_batch(
                list(pdb_codes), device=device)
            table = self.residue_slots.table(iface)
            edge_from = str(self.residue_slots_cfg.get('edges', 'residue'))
            if edge_from == 'residue':
                freq_raw = table
            if self.residue_slots.mode == 'table':
                mu = table.mean(dim=(1, 2), keepdim=True)
                sd = table.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
                x = (table - mu) / sd
                if self.graph_input_stage == 'post_block':
                    shell_embed = self.backbone(x, return_nodes=True)
                else:
                    shell_embed = self.backbone(x, return_pre_block=True)
                return shell_embed, freq_raw
            shell_embed = self.residue_slots.encode_nodes(iface)
            return shell_embed, freq_raw
        x = (x_raw - self.feat_mean) / self.feat_std
        x = self._stored_flat_to_pair_shell(x)
        if self.return_graph_inputs:
            if self.graph_input_stage == 'post_block':
                shell_embed = self.backbone(x, return_nodes=True)
            else:
                shell_embed = self.backbone(x, return_pre_block=True)
            return shell_embed, freq_raw
        h = self.backbone(x, return_embedding=True)
        return self.proj(h)
