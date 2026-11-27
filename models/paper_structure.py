"""Load and normalize residue contacts for the MHGRL shell encoder."""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .paper_mhgrl import PaperShellEncoder


class PaperStructureEncoder(nn.Module):
    def __init__(self, features_path, scaler_path, n_shells, n_features, dim):
        super().__init__()
        if not Path(scaler_path).is_file():
            raise FileNotFoundError(f"Missing train-only scaler: {scaler_path}")
        self.n_shells = n_shells
        self.n_features = n_features
        width = n_shells * n_features
        with np.load(features_path, allow_pickle=False) as archive:
            if ("partition_family" not in archive
                    or archive["partition_family"].item() != "paper_residue_min_0_30p5A"
                    or archive["feature_layout"].item() != "shell_major"):
                raise ValueError(f"Expected features built by build_paper_shell_features.py: {features_path}")
            codes = [str(code).lower().strip() for code in archive["codes"]]
            self.counts = np.asarray(archive["feats"], dtype=np.float32)
        if self.counts.shape != (len(codes), width) or len(set(codes)) != len(codes):
            raise ValueError("Invalid shell feature shape or duplicate PDB codes")
        self.code_index = {code: index for index, code in enumerate(codes)}
        with np.load(scaler_path, allow_pickle=False) as stats:
            self.mean = np.asarray(stats["mean"], dtype=np.float32)
            std = np.asarray(stats["std"], dtype=np.float32)
        if (self.mean.shape != (width,) or std.shape != (width,)
                or not np.isfinite(self.mean).all() or not np.isfinite(std).all()
                or (std < 0).any()):
            raise ValueError("Invalid training scaler for the shell features")
        self.std = np.where(std < 1e-6, 1.0, std)
        self.encoder = PaperShellEncoder(dim=dim, n_features=n_features)

    def forward(self, pdb_codes, device):
        indices = [self.code_index[str(code).lower().strip()] for code in pdb_codes]
        normalized = (self.counts[indices] - self.mean) / self.std
        x = torch.as_tensor(normalized, device=device)
        return self.encoder(x.reshape(len(pdb_codes), self.n_shells, self.n_features))
