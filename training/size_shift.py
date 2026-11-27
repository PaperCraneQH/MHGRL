"""Optional importance weighting for ligand-size shifts.

Weights are target/source density ratios over log1p(heavy atoms), estimated
from smoothed histograms and scaled to unit mean. The target can be a size
range from the training data or a supplied unlabeled histogram."""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def _smooth(counts: torch.Tensor, kernel_sigma: float) -> torch.Tensor:
    """Gaussian-smooth a 1-D histogram (zero padding at both ends)."""
    if kernel_sigma <= 0:
        return counts
    half = max(int(math.ceil(3.0 * kernel_sigma)), 1)
    offsets = torch.arange(-half, half + 1, dtype=torch.float32)
    kernel = torch.exp(-offsets.pow(2) / (2.0 * kernel_sigma ** 2))
    kernel = kernel / kernel.sum()
    return F.conv1d(counts.view(1, 1, -1), kernel.view(1, 1, -1),
                    padding=half).view(-1)


class LigandSizeShiftWeights:
    """Per-sample importance weights that match the drug-like size regime."""

    def __init__(self, ligand_atom_counts: Sequence[float],
                 mode: str = 'size_range',
                 min_atoms: float = 4.0, max_atoms: float = 52.0,
                 hard_filter: bool = False,
                 target_counts: Optional[Sequence[float]] = None,
                 bin_width: float = 0.15, kernel_sigma: float = 1.0,
                 min_weight: float = 0.1, max_weight: float = 5.0):
        n = torch.as_tensor(ligand_atom_counts, dtype=torch.float32).view(-1)
        if n.numel() == 0:
            raise ValueError('size_shift requires ligand atom counts')
        if bin_width <= 0:
            raise ValueError('bin_width must be positive')
        if min_weight < 0 or max_weight <= min_weight:
            raise ValueError('require 0 <= min_weight < max_weight')

        mode = str(mode)
        if mode not in ('size_range', 'target_histogram'):
            raise ValueError(
                "size_shift.mode must be 'size_range' or 'target_histogram'")

        source = torch.log1p(n)
        if mode == 'size_range':
            keep = (n >= float(min_atoms)) & (n <= float(max_atoms))
            if int(keep.sum()) < 100:
                raise ValueError(
                    'size_range leaves too few training samples; widen it')
            target = source[keep]
        else:
            if target_counts is None:
                raise ValueError(
                    "mode='target_histogram' needs target_counts")
            t = torch.as_tensor(target_counts, dtype=torch.float32).view(-1)
            target = torch.log1p(t[t > 0])

        lo = float(min(source.min(), target.min()))
        hi = float(max(source.max(), target.max()))
        n_bins = max(int(math.ceil((hi - lo) / bin_width)), 2)
        edges = torch.linspace(lo, hi, n_bins + 1)[1:-1]

        src_bins = torch.bucketize(source, edges)
        tgt_bins = torch.bucketize(target, edges)
        src_hist = torch.bincount(src_bins, minlength=n_bins).float()
        tgt_hist = torch.bincount(tgt_bins, minlength=n_bins).float()
        src_den = _smooth(src_hist, kernel_sigma)
        tgt_den = _smooth(tgt_hist, kernel_sigma)
        src_den = src_den / src_den.sum().clamp_min(1e-8)
        tgt_den = tgt_den / tgt_den.sum().clamp_min(1e-8)

        ratio = tgt_den / src_den.clamp_min(1e-8)
        bin_w = ratio.clamp(float(min_weight), float(max_weight))
        weights = bin_w[src_bins]
        if hard_filter:
            outside = (n < float(min_atoms)) | (n > float(max_atoms))
            weights = weights.masked_fill(outside, 0.0)
        total = weights.sum().clamp_min(1e-8)
        # E_source[w] = 1 so the gradient scale matches unweighted training.
        weights = weights * (weights.numel() / total)

        self.weights = weights
        self.n_atoms = n
        self.mode = mode
        kept = int((weights > 0).sum())
        q = torch.tensor([0.05, 0.5, 0.95])
        wq = torch.quantile(weights[weights > 0], q).tolist()
        self.description = (
            f'ligand_size_shift mode={mode} bins={n_bins} '
            f'width={bin_width:g} sigma={kernel_sigma:g} '
            f'effective_samples={kept}/{weights.numel()} '
            f'weight_q05/50/95=[{wq[0]:.2f},{wq[1]:.2f},{wq[2]:.2f}] '
            f'mean_atoms {float(n.mean()):.1f} -> '
            f'{float((n * weights).sum() / total):.1f}')

    def sampler(self, num_samples_ratio: float = 1.0, seed: int = 0):
        """A ``WeightedRandomSampler`` realising these importance weights."""
        from torch.utils.data import WeightedRandomSampler
        num_samples = max(int(round(self.weights.numel()
                                    * float(num_samples_ratio))), 1)
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        return WeightedRandomSampler(
            weights=self.weights.double(), num_samples=num_samples,
            replacement=True, generator=generator)


def build_size_shift(cfg: Optional[Dict],
                     ligand_atom_counts: Sequence[float]
                     ) -> Optional[LigandSizeShiftWeights]:
    """Build importance weights from ``training.size_shift``; None when off."""
    cfg = dict(cfg or {})
    if not bool(cfg.get('enabled', False)):
        return None
    return LigandSizeShiftWeights(
        ligand_atom_counts=ligand_atom_counts,
        mode=str(cfg.get('mode', 'size_range')),
        min_atoms=float(cfg.get('min_atoms', 4)),
        max_atoms=float(cfg.get('max_atoms', 52)),
        hard_filter=bool(cfg.get('hard_filter', False)),
        target_counts=cfg.get('target_counts'),
        bin_width=float(cfg.get('bin_width', 0.15)),
        kernel_sigma=float(cfg.get('kernel_sigma', 1.0)),
        min_weight=float(cfg.get('min_weight', 0.1)),
        max_weight=float(cfg.get('max_weight', 5.0)),
    )
