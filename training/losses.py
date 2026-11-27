"""Configurable training loss factory."""
from __future__ import annotations

import math
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn


def _pairwise_margin_rank_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    margin: float = 0.5,
    same_y_eps: float = 1e-3,
) -> torch.Tensor:
    """Full-batch pairwise margin ranking loss (hinge form).

    For any pair (i, j) within the batch (i != j, |y_i - y_j| > same_y_eps):
        l_{i,j} = max(0, margin - sign(y_i - y_j) * (pred_i - pred_j))
    Returns the mean over all valid pairs.

    Parameters
    ----------
    pred       : predictions of shape [B, 1] or [B]
    target     : ground truth of shape [B, 1] or [B]
    margin     : hinge margin (default 0.5)
    same_y_eps : exclude pairs with |y_i - y_j| < same_y_eps to guard against label noise

    Returns
    -------
    A scalar tensor; returns 0 if there is no valid pair.
    """
    p = pred.view(-1)
    t = target.view(-1)
    B = p.size(0)
    if B < 2:
        return torch.zeros((), device=p.device, dtype=p.dtype)

    diff_p = p.unsqueeze(0) - p.unsqueeze(1)
    diff_t = t.unsqueeze(0) - t.unsqueeze(1)
    sign = torch.sign(diff_t)
    # Ignore self-pairs and near-tied labels.
    mask = (diff_t.abs() > same_y_eps).to(p.dtype)
    hinge = torch.relu(margin - sign * diff_p)
    valid = mask.sum().clamp(min=1.0)
    return (hinge * mask).sum() / valid


class _MSEWithRank(nn.Module):
    """nn.Module wrapper that also exposes last_mse / last_rank for training-log monitoring.

    forward returns (mse + rank_weight * rank); during backward the two gradient
    paths are automatically weighted by rank_weight.
    """

    def __init__(self, rank_weight: float, margin: float, same_y_eps: float):
        super().__init__()
        self.rank_weight = float(rank_weight)
        self.margin = float(margin)
        self.same_y_eps = float(same_y_eps)
        self.mse = nn.MSELoss()
        self.last_mse: Optional[float] = None
        self.last_rank: Optional[float] = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l_mse = self.mse(pred, target)
        l_rank = _pairwise_margin_rank_loss(
            pred, target, margin=self.margin, same_y_eps=self.same_y_eps,
        )
        with torch.no_grad():
            self.last_mse = float(l_mse.detach())
            self.last_rank = float(l_rank.detach())
        return l_mse + self.rank_weight * l_rank


class _MSEWithPearson(nn.Module):
    """MSE plus a batch Pearson-correlation penalty (1 - r).

    HBGSA-style hybrid objective: keep absolute-error pressure from MSE while
    encouraging linear correlation within each training batch.  Falls back to
    pure MSE when a batch has fewer than two finite samples or zero variance.
    """

    def __init__(self, pearson_weight: float = 0.10, eps: float = 1e-8):
        super().__init__()
        if pearson_weight < 0:
            raise ValueError('pearson_weight must be non-negative')
        self.pearson_weight = float(pearson_weight)
        self.eps = float(eps)
        self.mse = nn.MSELoss()
        self.last_mse: Optional[float] = None
        self.last_pearson: Optional[float] = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = pred.view(-1)
        t = target.view(-1)
        l_mse = self.mse(p, t)
        if p.numel() < 2:
            corr_term = p.new_zeros(())
            r_val = 0.0
        else:
            p_c = p - p.mean()
            t_c = t - t.mean()
            denom = torch.sqrt((p_c * p_c).sum() * (t_c * t_c).sum()).clamp_min(
                self.eps)
            r = (p_c * t_c).sum() / denom
            corr_term = 1.0 - r
            r_val = float(r.detach())
        with torch.no_grad():
            self.last_mse = float(l_mse.detach())
            self.last_pearson = r_val
        return l_mse + self.pearson_weight * corr_term


class _MSEExtremeWeighted(nn.Module):
    """Upweight labels far from the training-set median to fight range collapse.

    weight = clip(1 + (|y-median|/IQR)^power, min_weight, max_weight), then
    renormalized so the training-set mean weight equals 1.
    """

    def __init__(self, training_targets: torch.Tensor, power: float = 1.0,
                 max_weight: float = 4.0, min_weight: float = 1.0):
        super().__init__()
        y = torch.as_tensor(training_targets, dtype=torch.float32).view(-1).cpu()
        y = y[torch.isfinite(y)]
        if y.numel() == 0:
            raise ValueError('mse_extreme_weighted requires finite targets')
        if power <= 0:
            raise ValueError('power must be > 0')
        if not (0.0 < min_weight <= 1.0 <= max_weight):
            raise ValueError('need 0 < min_weight <= 1 <= max_weight')
        median = float(y.median())
        q75 = float(y.quantile(0.75))
        q25 = float(y.quantile(0.25))
        iqr = max(q75 - q25, 1e-3)
        raw = 1.0 + ((y - median).abs() / iqr).pow(float(power))
        raw = raw.clamp(float(min_weight), float(max_weight))
        raw = raw / raw.mean().clamp_min(1e-8)
        self.median = median
        self.iqr = iqr
        self.power = float(power)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.mse = nn.MSELoss(reduction='none')
        self.description = (
            f'mse_extreme_weighted median={median:.3f} iqr={iqr:.3f} '
            f'power={power:g} weight_clip=[{min_weight:g},{max_weight:g}]')

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = pred.view(-1)
        t = target.view(-1)
        w = 1.0 + ((t - self.median).abs() / self.iqr).pow(self.power)
        w = w.clamp(self.min_weight, self.max_weight)
        # Normalize weights to preserve the effective learning-rate scale.
        w = w / w.mean().clamp_min(1e-8)
        return (w * self.mse(p, t)).mean()


class _MSEWithSlope(nn.Module):
    """MSE plus a penalty that pushes batch regression slope toward 1.

    The slope penalty discourages compressing the predicted affinity range.
    """

    def __init__(self, slope_weight: float = 0.20, target_slope: float = 1.0,
                 eps: float = 1e-8):
        super().__init__()
        if slope_weight < 0:
            raise ValueError('slope_weight must be non-negative')
        self.slope_weight = float(slope_weight)
        self.target_slope = float(target_slope)
        self.eps = float(eps)
        self.mse = nn.MSELoss()
        self.last_mse: Optional[float] = None
        self.last_slope: Optional[float] = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = pred.view(-1)
        t = target.view(-1)
        l_mse = self.mse(p, t)
        if p.numel() < 2:
            slope = p.new_tensor(self.target_slope)
        else:
            t_c = t - t.mean()
            p_c = p - p.mean()
            var_t = (t_c * t_c).sum().clamp_min(self.eps)
            slope = (p_c * t_c).sum() / var_t
        l_slope = (slope - self.target_slope).pow(2)
        with torch.no_grad():
            self.last_mse = float(l_mse.detach())
            self.last_slope = float(slope.detach())
        return l_mse + self.slope_weight * l_slope


def _bounded_normalized_weights(
    raw_weights: torch.Tensor,
    counts: torch.Tensor,
    min_weight: float,
    max_weight: float,
) -> torch.Tensor:
    """Scale ``raw_weights`` so clipped weights satisfy E_train[w] = 1.

    Scaling *after* clipping would violate ``max_weight`` (e.g. a requested cap
    of 3 became 4.08 on base split 0), so the bounded normalization is solved
    directly by monotone bisection on the pre-clip scale.
    """
    lo, hi = 0.0, 1.0
    total = counts.sum()

    def weighted_mean(scale: float) -> torch.Tensor:
        bounded = (raw_weights * scale).clamp(min_weight, max_weight)
        return (bounded * counts).sum() / total

    while float(weighted_mean(hi)) < 1.0:
        hi *= 2.0
    for _ in range(64):
        mid = (lo + hi) / 2.0
        if float(weighted_mean(mid)) < 1.0:
            lo = mid
        else:
            hi = mid
    return (raw_weights * hi).clamp(min_weight, max_weight)


def _label_bin_edges(
    y: torch.Tensor, bin_width: float
) -> torch.Tensor:
    """Internal fixed-width bin boundaries covering the observed label range."""
    start = math.floor(float(y.min()) / bin_width) * bin_width
    stop = math.ceil(float(y.max()) / bin_width) * bin_width
    # Internal boundaries define one more bin than their count.
    return torch.arange(start + bin_width, stop, bin_width, dtype=torch.float32)


class _LDSWeightedMSE(nn.Module):
    """MSE reweighted by Label Distribution Smoothing (Yang et al., ICML 2021).

    Unlike plain inverse-frequency binning, LDS first convolves the empirical
    label histogram with a symmetric Gaussian kernel, so neighbouring affinity
    bins share statistical support before the inverse-density weight is taken.
    This is the intended fix for the sparse high/low-affinity tails that drive
    the observed ŷ~y slope collapse.
    """

    def __init__(self, training_targets: torch.Tensor, bin_width: float = 0.5,
                 kernel_size: int = 5, kernel_sigma: float = 2.0,
                 power: float = 0.5, min_weight: float = 0.5,
                 max_weight: float = 4.0):
        super().__init__()
        y = torch.as_tensor(training_targets, dtype=torch.float32).view(-1).cpu()
        y = y[torch.isfinite(y)]
        if y.numel() == 0:
            raise ValueError('lds_weighted_mse requires finite training targets')
        if bin_width <= 0 or not (0.0 < power <= 1.0):
            raise ValueError('bin_width must be >0 and power must be in (0,1]')
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError('kernel_size must be a positive odd integer')
        if kernel_sigma <= 0:
            raise ValueError('kernel_sigma must be positive')
        if (min_weight <= 0 or max_weight < min_weight
                or not min_weight <= 1.0 <= max_weight):
            raise ValueError(
                'weight bounds must satisfy 0 < min_weight <= 1 <= max_weight '
                'so E_train[w] can be normalized to one')

        edges = _label_bin_edges(y, bin_width)
        bin_ids = torch.bucketize(y, edges)
        counts = torch.bincount(bin_ids, minlength=edges.numel() + 1).float()

        half = kernel_size // 2
        offsets = torch.arange(-half, half + 1, dtype=torch.float32)
        kernel = torch.exp(-offsets.pow(2) / (2.0 * kernel_sigma ** 2))
        kernel = kernel / kernel.sum()
        # Zero padding matches scipy.ndimage.convolve1d in constant mode.
        smoothed = torch.nn.functional.conv1d(
            counts.view(1, 1, -1), kernel.view(1, 1, -1), padding=half
        ).view(-1)

        nonempty = counts > 0
        reference = smoothed[nonempty].mean()
        raw_weights = (reference / smoothed.clamp_min(1e-6)).pow(float(power))
        weights = _bounded_normalized_weights(
            raw_weights, counts, float(min_weight), float(max_weight))

        self.register_buffer('edges', edges, persistent=True)
        self.register_buffer('weights', weights, persistent=True)
        self.register_buffer('counts', counts, persistent=True)
        self.description = (
            f'lds_weighted_mse bins={counts.numel()} width={bin_width:g} '
            f'kernel={kernel_size}/{kernel_sigma:g} power={power:g} '
            f'normalized_weight_range='
            f'[{float(weights[nonempty].min()):.3f},'
            f'{float(weights[nonempty].max()):.3f}]')

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = pred.view(-1)
        t = target.view(-1)
        edges = self.edges.to(device=t.device, dtype=t.dtype)
        weights = self.weights.to(device=t.device, dtype=t.dtype)
        sample_weight = weights[torch.bucketize(t, edges)]
        return (sample_weight * (p - t).pow(2)).mean()


class _BalancedMSE(nn.Module):
    """Balanced MSE, batch-based BMC form (Ren et al., CVPR 2022 oral).

    Treats regression as a softmax over the labels present in the batch:
    logits_{ij} = -(pred_i - y_j)^2 / (2 sigma^2) with the diagonal as target.
    This restores calibration across the whole label range instead of merely
    rescaling per-sample gradients, so it attacks the prediction-range
    compression rather than its symptom.

    ``noise_sigma`` is learnable; the training runner adds it to the optimizer
    as a separate parameter group when ``sigma_lr`` > 0.
    """

    def __init__(self, init_noise_sigma: float = 1.0, sigma_lr: float = 1e-2,
                 min_noise_sigma: float = 0.05):
        super().__init__()
        if init_noise_sigma <= 0:
            raise ValueError('init_noise_sigma must be positive')
        if min_noise_sigma <= 0:
            raise ValueError('min_noise_sigma must be positive')
        self.noise_sigma = nn.Parameter(
            torch.tensor(float(init_noise_sigma), dtype=torch.float32))
        self.sigma_lr = float(sigma_lr)
        self.min_noise_sigma = float(min_noise_sigma)
        self.last_mse: Optional[float] = None
        self.last_sigma: Optional[float] = None
        self.description = (
            f'balanced_mse (BMC) init_sigma={init_noise_sigma:g} '
            f'sigma_lr={sigma_lr:g}')

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = pred.view(-1, 1)
        t = target.view(-1, 1).to(p.dtype)
        sigma = self.noise_sigma.to(p.dtype).clamp_min(self.min_noise_sigma)
        noise_var = sigma.pow(2)
        if p.size(0) < 2:
            return torch.nn.functional.mse_loss(p.view(-1), t.view(-1))
        # The diagonal pairs each prediction with its own label.
        logits = -(p - t.T).pow(2) / (2.0 * noise_var)
        labels = torch.arange(p.size(0), device=p.device)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        # Keep the gradient scale comparable to ordinary MSE.
        loss = loss * (2.0 * noise_var).detach()
        with torch.no_grad():
            self.last_mse = float(
                torch.nn.functional.mse_loss(p.view(-1), t.view(-1)).detach())
            self.last_sigma = float(sigma.detach())
        return loss


class _HistogramBalancedMSE(nn.Module):
    """MSE reweighted by fixed-width bins estimated from training labels only.

    Square-root inverse frequency avoids the extreme weights of full inverse
    frequency.  Weights are clipped and then normalized so their training-set
    expectation is one, keeping the effective learning-rate scale comparable
    with ordinary MSE.
    """

    def __init__(self, training_targets: torch.Tensor, bin_width: float = 1.0,
                 power: float = 0.5, min_weight: float = 0.5,
                 max_weight: float = 3.0):
        super().__init__()
        y = torch.as_tensor(training_targets, dtype=torch.float32).view(-1).cpu()
        y = y[torch.isfinite(y)]
        if y.numel() == 0:
            raise ValueError('histogram_balanced_mse requires finite training targets')
        if bin_width <= 0 or not (0.0 < power <= 1.0):
            raise ValueError('bin_width must be >0 and power must be in (0,1]')
        if (min_weight <= 0 or max_weight < min_weight
                or not min_weight <= 1.0 <= max_weight):
            raise ValueError(
                'histogram weight bounds must satisfy 0 < min_weight <= 1 '
                '<= max_weight so E_train[w] can be normalized to one')

        edges = _label_bin_edges(y, bin_width)
        bin_ids = torch.bucketize(y, edges)
        counts = torch.bincount(bin_ids, minlength=edges.numel() + 1).float()
        nonempty = counts > 0
        reference = counts[nonempty].mean()
        raw_weights = (reference / counts.clamp_min(1.0)).pow(float(power))
        weights = _bounded_normalized_weights(
            raw_weights, counts, float(min_weight), float(max_weight))
        self.register_buffer('edges', edges, persistent=True)
        self.register_buffer('weights', weights, persistent=True)
        self.register_buffer('counts', counts, persistent=True)
        self.description = (
            f'histogram_balanced_mse bins={counts.numel()} width={bin_width:g} '
            f'power={power:g} normalized_weight_range='
            f'[{float(weights[nonempty].min()):.3f},'
            f'{float(weights[nonempty].max()):.3f}]')

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = pred.view(-1)
        t = target.view(-1)
        edges = self.edges.to(device=t.device, dtype=t.dtype)
        weights = self.weights.to(device=t.device, dtype=t.dtype)
        sample_weight = weights[torch.bucketize(t, edges)]
        return (sample_weight * (p - t).pow(2)).mean()


def make_loss_fn(
    cfg: Optional[Dict] = None,
    training_targets: Optional[torch.Tensor] = None,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build a loss function from the yaml configuration.

    yaml field location:
        training:
          loss:
            type: mse_only | mse_with_rank | histogram_balanced_mse | mse_with_pearson | mse_extreme_weighted | mse_with_slope | lds_weighted_mse | balanced_mse
            rank_weight: 0.10           # only used by mse_with_rank
            margin: 0.5                 # only used by mse_with_rank
            same_y_eps: 1.0e-3          # only used by mse_with_rank
            pearson_weight: 0.10        # only used by mse_with_pearson
            slope_weight: 0.20          # only used by mse_with_slope
            target_slope: 1.0           # only used by mse_with_slope
            power: 1.0                  # extreme / histogram / lds
            min_weight: 1.0             # extreme / histogram / lds
            max_weight: 4.0             # extreme / histogram / lds
            bin_width: 1.0              # histogram / lds
            kernel_size: 5              # only used by lds_weighted_mse
            kernel_sigma: 2.0           # only used by lds_weighted_mse
            init_noise_sigma: 1.0       # only used by balanced_mse
            sigma_lr: 1.0e-2            # only used by balanced_mse

    Parameters
    ----------
    cfg : the training.loss sub-dict; may be None / an empty dict (falls back to mse_only)
    training_targets : labels from the current training split only; required by
        histogram_balanced_mse / mse_extreme_weighted / lds_weighted_mse and
        ignored by the other loss types

    Returns
    -------
    callable(pred, target) -> scalar tensor

    Notes
    -----
    - The returned object additionally carries .last_mse / .last_rank attributes
      (only for mse_with_rank), usable for dual-axis training-log monitoring.
      These two attributes do not exist on the mse_only path.
    """
    cfg = dict(cfg or {})
    loss_type = str(cfg.get('type', 'mse_only')).strip()

    if loss_type == 'mse_only':
        return nn.MSELoss()

    if loss_type == 'mse_with_rank':
        return _MSEWithRank(
            rank_weight=float(cfg.get('rank_weight', 0.10)),
            margin=float(cfg.get('margin', 0.5)),
            same_y_eps=float(cfg.get('same_y_eps', 1.0e-3)),
        )

    if loss_type == 'histogram_balanced_mse':
        if training_targets is None:
            raise ValueError(
                'histogram_balanced_mse requires training_targets from the '
                'current training split')
        return _HistogramBalancedMSE(
            training_targets=training_targets,
            bin_width=float(cfg.get('bin_width', 1.0)),
            power=float(cfg.get('power', 0.5)),
            min_weight=float(cfg.get('min_weight', 0.5)),
            max_weight=float(cfg.get('max_weight', 3.0)),
        )

    if loss_type == 'mse_with_pearson':
        return _MSEWithPearson(
            pearson_weight=float(cfg.get('pearson_weight', 0.10)),
            eps=float(cfg.get('eps', 1.0e-8)),
        )

    if loss_type == 'mse_extreme_weighted':
        if training_targets is None:
            raise ValueError(
                'mse_extreme_weighted requires training_targets from the '
                'current training split')
        return _MSEExtremeWeighted(
            training_targets=training_targets,
            power=float(cfg.get('power', 1.0)),
            max_weight=float(cfg.get('max_weight', 4.0)),
            min_weight=float(cfg.get('min_weight', 1.0)),
        )

    if loss_type == 'mse_with_slope':
        return _MSEWithSlope(
            slope_weight=float(cfg.get('slope_weight', 0.20)),
            target_slope=float(cfg.get('target_slope', 1.0)),
            eps=float(cfg.get('eps', 1.0e-8)),
        )

    if loss_type == 'lds_weighted_mse':
        if training_targets is None:
            raise ValueError(
                'lds_weighted_mse requires training_targets from the current '
                'training split')
        return _LDSWeightedMSE(
            training_targets=training_targets,
            bin_width=float(cfg.get('bin_width', 0.5)),
            kernel_size=int(cfg.get('kernel_size', 5)),
            kernel_sigma=float(cfg.get('kernel_sigma', 2.0)),
            power=float(cfg.get('power', 0.5)),
            min_weight=float(cfg.get('min_weight', 0.5)),
            max_weight=float(cfg.get('max_weight', 4.0)),
        )

    if loss_type == 'balanced_mse':
        return _BalancedMSE(
            init_noise_sigma=float(cfg.get('init_noise_sigma', 1.0)),
            sigma_lr=float(cfg.get('sigma_lr', 1.0e-2)),
            min_noise_sigma=float(cfg.get('min_noise_sigma', 0.05)),
        )

    raise ValueError(
        f'Unknown training.loss.type={loss_type!r}. '
        f'Valid: mse_only | mse_with_rank | histogram_balanced_mse | '
        f'mse_with_pearson | mse_extreme_weighted | mse_with_slope | '
        f'lds_weighted_mse | balanced_mse'
    )
