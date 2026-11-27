"""Optional feature-level imbalanced regression.

Feature Distribution Smoothing calibrates penultimate representations using
running statistics for affinity bins (Yang et al., ICML 2021).
RankSim regularizes neighborhood rankings in label and feature space
(Gong et al., ICML 2022). The training loop supplies affinity and epoch
context. At inference, smoothing uses the bin of a provisional prediction.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ProgressiveBody(nn.Module):
    """Width-decreasing MLP that stops at the penultimate representation."""

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
        self.net = nn.Sequential(*layers)
        self.out_dim = int(dims[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureDistributionSmoothing(nn.Module):
    """Calibrate penultimate features so rare affinity bins inherit statistics
    from neighbouring bins.

    Running mean / variance are tracked per fixed-width label bin.  A Gaussian
    kernel smooths those statistics across adjacent bins; each sample is then
    whitened by its own bin's running stats and recoloured by the smoothed
    stats.  This is the feature-space counterpart of LDS and is orthogonal to
    loss reweighting.
    """

    def __init__(self, feature_dim: int, training_targets: torch.Tensor,
                 bin_width: float = 0.5, kernel_size: int = 5,
                 kernel_sigma: float = 2.0, momentum: float = 0.9,
                 start_update: int = 0, start_smooth: int = 1,
                 eps: float = 1e-5):
        super().__init__()
        if feature_dim <= 0:
            raise ValueError('feature_dim must be positive')
        if bin_width <= 0:
            raise ValueError('bin_width must be positive')
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError('kernel_size must be a positive odd integer')
        if not (0.0 <= momentum < 1.0):
            raise ValueError('momentum must be in [0, 1)')

        y = torch.as_tensor(training_targets, dtype=torch.float32).view(-1)
        y = y[torch.isfinite(y)]
        if y.numel() == 0:
            raise ValueError('FDS requires finite training targets')

        start = math.floor(float(y.min()) / bin_width) * bin_width
        stop = math.ceil(float(y.max()) / bin_width) * bin_width
        edges = torch.arange(start + bin_width, stop, bin_width,
                             dtype=torch.float32)
        n_bins = int(edges.numel()) + 1

        half = kernel_size // 2
        offsets = torch.arange(-half, half + 1, dtype=torch.float32)
        kernel = torch.exp(-offsets.pow(2) / (2.0 * kernel_sigma ** 2))
        kernel = kernel / kernel.sum()

        self.feature_dim = int(feature_dim)
        self.bin_width = float(bin_width)
        self.momentum = float(momentum)
        self.start_update = int(start_update)
        self.start_smooth = int(start_smooth)
        self.eps = float(eps)
        self.register_buffer('edges', edges)
        self.register_buffer('kernel', kernel)
        self.register_buffer(
            'running_mean', torch.zeros(n_bins, feature_dim))
        self.register_buffer(
            'running_var', torch.ones(n_bins, feature_dim))
        self.register_buffer(
            'running_count', torch.zeros(n_bins))
        self.register_buffer(
            'smoothed_mean', torch.zeros(n_bins, feature_dim))
        self.register_buffer(
            'smoothed_var', torch.ones(n_bins, feature_dim))
        self._smoothed_ready = False
        self.description = (
            f'FDS bins={n_bins} width={bin_width:g} '
            f'kernel={kernel_size}/{kernel_sigma:g} '
            f'momentum={momentum:g} '
            f'update@{start_update} smooth@{start_smooth}')

    def _bin_ids(self, labels: torch.Tensor) -> torch.Tensor:
        edges = self.edges.to(device=labels.device, dtype=labels.dtype)
        return torch.bucketize(labels.view(-1), edges)

    @torch.no_grad()
    def _update_running_stats(self, features: torch.Tensor,
                              bin_ids: torch.Tensor) -> None:
        feats = features.detach().float()
        for b in bin_ids.unique().tolist():
            mask = bin_ids == int(b)
            n = int(mask.sum())
            if n == 0:
                continue
            batch_mean = feats[mask].mean(dim=0)
            batch_var = feats[mask].var(dim=0, unbiased=False).clamp_min(self.eps)
            count = float(self.running_count[b])
            if count <= 0:
                self.running_mean[b] = batch_mean
                self.running_var[b] = batch_var
                self.running_count[b] = float(n)
            else:
                m = self.momentum
                self.running_mean[b] = (
                    m * self.running_mean[b] + (1.0 - m) * batch_mean)
                self.running_var[b] = (
                    m * self.running_var[b] + (1.0 - m) * batch_var)
                self.running_count[b] = count + float(n)

    @torch.no_grad()
    def _refresh_smoothed_stats(self) -> None:
        # Conv1d over the bin axis; each feature channel is independent.
        mean = self.running_mean.transpose(0, 1).unsqueeze(0)  # [1, D, B]
        var = self.running_var.transpose(0, 1).unsqueeze(0)
        pad = self.kernel.numel() // 2
        kernel = self.kernel.view(1, 1, -1).to(mean.dtype)
        # Grouped conv: one kernel shared across feature channels.
        D = mean.size(1)
        kernel_g = kernel.expand(D, 1, -1).contiguous()
        sm = F.conv1d(mean, kernel_g, padding=pad, groups=D)
        sv = F.conv1d(var, kernel_g, padding=pad, groups=D)
        self.smoothed_mean.copy_(sm.squeeze(0).transpose(0, 1))
        self.smoothed_var.copy_(sv.squeeze(0).transpose(0, 1).clamp_min(self.eps))
        self._smoothed_ready = True

    def forward(self, features: torch.Tensor, labels: torch.Tensor,
                epoch: int) -> torch.Tensor:
        if features.dim() != 2:
            raise ValueError(f'FDS expects [B, D], got {tuple(features.shape)}')
        if features.size(1) != self.feature_dim:
            raise ValueError(
                f'FDS feature_dim={self.feature_dim} but got {features.size(1)}')

        bin_ids = self._bin_ids(labels)
        if self.training and epoch >= self.start_update:
            self._update_running_stats(features, bin_ids)
            self._refresh_smoothed_stats()

        if epoch < self.start_smooth or not self._smoothed_ready:
            return features

        # Whiten by running stats of the sample's own bin, recolour by smoothed.
        mean = self.running_mean[bin_ids].to(features.dtype)
        var = self.running_var[bin_ids].to(features.dtype).clamp_min(self.eps)
        sm = self.smoothed_mean[bin_ids].to(features.dtype)
        sv = self.smoothed_var[bin_ids].to(features.dtype).clamp_min(self.eps)
        calibrated = (features - mean) / var.sqrt()
        calibrated = calibrated * sv.sqrt() + sm
        return calibrated


def ranksim_loss(features: torch.Tensor, labels: torch.Tensor,
                 temperature: float = 2.0) -> torch.Tensor:
    """Soft ranking-similarity regulariser.

    Encourages the neighbour ranking of each anchor in feature space to match
    its neighbour ranking in label space.  Implemented as a batch-mean KL
    between the two tempered softmax distributions over off-diagonal pairs
    (a stable surrogate for the ranking-matrix objective in the paper).
    """
    if features.dim() != 2:
        raise ValueError(f'RankSim expects [B, D], got {tuple(features.shape)}')
    y = labels.view(-1).to(features.dtype)
    B = features.size(0)
    if B < 3:
        return features.new_zeros(())
    if temperature <= 0:
        raise ValueError('temperature must be positive')

    # Label-space similarity: nearer affinities → larger score.
    sim_y = -(y.unsqueeze(0) - y.unsqueeze(1)).abs()
    # Feature-space similarity: cosine on L2-normalised penultimate features.
    z = F.normalize(features, p=2, dim=1)
    sim_z = z @ z.T

    eye = torch.eye(B, device=features.device, dtype=torch.bool)
    neg_inf = torch.finfo(features.dtype).min
    sim_y = sim_y.masked_fill(eye, neg_inf)
    sim_z = sim_z.masked_fill(eye, neg_inf)

    log_pz = F.log_softmax(sim_z / temperature, dim=1)
    py = F.softmax(sim_y / temperature, dim=1)
    return F.kl_div(log_pz, py, reduction='batchmean')


class FeatureDirReadout(nn.Module):
    """Progressive smooth-MLP readout with optional FDS and feature export.

    Drop-in replacement for ``ProgressiveAllNodeMLP``.  ``forward`` still
    returns a scalar prediction ``[B, 1]`` so existing fusion call sites stay
    unchanged; the penultimate features are cached on ``last_features`` for
    the training loop to pick up (RankSim) and FDS is applied inside when a
    context has been set via :meth:`set_context`.
    """

    returns_features = True

    def __init__(self, in_dim: int, hidden_dims: List[int],
                 dropout: float = 0.2,
                 fds: Optional[FeatureDistributionSmoothing] = None):
        super().__init__()
        self.body = _ProgressiveBody(in_dim, hidden_dims, dropout=dropout)
        self.regressor = nn.Linear(self.body.out_dim, 1)
        self.fds = fds
        self.feature_dim = self.body.out_dim
        self.last_features: Optional[torch.Tensor] = None
        self._ctx_labels: Optional[torch.Tensor] = None
        self._ctx_epoch: int = 0
        parts = [f'FeatureDirReadout in={in_dim} dims={list(hidden_dims)}']
        if fds is not None:
            parts.append(fds.description)
        self.description = ' | '.join(parts)

    def set_context(self, labels: Optional[torch.Tensor] = None,
                    epoch: int = 0) -> None:
        """Inject batch labels / epoch so FDS can run inside ``forward``."""
        if labels is None:
            self._ctx_labels = None
        else:
            self._ctx_labels = labels.detach().view(-1)
        self._ctx_epoch = int(epoch)

    def clear_context(self) -> None:
        self._ctx_labels = None
        self._ctx_epoch = 0

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.body(x)
        if self.fds is None:
            return feat
        if (self._ctx_labels is not None
                and self._ctx_labels.numel() == feat.size(0)):
            return self.fds(feat, self._ctx_labels, self._ctx_epoch)
        # At inference, select the smoothing bin using a provisional prediction.
        if (not self.training and self.fds._smoothed_ready
                and self._ctx_epoch >= self.fds.start_smooth):
            with torch.no_grad():
                rough = self.regressor(feat).view(-1)
            return self.fds(feat, rough, self._ctx_epoch)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encode(x)
        self.last_features = feat
        return self.regressor(feat)


def build_feature_dir_readout(
    in_dim: int,
    hidden_dims: List[int],
    dropout: float,
    feature_dir_cfg: Optional[Dict] = None,
    training_targets: Optional[torch.Tensor] = None,
) -> FeatureDirReadout:
    """Factory used by the fusion builder when ``feature_dir.enabled``."""
    cfg = dict(feature_dir_cfg or {})
    fds = None
    fds_cfg = dict(cfg.get('fds') or {})
    if bool(fds_cfg.get('enabled', True)):
        if training_targets is None:
            raise ValueError(
                'feature_dir.fds requires training_targets from the current '
                'training split (passed through build_model / fusion factory)')
        feature_dim = int(cfg.get('feature_dim', hidden_dims[-1]))
        if feature_dim != int(hidden_dims[-1]):
            raise ValueError(
                f'feature_dir.feature_dim={feature_dim} must equal the last '
                f'smooth_mlp_dims entry ({hidden_dims[-1]})')
        fds = FeatureDistributionSmoothing(
            feature_dim=feature_dim,
            training_targets=training_targets,
            bin_width=float(fds_cfg.get('bin_width', 0.5)),
            kernel_size=int(fds_cfg.get('kernel_size', 5)),
            kernel_sigma=float(fds_cfg.get('kernel_sigma', 2.0)),
            momentum=float(fds_cfg.get('momentum', 0.9)),
            start_update=int(fds_cfg.get('start_update', 0)),
            start_smooth=int(fds_cfg.get('start_smooth', 1)),
        )
    return FeatureDirReadout(
        in_dim=in_dim, hidden_dims=hidden_dims, dropout=dropout, fds=fds)


def attach_fds_to_model(model: nn.Module, feature_dir_cfg: Optional[Dict],
                        training_targets: torch.Tensor) -> Optional[str]:
    """Attach an FDS module to ``model.fusion.smooth_head`` after data load.

    The fusion factory builds the readout before training labels are known;
    call this once per repeat with the current split's targets.
    Returns a human-readable description, or None if FDS is not applicable.
    """
    cfg = dict(feature_dir_cfg or {})
    if not bool(cfg.get('enabled', False)):
        return None
    fds_cfg = dict(cfg.get('fds') or {})
    if not bool(fds_cfg.get('enabled', True)):
        return None
    fusion = getattr(model, 'fusion', None)
    head = getattr(fusion, 'smooth_head', None)
    if not isinstance(head, FeatureDirReadout):
        return None
    feature_dim = int(cfg.get('feature_dim', head.feature_dim))
    if feature_dim != head.feature_dim:
        raise ValueError(
            f'feature_dir.feature_dim={feature_dim} != readout '
            f'feature_dim={head.feature_dim}')
    head.fds = FeatureDistributionSmoothing(
        feature_dim=feature_dim,
        training_targets=training_targets,
        bin_width=float(fds_cfg.get('bin_width', 0.5)),
        kernel_size=int(fds_cfg.get('kernel_size', 5)),
        kernel_sigma=float(fds_cfg.get('kernel_sigma', 2.0)),
        momentum=float(fds_cfg.get('momentum', 0.9)),
        start_update=int(fds_cfg.get('start_update', 0)),
        start_smooth=int(fds_cfg.get('start_smooth', 1)),
    )
    head.description = (
        f'FeatureDirReadout dim={head.feature_dim} | {head.fds.description}')
    return head.description


def get_feature_dir_readout(model: nn.Module) -> Optional[FeatureDirReadout]:
    fusion = getattr(model, 'fusion', None)
    head = getattr(fusion, 'smooth_head', None)
    return head if isinstance(head, FeatureDirReadout) else None


def ranksim_config(feature_dir_cfg: Optional[Dict]) -> Tuple[bool, float, float]:
    """Return ``(enabled, weight, temperature)`` from the yaml sub-dict."""
    cfg = dict(feature_dir_cfg or {})
    rs = dict(cfg.get('ranksim') or {})
    enabled = bool(cfg.get('enabled', False)) and bool(rs.get('enabled', True))
    return enabled, float(rs.get('weight', 0.1)), float(rs.get('temperature', 2.0))
