"""General training runner invoked by the five split entry scripts."""
from __future__ import annotations

import glob
import fcntl
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data import (  # noqa: E402
    PDBbindESMDataset, esm_collate_fn, LengthBucketSampler,
)
from models import build_model, setup_struct_branch  # noqa: E402
from training.prediction_reports import (  # noqa: E402
    aggregate_prediction_tables,
    write_prediction_table,
)
from utils import Evaluate, Logger, cal_final_results, fmt_secs, mse, set_seed  # noqa: E402


@dataclass
class SplitSpec:
    """Describes the training workflow for one data split."""
    name: str
    train_template: str
    valid_template: str
    # Each test specification pairs a reporting tag with a dataset template.
    test_specs: List[tuple]
    n_repeats: int = 5

    def train_name(self, repeat: int) -> str:
        return self.train_template.format(split=self.name, repeat=repeat)

    def valid_name(self, repeat: int) -> str:
        return self.valid_template.format(split=self.name, repeat=repeat)

    def test_name(self, template: str, repeat: int) -> str:
        # Resolve fixed holdout templates even when they have no repeat field.
        if '{split}' in template or '{repeat}' in template:
            return template.format(split=self.name, repeat=repeat)
        return template


SPLIT_SPECS: Dict[str, 'SplitSpec'] = {
    'base': SplitSpec(
        name='base',
        train_template='train_{split}_{repeat}',
        valid_template='valid_{split}_{repeat}',
        test_specs=[('CASF', 'CASF-2016'), ('CSAR', 'CSAR-HiQ')],
        n_repeats=5,
    ),
    'random': SplitSpec(
        name='random',
        train_template='train_{split}_{repeat}',
        valid_template='valid_{split}_{repeat}',
        test_specs=[('test', 'test_{split}_{repeat}')],
        n_repeats=5,
    ),
    'holdout': SplitSpec(
        name='holdout',
        train_template='train_{split}',
        valid_template='valid_{split}',
        test_specs=[('test', 'test_{split}')],
        n_repeats=5,
    ),
    'scaffold': SplitSpec(
        name='scaffold',
        train_template='train_{split}_{repeat}',
        valid_template='valid_{split}_{repeat}',
        test_specs=[('test', 'test_{split}_{repeat}')],
        n_repeats=5,
    ),
    'seq_identity': SplitSpec(
        name='seq_identity',
        train_template='train_{split}_{repeat}',
        valid_template='valid_{split}_{repeat}',
        test_specs=[('test', 'test_{split}_{repeat}')],
        n_repeats=5,
    ),
    # Optional GEMS CleanSplit cross-validation.
    'gems_5fold': SplitSpec(
        name='gems_5fold',
        train_template='train_{split}_{repeat}',
        valid_template='valid_{split}_{repeat}',
        test_specs=[('CASF', 'CASF-2016')],
        n_repeats=5,
    ),
    # Optional EHIGN-style fixed development split with three test years.
    'paper': SplitSpec(
        name='paper',
        train_template='train_paper',
        valid_template='valid_paper',
        test_specs=[
            ('test2013', 'test2013'),
            ('test2016', 'test2016'),
            ('test2019', 'test2019'),
        ],
        n_repeats=3,
    ),
}


def get_split_spec(name: str) -> 'SplitSpec':
    if name not in SPLIT_SPECS:
        raise KeyError(f'unknown split {name!r}. Available: {list(SPLIT_SPECS.keys())}')
    return SPLIT_SPECS[name]


def _config_for_repeat(config: dict, split: str, repeat: int) -> dict:
    """Resolve train-only scaler paths for {split} and {repeat}; {fold} aliases {repeat}."""
    pli_cfg = config.get('plitext') or {}
    scaler_path = pli_cfg.get('scaler_path')
    if not isinstance(scaler_path, str) or '{' not in scaler_path:
        return config
    try:
        resolved = scaler_path.format(
            split=split, repeat=repeat, fold=repeat)
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(
            f'invalid plitext.scaler_path template {scaler_path!r}; '
            'supported placeholders are {split}, {repeat}, and {fold}'
        ) from exc
    resolved_check = (
        resolved if os.path.isabs(resolved) else os.path.join(_ROOT, resolved))
    if not os.path.isfile(resolved_check):
        # Require the repeat-specific scaler to avoid using global statistics.
        raise FileNotFoundError(
            f'resolved train-only scaler does not exist: {resolved_check}')

    repeat_config = dict(config)
    repeat_pli_cfg = dict(pli_cfg)
    repeat_pli_cfg['scaler_path'] = resolved
    repeat_config['plitext'] = repeat_pli_cfg
    return repeat_config


def _resolve_amp_dtype(name) -> Optional[torch.dtype]:
    """Parse the configured autocast dtype; an empty value disables autocast."""
    if name is None:
        return None
    s = str(name).strip().lower()
    if s in ('', 'none', 'null', 'off', 'disable', 'false'):
        return None
    if s in ('bf16', 'bfloat16'):
        return torch.bfloat16
    if s in ('fp16', 'float16', 'half'):
        return torch.float16
    raise ValueError(f'unknown amp_dtype: {name!r}. '
                     f'Supported: bf16/bfloat16/fp16/float16/none')


def _unpack_fusion_output(output):
    """Fusion head may return (main, [aux1, aux2, ...]) for deep supervision, or a single tensor.

    Returns (main_pred, aux_list). aux_list is None when there is no deep supervision.
    """
    if isinstance(output, dict):
        if 'prediction' not in output:
            raise KeyError('fusion output dict must contain prediction')
        return output['prediction'], output
    if isinstance(output, (tuple, list)):
        return output[0], (output[1] if len(output) > 1 else None)
    return output, None


def _fusion_supervised_loss(main_pred, aux_outs, targets, loss_fn,
                            aux_loss_weight: float = 0.0):
    """Compute ordinary or EHIGN bidirectional supervised loss."""
    loss_mode = (
        aux_outs.get('loss_mode')
        if isinstance(aux_outs, dict) else None)
    if loss_mode in ('ehign_bidirectional', 'main_with_ehign_aux'):
        directional = aux_outs.get('directional_predictions') or []
        if len(directional) != 2:
            raise ValueError('EHIGN loss requires exactly two directional predictions')
        pred_to_shell, pred_from_shell = directional
        directional_loss = (
            loss_fn(pred_to_shell, targets)
            + loss_fn(pred_from_shell, targets)
            + loss_fn(pred_to_shell, pred_from_shell)
        ) / 3.0
        if loss_mode == 'ehign_bidirectional':
            return directional_loss
        return (
            loss_fn(main_pred, targets)
            + float(aux_loss_weight) * directional_loss)

    loss = loss_fn(main_pred, targets)
    if aux_outs and aux_loss_weight > 0.0:
        for a in aux_outs:
            loss = loss + aux_loss_weight * loss_fn(a, targets)
    return loss


def train_one_epoch(model, device, loader, optimizer, loss_fn,
                    epoch: int, log_interval: int, logger: Logger,
                    grad_clip: Optional[float] = 1.0,
                    amp_dtype: Optional[torch.dtype] = None,
                    grad_scaler: Optional[torch.cuda.amp.GradScaler] = None,
                    trainable_params: Optional[List[torch.nn.Parameter]] = None,
                    aux_loss_weight: float = 0.0,
                    ema=None,
                    ranksim_weight: float = 0.0,
                    ranksim_temperature: float = 2.0):
    """Train for one epoch and return loss, RMSE, Pearson r, and throughput."""
    from models.feature_dir import get_feature_dir_readout, ranksim_loss

    model.train()
    # Keep the frozen backbone in evaluation mode while training adapters.
    prot_bb = getattr(model.prot, 'esm', None) or getattr(model.prot, 'backbone', None)
    if prot_bb is not None and (
            not getattr(model.prot, 'has_trainable_esm', False)
            or getattr(model.prot, 'adapter_only', False)):
        prot_bb.eval()
    lig_lm = getattr(model.lig, 'lm', None)
    if lig_lm is not None and getattr(model.lig, 'adapter_only', False):
        lig_lm.eval()

    n_samples = len(loader.dataset)
    n_batches = len(loader)
    running_sum = 0.0
    running_n = 0
    # Keep metric accumulators on the GPU to avoid per-batch transfers.
    pred_chunks: List[torch.Tensor] = []
    label_chunks: List[torch.Tensor] = []
    pred_to_shell_chunks: List[torch.Tensor] = []
    pred_from_shell_chunks: List[torch.Tensor] = []

    use_amp = amp_dtype is not None and device.type == 'cuda'
    use_scaler = use_amp and amp_dtype == torch.float16 and grad_scaler is not None

    if trainable_params is None:
        trainable_params = [p for p in model.parameters() if p.requires_grad]

    dir_head = get_feature_dir_readout(model)
    use_ranksim = bool(ranksim_weight > 0.0 and dir_head is not None)

    t_start = time.time()
    for batch_idx, (pyg_batch, seqs, smiles, pdb_codes) in enumerate(loader):
        pyg_batch = pyg_batch.to(device, non_blocking=True)
        targets = pyg_batch.y.view(-1, 1).float().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if dir_head is not None:
            dir_head.set_context(labels=targets, epoch=epoch)

        if use_amp:
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                output = model(pyg_batch, seqs, smiles, pdb_codes)
                output, aux_outs = _unpack_fusion_output(output)
                loss = _fusion_supervised_loss(
                    output, aux_outs, targets, loss_fn, aux_loss_weight)
                if use_ranksim and dir_head.last_features is not None:
                    loss = loss + ranksim_weight * ranksim_loss(
                        dir_head.last_features, targets,
                        temperature=ranksim_temperature)
        else:
            output = model(pyg_batch, seqs, smiles, pdb_codes)
            output, aux_outs = _unpack_fusion_output(output)
            loss = _fusion_supervised_loss(
                output, aux_outs, targets, loss_fn, aux_loss_weight)
            if use_ranksim and dir_head.last_features is not None:
                loss = loss + ranksim_weight * ranksim_loss(
                    dir_head.last_features, targets,
                    temperature=ranksim_temperature)

        if use_scaler:
            grad_scaler.scale(loss).backward()
            if grad_clip is not None:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(grad_clip))
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(grad_clip))
            optimizer.step()

        if ema is not None and epoch >= ema.start_epoch:
            ema.update(model)

        bsz = targets.size(0)
        running_sum += loss.item() * bsz
        running_n += bsz
        pred_chunks.append(output.detach().float().view(-1))
        label_chunks.append(targets.detach().float().view(-1))

        if batch_idx % log_interval == 0 or batch_idx == n_batches - 1:
            # A batch sampler can leave DataLoader.batch_size unset.
            seen = running_n
            pct = 100. * (batch_idx + 1) / n_batches
            elapsed = time.time() - t_start
            sps = running_n / max(elapsed, 1e-6)
            remaining_batches = n_batches - (batch_idx + 1)
            eta_epoch = elapsed * remaining_batches / max(batch_idx + 1, 1)
            preds_so_far = torch.cat(pred_chunks).cpu().numpy()
            labels_so_far = torch.cat(label_chunks).cpu().numpy()
            tr_rmse = float(np.sqrt(((preds_so_far - labels_so_far) ** 2).mean()))
            if preds_so_far.std() > 1e-8 and labels_so_far.std() > 1e-8:
                tr_r = float(np.corrcoef(preds_so_far, labels_so_far)[0, 1])
            else:
                tr_r = float('nan')

            logger.log(
                f'  [epoch {epoch:>3}] batch {batch_idx + 1:>4}/{n_batches} '
                f'({seen:>6}/{n_samples} | {pct:5.1f}%)  '
                f'batch_loss={loss.item():.4f}  avg_loss={running_sum / max(running_n, 1):.4f}  '
                f'tr_RMSE={tr_rmse:.4f}  tr_r={tr_r:.4f}  '
                f'speed={sps:.0f} sps  eta(epoch)={fmt_secs(eta_epoch)}',
                to_terminal=False,
            )

    avg_loss = running_sum / max(running_n, 1)
    if pred_chunks:
        preds = torch.cat(pred_chunks).cpu().numpy()
        labels = torch.cat(label_chunks).cpu().numpy()
        tr_rmse = float(np.sqrt(((preds - labels) ** 2).mean()))
        if preds.std() > 1e-8 and labels.std() > 1e-8:
            tr_r = float(np.corrcoef(preds, labels)[0, 1])
        else:
            tr_r = float('nan')
    else:
        tr_rmse = float('nan')
        tr_r = float('nan')
    sps = running_n / max(time.time() - t_start, 1e-6)
    if dir_head is not None:
        dir_head.clear_context()
    return avg_loss, tr_rmse, tr_r, sps


@torch.no_grad()
def predict(model, device, loader, amp_dtype: Optional[torch.dtype] = None,
            return_details: bool = False,
            feature_dir_epoch: int = 10_000):
    """Evaluate without labels for feature calibration; return predictions and targets."""
    from models.feature_dir import get_feature_dir_readout

    model.eval()
    dir_head = get_feature_dir_readout(model)
    if dir_head is not None:
        # Use provisional predictions to select FDS bins without test labels.
        dir_head.set_context(labels=None, epoch=feature_dir_epoch)
    use_amp = amp_dtype is not None and device.type == 'cuda'
    pred_chunks: List[torch.Tensor] = []
    label_chunks: List[torch.Tensor] = []
    pred_to_shell_chunks: List[torch.Tensor] = []
    pred_from_shell_chunks: List[torch.Tensor] = []
    pdb_code_rows: List[str] = []
    protein_lengths: List[int] = []
    ligand_atom_counts: List[int] = []
    total_contacts: List[float] = []
    near_contact_fractions: List[float] = []
    nonempty_shell_counts: List[int] = []
    struct_store = getattr(getattr(model, 'plitext', None), 'store', None)
    for pyg_batch, seqs, smiles, pdb_codes in loader:
        if return_details:
            pdb_code_rows.extend(str(x or '').lower().strip() for x in pdb_codes)
            protein_lengths.extend(len(str(x or '').replace(' ', '')) for x in seqs)
            c_size = getattr(pyg_batch, 'c_size', None)
            if c_size is None:
                ligand_atom_counts.extend([-1] * len(pdb_codes))
            else:
                ligand_atom_counts.extend(
                    int(x) for x in c_size.view(-1).detach().cpu().tolist()
                )
            if struct_store is None:
                total_contacts.extend([float('nan')] * len(pdb_codes))
                near_contact_fractions.extend([float('nan')] * len(pdb_codes))
                nonempty_shell_counts.extend([-1] * len(pdb_codes))
            else:
                raw_batch = struct_store.lookup_batch(list(pdb_codes))
                n_pairs = int(getattr(model.plitext, 'n_pairs', 168))
                n_shells = int(getattr(model.plitext, 'n_shells', 60))
                raw_batch = raw_batch.reshape(-1, n_pairs, n_shells)
                for raw in raw_batch:
                    shell_totals = raw.sum(axis=0)
                    total = float(shell_totals.sum())
                    # Use the first nine shells for the near-contact diagnostic.
                    near = float(shell_totals[:9].sum())
                    total_contacts.append(total)
                    near_contact_fractions.append(near / total if total > 0 else 0.0)
                    nonempty_shell_counts.append(int(np.count_nonzero(shell_totals)))
        pyg_batch = pyg_batch.to(device, non_blocking=True)
        if use_amp:
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                out = model(pyg_batch, seqs, smiles, pdb_codes)
        else:
            out = model(pyg_batch, seqs, smiles, pdb_codes)
        out, aux_outs = _unpack_fusion_output(out)
        if return_details and isinstance(aux_outs, dict) \
                and aux_outs.get('loss_mode') in (
                    'ehign_bidirectional', 'main_with_ehign_aux'):
            directional = aux_outs.get('directional_predictions') or []
            if len(directional) == 2:
                pred_to_shell_chunks.append(directional[0].detach().float().view(-1))
                pred_from_shell_chunks.append(directional[1].detach().float().view(-1))
        pred_chunks.append(out.detach().float().view(-1))
        label_chunks.append(pyg_batch.y.view(-1).detach().float())
    if not pred_chunks:
        if return_details:
            return np.array([]), np.array([]), {
                'pdb_code': [], 'protein_length': [], 'ligand_atom_count': [],
                'total_contacts': [], 'near_contact_fraction': [],
                'nonempty_shell_count': [],
                'prediction_to_shell': [], 'prediction_from_shell': [],
            }
        return np.array([]), np.array([])
    preds = torch.cat(pred_chunks).cpu().numpy()
    labels = torch.cat(label_chunks).cpu().numpy()
    if return_details:
        pred_to_shell = (
            torch.cat(pred_to_shell_chunks).cpu().numpy().tolist()
            if pred_to_shell_chunks else [])
        pred_from_shell = (
            torch.cat(pred_from_shell_chunks).cpu().numpy().tolist()
            if pred_from_shell_chunks else [])
        return labels, preds, {
            'pdb_code': pdb_code_rows,
            'protein_length': protein_lengths,
            'ligand_atom_count': ligand_atom_counts,
            'total_contacts': total_contacts,
            'near_contact_fraction': near_contact_fractions,
            'nonempty_shell_count': nonempty_shell_counts,
            'prediction_to_shell': pred_to_shell,
            'prediction_from_shell': pred_from_shell,
        }
    return labels, preds


def evaluate_and_log(metric: Evaluate, G, P, name: str, logger: Logger):
    rmse, mae, r, sd = metric.evaluate(G, P)
    logger.log(f'  >>> {name:<18} '
               f'RMSE={rmse:.4f}  MAE={mae:.4f}  SD={sd:.4f}  Pearson_r={r:.4f}  '
               f'(N={len(G)})')
    return rmse, mae, sd, r


def make_optimizer(model, base_lr: float, esm_lr: float, weight_decay: float,
                   lig_lr: Optional[float] = None,
                   struct_lr: Optional[float] = None,
                   readout_weight_decay: Optional[float] = None):
    """Assign the configured learning rates to the backbone, structure, and fusion groups."""
    esm_params, lig_params, struct_params = [], [], []
    readout_params, other_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith('prot.esm'):
            esm_params.append(p)
        elif n.startswith('lig.lm'):
            lig_params.append(p)
        elif n.startswith('plitext.backbone'):
            struct_params.append(p)
        elif readout_weight_decay is not None and n.startswith(
                ('fusion.smooth_head.',
                 'fusion.full_ordered_residual_readout.',
                 'fusion.ordered_lowrank_readout.')):
            readout_params.append(p)
        else:
            other_params.append(p)

    groups = [{'params': other_params, 'lr': base_lr, 'weight_decay': weight_decay}]
    if esm_params:
        groups.append({'params': esm_params, 'lr': esm_lr, 'weight_decay': weight_decay})
    if lig_params:
        groups.append({'params': lig_params,
                       'lr': (lig_lr if lig_lr is not None else base_lr),
                       'weight_decay': weight_decay})
    if struct_params:
        groups.append({'params': struct_params,
                       'lr': (struct_lr if struct_lr is not None else base_lr),
                       'weight_decay': weight_decay})
    if readout_params:
        groups.append({
            'params': readout_params,
            'lr': base_lr,
            'weight_decay': float(readout_weight_decay),
        })
    return torch.optim.AdamW(groups)


def make_scheduler(optimizer, sched_cfg: Optional[Dict[str, Any]]):
    """Return the scheduler and its step mode: plateau, epoch, or None."""
    if not sched_cfg:
        return None, None
    name = str(sched_cfg.get('name', '')).strip().lower()
    if name in ('', 'none', 'null', 'off', 'disable'):
        return None, None

    if name in ('reduce_on_plateau', 'plateau', 'reducelronplateau'):
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        sched = ReduceLROnPlateau(
            optimizer,
            mode=str(sched_cfg.get('mode', 'min')),
            factor=float(sched_cfg.get('factor', 0.5)),
            patience=int(sched_cfg.get('patience', 5)),
            min_lr=float(sched_cfg.get('min_lr', 0.0)),
            cooldown=int(sched_cfg.get('cooldown', 0)),
            threshold=float(sched_cfg.get('threshold', 1e-4)),
            threshold_mode=str(sched_cfg.get('threshold_mode', 'rel')),
        )
        return sched, 'plateau'

    if name in ('cosine', 'cosineannealing', 'cosineannealinglr'):
        from torch.optim.lr_scheduler import CosineAnnealingLR
        sched = CosineAnnealingLR(
            optimizer,
            T_max=int(sched_cfg.get('t_max', 200)),
            eta_min=float(sched_cfg.get('min_lr', 0.0)),
        )
        return sched, 'epoch'

    raise ValueError(f'unknown scheduler name: {name!r}. '
                     f'Supported: reduce_on_plateau / cosine')


def _to_jsonable(obj):
    """Recursively convert numpy scalars/arrays/containers to native Python types
    for ``json.dump`` (numpy.float32/64 are not JSON-serializable by default)."""
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return obj


@dataclass
class TrainingState:
    """Scalar state to persist during one (model, repeat) training run."""
    start_epoch: int = 1
    best_val_mse: float = float('inf')
    best_epoch: int = -1
    best_val_metrics: Optional[tuple] = None       # (rmse, mae, sd, r)
    no_improve: int = 0
    epoch_times: List[float] = field(default_factory=list)
    best_state_dict: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = field(default_factory=list)


# Omit pretrained backbones that can be reloaded independently.
_PRETRAINED_BACKBONE_PREFIXES = ('prot.esm.', 'lig.lm.')


def _is_backbone_key(k: str) -> bool:
    return any(k.startswith(pfx) for pfx in _PRETRAINED_BACKBONE_PREFIXES)


def _extract_savable_state(model) -> Dict[str, Any]:
    """Strip frozen pretrained backbones (ESM-2 / ChemBERTa) from model.state_dict().

    Keeps trainable params (incl. LoRA adapters) and all non-backbone parts (GNN / fusion / proj / buffers).
    """
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    out: Dict[str, Any] = {}
    for k, v in model.state_dict().items():
        is_frozen_backbone = _is_backbone_key(k) and k not in trainable_names
        if is_frozen_backbone:
            continue
        out[k] = v
    return out


def _clone_savable_state(model) -> Dict[str, Any]:
    """Make a true weight snapshot rather than state_dict storage aliases."""
    return {
        k: v.detach().clone()
        for k, v in _extract_savable_state(model).items()
    }


class ModelEMA:
    """Per-optimizer-step exponential moving average of savable model state."""

    def __init__(self, decay: float = 0.999, start_epoch: int = 5):
        if not 0.0 < float(decay) < 1.0:
            raise ValueError(f'EMA decay must be in (0,1), got {decay}')
        if int(start_epoch) < 1:
            raise ValueError('EMA start_epoch must be >= 1')
        self.decay = float(decay)
        self.start_epoch = int(start_epoch)
        self.shadow: Dict[str, Any] = {}
        self.num_updates = 0

    @property
    def initialized(self) -> bool:
        return bool(self.shadow)

    @torch.no_grad()
    def update(self, model) -> None:
        current = _extract_savable_state(model)
        if not self.shadow:
            self.shadow = {
                k: v.detach().clone() for k, v in current.items()
            }
        else:
            one_minus = 1.0 - self.decay
            for key, value in current.items():
                source = value.detach()
                target = self.shadow[key]
                if target.is_floating_point() or target.is_complex():
                    target.mul_(self.decay).add_(source, alpha=one_minus)
                else:
                    target.copy_(source)
        self.num_updates += 1

    def state_dict(self) -> Dict[str, Any]:
        return {
            'decay': self.decay,
            'start_epoch': self.start_epoch,
            'num_updates': self.num_updates,
            'shadow': {k: v.detach().clone() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.decay = float(state.get('decay', self.decay))
        self.start_epoch = int(state.get('start_epoch', self.start_epoch))
        self.num_updates = int(state.get('num_updates', 0))
        self.shadow = {
            k: v.detach().clone()
            for k, v in (state.get('shadow') or {}).items()
        }

    @torch.no_grad()
    def apply_to(self, model) -> Dict[str, Any]:
        if not self.initialized:
            raise RuntimeError('cannot apply an uninitialized EMA')
        backup = _clone_savable_state(model)
        model.load_state_dict(self.shadow, strict=False)
        return backup

    @torch.no_grad()
    def restore(self, model, backup: Dict[str, Any]) -> None:
        model.load_state_dict(backup, strict=False)

    def weight_snapshot(self) -> Dict[str, Any]:
        return {k: v.detach().clone() for k, v in self.shadow.items()}


def _capture_rng_state() -> Dict[str, Any]:
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(rng: Optional[Dict[str, Any]]) -> None:
    if not rng:
        return
    try:
        if 'python' in rng:
            random.setstate(rng['python'])
        if 'numpy' in rng:
            np.random.set_state(rng['numpy'])
        if 'torch' in rng:
            torch.set_rng_state(rng['torch'])
        if rng.get('cuda') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng['cuda'])
    except Exception:
        # A failed RNG-state restore affects reproducibility, not checkpoint weights.
        pass


def _save_checkpoint(path: str, *, model, optimizer, scheduler, epoch: int,
                     state: TrainingState, done_training: bool,
                     ema: Optional[ModelEMA] = None) -> None:
    """Save the best weights and the optimizer and RNG state needed to resume."""
    latest = _extract_savable_state(model)
    payload = {
        'state_dict': state.best_state_dict if state.best_state_dict is not None else latest,
        'latest_state_dict': latest,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'epoch': int(epoch),
        'best_val_mse': float(state.best_val_mse),
        'best_epoch': int(state.best_epoch),
        'best_val_metrics': state.best_val_metrics,
        'no_improve': int(state.no_improve),
        'epoch_times': list(state.epoch_times),
        'history': list(state.history),
        'rng_state': _capture_rng_state(),
        'ema': ema.state_dict() if ema is not None else None,
        'done_training': bool(done_training),
        'format_version': 5,
    }
    # Rename the completed temporary file to avoid partial checkpoints.
    tmp = path + '.tmp'
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _save_final_best_checkpoint(path: str, *, model, epoch: int,
                                state: TrainingState) -> None:
    """Save best validation weights without the state needed to resume training."""
    best = (state.best_state_dict if state.best_state_dict is not None
            else _extract_savable_state(model))
    payload = {
        'state_dict': best,
        'latest_state_dict': best,
        'epoch': int(epoch),
        'best_val_mse': float(state.best_val_mse),
        'best_epoch': int(state.best_epoch),
        'best_val_metrics': state.best_val_metrics,
        'no_improve': int(state.no_improve),
        'epoch_times': list(state.epoch_times),
        'history': list(state.history),
        'done_training': True,
        'checkpoint_kind': 'final_best_only',
        'format_version': 6,
    }
    tmp = path + '.tmp'
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _try_load_checkpoint(path: str, device, logger: Logger,
                         model, optimizer, scheduler=None,
                         ema: Optional[ModelEMA] = None,
                         allow_mid_training_resume: bool = True
                         ) -> Optional[TrainingState]:
    """Try loading full checkpoint. On success returns TrainingState and injects
    weights/optimizer/scheduler; legacy format or missing fields -> None (train from scratch)."""
    if not os.path.isfile(path):
        return None
    try:
        ckpt = torch.load(path, map_location=device)
    except Exception as e:
        logger.log(f'[resume][warn] failed to load {os.path.basename(path)}: {e}; '
                   f'will retrain from scratch')
        return None
    if 'epoch' not in ckpt or 'latest_state_dict' not in ckpt:
        # Weights-only checkpoints cannot resume optimizer state.
        logger.log(f'[resume][warn] {os.path.basename(path)} is legacy format '
                   f'(no resume info); will retrain from scratch (existing best weights ignored)')
        return None

    state = TrainingState()
    state.best_val_mse = float(ckpt.get('best_val_mse', float('inf')))
    state.best_epoch = int(ckpt.get('best_epoch', -1))
    state.best_val_metrics = ckpt.get('best_val_metrics')
    state.no_improve = int(ckpt.get('no_improve', 0))
    state.epoch_times = list(ckpt.get('epoch_times', []))
    state.history = list(ckpt.get('history', []))
    state.best_state_dict = ckpt.get('state_dict')
    state.start_epoch = int(ckpt['epoch']) + 1
    state._done_training = bool(ckpt.get('done_training', False))  # type: ignore[attr-defined]

    if not state._done_training and not allow_mid_training_resume:
        logger.log(
            f'[resume] ignoring unfinished checkpoint at epoch {ckpt["epoch"]}; '
            'checkpoint mode is final_best_only, so this seed restarts from epoch 1.'
        )
        return None

    try:
        missing, unexpected = model.load_state_dict(ckpt['latest_state_dict'], strict=False)
        non_backbone_missing = [k for k in missing if not _is_backbone_key(k)]
        if non_backbone_missing:
            logger.log(f'[resume][warn] non-backbone missing keys when loading latest: '
                       f'{non_backbone_missing[:5]}...')
        if unexpected:
            logger.log(f'[resume][warn] unexpected keys when loading latest: {len(unexpected)}')
    except Exception as e:
        logger.log(f'[resume][warn] model state load failed: {e}; will retrain from scratch')
        return None

    if 'optimizer' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
        except Exception as e:
            logger.log(f'[resume][warn] optimizer state load failed: {e}; '
                       f'will keep freshly-built optimizer')

    sched_state = ckpt.get('scheduler')
    if scheduler is not None and sched_state is not None:
        try:
            scheduler.load_state_dict(sched_state)
        except Exception as e:
            logger.log(f'[resume][warn] scheduler state load failed: {e}; '
                       f'will keep freshly-built scheduler')
    elif scheduler is not None and sched_state is None:
        logger.log('[resume][warn] checkpoint has no scheduler state '
                   '(pre-format_version=4); scheduler will start from fresh state.')

    if ema is not None and ckpt.get('ema') is not None:
        try:
            ema.load_state_dict(ckpt['ema'])
        except Exception as e:
            logger.log(f'[resume][warn] EMA state load failed: {e}; '
                       f'EMA will restart at epoch {ema.start_epoch}')

    _restore_rng_state(ckpt.get('rng_state'))
    return state


def _completed_repeats_in_csv(csv_path: str) -> set:
    """Read result CSV and return set of completed repeat ids (int).

    CSV may have trailing 'avg' / 'std' rows (appended after all repeats);
    that means the experiment group is **fully done** -> return special set {-1}.
    """
    if not os.path.isfile(csv_path):
        return set()
    try:
        import pandas as pd
        lock_path = csv_path + '.lock'
        with open(lock_path, 'a', encoding='utf-8') as lock_handle:
            fcntl.flock(lock_handle, fcntl.LOCK_SH)
            df = pd.read_csv(csv_path)
            fcntl.flock(lock_handle, fcntl.LOCK_UN)
        if 'repeat' not in df.columns:
            return set()
        vals = df['repeat'].astype(str).tolist()
        if 'avg' in vals or 'std' in vals:
            return {-1}  # The aggregate row marks every repeat as complete.
        done = set()
        for v in vals:
            try:
                done.add(int(v))
            except ValueError:
                pass
        return done
    except Exception:
        return set()


def _ensure_result_csv(csv_path: str, reset: bool = False) -> None:
    """Create a result CSV header exactly once, safe for dual-GPU shards."""
    lock_path = csv_path + '.lock'
    with open(lock_path, 'a', encoding='utf-8') as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        if reset or not os.path.isfile(csv_path):
            with open(csv_path, 'w', encoding='utf-8') as handle:
                handle.write('repeat,rmse,mae,sd,r\n')
        fcntl.flock(lock_handle, fcntl.LOCK_UN)


def _append_result_row(csv_path: str, repeat: int,
                       rmse_value: float, mae_value: float,
                       sd_value: float, r_value: float) -> bool:
    """Append one repeat atomically; return False if it already exists."""
    lock_path = csv_path + '.lock'
    with open(lock_path, 'a', encoding='utf-8') as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        done = _completed_repeats_in_csv_unlocked(csv_path)
        if repeat in done or -1 in done:
            fcntl.flock(lock_handle, fcntl.LOCK_UN)
            return False
        with open(csv_path, 'a+', encoding='utf-8') as handle:
            handle.seek(0, os.SEEK_END)
            pos = handle.tell()
            if pos > 0:
                handle.seek(pos - 1)
                if handle.read(1) != '\n':
                    handle.write('\n')
            handle.write(','.join([
                str(repeat), f'{rmse_value:.4f}', f'{mae_value:.4f}',
                f'{sd_value:.4f}', f'{r_value:.4f}',
            ]) + '\n')
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
    return True


def _completed_repeats_in_csv_unlocked(csv_path: str) -> set:
    """Small CSV parser used only while the caller owns the result lock."""
    if not os.path.isfile(csv_path):
        return set()
    done = set()
    with open(csv_path, encoding='utf-8') as handle:
        next(handle, None)
        for line in handle:
            value = line.split(',', 1)[0].strip()
            if value in ('avg', 'std'):
                return {-1}
            try:
                done.add(int(value))
            except ValueError:
                pass
    return done


def run_experiment(config_path: str, split: SplitSpec,
                   data_root: str = 'data_processed_esm',
                   device_override: Optional[str] = None,
                   models_override: Optional[List[str]] = None,
                   resume: bool = True,
                   batch_train_override: Optional[int] = None,
                   batch_eval_override: Optional[int] = None,
                   num_workers_override: Optional[int] = None,
                   repeat_indices_override: Optional[List[int]] = None,
                   validation_only: bool = False):
    """Train each repeat and evaluate its best validation checkpoint.

    With validation_only, skip loading test data and writing test results.
    A later invocation can reuse the completed checkpoints for evaluation."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    exp_cfg = cfg['experiment']
    train_cfg = cfg['training']

    # Older configurations may still require resumable checkpoints.
    exp_tag_for_policy = str(exp_cfg.get('tag') or '')
    configured_checkpoint_mode = train_cfg.get('checkpoint_mode')
    if configured_checkpoint_mode is None:
        checkpoint_mode = (
            'final_best_only'
            if exp_tag_for_policy.startswith('hs62_strictpk3_20261001_down_')
            else 'full_resume'
        )
    else:
        checkpoint_mode = str(configured_checkpoint_mode).strip().lower()
    if checkpoint_mode not in {'full_resume', 'final_best_only'}:
        raise ValueError(
            'training.checkpoint_mode must be full_resume or final_best_only, '
            f'got {checkpoint_mode!r}')

    esm_mode = exp_cfg.get('esm_mode', 'frozen')
    lr = float(train_cfg.get('lr', 1e-4))
    lr_esm = float(train_cfg.get('lr_esm', lr))
    lr_lig = float(train_cfg.get('lr_lig', lr))
    lr_struct = float(train_cfg.get('lr_struct', lr))
    wd = float(train_cfg.get('weight_decay', 1e-4))
    readout_wd = train_cfg.get('readout_weight_decay')
    if readout_wd is not None:
        readout_wd = float(readout_wd)
    batch_train = int(batch_train_override or train_cfg.get('batch_size_train', 32))
    batch_eval = int(batch_eval_override or train_cfg.get('batch_size_eval', batch_train))
    num_epochs = int(train_cfg.get('epochs', 200))
    patience = int(train_cfg.get('early_stop_patience', 15))
    result_std_ddof = int(train_cfg.get('result_std_ddof', 0))
    if result_std_ddof not in (0, 1):
        raise ValueError('training.result_std_ddof must be 0 or 1')
    external_counts = train_cfg.get('required_external_counts') or {}
    sched_cfg = train_cfg.get('scheduler')
    # The temporal holdout reuses one partition across five training seeds.
    n_repeats = (
        5 if split.name == 'holdout'
        else int(train_cfg.get('n_repeats', split.n_repeats)))
    seed_offset = int(train_cfg.get('seed_offset', 0))
    seed_sequence_cfg = train_cfg.get('seed_sequence')
    seed_sequence = None
    if seed_sequence_cfg is not None:
        seed_sequence = [int(value) for value in seed_sequence_cfg]
        if not seed_sequence:
            raise ValueError(
                'training.seed_sequence must be a non-empty list')
    if repeat_indices_override is None:
        repeat_indices = list(range(n_repeats))
        sharded_repeats = False
    else:
        repeat_indices = sorted(set(int(x) for x in repeat_indices_override))
        if not repeat_indices or any(x < 0 or x >= n_repeats for x in repeat_indices):
            raise ValueError(
                f'repeat indices must be a non-empty subset of 0..{n_repeats - 1}, '
                f'got {repeat_indices}')
        sharded_repeats = True
    log_interval = int(train_cfg.get('log_interval', 20))
    num_workers = int(num_workers_override if num_workers_override is not None
                       else train_cfg.get('num_workers', 4))
    pin_memory = bool(train_cfg.get('pin_memory', True))
    persistent_workers = bool(train_cfg.get('persistent_workers', True)) and num_workers > 0
    prefetch_factor = train_cfg.get('prefetch_factor')   # None uses the DataLoader default.
    if prefetch_factor is not None:
        prefetch_factor = int(prefetch_factor)
    amp_dtype = _resolve_amp_dtype(train_cfg.get('amp_dtype'))
    ema_cfg = train_cfg.get('ema') or {}
    ema_enabled = bool(ema_cfg.get('enabled', False))
    ema_decay = float(ema_cfg.get('decay', 0.999))
    ema_start_epoch = int(ema_cfg.get('start_epoch', 5))
    bucket_by_length = bool(train_cfg.get('bucket_by_length', False))
    bucket_count = train_cfg.get('bucket_count')      # None lets the sampler choose the bucket count.
    if bucket_count is not None:
        bucket_count = int(bucket_count)
    device_str = device_override or train_cfg.get('device', 'cuda:0')
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available() and device.type == 'cuda':
        frac = os.environ.get('CUDA_MEMORY_FRACTION')
        if frac:
            idx = device.index if device.index is not None else torch.cuda.current_device()
            torch.cuda.set_per_process_memory_fraction(float(frac), idx)

    ligand_models = list(models_override) if models_override else list(cfg['ligand']['models'])
    lig_type = str(cfg.get('ligand', {}).get('type', 'chemberta')).lower()

    exp_tag = exp_cfg.get('tag') or esm_mode

    save_root = os.path.join(_ROOT, 'results',
                             f'{split.name}_{exp_tag}_lr{lr:g}')
    os.makedirs(save_root, exist_ok=True)

    # Use separate log files for different model subsets.
    mode_suffix = '_validation_only' if validation_only else ''
    if sharded_repeats:
        shard_label = '_'.join(str(x) for x in repeat_indices)
        log_filename = f'train_repeats_{shard_label}{mode_suffix}.log'
    elif models_override:
        log_filename = (
            'train_' + '_'.join(models_override) + mode_suffix + '.log')
    else:
        log_filename = f'train{mode_suffix}.log'
    log_path = os.path.join(save_root, log_filename)
    logger = Logger(log_path)

    logger.section(
        f'MHGRL  |  split={split.name}  |  tag={exp_tag}  |  '
        f'esm_mode={esm_mode}  |  lr={lr:g}',
        char='#',
    )
    logger.log(f'config              : {config_path}')
    logger.log(f'device              : {device}')
    logger.log(f'train batch size    : {batch_train}')
    logger.log(f'eval  batch size    : {batch_eval}')
    logger.log(f'learning rate       : {lr:g}  (esm_lr={lr_esm:g}, lig_lr={lr_lig:g})')
    logger.log(f'ligand encoder type : {lig_type}')
    logger.log(f'weight_decay        : {wd:g}')
    logger.log(
        f'readout weight_decay: '
        f'{readout_wd if readout_wd is not None else "same as global"}')
    logger.log(f'max epochs          : {num_epochs}')
    logger.log(f'early-stop patience : {patience}')
    logger.log(f'checkpoint mode     : {checkpoint_mode}')
    if sched_cfg:
        sched_name = str(sched_cfg.get('name', 'none'))
        if sched_name.lower() in ('', 'none', 'null', 'off', 'disable'):
            logger.log(f'lr scheduler        : (disabled)')
        else:
            extra = {k: v for k, v in sched_cfg.items() if k != 'name'}
            logger.log(f'lr scheduler        : {sched_name}  {extra}')
    else:
        logger.log(f'lr scheduler        : (none, constant lr)')
    logger.log(f'repeats per model   : {n_repeats}')
    logger.log(f'repeats selected    : {repeat_indices}')
    logger.log(f'seed offset         : {seed_offset}')
    if seed_sequence is not None:
        logger.log(f'seed sequence       : {seed_sequence}')
    logger.log(f'num_workers         : {num_workers}  '
               f'(prefetch_factor={prefetch_factor if prefetch_factor is not None else "default(2)"})')
    logger.log(f'amp_dtype           : {amp_dtype if amp_dtype is not None else "off (fp32)"}')
    logger.log(
        f'EMA weights         : '
        f'{f"decay={ema_decay:g}, start_epoch={ema_start_epoch}" if ema_enabled else "off"}')
    logger.log(f'bucket_by_length    : {bucket_by_length}'
               f'{f" (num_buckets={bucket_count})" if bucket_by_length and bucket_count else ""}')
    logger.log(f'log file            : {log_path}')
    logger.log(f'results dir         : {save_root}')
    logger.log(f'ligand models       : {ligand_models}')
    logger.log(
        'run mode            : '
        + ('validation-only (test data is not loaded)' if validation_only
           else 'train/resume + test evaluation'))

    torch.backends.cudnn.benchmark = True

    overall_start = time.time()

    for m_idx, model_name in enumerate(ligand_models):
        logger.section(f'MODEL {m_idx + 1}/{len(ligand_models)} : {model_name}', char='#')

        # Do not create test result files during validation-only runs.
        result_files = {}
        already_done = {}
        if not validation_only:
            for tag, _tpl in split.test_specs:
                rf = os.path.join(
                    save_root,
                    f'result_{model_name}_{split.name}_{tag}.csv')
                _ensure_result_csv(rf, reset=(not resume))
                done = _completed_repeats_in_csv(rf) if resume else set()
                already_done[tag] = done
                result_files[tag] = rf

        all_aggregated = all((-1 in already_done[tag]) for tag in already_done)
        if resume and all_aggregated and already_done:
            logger.log(f'[resume] {model_name}: all test sets aggregated (avg/std present), skipping.')
            continue

        for repeat in repeat_indices:
            # Skip a repeat only when every requested test set is complete.
            if resume and not validation_only:
                already_repeat = all(
                    (repeat in already_done[tag]) or (-1 in already_done[tag])
                    for tag in already_done
                )
                if already_repeat:
                    logger.log(f'[resume] {model_name} repeat {repeat} already done '
                               f'(all test tags), skipping.')
                    continue

            if seed_sequence is not None and repeat < len(seed_sequence):
                seed = seed_sequence[repeat]
            else:
                seed = 42 + 2 ** repeat + seed_offset
            set_seed(seed)
            logger.section(
                f'{model_name}  |  repeat {repeat + 1}/{n_repeats}  |  seed={seed}',
                char='='
            )

            t0 = time.time()
            train_ds = PDBbindESMDataset(root=data_root, dataset=split.train_name(repeat))
            valid_ds = PDBbindESMDataset(root=data_root, dataset=split.valid_name(repeat))
            test_loaders = {}

            loader_kwargs = dict(
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                collate_fn=esm_collate_fn,
            )
            if prefetch_factor is not None and num_workers > 0:
                loader_kwargs['prefetch_factor'] = prefetch_factor

            # Optional ligand-size resampling applies to training only.
            from training.size_shift import build_size_shift  # noqa: E402
            size_shift_cfg = train_cfg.get('size_shift') or {}
            size_shift = build_size_shift(
                size_shift_cfg,
                getattr(getattr(train_ds, '_data', None), 'c_size', []))
            if size_shift is not None and bucket_by_length:
                raise ValueError(
                    'training.size_shift and bucket_by_length are mutually '
                    'exclusive (both drive the train sampler)')
            if size_shift is not None:
                train_sampler = size_shift.sampler(
                    num_samples_ratio=float(
                        size_shift_cfg.get('num_samples_ratio', 1.0)),
                    seed=seed)
                train_loader = DataLoader(train_ds, batch_size=batch_train,
                                          sampler=train_sampler,
                                          drop_last=False, **loader_kwargs)
            elif bucket_by_length:
                train_sampler = LengthBucketSampler(
                    train_ds, batch_size=batch_train,
                    num_buckets=bucket_count, shuffle=True,
                    drop_last=False, seed=seed,
                )
                train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                                          **loader_kwargs)
            else:
                train_loader = DataLoader(train_ds, batch_size=batch_train,
                                          shuffle=True, drop_last=False,
                                          **loader_kwargs)
            valid_loader = DataLoader(valid_ds, batch_size=batch_eval,
                                      shuffle=False, **loader_kwargs)

            test_sizes = {}
            if not validation_only:
                for tag, tpl in split.test_specs:
                    ds = PDBbindESMDataset(
                        root=data_root,
                        dataset=split.test_name(tpl, repeat))
                    if tag in external_counts:
                        if len(ds) != int(external_counts[tag]):
                            raise ValueError(
                                f'{tag}: expected {external_counts[tag]} test complexes, found {len(ds)}')
                        eval_codes = {c.lower() for c in ds._data.pdb_code if c}
                        develop_codes = {
                            c.lower() for dataset in (train_ds, valid_ds)
                            for c in dataset._data.pdb_code if c
                        }
                        if len(eval_codes) != len(ds) or len(develop_codes) != len(train_ds) + len(valid_ds):
                            raise ValueError(f'{tag}: missing or duplicate train, validation, or test codes')
                        if eval_codes & develop_codes:
                            raise ValueError(f'{tag}: test complexes overlap training or validation')
                    test_loaders[tag] = DataLoader(
                        ds, batch_size=batch_eval,
                        shuffle=False, **loader_kwargs)
                    test_sizes[tag] = len(ds)

            size_str = '  '.join(f'#test({k})={v:>4}' for k, v in test_sizes.items())
            logger.log(f'#train={len(train_ds):>6}  '
                       f'#valid={len(valid_ds):>6}  '
                       f'{size_str}  '
                       f'(loaded in {time.time() - t0:.1f}s)')
            if size_shift is not None:
                logger.log(f'size_shift           : {size_shift.description}')

            # Resolve the train-only scaler before constructing the model.
            repeat_cfg = _config_for_repeat(cfg, split.name, repeat)
            pli_cfg = repeat_cfg.get('plitext') or {}
            scaler_path = pli_cfg.get('scaler_path')
            if scaler_path:
                logger.log(f'structure scaler    : {scaler_path}')
            model = build_model(repeat_cfg, ligand_model_name=model_name).to(device)
            setup_struct_branch(model, pli_cfg, split=split.name, repeat=repeat)
            n_total = sum(p.numel() for p in model.parameters())
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.log(f'model={model_name}  '
                       f'total_params={n_total:,}  trainable_params={n_train:,}')

            from training.losses import make_loss_fn  # noqa: E402
            train_storage = getattr(train_ds, '_data', None)
            train_targets = getattr(train_storage, 'y', None)
            loss_fn = make_loss_fn(
                cfg['training'].get('loss'), training_targets=train_targets)
            if hasattr(loss_fn, 'description'):
                logger.log(f'loss function        : {loss_fn.description}')

            # Initialize FDS from training labels after creating the readout.
            from models.feature_dir import (  # noqa: E402
                attach_fds_to_model, ranksim_config)
            feature_dir_cfg = (cfg.get('fusion') or {}).get('feature_dir') or {}
            fds_desc = attach_fds_to_model(
                model, feature_dir_cfg, train_targets)
            if fds_desc:
                head = model.fusion.smooth_head
                if getattr(head, 'fds', None) is not None:
                    head.fds = head.fds.to(device)
                logger.log(f'feature_dir          : {fds_desc}')
            rs_enabled, rs_weight, rs_temperature = ranksim_config(
                feature_dir_cfg)
            if rs_enabled:
                logger.log(
                    f'ranksim              : weight={rs_weight:g} '
                    f'temperature={rs_temperature:g}')
            else:
                rs_weight = 0.0

            optimizer = make_optimizer(model, base_lr=lr, esm_lr=lr_esm,
                                       weight_decay=wd, lig_lr=lr_lig,
                                       struct_lr=lr_struct,
                                       readout_weight_decay=readout_wd)
            # Register trainable loss parameters before constructing the scheduler.
            loss_params = []
            if isinstance(loss_fn, torch.nn.Module):
                loss_fn = loss_fn.to(device)
                loss_params = [p for p in loss_fn.parameters() if p.requires_grad]
            if loss_params:
                sigma_lr = float(getattr(loss_fn, 'sigma_lr', 0.0)) or lr
                optimizer.add_param_group(
                    {'params': loss_params, 'lr': sigma_lr, 'weight_decay': 0.0})
                logger.log(
                    f'loss params          : '
                    f'{sum(p.numel() for p in loss_params)} '
                    f'(lr={sigma_lr:g}, no weight decay)')
            scheduler, sched_step_kind = make_scheduler(optimizer, sched_cfg)
            aux_loss_weight = float((cfg.get('fusion') or {}).get('aux_weight', 0.0))
            metric = Evaluate()
            # Use gradient scaling for fp16, not bf16.
            grad_scaler = (
                torch.cuda.amp.GradScaler() if amp_dtype == torch.float16 else None
            )
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            ema = (
                ModelEMA(decay=ema_decay, start_epoch=ema_start_epoch)
                if ema_enabled else None
            )

            model_file = os.path.join(
                save_root, f'{model_name}_{split.name}_{exp_tag}_{repeat}.pt')

            state = TrainingState()
            skip_training = False
            if resume:
                loaded = _try_load_checkpoint(model_file, device, logger,
                                              model, optimizer,
                                              scheduler=scheduler, ema=ema,
                                              allow_mid_training_resume=(
                                                  checkpoint_mode == 'full_resume'))
                if loaded is not None:
                    state = loaded
                    if getattr(state, '_done_training', False):
                        logger.log(
                            f'[resume] training already finished at epoch {state.start_epoch - 1} '
                            f'(best epoch={state.best_epoch}, best_val_MSE='
                            f'{state.best_val_mse:.4f}); going to test phase.'
                        )
                        skip_training = True
                    else:
                        logger.log(
                            f'[resume] continuing from epoch {state.start_epoch} '
                            f'(best epoch={state.best_epoch}, best_val_MSE='
                            f'{state.best_val_mse:.4f}, no_improve={state.no_improve}/{patience}, '
                            f'completed_so_far={len(state.epoch_times)} epochs).'
                        )

            if not skip_training:
                logger.subsection(
                    'Per-epoch summary  '
                    '(train + val metrics, ETA = est. time to early-stop or max-epoch)'
                )
                header = (
                    f'  {"ep":>3} | '
                    f'{"tr_loss":>8} {"tr_RMSE":>8} {"tr_r":>6} | '
                    f'{"v_RMSE":>7} {"v_MAE":>7} {"v_SD":>7} {"v_r":>6} | '
                    f'{"best_vR":>7} | '
                    f'{"lr":>9} | '
                    f'{"ep_time":>8} | {"ETA":>9} | '
                    f'{"best":>4} {"pat":>5}'
                )
                logger.log(header, with_ts=False)

                early_stopped = False
                last_epoch = state.start_epoch - 1
                for epoch in range(state.start_epoch, num_epochs + 1):
                    ep_t0 = time.time()
                    # Record the learning rate before the scheduler changes it.
                    current_lr = float(optimizer.param_groups[0]['lr'])
                    current_lr_esm = (
                        float(optimizer.param_groups[1]['lr'])
                        if len(optimizer.param_groups) > 1 else None
                    )
                    tr_loss, tr_rmse, tr_r, _ = train_one_epoch(
                        model, device, train_loader, optimizer, loss_fn,
                        epoch, log_interval, logger,
                        amp_dtype=amp_dtype, grad_scaler=grad_scaler,
                        trainable_params=trainable_params,
                        aux_loss_weight=aux_loss_weight,
                        ema=ema,
                        ranksim_weight=rs_weight,
                        ranksim_temperature=rs_temperature,
                    )
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    evaluated_with_ema = bool(
                        ema is not None and ema.initialized)
                    ema_backup = None
                    if evaluated_with_ema:
                        ema_backup = ema.apply_to(model)
                    try:
                        G, P = predict(
                            model, device, valid_loader,
                            amp_dtype=amp_dtype)
                    finally:
                        if evaluated_with_ema:
                            ema.restore(model, ema_backup)
                    v_rmse, v_mae, v_r, v_sd = metric.evaluate(G, P)
                    val_mse = float(mse(G, P))

                    improved = val_mse < state.best_val_mse
                    if improved:
                        state.best_val_mse = val_mse
                        state.best_epoch = epoch
                        # Convert NumPy scalars before writing the history JSON.
                        state.best_val_metrics = (
                            float(v_rmse), float(v_mae), float(v_sd), float(v_r)
                        )
                        state.no_improve = 0
                        state.best_state_dict = (
                            ema.weight_snapshot()
                            if evaluated_with_ema
                            else _clone_savable_state(model)
                        )
                        flag = '*'
                    else:
                        state.no_improve += 1
                        flag = ' '

                    ep_time = time.time() - ep_t0
                    state.epoch_times.append(ep_time)

                    avg_ep = float(np.mean(state.epoch_times[-5:]))
                    remain_by_max = num_epochs - epoch
                    remain_by_patience = max(patience - state.no_improve, 0)
                    remain = max(min(remain_by_max, remain_by_patience), 0)
                    eta_sec = avg_ep * remain
                    best_val_rmse = (state.best_val_metrics[0]
                                     if state.best_val_metrics is not None else float('inf'))

                    # Normalized entropy near one indicates uniform pocket attention.
                    fusion_mod = getattr(model, 'fusion', None)
                    pocket_attn = getattr(fusion_mod, 'pocket_attn', None)
                    if pocket_attn is None:
                        pocket_attn = getattr(fusion_mod, 'ligand_attn', None)
                    pocket_entropy = (
                        None if pocket_attn is None
                        else pocket_attn.last_attn_entropy)
                    pocket_gate = (
                        None if pocket_attn is None or not pocket_attn.use_gate
                        else float(torch.sigmoid(pocket_attn.gate).detach()))
                    # The gate should favor pocket features when they cover less of the chain.
                    dil = getattr(
                        getattr(model, 'fusion', None), 'pocket_gate', None)
                    dil_mean = None if dil is None else dil.last_gate_mean
                    dil_corr = (None if dil is None
                                else dil.last_gate_share_corr)
                    pair_res = getattr(fusion_mod, 'pair_residual', None)
                    pair_abs = (
                        None if pair_res is None
                        else getattr(pair_res, 'last_abs_mean', None))
                    pair_pri = getattr(fusion_mod, 'pair_primary', None)
                    pair_pri_abs = (
                        None if pair_pri is None
                        else getattr(pair_pri, 'last_abs_mean', None))
                    pocket_pool = getattr(fusion_mod, 'pocket_pool', None)
                    pocket_n_res = (
                        None if pocket_pool is None
                        else getattr(pocket_pool, 'last_n_res', None))
                    pocket_prot_l2 = (
                        None if pocket_pool is None
                        else getattr(pocket_pool, 'last_prot_l2', None))
                    signed_shell = getattr(fusion_mod, 'signed_shell', None)
                    signed_abs = (
                        None if signed_shell is None
                        else getattr(signed_shell, 'last_abs_mean', None))
                    iface = getattr(fusion_mod, 'coarse_interface', None)
                    iface_shell_abs = (
                        None if iface is None
                        else getattr(iface, 'last_shell_abs', None))
                    iface_edge_abs = (
                        None if iface is None
                        else getattr(iface, 'last_edge_abs', None))
                    iface_n_pl = (
                        None if iface is None
                        else getattr(iface, 'last_n_pl', None))
                    iface_n_prot = (
                        None if iface is None
                        else getattr(iface, 'last_n_prot', None))
                    iface_n_lig = (
                        None if iface is None
                        else getattr(iface, 'last_n_lig', None))
                    geo_n_cb = getattr(fusion_mod, 'last_n_cb', None)
                    geo_n_atom = getattr(fusion_mod, 'last_n_atom', None)
                    geo_n_hier = getattr(fusion_mod, 'last_n_hier', None)
                    geo_n_pl = getattr(fusion_mod, 'last_n_pl', None)
                    shell_geom = getattr(fusion_mod, 'shell_geometry', None)
                    shell_geom_abs = (
                        None if shell_geom is None
                        else getattr(shell_geom, 'last_abs', None))
                    shell_geom_n_pl = (
                        None if shell_geom is None
                        else getattr(shell_geom, 'last_n_pl', None))
                    shell_to_3d = getattr(fusion_mod, 'shell_to_3d', None)
                    shell_to_3d_abs = (
                        None if shell_to_3d is None
                        else getattr(shell_to_3d, 'last_abs', None))
                    shell_to_3d_entropy = (
                        None if shell_to_3d is None
                        else getattr(shell_to_3d, 'last_entropy', None))
                    geo_readout = getattr(fusion_mod, 'shell_geo_readout', None)
                    geo_ch_abs = (
                        None if geo_readout is None
                        else getattr(geo_readout, 'last_abs', None))
                    geo_col_abs = (
                        None if geo_readout is None
                        else getattr(fusion_mod, 'last_geo_col_abs', None))
                    geo_edge_abs = (
                        None if getattr(fusion_mod, 'geo_edge_proj', None) is None
                        else getattr(fusion_mod, 'last_geo_edge_abs', None))
                    geo_head = getattr(fusion_mod, 'geo_head', None)
                    geo_head_abs = (
                        None if geo_head is None
                        else getattr(geo_head, 'last_abs', None))
                    geo_film = getattr(fusion_mod, 'geo_film', None)
                    geo_film_abs = (
                        None if geo_film is None
                        else getattr(geo_film, 'last_abs', None))
                    iface_tok = getattr(fusion_mod, 'interface_tokens', None)
                    iface_tok_abs = (
                        None if iface_tok is None
                        else getattr(iface_tok, 'last_abs', None))
                    iface_col_abs = getattr(fusion_mod, 'last_iface_col_abs', None)
                    side_tower = getattr(fusion_mod, 'side_tower', None)
                    side_tower_abs = (
                        None if side_tower is None
                        else getattr(side_tower, 'last_abs', None))
                    freq_mix = getattr(getattr(model, 'plitext', None), 'freq_mix', None)
                    freq_mix_gate = (
                        None if freq_mix is None
                        else getattr(freq_mix, 'last_gate', None))
                    freq_mix_abs = (
                        None if freq_mix is None
                        else getattr(freq_mix, 'last_abs', None))
                    residue_slots = getattr(
                        getattr(model, 'plitext', None), 'residue_slots', None)
                    residue_slot_abs = (
                        None if residue_slots is None
                        else getattr(residue_slots, 'last_abs', None))
                    residue_slot_n = (
                        None if residue_slots is None
                        else getattr(residue_slots, 'last_n_res', None))
                    contact_pri = getattr(fusion_mod, 'contact_primary', None)
                    contact_pri_abs = (
                        None if contact_pri is None
                        else getattr(contact_pri, 'last_abs', None))
                    contact_n_pl = (
                        None if contact_pri is None
                        else getattr(contact_pri, 'last_n_pl', None))

                    state.history.append({
                        'epoch': int(epoch),
                        'tr_loss': float(tr_loss),
                        'tr_rmse': float(tr_rmse),
                        'tr_r': float(tr_r),
                        'v_rmse': float(v_rmse),
                        'v_mae': float(v_mae),
                        'v_sd': float(v_sd),
                        'v_r': float(v_r),
                        'val_mse': float(val_mse),
                        'lr': current_lr,
                        'lr_esm': current_lr_esm,
                        'ep_time_sec': float(ep_time),
                        'improved': bool(improved),
                        'ema_eval': bool(evaluated_with_ema),
                        'pocket_attn_entropy': pocket_entropy,
                        'pocket_attn_gate': pocket_gate,
                        'dilution_gate_mean': dil_mean,
                        'dilution_gate_share_corr': dil_corr,
                        'pair_residual_abs_mean': pair_abs,
                        'pair_primary_abs_mean': pair_pri_abs,
                        'pocket_node_n_res': pocket_n_res,
                        'pocket_prot_l2': pocket_prot_l2,
                        'signed_shell_abs_mean': signed_abs,
                        'interface_shell_abs': iface_shell_abs,
                        'interface_edge_abs': iface_edge_abs,
                        'interface_n_pl': iface_n_pl,
                        'interface_n_prot': iface_n_prot,
                        'interface_n_lig': iface_n_lig,
                        'geo_n_cb': geo_n_cb,
                        'geo_n_atom': geo_n_atom,
                        'geo_n_hier': geo_n_hier,
                        'geo_n_pl': geo_n_pl,
                        'shell_geom_abs': shell_geom_abs,
                        'shell_geom_n_pl': shell_geom_n_pl,
                        'shell_to_3d_abs': shell_to_3d_abs,
                        'shell_to_3d_entropy': shell_to_3d_entropy,
                        'geo_channel_abs': geo_ch_abs,
                        'geo_col_abs': geo_col_abs,
                        'geo_edge_abs': geo_edge_abs,
                        'geo_head_abs': geo_head_abs,
                        'geo_film_abs': geo_film_abs,
                        'iface_tok_abs': iface_tok_abs,
                        'iface_col_abs': iface_col_abs,
                        'side_tower_abs': side_tower_abs,
                        'freq_mix_gate': freq_mix_gate,
                        'freq_mix_abs': freq_mix_abs,
                        'residue_slot_abs': residue_slot_abs,
                        'residue_slot_n': residue_slot_n,
                        'contact_pri_abs': contact_pri_abs,
                        'contact_n_pl': contact_n_pl,
                    })
                    if (epoch == 1 or epoch % 10 == 0) and (
                            shell_geom_abs is not None):
                        logger.log(
                            f'  shell_geom: abs={shell_geom_abs:.4f}'
                            f'  n_pl={0.0 if shell_geom_n_pl is None else shell_geom_n_pl:.1f}',
                            with_ts=False)
                    if (epoch == 1 or epoch % 10 == 0) and (
                            shell_to_3d_abs is not None):
                        logger.log(
                            f'  shell_to_3d: abs={shell_to_3d_abs:.4f}'
                            f'  entropy={0.0 if shell_to_3d_entropy is None else shell_to_3d_entropy:.3f}',
                            with_ts=False)
                    if (epoch == 1 or epoch % 10 == 0) and (
                            geo_ch_abs is not None):
                        logger.log(
                            f'  geo_readout: ch={geo_ch_abs:.4f}'
                            f'  col={0.0 if geo_col_abs is None else geo_col_abs:.4f}'
                            f'  edge={0.0 if geo_edge_abs is None else geo_edge_abs:.4f}'
                            f'  head={0.0 if geo_head_abs is None else geo_head_abs:.4f}'
                            f'  film={0.0 if geo_film_abs is None else geo_film_abs:.4f}',
                            with_ts=False)
                    if (epoch == 1 or epoch % 10 == 0) and (
                            iface_tok_abs is not None):
                        logger.log(
                            f'  iface_tok: abs={iface_tok_abs:.4f}'
                            f'  col={0.0 if iface_col_abs is None else iface_col_abs:.4f}',
                            with_ts=False)
                    if (epoch == 1 or epoch % 10 == 0) and (
                            side_tower_abs is not None):
                        logger.log(
                            f'  side_tower: abs={side_tower_abs:.4f}',
                            with_ts=False)
                    if (epoch == 1 or epoch % 10 == 0) and (
                            freq_mix_gate is not None):
                        logger.log(
                            f'  freq_mix: gate={freq_mix_gate:.4f}'
                            f'  abs={0.0 if freq_mix_abs is None else freq_mix_abs:.4f}',
                            with_ts=False)
                    if (epoch == 1 or epoch % 10 == 0) and (
                            residue_slot_abs is not None):
                        logger.log(
                            f'  residue_slots: abs={residue_slot_abs:.4f}'
                            f'  n_res={0.0 if residue_slot_n is None else residue_slot_n:.1f}',
                            with_ts=False)
                    if (epoch == 1 or epoch % 10 == 0) and (
                            contact_pri_abs is not None):
                        logger.log(
                            f'  contact_pri: abs={contact_pri_abs:.4f}'
                            f'  n_pl={0.0 if contact_n_pl is None else contact_n_pl:.1f}',
                            with_ts=False)
                    if (epoch == 1 or epoch % 10 == 0) and (
                            geo_n_cb is not None):
                        logger.log(
                            f'  multiscale: n_cb={geo_n_cb:.1f}'
                            f'  n_atom={0.0 if geo_n_atom is None else geo_n_atom:.1f}'
                            f'  n_hier={0.0 if geo_n_hier is None else geo_n_hier:.1f}'
                            f'  n_pl={0.0 if geo_n_pl is None else geo_n_pl:.1f}',
                            with_ts=False)
                    if (epoch == 1 or epoch % 10 == 0) and (
                            iface_shell_abs is not None):
                        logger.log(
                            f'  interface: shell_abs={iface_shell_abs:.4f}'
                            f'  edge_abs={0.0 if iface_edge_abs is None else iface_edge_abs:.4f}'
                            f'  n_prot={0.0 if iface_n_prot is None else iface_n_prot:.1f}'
                            f'  n_lig={0.0 if iface_n_lig is None else iface_n_lig:.1f}'
                            f'  n_pl={0.0 if iface_n_pl is None else iface_n_pl:.1f}',
                            with_ts=False)
                    if (epoch == 1 or epoch % 10 == 0) and (
                            pocket_entropy is not None or dil_mean is not None
                            or pocket_prot_l2 is not None):
                        parts = []
                        if pocket_entropy is not None:
                            parts.append(f'attn_entropy={pocket_entropy:.4f}')
                        if pocket_gate is not None:
                            parts.append(f'attn_gate={pocket_gate:.4f}')
                        if dil_mean is not None:
                            parts.append(f'dilution_g={dil_mean:.4f}')
                        if dil_corr is not None:
                            parts.append(f'corr(g,share)={dil_corr:+.4f}')
                        if pocket_n_res is not None:
                            parts.append(f'pocket_n_res={pocket_n_res:.1f}')
                        if pocket_prot_l2 is not None:
                            parts.append(f'pocket_prot_l2={pocket_prot_l2:.4f}')
                        logger.log('  pocket: ' + '  '.join(parts),
                                   with_ts=False)

                    logger.log(
                        f'  {epoch:>3} | '
                        f'{tr_loss:>8.4f} {tr_rmse:>8.4f} {tr_r:>6.3f} | '
                        f'{v_rmse:>7.4f} {v_mae:>7.4f} {v_sd:>7.4f} {v_r:>6.3f} | '
                        f'{best_val_rmse:>7.4f} | '
                        f'{current_lr:>9.2e} | '
                        f'{fmt_secs(ep_time):>8} | {fmt_secs(eta_sec):>9} | '
                        f'{state.best_epoch:>3}{flag} {state.no_improve:>2}/{patience:<2}',
                        with_ts=False,
                    )

                    if scheduler is not None:
                        prev_lr = current_lr
                        prev_lr_esm = current_lr_esm
                        if sched_step_kind == 'plateau':
                            scheduler.step(val_mse)
                        else:
                            scheduler.step()
                        new_lr = float(optimizer.param_groups[0]['lr'])
                        new_lr_esm = (
                            float(optimizer.param_groups[1]['lr'])
                            if len(optimizer.param_groups) > 1 else None
                        )
                        if abs(new_lr - prev_lr) > 1e-12:
                            msg = f'  [lr] scheduler reduced base lr: {prev_lr:.2e} -> {new_lr:.2e}'
                            if prev_lr_esm is not None and new_lr_esm is not None \
                                    and abs(new_lr_esm - prev_lr_esm) > 1e-12:
                                msg += f'  | esm lr: {prev_lr_esm:.2e} -> {new_lr_esm:.2e}'
                            logger.log(msg)

                    last_epoch = epoch
                    early_stopped = state.no_improve >= patience
                    is_last_epoch = early_stopped or epoch >= num_epochs

                    if checkpoint_mode == 'full_resume':
                        _save_checkpoint(
                            model_file, model=model, optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch, state=state,
                            done_training=is_last_epoch, ema=ema,
                        )
                    elif is_last_epoch:
                        # Compact checkpoints need only the best validation weights.
                        _save_final_best_checkpoint(
                            model_file, model=model, epoch=epoch, state=state)

                    if early_stopped:
                        logger.log(f'>>> Early stop at epoch {epoch} '
                                   f'(no improvement for {patience} epochs)')
                        break

                if not early_stopped and last_epoch >= num_epochs:
                    logger.log(f'>>> Reached max_epochs={num_epochs}')

            logger.subsection('Best checkpoint (selected by validation MSE)')
            logger.log(f'  best_epoch    : {state.best_epoch}')
            logger.log(f'  best_val_MSE  : {state.best_val_mse:.4f}')
            if state.best_val_metrics is not None:
                br, bm, bs, bp = state.best_val_metrics
                logger.log(f'  val_RMSE      : {br:.4f}')
                logger.log(f'  val_MAE       : {bm:.4f}')
                logger.log(f'  val_SD        : {bs:.4f}')
                logger.log(f'  val_Pearson_r : {bp:.4f}')
            logger.log(f'  saved weights : {model_file}')

            history_file = os.path.join(
                save_root,
                f'{model_name}_{split.name}_{exp_tag}_{repeat}_history.json')
            try:
                payload = _to_jsonable({
                    'model': model_name,
                    'split': split.name,
                    'exp_tag': exp_tag,
                    'repeat': repeat,
                    'seed': seed,
                    'total_params': n_total,
                    'trainable_params': n_train,
                    'epochs_completed': len(state.history),
                    'train_epoch_time_total_sec': sum(
                        float(row.get('ep_time_sec', 0.0))
                        for row in state.history),
                    'train_epoch_time_mean_sec': (
                        float(np.mean([
                            float(row.get('ep_time_sec', 0.0))
                            for row in state.history]))
                        if state.history else 0.0),
                    'train_epoch_time_std_sec': (
                        float(np.std([
                            float(row.get('ep_time_sec', 0.0))
                            for row in state.history], ddof=0))
                        if state.history else 0.0),
                    'best_epoch': state.best_epoch,
                    'best_val_mse': state.best_val_mse,
                    'best_val_metrics': state.best_val_metrics,
                    'config': config_path,
                    'history': state.history,
                })
                with open(history_file, 'w') as f:
                    json.dump(payload, f, indent=2)
                logger.log(f'  history json  : {history_file}')
            except Exception as e:
                logger.log(f'  [warn] failed to save history json: {e}')

            # Validation-only runs never load test weights or write test results.
            if validation_only:
                logger.log(
                    f'>>> repeat {repeat + 1}/{n_repeats} validation-only done; '
                    'checkpoint/history saved, test phase intentionally skipped.')
                continue

            if state.best_state_dict is not None:
                missing, unexpected = model.load_state_dict(
                    state.best_state_dict, strict=False)
                non_backbone_missing = [k for k in missing if not _is_backbone_key(k)]
                if non_backbone_missing:
                    logger.log(f'  [load][warn] missing non-backbone keys: {non_backbone_missing[:5]} ...')
                if unexpected:
                    logger.log(f'  [load][warn] unexpected keys: {len(unexpected)} (ignored)')
            else:
                logger.log('  [warn] no best_state_dict in memory; using current model state.')

            logger.subsection('Final test on external sets '
                              '(using the best-val checkpoint above)')
            for tag, loader in test_loaders.items():
                if resume and (repeat in already_done.get(tag, set())):
                    logger.log(f'  [resume] {tag}: repeat {repeat} already in CSV, skipping test.')
                    continue
                if len(loader.dataset) == 0:
                    logger.log(f'  [skip] {tag}: empty test set')
                    continue
                G, P, sample_details = predict(
                    model, device, loader, amp_dtype=amp_dtype, return_details=True)
                rmse, mae, sd, r = evaluate_and_log(metric, G, P, tag, logger)
                pred_json = os.path.join(
                    save_root,
                    f'pred_{tag.lower()}_{model_name}_{split.name}_{exp_tag}_{repeat}.json')
                pred_payload = {
                        'pdb_code': sample_details['pdb_code'],
                        'pred': P.tolist(),
                        'ground_truth': G.tolist(),
                        'residual': (P - G).tolist(),
                    }
                if sample_details['prediction_to_shell']:
                    pred_payload['prediction_to_shell'] = sample_details['prediction_to_shell']
                    pred_payload['prediction_from_shell'] = sample_details['prediction_from_shell']
                with open(pred_json, 'w') as f:
                    json.dump(pred_payload, f)
                comparison_csv = os.path.join(
                    save_root,
                    f'comparison_{tag.lower()}_{model_name}_{split.name}_{exp_tag}_{repeat}.csv')
                prediction_extras = {
                    'protein_length': sample_details['protein_length'],
                    'ligand_atom_count': sample_details['ligand_atom_count'],
                    'total_contacts': sample_details['total_contacts'],
                    'near_contact_fraction': sample_details['near_contact_fraction'],
                    'nonempty_shell_count': sample_details['nonempty_shell_count'],
                }
                if sample_details['prediction_to_shell']:
                    prediction_extras['prediction_to_shell'] = \
                        sample_details['prediction_to_shell']
                    prediction_extras['prediction_from_shell'] = \
                        sample_details['prediction_from_shell']
                    prediction_extras['direction_gap'] = [
                        a - b for a, b in zip(
                            sample_details['prediction_to_shell'],
                            sample_details['prediction_from_shell'])
                    ]
                write_prediction_table(
                    comparison_csv,
                    test_set=tag,
                    repeat=repeat,
                    pdb_codes=sample_details['pdb_code'],
                    ground_truth=G,
                    predictions=P,
                    extra_columns=prediction_extras,
                )
                logger.log(f'  prediction table: {comparison_csv}')
                appended = _append_result_row(
                    result_files[tag], repeat, rmse, mae, sd, r)
                if not appended:
                    logger.log(
                        f'  [shard/resume] {tag} repeat {repeat} was already '
                        'written by another process; keeping the existing row.')
                already_done.setdefault(tag, set()).add(repeat)

            logger.log(f'>>> repeat {repeat + 1}/{n_repeats} done. '
                       f'elapsed since start: {fmt_secs(time.time() - overall_start)}')

            time.sleep(2)

        if validation_only:
            logger.log(
                f'>>> {model_name}: validation-only run complete; '
                'no test artifacts were created.')
            continue

        # The first shard that sees all repeats can aggregate under the result lock.
        if sharded_repeats:
            for tag, rf in result_files.items():
                done = _completed_repeats_in_csv(rf)
                if done == set(range(n_repeats)):
                    try:
                        cal_final_results(
                            result_file=rf, n_repeat=n_repeats, logger=logger,
                            std_ddof=result_std_ddof)
                        comparison_tables = sorted(glob.glob(os.path.join(
                            save_root,
                            f'comparison_{tag.lower()}_{model_name}_{split.name}_'
                            f'{exp_tag}_[0-9]*.csv')))
                        if len(comparison_tables) >= n_repeats:
                            hard_path = os.path.join(
                                save_root,
                                f'difficult_samples_{tag.lower()}_{model_name}_'
                                f'{split.name}_{exp_tag}.csv')
                            aggregate_prediction_tables(
                                comparison_tables, hard_path)
                            logger.log(
                                f'  difficult-sample summary: {hard_path}')
                    except Exception as e:
                        logger.log(
                            f'[warn] shard final aggregation failed for {tag}: {e}')
                elif -1 in done:
                    logger.log(f'[shard] {tag} already contains avg/std.')
                else:
                    logger.log(
                        f'[shard] {tag} currently has repeats {sorted(done)}; '
                        'the last shard will add avg/std.')
            continue

        for tag, rf in result_files.items():
            if resume and (-1 in already_done.get(tag, set())):
                logger.log(f'[resume] {rf} already aggregated, skipping cal_final_results.')
                continue
            try:
                cal_final_results(
                    result_file=rf, n_repeat=n_repeats, logger=logger,
                    std_ddof=result_std_ddof)
            except Exception as e:
                logger.log(f'[warn] cal_final_results failed for {rf}: {e}')
            comparison_tables = sorted(glob.glob(os.path.join(
                save_root,
                f'comparison_{tag.lower()}_{model_name}_{split.name}_{exp_tag}_[0-9]*.csv')))
            if comparison_tables:
                hard_path = os.path.join(
                    save_root,
                    f'difficult_samples_{tag.lower()}_{model_name}_{split.name}_{exp_tag}.csv')
                try:
                    aggregate_prediction_tables(comparison_tables, hard_path)
                    logger.log(f'  difficult-sample summary: {hard_path}')
                except Exception as e:
                    logger.log(
                        f'[warn] difficult-sample aggregation failed for {tag}: {e}')

    logger.section(f'ALL DONE  |  total elapsed: {fmt_secs(time.time() - overall_start)}',
                   char='#')
    logger.close()
