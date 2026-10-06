"""
M06 upstream fork: SiT-XL/2 train with ONLINE SDVAE encoding — FiLM-advanced.

DEVIATION (FiLM-advanced): per-layer scalar gates × 28 (one per SiT block),
Kaiming-uniform init on proj weights, zero-init on gates.
Gates zero-init ensures step-0 forward is numerically identical to upstream
(identity preservation). Ablation vs film_simple (additive bias at input only).

Forked from: scripts/train_m06_upstream_sdvae_online_mask.py (online SDVAE)
RAE reference: scripts/train_m06_upstream_rae_film_advanced.py

PRESERVED from sdvae_online_mask.py (online VAE pipeline):
  - Online per-batch vae.encode() — NOT ImgLatentDataset (offline latent shards)
  - SiT path setup + SiT_XL_2 import
  - SDVAE_MULTIPLIER=0.18215
  - Transport: SiT create_transport (not RAE stage2.transport)
  - center_crop_arr ADM pipeline
  - EMA / W&B / best.pt / early-stop / ckpt save
  - W1/W5 defensive guards, C3 DDP early-stop broadcast

CHANGED vs sdvae_online_mask.py:
  - SiT_M06_FiLM: Kaiming proj + zero-init per-layer gates (28 blocks)
  - Dataset: RawImageMaskCannyDataset (raw images + mask + canny shards)
    Supports all 4 phases: warmup/mask/canny/canny_mask
  - Phase guard: accepts warmup/mask/canny/canny_mask (not mask-only)
  - set_phase_freezing: warmup freezes all adapters; mask freezes canny_*;
    canny freezes mask_*; canny_mask unfreezes all
  - load_pretrained drops adapter keys (preserve Kaiming proj init)
  - update_ema strips _orig_mod. prefix (torch.compile compat)
  - _build_model_kwargs helper (per-phase)
  - No [0,1] fix: SDVAE does NOT use DINOv2 (no image_01 conversion needed)

Phase semantics:
  warmup:     mask_proj_x + canny_proj_x + gates frozen; identical to upstream
  mask:       mask_proj_x + mask_gates trainable; canny_* frozen
  canny:      canny_proj_x + canny_gates trainable; mask_* frozen
  canny_mask: all four adapter params trainable

Smoke validation (5 min after launch):
  [sanity] latent std=<value> — expected ~0.15 (= 0.83 × 0.18215)
  [sanity] mask std=<value>   — expected ~0.5 for binary masks
  [M06][freeze-audit] mask_proj_x.weight: trainable=<True/False>
  [M06][freeze-audit] mask_gates: trainable=<True/False>
"""
from __future__ import annotations
from typing import Optional
from pathlib import Path
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cuda
import torch.backends.cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import sys
import yaml
import json
import numpy as np
import logging
import os
import argparse
import subprocess
from datetime import datetime
from time import time
from glob import glob
from copy import deepcopy
from collections import OrderedDict
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
from models import SiT_XL_2
from transport import create_transport
from accelerate import Accelerator
sys.path.insert(0, str(_SCRIPT_DIR))
from feature_information_dynamics.data.raw_image_mask_canny_dataset import RawImageMaskCannyDataset
SIT_HIDDEN: int = 1152
SIT_NUM_BLOCKS: int = 28
SIT_TOKENS: int = 256
MASK_KERNEL: int = 16
SDVAE_MULTIPLIER: float = 0.18215
CHAIN_NUM_CLASSES: int = 1001
CHAIN_CLASS_DROPOUT: float = 0.0
PHASE_CHOICES = ('warmup', 'mask', 'canny', 'canny_mask')

class SiT_M06_FiLM(nn.Module):
    """SiT-XL/2 wrapper with FiLM-advanced per-layer mask/canny conditioning.

    DEVIATION (FiLM-advanced vs film_simple):
      - mask_proj_x / canny_proj_x: Kaiming-uniform init (not zeros).
        Zeros init on proj + zeros init on gates = dead gradient at step 0.
        Kaiming init on proj + zeros init on gates = smooth start (gate scales
        proj output to zero at step 0) + healthy gradients from step 1.
      - mask_gates / canny_gates: nn.Parameter(zeros(num_blocks=28)), zero-init.
        At step 0: x_tok += 0 * proj(mask) = x_tok (identity preservation).
        Gates are learned per-layer; model decides which blocks benefit most.

    Architecture:
      mask_proj_x  = Conv2d(1, 1152, kernel=16, stride=16, bias=False)
      canny_proj_x = Conv2d(1, 1152, kernel=16, stride=16, bias=False)
          -> maps [B, 1, 256, 256] -> [B, 1152, 16, 16] -> [B, 256, 1152]

      mask_gates   = nn.Parameter(zeros(28))  # per-block scalar gate
      canny_gates  = nn.Parameter(zeros(28))

    Per-block injection (inside the 28-block SiT loop):
        x_tok += mask_gates[i]  * mask_full   # [B, 256, 1152]
        x_tok += canny_gates[i] * canny_full  # [B, 256, 1152]

    When all gates=0 (at init), forward == upstream SiT baseline (identity).
    When mask=None and canny=None (warmup), forward is exactly upstream.
    """

    def __init__(self, input_size: int=32, num_classes: int=1001, learn_sigma: bool=True, use_checkpoint: bool=False, class_dropout_prob: float=0.0) -> None:
        """Build SiT_XL_2 backbone + Kaiming proj + zero-init per-layer gates.

        Args:
            input_size:         Latent spatial size (32 for SDVAE f8d4).
            num_classes:        1001 for chained phases (1000 classes + null slot).
            learn_sigma:        Whether model outputs sigma (True for SiT-XL/2).
            use_checkpoint:     Gradient checkpointing.
            class_dropout_prob: 0.0 for chained phases (no label drop).
        """
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.learn_sigma = learn_sigma
        self.backbone = SiT_XL_2(input_size=input_size, num_classes=num_classes, learn_sigma=learn_sigma, class_dropout_prob=class_dropout_prob)
        self.mask_proj_x = nn.Conv2d(1, SIT_HIDDEN, kernel_size=MASK_KERNEL, stride=MASK_KERNEL, bias=False)
        nn.init.kaiming_uniform_(self.mask_proj_x.weight, a=5 ** 0.5)
        self.canny_proj_x = nn.Conv2d(1, SIT_HIDDEN, kernel_size=MASK_KERNEL, stride=MASK_KERNEL, bias=False)
        nn.init.kaiming_uniform_(self.canny_proj_x.weight, a=5 ** 0.5)
        self.mask_gates = nn.Parameter(torch.zeros(SIT_NUM_BLOCKS))
        self.canny_gates = nn.Parameter(torch.zeros(SIT_NUM_BLOCKS))

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor]=None, canny: Optional[torch.Tensor]=None) -> torch.Tensor:
        """Forward with per-layer FiLM-advanced gate injection.

        Inlines backbone forward to inject mask/canny per block.
        Calling backbone.forward() directly is not used because SiT_XL_2
        has no per-block conditioning hook.

        Args:
            x:     [B, 4, 32, 32]   noisy SDVAE latent
            t:     [B]              timestep
            y:     [B]              class label (0-999 or 1000 for null)
            mask:  [B, 256, 256] or [B, 1, 256, 256] binary mask, or None
            canny: [B, 256, 256] or [B, 1, 256, 256] canny edges, or None

        Returns:
            [B, 4, 32, 32] velocity prediction
        """
        from torch.utils.checkpoint import checkpoint
        b = self.backbone
        mask_full = None
        canny_full = None
        if mask is not None:
            m = mask if mask.dtype.is_floating_point else mask.float()
            if m.dim() == 3:
                m = m.unsqueeze(1)
            mask_full = self.mask_proj_x(m).flatten(2).transpose(1, 2)
        if canny is not None:
            ce = canny if canny.dtype.is_floating_point else canny.float()
            if ce.dim() == 3:
                ce = ce.unsqueeze(1)
            canny_full = self.canny_proj_x(ce).flatten(2).transpose(1, 2)
        x_tok = b.x_embedder(x) + b.pos_embed
        t_emb = b.t_embedder(t)
        y_emb = b.y_embedder(y, b.training)
        c = t_emb + y_emb
        for i, block in enumerate(b.blocks):
            if self.use_checkpoint:
                x_tok = checkpoint(block, x_tok, c, use_reentrant=False)
            else:
                x_tok = block(x_tok, c)
            if mask_full is not None:
                x_tok = x_tok + self.mask_gates[i] * mask_full
            if canny_full is not None:
                x_tok = x_tok + self.canny_gates[i] * canny_full
        out = b.final_layer(x_tok, c)
        out = b.unpatchify(out)
        if self.learn_sigma:
            out, _ = out.chunk(2, dim=1)
        return out

def load_pretrained_sit_backbone(model: SiT_M06_FiLM, ckpt_path: str, rank: int=0) -> SiT_M06_FiLM:
    """Load SiT paper ckpt into model.backbone; adapter keys dropped for Kaiming.

    FiLM-advanced deviation: ALWAYS drop mask_proj_x/canny_proj_x/mask_gates/
    canny_gates from ckpt to preserve fresh Kaiming proj init + zero gate init.
    If ckpt contained old zero proj weights, gates=0 + proj=0 would dead-lock
    gradient flow (no gradient path through identity-zero combination).

    Handles flat state_dict or {'ema'/'model': {...}} wrapper.
    Priority: 'model' for chained M06 ckpts (has best_val_loss), 'ema' for
    paper pretrain ckpts (per feedback_ckpt_load_prefer_ema).

    Args:
        model:     SiT_M06_FiLM instance.
        ckpt_path: Path to SiT-XL-2-256.pt or M06 best.pt.
        rank:      Logging rank (only rank 0 prints).

    Returns:
        model with backbone weights loaded, adapter keys at Kaiming/zero init.
    """
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f'Pretrain ckpt not found: {path}')
    raw = torch.load(str(path), map_location='cpu', weights_only=False)
    _is_chained = isinstance(raw, dict) and 'best_val_loss' in raw
    _priority = ('model', 'ema') if _is_chained else ('ema', 'model')
    src_sd, key_used = (None, None)
    for cand in _priority:
        if isinstance(raw, dict) and cand in raw:
            src_sd, key_used = (raw[cand], cand)
            break
    if src_sd is None:
        src_sd, key_used = (raw, '<flat>')
    src_sd = {k.removeprefix('module.').removeprefix('_orig_mod.').removeprefix('net.'): v for k, v in src_sd.items() if isinstance(v, torch.Tensor)}
    _adapter_pfx = ('mask_proj_x.', 'canny_proj_x.', 'mask_gates', 'canny_gates')
    _dropped = []
    for k in list(src_sd.keys()):
        if any((k.startswith(p) for p in _adapter_pfx)):
            del src_sd[k]
            _dropped.append(k)
    if _dropped and rank == 0:
        print(f'[sit_ckpt] DROP {_dropped} from ckpt (FiLM-advanced: preserve Kaiming init for proj, zero init for gates)')
    _has_backbone_prefix = any((k.startswith('backbone.') for k in src_sd))
    if _has_backbone_prefix:
        result = model.load_state_dict(src_sd, strict=False)
    else:
        result = model.backbone.load_state_dict(src_sd, strict=False)
    if rank == 0:
        non_adapter = [k for k in result.missing_keys if not any((k.startswith(p) for p in _adapter_pfx))]
        if non_adapter:
            print(f'[sit_ckpt] non-adapter missing ({len(non_adapter)}): {non_adapter[:6]}')
        print(f"[sit_ckpt] key='{key_used}' missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}")
    return model

def _build_model_kwargs(y: torch.Tensor, mask: torch.Tensor, canny: torch.Tensor, phase: str) -> dict:
    """Build model_kwargs from batch tensors based on training phase.

    Phase semantics:
      warmup:     mask=None, canny=None (all adapters frozen)
      mask:       mask passed, canny=None
      canny:      canny passed, mask=None
      canny_mask: both mask and canny passed

    Args:
        y:     [B] class label tensor.
        mask:  [B, 256, 256] float32 mask tensor.
        canny: [B, 256, 256] float32 canny tensor.
        phase: One of PHASE_CHOICES.

    Returns:
        Dict with 'y' and optionally 'mask' and/or 'canny'.
    """
    kwargs: dict = {'y': y}
    if phase in ('mask', 'canny_mask'):
        kwargs['mask'] = mask
    if phase in ('canny', 'canny_mask'):
        kwargs['canny'] = canny
    return kwargs

@torch.no_grad()
def evaluate_deterministic(model: torch.nn.Module, val_loader: DataLoader, device: torch.device, transport: object, vae: torch.nn.Module, phase: str, val_seed: int=12345, val_n_batches: int=8) -> torch.Tensor:
    """Deterministic val: MSE loss with fixed seed per batch.

    Mirrors train protocol: injects mask/canny per phase.
    Val loader must yield RawImageMaskCannyDataset tuples: (image, mask, canny, label).

    Per batch: torch.manual_seed(val_seed + batch_idx) ensures identical
    noise/timestep sampling across calls (same model state → same val loss).

    Args:
        model:         The model (may be DDP-wrapped).
        val_loader:    DataLoader yielding (image, mask, canny, label).
        device:        CUDA device.
        transport:     Transport object with training_losses method.
        vae:           Frozen VAE for online encoding.
        phase:         Training phase — controls model_kwargs.
        val_seed:      Base seed for deterministic eval.
        val_n_batches: Maximum batches to evaluate.

    Returns:
        Scalar MSE loss tensor on device, NOT yet all-reduced.
    """
    model.eval()
    total_mse = torch.tensor(0.0, device=device)
    n_seen = 0
    for batch_idx, (image, mask, canny, label) in enumerate(val_loader):
        if batch_idx >= val_n_batches:
            break
        image = image.to(device)
        mask = mask.to(device)
        canny = canny.to(device)
        label = label.to(device)
        torch.manual_seed(val_seed + batch_idx)
        torch.cuda.manual_seed_all(val_seed + batch_idx)
        x = vae.encode(image).latent_dist.sample().mul_(SDVAE_MULTIPLIER)
        model_kwargs = _build_model_kwargs(label, mask, canny, phase)
        loss_dict = transport.training_losses(model, x, model_kwargs)
        mse = loss_dict['loss'].mean()
        total_mse += mse
        n_seen += 1
    if n_seen == 0:
        return torch.tensor(0.0, device=device)
    model.train()
    return total_mse / n_seen

def spawn_fid_eval(ckpt_path: str, experiment_dir: str, train_config: dict) -> None:
    raise NotImplementedError('FID hooks are separate; disable train.fid_every and use the upstream FID tool explicitly.')

def load_weights_with_shape_check(model: nn.Module, checkpoint: dict, rank: int=0) -> nn.Module:
    """Load weights from checkpoint, skipping mismatched shapes.

    Args:
        model:      Target model.
        checkpoint: Dict with 'model' key containing state_dict.
        rank:       Logging rank (only rank 0 prints skips).

    Returns:
        model with weights loaded in-place.
    """
    model_state_dict = model.state_dict()
    for name, param in checkpoint['model'].items():
        if name in model_state_dict:
            if param.shape == model_state_dict[name].shape:
                model_state_dict[name].copy_(param)
            elif rank == 0:
                print(f"Skipping parameter '{name}': checkpoint shape {param.shape}, model shape {model_state_dict[name].shape}")
        elif rank == 0:
            print(f"Parameter '{name}' not found in model, skipping.")
    model.load_state_dict(model_state_dict, strict=False)
    return model

@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float=0.9999) -> None:
    """Step the EMA model towards the current model.

    Strips 'module.' (DDP) and '_orig_mod.' (torch.compile) prefixes.

    Args:
        ema_model: EMA copy of model.
        model:     Live model (may have 'module.' or '_orig_mod.' prefix).
        decay:     EMA decay rate (default 0.9999).
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        name = name.replace('module.', '').removeprefix('_orig_mod.')
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model: nn.Module, flag: bool=True) -> None:
    """Set requires_grad flag for all parameters.

    Args:
        model: The model to update.
        flag:  Whether parameters require gradients.
    """
    for p in model.parameters():
        p.requires_grad = flag

def load_config(config_path: str) -> dict:
    """Load YAML config file.

    Args:
        config_path: Path to YAML file.

    Returns:
        Parsed config dict.
    """
    with open(config_path, 'r') as fh:
        config = yaml.safe_load(fh)
    return config

def create_logger(logging_dir: str) -> logging.Logger:
    """Create a logger that writes to a log file and stdout (rank 0 only).

    Args:
        logging_dir: Directory to write log.txt into.

    Returns:
        Configured logger.
    """
    if dist.get_rank() == 0:
        logging.basicConfig(level=logging.INFO, format='[\x1b[34m%(asctime)s\x1b[0m] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', handlers=[logging.StreamHandler(), logging.FileHandler(f'{logging_dir}/log.txt')])
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def do_train(train_config: dict, accelerator: Accelerator) -> Accelerator:
    """Train SiT_M06_FiLM with ONLINE SDVAE encoding — FiLM-advanced ablation."""
    device = accelerator.device
    _use_tf32 = train_config.get('train', {}).get('allow_tf32', True)
    torch.backends.cuda.matmul.allow_tf32 = _use_tf32
    torch.backends.cudnn.allow_tf32 = _use_tf32
    if accelerator.is_main_process:
        print(f'[speedup] allow_tf32={_use_tf32} | gradient_as_bucket_view=True | fused_adamw=True')
    _phase = train_config['data'].get('phase', 'warmup')
    if _phase not in PHASE_CHOICES:
        raise RuntimeError(f"train_m06_upstream_sdvae_online_film_advanced.py: data.phase='{_phase}' not in {PHASE_CHOICES}.")
    if train_config['data'].get('latent_norm', False):
        raise RuntimeError('[sdvae-film] data.latent_norm=true detected. Online encoding uses raw VAE output × 0.18215; z-score normalization would double-process the latent. Set data.latent_norm: false.')
    _lm = train_config['data'].get('latent_multiplier', 1.0)
    if abs(_lm - 1.0) > 1e-06:
        raise RuntimeError(f'[sdvae-film] data.latent_multiplier={_lm}. Online encoding bakes 0.18215 via .mul_(SDVAE_MULTIPLIER); yaml latent_multiplier MUST be 1.0.')
    experiment_dir = f"{train_config['train']['output_dir']}/{train_config['train']['exp_name']}"
    checkpoint_dir = f'{experiment_dir}/checkpoints'
    if accelerator.is_main_process:
        os.makedirs(train_config['train']['output_dir'], exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f'Experiment directory created at {experiment_dir}')
        tensorboard_dir_log = f"tensorboard_logs/{train_config['train']['exp_name']}"
        os.makedirs(tensorboard_dir_log, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_dir_log)
        config_str = json.dumps(train_config, indent=4)
        writer.add_text('training configs', config_str, global_step=0)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
        writer = None
    rank = accelerator.local_process_index
    downsample_ratio = train_config.get('vae', {}).get('downsample_ratio', 8)
    assert train_config['data']['image_size'] % downsample_ratio == 0
    latent_size = train_config['data']['image_size'] // downsample_ratio
    _num_classes = train_config['data'].get('num_classes', CHAIN_NUM_CLASSES)
    _class_dropout = train_config['model'].get('class_dropout_prob', CHAIN_CLASS_DROPOUT)
    if _num_classes != CHAIN_NUM_CLASSES and accelerator.is_main_process:
        logger.warning(f'[film-chain] num_classes={_num_classes} != {CHAIN_NUM_CLASSES}; warmup ckpt LabelEmbedder has {CHAIN_NUM_CLASSES} slots — shape mismatch will cause silent embedding drop or error.')
    if abs(_class_dropout) > 1e-06 and _phase != 'warmup' and accelerator.is_main_process:
        logger.warning(f'[film-chain] class_dropout_prob={_class_dropout} != 0.0 in chained phase.')
    model = SiT_M06_FiLM(input_size=latent_size, num_classes=_num_classes, learn_sigma=train_config['model'].get('learn_sigma', True), use_checkpoint=train_config['model'].get('use_checkpoint', False), class_dropout_prob=_class_dropout)
    ema = deepcopy(model).to(device)
    _init_best_val_loss = float('inf')
    if 'weight_init' not in train_config['train']:
        raise RuntimeError('[sdvae-film] train.weight_init is required. Set to path of warmup best.pt or SiT-XL-2-256.pt.')
    ckpt_path = train_config['train']['weight_init']
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'[sdvae-film] train.weight_init not found: {ckpt_path}')
    raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    _bvl = raw.get('best_val_loss', None) if isinstance(raw, dict) else None
    if _bvl is not None and isinstance(_bvl, (int, float)) and (float(_bvl) > 0):
        _init_best_val_loss = float(_bvl)
        if accelerator.is_main_process:
            logger.info(f'[chained] inherited best_val_loss = {_init_best_val_loss:.6f} (save threshold)')
    _adapter_keys_detect = ('mask_proj_x.', 'canny_proj_x.', 'mask_gates', 'canny_gates')
    _is_m06_ckpt = isinstance(raw, dict) and 'model' in raw and any((k.startswith(pk) or k.startswith('backbone.') for k in raw['model'].keys() for pk in _adapter_keys_detect))
    if _is_m06_ckpt:
        ckpt_data = raw
        ckpt_data['model'] = {k.replace('module.', '').removeprefix('_orig_mod.'): v for k, v in ckpt_data['model'].items()}
        model = load_weights_with_shape_check(model, ckpt_data, rank=rank)
        ema = load_weights_with_shape_check(ema, ckpt_data, rank=rank)
    else:
        model = load_pretrained_sit_backbone(model, ckpt_path, rank=rank)
        ema = load_pretrained_sit_backbone(ema, ckpt_path, rank=rank)
    if accelerator.is_main_process:
        logger.info(f'Loaded weight_init from {ckpt_path} (m06_ckpt={_is_m06_ckpt})')
    requires_grad(ema, False)
    model = model.to(device)
    transport = create_transport(train_config['transport']['path_type'], train_config['transport']['prediction'], train_config['transport']['loss_weight'], train_config['transport']['train_eps'], train_config['transport']['sample_eps'])
    if accelerator.is_main_process:
        total_params = sum((p.numel() for p in model.parameters())) / 1000000.0
        logger.info(f'SiT_M06_FiLM Parameters: {total_params:.2f}M')
    _base_lr = train_config['optimizer']['lr']
    _adapter_mult = float(train_config['optimizer'].get('adapter_lr_multiplier', 1.0))
    _adapter_param_names = ('mask_proj_x', 'canny_proj_x', 'mask_gates', 'canny_gates')
    _adapter_params, _backbone_params = ([], [])
    for n, p in model.named_parameters():
        if any((ak in n for ak in _adapter_param_names)):
            _adapter_params.append(p)
        else:
            _backbone_params.append(p)
    if accelerator.is_main_process:
        logger.info(f'[M06] param groups: backbone={len(_backbone_params)} @ lr={_base_lr}, adapters={len(_adapter_params)} @ lr={_base_lr * _adapter_mult} (multiplier={_adapter_mult})')
    _fused_adamw = train_config['optimizer'].get('fused_adamw', False)
    opt = torch.optim.AdamW([{'params': _backbone_params, 'lr': _base_lr}, {'params': _adapter_params, 'lr': _base_lr * _adapter_mult}], weight_decay=0, betas=(0.9, train_config['optimizer']['beta2']), fused=_fused_adamw)
    vae_path = train_config['data'].get('vae_path', '')
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(vae_path)
    vae = vae.to(device).eval()
    requires_grad(vae, False)
    if accelerator.is_main_process:
        logger.info(f'[online] VAE loaded from {vae_path}')

    def _freeze_canny_adapters() -> None:
        """Freeze canny_proj_x and canny_gates."""
        model.canny_proj_x.weight.requires_grad_(False)
        model.canny_gates.requires_grad_(False)

    def _freeze_mask_adapters() -> None:
        """Freeze mask_proj_x and mask_gates."""
        model.mask_proj_x.weight.requires_grad_(False)
        model.mask_gates.requires_grad_(False)

    def _log_trainable() -> None:
        """Print trainable=True/False for each adapter param (audit)."""
        if accelerator.is_main_process:
            for n, p in model.named_parameters():
                if any((ak in n for ak in _adapter_param_names)):
                    logger.info(f'[M06][freeze-audit] {n}: trainable={p.requires_grad}')
    if _phase == 'mask':
        _freeze_canny_adapters()
        if accelerator.is_main_process:
            logger.info('[M06] phase=mask → mask_proj_x + mask_gates trainable, canny_proj_x + canny_gates frozen')
        _log_trainable()
    elif _phase == 'canny':
        _freeze_mask_adapters()
        if accelerator.is_main_process:
            logger.info('[M06] phase=canny → canny_proj_x + canny_gates trainable, mask_proj_x + mask_gates frozen')
        _log_trainable()
    elif _phase == 'canny_mask':
        if accelerator.is_main_process:
            logger.info('[M06] phase=canny_mask → all four adapter params trainable')
        _log_trainable()
    else:
        _freeze_mask_adapters()
        _freeze_canny_adapters()
        if accelerator.is_main_process:
            logger.info(f'[M06] phase={_phase} → all adapters frozen (no conditioning; avoids find_unused_parameters crash)')
        _log_trainable()
    if _phase == 'mask':
        assert any((p.requires_grad for p in model.mask_proj_x.parameters())), '[ASSERT] mask_proj_x must NOT be frozen in mask phase'
        assert model.mask_gates.requires_grad, '[ASSERT] mask_gates must NOT be frozen in mask phase'
        assert not any((p.requires_grad for p in model.canny_proj_x.parameters())), '[ASSERT] canny_proj_x MUST be frozen in mask phase'
    elif _phase == 'canny':
        assert any((p.requires_grad for p in model.canny_proj_x.parameters())), '[ASSERT] canny_proj_x must NOT be frozen in canny phase'
        assert model.canny_gates.requires_grad, '[ASSERT] canny_gates must NOT be frozen in canny phase'
        assert not any((p.requires_grad for p in model.mask_proj_x.parameters())), '[ASSERT] mask_proj_x MUST be frozen in canny phase'
    if accelerator.is_main_process:
        effective_hp = {'phase': _phase, 'online_encode': True, 'vae_multiplier': SDVAE_MULTIPLIER, 'film_variant': 'film_advanced (per-layer gates)', 'proj_init': 'kaiming_uniform (a=5**0.5)', 'gates_init': 'zeros', 'num_blocks_gates': SIT_NUM_BLOCKS, 'no_01_fix': 'SDVAE does not use DINOv2 (no [-1,1] -> [0,1] conversion)', 'latent_norm': 'DISABLED (raises if yaml=true)', 'num_classes': _num_classes, 'class_dropout_prob': _class_dropout, 'adapter_lr_multiplier': _adapter_mult, 'weight_init': ckpt_path, 'best_val_loss_init': f'{_init_best_val_loss:.6f}', 'lr': _base_lr, 'global_batch_size': train_config['train']['global_batch_size'], 'max_steps': train_config['train']['max_steps']}
        logger.info('[sdvae-film] effective_hp: %s', json.dumps(effective_hp, indent=2))
    image_size = train_config['data'].get('image_size', 256)
    _mask_shard_dir = train_config['data'].get('mask_shard_dir', '')
    _canny_shard_dir = train_config['data'].get('canny_shard_dir', '')
    _imagenet_root = train_config['data'].get('imagenet_root', '')
    if 'data_path' in train_config['data'] and 'imagenet_root' not in train_config['data']:
        _imagenet_root = train_config['data']['data_path']
    _paired_train_wl = train_config['data'].get('paired_train_whitelist', train_config['data'].get('whitelist_json', ''))
    dataset = RawImageMaskCannyDataset(imagenet_root=_imagenet_root, mask_shard_dir=_mask_shard_dir, canny_shard_dir=_canny_shard_dir, paired_whitelist_json=_paired_train_wl, image_size=image_size, is_val=False)
    batch_size_per_gpu = int(np.round(train_config['train']['global_batch_size'] / accelerator.num_processes / accelerator.gradient_accumulation_steps))
    global_batch_size = batch_size_per_gpu * accelerator.num_processes * accelerator.gradient_accumulation_steps
    loader = DataLoader(dataset, batch_size=batch_size_per_gpu, shuffle=True, num_workers=train_config['data']['num_workers'], pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=4)
    if accelerator.is_main_process:
        logger.info(f'[M06] RawImageMaskCannyDataset train N={len(dataset):,} path={_imagenet_root}')
        logger.info(f'Batch size {batch_size_per_gpu} per gpu, global batch size {global_batch_size}')
    val_loader = None
    val_n_batches = 0
    _paired_val_wl = train_config['data'].get('paired_val_whitelist', None)
    if _paired_val_wl is not None:
        val_ds = RawImageMaskCannyDataset(imagenet_root=_imagenet_root, mask_shard_dir=_mask_shard_dir, canny_shard_dir=_canny_shard_dir, paired_whitelist_json=_paired_val_wl, image_size=image_size, is_val=True)
        val_n_batches = train_config['train'].get('val_n_batches', 8)
        val_loader = DataLoader(val_ds, batch_size=batch_size_per_gpu, shuffle=False, num_workers=2, pin_memory=True, drop_last=False, persistent_workers=False)
        if accelerator.is_main_process:
            _val_seed = train_config['train'].get('val_seed', 12345)
            logger.info(f'[val] RawImageMaskCannyDataset N={len(val_ds):,} | phase={_phase} | n_batches={val_n_batches} | seed={_val_seed}')
    train_steps = 0
    resume_epoch = 0
    resume_best_val_loss = _init_best_val_loss
    if train_config['train'].get('resume', False):
        ckpt_files = [path for path in glob(f'{checkpoint_dir}/*.pt') if os.path.splitext(os.path.basename(path))[0].isdigit()]
        if ckpt_files:
            ckpt_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
            latest_ckpt = ckpt_files[-1]
            ckpt = torch.load(latest_ckpt, map_location=lambda storage, loc: storage)
            model.load_state_dict(ckpt['model'])
            ema.load_state_dict(ckpt['ema'])
            opt.load_state_dict(ckpt['opt'])
            train_steps = int(ckpt.get('train_steps', os.path.basename(latest_ckpt).split('.')[0]))
            resume_epoch = int(ckpt.get('epoch', 0))
            resume_best_val_loss = float(ckpt.get('best_val_loss', _init_best_val_loss))
            if accelerator.is_main_process:
                logger.info(f'Resuming full state from checkpoint: {latest_ckpt} (step={train_steps}, epoch={resume_epoch}, best={resume_best_val_loss:.6f}, optimizer=True)')
        else:
            raise RuntimeError(f'resume=True but no numeric checkpoint found in {checkpoint_dir}')
    model, opt, loader = accelerator.prepare(model, opt, loader)
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)
    if train_config['train'].get('val_only_first', False):
        if val_loader is None:
            raise RuntimeError('val_only_first=True but val_loader is None')
        results = {}
        for probe_phase in ('warmup', _phase):
            val_mse = evaluate_deterministic(model, val_loader, device, transport, vae, phase=probe_phase, val_seed=train_config['train'].get('val_seed', 12345), val_n_batches=val_n_batches)
            accelerator.wait_for_everyone()
            results[probe_phase] = accelerator.gather(val_mse.detach().unsqueeze(0)).mean().item()
        if accelerator.is_main_process:
            _unwrapped = accelerator.unwrap_model(model)
            logger.info(f"[init_check] val_only_first=True | phase=warmup val_mse={results['warmup']:.6f} | phase={_phase} val_mse={results[_phase]:.6f} | inherited_best={_init_best_val_loss:.6f} | delta(phase-warmup)={results[_phase] - results['warmup']:+.6f} | mask_gates_max_abs={_unwrapped.mask_gates.abs().max().item():.6e} | mask_proj_x_max_abs={_unwrapped.mask_proj_x.weight.abs().max().item():.6e}")
        return accelerator
    log_steps = 0
    running_loss = 0.0
    start_time = time()
    epoch = resume_epoch
    best_val_loss = resume_best_val_loss
    no_improve_epochs = 0
    val_seed = train_config['train'].get('val_seed', 12345)
    early_stop_patience = train_config['train'].get('early_stop_patience', 5)
    early_stop_min_epochs = train_config['train'].get('early_stop_min_epochs', 10)
    val_every_n_epochs = train_config['train'].get('val_every_n_epochs', 1)
    val_loss_log_path = os.path.join(experiment_dir, 'val_loss_log.jsonl') if accelerator.is_main_process else None
    _max_steps = train_config['train']['max_steps']
    _ckpt_every = train_config['train'].get('ckpt_every', max(1, _max_steps // 6))
    if accelerator.is_main_process:
        logger.info(f'Training for up to {_max_steps} steps; ckpt_every={_ckpt_every}…')
    _early_stop = False
    _sanity_logged = False
    while True:
        for batch in loader:
            with accelerator.accumulate(model):
                image, mask, canny, label = batch
                image = image.to(device)
                mask = mask.to(device)
                canny = canny.to(device)
                label = label.to(device)
                with torch.no_grad():
                    x = vae.encode(image).latent_dist.sample().mul_(SDVAE_MULTIPLIER)
                if not _sanity_logged and accelerator.is_main_process:
                    _std = x.std().item()
                    logger.info(f'[sanity] latent std={_std:.4f} (expected ~0.15 = 0.83×0.18215; if ~0.83 then mul_ failed)')
                    _mstd = mask.float().std().item()
                    _mcov = mask.float().mean().item() * 100.0
                    logger.info(f'[sanity] mask std={_mstd:.4f} (expected ~0.5 for binary mask)')
                    logger.info(f'[sanity] mask coverage={_mcov:.1f}%')
                    if not 0.05 <= mask.float().mean().item() <= 0.95:
                        logger.warning(f'[sanity] mask coverage={_mcov:.1f}% outside [5%, 95%] — check mask shard loading')
                    _sanity_logged = True
                model_kwargs = _build_model_kwargs(label, mask, canny, _phase)
                loss_dict = transport.training_losses(model, x, model_kwargs)
                if 'cos_loss' in loss_dict:
                    mse_loss = loss_dict['loss'].mean()
                    loss = loss_dict['cos_loss'].mean() + mse_loss
                else:
                    loss = loss_dict['loss'].mean()
                if not torch.isfinite(loss).all():
                    raise RuntimeError(f'[NaN guard] non-finite loss at step={train_steps}: {loss.item()}')
                opt.zero_grad()
                accelerator.backward(loss)
                if 'max_grad_norm' in train_config['optimizer']:
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), train_config['optimizer']['max_grad_norm'])
                opt.step()
            if accelerator.sync_gradients:
                update_ema(ema, model.module)
                if 'cos_loss' in loss_dict:
                    running_loss += mse_loss.item()
                else:
                    running_loss += loss.item()
                log_steps += 1
                train_steps += 1
            if accelerator.sync_gradients and train_steps > 0 and (train_steps % train_config['train']['log_every'] == 0):
                torch.cuda.synchronize()
                end_time = time()
                sps = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                if accelerator.is_main_process:
                    logger.info(f'(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Steps/Sec: {sps:.2f}')
                    writer.add_scalar('Loss/train', avg_loss, train_steps)
                running_loss = 0.0
                log_steps = 0
                start_time = time()
            if accelerator.sync_gradients and train_steps % _ckpt_every == 0 and accelerator.is_main_process:
                periodic_path = os.path.join(checkpoint_dir, f'{train_steps:07d}.pt')
                unwrapped = accelerator.unwrap_model(model)
                torch.save({'model': unwrapped.state_dict(), 'opt': opt.state_dict(), 'ema': ema.state_dict(), 'train_steps': train_steps, 'best_val_loss': best_val_loss, 'epoch': epoch}, periodic_path)
                logger.info(f'[ckpt] periodic save → {periodic_path}')
            if train_steps >= _max_steps:
                break
        epoch += 1
        if val_loader is not None and epoch % val_every_n_epochs == 0:
            val_mse = evaluate_deterministic(model, val_loader, device, transport, vae, phase=_phase, val_seed=val_seed, val_n_batches=val_n_batches)
            accelerator.wait_for_everyone()
            val_mse_all = accelerator.gather(val_mse.detach().unsqueeze(0)).mean().item()
            if not (val_mse_all == val_mse_all and val_mse_all != float('inf') and (val_mse_all != float('-inf'))):
                raise RuntimeError(f'[NaN guard] non-finite val_mse at epoch={epoch}: {val_mse_all}')
            if accelerator.is_main_process:
                logger.info(f'[val] epoch={epoch} step={train_steps} mse={val_mse_all:.6f} best={best_val_loss:.6f}')
                writer.add_scalar('Loss/val_epoch', val_mse_all, epoch)
                record = {'epoch': epoch, 'step': train_steps, 'val_mse': val_mse_all, 'best_val_loss': best_val_loss, 'wall_time': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}
                with open(val_loss_log_path, 'a') as fh:
                    fh.write(json.dumps(record) + '\n')
                if val_mse_all < best_val_loss:
                    best_val_loss = val_mse_all
                    no_improve_epochs = 0
                    best_path = os.path.join(checkpoint_dir, 'best.pt')
                    unwrapped = accelerator.unwrap_model(model)
                    torch.save({'model': unwrapped.state_dict(), 'opt': opt.state_dict(), 'ema': ema.state_dict(), 'train_steps': train_steps, 'best_val_loss': best_val_loss, 'epoch': epoch}, best_path)
                    logger.info(f'[val] new best mse={best_val_loss:.6f} → saved {best_path}')
                else:
                    no_improve_epochs += 1
                    logger.info(f'[val] no improvement {no_improve_epochs}/{early_stop_patience}')
            _no_improve_t = torch.tensor(no_improve_epochs, device=device)
            dist.broadcast(_no_improve_t, src=0)
            no_improve_epochs = int(_no_improve_t.item())
            model.train()
            if epoch >= early_stop_min_epochs and no_improve_epochs >= early_stop_patience:
                _es_buf = torch.zeros(1, device=device, dtype=torch.int)
                if accelerator.is_main_process:
                    _es_buf.fill_(1)
                    logger.info(f'[val] EARLY STOP triggered at epoch {epoch}')
                dist.broadcast(_es_buf, src=0)
                if _es_buf.item() == 1:
                    _early_stop = True
        if train_steps >= _max_steps or _early_stop:
            break
    if accelerator.is_main_process:
        last_path = os.path.join(checkpoint_dir, 'last.pt')
        unwrapped = accelerator.unwrap_model(model)
        torch.save({'model': unwrapped.state_dict(), 'opt': opt.state_dict(), 'ema': ema.state_dict(), 'train_steps': train_steps}, last_path)
        logger.info(f'[end] saved last ckpt → {last_path}')
        best_pt_path = os.path.join(checkpoint_dir, 'best.pt')
        if not os.path.exists(best_pt_path):
            fail_marker = os.path.join(experiment_dir, '_FAIL')
            with open(fail_marker, 'w') as _f:
                _f.write(f'training ended without best.pt at {best_pt_path}\n')
            raise RuntimeError(f'[FAIL] sdvae-film training ended without best.pt at {best_pt_path}')
        logger.info('Done!')
    return accelerator
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SiT_M06_FiLM — online SDVAE encode + FiLM-advanced per-layer gates')
    parser.add_argument('--config', type=str, default='configs/debug.yaml', help='Path to YAML training config.')
    parser.add_argument('--vae_path', type=str, default='', help='Local diffusers AutoencoderKL cache directory.')
    args = parser.parse_args()
    train_config = load_config(args.config)
    train_config['_config_path'] = args.config
    if train_config['data'].get('latent_norm', False):
        raise RuntimeError('[sdvae-film] data.latent_norm=true in yaml. Online encoding uses raw VAE output × 0.18215; z-score norm would double-process the latent.')
    _lm = train_config['data'].get('latent_multiplier', 1.0)
    if abs(_lm - 1.0) > 1e-06:
        raise RuntimeError(f'[sdvae-film] data.latent_multiplier={_lm}. Must be 1.0.')
    _phase_early = train_config['data'].get('phase', 'warmup')
    if _phase_early not in PHASE_CHOICES:
        raise RuntimeError(f"[sdvae-film] data.phase='{_phase_early}' not in {PHASE_CHOICES}.")
    if 'weight_init' not in train_config.get('train', {}):
        raise RuntimeError('[sdvae-film] train.weight_init is required (path to warmup best.pt or SiT-XL-2-256.pt).')
    _wi = train_config['train']['weight_init']
    if not os.path.exists(_wi):
        raise FileNotFoundError(f'[sdvae-film] train.weight_init not found: {_wi}')
    train_config['data'].setdefault('vae_path', args.vae_path)
    _grad_accum = int(train_config.get('train', {}).get('gradient_accumulation_steps', 1))
    _ddp_bucket_cap_mb = train_config.get('train', {}).get('ddp_bucket_cap_mb', None)
    if _ddp_bucket_cap_mb is not None:
        from accelerate import DistributedDataParallelKwargs
        ddp_kwargs = DistributedDataParallelKwargs(bucket_cap_mb=_ddp_bucket_cap_mb, find_unused_parameters=False, gradient_as_bucket_view=True)
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], gradient_accumulation_steps=_grad_accum)
    else:
        accelerator = Accelerator(gradient_accumulation_steps=_grad_accum)
    do_train(train_config, accelerator)
