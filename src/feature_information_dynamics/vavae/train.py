"""
M06 upstream fork: LightningDiT train + FiLM-advanced per-layer gate injection.

DEVIATION (FiLM-advanced): per-layer scalar gates × 28 (one per LightningDiT block),
single proj pair at hidden_dim=1152. All gates zero-init for smooth start. Mirrors
RAE FiLM-advanced design (train_m06_upstream_rae_film_advanced.py) for VAVAE backbone.

Gold-standard reference: scripts/train_m06_upstream_rae_film_advanced.py (RAE, LIVE validated)
Vanilla VAVAE base:       scripts/train_m06_upstream.py

Key differences vs RAE FiLM-advanced:
  - Backbone: LightningDiT single stream (not RAE dual encoder-decoder)
  - hidden_dim = 1152 (no dec/enc split; single proj output dim)
  - num_gates = len(self.blocks) = 28 (not 30)
  - No [0,1] fix (VAVAE does not use DINOv2 encoder; RAE needs it, VAVAE does not)
  - Transport: LightningDiT SiT transport (use_cosine_loss / use_lognorm / alpha_train)

Phase semantics (identical to all M06 scripts):
  warmup:     mask=None, canny=None (adapter + gates frozen; identical to upstream)
  mask:       mask input active; canny dropped
  canny:      canny input active; mask dropped
  canny_mask: both mask and canny active

Modifications vs train_m06_upstream.py (minimal):
  1. VAVAE_M06_FiLM subclass replaces LightningDiT_M06
     - mask_proj_x / canny_proj_x Conv2d (Kaiming-uniform init)
     - mask_gates / canny_gates nn.Parameter (zero-init, per block)
     - forward inlines block loop with per-layer gate injection
  2. model = VAVAE_M06_FiLM(...) instead of LightningDiT_M06(...)
  3. load_pretrained_vavae_backbone: force-drops proj keys (preserve Kaiming init)
  4. Phase-aware freeze: FiLM-advanced variant freezes proj + gates together
  5. update_ema: strip _orig_mod. prefix (torch.compile compatibility)
  6. All dataset / transport / optimizer / val / early-stop logic unchanged from base
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cuda
import torch.backends.cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import math
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
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from models.lightningdit import LightningDiT_models, LightningDiT
from transport import create_transport, Sampler
from accelerate import Accelerator
from datasets.img_latent_dataset import ImgLatentDataset, PixelImageNetDataset
from safetensors import safe_open
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from feature_information_dynamics.data.whitelisted_img_latent_dataset import WhitelistedImgLatentDataset
from feature_information_dynamics.data.sharded_mask_dataset import MaskedShardedDataset
from feature_information_dynamics.data.sharded_mask_canny_dataset import MaskedCannyShardedDataset
PHASE_CHOICES: tuple[str, ...] = ('warmup', 'mask', 'canny', 'canny_mask')
_DEFAULT_CANNY_SHARD_DIR = ''

class VAVAE_M06_FiLM(LightningDiT):
    """LightningDiT subclass with FiLM-advanced per-layer mask/canny conditioning.

    DEVIATION (FiLM-advanced): per-layer scalar gates × num_blocks (=28 for XL/1).
    Mirrors RAE FiLM-advanced design adapted for LightningDiT single stream.

    Architecture:
      mask_proj_x  = Conv2d(1, hidden, kernel=mask_kernel, stride=mask_kernel, bias=False)
      canny_proj_x = Conv2d(1, hidden, kernel=mask_kernel, stride=mask_kernel, bias=False)
          -> maps [B, 1, 256, 256] -> [B, hidden, tokens_per_side, tokens_per_side]
          -> flatten+transpose -> [B, N_tokens, hidden]

      mask_gates   = nn.Parameter(zeros(num_blocks))  # per-layer scalar gate
      canny_gates  = nn.Parameter(zeros(num_blocks))

    Forward: for each block i in self.blocks:
        x = block(x, c)
        if mask_full is not None:
            x = x + mask_gates[i] * mask_full   # [B, N, hidden]
        if canny_full is not None:
            x = x + canny_gates[i] * canny_full  # [B, N, hidden]

    Gates zero-init: at step-0 forward is numerically identical to upstream.
    When mask=None and canny=None (warmup phase), forward is identical to upstream.

    Projection init: Kaiming-uniform (not zeros). Combined with gates=0, this
    ensures step-0 identity while allowing meaningful gradients from step 1
    (zeros proj + zeros gates would dead-lock gradient flow for proj weights).
    """

    def __init__(self, *args, **kwargs) -> None:
        """Build LightningDiT backbone + Kaiming-init proj + zero-init per-layer gates."""
        super().__init__(*args, **kwargs)
        hidden: int = self.x_embedder.proj.out_channels
        tokens_per_side: int = int(self.x_embedder.num_patches ** 0.5)
        mask_kernel: int = 256 // tokens_per_side
        assert mask_kernel * tokens_per_side == 256, f'mask_kernel {mask_kernel} × tokens_per_side {tokens_per_side} != 256'
        self.mask_proj_x = nn.Conv2d(1, hidden, kernel_size=mask_kernel, stride=mask_kernel, bias=False)
        self.canny_proj_x = nn.Conv2d(1, hidden, kernel_size=mask_kernel, stride=mask_kernel, bias=False)
        nn.init.kaiming_uniform_(self.mask_proj_x.weight, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.canny_proj_x.weight, a=5 ** 0.5)
        num_blocks: int = len(self.blocks)
        self.mask_gates = nn.Parameter(torch.zeros(num_blocks))
        self.canny_gates = nn.Parameter(torch.zeros(num_blocks))

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor]=None, canny: Optional[torch.Tensor]=None) -> torch.Tensor:
        """Forward: LightningDiT block loop + FiLM-advanced per-layer gate injection.

        Inlines parent block loop (lightningdit.py forward) to inject mask/canny
        conditioning via per-layer scalar gates across all 28 blocks (XL/1).
        Using super().forward() is not feasible here as the parent has no injection hook.

        Args:
            x:     [B, in_chans, H, W] latent input (16x16 spatial after VAVAE)
            t:     [B] diffusion timesteps (SiT convention: t=1=clean, t=0=noise)
            y:     [B] class labels
            mask:  [B, 256, 256] or [B, 1, 256, 256] binary mask, or None
            canny: [B, 256, 256] or [B, 1, 256, 256] canny edges, or None

        Returns:
            [B, out_chans, H, W] velocity prediction
        """
        from torch.utils.checkpoint import checkpoint as grad_ckpt
        use_checkpoint = self.use_checkpoint
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
        x = self.x_embedder(x) + self.pos_embed
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y, self.training)
        c = t_emb + y_emb
        for i, block in enumerate(self.blocks):
            if use_checkpoint:
                x = grad_ckpt(block, x, c, self.feat_rope, use_reentrant=True)
            else:
                x = block(x, c, self.feat_rope)
            if mask_full is not None:
                x = x + self.mask_gates[i] * mask_full
            if canny_full is not None:
                x = x + self.canny_gates[i] * canny_full
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        return x

def load_pretrained_vavae_backbone(model: VAVAE_M06_FiLM, ckpt_path: str, rank: int=0) -> VAVAE_M06_FiLM:
    """Load VAVAE/LightningDiT pretrain weights; adapter keys stay at Kaiming init.

    Key priority: 'ema' (clean) -> 'model' -> flat.  Strips 'module.' / '_orig_mod.'
    DDP prefixes.  Force-drops mask_proj_x / canny_proj_x / *_gates from ckpt so
    FiLM-advanced Kaiming init is preserved (gates=0 + proj=Kaiming for gradient flow).

    This is the FiLM-advanced equivalent of load_pretrained_rae_backbone in the RAE script.

    Args:
        model:     VAVAE_M06_FiLM instance.
        ckpt_path: Path to upstream LightningDiT checkpoint.
        rank:      Logging rank (only rank 0 prints).

    Returns:
        model with backbone weights loaded (in-place), same object returned.
    """
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f'VAVAE pretrain ckpt not found: {path}')
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
    src_sd = {k.replace('module.', '').removeprefix('_orig_mod.'): v for k, v in src_sd.items() if isinstance(v, torch.Tensor)}
    _adapter_keys = ('mask_proj_x.weight', 'canny_proj_x.weight', 'mask_gates', 'canny_gates')
    _dropped = [k for k in _adapter_keys if k in src_sd]
    for k in _dropped:
        del src_sd[k]
    if _dropped and rank == 0:
        print(f'[vavae_ckpt] DROP {_dropped} from ckpt (FiLM-advanced: preserve Kaiming+zero-gates init)')
    result = model.load_state_dict(src_sd, strict=False)
    if rank == 0:
        _adapter_pfx = ('mask_proj_x.', 'canny_proj_x.', 'mask_gates', 'canny_gates')
        non_adapter_missing = [k for k in result.missing_keys if not any((k.startswith(p) for p in _adapter_pfx))]
        if non_adapter_missing:
            print(f'[vavae_ckpt] unexpected missing ({len(non_adapter_missing)}): {non_adapter_missing[:6]}')
        print(f"[vavae_ckpt] key='{key_used}' missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}")
    return model

@torch.no_grad()
def evaluate_deterministic(model: nn.Module, val_loader: DataLoader, device: torch.device, transport: object, val_seed: int=12345, val_n_batches: int=8, phase: str='mask') -> torch.Tensor:
    """Deterministic val: MSE-only loss, fixed seed per batch.

    Handles four batch shapes returned by datasets:
      - dict with 'latent'/'label' keys
      - tuple (latent, label, mask, canny)
      - tuple (latent, label, mask)
      - tuple (latent, label)

    Args:
        model:         The model to evaluate (may be DDP-wrapped).
        val_loader:    DataLoader yielding batches from a dataset.
        device:        CUDA device.
        transport:     Transport object with training_losses method.
        val_seed:      Base seed; batch i uses val_seed + i for determinism.
        val_n_batches: Maximum batches to evaluate.
        phase:         Training phase for model_kwargs construction.

    Returns:
        Scalar MSE loss tensor on ``device``, NOT yet all-reduced across ranks.
    """
    model.eval()
    total_mse = torch.tensor(0.0, device=device)
    n_seen = 0
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= val_n_batches:
            break
        if isinstance(batch, dict):
            x = batch['latent'].to(device)
            y = batch['label'].to(device)
            mask, canny_t = (None, None)
        elif len(batch) == 4:
            x, y, mask, canny_t = batch
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device, non_blocking=True)
            canny_t = canny_t.to(device, non_blocking=True)
        elif len(batch) == 3:
            x, y, mask = batch
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device, non_blocking=True)
            canny_t = None
        else:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            mask, canny_t = (None, None)
        torch.manual_seed(val_seed + batch_idx)
        torch.cuda.manual_seed_all(val_seed + batch_idx)
        model_kwargs: dict = {'y': y}
        if mask is not None and phase in ('mask', 'canny_mask'):
            model_kwargs['mask'] = mask
        if canny_t is not None and phase in ('canny', 'canny_mask'):
            model_kwargs['canny'] = canny_t
        loss_dict = transport.training_losses(model, x, model_kwargs)
        mse = loss_dict['loss'].mean()
        total_mse += mse
        n_seen += 1
    if n_seen == 0:
        return torch.tensor(0.0, device=device)
    return total_mse / n_seen

def spawn_fid_eval(ckpt_path: str, experiment_dir: str, train_config: dict) -> None:
    raise NotImplementedError('FID hooks are separate; disable train.fid_every and use the upstream FID tool explicitly.')

def load_weights_with_shape_check(model: nn.Module, checkpoint: dict, rank: int=0) -> nn.Module:
    """Load weights from checkpoint, skipping mismatched shapes.

    Args:
        model:      Target model.
        checkpoint: Dict with 'model' key containing state_dict.
        rank:       Logging rank.

    Returns:
        model with weights loaded in-place.
    """
    model_state_dict = model.state_dict()
    for name, param in checkpoint['model'].items():
        if name in model_state_dict:
            if param.shape == model_state_dict[name].shape:
                model_state_dict[name].copy_(param)
            elif name == 'x_embedder.proj.weight':
                weight = torch.zeros_like(model_state_dict[name])
                weight[:, :16] = param[:, :16]
                model_state_dict[name] = weight
            elif rank == 0:
                print(f"Skipping '{name}': ckpt shape {param.shape}, model shape {model_state_dict[name].shape}")
        elif rank == 0:
            print(f"Parameter '{name}' not found in model, skipping.")
    model.load_state_dict(model_state_dict, strict=False)
    return model

@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float=0.9999) -> None:
    """Step the EMA model towards the current model.

    Strips both 'module.' (DDP) and '_orig_mod.' (torch.compile) prefixes.

    Args:
        ema_model: EMA copy of the model.
        model:     Live model (may have DDP or compile prefixes).
        decay:     EMA decay rate.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        name = name.replace('module.', '').removeprefix('_orig_mod.')
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model: nn.Module, flag: bool=True) -> None:
    """Set requires_grad flag for all parameters in a model.

    Args:
        model: The model.
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
    """Create a logger that writes to a log file and stdout.

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
    """Train VAVAE_M06_FiLM (LightningDiT + FiLM-advanced per-layer gates, M06 protocol)."""
    device = accelerator.device
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
    rank = accelerator.local_process_index
    data_type = train_config['data'].get('type', 'latent')
    if data_type == 'online_rae':
        latent_size = 16
    else:
        downsample_ratio = train_config['vae'].get('downsample_ratio', 16) if 'vae' in train_config else 16
        assert train_config['data']['image_size'] % downsample_ratio == 0, 'image_size must be divisible by downsample_ratio'
        latent_size = train_config['data']['image_size'] // downsample_ratio
    _model_type = train_config['model'].get('model_type', 'LightningDiT-XL/2')
    _patch_size = int(_model_type.split('/')[-1])
    if accelerator.is_main_process:
        logger.info(f'[M06-FiLM] model_type={_model_type} → patch_size={_patch_size}')
    model = VAVAE_M06_FiLM(input_size=latent_size, patch_size=_patch_size, num_classes=train_config['data']['num_classes'], use_qknorm=train_config['model']['use_qknorm'], use_swiglu=train_config['model'].get('use_swiglu', False), use_rope=train_config['model'].get('use_rope', False), use_rmsnorm=train_config['model'].get('use_rmsnorm', False), wo_shift=train_config['model'].get('wo_shift', False), in_channels=train_config['model'].get('in_chans', 4), use_checkpoint=train_config['model'].get('use_checkpoint', False), class_dropout_prob=train_config['model'].get('class_dropout_prob', 0.1))
    ema = deepcopy(model).to(device)
    _init_best_val_loss = float('inf')
    _wi_path = train_config['train'].get('weight_init') or train_config.get('ckpt_path')
    if _wi_path:
        if accelerator.is_main_process and 'weight_init' not in train_config['train']:
            logger.warning(f'[I-002] train.weight_init missing; falling back to top-level ckpt_path={_wi_path}')
        raw = torch.load(_wi_path, map_location='cpu', weights_only=False)
        _bvl = raw.get('best_val_loss', None) if isinstance(raw, dict) else None
        if _bvl is not None and isinstance(_bvl, (int, float)) and (_bvl > 0):
            _init_best_val_loss = float(_bvl)
        _adapter_keys_check = ('mask_proj_x.', 'canny_proj_x.', 'mask_gates', 'canny_gates')
        _is_m06_ckpt = isinstance(raw, dict) and 'model' in raw and any((k.startswith(pk) for k in raw['model'].keys() for pk in _adapter_keys_check))
        if _is_m06_ckpt:
            ckpt_data = dict(raw)
            ckpt_data['model'] = {k.replace('module.', '').removeprefix('_orig_mod.'): v for k, v in ckpt_data['model'].items()}
            model = load_weights_with_shape_check(model, ckpt_data, rank=rank)
            ema = load_weights_with_shape_check(ema, ckpt_data, rank=rank)
        else:
            model = load_pretrained_vavae_backbone(model, _wi_path, rank=rank)
            ema = load_pretrained_vavae_backbone(ema, _wi_path, rank=rank)
        if accelerator.is_main_process:
            logger.info(f'Loaded weight_init from {_wi_path} (m06_ckpt={_is_m06_ckpt})')
            if _init_best_val_loss < float('inf'):
                logger.info(f'[chained] inherited best_val_loss = {_init_best_val_loss:.6f}')
    elif accelerator.is_main_process:
        logger.warning('[I-002] No weight_init or ckpt_path set — TRAINING FROM SCRATCH')
    requires_grad(ema, False)
    model = model.to(device)
    transport = create_transport(train_config['transport']['path_type'], train_config['transport']['prediction'], train_config['transport']['loss_weight'], train_config['transport']['train_eps'], train_config['transport']['sample_eps'], use_cosine_loss=train_config['transport'].get('use_cosine_loss', False), use_lognorm=train_config['transport'].get('use_lognorm', False), alpha_train=train_config['transport'].get('alpha_train', 1.0))
    if accelerator.is_main_process:
        total_params = sum((p.numel() for p in model.parameters())) / 1000000.0
        adapter_params = sum((p.numel() for p in model.mask_proj_x.parameters())) + sum((p.numel() for p in model.canny_proj_x.parameters())) + model.mask_gates.numel() + model.canny_gates.numel()
        logger.info(f'[M06-FiLM] VAVAE_M06_FiLM Parameters: {total_params:.2f}M (adapter: {adapter_params / 1000000.0:.4f}M | num_gates={len(model.mask_gates)} per modality)')
        logger.info(f"Optimizer: AdamW, lr={train_config['optimizer']['lr']}, beta2={train_config['optimizer']['beta2']}")
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
        logger.info(f'[M06-FiLM] param groups: backbone={len(_backbone_params)} params @ lr={_base_lr}, adapters={len(_adapter_params)} params @ lr={_base_lr * _adapter_mult} (multiplier={_adapter_mult})')
    opt = torch.optim.AdamW([{'params': _backbone_params, 'lr': _base_lr}, {'params': _adapter_params, 'lr': _base_lr * _adapter_mult}], weight_decay=0, betas=(0.9, train_config['optimizer']['beta2']))
    sr95_filter = train_config['data'].get('sr95_filter', False)
    data_split = train_config['data'].get('split', 'train')
    _paired_wl = train_config['data'].get('paired_train_whitelist', None)
    _dataset_class = train_config['data'].get('dataset_class', 'auto')
    _phase = train_config['data'].get('phase', 'warmup')
    if data_type in ('pixel', 'online_rae'):
        dataset = PixelImageNetDataset(root=train_config['data']['data_path'], sr95_filter=sr95_filter, split=data_split, image_size=train_config['data'].get('image_size', 256), use_hflip=train_config['data'].get('use_hflip', False))
    elif _dataset_class == 'sharded_mask_canny':
        _canny_shard_dir = train_config['data'].get('canny_shard_dir', None)
        _default_mask_shard_dir = ''
        if _phase in ('canny', 'canny_mask') and _canny_shard_dir is None:
            logging.warning('[M06-FiLM] phase=%s requires data.canny_shard_dir but not set — defaulting to %s', _phase, _DEFAULT_CANNY_SHARD_DIR)
            _canny_shard_dir = _DEFAULT_CANNY_SHARD_DIR
        elif _canny_shard_dir is None:
            _canny_shard_dir = _DEFAULT_CANNY_SHARD_DIR
        dataset = MaskedCannyShardedDataset(data_dir=train_config['data']['data_path'], mask_shard_dir=train_config['data'].get('mask_shard_dir', _default_mask_shard_dir), canny_shard_dir=_canny_shard_dir, latent_norm=train_config['data'].get('latent_norm', False), latent_multiplier=train_config['data'].get('latent_multiplier', 0.18215), sr95_filter=sr95_filter, split=data_split, paired_whitelist_json=_paired_wl, phase=_phase)
    elif _dataset_class == 'sharded_mask':
        _default_mask_shard_dir = ''
        dataset = MaskedShardedDataset(data_dir=train_config['data']['data_path'], mask_shard_dir=train_config['data'].get('mask_shard_dir', _default_mask_shard_dir), latent_norm=train_config['data'].get('latent_norm', False), latent_multiplier=train_config['data'].get('latent_multiplier', 0.18215), sr95_filter=sr95_filter, split=data_split, paired_whitelist_json=_paired_wl, phase=_phase)
    else:
        _ds_kwargs = dict(data_dir=train_config['data']['data_path'], latent_norm=train_config['data'].get('latent_norm', False), latent_multiplier=train_config['data'].get('latent_multiplier', 0.18215), sr95_filter=sr95_filter, split=data_split)
        if _paired_wl is not None:
            dataset = WhitelistedImgLatentDataset(**_ds_kwargs, paired_whitelist_json=_paired_wl)
        else:
            dataset = ImgLatentDataset(**_ds_kwargs)
    batch_size_per_gpu = int(np.round(train_config['train']['global_batch_size'] / accelerator.num_processes))
    global_batch_size = batch_size_per_gpu * accelerator.num_processes
    loader = DataLoader(dataset, batch_size=batch_size_per_gpu, shuffle=True, num_workers=train_config['data']['num_workers'], pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=4)
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} samples {train_config['data']['data_path']}")
        logger.info(f'Batch size {batch_size_per_gpu} per gpu, global batch size {global_batch_size}')
    _dataset_class_eff = _dataset_class
    val_loader = None
    val_n_batches = 0
    if _dataset_class_eff in ('sharded_mask', 'sharded_mask_canny'):
        _paired_val_wl = train_config['data'].get('paired_val_whitelist', None)
        if _paired_val_wl is not None:
            if _dataset_class_eff == 'sharded_mask_canny':
                val_dataset = MaskedCannyShardedDataset(data_dir=train_config['data']['data_path'], mask_shard_dir=train_config['data'].get('mask_shard_dir', _default_mask_shard_dir), canny_shard_dir=_canny_shard_dir, latent_norm=train_config['data'].get('latent_norm', False), latent_multiplier=train_config['data'].get('latent_multiplier', 0.18215), sr95_filter=sr95_filter, split=data_split, paired_whitelist_json=_paired_val_wl, phase=_phase, is_val=True)
            else:
                val_dataset = MaskedShardedDataset(data_dir=train_config['data']['data_path'], mask_shard_dir=train_config['data'].get('mask_shard_dir', _default_mask_shard_dir), latent_norm=train_config['data'].get('latent_norm', False), latent_multiplier=train_config['data'].get('latent_multiplier', 0.18215), sr95_filter=sr95_filter, split=data_split, paired_whitelist_json=_paired_val_wl, phase=_phase, is_val=True)
            val_n_batches = train_config['train'].get('val_n_batches', 8)
            val_loader = DataLoader(val_dataset, batch_size=batch_size_per_gpu, shuffle=False, num_workers=2, pin_memory=True, drop_last=False, persistent_workers=False)
            if accelerator.is_main_process:
                _val_seed_log = train_config['train'].get('val_seed', 12345)
                logger.info(f'[val] {_dataset_class_eff} dataset N={len(val_dataset):,} | n_batches={val_n_batches} | seed={_val_seed_log}')
    elif 'valid_path' in train_config['data']:
        if data_type in ('pixel', 'online_rae'):
            valid_dataset_full = PixelImageNetDataset(root=train_config['data']['valid_path'], sr95_filter=sr95_filter, split='val', image_size=train_config['data'].get('image_size', 256))
        else:
            valid_dataset_full = ImgLatentDataset(data_dir=train_config['data']['valid_path'], latent_norm=train_config['data'].get('latent_norm', False), latent_multiplier=train_config['data'].get('latent_multiplier', 0.18215), sr95_filter=sr95_filter, split='val')
        val_subset_size = train_config['data'].get('val_subset_size', 5000)
        if 0 < val_subset_size < len(valid_dataset_full):
            valid_dataset = Subset(valid_dataset_full, list(range(val_subset_size)))
        else:
            valid_dataset = valid_dataset_full
        val_loader = DataLoader(valid_dataset, batch_size=batch_size_per_gpu, shuffle=False, num_workers=train_config['data']['num_workers'], pin_memory=True, drop_last=False, persistent_workers=True, prefetch_factor=2)
        val_n_batches = len(val_loader)
        if accelerator.is_main_process:
            logger.info(f"Validation Dataset ({data_type}): {len(valid_dataset):,} samples from {train_config['data']['valid_path']}")
    update_ema(ema, model, decay=0)
    model.train()
    ema.eval()

    def _freeze_mask_adapters() -> None:
        """Freeze mask_proj_x and mask_gates."""
        model.mask_proj_x.weight.requires_grad_(False)
        model.mask_gates.requires_grad_(False)

    def _freeze_canny_adapters() -> None:
        """Freeze canny_proj_x and canny_gates."""
        model.canny_proj_x.weight.requires_grad_(False)
        model.canny_gates.requires_grad_(False)

    def _log_trainable() -> None:
        """Print trainable=True/False for each adapter param (freeze audit)."""
        if accelerator.is_main_process:
            for n, p in model.named_parameters():
                if any((ak in n for ak in ('mask_proj_x', 'canny_proj_x', 'mask_gates', 'canny_gates'))):
                    logger.info(f'[M06-FiLM][freeze-audit] {n}: trainable={p.requires_grad}')
    if _phase == 'mask':
        _freeze_canny_adapters()
        if accelerator.is_main_process:
            logger.info('[M06-FiLM] phase=mask -> mask_proj_x + mask_gates trainable, canny_proj_x + canny_gates frozen')
        _log_trainable()
    elif _phase == 'canny':
        _freeze_mask_adapters()
        if accelerator.is_main_process:
            logger.info('[M06-FiLM] phase=canny -> canny_proj_x + canny_gates trainable, mask_proj_x + mask_gates frozen')
        _log_trainable()
    elif _phase == 'canny_mask':
        if accelerator.is_main_process:
            logger.info('[M06-FiLM] phase=canny_mask -> mask_proj_x + mask_gates + canny_proj_x + canny_gates trainable')
        _log_trainable()
    else:
        _freeze_mask_adapters()
        _freeze_canny_adapters()
        if accelerator.is_main_process:
            logger.info(f'[M06-FiLM] phase={_phase} -> all adapters (proj + gates) frozen (no conditioning input; avoids find_unused_parameters crash)')
        _log_trainable()
    model, opt, loader = accelerator.prepare(model, opt, loader)
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)
    train_config['train']['resume'] = train_config['train'].get('resume', False)
    if train_config['train']['resume']:
        ckpt_files = glob(f'{checkpoint_dir}/*.pt')
        if ckpt_files:
            ckpt_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
            latest_ckpt = ckpt_files[-1]
            ckpt = torch.load(latest_ckpt, map_location=lambda storage, loc: storage)
            model.load_state_dict(ckpt['model'])
            ema.load_state_dict(ckpt['ema'])
            train_steps = int(latest_ckpt.split('/')[-1].split('.')[0])
            if accelerator.is_main_process:
                logger.info(f'Resuming from checkpoint: {latest_ckpt}')
        elif accelerator.is_main_process:
            logger.info('No checkpoint found. Starting from scratch.')
    if not train_config['train']['resume']:
        train_steps = 0
    log_steps = 0
    running_loss = 0.0
    start_time = time()
    epoch = 0
    best_val_loss = _init_best_val_loss
    no_improve_epochs = 0
    val_seed = train_config['train'].get('val_seed', 12345)
    early_stop_patience = train_config['train'].get('early_stop_patience', 5)
    early_stop_min_epochs = train_config['train'].get('early_stop_min_epochs', 10)
    val_every_n_epochs = train_config['train'].get('val_every_n_epochs', 1)
    val_loss_log_path = os.path.join(experiment_dir, 'val_loss_log.jsonl') if accelerator.is_main_process else None
    if accelerator.is_main_process:
        logger.info(f"Training for up to {train_config['train']['max_steps']} steps...")
    _early_stop = False
    while True:
        for batch in loader:
            mask_for_model = None
            canny_for_model = None
            if isinstance(batch, dict):
                x, y = (batch['latent'], batch['label'])
            elif len(batch) == 4:
                x, y, mask_for_model, canny_for_model = batch
            elif len(batch) == 3:
                x, y, mask_for_model = batch
            else:
                x, y = batch
            if accelerator.mixed_precision == 'no':
                x = x.to(device, dtype=torch.float32)
                y = y
            else:
                x = x.to(device)
                y = y.to(device)
            if mask_for_model is not None:
                mask_for_model = mask_for_model.to(device, non_blocking=True)
            if canny_for_model is not None:
                canny_for_model = canny_for_model.to(device, non_blocking=True)
            model_kwargs: dict = {'y': y}
            if mask_for_model is not None and _phase in ('mask', 'canny_mask'):
                model_kwargs['mask'] = mask_for_model
            if canny_for_model is not None and _phase in ('canny', 'canny_mask'):
                model_kwargs['canny'] = canny_for_model
            loss_dict = transport.training_losses(model, x, model_kwargs)
            if 'cos_loss' in loss_dict:
                mse_loss = loss_dict['loss'].mean()
                loss = loss_dict['cos_loss'].mean() + mse_loss
            else:
                loss = loss_dict['loss'].mean()
            if not torch.isfinite(loss).all():
                raise RuntimeError(f'[NaN guard] non-finite loss at step={train_steps}: {loss.item()}. Lower lr or strengthen grad clip and restart.')
            opt.zero_grad()
            accelerator.backward(loss)
            if 'max_grad_norm' in train_config['optimizer']:
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), train_config['optimizer']['max_grad_norm'])
            opt.step()
            update_ema(ema, model.module)
            if 'cos_loss' in loss_dict:
                running_loss += mse_loss.item()
            else:
                running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % train_config['train']['log_every'] == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                if accelerator.is_main_process:
                    logger.info(f'(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Steps/Sec: {steps_per_sec:.2f}')
                    writer.add_scalar('Loss/train', avg_loss, train_steps)
                running_loss = 0.0
                log_steps = 0
                start_time = time()
            if train_steps >= train_config['train']['max_steps']:
                break
        epoch += 1
        if val_loader is not None and epoch % val_every_n_epochs == 0:
            val_mse = evaluate_deterministic(model, val_loader, device, transport, val_seed=val_seed, val_n_batches=val_n_batches, phase=_phase)
            accelerator.wait_for_everyone()
            val_mse_all = accelerator.gather(val_mse.detach().unsqueeze(0)).mean().item()
            if not (val_mse_all == val_mse_all and val_mse_all not in (float('inf'), float('-inf'))):
                raise RuntimeError(f'[NaN guard] non-finite val_mse at epoch={epoch}: {val_mse_all}. Lower lr and restart.')
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
                    logger.info(f'[val] new best mse={best_val_loss:.6f} -> saved {best_path}')
                else:
                    no_improve_epochs += 1
                    logger.info(f'[val] no improvement {no_improve_epochs}/{early_stop_patience}')
            _no_improve_t = torch.tensor(no_improve_epochs, device=device)
            dist.broadcast(_no_improve_t, src=0)
            no_improve_epochs = int(_no_improve_t.item())
            model.train()
            if epoch >= early_stop_min_epochs and no_improve_epochs >= early_stop_patience:
                if accelerator.is_main_process:
                    logger.info(f'[val] EARLY STOP triggered at epoch {epoch} (patience {early_stop_patience} reached)')
                _early_stop = True
        if train_steps >= train_config['train']['max_steps'] or _early_stop:
            break
    if accelerator.is_main_process:
        last_path = os.path.join(checkpoint_dir, 'last.pt')
        unwrapped = accelerator.unwrap_model(model)
        torch.save({'model': unwrapped.state_dict(), 'opt': opt.state_dict(), 'ema': ema.state_dict(), 'train_steps': train_steps}, last_path)
        logger.info(f'[end] saved last ckpt -> {last_path}')
        best_pt_path = os.path.join(checkpoint_dir, 'best.pt')
        if not os.path.exists(best_pt_path):
            fail_marker = os.path.join(experiment_dir, '_FAIL')
            with open(fail_marker, 'w') as _f:
                _f.write(f'training ended without best.pt at {best_pt_path}\n')
            raise RuntimeError(f'[FAIL] training ended without best.pt at {best_pt_path}')
        logger.info('Done!')
    return accelerator
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/debug_vavae_m06_film_advanced.yaml', help='Path to YAML training config.')
    args = parser.parse_args()
    train_config = load_config(args.config)
    train_config['_config_path'] = args.config
    _ddp_bucket_cap_mb = train_config.get('train', {}).get('ddp_bucket_cap_mb', None)
    if _ddp_bucket_cap_mb is not None:
        from accelerate import DistributedDataParallelKwargs
        ddp_kwargs = DistributedDataParallelKwargs(bucket_cap_mb=_ddp_bucket_cap_mb, find_unused_parameters=False, gradient_as_bucket_view=False)
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    else:
        accelerator = Accelerator()
    do_train(train_config, accelerator)
