"""
M06 upstream fork: RAE/DiTwDDTHead train — FiLM-advanced ablation.

DEVIATION (FiLM-advanced): per-layer scalar gates × 30 (28 enc + 2 dec), proj output
dec_h=2048, encoder uses [:enc_h] slice.  Single proj pair for both streams.
All gates zero-init for smooth start.  Ablation vs film_simple.

Original: M06 mask/canny-conditioning adapter.

Raw-image + online RAE encode pipeline. Inputs raw ImageNet images from
RawImageMaskCannyDataset; encodes them with RAE stage1 (frozen) online per batch;
feeds latents into RAE_M06 (DiTwDDTHead subclass with mask_proj_x Conv2d adapter).

Phase semantics:
  warmup:     mask and canny inputs dropped (mask_proj_x frozen); identical to upstream.
  mask:       mask input active; canny dropped.
  canny:      canny input active; mask dropped.
  canny_mask: both mask and canny active.

Forked from: third_party/RAE/src/train.py (OmegaConf + DDP variant)
Gold-standard reference: scripts/train_m06_upstream.py (VAVAE/LightningDiT)
Second reference:        scripts/train_m06_upstream_sdvae.py (SDVAE/SiT)
Legacy RAE reference:    scripts/train_rae_mask_patch.py (DiTwDDTHead_MaskPatch)

RAE specifics vs VAVAE/SDVAE versions:
  - Backbone: DiTwDDTHead (NOT LightningDiT)
    dual encoder-decoder: enc hidden=1152 (depth=28), dec hidden=2048 (depth=2)
  - latent shape: [B, 768, 16, 16]  (DINOv2-B RAE; 768-dim, 16x16 spatial)
  - tokens_per_side = 16 → 256 tokens (patch_size=1, input_size=16)
  - mask_kernel = 256 // 16 = 16 (same value as SDVAE)
  - Model config: loaded from OmegaConf yaml stage_2.params dict
  - RAE transport: stage2.transport.create_transport (different from VAVAE)
    adds time_dist_type + time_dist_shift; no use_cosine_loss/use_lognorm
  - Pretrain ckpt: stage2_model.pt -> prefers 'ema' key, fallback 'model', flat
  - mask injection: Conv2d on raw 256x256 mask -> [B, dec_h, 16, 16] -> add to
    x_tok (decoder stream) ONLY. Encoder stream untouched in mask phase.

t-axis convention note (from project memory):
  - RAE upstream uses t=0=clean convention in its path sampler.
  - This fork uses RAE's own stage2.transport end-to-end; no t-axis flip needed.
  - The t_rae=1-t + return -net() flip is ONLY needed in *eval* wrappers that
    interface with project eval_lib (t=1=clean convention).

Modifications vs upstream (minimal):
  1. RAE_M06 subclass of DiTwDDTHead added (mask_proj_x Conv2d, zeros-init)
  2. model = RAE_M06(**model_params) instead of DiTwDDTHead(**model_params)
  3. weight_init dispatch: paper RAE ckpt (ema/model key) vs M06 ckpt (mask_proj_x)
  4. Phase-aware mask_proj_x freeze (warmup/canny->frozen, mask/canny_mask->trainable)
  5. RawImageMaskCannyDataset wired (raw images + sharded mask/canny)
  6. RAE stage1 instantiated once from yaml; frozen eval-mode encoder
  7. Online encode: x_rgb -> rae_stage1.encode(x_rgb) -> latent per batch
  8. val_loader build + evaluate_deterministic
  9. Per-epoch early-stop loop
  10. val_every_n_epochs field (default 1)
"""
from __future__ import annotations
from typing import Optional
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cuda
import torch.backends.cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import sys
import yaml
import json
import numpy as np
import logging
import os
import argparse
import subprocess
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from glob import glob
from time import time
from omegaconf import OmegaConf
from accelerate import Accelerator
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
from stage2.models.DDT import DiTwDDTHead
from stage2.transport import create_transport
from utils.model_utils import instantiate_from_config
sys.path.insert(0, str(_SCRIPT_DIR))
from feature_information_dynamics.data.raw_image_mask_canny_dataset import RawImageMaskCannyDataset
RAE_ENC_HIDDEN: int = 1152
RAE_DEC_HIDDEN: int = 2048
RAE_TOKENS: int = 256
MASK_KERNEL: int = 16
RAE_IN_CHANNELS: int = 768
_RAE_PRETRAIN_CKPT = ''
_RAE_CONFIG_YAML = ''
PHASE_CHOICES = ('warmup', 'mask', 'canny', 'canny_mask')

class RAE_M06(DiTwDDTHead):
    """DiTwDDTHead subclass with M06 mask/canny conditioning — FiLM-advanced variant.

    DEVIATION (FiLM-advanced): per-layer scalar gates × 30 (28 enc + 2 dec),
    proj output dec_h=2048, encoder uses [:enc_h] slice. Single proj pair for both
    streams. All gates zero-init for smooth start. Ablation vs film_simple.

    Architecture:
      mask_proj_x  = Conv2d(1, dec_h=2048, kernel=16, stride=16, bias=False)
      canny_proj_x = Conv2d(1, dec_h=2048, kernel=16, stride=16, bias=False)
          -> maps [B, 1, 256, 256] -> [B, dec_h, 16, 16]
          -> flatten+transpose -> [B, 256, dec_h] full-dim token feature

      mask_gates   = nn.Parameter(zeros(num_blocks=30))  # per-layer scalar gate
      canny_gates  = nn.Parameter(zeros(num_blocks=30))

    For encoder blocks (i < num_encoder_blocks=28):
        s_tok += mask_gates[i] * mask_full[..., :enc_h]  # [B, 256, enc_h]
    For decoder blocks (i >= num_encoder_blocks):
        x_tok += mask_gates[i] * mask_full               # [B, 256, dec_h]

    Gates zero-init: at step-0 forward is identical to upstream. Gates learned
    during mask/canny_mask phases, one per layer, enabling the model to learn which
    layers benefit most from conditioning signal.

    When mask=None and canny=None (warmup phase), forward is numerically identical
    to upstream DiTwDDTHead.
    """

    def __init__(self, **kwargs) -> None:
        """Build DiTwDDTHead backbone + Kaiming-init proj + zero-init per-layer gates."""
        super().__init__(**kwargs)
        dec_h = self.decoder_hidden_size
        enc_h = self.s_embedder.proj.weight.shape[0]
        self.mask_proj_x = nn.Conv2d(1, dec_h, kernel_size=MASK_KERNEL, stride=MASK_KERNEL, bias=False)
        self.canny_proj_x = nn.Conv2d(1, dec_h, kernel_size=MASK_KERNEL, stride=MASK_KERNEL, bias=False)
        nn.init.kaiming_uniform_(self.mask_proj_x.weight, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.canny_proj_x.weight, a=5 ** 0.5)
        self.mask_gates = nn.Parameter(torch.zeros(self.num_blocks))
        self.canny_gates = nn.Parameter(torch.zeros(self.num_blocks))
        self._enc_h: int = enc_h

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, s: Optional[torch.Tensor]=None, mask: Optional[torch.Tensor]=None, canny: Optional[torch.Tensor]=None) -> torch.Tensor:
        """Forward: inlined DiTwDDTHead.forward + FiLM-advanced per-layer gate injection.

        Inlines parent forward (DDT.py:346-370) to inject mask/canny conditioning via
        per-layer scalar gates across all 30 blocks (28 enc + 2 dec).  Encoder blocks
        use dec_h proj sliced to [:enc_h]; decoder blocks use full dec_h proj.
        Calling super().forward() is not used because parent has no injection hook.

        Args:
            x:     [B, 768, 16, 16]  noisy RAE latent (decoder input)
            t:     [B]               timestep (RAE convention: t=0 clean, t=1 noise)
            y:     [B]               class label
            s:     [B, 768, 16, 16]  optional precomputed encoder features (or None)
            mask:  [B, 256, 256] or [B, 1, 256, 256] binary mask, or None
            canny: [B, 256, 256] or [B, 1, 256, 256] canny edges, or None

        Returns:
            [B, 768, 16, 16] velocity prediction
        """
        enc_h = self._enc_h
        mask_full = mask_enc = canny_full = canny_enc = None
        if mask is not None:
            m = mask if mask.dtype.is_floating_point else mask.float()
            if m.dim() == 3:
                m = m.unsqueeze(1)
            mask_full = self.mask_proj_x(m).flatten(2).transpose(1, 2)
            mask_enc = mask_full[..., :enc_h]
        if canny is not None:
            ce = canny if canny.dtype.is_floating_point else canny.float()
            if ce.dim() == 3:
                ce = ce.unsqueeze(1)
            canny_full = self.canny_proj_x(ce).flatten(2).transpose(1, 2)
            canny_enc = canny_full[..., :enc_h]
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y, self.training)
        c = F.silu(t_emb + y_emb)
        if s is None:
            s_tok = self.s_embedder(x)
            if self.use_pos_embed:
                s_tok = s_tok + self.pos_embed
            for i in range(self.num_encoder_blocks):
                s_tok = self.blocks[i](s_tok, c, feat_rope=self.enc_feat_rope)
                if mask_enc is not None:
                    s_tok = s_tok + self.mask_gates[i] * mask_enc
                if canny_enc is not None:
                    s_tok = s_tok + self.canny_gates[i] * canny_enc
            t_bc = t_emb.unsqueeze(1).expand(-1, s_tok.shape[1], -1)
            s_tok = F.silu(t_bc + s_tok)
        else:
            s_tok = s
        s_proj = self.s_projector(s_tok)
        x_tok = self.x_embedder(x)
        if self.use_pos_embed and self.x_pos_embed is not None:
            x_tok = x_tok + self.x_pos_embed
        for i in range(self.num_encoder_blocks, self.num_blocks):
            x_tok = self.blocks[i](x_tok, s_proj, feat_rope=self.dec_feat_rope)
            if mask_full is not None:
                x_tok = x_tok + self.mask_gates[i] * mask_full
            if canny_full is not None:
                x_tok = x_tok + self.canny_gates[i] * canny_full
        x_tok = self.final_layer(x_tok, s_proj)
        return self.unpatchify(x_tok)

def load_pretrained_rae_backbone(model: RAE_M06, ckpt_path: str, rank: int=0) -> RAE_M06:
    """Load RAE DiTDH-XL pretrain weights; adapter keys stay at zero-init.

    Ckpt key priority: 'ema' -> 'model' -> flat OrderedDict (mirrors
    load_pretrained_for_finetune_rae in train_rae_mask_patch.py:247-288).
    Strips 'module.' / 'net.' DDP prefixes.
    mask_proj_x / canny_proj_x are absent in pretrain ckpt -> zero-init preserved
    via strict=False.

    Args:
        model:     RAE_M06 instance.
        ckpt_path: Path to stage2_model.pt.
        rank:      Logging rank (only rank 0 prints warnings).

    Returns:
        model with backbone weights loaded (in-place), same object returned.
    """
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f'RAE pretrain ckpt not found: {path}')
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
    _force_drop = []
    for k in ('mask_proj_x.weight', 'canny_proj_x.weight'):
        if k in src_sd:
            _force_drop.append(k)
            del src_sd[k]
    if _force_drop and rank == 0:
        print(f'[rae_ckpt] DROP {_force_drop} from ckpt (FiLM-advanced: preserve Kaiming init for proj)')
    result = model.load_state_dict(src_sd, strict=False)
    if rank == 0:
        adapter_pfx = ('mask_proj_x.', 'canny_proj_x.')
        non_adapter = [k for k in result.missing_keys if not any((k.startswith(p) for p in adapter_pfx))]
        if non_adapter:
            print(f'[rae_ckpt] unexpected missing ({len(non_adapter)}): {non_adapter[:6]}')
        print(f"[rae_ckpt] key='{key_used}' missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}")
    return model

def build_rae_stage1(rae_config_yaml: str, device: torch.device) -> nn.Module:
    """Instantiate RAE stage1 (DINOv2 encoder) from OmegaConf yaml.

    Reads stage_1 section from the same yaml used for stage_2. Sets stage1 to
    eval mode and freezes all parameters; used for online encode in the training
    loop.

    Args:
        rae_config_yaml: Path to OmegaConf yaml containing 'stage_1' section.
        device:          CUDA device to move stage1 onto.

    Returns:
        Frozen eval-mode RAE stage1 nn.Module with .encode(x) -> [B,768,16,16].
    """
    cfg_all = OmegaConf.load(rae_config_yaml)
    rae_stage1_cfg = cfg_all.get('stage_1')
    if rae_stage1_cfg is None:
        raise KeyError(f"'stage_1' section not found in {rae_config_yaml}")
    stage1: nn.Module = instantiate_from_config(rae_stage1_cfg).to(device)
    stage1.eval()
    for p in stage1.parameters():
        p.requires_grad_(False)
    return stage1

@torch.no_grad()
def evaluate_deterministic(model: nn.Module, rae_stage1: nn.Module, val_loader: DataLoader, device: torch.device, transport: object, phase: str, condition_mode: str='default', val_seed: int=12345, val_n_batches: int=8) -> torch.Tensor:
    """Deterministic val: MSE-only loss, fixed seed per batch.

    Per batch: torch.manual_seed(val_seed + batch_idx) ensures identical
    noise/timestep sampling across calls (same model state -> same val loss).

    Encodes raw images online with frozen rae_stage1 before computing loss.

    Args:
        model:        The model (may be DDP-wrapped).
        rae_stage1:   Frozen RAE stage1 encoder.
        val_loader:   DataLoader yielding RawImageMaskCannyDataset(is_val=True).
        device:       CUDA device.
        transport:    Transport object with training_losses method.
        phase:        Training phase; controls mask/canny pass-through.
        val_seed:     Base seed; batch i uses val_seed + i.
        val_n_batches: Maximum batches to evaluate.

    Returns:
        Scalar MSE loss tensor on ``device``, NOT yet all-reduced across ranks.
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
        if condition_mode in ('mask_only', 'uncond_only', 'canny_only', 'mask_canny'):
            label = torch.full_like(label, 1000)
        with torch.no_grad():
            image_01 = (image + 1.0) / 2.0
            x = rae_stage1.encode(image_01)
        torch.manual_seed(val_seed + batch_idx)
        torch.cuda.manual_seed_all(val_seed + batch_idx)
        model_kwargs = _build_model_kwargs(label, mask, canny, phase)
        loss_dict = transport.training_losses(model, x, model_kwargs)
        mse = loss_dict['loss'].mean()
        total_mse += mse
        n_seen += 1
    if n_seen == 0:
        return torch.tensor(0.0, device=device)
    return total_mse / n_seen

def _build_model_kwargs(y: torch.Tensor, mask: torch.Tensor, canny: torch.Tensor, phase: str) -> dict:
    """Build model_kwargs dict from batch tensors based on training phase.

    Phase semantics:
      warmup:     mask=None, canny=None (adapter frozen; throughput = upstream)
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

def spawn_fid_eval(ckpt_path: str, experiment_dir: str, train_config: dict) -> None:
    raise NotImplementedError('FID hooks are separate; disable train.fid_every and use the upstream FID tool explicitly.')

def load_weights_with_shape_check(model: nn.Module, checkpoint: dict, rank: int=0) -> nn.Module:
    """Load weights from checkpoint, skipping mismatched shapes.

    Args:
        model:      Target model (RAE_M06 or any nn.Module).
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
            elif rank == 0:
                print(f"Skipping parameter '{name}': checkpoint shape {param.shape}, model shape {model_state_dict[name].shape}")
        elif rank == 0:
            print(f"Parameter '{name}' not found in model, skipping.")
    model.load_state_dict(model_state_dict, strict=False)
    return model

@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float=0.9999) -> None:
    """Step the EMA model towards the current model.

    Args:
        ema_model: EMA copy of the model.
        model:     Live model (may have 'module.' prefix from DDP).
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
    """Train RAE_M06 (raw images + online RAE encode + sharded mask/canny, M06 protocol)."""
    device = accelerator.device
    _use_tf32 = train_config.get('train', {}).get('allow_tf32', True)
    torch.backends.cuda.matmul.allow_tf32 = _use_tf32
    torch.backends.cudnn.allow_tf32 = _use_tf32
    if accelerator.is_main_process:
        print(f'[speedup] allow_tf32={_use_tf32} | gradient_as_bucket_view=True | fused_adamw=True')
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
    _rae_cfg_path = train_config.get('model', {}).get('rae_config_yaml', _RAE_CONFIG_YAML)
    rae_stage1 = build_rae_stage1(_rae_cfg_path, device)
    if accelerator.is_main_process:
        logger.info(f'[M06] RAE stage1 loaded from {_rae_cfg_path} (frozen, eval mode)')
    _rae_model_params = OmegaConf.to_container(OmegaConf.load(_rae_cfg_path).get('stage_2').get('params', {}), resolve=True)
    for _k in ('num_classes', 'class_dropout_prob'):
        if _k in train_config.get('model', {}):
            _rae_model_params[_k] = train_config['model'][_k]
    model = RAE_M06(**_rae_model_params)
    ema = deepcopy(model).to(device)
    _init_best_val_loss = float('inf')
    _reset_best = bool(train_config['train'].get('reset_best_val_loss', False))
    if 'weight_init' in train_config['train']:
        ckpt_path = train_config['train']['weight_init']
        raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        _bvl = raw.get('best_val_loss', None) if isinstance(raw, dict) else None
        if _bvl is not None and isinstance(_bvl, (int, float)) and (_bvl > 0) and (not _reset_best):
            _init_best_val_loss = float(_bvl)
            if accelerator.is_main_process:
                logger.info(f'[chained] inherited best_val_loss = {_init_best_val_loss:.6f} (save threshold)')
        elif _reset_best and accelerator.is_main_process:
            logger.info('[oracle] reset_best_val_loss=true -> using inf save threshold')
        _adapter_keys = ('mask_proj_x.', 'canny_proj_x.')
        _is_m06_ckpt = isinstance(raw, dict) and 'model' in raw and any((k.startswith(pk) for k in raw['model'].keys() for pk in _adapter_keys))
        if _is_m06_ckpt:
            ckpt_data = raw
            ckpt_data['model'] = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in ckpt_data['model'].items()}
            model = load_weights_with_shape_check(model, ckpt_data, rank=rank)
            ema = load_weights_with_shape_check(ema, ckpt_data, rank=rank)
        else:
            model = load_pretrained_rae_backbone(model, ckpt_path, rank=rank)
            ema = load_pretrained_rae_backbone(ema, ckpt_path, rank=rank)
        if accelerator.is_main_process:
            logger.info(f'Loaded weight_init from {ckpt_path} (m06_ckpt={_is_m06_ckpt})')
    requires_grad(ema, False)
    model = model.to(device)
    transport = create_transport(train_config['transport']['path_type'], train_config['transport']['prediction'], train_config['transport'].get('loss_weight', None), train_config['transport'].get('train_eps', None), train_config['transport'].get('sample_eps', None), time_dist_type=train_config['transport'].get('time_dist_type', 'uniform'), time_dist_shift=train_config['transport'].get('time_dist_shift', 1.0))
    if accelerator.is_main_process:
        total_params = sum((p.numel() for p in model.parameters())) / 1000000.0
        logger.info(f'[M06] RAE_M06 Parameters: {total_params:.2f}M')
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
        logger.info(f'[M06] param groups: backbone={len(_backbone_params)} params @ lr={_base_lr}, adapters={len(_adapter_params)} params @ lr={_base_lr * _adapter_mult} (multiplier={_adapter_mult})')
    _fused_adamw = train_config['optimizer'].get('fused_adamw', True)
    opt = torch.optim.AdamW([{'params': _backbone_params, 'lr': _base_lr}, {'params': _adapter_params, 'lr': _base_lr * _adapter_mult}], weight_decay=0, betas=(0.9, train_config['optimizer']['beta2']), fused=_fused_adamw)
    _phase = train_config['data'].get('phase', 'warmup')
    _condition_mode = train_config['train'].get('condition_mode', 'default')
    _condition_phases = {'uncond_only': 'warmup', 'class_only': 'warmup', 'mask_only': 'mask', 'class_mask': 'mask', 'canny_only': 'canny', 'class_canny': 'canny', 'mask_canny': 'canny_mask', 'class_mask_canny': 'canny_mask'}
    if _condition_mode not in ('default', *_condition_phases):
        raise ValueError(f"unsupported train.condition_mode: {_condition_mode!r}; expected one of {sorted(('default', *_condition_phases))}")
    if _condition_mode in _condition_phases and _phase != _condition_phases[_condition_mode]:
        raise ValueError(f'condition_mode={_condition_mode!r} requires data.phase={_condition_phases[_condition_mode]!r}, got {_phase!r}')
    if accelerator.is_main_process:
        logger.info('[conditioning] mode=%s phase=%s', _condition_mode, _phase)
    _rae_cfg_path = train_config.get('model', {}).get('rae_config_yaml', _RAE_CONFIG_YAML)
    _load_mask = _phase in ('mask', 'canny_mask')
    _load_canny = _phase in ('canny', 'canny_mask')
    train_dataset = RawImageMaskCannyDataset(imagenet_root=train_config['data']['data_path'], mask_shard_dir=train_config['data']['mask_shard_dir'], canny_shard_dir=train_config['data']['canny_shard_dir'], paired_whitelist_json=train_config['data']['paired_train_whitelist'], image_size=256, is_val=False, load_mask=_load_mask, load_canny=_load_canny)
    batch_size_per_gpu = int(np.round(train_config['train']['global_batch_size'] / accelerator.num_processes / accelerator.gradient_accumulation_steps))
    global_batch_size = batch_size_per_gpu * accelerator.num_processes * accelerator.gradient_accumulation_steps
    loader = DataLoader(train_dataset, batch_size=batch_size_per_gpu, shuffle=True, num_workers=train_config['data']['num_workers'], pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=4)
    if accelerator.is_main_process:
        logger.info(f"[M06] RawImageMaskCannyDataset train N={len(train_dataset):,} path={train_config['data']['data_path']}")
        logger.info(f'Batch size {batch_size_per_gpu} per gpu, global batch size {global_batch_size}')
    val_loader = None
    val_n_batches = 0
    _paired_val_wl = train_config['data'].get('paired_val_whitelist', None)
    if _paired_val_wl is not None:
        val_dataset = RawImageMaskCannyDataset(imagenet_root=train_config['data']['data_path'], mask_shard_dir=train_config['data']['mask_shard_dir'], canny_shard_dir=train_config['data']['canny_shard_dir'], paired_whitelist_json=_paired_val_wl, image_size=256, is_val=True, load_mask=_load_mask, load_canny=_load_canny)
        val_n_batches = train_config['train'].get('val_n_batches', 8)
        val_loader = DataLoader(val_dataset, batch_size=batch_size_per_gpu, shuffle=False, num_workers=2, pin_memory=True, drop_last=False, persistent_workers=False)
        if accelerator.is_main_process:
            _val_seed = train_config['train'].get('val_seed', 12345)
            logger.info(f'[val] RawImageMaskCannyDataset val N={len(val_dataset):,} | n_batches={val_n_batches} | seed={_val_seed}')
    update_ema(ema, model, decay=0)
    model.train()
    ema.eval()
    train_config['train']['resume'] = train_config['train'].get('resume', False)
    _resume_opt_state = None
    _resume_epoch = 0
    _resume_best_val_loss = None
    if train_config['train']['resume']:
        explicit_ckpt = train_config['train'].get('resume_checkpoint')
        ckpt_files = glob(f'{checkpoint_dir}/[0-9]*.pt')
        if explicit_ckpt:
            ckpt_files = [explicit_ckpt]
        if ckpt_files:
            ckpt_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]) if os.path.basename(x).split('.')[0].isdigit() else -1)
            latest_ckpt = ckpt_files[-1]
            ckpt = torch.load(latest_ckpt, map_location=lambda storage, loc: storage)
            model.load_state_dict(ckpt['model'])
            ema.load_state_dict(ckpt['ema'])
            if 'train_steps' in ckpt:
                train_steps = int(ckpt['train_steps'])
            else:
                train_steps = int(latest_ckpt.split('/')[-1].split('.')[0])
            _resume_opt_state = ckpt.get('opt')
            _resume_epoch = int(ckpt.get('epoch', train_config['train'].get('resume_epoch', 0)))
            _resume_best_val_loss = ckpt.get('best_val_loss')
            if accelerator.is_main_process:
                logger.info('Resuming from checkpoint: %s | step=%d epoch=%d opt=%s best=%s', latest_ckpt, train_steps, _resume_epoch, 'restored' if _resume_opt_state is not None else 'missing', _resume_best_val_loss)
        elif accelerator.is_main_process:
            logger.info('No checkpoint found. Starting from scratch.')

    def _freeze_canny_adapters() -> None:
        """Freeze canny_proj_x and canny_gates parameters."""
        model.canny_proj_x.weight.requires_grad_(False)
        model.canny_gates.requires_grad_(False)

    def _freeze_mask_adapters() -> None:
        """Freeze mask_proj_x and mask_gates parameters."""
        model.mask_proj_x.weight.requires_grad_(False)
        model.mask_gates.requires_grad_(False)

    def _log_trainable_advanced() -> None:
        """Print trainable=True/False for each adapter param (audit)."""
        if accelerator.is_main_process:
            for n, p in model.named_parameters():
                if any((ak in n for ak in ('mask_proj_x', 'canny_proj_x', 'mask_gates', 'canny_gates'))):
                    logger.info(f'[M06][freeze-audit] {n}: trainable={p.requires_grad}')
    if _phase == 'mask':
        _freeze_canny_adapters()
        if accelerator.is_main_process:
            logger.info('[M06] phase=mask -> mask_proj_x + mask_gates trainable, canny_proj_x + canny_gates frozen')
        _log_trainable_advanced()
    elif _phase == 'canny':
        _freeze_mask_adapters()
        if accelerator.is_main_process:
            logger.info('[M06] phase=canny -> canny_proj_x + canny_gates trainable, mask_proj_x + mask_gates frozen')
        _log_trainable_advanced()
    elif _phase == 'canny_mask':
        if accelerator.is_main_process:
            logger.info('[M06] phase=canny_mask -> mask_proj_x + mask_gates + canny_proj_x + canny_gates trainable')
        _log_trainable_advanced()
    else:
        _freeze_mask_adapters()
        _freeze_canny_adapters()
        if accelerator.is_main_process:
            logger.info(f'[M06] phase={_phase} -> all adapters (proj + gates) frozen (no conditioning input; avoids find_unused_parameters crash)')
        _log_trainable_advanced()
    model, opt, loader = accelerator.prepare(model, opt, loader)
    if _resume_opt_state is not None:
        opt.load_state_dict(_resume_opt_state)
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)
    if not train_config['train']['resume']:
        train_steps = 0
    log_steps = 0
    running_loss = 0.0
    start_time = time()
    epoch = _resume_epoch
    best_val_loss = float(_resume_best_val_loss) if _resume_best_val_loss is not None else _init_best_val_loss
    no_improve_epochs = 0
    val_seed = train_config['train'].get('val_seed', 12345)
    early_stop_patience = train_config['train'].get('early_stop_patience', 5)
    early_stop_min_epochs = train_config['train'].get('early_stop_min_epochs', 10)
    val_every_n_epochs = train_config['train'].get('val_every_n_epochs', 1)
    val_loss_log_path = os.path.join(experiment_dir, 'val_loss_log.jsonl') if accelerator.is_main_process else None
    if train_config['train'].get('val_only_first', False):
        if val_loader is None:
            raise RuntimeError('val_only_first=True but val_loader is None')
        results = {}
        for probe_phase in ('warmup', _phase):
            val_mse = evaluate_deterministic(model, rae_stage1, val_loader, device, transport, phase=probe_phase, condition_mode=_condition_mode, val_seed=val_seed, val_n_batches=val_n_batches)
            accelerator.wait_for_everyone()
            results[probe_phase] = accelerator.gather(val_mse.detach().unsqueeze(0)).mean().item()
        if accelerator.is_main_process:
            logger.info(f"[init_check] val_only_first=True | phase=warmup val_mse={results['warmup']:.6f} | phase={_phase} val_mse={results[_phase]:.6f} | inherited_best={_init_best_val_loss:.6f} | delta(phase-warmup)={results[_phase] - results['warmup']:+.6f} | gates_max_abs={accelerator.unwrap_model(model).mask_gates.abs().max().item():.6e} | mask_proj_x_max_abs={accelerator.unwrap_model(model).mask_proj_x.weight.abs().max().item():.6e}")
        return accelerator
    if accelerator.is_main_process:
        logger.info(f"Training for up to {train_config['train']['max_steps']} steps...")
    _early_stop = False
    while True:
        for image, mask, canny, label in loader:
            with accelerator.accumulate(model):
                image = image.to(device)
                mask = mask.to(device)
                canny = canny.to(device)
                label = label.to(device)
                if _condition_mode in ('mask_only', 'uncond_only', 'canny_only', 'mask_canny'):
                    label = torch.full_like(label, 1000)
                with torch.no_grad():
                    image_01 = (image + 1.0) / 2.0
                    x = rae_stage1.encode(image_01)
                model_kwargs = _build_model_kwargs(label, mask, canny, _phase)
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
            if train_steps >= train_config['train']['max_steps']:
                break
        epoch += 1
        checkpoint_every_n_epochs = int(train_config['train'].get('checkpoint_every_n_epochs', 0))
        if checkpoint_every_n_epochs > 0 and epoch % checkpoint_every_n_epochs == 0 and accelerator.is_main_process:
            epoch_path = os.path.join(checkpoint_dir, f'epoch_{epoch:04d}.pt')
            unwrapped = accelerator.unwrap_model(model)
            torch.save({'model': unwrapped.state_dict(), 'opt': opt.state_dict(), 'ema': ema.state_dict(), 'train_steps': train_steps, 'best_val_loss': best_val_loss, 'epoch': epoch}, epoch_path)
            logger.info('[checkpoint] fixed-epoch snapshot -> %s', epoch_path)
        if val_loader is not None and epoch % val_every_n_epochs == 0:
            val_mse = evaluate_deterministic(model, rae_stage1, val_loader, device, transport, phase=_phase, condition_mode=_condition_mode, val_seed=val_seed, val_n_batches=val_n_batches)
            accelerator.wait_for_everyone()
            val_mse_all = accelerator.gather(val_mse.detach().unsqueeze(0)).mean().item()
            if not (val_mse_all == val_mse_all and val_mse_all != float('inf') and (val_mse_all != -float('inf'))):
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
        torch.save({'model': unwrapped.state_dict(), 'opt': opt.state_dict(), 'ema': ema.state_dict(), 'train_steps': train_steps, 'best_val_loss': best_val_loss, 'epoch': epoch}, last_path)
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
    parser.add_argument('--config', type=str, default='configs/debug_rae_m06.yaml', help='Path to YAML training config.')
    args = parser.parse_args()
    train_config = load_config(args.config)
    train_config['_config_path'] = args.config
    _grad_accum = int(train_config.get('train', {}).get('gradient_accumulation_steps', 1))
    _ddp_bucket_cap_mb = train_config.get('train', {}).get('ddp_bucket_cap_mb', 256)
    from accelerate import DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(bucket_cap_mb=_ddp_bucket_cap_mb, find_unused_parameters=False, gradient_as_bucket_view=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], gradient_accumulation_steps=_grad_accum)
    do_train(train_config, accelerator)
