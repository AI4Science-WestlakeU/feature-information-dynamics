"""Native JiT conditional trainer.

The loss, time sampling, optimizer, EMA, phase freezing, validation and resume
logic are retained from did_q01's executed FiLM-advanced trainer. The conditional
model is imported from the public backend, which corrects spatial-token alignment.
Set resources.jit_source in the training YAML to the supplied JiT source checkout.
"""
from __future__ import annotations

from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cuda
import torch.backends.cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import sys
import math
import yaml
import json
import numpy as np
import logging
import os
import argparse
from datetime import datetime
from time import time
from glob import glob
from copy import deepcopy
from collections import OrderedDict

from accelerate import Accelerator
from feature_information_dynamics.data.raw_image_mask_canny_dataset import RawImageMaskCannyDataset

NULL_CLASS: int = 1000   # ImageNet null class for CFG (JiT uses num_classes=1000)

# Phase choices (shared with all M06 scripts)
PHASE_CHOICES: tuple[str, ...] = ("warmup", "mask", "canny", "canny_mask")


# ---------------------------------------------------------------------------
# JiT_M06_FiLM: FiLM-advanced subclass wrapping JiT-L/16
# ---------------------------------------------------------------------------

def load_pretrained_jit_backbone_film(
    model: JiT_M06_FiLM,
    ckpt_path: str,
    rank: int = 0,
) -> JiT_M06_FiLM:
    """Load JiT paper ckpt into model.backbone; FiLM adapter keys stay at Kaiming init.

    Handles JiT paper flat state_dict or {'model_ema2'/'model_ema1'/'model': {...}}.
    Strips 'module.' and 'net.' DDP prefixes.  Force-drops mask_proj_x / canny_proj_x /
    *_gates from ckpt so FiLM-advanced Kaiming+zero-gates init is preserved.

    Args:
        model:     JiT_M06_FiLM instance.
        ckpt_path: Path to checkpoint-last.pth.
        rank:      Logging rank (only rank 0 prints).

    Returns:
        model with backbone weights loaded (in-place).
    """
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"Pretrain ckpt not found: {path}")
    raw = torch.load(str(path), map_location="cpu", weights_only=False)

    src_sd, key_used = None, None
    for cand in ("model_ema2", "model_ema1", "model"):
        if isinstance(raw, dict) and cand in raw:
            src_sd, key_used = raw[cand], cand
            break
    if src_sd is None:
        src_sd, key_used = raw, "<flat>"

    src_sd = {
        k.removeprefix("module.").removeprefix("net."): v
        for k, v in src_sd.items()
        if isinstance(v, torch.Tensor)
    }

    # FiLM-advanced: ALWAYS drop adapter keys to preserve Kaiming+zero-gates init.
    # loading ckpt gates=0+proj=0 dead-locks gradient flow; Kaiming+gates=0 does not.
    _adapter_keys = (
        "mask_proj_x.weight",
        "canny_proj_x.weight",
        "mask_gates",
        "canny_gates",
    )
    _dropped = [k for k in _adapter_keys if k in src_sd]
    for k in _dropped:
        del src_sd[k]
    if _dropped and rank == 0:
        print(f"[jit_film_ckpt] DROP {_dropped} (FiLM-advanced: preserve Kaiming+zero-gates init)")

    result = model.backbone.load_state_dict(src_sd, strict=False)
    if rank == 0:
        _adapter_pfx = ("mask_proj_x.", "canny_proj_x.", "mask_gates", "canny_gates")
        non_adapter = [
            k for k in result.missing_keys
            if not any(k.startswith(p) for p in _adapter_pfx)
        ]
        if non_adapter:
            print(f"[jit_film_ckpt] unexpected missing ({len(non_adapter)}): {non_adapter[:6]}")
        print(
            f"[jit_film_ckpt] key='{key_used}' "
            f"missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}"
        )
    return model


def load_weights_with_shape_check(
    model: nn.Module, checkpoint: dict, rank: int = 0
) -> nn.Module:
    """Load weights from checkpoint, skipping mismatched shapes.

    Args:
        model:      Target model (JiT_M06_FiLM or any nn.Module).
        checkpoint: Dict with 'model' key containing state_dict.
        rank:       Logging rank.

    Returns:
        model with weights loaded in-place.
    """
    model_state_dict = model.state_dict()
    for name, param in checkpoint["model"].items():
        if name in model_state_dict:
            if param.shape == model_state_dict[name].shape:
                model_state_dict[name].copy_(param)
            else:
                if rank == 0:
                    print(
                        f"Skipping '{name}': ckpt {param.shape} vs model "
                        f"{model_state_dict[name].shape}"
                    )
        else:
            if rank == 0:
                print(f"Parameter '{name}' not found in model, skipping.")
    model.load_state_dict(model_state_dict, strict=False)
    return model


# ---------------------------------------------------------------------------
# Pixel-specific velocity loss + evaluation (unchanged logic from base)
# ---------------------------------------------------------------------------

def _sample_t_lognormal(
    batch_size: int,
    device: torch.device,
    p_mean: float = -0.8,
    p_std: float = 0.8,
) -> torch.Tensor:
    """Sample [B] timesteps from lognormal: t = sigmoid(N(p_mean, p_std)).

    t=1 corresponds to clean data; t=0 to pure noise (project convention).

    Args:
        batch_size: Number of samples.
        device:     Target device.
        p_mean:     Lognormal location parameter (default -0.8).
        p_std:      Lognormal scale parameter (default 0.8).

    Returns:
        [B] tensor of timesteps in (0, 1).
    """
    z = torch.randn(batch_size, device=device) * p_std + p_mean
    return torch.sigmoid(z)  # [B] in (0, 1)


def _sample_t_full_snr_mixture(
    batch_size: int,
    device: torch.device,
    p_mean: float,
    p_std: float,
    sampler_config: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed-quota official + stratified log-SNR sampling.

    Returns timesteps and a boolean mask identifying official samples. The
    reliable and high-SNR quotas are uniform in log10(gamma), hence uniform in
    logit(t), and are shuffled so quota type is not tied to sample order.
    """
    f_official = float(sampler_config.get("original_fraction", 0.5))
    f_reliable = float(sampler_config.get("reliable_fraction", 0.4))
    f_high = float(sampler_config.get("high_fraction", 0.1))
    if min(f_official, f_reliable, f_high) < 0:
        raise ValueError("full-SNR mixture fractions must be non-negative")
    if abs(f_official + f_reliable + f_high - 1.0) > 1e-8:
        raise ValueError("full-SNR mixture fractions must sum to one")

    n_official = int(round(batch_size * f_official))
    n_reliable = int(round(batch_size * f_reliable))
    n_high = batch_size - n_official - n_reliable
    if n_high < 0:
        raise ValueError("rounded full-SNR quotas exceed batch size")

    z_parts = []
    official_parts = []
    if n_official:
        z_parts.append(torch.randn(n_official, device=device) * p_std + p_mean)
        official_parts.append(torch.ones(n_official, dtype=torch.bool, device=device))

    def stratified_log_gamma(n: int, lo: float, hi: float, bins: int) -> torch.Tensor:
        if bins <= 0:
            raise ValueError("log-SNR stratum count must be positive")
        edges = torch.linspace(lo, hi, bins + 1, device=device)
        base, remainder = divmod(n, bins)
        draws = []
        for idx in range(bins):
            count = base + int(idx < remainder)
            if count:
                draws.append(
                    torch.rand(count, device=device) * (edges[idx + 1] - edges[idx])
                    + edges[idx]
                )
        lg = torch.cat(draws) if draws else torch.empty(0, device=device)
        return lg * (math.log(10.0) / 2.0)

    if n_reliable:
        z_parts.append(stratified_log_gamma(
            n_reliable,
            float(sampler_config.get("reliable_log10_gamma_min", -8.0)),
            float(sampler_config.get("reliable_log10_gamma_max", 0.75)),
            int(sampler_config.get("reliable_bins", 8)),
        ))
        official_parts.append(torch.zeros(n_reliable, dtype=torch.bool, device=device))
    if n_high:
        z_parts.append(stratified_log_gamma(
            n_high,
            float(sampler_config.get("high_log10_gamma_min", 0.75)),
            float(sampler_config.get("high_log10_gamma_max", 4.0)),
            int(sampler_config.get("high_bins", 3)),
        ))
        official_parts.append(torch.zeros(n_high, dtype=torch.bool, device=device))

    z = torch.cat(z_parts)
    is_official = torch.cat(official_parts)
    order = torch.randperm(batch_size, device=device)
    return torch.sigmoid(z[order]), is_official[order]


def _sample_t_fixed(
    batch_size: int,
    device: torch.device,
    sampler_config: dict,
) -> torch.Tensor:
    """Return a batch at one exact t for single-SNR MMSE training."""
    fixed_t = float(sampler_config.get("fixed_t", float("nan")))
    if not math.isfinite(fixed_t) or not 0.0 < fixed_t < 1.0:
        raise ValueError("fixed_t sampler requires finite 0 < fixed_t < 1")
    return torch.full((batch_size,), fixed_t, device=device)


def _fixed_quota_drop_mask(
    batch_size: int,
    fraction: float,
    device: torch.device,
) -> torch.Tensor:
    """Shuffle an exact per-rank class-drop quota for a training batch."""
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("class_drop_fraction must be finite and in [0, 1]")
    n_drop = int(round(batch_size * fraction))
    mask = torch.cat(
        (
            torch.ones(n_drop, dtype=torch.bool, device=device),
            torch.zeros(batch_size - n_drop, dtype=torch.bool, device=device),
        )
    )
    return mask[torch.randperm(batch_size, device=device)]


def _compute_velocity_loss(
    model: nn.Module,
    x_clean: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    p_mean: float,
    p_std: float,
    t_eps: float,
    mask: Optional[torch.Tensor] = None,
    canny: Optional[torch.Tensor] = None,
    mask_drop: Optional[torch.Tensor] = None,
    canny_drop: Optional[torch.Tensor] = None,
    sampler_config: Optional[dict] = None,
) -> torch.Tensor:
    """Compute flow-matching MSE velocity loss for pixel JiT FiLM-advanced.

    DEVIATION: JiT.forward returns x_pred not v_pred -> velocity computed manually.
    v = (x_clean - x_noisy) / max(1 - t, t_eps). Matches the implemented
    denominator floor below and the corrected Pixel MMSE evaluator.

    Args:
        model:    JiT_M06_FiLM (may be DDP-wrapped).
        x_clean:  [B, 3, H, W] clean images in [-1, 1].
        y:        [B] class labels.
        device:   Target device.
        p_mean:   t-sampling mean.
        p_std:    t-sampling std.
        t_eps:    Floor for (1 - t) denominator.
        mask:     [B, H, W] binary mask or None.
        canny:    [B, H, W] binary canny edges or None.

    Returns:
        Scalar MSE loss.
    """
    B = x_clean.shape[0]
    sampler_mode = sampler_config.get("mode") if sampler_config else None
    use_mixture = sampler_mode == "mixed_full_snr"
    use_fixed = sampler_mode == "fixed_t"
    if use_mixture:
        t, is_official = _sample_t_full_snr_mixture(
            B, device, p_mean, p_std, sampler_config
        )
    elif use_fixed:
        t = _sample_t_fixed(B, device, sampler_config)
        is_official = torch.zeros(B, dtype=torch.bool, device=device)
    else:
        t = _sample_t_lognormal(B, device, p_mean, p_std)
        is_official = torch.ones(B, dtype=torch.bool, device=device)
    noise = torch.randn_like(x_clean)                    # [B, 3, H, W]
    t4 = t.view(B, 1, 1, 1)                              # [B, 1, 1, 1]
    x_noisy = t4 * x_clean + (1.0 - t4) * noise         # [B, 3, H, W]
    v_target = (x_clean - x_noisy) / (1.0 - t4).clamp_min(t_eps)   # [B, 3, H, W]

    x_pred = model(x_noisy, t, y, mask=mask, canny=canny,
                   mask_drop=mask_drop, canny_drop=canny_drop)      # [B, 3, H, W]
    v_pred = (x_pred - x_noisy) / (1.0 - t4).clamp_min(t_eps)       # [B, 3, H, W]

    velocity_per_sample = ((v_pred.float() - v_target.float()) ** 2).mean(dim=(1, 2, 3))
    if not use_mixture and not use_fixed:
        return velocity_per_sample.mean()
    # Official draws retain the paper velocity objective. Stratified draws use
    # direct x1 reconstruction MSE, which aligns with the evaluated MMSE and
    # avoids reintroducing a (1-t)^-2 high-SNR bias.
    x1_per_sample = ((x_pred.float() - x_clean.float()) ** 2).mean(dim=(1, 2, 3))
    per_sample = torch.where(is_official, velocity_per_sample, x1_per_sample)
    return per_sample.mean()


@torch.no_grad()
def evaluate_deterministic_pixel(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    p_mean: float,
    p_std: float,
    t_eps: float,
    phase: str = "warmup",
    condition_mode: str = "default",
    val_seed: int = 12345,
    val_n_batches: int = 8,
    sampler_config: Optional[dict] = None,
) -> torch.Tensor:
    """Deterministic pixel-space val: MSE velocity loss, fixed seed per batch.

    Batch shape from RawImageMaskCannyDataset: (image, mask, canny, label) 4-tuple.

    Args:
        model:         JiT_M06_FiLM (may be DDP-wrapped).
        val_loader:    DataLoader from RawImageMaskCannyDataset(is_val=True).
        device:        CUDA device.
        p_mean:        t lognormal mean.
        p_std:         t lognormal std.
        t_eps:         Floor for velocity denominator.
        phase:         Training phase controlling which signals to pass.
        val_seed:      Base seed; batch i uses val_seed + i.
        val_n_batches: Max batches to evaluate.

    Returns:
        Scalar MSE loss tensor on device, NOT yet all-reduced.
    """
    model.eval()
    total_mse = torch.tensor(0.0, device=device)
    n_seen = 0
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= val_n_batches:
            break
        image, mask_2d, canny_2d, label = batch
        image = image.to(device)                         # [B, 3, 256, 256]
        mask_2d = mask_2d.to(device, non_blocking=True)  # [B, 256, 256]
        canny_2d = canny_2d.to(device, non_blocking=True)  # [B, 256, 256]
        label = label.to(device)                          # [B]
        if condition_mode in ("uncond", "mask_only", "canny_only", "mask_canny"):
            # JiT's LabelEmbedder does not itself implement CFG dropout.
            # A mask-only model must therefore receive the null class explicitly.
            label = torch.full_like(label, NULL_CLASS)

        mask_for_model = mask_2d if phase in ("mask", "canny_mask") else None
        canny_for_model = canny_2d if phase in ("canny", "canny_mask") else None

        torch.manual_seed(val_seed + batch_idx)
        torch.cuda.manual_seed_all(val_seed + batch_idx)

        mse = _compute_velocity_loss(
            model, image, label, device, p_mean, p_std, t_eps,
            mask=mask_for_model,
            canny=canny_for_model,
            sampler_config=sampler_config,
        )
        total_mse += mse.detach()
        n_seen += 1

    if n_seen == 0:
        return torch.tensor(0.0, device=device)
    return total_mse / n_seen


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = 0.9999) -> None:
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
        name = name.replace("module.", "").removeprefix("_orig_mod.")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model: nn.Module, flag: bool = True) -> None:
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
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


def create_logger(logging_dir: str) -> logging.Logger:
    """Create logger writing to file and stdout.

    Args:
        logging_dir: Directory for log.txt.

    Returns:
        Configured logger.
    """
    # ``accelerate launch --num_processes=1`` does not initialize a torch
    # distributed process group.  Keep this helper usable in both the
    # single-process specialization runs and the original DDP runs.
    is_main_process = not dist.is_initialized() or dist.get_rank() == 0
    if is_main_process:
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


# ---------------------------------------------------------------------------
# do_train — main training function (FiLM-advanced variant)
# ---------------------------------------------------------------------------

def do_train(train_config: dict, accelerator: Accelerator) -> Accelerator:
    """Train JiT_M06_FiLM (JiT-L/16 + FiLM-advanced per-layer gates, M06 protocol).

    Args:
        train_config: Parsed YAML config dict.
        accelerator:  Hugging Face Accelerate context.

    Returns:
        accelerator (for downstream use).
    """
    # The caller configures the explicit JiT source before loading the backend.
    from feature_information_dynamics.pixel.model import JiT_M06_FiLM

    device = accelerator.device

    experiment_dir = (
        f"{train_config['train']['output_dir']}/{train_config['train']['exp_name']}"
    )
    checkpoint_dir = f"{experiment_dir}/checkpoints"

    if accelerator.is_main_process:
        os.makedirs(train_config["train"]["output_dir"], exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory: {experiment_dir}")
        tb_dir = f"tensorboard_logs/{train_config['train']['exp_name']}"
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
        writer.add_text("training configs", json.dumps(train_config, indent=4), 0)

    rank = accelerator.local_process_index

    assert train_config["data"].get("image_size", 256) == 256, (
        "Pixel JiT FiLM-advanced fork only supports image_size=256"
    )

    # ── Build JiT_M06_FiLM model ──────────────────────────────────────────────
    _global_seed = int(train_config["train"].get("global_seed", 0))
    torch.manual_seed(_global_seed)
    torch.cuda.manual_seed_all(_global_seed)
    np.random.seed(_global_seed)
    model = JiT_M06_FiLM(
        input_size=256,
        num_classes=train_config["data"]["num_classes"],
        use_checkpoint=train_config["model"].get("use_checkpoint", False),
        class_dropout_prob=train_config["model"].get("class_dropout_prob", 0.1),
        model_name=train_config["model"].get("name", "JiT-L/16"),
    )

    ema = deepcopy(model).to(device)

    # ── Weight init dispatch (paper ckpt vs M06 chained ckpt) ────────────────
    _init_best_val_loss = float("inf")
    _preserve_loaded_ema = False
    # Oracle / per-γ delta-t protocol: explicit opt-out from chained inheritance
    _reset_best = bool(train_config["train"].get("reset_best_val_loss", False))
    if "weight_init" in train_config["train"]:
        ckpt_path = train_config["train"]["weight_init"]
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Chained protocol: inherit best_val_loss as save threshold (UNLESS reset)
        _bvl = raw.get("best_val_loss", None) if isinstance(raw, dict) else None
        if _reset_best:
            if accelerator.is_main_process:
                logger.info(
                    f"[oracle] reset_best_val_loss=true → ignoring inherited "
                    f"best_val_loss={_bvl} ; using inf threshold"
                )
        elif _bvl is not None and isinstance(_bvl, (int, float)) and _bvl > 0:
            _init_best_val_loss = float(_bvl)
            if accelerator.is_main_process:
                logger.info(
                    f"[chained] inherited best_val_loss = {_init_best_val_loss:.6f} (save threshold)"
                )
        # Detect M06 FiLM ckpt: 'model' key with 'backbone.' + FiLM adapter prefixes
        _adapter_keys_check = (
            "mask_proj_x.", "canny_proj_x.", "mask_gates", "canny_gates"
        )
        _is_m06_ckpt = (
            isinstance(raw, dict)
            and "model" in raw
            and any(k.startswith("backbone.") for k in raw["model"].keys())
        )
        if _is_m06_ckpt:
            _init_model_from_ema = bool(
                train_config["train"].get("init_model_from_ema", False)
            )
            _model_state_key = (
                "ema"
                if _init_model_from_ema
                and isinstance(raw.get("ema"), dict)
                else "model"
            )
            ckpt_data = {
                "model": {
                    k.replace("module.", "").removeprefix("_orig_mod."): v
                    for k, v in raw[_model_state_key].items()
                }
            }
            model = load_weights_with_shape_check(model, ckpt_data, rank=rank)
            _ema_state = raw.get("ema", raw[_model_state_key])
            ema_ckpt_data = {
                "model": {
                    k.replace("module.", "").removeprefix("_orig_mod."): v
                    for k, v in _ema_state.items()
                }
            }
            ema = load_weights_with_shape_check(ema, ema_ckpt_data, rank=rank)
            _preserve_loaded_ema = True
            if accelerator.is_main_process:
                logger.info(
                    "[chained] initialized model from checkpoint[%s] and "
                    "preserved checkpoint EMA",
                    _model_state_key,
                )
        else:
            # Paper JiT pretrain ckpt -> load backbone; adapter keys stay at Kaiming+zero-gates
            model = load_pretrained_jit_backbone_film(model, ckpt_path, rank=rank)
            ema = load_pretrained_jit_backbone_film(ema, ckpt_path, rank=rank)
        if accelerator.is_main_process:
            logger.info(f"Loaded weight_init from {ckpt_path} (m06_ckpt={_is_m06_ckpt})")

    requires_grad(ema, False)
    model = model.to(device)

    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        adapter_params = (
            sum(p.numel() for p in model.mask_proj_x.parameters())
            + sum(p.numel() for p in model.canny_proj_x.parameters())
            + model.mask_gates.numel()
            + model.canny_gates.numel()
        )
        logger.info(
            f"[M06-FiLM] {model.model_name} Parameters: {total_params:.2f}M "
            f"(adapter: {adapter_params / 1e6:.4f}M | "
            f"num_gates={len(model.mask_gates)} per modality)"
        )
        logger.info(
            f"Optimizer: AdamW, lr={train_config['optimizer']['lr']}, "
            f"beta2={train_config['optimizer']['beta2']}"
        )

    # M06: adapter param group with optional lr multiplier (FiLM-advanced: gates included)
    _base_lr = train_config["optimizer"]["lr"]
    _adapter_mult = float(train_config["optimizer"].get("adapter_lr_multiplier", 1.0))
    _adapter_param_names = (
        "mask_proj_x", "canny_proj_x", "mask_gates", "canny_gates",
        "mask_null_embed", "canny_null_embed",
    )
    _adapter_params, _backbone_params = [], []
    for n, p in model.named_parameters():
        if any(ak in n for ak in _adapter_param_names):
            _adapter_params.append(p)
        else:
            _backbone_params.append(p)
    if accelerator.is_main_process:
        logger.info(
            f"[M06-FiLM] param groups: backbone={len(_backbone_params)} @ lr={_base_lr}, "
            f"adapters={len(_adapter_params)} @ lr={_base_lr * _adapter_mult} "
            f"(multiplier={_adapter_mult})"
        )

    opt = torch.optim.AdamW(
        [
            {"params": _backbone_params, "lr": _base_lr},
            {"params": _adapter_params, "lr": _base_lr * _adapter_mult},
        ],
        weight_decay=0,
        betas=(0.9, train_config["optimizer"]["beta2"]),
    )

    # ── Dataset construction (raw image + sharded mask/canny) ─────────────────
    _data_phase = train_config["data"].get("phase", "warmup")
    _condition_mode = train_config["train"].get("condition_mode", "default")
    _class_drop_fraction = float(
        train_config["train"].get("class_drop_fraction", 0.0)
    )
    _condition_phases = {
        "uncond": "warmup",
        "class_only": "warmup",
        "mask_only": "mask",
        "mask_class": "mask",
        "canny_only": "canny",
        "class_canny": "canny",
        "mask_canny": "canny_mask",
        "mask_class_canny": "canny_mask",
    }
    if _condition_mode not in ("default", *_condition_phases):
        raise ValueError(
            "train.condition_mode must be one of "
            "{'default','uncond','class_only','mask_only','mask_class',"
            "'canny_only','class_canny','mask_canny','mask_class_canny'}, "
            f"got {_condition_mode!r}"
        )
    if not math.isfinite(_class_drop_fraction) or not 0.0 <= _class_drop_fraction <= 1.0:
        raise ValueError("train.class_drop_fraction must be finite and in [0, 1]")
    if _class_drop_fraction > 0 and _condition_mode in (
        "uncond", "mask_only", "canny_only", "mask_canny"
    ):
        raise ValueError(
            "train.class_drop_fraction is invalid when condition_mode already forces NULL labels"
        )
    if (
        _condition_mode in _condition_phases
        and _data_phase != _condition_phases[_condition_mode]
    ):
        raise ValueError(
            f"condition_mode={_condition_mode!r} requires "
            f"data.phase={_condition_phases[_condition_mode]!r}, "
            f"got {_data_phase!r}"
        )
    if accelerator.is_main_process:
        logger.info("[conditioning] mode=%s phase=%s", _condition_mode, _data_phase)
        logger.info(
            "[conditioning] explicit fixed-quota class_drop_fraction=%.6f",
            _class_drop_fraction,
        )

    dataset = RawImageMaskCannyDataset(
        imagenet_root=train_config["data"]["data_path"],
        mask_shard_dir=train_config["data"]["mask_shard_dir"],
        canny_shard_dir=train_config["data"]["canny_shard_dir"],
        paired_whitelist_json=train_config["data"]["paired_train_whitelist"],
        image_size=train_config["data"].get("image_size", 256),
        is_val=False,
    )

    batch_size_per_gpu = int(
        np.round(train_config["train"]["global_batch_size"] / accelerator.num_processes)
    )
    global_batch_size = batch_size_per_gpu * accelerator.num_processes
    loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        shuffle=True,
        num_workers=train_config["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset N={len(dataset):,} from {train_config['data']['data_path']}")
        logger.info(f"Batch per GPU={batch_size_per_gpu}, global={global_batch_size}")

    # ── Val loader construction ───────────────────────────────────────────────
    val_loader = None
    val_n_batches = 0
    _paired_val_wl = train_config["data"].get("paired_val_whitelist", None)
    if _paired_val_wl is not None:
        val_dataset = RawImageMaskCannyDataset(
            imagenet_root=train_config["data"]["data_path"],
            mask_shard_dir=train_config["data"]["mask_shard_dir"],
            canny_shard_dir=train_config["data"]["canny_shard_dir"],
            paired_whitelist_json=_paired_val_wl,
            image_size=train_config["data"].get("image_size", 256),
            is_val=True,
        )
        val_n_batches = train_config["train"].get("val_n_batches", 8)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size_per_gpu,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
            persistent_workers=False,
        )
        if accelerator.is_main_process:
            _val_seed_log = train_config["train"].get("val_seed", 12345)
            logger.info(
                f"[val] RawImageMaskCannyDataset N={len(val_dataset):,} | "
                f"n_batches={val_n_batches} | seed={_val_seed_log}"
            )

    # ── Prepare for training ──────────────────────────────────────────────────
    if not _preserve_loaded_ema:
        update_ema(ema, model, decay=0)
    model.train()
    ema.eval()

    # ── Phase-aware adapter freeze (FiLM-advanced: includes gates) ────────────
    def _freeze_mask_adapters() -> None:
        """Freeze mask parameters, including its unused-null embedding."""
        model.mask_proj_x.weight.requires_grad_(False)
        model.mask_gates.requires_grad_(False)
        model.mask_null_embed.requires_grad_(False)

    def _freeze_canny_adapters() -> None:
        """Freeze canny parameters, including its unused-null embedding."""
        model.canny_proj_x.weight.requires_grad_(False)
        model.canny_gates.requires_grad_(False)
        model.canny_null_embed.requires_grad_(False)

    def _log_trainable() -> None:
        """Print trainable=True/False for each adapter param (freeze audit)."""
        if accelerator.is_main_process:
            for n, p in model.named_parameters():
                if any(ak in n for ak in _adapter_param_names):
                    logger.info(f"[M06-FiLM][freeze-audit] {n}: trainable={p.requires_grad}")

    if _data_phase == "mask":
        _freeze_canny_adapters()
        # No mask dropout in this branch: its null embedding is not in the
        # forward graph and must be frozen for DDP unused-parameter safety.
        model.mask_null_embed.requires_grad_(False)
        if accelerator.is_main_process:
            logger.info(
                "[M06-FiLM] phase=mask -> mask_proj_x + mask_gates trainable, "
                "canny_proj_x + canny_gates frozen"
            )
        _log_trainable()
    elif _data_phase == "canny":
        _freeze_mask_adapters()
        model.canny_null_embed.requires_grad_(False)
        if accelerator.is_main_process:
            logger.info(
                "[M06-FiLM] phase=canny -> canny_proj_x + canny_gates trainable, "
                "mask_proj_x + mask_gates frozen"
            )
        _log_trainable()
    elif _data_phase == "canny_mask":
        if float(train_config["train"].get("mask_dropout_prob", 0.0)) == 0.0:
            model.mask_null_embed.requires_grad_(False)
        if float(train_config["train"].get("canny_dropout_prob", 0.0)) == 0.0:
            model.canny_null_embed.requires_grad_(False)
        if accelerator.is_main_process:
            logger.info(
                "[M06-FiLM] phase=canny_mask -> mask_proj_x + mask_gates + "
                "canny_proj_x + canny_gates trainable"
            )
        _log_trainable()
    else:
        # warmup: freeze all four adapter params (no conditioning; avoid DDP crash)
        _freeze_mask_adapters()
        _freeze_canny_adapters()
        if accelerator.is_main_process:
            logger.info(
                f"[M06-FiLM] phase={_data_phase} -> all adapters (proj + gates) frozen "
                "(no conditioning input; avoids find_unused_parameters crash)"
            )
        _log_trainable()

    train_config["train"]["resume"] = train_config["train"].get("resume", False)
    _resume_opt_state = None
    _resume_epoch = 0
    _resume_best_val_loss = None
    if train_config["train"]["resume"]:
        explicit_ckpt = train_config["train"].get("resume_checkpoint")
        ckpt_files = glob(f"{checkpoint_dir}/[0-9]*.pt")
        if explicit_ckpt:
            ckpt_files = [explicit_ckpt]
        if ckpt_files:
            ckpt_files.sort(
                key=lambda x: int(os.path.basename(x).split(".")[0])
                if os.path.basename(x).split(".")[0].isdigit() else -1
            )
            latest_ckpt = ckpt_files[-1]
            ckpt = torch.load(latest_ckpt, map_location=lambda storage, loc: storage)
            model.load_state_dict(ckpt["model"])
            ema.load_state_dict(ckpt["ema"])
            if "train_steps" in ckpt:
                train_steps = int(ckpt["train_steps"])
            else:
                train_steps = int(latest_ckpt.split("/")[-1].split(".")[0])
            _resume_opt_state = ckpt.get("opt")
            _resume_epoch = int(
                ckpt.get("epoch", train_config["train"].get("resume_epoch", 0))
            )
            _resume_best_val_loss = ckpt.get("best_val_loss")
            if accelerator.is_main_process:
                logger.info(
                    "Resuming from checkpoint: %s | step=%d epoch=%d opt=%s best=%s",
                    latest_ckpt, train_steps, _resume_epoch,
                    "restored" if _resume_opt_state is not None else "missing",
                    _resume_best_val_loss,
                )
        else:
            if accelerator.is_main_process:
                logger.info("No checkpoint found. Starting from scratch.")

    model, opt, loader = accelerator.prepare(model, opt, loader)
    if _resume_opt_state is not None:
        opt.load_state_dict(_resume_opt_state)
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)
    # Identical initialization across jobs comes from _global_seed above.
    # Rank-specific streams avoid duplicating t/noise across DDP workers.
    torch.manual_seed(_global_seed + accelerator.process_index)
    torch.cuda.manual_seed_all(_global_seed + accelerator.process_index)
    np.random.seed(_global_seed + accelerator.process_index)

    if not train_config["train"]["resume"]:
        train_steps = 0
    log_steps = 0
    running_loss = 0.0
    start_time = time()

    # Pixel-specific t-sampling hyperparams (JiT lognormal convention)
    p_mean = train_config["train"].get("p_mean", -0.8)
    p_std = train_config["train"].get("p_std", 0.8)
    t_eps = train_config["train"].get("t_eps", 5e-2)
    sampler_config = train_config["train"].get("t_sampler")

    if accelerator.is_main_process:
        if sampler_config and sampler_config.get("mode") == "mixed_full_snr":
            local_batch = int(train_config["train"]["global_batch_size"]) // int(
                accelerator.num_processes
            )
            n_official = int(round(local_batch * float(sampler_config["original_fraction"])))
            n_reliable = int(round(local_batch * float(sampler_config["reliable_fraction"])))
            n_high = local_batch - n_official - n_reliable
            logger.info(
                "Full-SNR sampler protocol: %s; per-rank fixed quotas "
                "official/reliable/high=%d/%d/%d",
                json.dumps(sampler_config, sort_keys=True),
                n_official, n_reliable, n_high,
            )
        elif sampler_config and sampler_config.get("mode") == "fixed_t":
            fixed_t = float(sampler_config.get("fixed_t", float("nan")))
            if not math.isfinite(fixed_t) or not 0.0 < fixed_t < 1.0:
                raise ValueError("fixed_t sampler requires finite 0 < fixed_t < 1")
            logger.info(
                "Single-SNR sampler protocol: fixed_t=%.9f; objective=direct_x1_mse",
                fixed_t,
            )
        elif sampler_config:
            raise ValueError(
                f"unsupported train.t_sampler mode: {sampler_config.get('mode')!r}"
            )
        else:
            logger.info("Timestep sampler protocol: legacy lognormal only")

    # Per-epoch val + early stop state
    epoch = _resume_epoch
    best_val_loss = (
        float(_resume_best_val_loss)
        if _resume_best_val_loss is not None else _init_best_val_loss
    )
    no_improve_epochs = 0
    val_seed = train_config["train"].get("val_seed", 12345)
    early_stop_patience = train_config["train"].get("early_stop_patience", 5)
    early_stop_min_epochs = train_config["train"].get("early_stop_min_epochs", 10)
    val_every_n_epochs = train_config["train"].get("val_every_n_epochs", 1)

    val_loss_log_path = (
        os.path.join(experiment_dir, "val_loss_log.jsonl")
        if accelerator.is_main_process
        else None
    )

    if accelerator.is_main_process:
        logger.info(f"Training for up to {train_config['train']['max_steps']} steps...")

    _early_stop = False
    while True:
        for batch in loader:
            image, mask_2d, canny_2d, label = batch
            image = image.to(device)                          # [B, 3, 256, 256]
            mask_2d = mask_2d.to(device, non_blocking=True)   # [B, 256, 256]
            canny_2d = canny_2d.to(device, non_blocking=True)  # [B, 256, 256]
            label = label.to(device)                           # [B]
            if _condition_mode in ("uncond", "mask_only", "canny_only", "mask_canny"):
                # Explicitly remove class information for every training sample.
                label = torch.full_like(label, NULL_CLASS)
            elif _class_drop_fraction > 0:
                class_drop_t = _fixed_quota_drop_mask(
                    label.shape[0], _class_drop_fraction, device
                )
                label = torch.where(
                    class_drop_t,
                    torch.full_like(label, NULL_CLASS),
                    label,
                )

            mask_for_model = mask_2d if _data_phase in ("mask", "canny_mask") else None
            canny_for_model = canny_2d if _data_phase in ("canny", "canny_mask") else None

            # Multi-drop oracle v2: per-sample independent spatial-condition drop.
            # Class drop is handled explicitly above with a fixed per-rank quota;
            # Pixel JiT's LabelEmbedder itself does not perform CFG dropout.
            # Yaml flags: train.mask_dropout_prob, train.canny_dropout_prob (default 0).
            # Compute per-sample drop bool tensor → model uses learnable null embed for dropped.
            _mask_drop_p = float(train_config["train"].get("mask_dropout_prob", 0.0))
            _canny_drop_p = float(train_config["train"].get("canny_dropout_prob", 0.0))
            mask_drop_t = None
            canny_drop_t = None
            if mask_for_model is not None and _mask_drop_p > 0:
                B_ = mask_for_model.shape[0]
                mask_drop_t = (torch.rand(B_, device=device) < _mask_drop_p)
            if canny_for_model is not None and _canny_drop_p > 0:
                B_ = canny_for_model.shape[0]
                canny_drop_t = (torch.rand(B_, device=device) < _canny_drop_p)

            # DEVIATION: pixel JiT uses manual velocity loss (no transport.training_losses)
            loss = _compute_velocity_loss(
                model, image, label, device, p_mean, p_std, t_eps,
                mask=mask_for_model,
                canny=canny_for_model,
                mask_drop=mask_drop_t,
                canny_drop=canny_drop_t,
                sampler_config=sampler_config,
            )

            # NaN guard: abort instead of propagating NaN weights
            if not torch.isfinite(loss).all():
                raise RuntimeError(
                    f"[NaN guard] non-finite loss at step={train_steps}: {loss.item()}. "
                    "Lower lr or strengthen grad clip and restart."
                )

            opt.zero_grad()
            accelerator.backward(loss)
            if "max_grad_norm" in train_config["optimizer"]:
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), train_config["optimizer"]["max_grad_norm"]
                    )
            opt.step()
            update_ema(ema, accelerator.unwrap_model(model))

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % train_config["train"]["log_every"] == 0:
                torch.cuda.synchronize()
                end_time = time()
                sps = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                if dist.is_initialized():
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / dist.get_world_size()
                else:
                    avg_loss = avg_loss.item()
                if accelerator.is_main_process:
                    logger.info(
                        f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, "
                        f"Steps/Sec: {sps:.2f}"
                    )
                    writer.add_scalar("Loss/train", avg_loss, train_steps)
                running_loss = 0.0
                log_steps = 0
                start_time = time()

            if train_steps >= train_config["train"]["max_steps"]:
                break

        # ── End of epoch ──────────────────────────────────────────────────────
        epoch += 1

        if val_loader is not None and (epoch % val_every_n_epochs == 0):
            # Pure-uncond oracle: force null label in val (matches train cls_dropout=1.0)
            _val_force_null = bool(train_config["train"].get("val_force_null_label", False))
            if _val_force_null:
                _num_classes = train_config["data"].get("num_classes", 1000)
                @torch.no_grad()
                def evaluate_deterministic_pixel_null(model, val_loader_, device_, p_mean_, p_std_, t_eps_, phase=None, val_seed=12345, val_n_batches=8):
                    model.eval()
                    total = torch.tensor(0.0, device=device_); n = 0
                    for bi, batch in enumerate(val_loader_):
                        if bi >= val_n_batches: break
                        image, mask_2d, canny_2d, label = batch
                        image = image.to(device_); label = label.to(device_)
                        null_y = torch.full_like(label, _num_classes)
                        torch.manual_seed(val_seed + bi)
                        torch.cuda.manual_seed_all(val_seed + bi)
                        mse = _compute_velocity_loss(
                            model, image, null_y, device_, p_mean_, p_std_, t_eps_,
                            sampler_config=sampler_config,
                        )
                        total += mse.detach(); n += 1
                    return total / max(n, 1)
                val_mse = evaluate_deterministic_pixel_null(
                    model, val_loader, device, p_mean, p_std, t_eps,
                    phase=_data_phase,
                    val_seed=val_seed, val_n_batches=val_n_batches,
                )
            else:
                val_mse = evaluate_deterministic_pixel(
                    model, val_loader, device, p_mean, p_std, t_eps,
                    phase=_data_phase,
                    # Must match training semantics; in particular mask-only
                    # validation requires y=NULL_CLASS rather than GT class.
                    condition_mode=_condition_mode,
                    val_seed=val_seed, val_n_batches=val_n_batches,
                    sampler_config=sampler_config,
                )
            accelerator.wait_for_everyone()
            val_mse_all = accelerator.gather(val_mse.detach().unsqueeze(0)).mean().item()

            # NaN guard on val
            if not (
                val_mse_all == val_mse_all
                and val_mse_all not in (float("inf"), float("-inf"))
            ):
                raise RuntimeError(
                    f"[NaN guard] non-finite val_mse at epoch={epoch}: {val_mse_all}. "
                    "Lower lr and restart."
                )

            if accelerator.is_main_process:
                logger.info(
                    f"[val] epoch={epoch} step={train_steps} "
                    f"mse={val_mse_all:.6f} best={best_val_loss:.6f}"
                )
                writer.add_scalar("Loss/val_epoch", val_mse_all, epoch)
                record = {
                    "epoch": epoch,
                    "step": train_steps,
                    "val_mse": val_mse_all,
                    "best_val_loss": best_val_loss,
                    "wall_time": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
                with open(val_loss_log_path, "a") as fh:
                    fh.write(json.dumps(record) + "\n")
                if val_mse_all < best_val_loss:
                    best_val_loss = val_mse_all
                    no_improve_epochs = 0
                    best_path = os.path.join(checkpoint_dir, "best.pt")
                    unwrapped = accelerator.unwrap_model(model)
                    try:
                        torch.save(
                            {
                                "model": unwrapped.state_dict(),
                                "opt": opt.state_dict(),
                                "ema": ema.state_dict(),
                                "train_steps": train_steps,
                                "best_val_loss": best_val_loss,
                                "epoch": epoch,
                            },
                            best_path,
                        )
                        logger.info(f"[val] new best mse={best_val_loss:.6f} -> saved {best_path}")
                    except Exception as _save_err:
                        # NFS write failures (e.g. PytorchStreamWriter mid-write) shouldn't
                        # kill the training — val trajectory is the deliverable, not best.pt.
                        # Remove truncated file so next save_best attempt is fresh.
                        try: os.remove(best_path)
                        except Exception: pass
                        logger.warning(f"[val] save best failed (NFS hiccup?): {_save_err}; continuing")
                else:
                    no_improve_epochs += 1
                    logger.info(
                        f"[val] no improvement {no_improve_epochs}/{early_stop_patience}"
                    )

                snapshot_epochs = {
                    int(value)
                    for value in train_config["train"].get("snapshot_epochs", [])
                }
                if epoch in snapshot_epochs:
                    snapshot_path = os.path.join(
                        checkpoint_dir, f"epoch_{epoch:04d}.pt"
                    )
                    unwrapped = accelerator.unwrap_model(model)
                    torch.save(
                        {
                            "model": unwrapped.state_dict(),
                            "opt": opt.state_dict(),
                            "ema": ema.state_dict(),
                            "train_steps": train_steps,
                            "best_val_loss": best_val_loss,
                            "epoch": epoch,
                        },
                        snapshot_path,
                    )
                    logger.info(
                        f"[snapshot] epoch={epoch} step={train_steps} "
                        f"-> saved {snapshot_path}"
                    )

            _no_improve_t = torch.tensor(no_improve_epochs, device=device)
            if dist.is_initialized():
                dist.broadcast(_no_improve_t, src=0)
            no_improve_epochs = int(_no_improve_t.item())
            model.train()

            if epoch >= early_stop_min_epochs and no_improve_epochs >= early_stop_patience:
                if accelerator.is_main_process:
                    logger.info(
                        f"[val] EARLY STOP triggered at epoch {epoch} "
                        f"(patience {early_stop_patience} reached)"
                    )
                _early_stop = True

        if train_steps >= train_config["train"]["max_steps"] or _early_stop:
            break

    # ── Save last checkpoint ──────────────────────────────────────────────────
    if accelerator.is_main_process:
        last_path = os.path.join(checkpoint_dir, "last.pt")
        unwrapped = accelerator.unwrap_model(model)
        torch.save(
            {
                "model": unwrapped.state_dict(),
                "opt": opt.state_dict(),
                "ema": ema.state_dict(),
                "train_steps": train_steps,
                "best_val_loss": best_val_loss,
                "epoch": epoch,
            },
            last_path,
        )
        logger.info(f"[end] saved last ckpt -> {last_path}")
        # Fail-loud if no best.pt produced (per feedback_train_end_best_pt_required)
        # EXCEPT for oracle-style runs where val trajectory is the deliverable
        # (reset_best_val_loss=true marks oracle/delta-t phase)
        best_pt_path = os.path.join(checkpoint_dir, "best.pt")
        _is_oracle = bool(train_config["train"].get("reset_best_val_loss", False))
        if not os.path.exists(best_pt_path) and not _is_oracle:
            fail_marker = os.path.join(experiment_dir, "_FAIL")
            with open(fail_marker, "w") as _f:
                _f.write(f"training ended without best.pt at {best_pt_path}\n")
            raise RuntimeError(
                f"[FAIL] training ended without best.pt at {best_pt_path}"
            )
        if not os.path.exists(best_pt_path) and _is_oracle:
            logger.warning(f"[oracle] no best.pt saved (NFS hiccups?) — val trajectory is in training_results.txt / log; continuing")
        logger.info("Done!")

    accelerator.wait_for_everyone()
    if dist.is_initialized():
        dist.destroy_process_group()
    return accelerator


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(args=None):
    parser = argparse.ArgumentParser(description="Native JiT conditional training")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML training config.",
    )
    if args is None:
        args = parser.parse_args()

    train_config = load_config(args.config)
    train_config["_config_path"] = args.config
    source = train_config.get("resources", {}).get("jit_source")
    if source:
        sys.path.insert(0, str(Path(source).expanduser().resolve()))

    _ddp_bucket_cap_mb = train_config.get("train", {}).get("ddp_bucket_cap_mb", None)
    if _ddp_bucket_cap_mb is not None:
        from accelerate import DistributedDataParallelKwargs
        ddp_kwargs = DistributedDataParallelKwargs(
            bucket_cap_mb=_ddp_bucket_cap_mb,
            find_unused_parameters=False,
            gradient_as_bucket_view=False,
        )
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    else:
        accelerator = Accelerator()

    do_train(train_config, accelerator)


if __name__ == "__main__":
    main()
