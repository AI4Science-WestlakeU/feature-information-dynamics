"""Shared utilities for mask+patch conditional finetune training scripts.

Backbone-agnostic helpers shared by train_rae_mask_patch.py, train_sit_mask_patch.py,
and train_lightningdit_mask_patch.py.  Each backbone-specific script imports from
here and only implements: backbone instantiation, paper ckpt loading, dataset
subclass, and the forward/loss step.

Token grid convention (all backbones):
    NUM_TOKENS = 256  (16×16 grid)
    TOKEN_GRID  = 16
    BLOCK_PX    = 16  (pixels per token side)
    DINO_DIM    = 768 (16×16×3 RGB pixels per token, matches DINOv2-B)

Phase drop-probability protocol (M06 chained-only, 2026-04-28 revised):
    warmup : label_drop=0.10  mask_drop=1.0  patch_drop=1.0
    mask   : label_drop=0.00  mask_drop=0.0  patch_drop=1.0
    patch  : label_drop=0.00  mask_drop=0.0  patch_drop=0.0
    canny  : label_drop=0.00  mask_drop=0.0  patch_drop=1.0  canny_drop=0.0
"""
from __future__ import annotations

import argparse
import logging
import os
from collections import OrderedDict
from glob import glob
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: F401

# ── Constants ──────────────────────────────────────────────────────────────────
NUM_TOKENS: int = 256
TOKEN_GRID: int = 16
BLOCK_PX: int = 16
DINO_DIM: int = 768  # 16×16×3

# ── Phase defaults (M06 chained protocol, 2026-04-28 revised) ─────────────────
# Chained strict-conditioning protocol (user decision 2026-04-28):
#   warmup: class-cond warmup, mask/patch always dropped (=unconditional on mask/patch)
#   mask:   strict mask conditioning, no drop — model learns mask always present
#   patch:  strict mask+patch conditioning, no drop — model learns both always present
# drop_prob=0.0 → condition always passed; drop_prob=1.0 → condition always zeroed.
_PHASE_DEFAULTS: dict[str, dict[str, float]] = {
    "warmup": dict(label_drop_prob=0.10, mask_drop_prob=1.0, patch_drop_prob=1.0, canny_drop_prob=1.0),
    "mask":   dict(label_drop_prob=0.00, mask_drop_prob=0.0, patch_drop_prob=1.0, canny_drop_prob=1.0),
    "patch":  dict(label_drop_prob=0.00, mask_drop_prob=0.0, patch_drop_prob=0.0, canny_drop_prob=1.0),
    "canny":  dict(label_drop_prob=0.00, mask_drop_prob=0.0, patch_drop_prob=1.0, canny_drop_prob=0.0),
}

PHASE_CHOICES = ("warmup", "mask", "patch", "canny")


def apply_phase_defaults(args: argparse.Namespace, phase: str) -> None:
    """Apply _PHASE_DEFAULTS for phase to any drop arg still at argparse default (0.0).

    Args:
        args: argparse.Namespace with label_drop_prob, mask_drop_prob, patch_drop_prob.
        phase: one of "warmup", "mask", "patch".
    """
    for k, v in _PHASE_DEFAULTS[phase].items():
        if getattr(args, k, 0.0) == 0.0 and v != 0.0:
            setattr(args, k, v)


# ── Per-token feature builders ─────────────────────────────────────────────────

def build_mask_per_token(mask_2d: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Reshape pixel mask to per-token block vectors.

    Args:
        mask_2d: [B, 256, 256] float32 in {0, 1}
        device:  target device

    Returns:
        [B, 256, 256] — dim-1 = token index, dim-2 = 256 pixels in 16×16 block
    """
    B = mask_2d.shape[0]
    m = mask_2d.to(device)                                      # [B, 256, 256]
    m = m.view(B, TOKEN_GRID, BLOCK_PX, TOKEN_GRID, BLOCK_PX)
    m = m.permute(0, 1, 3, 2, 4).contiguous()                  # [B, 16, 16, 16, 16]
    return m.view(B, NUM_TOKENS, BLOCK_PX * BLOCK_PX)           # [B, 256, 256]


def build_canny_per_token(canny_2d: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Reshape dense Canny edge map to per-token block vectors.

    Args:
        canny_2d: [B, 256, 256] float32/bool in {0,1}
        device: target device

    Returns:
        [B, 256, 256] — same token/block convention as mask.
    """
    return build_mask_per_token(canny_2d.float(), device)


def build_patch_per_token(
    x_3ch: torch.Tensor,
    patch_tok_idx: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Build sparse per-token patch RGB vector from a chosen token index.

    Only the token at patch_tok_idx is filled; all others are zero.

    Args:
        x_3ch:         [B, 3, 256, 256] float32 in [-1, 1]
        patch_tok_idx: [B] int64 in [0, 255]
        device:        target device

    Returns:
        [B, 256, 768] float32
    """
    B = x_3ch.shape[0]
    x = x_3ch.to(device)                                               # [B, 3, 256, 256]
    x = x.view(B, 3, TOKEN_GRID, BLOCK_PX, TOKEN_GRID, BLOCK_PX)
    all_toks = x.permute(0, 2, 4, 3, 5, 1).contiguous()               # [B, 16, 16, 16, 16, 3]
    all_toks = all_toks.view(B, NUM_TOKENS, BLOCK_PX * BLOCK_PX * 3)  # [B, 256, 768]
    patch = torch.zeros_like(all_toks)                                  # [B, 256, 768]
    idx = patch_tok_idx.to(device).view(B, 1, 1).expand(B, 1, DINO_DIM)  # [B, 1, 768]
    patch.scatter_(dim=1, index=idx, src=all_toks.gather(dim=1, index=idx))
    return patch                                                         # [B, 256, 768]


# ── Conditioning drop ──────────────────────────────────────────────────────────

def drop_cond_batch(tensor: torch.Tensor, drop_prob: float) -> torch.Tensor:
    """Zero-out entire samples in batch with probability drop_prob.

    Args:
        tensor:    [B, ...] conditioning tensor
        drop_prob: per-sample zeroing probability in [0, 1]

    Returns:
        Tensor with dropped rows zeroed; same device and dtype.
    """
    if drop_prob <= 0.0:
        return tensor
    if drop_prob >= 1.0:
        return torch.zeros_like(tensor)
    keep = torch.rand(tensor.shape[0], device=tensor.device) >= drop_prob  # [B]
    return tensor * keep.view((-1,) + (1,) * (tensor.ndim - 1)).to(tensor.dtype)


# ── EMA / grad helpers ─────────────────────────────────────────────────────────

@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float = 0.9999) -> None:
    """Exponential moving average update: ema ← decay*ema + (1-decay)*model.

    Strips _orig_mod. prefix from torch.compile() wrapper if present.
    """
    src = model._orig_mod if hasattr(model, "_orig_mod") else model
    ema_p = OrderedDict(ema.named_parameters())
    for name, param in OrderedDict(src.named_parameters()).items():
        ema_p[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model: nn.Module, flag: bool = True) -> None:
    """Set requires_grad for all parameters of model."""
    for p in model.parameters():
        p.requires_grad = flag


# ── Logger ─────────────────────────────────────────────────────────────────────

def create_logger(logging_dir: Optional[str]) -> logging.Logger:
    """Rank-0 logger (stdout + optional file); non-rank-0 → NullHandler.

    Args:
        logging_dir: directory for log.txt file, or None for stdout only.

    Returns:
        Configured Logger for this module.
    """
    if dist.get_rank() == 0:
        handlers: list = [logging.StreamHandler()]
        if logging_dir:
            handlers.append(logging.FileHandler(f"{logging_dir}/log.txt"))
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=handlers,
        )
        return logging.getLogger(__name__)
    lg = logging.getLogger(__name__)
    lg.addHandler(logging.NullHandler())
    return lg


# ── DDP init ───────────────────────────────────────────────────────────────────

def init_ddp(
    global_batch_size: int,
    grad_accum_steps: int,
    seed: int,
    precision: str,
) -> tuple[int, int, torch.device, int, dict]:
    """Initialize NCCL process group and return DDP context.

    Args:
        global_batch_size: total batch size across all ranks.
        grad_accum_steps:  gradient accumulation steps.
        seed:              base random seed (rank-offset applied internally).
        precision:         "bf16" or "fp32".

    Returns:
        (rank, world_size, device, micro_batch_size, autocast_kwargs)
    """
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device_idx = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_idx)
    device = torch.device("cuda", device_idx)
    torch.manual_seed(seed * world_size + rank)
    torch.cuda.manual_seed(seed * world_size + rank)
    denom = world_size * grad_accum_steps
    if global_batch_size % denom != 0:
        raise ValueError(
            f"global_batch_size={global_batch_size} must be divisible by "
            f"world_size*grad_accum_steps={world_size}*{grad_accum_steps}={denom}"
        )
    micro_bs = global_batch_size // (world_size * grad_accum_steps)
    use_bf16 = precision == "bf16"
    # Caller uses: with torch.amp.autocast('cuda', **ac_kw)
    return rank, world_size, device, micro_bs, dict(dtype=torch.bfloat16, enabled=use_bf16)


# ── Experiment directory ───────────────────────────────────────────────────────

def make_exp_dir(
    rank: int,
    results_dir: str,
    backbone_tag: str,
    phase: str,
    precision: str,
) -> tuple[Optional[str], Optional[str]]:
    """Create experiment + checkpoint directories on rank 0.

    Args:
        rank:         DDP rank; only rank 0 creates directories.
        results_dir:  base results directory.
        backbone_tag: short identifier for backbone, e.g. "rae", "sit", "ldit".
        phase:        training phase, e.g. "warmup".
        precision:    "bf16" or "fp32".

    Returns:
        (exp_dir, ckpt_dir) on rank 0; (None, None) on other ranks.
    """
    if rank != 0:
        return None, None
    os.makedirs(results_dir, exist_ok=True)
    idx = len(glob(f"{results_dir}/*"))
    exp_dir = os.path.join(results_dir, f"{idx:03d}-{backbone_tag}-mask-patch-{phase}-{precision}")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    return exp_dir, ckpt_dir


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_checkpoint(
    rank: int,
    ckpt_dir: Optional[str],
    model: DDP,
    ema: nn.Module,
    opt: torch.optim.Optimizer,
    schedl: object,
    train_steps: int,
    epoch: int,
    phase: str,
    label_drop_prob: float,
    mask_drop_prob: float,
    patch_drop_prob: float,
    save_last_freq: int,
    save_ckpt_epochs: set,
    extra: Optional[dict] = None,
) -> None:
    """Save named epoch checkpoint and checkpoint-last.pt on rank 0.

    Args:
        rank:              DDP rank; only rank 0 saves.
        ckpt_dir:          checkpoint directory (None on non-rank-0).
        model:             DDP-wrapped model.
        ema:               EMA model.
        opt:               optimizer.
        schedl:            lr scheduler.
        train_steps:       global training step counter.
        epoch:             current epoch index.
        phase:             training phase label.
        label_drop_prob:   label drop probability used this phase.
        mask_drop_prob:    mask drop probability used this phase.
        patch_drop_prob:   patch drop probability used this phase.
        save_last_freq:    save every N epochs regardless of save_ckpt_epochs.
        save_ckpt_epochs:  set of epoch indices for named checkpoint saves.
        extra:             optional dict merged into checkpoint payload.
    """
    if rank != 0 or ckpt_dir is None:
        return
    payload: dict = {
        "model": model.module.state_dict(),
        "ema": ema.state_dict(),
        "opt": opt.state_dict(),
        "scheduler": schedl.state_dict(),  # type: ignore[attr-defined]
        "train_steps": train_steps,
        "epoch": epoch,
        "phase": phase,
        "label_drop_prob": label_drop_prob,
        "mask_drop_prob": mask_drop_prob,
        "patch_drop_prob": patch_drop_prob,
    }
    if extra:
        payload.update(extra)
    # Rolling save: always write (matches legacy SiT/RAE behavior — prevents total
    # loss of training state if a scheduled epoch save is never reached).
    torch.save(payload, os.path.join(ckpt_dir, "checkpoint-last.pt"))
    # Named save: only at scheduled epochs.
    should_named = (epoch in save_ckpt_epochs
                    or (save_last_freq > 0 and epoch % save_last_freq == 0))
    if should_named:
        torch.save(payload, os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt"))


def _strip_backbone_prefix(sd: dict) -> dict:
    """Strip legacy 'backbone.' prefix from state dict keys (compat with old wrapper).

    Old LightningDiT_MaskPatch wrapped LightningDiT as self.backbone, so saved keys had
    'backbone.x_embedder.*' etc. New version directly inherits LightningDiT, so keys are
    'x_embedder.*'. If no 'backbone.' keys present, returns sd unchanged (idempotent).
    """
    if not any(k.startswith("backbone.") for k in sd):
        return sd
    return {k.removeprefix("backbone."): v for k, v in sd.items()}


def load_resume_checkpoint(
    resume_path: str,
    model: nn.Module,
    ema: nn.Module,
) -> tuple[int, int, Optional[dict], Optional[dict]]:
    """Load resume checkpoint into model and EMA.

    Args:
        resume_path: path to checkpoint-last.pt from this script family.
        model:       model to load state dict into (strict=True).
        ema:         EMA model to load state dict into (strict=True).

    Returns:
        (train_steps, start_epoch, opt_state, sched_state)
    """
    ck = torch.load(resume_path, map_location="cpu", weights_only=False)
    model.load_state_dict(_strip_backbone_prefix(ck["model"]), strict=True)
    ema.load_state_dict(_strip_backbone_prefix(ck["ema"]), strict=True)
    start_epoch = int(ck.get("epoch", -1)) + 1
    return int(ck.get("train_steps", 0)), start_epoch, ck.get("opt"), ck.get("scheduler")


def load_phase_checkpoint(
    init_from_phase: str,
    model: nn.Module,
    ema: nn.Module,
    expected_source_phase: Optional[str] = None,
) -> dict:
    """Initialize a new phase from a wrapper checkpoint without optimizer/scheduler tail.

    This is intentionally separate from load_paper_checkpoint and load_resume_checkpoint:
    - paper checkpoints initialize only the official backbone with strict=False;
    - phase checkpoints initialize the full wrapper model/EMA with strict=True;
    - optimizer, scheduler, epoch, and train_steps are discarded for a fresh phase schedule.
    """
    ck = torch.load(init_from_phase, map_location="cpu", weights_only=False)
    if "model" not in ck:
        raise KeyError(
            f"{init_from_phase} is not a wrapper phase checkpoint: missing key 'model'. "
            "Use --pretrained_ckpt/--ckpt only for paper pretrain checkpoints."
        )
    source_phase = ck.get("phase")
    if expected_source_phase and source_phase != expected_source_phase:
        raise ValueError(
            f"init_from_phase source phase mismatch: expected {expected_source_phase!r}, "
            f"got {source_phase!r} from {init_from_phase}"
        )
    model.load_state_dict(_strip_backbone_prefix(ck["model"]), strict=True)
    if "ema" in ck:
        ema.load_state_dict(_strip_backbone_prefix(ck["ema"]), strict=True)
    else:
        ema.load_state_dict(_strip_backbone_prefix(ck["model"]), strict=True)
    return {
        "source_phase": source_phase,
        "source_epoch": ck.get("epoch"),
        "source_train_steps": ck.get("train_steps"),
    }


def validate_checkpoint_policy(
    *,
    paper_ckpt: Optional[str],
    resume: Optional[str],
    init_from_phase: Optional[str],
) -> str:
    """Return checkpoint mode after enforcing mutually exclusive load policies."""
    modes = [
        ("paper-pretrain", paper_ckpt),
        ("resume", resume),
        ("phase-init", init_from_phase),
    ]
    active = [name for name, path in modes if path]
    if len(active) > 1:
        raise ValueError(
            "Checkpoint load policy is mutually exclusive: choose exactly one of "
            "--pretrained_ckpt/--ckpt, --resume, or --init_from_phase. "
            f"Got {active}."
        )
    return active[0] if active else "scratch"


def set_backbone_class_dropout(model: nn.Module, class_dropout_prob: float) -> list[str]:
    """Set internal LabelEmbedder dropout knobs so external label_drop_prob is authoritative."""
    touched: list[str] = []
    for module_name, module in model.named_modules():
        if hasattr(module, "dropout_prob"):
            setattr(module, "dropout_prob", class_dropout_prob)
            touched.append(f"{module_name}.dropout_prob")
    return touched


def log_effective_train_config(
    logger: logging.Logger,
    *,
    checkpoint_mode: str,
    class_dropout_prob: float,
    class_dropout_targets: list[str],
    global_batch_size: int,
    micro_batch_size: int,
    grad_accum_steps: int,
    world_size: int,
    lr: float,
    betas: tuple[float, float],
    scheduler: str,
    t_eps: float,
    timestep_distribution: str,
    latent_norm: bool,
    latent_multiplier: float,
) -> None:
    """Emit the launch-critical effective config in one stable log block."""
    logger.info(
        "[effective_config] checkpoint_mode=%s class_dropout_prob=%.4f targets=%s",
        checkpoint_mode, class_dropout_prob, ",".join(class_dropout_targets) or "<none>",
    )
    logger.info(
        "[effective_config] global_batch=%d micro_batch=%d world_size=%d grad_accum=%d",
        global_batch_size, micro_batch_size, world_size, grad_accum_steps,
    )
    logger.info(
        "[effective_config] lr=%.6g optimizer=AdamW betas=(%.3f,%.3f) scheduler=%s",
        lr, betas[0], betas[1], scheduler,
    )
    logger.info(
        "[effective_config] t_eps=%.6g timestep_distribution=%s latent_norm=%s latent_multiplier=%.6g",
        t_eps, timestep_distribution, latent_norm, latent_multiplier,
    )


def load_paper_checkpoint(
    model_or_submodule: nn.Module,
    ckpt_path: str,
    adapter_prefixes: tuple[str, ...],
    logger: Optional[logging.Logger] = None,
) -> dict:
    """Load a paper pretrain checkpoint into model or submodule with strict=False.

    Handles flat state_dict or {"ema", "model"} dict formats.  Strips "module."
    and "net." DDP prefixes.  Keys in adapter_prefixes are expected missing
    (zero-init adapters) and suppressed from warnings.  Non-tensor entries in the
    source state_dict (e.g. integer step counters, config dicts) are filtered out.

    Args:
        model_or_submodule: target nn.Module (may be backbone sub-module).
        ckpt_path:          path to checkpoint file (.pt/.pth).
        adapter_prefixes:   key prefixes for zero-init adapter layers to suppress.
        logger:             optional logger for info/warning messages.

    Returns:
        dict with keys: key_used, missing_keys, unexpected_keys.
    """
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    raw = torch.load(str(path), map_location="cpu", weights_only=False)
    src_sd, key_used = None, None
    for cand in ("ema", "model"):
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
    result = model_or_submodule.load_state_dict(src_sd, strict=False)
    non_adapter_missing = [
        k for k in result.missing_keys
        if not any(k.startswith(p) for p in adapter_prefixes)
    ]
    if non_adapter_missing and logger:
        logger.warning("[load_paper_ckpt] non-adapter missing keys (%d): %s",
                       len(non_adapter_missing), non_adapter_missing[:8])
    if logger:
        logger.info("[load_paper_ckpt] key='%s' missing=%d unexpected=%d",
                    key_used, len(result.missing_keys), len(result.unexpected_keys))
    return dict(
        key_used=key_used,
        missing_keys=list(result.missing_keys),
        unexpected_keys=list(result.unexpected_keys),
    )


# ── Param count logging ────────────────────────────────────────────────────────

def log_param_counts(model: nn.Module, logger: logging.Logger,
                     warn_threshold: float = 700e6) -> None:
    """Log trainable vs total parameter counts; warn if trainable > threshold.

    Args:
        model:           model to inspect.
        logger:          rank-0 logger.
        warn_threshold:  trainable param count above which a warning is issued.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Params — trainable=%.2fM total=%.2fM", trainable / 1e6, total / 1e6)
    if trainable > warn_threshold:
        logger.warning(
            "[param_check] trainable=%.2fM > %.0fM: verify frozen components.",
            trainable / 1e6, warn_threshold / 1e6,
        )


# ── Adapter sanity check ───────────────────────────────────────────────────────

def log_adapter_norms(
    model_module: nn.Module,
    adapter_names: tuple[str, ...],
    logger: logging.Logger,
) -> None:
    """Log weight norms of zero-init adapters; any non-zero indicates contamination.

    Args:
        model_module:  unwrapped model (model.module from DDP).
        adapter_names: attribute names of adapter Linear layers to check.
        logger:        rank-0 logger.
    """
    for name in adapter_names:
        layer = getattr(model_module, name, None)
        if layer is not None and hasattr(layer, "weight"):
            logger.info("[sanity] adapter %s weight_norm=%.6f (expect ~0 at init)",
                        name, layer.weight.norm().item())
