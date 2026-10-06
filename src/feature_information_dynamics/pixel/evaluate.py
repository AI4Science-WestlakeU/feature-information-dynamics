"""Native Pixel measurement with the historical evaluation protocol.

Preserves checkpoint selection, condition modes, time grid, per-batch random
noise and distributed reduction. The model uses the corrected public backend.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader, Subset

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
from feature_information_dynamics.data.raw_image_mask_canny_dataset import RawImageMaskCannyDataset
from feature_information_dynamics.data.train_lib_mask_patch import (
    NUM_TOKENS, build_mask_per_token, build_canny_per_token,
)
NULL_CLASS = 1000

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ── Constants ─────────────────────────────────────────────────────────────────

_IMG_SIZE: int = 256
_T_EPS: float = 1e-4   # floor for (1-t) denominator; avoids div-by-zero near t=1
_DEFAULT_IMAGENET_ROOT: str = ""
# JiT LabelEmbedder = nn.Embedding(num_classes+1, hidden). Internal +1 covers null.
# Eval passes num_classes=1000 (NOT 1001); null class index = NULL_CLASS = 1000.
# This differs from chained train protocol where num_classes=1001 is required for
# LabelEmbedder to host an extra "mask/canny conditioning" slot — see project memory
# `feedback_label_embed_chained_protocol`. Eval here matches the inference-time
# convention used by all single-phase JiT_M06 ckpts.
_JIT_NUM_CLASSES: int = 1000


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for JiT-L/16 (Pixel) MMSE canny-aware evaluation."""
    p = argparse.ArgumentParser(
        description="JiT-L/16 (Pixel) MMSE eval per t — M06 canny conditioning"
    )
    p.add_argument("--ckpt_path", required=True,
                   help="Path to JiT_M06 checkpoint (.pt). Keys: 'ema' > 'model' > flat.")
    p.add_argument(
        "--checkpoint_state",
        choices=("auto", "ema", "model"),
        default="auto",
        help="Checkpoint state to evaluate. 'auto' preserves ema > model priority.",
    )
    p.add_argument("--config_yaml", default=None,
                   help="Optional training YAML (used for logging/meta only).")
    p.add_argument("--paired_val_whitelist", required=True,
                   help="Path to paired-val filename whitelist JSON {filenames:[...]}.")
    p.add_argument("--mask_shard_dir", required=True,
                   help="Directory containing masks_rank*_shard*.safetensors shards.")
    p.add_argument("--canny_shard_dir", required=True,
                   help="Directory containing cannys_rank*_shard*.{safetensors,json} shards.")
    p.add_argument("--imagenet_root", required=True)
    p.add_argument("--jit-source", default=None, help="Path to the supplied JiT source checkout.")
    p.add_argument("--n_samples", type=int, default=512,
                   help="Number of val samples to evaluate.")
    p.add_argument("--t_grid", type=int, default=21,
                   help="Number of t points in [0, 1] (inclusive endpoints).")
    p.add_argument("--t_list_json", default=None,
                   help="JSON-encoded list of explicit t values (overrides --t_grid linspace).")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Forward batch size (image-domain: 3×256×256, memory heavy).")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--save_per_sample", action="store_true",
        help=("Save an [N,T] float32 squared-error matrix for paired factorial "
              "bootstrap acceptance; requires equal DDP shard sizes."),
    )
    # Modes 5-8 complete the class/mask/canny 2^3 factorial.
    p.add_argument("--modes", type=str, default="1,2,3,4",
                   help=("Comma-separated modes: 1=uncond 2=class 3=class+mask "
                         "4=class+mask+canny 5=mask-only 6=canny-only "
                         "7=class+canny 8=mask+canny"))
    p.add_argument("--out_dir", required=True,
                   help="Output directory for mmse_curve_mode{m}.{json,png,npz}.")
    return p.parse_args()


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_state_dict_from_ckpt(
    ckpt_path: str,
    checkpoint_state: str = "auto",
) -> Dict[str, torch.Tensor]:
    """Load and unwrap ckpt: priority 'ema' > 'model' > flat; strips 'module.' DDP prefix."""
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        if checkpoint_state != "auto":
            if checkpoint_state not in raw or not isinstance(raw[checkpoint_state], dict):
                raise KeyError(
                    f"Checkpoint has no dict state {checkpoint_state!r}: {ckpt_path}"
                )
            src = raw[checkpoint_state]
        else:
            for key in ("ema", "model"):
                if key in raw and isinstance(raw[key], dict):
                    src = raw[key]
                    break
            else:
                src = raw  # flat dict
    else:
        raise ValueError(f"Unexpected ckpt type: {type(raw)}")

    return {
        k.removeprefix("module."): v
        for k, v in src.items()
        if isinstance(v, torch.Tensor)
    }


def build_jit_model(
    ckpt_path: str,
    device: torch.device,
    checkpoint_state: str = "auto",
    model_name: str = "JiT-L/16",
) -> JiT_M06:
    """Build JiT_M06 (num_classes=1000) and load ckpt; strict=False for pre-canny ckpts.

    num_classes=_JIT_NUM_CLASSES=1000 → LabelEmbedder table = 1001 slots
    (null at index NULL_CLASS=1000). Do NOT add +1 externally — JiT_M06 internal.
    Allows canny_proj.* missing (warmup/mask-only ckpts) → stays at zero-init.
    """
    from feature_information_dynamics.pixel.model import JiT_M06_FiLM as JiT_M06

    net = JiT_M06(
        input_size=_IMG_SIZE,
        num_classes=_JIT_NUM_CLASSES,
        model_name=model_name,
    )
    sd = _load_state_dict_from_ckpt(ckpt_path, checkpoint_state)
    # Allow canny_proj.* missing for pre-canny-phase ckpts (mirrors SDVAE/VAVAE eval).
    _allowed_missing: set[str] = {"canny_proj.weight", "canny_proj.bias"}
    result = net.load_state_dict(sd, strict=False)
    _bad_missing = [k for k in result.missing_keys if k not in _allowed_missing]
    if _bad_missing:
        log.warning(
            "JiT_M06 ckpt unexpected missing keys (first 5): %s",
            _bad_missing[:5],
        )
    log.info(
        "JiT_M06 ckpt loaded: missing=%d unexpected=%d bad_missing=%d num_classes=%d",
        len(result.missing_keys), len(result.unexpected_keys),
        len(_bad_missing), _JIT_NUM_CLASSES,
    )
    return net.to(device).eval()


# ── Dataset ───────────────────────────────────────────────────────────────────

# EDIT C: RawImageMaskCannyDataset is the Pixel-domain analogue of
# MaskedCannyShardedDataset(phase="canny_mask"); it always returns the 4-tuple
# (image, mask, canny, label) so both mask AND canny are available for mode 4.
def build_val_dataset(args: argparse.Namespace) -> RawImageMaskCannyDataset:
    """Build val-split RawImageMaskCannyDataset (pixel-domain analogue of
    MaskedCannyShardedDataset(phase="canny_mask", is_val=True)).
    """
    return RawImageMaskCannyDataset(
        imagenet_root=args.imagenet_root,
        mask_shard_dir=args.mask_shard_dir,
        canny_shard_dir=args.canny_shard_dir,
        paired_whitelist_json=args.paired_val_whitelist,
        image_size=_IMG_SIZE,
        is_val=True,
    )


def sample_subset_indices(
    ds: RawImageMaskCannyDataset,
    n_samples: int,
    seed: int,
) -> List[int]:
    """Sample a random subset of dataset indices (n capped at len(ds))."""
    rng = np.random.default_rng(seed)
    total = len(ds)
    n = min(n_samples, total)
    return rng.choice(total, n, replace=False).tolist()


def shard_indices_for_rank(
    indices: List[int],
    rank: int,
    world_size: int,
) -> List[int]:
    """Split a global ordered subset into world_size contiguous shards.

    Uses ``np.array_split`` semantics: the first ``N % world_size`` shards get
    one extra element. Rank receives shard ``rank``. Determinism: same global
    indices + same world_size → same partition.
    """
    if world_size <= 1:
        return list(indices)
    parts = np.array_split(np.asarray(indices), world_size)
    return parts[rank].tolist()


# ── EDIT A: canny-aware adapter — canny_per_token replaces patch_per_token ────

class _JiT_M06_CannyAdapter(torch.nn.Module):
    """Adapter wrapping JiT_M06 for canny-aware 5-arg eval interface.

    Eval interface: forward(x, t, y, mask_per_token, canny_per_token) →
    JiT_M06.forward(x, t, y, mask=mask_per_token, canny=canny_per_token).
    Both mask and canny are [B, 256, 256] per-token format, or None.

    NOTE: Requires JiT_M06 to accept ``canny=`` kwarg. Current train script
    only accepts ``mask=``; mode 4 will TypeError until JiT_M06 is extended
    with a canny_proj Conv2d adapter (mirror SiT_M06 / LightningDiT_M06 /
    RAE_M06). Modes 1-3 (canny=None) work against the current code unchanged.
    """

    def __init__(self, inner: JiT_M06) -> None:
        """Wrap a JiT_M06 module."""
        super().__init__()
        self.inner = inner

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        mask_per_token: Optional[torch.Tensor] = None,
        canny_per_token: Optional[torch.Tensor] = None,  # EDIT A: was patch_per_token
    ) -> torch.Tensor:
        """Delegate to JiT_M06 passing mask and canny as kwargs.

        Mirrors SDVAE/VAVAE eval pattern: mask_per_token / canny_per_token are
        forwarded into JiT_M06's mask= / canny= kwargs. Both shapes [B,256,256]
        or None. Returns [B, 3, 256, 256] x_pred (caller derives velocity).
        """
        return self.inner(x, t, y, mask=mask_per_token, canny=canny_per_token)


# ── EDIT B: mode 4 = class+mask+canny (replaces class+mask+patch) ─────────────

def make_mode_kwargs_fn(
    labels: torch.Tensor,
    mask_pt: torch.Tensor,
    canny_pt: torch.Tensor,
    device: torch.device,
) -> Callable[[int], Dict]:
    """Closure: mode_idx -> forward kwargs for _JiT_M06_CannyAdapter.

    Modes:
        1 = uncond           (null class, zero mask, zero canny)
        2 = class            (real class, zero mask, zero canny)
        3 = class+mask       (real class, real mask, zero canny)
        4 = class+mask+canny (real class, real mask, real canny)
        5 = mask-only       (null class, real mask, zero canny)
        6 = canny-only      (null class, zero mask, real canny)
        7 = class+canny     (real class, zero mask, real canny)
        8 = mask+canny      (null class, real mask, real canny)

    NULL class = NULL_CLASS = 1000 (matches num_classes=1000; LabelEmbedder
    table size = 1001 covers the null slot).
    """
    N = labels.shape[0]
    null_y = torch.full((N,), NULL_CLASS, dtype=torch.long, device=device)
    zero_mask = torch.zeros(N, NUM_TOKENS, NUM_TOKENS, device=device)    # [N, 256, 256]
    zero_canny = torch.zeros(N, NUM_TOKENS, NUM_TOKENS, device=device)   # [N, 256, 256]

    _CFG: Dict[int, Tuple] = {
        1: (null_y, zero_mask, zero_canny),
        2: (labels, zero_mask, zero_canny),
        3: (labels, mask_pt,   zero_canny),
        4: (labels, mask_pt,   canny_pt),    # EDIT B: was patch_pt
        5: (null_y, mask_pt,   zero_canny),
        6: (null_y, zero_mask, canny_pt),
        7: (labels, zero_mask, canny_pt),
        8: (null_y, mask_pt,   canny_pt),
    }

    def _fn(m: int) -> Dict:
        y, mp, cp = _CFG[m]
        return {"y": y, "mask_per_token": mp, "canny_per_token": cp}

    return _fn


# ── x0_pred from JiT output ───────────────────────────────────────────────────

def _x0_pred_from_jit(
    x_pred: torch.Tensor,
    x_t: torch.Tensor,
    t_scalar: float,
    t_eps: float,
) -> torch.Tensor:
    """Convert JiT x_pred [B,3,H,W] to clean-image estimate via velocity.

    Matches train_m06_upstream_pixel_film_advanced._compute_velocity_loss:
      v_pred  = (x_pred - x_t) / max(1 - t, t_eps)
      x0_pred = x_t + (1-t) * v_pred
    For t <= 1-t_eps this is exactly x_pred; only the final high-SNR tail is
    stabilized by the denominator floor used during training.
    """
    one_minus_t = 1.0 - t_scalar
    denom = max(one_minus_t, t_eps)
    v_pred = (x_pred - x_t) / denom                        # [B, 3, H, W]
    x0_pred = x_t + one_minus_t * v_pred                   # [B, 3, H, W]
    return x0_pred


# ── MMSE accumulator ──────────────────────────────────────────────────────────

@torch.no_grad()
def compute_mmse_curve(
    adapter: _JiT_M06_CannyAdapter,
    loader: DataLoader,
    t_grid: np.ndarray,
    mode: int,
    device: torch.device,
    t_eps: float = _T_EPS,
    seed: int = 42,
    accelerator: Optional[Accelerator] = None,
    return_per_sample: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Accumulate MMSE_x1 over val samples for each t in t_grid (mode-specific kwargs).

    DDP shard semantics: ``loader`` is expected to yield ONLY this rank's shard
    (a contiguous slice of the global subset, see :func:`shard_indices_for_rank`).
    Per-rank local sums + counts are all-reduced via ``accelerator.reduce`` to
    produce the global mean per t. ``accelerator=None`` falls back to single-GPU
    behaviour (no reduction).
    """
    is_main = (accelerator is None) or accelerator.is_main_process
    world_size = 1 if accelerator is None else accelerator.num_processes
    rank = 0 if accelerator is None else accelerator.process_index

    # Per-rank seed offset keeps eps deterministic across ranks (different shards
    # → different noise) while remaining reproducible at fixed seed+world_size.
    rng = torch.Generator(device=device)
    rng.manual_seed(seed + rank * 100003)

    T = len(t_grid)

    # Collect this rank's shard into device memory
    images_list: List[torch.Tensor] = []
    masks_list: List[torch.Tensor] = []
    cannys_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []

    if is_main:
        log.info("Loading val samples into device memory (mode=%d)...", mode)
    for batch in loader:
        image, mask_2d, canny_2d, label = batch
        images_list.append(image.to(device).float())          # [b, 3, 256, 256]
        masks_list.append(mask_2d.to(device).float())         # [b, 256, 256]
        cannys_list.append(canny_2d.to(device).float())       # [b, 256, 256]
        if isinstance(label, torch.Tensor):
            labels_list.append(label.to(device))
        else:
            labels_list.append(torch.tensor(label, device=device))

    if not images_list:
        # Empty shard (world_size > N_subset edge case); fill zeros for all-reduce.
        local_n = 0
        local_sum = torch.zeros(T, dtype=torch.float64, device=device)
        local_count = torch.zeros(1, dtype=torch.float64, device=device)
        local_per_sample = torch.empty((0, T), dtype=torch.float32, device=device)
    else:
        rgb_all = torch.cat(images_list, dim=0)          # [n_local, 3, 256, 256]
        mask_2d_all = torch.cat(masks_list, dim=0)       # [n_local, 256, 256]
        canny_2d_all = torch.cat(cannys_list, dim=0)     # [n_local, 256, 256]
        labels_all = torch.cat(labels_list, dim=0)       # [n_local]
        local_n = rgb_all.shape[0]

        # Per-token format expected by _JiT_M06_CannyAdapter (mirrors SDVAE/VAVAE eval)
        # Pass raw 2D [n_local, 256, 256] to JiT_M06.mask_proj Conv2d (k=16, s=16);
        # build_mask_per_token would PERMUTE pixels destroying spatial layout the
        # Conv2d expects (training feeds raw 2D — see train_m06_upstream_pixel.py:382).
        mask_pt_all = mask_2d_all                                   # [n_local, 256, 256]
        canny_pt_all = canny_2d_all                                 # [n_local, 256, 256]

        if is_main:
            log.info(
                "Rank %d/%d loaded %d local samples; t-sweep over %d points (mode=%d)...",
                rank, world_size, local_n, T, mode,
            )

        # Pre-generate fixed noise for reproducibility across t values (per shard)
        eps_all = torch.randn(local_n, 3, _IMG_SIZE, _IMG_SIZE, device=device,
                              generator=rng)              # [n_local, 3, 256, 256]

        # Build mode-specific kwargs closure once; subset per micro-batch via slice
        mode_kwargs_fn = make_mode_kwargs_fn(labels_all, mask_pt_all, canny_pt_all, device)
        full_kwargs = mode_kwargs_fn(mode)
        y_all = full_kwargs["y"]                          # [n_local]
        mp_all = full_kwargs["mask_per_token"]            # [n_local, 256, 256]
        cp_all = full_kwargs["canny_per_token"]           # [n_local, 256, 256]

        local_sum = torch.zeros(T, dtype=torch.float64, device=device)
        local_count = torch.tensor([float(local_n)], dtype=torch.float64, device=device)
        local_per_sample = (
            torch.empty((local_n, T), dtype=torch.float32, device=device)
            if return_per_sample else None
        )

        for t_idx, t_val in enumerate(t_grid):
            t_f = float(t_val)
            mse_accum = 0.0

            batch_size = min(32, local_n)

            for s in range(0, local_n, batch_size):
                e = min(s + batch_size, local_n)
                rgb_b = rgb_all[s:e]                               # [b, 3, H, W]
                eps_b = eps_all[s:e]                               # [b, 3, H, W]
                y_b = y_all[s:e]                                   # [b]
                mp_b = mp_all[s:e]                                 # [b, 256, 256]
                cp_b = cp_all[s:e]                                 # [b, 256, 256]
                b = rgb_b.shape[0]

                # Linear interpolation: x_t = t*x1 + (1-t)*eps
                t4 = torch.full((b, 1, 1, 1), t_f, device=device, dtype=rgb_b.dtype)
                x_t = t4 * rgb_b + (1.0 - t4) * eps_b              # [b, 3, H, W]
                t_tensor = torch.full((b,), t_f, device=device, dtype=rgb_b.dtype)

                # Forward via canny-aware adapter (mode dispatches mask/canny to None or real)
                x_pred = adapter(x_t, t_tensor, y_b,
                                 mask_per_token=mp_b,
                                 canny_per_token=cp_b)              # [b, 3, H, W]

                x0_pred = _x0_pred_from_jit(x_pred, x_t, t_f, t_eps)  # [b, 3, H, W]

                per_sample_mse = ((rgb_b - x0_pred) ** 2).sum(dim=(1, 2, 3))  # [b]
                mse_accum += per_sample_mse.sum().item()
                if local_per_sample is not None:
                    local_per_sample[s:e, t_idx] = per_sample_mse.float()

            local_sum[t_idx] = mse_accum
            if is_main and ((t_idx + 1) % 5 == 0 or t_idx == 0 or t_idx == T - 1):
                # Note: rank-0-only; partial mean (this rank's shard) for progress tracking
                log.info(
                    "  mode=%d t[%d/%d]=%.3f  rank0_local_mean=%.4f",
                    mode, t_idx + 1, T, t_f,
                    mse_accum / max(local_n, 1),
                )

    # All-reduce across ranks (sum). Single-GPU path: accelerator=None → no-op.
    if accelerator is not None and world_size > 1:
        global_sum = accelerator.reduce(local_sum, reduction="sum")     # [T]
        global_count = accelerator.reduce(local_count, reduction="sum")  # [1]
    else:
        global_sum = local_sum
        global_count = local_count

    global_n = float(global_count.item())
    if global_n <= 0:
        raise RuntimeError("compute_mmse_curve: global sample count is 0")
    mmse_curve = global_sum.cpu().numpy() / global_n
    if not return_per_sample:
        return mmse_curve
    if accelerator is not None and world_size > 1:
        shard_sizes = accelerator.gather(
            torch.tensor([local_n], dtype=torch.int64, device=device)
        )
        if not torch.all(shard_sizes == shard_sizes[0]):
            raise RuntimeError(
                "--save_per_sample requires n_samples divisible by world_size"
            )
        global_per_sample = accelerator.gather(local_per_sample)
    else:
        global_per_sample = local_per_sample
    if global_per_sample.shape != (int(global_n), T):
        raise RuntimeError(
            f"per-sample gather shape mismatch: {tuple(global_per_sample.shape)} "
            f"!= {(int(global_n), T)}"
        )
    return mmse_curve, global_per_sample.cpu().numpy()


# ── Output helpers ────────────────────────────────────────────────────────────

_MODE_LABELS: Dict[int, str] = {
    1: "uncond",
    2: "class",
    3: "class+mask",
    4: "class+mask+canny",
    5: "mask-only",
    6: "canny-only",
    7: "class+canny",
    8: "mask+canny",
}


def save_mode_outputs(
    out_dir: Path,
    mode: int,
    t_grid: np.ndarray,
    mmse_curve: np.ndarray,
    meta: Dict,
) -> None:
    """Save per-mode MMSE curve as JSON/PNG/NPZ → mmse_curve_mode{m}.{json,png,npz}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    label = _MODE_LABELS.get(mode, f"mode{mode}")

    # JSON
    result = {
        "t_grid": t_grid.tolist(),
        "mmse_curve": mmse_curve.tolist(),
        "mode": mode,
        "mode_label": label,
        "meta": meta,
    }
    json_path = out_dir / f"mmse_curve_mode{mode}.json"
    with open(json_path, "w") as fh:
        json.dump(result, fh, indent=2)
    log.info("Saved JSON: %s", json_path)

    # NPZ — meta may already contain 'mode'; drop to avoid kwarg collision with explicit mode arg.
    npz_path = out_dir / f"mmse_curve_mode{mode}.npz"
    npz_meta = {k: v for k, v in meta.items() if k not in {"mode", "t_grid", "mmse_curve"}}
    np.savez(npz_path, t_grid=t_grid, mmse_curve=mmse_curve, mode=mode, **npz_meta)
    log.info("Saved NPZ: %s", npz_path)

    # PNG
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_grid, mmse_curve, linewidth=1.5, label=f"mode={mode} ({label})")
    ax.set_xlabel("t (t=1=clean, t=0=noise)")
    ax.set_ylabel("Mean MMSE_x1 (raw spatial sum)")
    ax.set_title(f"JiT-L/16 (Pixel) MMSE — mode {mode}: {label}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    png_path = out_dir / f"mmse_curve_mode{mode}.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    log.info("Saved PNG: %s", png_path)


def save_combined_outputs(
    out_dir: Path,
    t_grid: np.ndarray,
    curves: Dict[int, np.ndarray],
    meta: Dict,
) -> None:
    """Save multi-mode summary JSON + combined PNG across all active modes."""
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "t_grid": t_grid.tolist(),
        "modes": sorted(curves.keys()),
        "mode_labels": {str(m): _MODE_LABELS[m] for m in curves},
        "curves": {str(m): c.tolist() for m, c in curves.items()},
        "meta": meta,
    }
    summary_path = out_dir / "mmse_curves_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info("Saved summary JSON: %s", summary_path)

    fig, ax = plt.subplots(figsize=(7, 4))
    for m in sorted(curves.keys()):
        ax.plot(t_grid, curves[m], linewidth=1.5,
                label=f"mode={m} ({_MODE_LABELS[m]})")
    ax.set_xlabel("t (t=1=clean, t=0=noise)")
    ax.set_ylabel("Mean MMSE_x1 (raw spatial sum)")
    ax.set_title("JiT-L/16 (Pixel) MMSE — canny-aware modes")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    png_path = out_dir / "mmse_curves_combined.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    log.info("Saved combined PNG: %s", png_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args=None) -> None:
    """Pixel/JiT MMSE canny-aware evaluation entry point (DDP-aware via accelerate).

    Each rank loads the same ckpt + a contiguous shard of the val subset.
    Per-rank local sums + counts are all-reduced; rank 0 writes outputs.
    With ``accelerate launch --num_processes=1`` (or plain python) the path
    is functionally equivalent to the prior single-GPU implementation.
    """
    if args is None:
        args = parse_args()
    source = getattr(args, "jit_source", None)
    if source is None and args.config_yaml:
        with open(args.config_yaml) as handle:
            source = (yaml.safe_load(handle) or {}).get("resources", {}).get("jit_source")
    if source:
        sys.path.insert(0, str(Path(source).expanduser().resolve()))
    try:
        active_modes = sorted({int(m.strip()) for m in args.modes.split(",")})
    except ValueError as exc:
        raise ValueError(f"--modes must be comma-separated ints: {args.modes!r}") from exc
    if bad := [m for m in active_modes if m not in (1, 2, 3, 4, 5, 6, 7, 8)]:
        raise ValueError(f"Invalid mode(s) {bad}; valid range 1-8")

    # Resolve the effective grid before logging it.  Previously the startup
    # message always printed ``args.t_grid`` (default 21), even when an
    # explicit 49-point ``--t_list_json`` correctly overrode it later.
    if args.t_list_json:
        import json as _json
        t_grid = np.array(_json.loads(args.t_list_json), dtype=np.float64)
    else:
        t_grid = np.linspace(0.0, 1.0, args.t_grid, dtype=np.float64)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    accelerator = Accelerator()
    device = accelerator.device
    is_main = accelerator.is_main_process
    world_size = accelerator.num_processes
    rank = accelerator.process_index

    # Quiet non-main ranks (warning level only) — rank 0 keeps INFO.
    if not is_main:
        logging.getLogger().setLevel(logging.WARNING)

    if is_main:
        log.info(
            "Accelerator: world_size=%d  rank=%d  device=%s  Modes=%s  "
            "n_samples=%d  t_grid=%d",
            world_size, rank, device, active_modes, args.n_samples, len(t_grid),
        )

    # Load config for meta logging (all ranks read; only rank 0 writes meta).
    config_meta: Dict = {}
    if args.config_yaml:
        with open(args.config_yaml) as fh:
            config_meta = yaml.safe_load(fh) or {}

    # Build model — num_classes=1000 (NOT 1001); JiT_M06 internal +1 covers null
    if is_main:
        log.info("Loading JiT_M06 from: %s (num_classes=%d)",
                 args.ckpt_path, _JIT_NUM_CLASSES)
    model_name = config_meta.get("model", {}).get("name", "JiT-L/16")
    net = build_jit_model(
        args.ckpt_path,
        device,
        args.checkpoint_state,
        model_name=model_name,
    )
    adapter = _JiT_M06_CannyAdapter(net).to(device).eval()

    # Build dataset + global subset (deterministic via seed; identical across ranks)
    if is_main:
        log.info("Building val dataset (paired_val_whitelist=%s)...",
                 args.paired_val_whitelist)
    ds = build_val_dataset(args)
    global_indices = sample_subset_indices(ds, args.n_samples, args.seed)
    if is_main:
        log.info("Global val subset: %d / %d samples", len(global_indices), len(ds))

    # Shard the global ordered subset across ranks (np.array_split semantics).
    rank_indices = shard_indices_for_rank(global_indices, rank, world_size)
    log_fn = log.info if is_main else log.debug
    log_fn("Rank %d/%d shard: %d samples (global %d)",
           rank, world_size, len(rank_indices), len(global_indices))

    loader = DataLoader(
        Subset(ds, rank_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Shared meta block (mirrors SDVAE/VAVAE canny eval structure)
    shared_meta: Dict = {
        "ckpt_path": args.ckpt_path,
        "checkpoint_state": args.checkpoint_state,
        "n_samples": len(global_indices),
        "t_grid_n": int(len(t_grid)),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "imagenet_root": args.imagenet_root,
        "paired_val_whitelist": args.paired_val_whitelist,
        "mask_shard_dir": args.mask_shard_dir,
        "canny_shard_dir": args.canny_shard_dir,
        "t_eps": _T_EPS,
        "model": f"JiT_M06 ({model_name})",
        "rep": "pixel",
        "num_classes": _JIT_NUM_CLASSES,
        "null_class": NULL_CLASS,
        "active_modes": active_modes,
        "modes_description": {str(m): _MODE_LABELS[m] for m in (1, 2, 3, 4, 5)},
        "mmse_formula": "MMSE_x1 = ||x1 - (x_t + (1-t)*v_pred)||^2 (raw spatial sum)",
        "v_formula": "v_pred = (x_pred - x_t) / max(1 - t, t_eps)",
        "normalization": "rgb in [-1,1] (mean=std=0.5); raw spatial sum, no /D",
        "t_inversion_applied": False,
        "sign_flip_applied": False,
        "channel_slice_applied": False,
        "phase": "canny_mask",  # semantic equivalent: 4-tuple dataset, mode-driven kwargs
        "dataset_class": "RawImageMaskCannyDataset",
        "config": config_meta,
        "ddp": {
            "world_size": world_size,
            "backend": "accelerate",
            "shard_strategy": "contiguous_array_split",
        },
    }

    # Compute MMSE curve per active mode (DDP all-reduce inside).
    curves: Dict[int, np.ndarray] = {}
    for mode in active_modes:
        if is_main:
            log.info("=== Computing mode %d (%s) ===", mode, _MODE_LABELS[mode])
        curve_result = compute_mmse_curve(
            adapter=adapter,
            loader=loader,
            t_grid=t_grid,
            mode=mode,
            device=device,
            t_eps=_T_EPS,
            seed=args.seed,
            accelerator=accelerator,
            return_per_sample=args.save_per_sample,
        )
        if args.save_per_sample:
            mmse_curve, per_sample_mse = curve_result
        else:
            mmse_curve = curve_result
        if is_main:
            log.info(
                "mode=%d  MMSE summary: min=%.4f @ t=%.3f  max=%.4f @ t=%.3f",
                mode,
                mmse_curve.min(), t_grid[mmse_curve.argmin()],
                mmse_curve.max(), t_grid[mmse_curve.argmax()],
            )
        curves[mode] = mmse_curve

        # Output write gated on rank 0 only.
        if is_main:
            per_mode_meta = dict(shared_meta)
            per_mode_meta["mode"] = mode
            per_mode_meta["mode_label"] = _MODE_LABELS[mode]
            save_mode_outputs(
                Path(args.out_dir), mode, t_grid, mmse_curve, per_mode_meta,
            )
            if args.save_per_sample:
                per_sample_path = Path(args.out_dir) / f"per_sample_mse_mode{mode}.npy"
                np.save(per_sample_path, per_sample_mse)
                log.info(
                    "Saved paired-bootstrap matrix: %s shape=%s",
                    per_sample_path, per_sample_mse.shape,
                )

    # Combined plot/summary across all active modes (rank 0 only).
    if is_main:
        save_combined_outputs(Path(args.out_dir), t_grid, curves, shared_meta)
        log.info("Done. Outputs written to: %s", args.out_dir)

    # Sync ranks before exit so non-main ranks don't tear down prematurely.
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
