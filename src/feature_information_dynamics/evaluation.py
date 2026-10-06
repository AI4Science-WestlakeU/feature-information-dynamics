"""Model-agnostic MMSE evaluation library for flow-matching diffusion models.

Provides pure-function building blocks for computing MMSE curves under
v-prediction flow-matching (RFC: MMSE_x1 = (1-t)^2 * MMSE_v).

This library contains all bug fixes from M05 reviewer R3 PASS:
  - C1: DDP all_reduce SUM+SUM (not mean-of-means) for unbiased global MMSE.
  - C2: t_eps clamp default 5e-2 matching training, preventing numerical
        blow-up near t=1.
  - W1: Per-rank distinct noise seeds (not broadcast) for disjoint shards.

Designed to be imported by thin model-specific wrappers (JiT, RAE, etc.).
No CLI, no argparse, no torch.distributed init, no model imports.

Example:
    >>> import torch
    >>> from feature_information_dynamics import evaluation as eval_lib_mmse
    >>> t = eval_lib_mmse.make_t_grid(0.01, 0.99, 4)
    >>> t.shape
    torch.Size([4])
    >>> t.dtype
    torch.float64
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default lower-bound clamp on (1-t) to prevent numerical blow-up near t=1.
#: Must match the t_eps used during model training (C2 fix).
_T_EPS_DEFAULT: float = 5e-2

#: Canonical output keys for conditioning modes.  The fifth key is used by
#: dedicated mask-only branches (null class + real mask).
MODE_KEYS: Tuple[str, ...] = (
    "mmse_uncond",
    "mmse_class_cond",
    "mmse_class_mask_cond",
    "mmse_class_mask_patch_cond",
    "mmse_mask_only_cond",
    "mmse_canny_only_cond",
    "mmse_class_canny_cond",
    "mmse_mask_canny_cond",
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# t-grid
# ---------------------------------------------------------------------------

def make_t_grid(
    t_min: float,
    t_max: float,
    t_points: int,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return a uniform t-grid as a 1-D tensor.

    Args:
        t_min: Minimum t value (exclusive boundary, e.g. 0.01).
        t_max: Maximum t value (exclusive boundary, e.g. 0.99).
        t_points: Number of grid points.
        dtype: Output tensor dtype (default float64 for numerical stability).

    Returns:
        t_grid: shape [t_points], values linearly spaced in [t_min, t_max].

    Example:
        >>> t = make_t_grid(0.01, 0.99, 5)
        >>> t.shape
        torch.Size([5])
        >>> float(t[0]), float(t[-1])
        (0.01, 0.99)
    """
    return torch.linspace(t_min, t_max, t_points, dtype=dtype)  # [t_points]


# ---------------------------------------------------------------------------
# Per-rank noise seeding (W1 fix)
# ---------------------------------------------------------------------------

def per_rank_noise_seed(
    base_seed: int,
    rank: int,
    t_idx: int = 0,
    sample_idx: int = 0,
) -> int:
    """Compute a deterministic per-rank distinct noise seed (W1 fix).

    Ensures each DDP rank uses independent noise rather than a broadcasted
    noise tensor, which would produce biased estimates when shard sizes differ.

    Args:
        base_seed: Global experiment seed.
        rank: DDP rank index.
        t_idx: t-grid index (optional, for finer per-t differentiation).
        sample_idx: Sample index within the rank shard (optional).

    Returns:
        An integer seed unique to this (base_seed, rank, t_idx, sample_idx).

    Example:
        >>> per_rank_noise_seed(42, 0) != per_rank_noise_seed(42, 1)
        True
        >>> per_rank_noise_seed(42, 0) == per_rank_noise_seed(42, 0)
        True
    """
    # Mix via polynomial hash; keep within int64 range (63-bit mask avoids sign bit).
    return (base_seed * 2654435761 + rank * 40503 + t_idx * 12347 + sample_idx) & ((1 << 63) - 1)


# ---------------------------------------------------------------------------
# Single-t MMSE compute
# ---------------------------------------------------------------------------

def xpred_to_v(
    x_pred: torch.Tensor,
    x_t: torch.Tensor,
    t: float,
    t_eps: float = _T_EPS_DEFAULT,
) -> torch.Tensor:
    """Convert x-prediction to velocity: v = (x_pred - x_t) / max(1-t, t_eps).

    Use this to wrap x-prediction models (JiT, RAE) so they conform to the
    velocity contract required by compute_mmse_at_t.

    Args:
        x_pred: Model output [B, ...] interpreted as predicted clean x1.
        x_t: Noisy input z_t = (1-t)*x0 + t*x1, shape [B, ...].
        t: Diffusion time scalar.
        t_eps: Floor for denominator (1-t) to avoid blowup at t->1. Default 5e-2.

    Returns:
        v_pred: Velocity prediction [B, ...] same shape as x_pred.
    """
    denom = max(1.0 - t, t_eps)
    return (x_pred - x_t) / denom  # [B, ...]


@torch.no_grad()
def compute_mmse_at_t(
    model_forward_fn: Callable[..., torch.Tensor],
    x_gt: torch.Tensor,
    t: float,
    noise: torch.Tensor,
    *,
    output_kind: Literal["v", "x"] = "v",
    t_eps: float = _T_EPS_DEFAULT,
    forward_kwargs: Optional[Dict[str, object]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute (sum_squared_errors, count) for MMSE_v at one t (C1 fix: no division here).

    The caller is responsible for all_reduce before dividing.

    MMSE transform (v-prediction -> x1-space):
        z_t = (1-t)*x0 + t*x1         # noisy image
        v_true = x1 - x0               # ground truth velocity
        MMSE_v  = mean((v_true - v_pred)^2)
        MMSE_x1 = denom^2 * MMSE_v    where denom = max(1-t, t_eps)

    Args:
        model_forward_fn: Callable (x_t, t, **forward_kwargs) -> Tensor [B, ...].
        x_gt: Clean target [B, ...] = x1 (per project t=1=clean convention).
        t: Diffusion time scalar in (0, 1).
        noise: Pure noise [B, ...] = x0; must be disjoint across ranks.
        output_kind: "v" if model_forward_fn returns velocity directly (default).
            "x" if model_forward_fn returns x-prediction; converted via xpred_to_v.
        t_eps: Floor for (1-t) when converting x-pred to v-pred (C2 fix).
        forward_kwargs: Extra kwargs forwarded to model_forward_fn (e.g. {"y": class_idx}).

    Returns:
        (sum_squared_errors, count): both scalar float32 Tensors suitable for
        downstream all_reduce_mmse(sum, count).

    Example:
        >>> # Distributed pattern (recommended):
        >>> # local_sum, local_count = compute_mmse_at_t(net.forward, x_gt, t=0.3, noise=x0, output_kind="x")
        >>> # mmse_v = all_reduce_mmse(local_sum, local_count)  # only one device per rank
        >>> # mmse_x1 = max(1-t, 5e-2)**2 * mmse_v  # convert MMSE_v -> MMSE_x1
    """
    fwd_kw = forward_kwargs or {}
    x1 = x_gt
    x0 = noise
    B = x1.shape[0]

    x_t = (1.0 - t) * x0 + t * x1                                 # [B, C, H, W] noisy interpolation
    v_target = x1 - x0                                             # [B, C, H, W] ground truth velocity

    pred = model_forward_fn(x_t, t, **fwd_kw)                      # [B, C, H, W]
    if output_kind == "x":
        v_pred = xpred_to_v(pred, x_t, t, t_eps=t_eps)            # [B, C, H, W]
    else:
        v_pred = pred                                               # [B, C, H, W]

    sq_err = ((v_target - v_pred) ** 2).sum(dim=(1, 2, 3)).float() # [B]
    mse_sum = sq_err.sum()                                          # scalar float32
    n_count = torch.tensor(B, dtype=torch.float32, device=x1.device)  # scalar float32
    return mse_sum, n_count


@torch.no_grad()
def compute_sample_errors_at_t(
    model_forward_fn: Callable[..., torch.Tensor],
    x_gt: torch.Tensor,
    t: float,
    noise: torch.Tensor,
    *,
    output_kind: Literal["v", "x"] = "v",
    t_eps: float = _T_EPS_DEFAULT,
    forward_kwargs: Optional[Dict[str, object]] = None,
) -> torch.Tensor:
    """Return per-sample MMSE_x1 squared errors at one t.

    This is the sample-level analogue of compute_mmse_at_t. It intentionally
    performs the same forward and unit conversion, but returns a vector [B]
    after summing over feature dimensions and applying the x1-space factor.
    """
    fwd_kw = forward_kwargs or {}
    x1 = x_gt
    x0 = noise

    x_t = (1.0 - t) * x0 + t * x1
    v_target = x1 - x0

    pred = model_forward_fn(x_t, t, **fwd_kw)
    if output_kind == "x":
        v_pred = xpred_to_v(pred, x_t, t, t_eps=t_eps)
    else:
        v_pred = pred

    denom = max(1.0 - t, t_eps)
    return ((v_target - v_pred) ** 2).sum(dim=(1, 2, 3)).float() * (denom ** 2)


# ---------------------------------------------------------------------------
# DDP all_reduce (C1 fix)
# ---------------------------------------------------------------------------

def all_reduce_mmse(
    local_sum: torch.Tensor,
    local_count: torch.Tensor,
) -> float:
    """All-reduce SUM+SUM across ranks, then divide (C1 fix).

    Correct implementation avoids the biased mean-of-means bug that arises
    when ranks have unequal shard sizes and we naive-average per-rank means.

    Args:
        local_sum: Scalar float32 tensor — rank-local sum of squared errors.
        local_count: Scalar float32 tensor — rank-local sample count.

    Returns:
        Global unweighted MMSE (float). Returns NaN if global count is 0.

    Example:
        >>> import torch
        >>> s = torch.tensor(4.0)
        >>> c = torch.tensor(2.0)
        >>> # In single-process mode (no dist), call all_reduce manually:
        >>> # result = all_reduce_mmse(s, c)  # == 2.0
    """
    dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
    if local_count.item() == 0:
        return float("nan")
    return (local_sum / local_count).item()


# ---------------------------------------------------------------------------
# JiT-weighted loss
# ---------------------------------------------------------------------------

def jit_weighted_loss(
    t_grid: torch.Tensor,
    mmse_curve: np.ndarray,
    p_mean: float = -0.8,
    p_std: float = 0.8,
) -> float:
    """Compute JiT-Karras weighted MMSE: E_t[w(t) * MMSE_x1(t)].

    Weight w(t) is log-normal in sigma = t/(1-t):
        w(t) ∝ exp(-0.5 * ((log(sigma) - p_mean) / p_std)^2)

    Discretized sum over t_grid (uniform spacing, trapezoidal-equivalent for
    dense grids). NaN entries in mmse_curve are excluded from numerator and
    denominator.

    Args:
        t_grid: [T] t values as float64 Tensor or numpy array.
        mmse_curve: [T] MMSE_x1 values (NaN for invalid/skipped t).
        p_mean: Log-normal mean for log(sigma). Default -0.8 (JiT default).
        p_std: Log-normal std for log(sigma). Default 0.8 (JiT default).

    Returns:
        Weighted scalar (float). Returns NaN if no valid points remain.

    Example:
        >>> import numpy as np, torch
        >>> t = make_t_grid(0.01, 0.99, 256)
        >>> mmse = np.ones(256, dtype=np.float64)  # constant MMSE=1 curve
        >>> loss = jit_weighted_loss(t, mmse)
        >>> abs(loss - 1.0) < 0.01  # should be close to 1
        True
    """
    t_np = t_grid.numpy() if isinstance(t_grid, torch.Tensor) else np.asarray(t_grid)
    eps = 1e-8
    t_safe = np.clip(t_np, eps, 1.0 - eps)                       # [T]
    sigma = t_safe / (1.0 - t_safe)                              # [T]
    log_sigma = np.log(sigma)
    weights = np.exp(-0.5 * ((log_sigma - p_mean) / p_std) ** 2)  # [T]

    valid = np.isfinite(mmse_curve) & (weights > 0)
    if valid.sum() == 0:
        return float("nan")
    return float((weights[valid] * mmse_curve[valid]).sum() / weights[valid].sum())


# ---------------------------------------------------------------------------
# Loss order check
# ---------------------------------------------------------------------------

#: Key aliases for M05 legacy naming convention (no _cond suffix on last two).
_LOSS_KEY_ALIAS: Dict[str, str] = {
    "class_mask": "class_mask_cond",
    "class_mask_patch": "class_mask_patch_cond",
}


def check_loss_order(
    losses_dict: Dict[str, float],
    eps_rel: float = 0.01,
) -> bool:
    """Returns True if uncond >= class >= mask >= patch (within eps_rel relative tolerance).

    Accepts both naming conventions:
    - canonical: 'uncond', 'class_cond', 'class_mask_cond', 'class_mask_patch_cond'
    - M05 legacy: 'uncond', 'class_cond', 'class_mask', 'class_mask_patch'

    Args:
        losses_dict: Dict with 4 loss values under either naming convention.
        eps_rel: Relative tolerance for the >= comparisons (default 0.01 = 1%).

    Returns:
        True if all ordering constraints hold within tolerance; False if any
        ordering constraint is violated or any required key is missing.

    Example:
        >>> check_loss_order({'uncond': 1.0, 'class_cond': 0.8, 'class_mask': 0.5, 'class_mask_patch': 0.3})
        True
        >>> check_loss_order({'uncond': 0.5, 'class_cond': 0.8, 'class_mask_cond': 0.6, 'class_mask_patch_cond': 0.4})
        False
    """
    normalized = {_LOSS_KEY_ALIAS.get(k, k): v for k, v in losses_dict.items()}
    required = ("uncond", "class_cond", "class_mask_cond", "class_mask_patch_cond")
    if not all(k in normalized for k in required):
        return False

    seq = [normalized[k] for k in required]
    for prev, curr in zip(seq[:-1], seq[1:]):
        if curr > prev * (1.0 + eps_rel):
            log.info(
                "loss_order: FAIL (%.4f > %.4f * (1 + %.2f))",
                curr, prev, eps_rel,
            )
            return False

    u, c, m, p = seq
    log.info(
        "loss_order: uncond=%.4f class=%.4f mask=%.4f patch=%.4f -> PASS",
        u, c, m, p,
    )
    return True


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

def save_mmse_pt(
    out_path: Path,
    t_grid: torch.Tensor,
    mmse_dict: Dict[str, np.ndarray],
    jit_weighted_losses: Optional[Dict[str, float]],
    loss_order_ok: Optional[bool],
    meta: Dict[str, object],
) -> None:
    """Save MMSE results in canonical output schema.

    Absent modes are stored as NaN tensors of shape [len(t_grid)] so that
    downstream plotters always find all 4 keys without conditional checks.

    Output schema (torch.save dict):
        t_grid:                    float64 Tensor [T]
        mmse_uncond:               float64 Tensor [T]  (NaN if not evaluated)
        mmse_class_cond:           float64 Tensor [T]
        mmse_class_mask_cond:      float64 Tensor [T]
        mmse_class_mask_patch_cond:float64 Tensor [T]
        mmse_mask_only_cond:       float64 Tensor [T]
        jit_weighted_losses:       dict[str, float] or None
        loss_order_ok:             bool or None
        meta:                      dict

    Args:
        out_path: Destination path (.pt). Parent dirs are created if needed.
        t_grid: [T] float64 Tensor — canonical t-axis.
        mmse_dict: Subset of MODE_KEYS mapped to [T] float64 numpy arrays.
            Keys not present are filled with NaN.
        jit_weighted_losses: Dict of mode_name -> scalar loss, or None.
        loss_order_ok: Bool result of check_loss_order, or None.
        meta: Arbitrary metadata dict (checkpoint path, config, etc.).

    Example:
        >>> import tempfile, torch, numpy as np
        >>> from pathlib import Path
        >>> t = make_t_grid(0.01, 0.99, 4)
        >>> mmse = {"mmse_uncond": np.ones(4)}
        >>> with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        ...     save_mmse_pt(Path(f.name), t, mmse, None, None, {})
    """
    T = len(t_grid)
    nan_arr = np.full(T, np.nan, dtype=np.float64)

    result_tensors = {
        key: torch.tensor(mmse_dict.get(key, nan_arr), dtype=torch.float64)
        for key in MODE_KEYS
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "t_grid": t_grid.to(torch.float64),                      # [T] float64
        **result_tensors,
        "jit_weighted_losses": jit_weighted_losses,
        "loss_order_ok": loss_order_ok,
        "meta": meta,
    }
    torch.save(payload, out_path)
    log.info("Saved MMSE results to %s", out_path)


def write_manifest_jsonl(
    out_path: Path,
    *,
    sample_ids: Sequence[int],
    labels: Sequence[int],
    meta: Optional[Dict[str, object]] = None,
) -> None:
    """Write a deterministic JSONL manifest for the evaluated sample columns."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        if meta is not None:
            f.write(json.dumps({"type": "meta", **meta}, sort_keys=True) + "\n")
        for col, (sid, label) in enumerate(zip(sample_ids, labels)):
            f.write(
                json.dumps(
                    {
                        "type": "sample",
                        "column": int(col),
                        "sample_id": int(sid),
                        "label": int(label),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    log.info("Saved MMSE manifest to %s", out_path)


def save_per_sample_errors_pt(
    out_path: Path,
    *,
    t_grid: torch.Tensor,
    per_sample: Dict[str, object],
    meta: Optional[Dict[str, object]] = None,
) -> None:
    """Save per-sample MMSE_x1 errors for offline bootstrap CI.

    Expected per_sample fields are produced by main_mmse_loop when
    per_sample_out is provided:
      - sample_ids: LongTensor [N]
      - labels: LongTensor [N]
      - errors: dict[str, FloatTensor [T, N]]
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "t_grid": t_grid.to(torch.float64).cpu(),
        "sample_ids": per_sample["sample_ids"],
        "labels": per_sample["labels"],
        "errors": per_sample["errors"],
        "meta": meta or {},
    }
    torch.save(payload, out_path)
    log.info("Saved per-sample MMSE errors to %s", out_path)


# ---------------------------------------------------------------------------
# Main MMSE loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def main_mmse_loop(
    rank: int,
    world_size: int,
    t_grid: torch.Tensor,
    x1: torch.Tensor,
    x0: torch.Tensor,
    model_forward_fn: Callable[..., torch.Tensor],
    active_modes: List[int],
    mode_kwargs_fn: Callable[[int], Dict[str, object]],
    t_eps: float = _T_EPS_DEFAULT,
    output_kind: Literal["v", "x"] = "v",
    device: Optional[torch.device] = None,
    logger: Optional[logging.Logger] = None,
    sample_ids: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    per_sample_out: Optional[Dict[str, object]] = None,
) -> Dict[str, np.ndarray]:
    """Run MMSE evaluation loop over t_grid for each active mode.

    Outer loop: t values. Inner loop: active modes (shared x1/x0 per t).
    Handles DDP all_reduce internally (C1 fix). Per-rank noise is supplied
    by the caller via x0 (W1 fix: caller ensures disjoint seeds per rank).

    Args:
        rank: DDP rank (used for progress logging).
        world_size: Total DDP ranks (logged at debug level).
        t_grid: [T] float64 Tensor of t values.
        x1: [N, C, H, W] clean samples on device for this rank.
        x0: [N, C, H, W] noise for this rank (distinct from other ranks).
        model_forward_fn: (x_noisy, t_scalar, **kwargs) -> prediction [N, C, H, W].
        active_modes: List of mode indices (1-based, e.g. [1, 2, 3, 4]).
        mode_kwargs_fn: Given mode index, returns dict of kwargs for model_forward_fn.
            E.g. mode_kwargs_fn(1) -> {"y": null_labels, "mask": zero_mask}.
        t_eps: Clamp for (1-t) denom (C2 fix). Default 5e-2.
        output_kind: "v" if model returns velocity (default); "x" if x-prediction.
            Forwarded to compute_mmse_at_t.
        device: Compute device (inferred from x1 if None).
        logger: Optional logger; falls back to module-level log.

    Returns:
        Dict mapping canonical key (e.g. "mmse_uncond") to float64 numpy array [T].
        Missing modes are NOT included; save_mmse_pt fills them with NaN.

    Example:
        >>> # result = main_mmse_loop(rank=0, world_size=1, t_grid=t,
        >>> #     x1=x1, x0=x0, model_forward_fn=fn,
        >>> #     active_modes=[1], mode_kwargs_fn=lambda m: {"y": labels})
        >>> # result["mmse_uncond"].shape == (T,)
    """
    _log = logger or log
    _device = device or x1.device
    T = len(t_grid)
    _log.debug("main_mmse_loop: rank=%d world_size=%d T=%d modes=%s", rank, world_size, T, active_modes)

    _MODE_KEY = {
        1: "mmse_uncond",
        2: "mmse_class_cond",
        3: "mmse_class_mask_cond",
        4: "mmse_class_mask_patch_cond",
        5: "mmse_mask_only_cond",
        6: "mmse_canny_only_cond",
        7: "mmse_class_canny_cond",
        8: "mmse_mask_canny_cond",
    }

    # Initialize per-mode result buffers
    mode_sums: Dict[int, np.ndarray] = {
        m: np.full(T, np.nan, dtype=np.float64) for m in active_modes
    }
    collect_per_sample = per_sample_out is not None
    if collect_per_sample:
        if sample_ids is None or labels is None:
            raise ValueError("sample_ids and labels are required when per_sample_out is provided")
        local_errors: Dict[int, np.ndarray] = {
            m: np.full((T, int(x1.shape[0])), np.nan, dtype=np.float32) for m in active_modes
        }

    for ti, t_val in enumerate(t_grid.tolist()):
        t_scalar = float(t_val)
        denom = max(1.0 - t_scalar, t_eps)                       # C2 fix

        for mode in active_modes:
            kwargs = mode_kwargs_fn(mode)
            try:
                if collect_per_sample:
                    sample_err = compute_sample_errors_at_t(
                        model_forward_fn, x1, t_scalar, x0,
                        output_kind=output_kind, t_eps=t_eps, forward_kwargs=kwargs,
                    )
                    local_errors[mode][ti, : sample_err.numel()] = sample_err.detach().cpu().numpy()
                    mse_sum = sample_err.sum()
                    n_count = torch.tensor(
                        sample_err.numel(), dtype=torch.float32, device=x1.device
                    )
                else:
                    mse_sum, n_count = compute_mmse_at_t(
                        model_forward_fn, x1, t_scalar, x0,
                        output_kind=output_kind, t_eps=t_eps, forward_kwargs=kwargs,
                    )
            except RuntimeError as exc:
                if rank == 0:
                    _log.warning("ti=%d mode=%d RuntimeError: %s — skip", ti, mode, exc)
                continue

            mse_v_global = all_reduce_mmse(mse_sum, n_count)     # C1 fix

            if not np.isfinite(mse_v_global):
                if rank == 0:
                    _log.warning("ti=%d mode=%d NaN/Inf in mse_v — skip", ti, mode)
                continue

            if collect_per_sample:
                mode_sums[mode][ti] = mse_v_global               # already MMSE_x1
            else:
                mode_sums[mode][ti] = (denom ** 2) * mse_v_global    # MMSE_x1

        if rank == 0 and (ti % 32 == 0 or ti == T - 1):
            sample_mode = active_modes[0]
            val = mode_sums[sample_mode][ti]
            _log.info(
                "t=%.4f (%d/%d) mode%d mmse_x1=%.4f",
                t_scalar, ti + 1, T, sample_mode,
                val if np.isfinite(val) else float("nan"),
            )

    if collect_per_sample:
        local_obj = {
            "sample_ids": sample_ids.detach().cpu().long(),
            "labels": labels.detach().cpu().long(),
            "errors": {_MODE_KEY[m]: torch.from_numpy(local_errors[m]) for m in active_modes},
        }
        gathered: Optional[List[object]]
        if rank == 0:
            gathered = [None for _ in range(world_size)]
        else:
            gathered = None
        dist.gather_object(local_obj, object_gather_list=gathered, dst=0)
        if rank == 0:
            assert gathered is not None
            sample_parts = [g["sample_ids"] for g in gathered if g is not None]  # type: ignore[index]
            label_parts = [g["labels"] for g in gathered if g is not None]       # type: ignore[index]
            sample_all = torch.cat(sample_parts).long()
            label_all = torch.cat(label_parts).long()
            order = torch.argsort(sample_all)
            errors_all = {}
            for m in active_modes:
                key = _MODE_KEY[m]
                err_parts = [g["errors"][key] for g in gathered if g is not None]  # type: ignore[index]
                errors_all[key] = torch.cat(err_parts, dim=1)[:, order].float()
            per_sample_out.update(
                {
                    "sample_ids": sample_all[order],
                    "labels": label_all[order],
                    "errors": errors_all,
                }
            )
    return {_MODE_KEY[m]: mode_sums[m] for m in active_modes}
