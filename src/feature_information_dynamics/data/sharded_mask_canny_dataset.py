"""Sharded mask + canny dataset (latent-domain).

Standalone module so SDVAE/SiT and other reps can import without dragging
LightningDiT model imports (which conflict with SiT's `models` package on
shared sys.path).

Phase semantics:
  warmup:     latent + label only (no mask, no canny)
  mask:       latent + label + mask (canny zeroed / not loaded)
  canny:      latent + label + canny (mask zeroed / not loaded)
  canny_mask: latent + label + mask + canny

Returns: (latent [C,H,W], label int, mask [256,256] f32, canny [256,256] f32)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from safetensors import safe_open

# Reuse parent class from existing module
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from feature_information_dynamics.data.sharded_mask_dataset import MaskedShardedDataset  # noqa: E402

# Valid training phases (mirrors RAE M06 phase semantics)
PHASE_CHOICES: tuple[str, ...] = ("warmup", "mask", "canny", "canny_mask")


class MaskedCannyShardedDataset(MaskedShardedDataset):
    """MaskedShardedDataset extended to also load canny shards.

    Canny shard schema (produced by build_canny_shards.py):
        cannys_rank{RR}_shard{SSS}.safetensors
            keys: cannys [N_shard, 256, 256] uint8 (binary edges, 0/1)

    Canny shards are 1:1 aligned with latent shards (same N, same order).

    Phase semantics:
      warmup:     latent + label only (no mask, no canny)
      mask:       latent + label + mask (canny zeroed / not loaded)
      canny:      latent + label + canny (mask zeroed / not loaded)
      canny_mask: latent + label + mask + canny

    Returns:
        All phases: (latent [C,H,W], label int, mask [256,256] f32, canny [256,256] f32)
        Mask is zeros when phase != mask/canny_mask.
        Canny is zeros when phase != canny/canny_mask.
    """

    _VALID_PHASES_EXT: frozenset[str] = frozenset(PHASE_CHOICES)

    def __init__(
        self,
        data_dir: str,
        mask_shard_dir: str,
        canny_shard_dir: str,
        latent_norm: bool = True,
        latent_multiplier: float = 1.0,
        sr95_filter: bool = False,
        split: str = "train",
        paired_whitelist_json: Optional[str] = None,
        phase: str = "warmup",
        is_val: bool = False,
    ) -> None:
        """Initialize dataset with mask + canny shard support.

        Args:
            data_dir:             Directory with latents_rank*.safetensors shards.
            mask_shard_dir:       Directory with masks_rank*.safetensors shards.
            canny_shard_dir:      Directory with cannys_rank*.safetensors shards.
            latent_norm:          Apply z-score normalisation via latents_stats.pt.
            latent_multiplier:    Extra scale factor after z-normalisation.
            sr95_filter:          Pass through to parent.
            split:                "train" or "val".
            paired_whitelist_json: Path to JSON whitelist {"filenames": [...]}.
            phase:                One of PHASE_CHOICES.
            is_val:               Deterministic non-flipped mode when True.

        Raises:
            ValueError:        If phase is not in PHASE_CHOICES.
            FileNotFoundError: If phase requires canny and a canny shard is missing.
        """
        if phase not in self._VALID_PHASES_EXT:
            raise ValueError(
                f"phase must be one of {sorted(self._VALID_PHASES_EXT)}, got {phase!r}"
            )
        # MaskedShardedDataset validates phase against its own _VALID_PHASES (warmup/mask).
        # Pass "warmup" to parent for non-mask phases to bypass parent's stricter check;
        # we manage mask loading ourselves via _phase below.
        _parent_phase = "mask" if phase in ("mask", "canny_mask") else "warmup"
        super().__init__(
            data_dir=data_dir,
            mask_shard_dir=mask_shard_dir,
            latent_norm=latent_norm,
            latent_multiplier=latent_multiplier,
            sr95_filter=sr95_filter,
            split=split,
            paired_whitelist_json=paired_whitelist_json,
            phase=_parent_phase,
            is_val=is_val,
        )
        # Override with full phase (parent stores "warmup" or "mask" only)
        self._phase = phase
        self._canny_shard_dir = Path(canny_shard_dir)
        self._latent_to_canny: dict[str, str] = {}

        if phase in ("canny", "canny_mask"):
            self._latent_to_canny = self._build_latent_to_canny_map()

    def _build_latent_to_canny_map(self) -> dict[str, str]:
        """Build {latent_shard_path: canny_shard_path} for every latent shard.

        Derives canny shard name by replacing "latents_" prefix with "cannys_".
        Raises FileNotFoundError for any missing counterpart.
        """
        mapping: dict[str, str] = {}
        for latent_safe_file in self.files:
            stem = Path(latent_safe_file).name          # latents_rank00_shard000.safetensors
            canny_name = stem.replace("latents_", "cannys_", 1)
            canny_path = str(self._canny_shard_dir / canny_name)
            if not os.path.exists(canny_path):
                raise FileNotFoundError(
                    f"Canny shard not found: {canny_path} "
                    f"(expected counterpart of {latent_safe_file})"
                )
            mapping[latent_safe_file] = canny_path
        print(
            f"[MaskedCannyShardedDataset] phase={self._phase} | "
            f"mapped {len(mapping)} latent->canny shard pairs in {self._canny_shard_dir}"
        )
        return mapping

    def _read_canny_slice(self, latent_safe_file: str, idx_in_file: int) -> torch.Tensor:
        """Open canny shard and read one slice at idx_in_file.

        Returns:
            [256, 256] float32 tensor with values in {0.0, 1.0}.
        """
        canny_path = self._latent_to_canny[latent_safe_file]
        with safe_open(canny_path, framework="pt", device="cpu") as f:
            canny_uint8 = f.get_slice("cannys")[idx_in_file]   # [256, 256] uint8
        return canny_uint8.float()                               # [256, 256] float32

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        """Return (latent, label, mask, canny) with synchronized horizontal flip.

        Mask and canny are zeros tensors when not active for the current phase.
        Flip decision is consistent across latent, mask, and canny.
        """
        global_idx = self.valid_indices[idx]
        info = self._full_map[global_idx]
        latent_safe_file: str = info["safe_file"]
        idx_in_file: int = info["idx_in_file"]

        # Val: no flip (deterministic). Train: 50/50 random flip.
        do_flip = (not self._is_val) and bool(np.random.uniform(0.0, 1.0) < 0.5)
        lat_key = "latents_flip" if do_flip else "latents"

        with safe_open(latent_safe_file, framework="pt", device="cpu") as f:
            feature = f.get_slice(lat_key)[idx_in_file : idx_in_file + 1]   # [1, C, H, W]
            label = f.get_slice("labels")[idx_in_file : idx_in_file + 1]    # [1]

        if self.latent_norm:
            feature = (feature - self._latent_mean) / self._latent_std
        feature = feature * self.latent_multiplier
        feature = feature.squeeze(0)                                        # [C, H, W]
        label = label.squeeze(0)                                            # scalar

        _zeros = torch.zeros(256, 256, dtype=torch.float32)

        # Load mask when phase requires it
        if self._phase in ("mask", "canny_mask"):
            mask = self._read_mask_slice(latent_safe_file, idx_in_file)    # [256, 256] f32
            if do_flip:
                mask = mask.flip(-1)
        else:
            mask = _zeros

        # Load canny when phase requires it
        if self._phase in ("canny", "canny_mask"):
            canny = self._read_canny_slice(latent_safe_file, idx_in_file)  # [256, 256] f32
            if do_flip:
                canny = canny.flip(-1)
        else:
            canny = _zeros

        return feature, label, mask, canny
