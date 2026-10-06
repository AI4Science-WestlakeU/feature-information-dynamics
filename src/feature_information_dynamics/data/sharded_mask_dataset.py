"""Sharded mask dataset: mirrors latent shard protocol for mask conditioning.

Replaces failed single-megafile approaches (mmap-on-NFS slow, eager-load fork-stall).
Per latent shard, read corresponding mask shard via safetensors mmap — same speed
profile as latent IO (proven 5+ sps in W warmup smoke).

Mask shard schema (produced by build_mask_shards.py):
    masks_rank{RR}_shard{SSS}.safetensors
        keys: masks [N_shard, 256, 256] uint8
              mask_valid [N_shard] uint8

Usage::
    from feature_information_dynamics.data.sharded_mask_dataset import MaskedShardedDataset
    ds = MaskedShardedDataset(
        data_dir="...",
        mask_shard_dir="DATA/mask_shards",
        latent_norm=True,
        latent_multiplier=1.0,
        paired_whitelist_json="...",
        phase="mask",   # "warmup" -> no mask read; "mask" -> read mask shard slice
    )
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from safetensors import safe_open

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from feature_information_dynamics.data.whitelisted_img_latent_dataset import WhitelistedImgLatentDataset  # noqa: E402

_VALID_PHASES = frozenset({"warmup", "mask"})


class MaskedShardedDataset(WhitelistedImgLatentDataset):
    """WhitelistedImgLatentDataset + per-shard mask read via safetensors.

    In "warmup" phase no mask shards are opened — zero-IO contract.
    In "mask" phase each __getitem__ opens the matching mask shard shard and
    reads one slice (same index as the latent slice) via get_slice.

    Returns:
        warmup: (latent [C,H,W] float32, label int)
        mask:   (latent [C,H,W] float32, label int, mask [256,256] float32)
    """

    def __init__(
        self,
        data_dir: str,
        mask_shard_dir: str,
        latent_norm: bool = True,
        latent_multiplier: float = 1.0,
        sr95_filter: bool = False,
        split: str = "train",
        paired_whitelist_json: Optional[str] = None,
        phase: str = "warmup",
        is_val: bool = False,
    ) -> None:
        """Initialize dataset.

        Args:
            data_dir:             Directory containing latents_rank*.safetensors shards.
            mask_shard_dir:       Directory containing masks_rank*.safetensors shards
                                  (one-to-one with latent shards, same N samples, same order).
            latent_norm:          Apply z-score normalisation using latents_stats.pt.
            latent_multiplier:    Extra scale factor after z-normalisation.
            sr95_filter:          Pass through to parent (reserved for future use).
            split:                "train" or "val" (9:1 shard-level deterministic split).
            paired_whitelist_json: Path to JSON whitelist {"filenames": [...]}.
            phase:                "warmup" (no mask IO) or "mask" (read mask shard slice).

        Raises:
            ValueError:      If phase is not "warmup" or "mask".
            FileNotFoundError: If phase=="mask" and any mask shard counterpart is missing.
        """
        if phase not in _VALID_PHASES:
            raise ValueError(
                f"phase must be one of {sorted(_VALID_PHASES)}, got {phase!r}"
            )

        super().__init__(
            data_dir=data_dir,
            latent_norm=latent_norm,
            latent_multiplier=latent_multiplier,
            sr95_filter=sr95_filter,
            split=split,
            paired_whitelist_json=paired_whitelist_json,
        )

        self._phase = phase
        self._is_val = is_val   # val mode: no random hflip → deterministic latent + mask
        self._mask_shard_dir = Path(mask_shard_dir)
        self._latent_to_mask: dict[str, str] = {}

        if phase == "mask":
            self._latent_to_mask = self._build_latent_to_mask_map()

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_latent_to_mask_map(self) -> dict[str, str]:
        """Build {latent_shard_path: mask_shard_path} for every latent shard.

        Derives mask shard name by replacing "latents_" prefix with "masks_"
        in the latent shard filename.  Raises FileNotFoundError for any missing
        counterpart so bad configurations fail at init rather than mid-epoch.

        Returns:
            Mapping from latent shard path string to mask shard path string.
        """
        mapping: dict[str, str] = {}
        for latent_safe_file in self.files:
            stem = Path(latent_safe_file).name          # latents_rank00_shard000.safetensors
            mask_name = stem.replace("latents_", "masks_", 1)
            mask_path = str(self._mask_shard_dir / mask_name)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(
                    f"Mask shard not found: {mask_path} "
                    f"(expected counterpart of {latent_safe_file})"
                )
            mapping[latent_safe_file] = mask_path

        print(
            f"[MaskedShardedDataset] phase=mask | mapped {len(mapping)} "
            f"latent->mask shard pairs in {self._mask_shard_dir}"
        )
        return mapping

    def _read_mask_slice(self, latent_safe_file: str, idx_in_file: int) -> torch.Tensor:
        """Open mask shard and read one slice at idx_in_file.

        Uses safe_open context manager per call — mirrors upstream ImgLatentDataset
        __getitem__ IO pattern.  Returns float32 mask in [0.0, 1.0].

        Args:
            latent_safe_file: Path to the latent shard (used to look up mask shard).
            idx_in_file:      Local index within the shard (same as latent local idx).

        Returns:
            [256, 256] float32 tensor with values in {0.0, 1.0}.
        """
        mask_path = self._latent_to_mask[latent_safe_file]
        with safe_open(mask_path, framework="pt", device="cpu") as f:
            # mask shard has SAME N samples in SAME order as latent shard
            mask_uint8 = f.get_slice("masks")[idx_in_file]   # [256, 256] uint8
        return mask_uint8.float()                              # [256, 256] float32

    # ── Public API ─────────────────────────────────────────────────────────────

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, int] | tuple[torch.Tensor, int, torch.Tensor]:
        """Return one sample with synchronized horizontal-flip data augmentation.

        Upstream ImgLatentDataset.__getitem__ randomly picks `latents` or
        `latents_flip` per sample (50/50). For mask phase we MUST mirror that
        decision and flip the mask in lockstep — otherwise latent and mask
        come from different orientations and conditioning is broken.

        Implementation: bypass super() to control the flip decision; replicate
        upstream's latent norm + multiplier logic.

        Args:
            idx: Dataset index in [0, len(self)).

        Returns:
            warmup phase: (latent [C,H,W] float32, label int)
            mask phase:   (latent [C,H,W] float32, label int, mask [256,256] float32)
        """
        if self._phase != "mask":
            # Warmup phase. For val, must avoid upstream's random hflip to keep determinism.
            if self._is_val:
                # Re-implement upstream __getitem__ but force non-flipped latent.
                global_idx = self.valid_indices[idx]
                info = self._full_map[global_idx]
                with safe_open(info["safe_file"], framework="pt", device="cpu") as f:
                    feature = f.get_slice("latents")[info["idx_in_file"] : info["idx_in_file"] + 1]
                    label = f.get_slice("labels")[info["idx_in_file"] : info["idx_in_file"] + 1]
                if self.latent_norm:
                    feature = (feature - self._latent_mean) / self._latent_std
                feature = feature * self.latent_multiplier
                return feature.squeeze(0), label.squeeze(0)
            # Train: upstream random hflip via latents/latents_flip 50/50
            return super().__getitem__(idx)

        # Mask phase: control the flip decision so mask mirrors latent.
        global_idx = self.valid_indices[idx]
        info = self._full_map[global_idx]
        latent_safe_file: str = info["safe_file"]
        idx_in_file: int = info["idx_in_file"]

        # Val: no flip (deterministic). Train: 50/50 random flip, mask flipped in lockstep.
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

        mask = self._read_mask_slice(latent_safe_file, idx_in_file)         # [256, 256] f32
        if do_flip:
            mask = mask.flip(-1)                                            # H-flip → synchronized

        return feature, label, mask
