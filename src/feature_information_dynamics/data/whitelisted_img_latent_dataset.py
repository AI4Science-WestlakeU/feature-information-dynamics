"""Whitelist-filtered ImgLatentDataset for IO isolation testing.

Minimal subclass of upstream ImgLatentDataset that applies an init-time
paired-whitelist filter via per-shard JSON sidecars. No per-step query.
__getitem__ is inherited unchanged from upstream — pure safetensors read.

Goal: isolate the contribution of "subset-of-shards" reads to sps regression.
A1 v3 (no whitelist, full 648K) hit 2.85 sps on H800. Drop to ~547K via
whitelist; if sps drops materially, scattered reads across shards are the
cause; if sps holds, whitelist filtering is innocent.

Usage::
    from feature_information_dynamics.data.whitelisted_img_latent_dataset import WhitelistedImgLatentDataset

    ds = WhitelistedImgLatentDataset(
        data_dir,
        latent_norm=True,
        latent_multiplier=1.0,
        paired_whitelist_json="DATA/paired_train_whitelist.json",
    )
"""
from __future__ import annotations

import json
import os
from typing import Optional

from datasets.img_latent_dataset import ImgLatentDataset


class WhitelistedImgLatentDataset(ImgLatentDataset):
    """ImgLatentDataset + paired-filename whitelist (init-time filter only)."""

    def __init__(
        self,
        data_dir: str,
        latent_norm: bool = True,
        latent_multiplier: float = 1.0,
        sr95_filter: bool = False,
        split: str = "train",
        paired_whitelist_json: Optional[str] = None,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            latent_norm=latent_norm,
            latent_multiplier=latent_multiplier,
            sr95_filter=sr95_filter,
            split=split,
        )
        if paired_whitelist_json is None:
            return

        with open(paired_whitelist_json) as fh:
            wl_data = json.load(fh)
        wl_set: set[str] = set(wl_data["filenames"])

        # Per-shard sidecar filenames (must exist for paired filtering).
        shard_to_fns: dict[str, Optional[list[str]]] = {}
        for safe_file in self.files:
            sidecar = safe_file.replace(".safetensors", ".json")
            if not os.path.exists(sidecar):
                shard_to_fns[safe_file] = None
                continue
            with open(sidecar) as fh:
                shard_to_fns[safe_file] = json.load(fh)["filenames"]

        n_before: int = len(self.valid_indices)
        seen: set[str] = set()
        new_valid: list[int] = []
        for global_idx in self.valid_indices:
            mapping = self._full_map[global_idx]
            fns = shard_to_fns.get(mapping["safe_file"])
            if fns is None:
                continue
            fn = fns[mapping["idx_in_file"]]
            if fn not in wl_set or fn in seen:
                continue
            seen.add(fn)
            new_valid.append(global_idx)
        self.valid_indices = new_valid
        print(
            f"[WhitelistedImgLatentDataset] paired whitelist filter ({split}): "
            f"{n_before} → {len(self.valid_indices)} samples"
        )
