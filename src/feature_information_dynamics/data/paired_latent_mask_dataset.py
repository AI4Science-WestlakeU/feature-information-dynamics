"""Offline latent + mask paired dataset for B-path training (task #10).

Design decision: wrapper class rather than patching third_party/LightningDiT/datasets/.
Rationale: third_party should remain upstream-pull-compatible; all project-specific
logic lives here.  Downstream train scripts (train_lightningdit_mask_patch.py,
train_sit_mask_patch.py) import PairedLatentMaskDataset from this module instead of
ImgLatentDataset directly.

Shard schema (produced by extract_latents_with_filenames.py):
    latents_rank{RR}_shard{SSS}.safetensors  — {latents, latents_flip, labels}
    latents_rank{RR}_shard{SSS}.json         — {"filenames": [...]}  (sidecar)

Union mask schema (from dev_docs/data/sr95_union_mask_recipe.md):
    {mask_root}/shard_{0..6}/{synset}/{stem}_masks.npz — keys: mask(H,W)bool, score, n_instances

Backward compat: old shards without JSON sidecar return None for filename/mask.

Usage::
    from feature_information_dynamics.data.paired_latent_mask_dataset import PairedLatentMaskDataset

    ds = PairedLatentMaskDataset(
        data_dir="DATA/imagenet_latents_f16d32_sr95_v2/.../imagenet_train_256",
        mask_root="DATA/masks_union",
        imagenet_root="DATA/imagenet",  # for patch-phase RGB load
        use_hflip=True,   # synchronized per-sample random hflip (lat+mask+image)
        load_image=True,  # load RGB for patch conditioning; False = zeros sentinel
        latent_norm=True,
        image_size=256,
        latent_multiplier=1.0,
    )
    # Returns dict: {latent, label, filename, mask_2d, mask_valid,
    #                image_3ch, image_valid, patch_tok}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open
from torch.utils.data import Dataset
from torchvision import transforms

# TOKEN_GRID constants (must match train_lib_mask_patch)
_TOKEN_GRID: int = 16

# ── Shard index helpers ───────────────────────────────────────────────────────

def _build_shard_index(
    data_dir: Path,
    latent_norm: bool,
    latent_multiplier: float,
    split: str,
    sr95_filter: bool,
) -> tuple[
    list[tuple[int, int]],
    list[str],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Build flat (shard_file_idx, local_sample_idx) index from all safetensors shards.

    Args:
        data_dir:          directory containing latents_rank*.safetensors shards.
        latent_norm:       if True, apply z-score normalization using latents_stats.pt.
        latent_multiplier: scale factor after z-score.
        split:             "train" or "val" (9:1 split by shard index).
        sr95_filter:       if True, only use shards with SR95 class labels.

    Returns:
        (index, shard_files, latent_mean, latent_std)
        - index:        list of (shard_file_idx, local_sample_idx) tuples.
        - shard_files:  list of safetensors paths (post-split filter).
        - latent_mean:  [1, C, 1, 1] float32 from latents_stats.pt (None if missing).
        - latent_std:   [1, C, 1, 1] float32 from latents_stats.pt (None if missing).
    """
    import glob as glob_mod
    shard_files = sorted(glob_mod.glob(str(data_dir / "latents_rank*.safetensors")))
    if not shard_files:
        raise FileNotFoundError(f"No latent shards found in {data_dir}")

    # 9:1 train/val split by shard index (deterministic)
    n = len(shard_files)
    if split == "val":
        shard_files = shard_files[n * 9 // 10:]
    else:
        shard_files = shard_files[:n * 9 // 10]

    index: list[tuple[int, int]] = []
    for sf_idx, sf in enumerate(shard_files):
        with safe_open(sf, framework="pt", device="cpu") as f:
            n_samples = f.get_slice("labels").get_shape()[0]
        for local_i in range(n_samples):
            index.append((sf_idx, local_i))

    latent_mean = latent_std = None
    if latent_norm:
        stats_path = data_dir / "latents_stats.pt"
        if stats_path.exists():
            stats = torch.load(str(stats_path), map_location="cpu")
            latent_mean = stats["mean"]  # [1, C, 1, 1]
            latent_std = stats["std"]    # [1, C, 1, 1]

    return index, shard_files, latent_mean, latent_std


def _load_sidecar_filenames(shard_path: str) -> Optional[list[str]]:
    """Load per-shard JSON sidecar for filenames. Returns None if absent (backward compat)."""
    json_path = Path(shard_path).with_suffix(".json")
    if not json_path.exists():
        return None
    with open(json_path) as f:
        data = json.load(f)
    return data.get("filenames")


def _build_canny_index(canny_root: Optional[str]) -> dict[str, tuple[Path, int]]:
    """Build filename -> (shard_path, local_idx) lookup for Canny npz shards."""
    if not canny_root:
        return {}
    root = Path(canny_root)
    out: dict[str, tuple[Path, int]] = {}
    for shard in sorted(root.glob("shard_*.npz")):
        with np.load(shard, allow_pickle=True) as d:
            for i, fn in enumerate(d["filename"]):
                out[str(fn)] = (shard, i)
    return out


def _load_canny_from_index(
    canny_index: dict[str, tuple[Path, int]],
    filename: str,
    image_size: int,
) -> tuple[torch.Tensor, bool]:
    """Load one Canny map from shard index; zeros sentinel when absent."""
    if not filename or filename not in canny_index:
        return torch.zeros(image_size, image_size, dtype=torch.float32), False
    shard, local_i = canny_index[filename]
    with np.load(shard, allow_pickle=True) as d:
        edge = d["edge"][local_i].astype(np.float32)
    if edge.max() > 1.0:
        edge = edge / 255.0
    return torch.from_numpy(edge).float(), True


# ── Mask lookup ───────────────────────────────────────────────────────────────

def _find_mask_npz(mask_root: Path, filename: str) -> Optional[Path]:
    """Locate union mask npz for a given "{synset}/{stem}.JPEG" filename.

    Searches mask_root/shard_*/{synset}/{stem}_masks.npz in sorted shard order.

    Args:
        mask_root: union mask root containing shard_0..6 subdirs.
        filename:  "{synset}/{stem}.JPEG" e.g. "n01440764/n01440764_10026.JPEG"

    Returns:
        Path to npz if found, else None.
    """
    parts = filename.split("/", 1)
    if len(parts) != 2:
        return None
    synset, basename = parts
    stem = basename.removesuffix(".JPEG").removesuffix(".jpeg").removesuffix(".jpg")
    for shard_dir in sorted(mask_root.glob("shard_*")):
        candidate = shard_dir / synset / f"{stem}_masks.npz"
        if candidate.exists():
            return candidate
    return None


def _load_mask_resize(npz_path: Path, image_size: int) -> torch.Tensor:
    """Load union mask from npz, resize preserving aspect ratio, then center-crop.

    Matches torchvision Resize(image_size) + CenterCrop(image_size) used during
    latent extraction — ensures mask spatial alignment with latent tokens.

    Args:
        npz_path:   path to *_masks.npz.
        image_size: target spatial size (both height and width).

    Returns:
        [image_size, image_size] float32 in {0, 1}.
    """
    with np.load(str(npz_path)) as d:
        mask_hw = d["mask"].copy()                          # (H, W) bool — must .copy()
    H, W = mask_hw.shape
    # Step 1: resize short edge to image_size, preserving aspect ratio.
    if H <= W:
        new_H, new_W = image_size, int(round(W * image_size / H))
    else:
        new_H, new_W = int(round(H * image_size / W)), image_size
    m = torch.from_numpy(mask_hw.astype(np.float32))[None, None]  # [1, 1, H, W]
    m = F.interpolate(m, size=(new_H, new_W), mode="nearest")      # [1, 1, new_H, new_W]
    # Step 2: center-crop to (image_size, image_size).
    top = (new_H - image_size) // 2
    left = (new_W - image_size) // 2
    m = m[:, :, top:top + image_size, left:left + image_size]      # [1, 1, image_size, image_size]
    return m[0, 0].contiguous()                                     # [image_size, image_size]


# ── Image loader ─────────────────────────────────────────────────────────────

_IMG_NORMALIZE_MEAN = [0.5, 0.5, 0.5]
_IMG_NORMALIZE_STD = [0.5, 0.5, 0.5]


# ── Dataset ───────────────────────────────────────────────────────────────────

class PairedLatentMaskDataset(Dataset):
    """Offline VAE latent + SAM3 union mask paired dataset.

    Returns dict when mask_root is given, else (latent, label) tuple for backward compat.

    Dict keys:
        latent      : [C, H, W] float32 (z-normalized if latent_norm=True)
        label       : int class id
        filename    : str "{synset}/{stem}.JPEG" (or "" if no sidecar)
        mask_2d     : [image_size, image_size] float32 in {0,1} (zeros if not found)
        mask_valid  : bool — True only when real mask loaded
        image_3ch   : [3, image_size, image_size] float32 in [-1, 1] (zeros if not loaded)
        image_valid : bool — True only when real image loaded
        patch_tok   : int in [0, 255] random token for patch conditioning
    """

    def __init__(
        self,
        data_dir: str,
        mask_root: Optional[str] = None,
        imagenet_root: Optional[str] = None,
        image_size: int = 256,
        latent_norm: bool = True,
        latent_multiplier: float = 1.0,
        split: str = "train",
        sr95_filter: bool = True,
        use_flip: bool = False,
        use_hflip: bool = True,
        load_image: bool = False,
        canny_root: Optional[str] = None,
        paired_train_whitelist_json: Optional[str] = None,
        paired_val_whitelist_json: Optional[str] = None,
        mask_required: bool = True,
        canny_required: bool = True,
    ) -> None:
        """Initialize dataset from latent shard directory.

        Args:
            data_dir:          directory with latents_rank*.safetensors shards.
            mask_root:         union mask root (shard_0..6/ layout). None = no mask.
            imagenet_root:     ImageNet root (train/ subdir expected). Required when
                               load_image=True; raises ValueError if mask_root set and
                               imagenet_root is None with load_image=True.
            image_size:        spatial size for mask/image resize (must match training crop).
            latent_norm:       z-normalize latents with latents_stats.pt.
            latent_multiplier: extra scale after z-norm (1.0 for vavae/sdvae default).
            split:             "train" or "val" (9:1 shard-level deterministic split).
            sr95_filter:       reserved for future filtering; currently pass-through.
            use_flip:          if True, always load latents_flip (static; legacy compat).
            use_hflip:         if True, apply per-sample random 50% horizontal flip,
                               synchronized across latent + mask + image.
            load_image:        if True and imagenet_root set, load RGB image_3ch for patch
                               conditioning. Default False (opt-in); warmup/mask phases
                               leave at False to skip JPEG IO — returns zeros sentinel.
            paired_train_whitelist_json: path to JSON whitelist {"filenames": [...]} for
                               train split. Only samples whose filename (synset/file or bare
                               file) is in the set are retained. None = no filter (default).
            paired_val_whitelist_json:   same as above for val split. Cannot pass both in
                               the same instance — build separate train/val datasets.
        """
        assert not (
            paired_train_whitelist_json is not None
            and paired_val_whitelist_json is not None
        ), (
            "Cannot pass both paired_train_whitelist_json and paired_val_whitelist_json "
            "to a single PairedLatentMaskDataset instance — build separate train/val datasets."
        )
        self._data_dir = Path(data_dir)
        self._mask_root = Path(mask_root) if mask_root else None
        self._imagenet_root = Path(imagenet_root) if imagenet_root else None
        # F2: when False, skip per-sample NFS mask/canny load (warmup phase saves
        # ~547k NPZ loads + ~4.4M NFS stat per epoch with 8 ranks). See team report
        # `_vavae_pipeline_full_diff_20260501v2.md` Root Cause #1.
        self._mask_required = mask_required
        self._canny_required = canny_required
        self._image_size = image_size
        self._latent_norm = latent_norm
        self._latent_multiplier = latent_multiplier
        self._use_flip = use_flip
        self._use_hflip = use_hflip
        self._load_image = load_image
        self._canny_index = _build_canny_index(canny_root)
        # Dual transforms hoisted at init — mirrors extract_latents_with_filenames.py:341-352.
        # Uses PIL-space flip (RandomHorizontalFlip p=1.0) to be bit-identical to latents_flip.
        self._image_tfm_normal = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(_IMG_NORMALIZE_MEAN, _IMG_NORMALIZE_STD),
        ])
        self._image_tfm_flip = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(_IMG_NORMALIZE_MEAN, _IMG_NORMALIZE_STD),
        ])

        if self._mask_root is not None and load_image and self._imagenet_root is None:
            raise ValueError(
                "imagenet_root must be provided when mask_root is set and load_image=True. "
                "Pass imagenet_root='/path/to/imagenet' or set load_image=False."
            )

        self._index, self._shard_files, self._latent_mean, self._latent_std = (
            _build_shard_index(self._data_dir, latent_norm, latent_multiplier, split, sr95_filter)
        )

        # Pre-load filenames sidecars (None per shard if absent — backward compat).
        self._shard_filenames: list[Optional[list[str]]] = [
            _load_sidecar_filenames(sf) for sf in self._shard_files
        ]

        # Paired whitelist filter: keep only samples whose filename is in the whitelist.
        # Symmetric to MaskCondImageNetDataset paired hook (mask_cond_dataset.py:198-203).
        self._paired_wl_set: Optional[set] = None
        _wl_json: Optional[str] = None
        if paired_train_whitelist_json is not None and split == "train":
            _wl_json = paired_train_whitelist_json
        elif paired_val_whitelist_json is not None and split == "val":
            _wl_json = paired_val_whitelist_json
        if _wl_json is not None:
            with open(_wl_json) as fh:
                _wl_data = json.load(fh)
            self._paired_wl_set = set(_wl_data["filenames"])
            n_before = len(self._index)
            # Manifest is canonical "synset/file.JPEG" format. Strict membership +
            # dedup by filename: latent shards can contain the same filename twice
            # (verify v2 found 6 dupes between 547454 vs 547448 manifest size).
            # Dedup ensures 4-rep training sees identical sample sets.
            _seen: set[str] = set()
            new_index: list[tuple[int, int]] = []
            for sf_idx, local_i in self._index:
                shard_fns = self._shard_filenames[sf_idx]
                if shard_fns is None:
                    continue
                fn = shard_fns[local_i]
                if fn not in self._paired_wl_set or fn in _seen:
                    continue
                _seen.add(fn)
                new_index.append((sf_idx, local_i))
            self._index = new_index
            n_after = len(self._index)
            print(
                f"[PairedLatentMaskDataset] paired whitelist filter ({split}): "
                f"{n_before} → {n_after} samples"
            )

        # W2: Validate mask_root schema — must be union mask (key="mask", H×W bool),
        # not legacy mask_root_v3 (key="masks", N×H×W). Fail fast at init, not per-sample.
        if self._mask_root is not None:
            sample_npz = next(self._mask_root.glob("shard_*/*/*_masks.npz"), None)
            if sample_npz is not None:
                with np.load(str(sample_npz)) as d:
                    if "mask" not in d.files:
                        raise ValueError(
                            f"mask_root={mask_root} npz lacks 'mask' key (found {sorted(d.files)}). "
                            "Use the union mask schema (dev_docs/data/sr95_union_mask_recipe.md), "
                            "not the legacy mask_root_v3 schema (key='masks', shape N×H×W)."
                        )

    def _load_image_3ch(self, jpeg_path: Path, do_flip: bool) -> torch.Tensor:
        """Load one JPEG using the same transform chain as extract_latents_with_filenames.py.

        Bit-identical to extract pipeline (lines 341-352):
            normal: Resize → CenterCrop → ToTensor → Normalize  (matches tfm)
            flip:   Resize → CenterCrop → RandomHFlip(p=1) → ToTensor → Normalize  (matches tfm_flip)

        Returns:
            [3, image_size, image_size] float32 in [-1, 1].
        """
        img = Image.open(str(jpeg_path)).convert("RGB")
        tfm = self._image_tfm_flip if do_flip else self._image_tfm_normal
        return tfm(img)   # [3, H, W] in [-1, 1]

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self._index)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, int] | dict:
        """Load one sample.

        Returns (latent, label) if mask_root is None (backward compat).
        Returns dict with latent/label/filename/mask_2d/mask_valid/image_3ch/image_valid/patch_tok.
        All tensor fields use zeros sentinel + bool valid for default_collate compatibility.
        """
        sf_idx, local_i = self._index[idx]
        shard_path = self._shard_files[sf_idx]

        # Per-sample random hflip decision (synchronized across lat+mask+image).
        do_flip = self._use_hflip and (torch.rand(1).item() < 0.5)

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            # use_flip: static legacy flag for always-flip. use_hflip: random per-sample.
            lat_key = "latents_flip" if (self._use_flip or do_flip) else "latents"
            latent = f.get_slice(lat_key)[local_i]          # [C, H, W]
            label = int(f.get_slice("labels")[local_i].item())

        latent = latent.float()
        if self._latent_norm and self._latent_mean is not None:
            # Reshape stats to [C, 1, 1] for channel-wise broadcast over [C, H, W] latent.
            # .squeeze() was wrong: collapses [1,C,1,1] → [C] which fails broadcast vs [C,H,W].
            c = latent.shape[0]
            mean = self._latent_mean.view(c, 1, 1)   # [C, 1, 1]
            std = self._latent_std.view(c, 1, 1)     # [C, 1, 1]
            latent = (latent - mean) / std            # [C, H, W]
            if self._latent_multiplier != 1.0:
                latent = latent * self._latent_multiplier

        if self._mask_root is None:
            return latent, label

        # Resolve filename from sidecar ("" sentinel if no sidecar — collation-safe).
        shard_fns = self._shard_filenames[sf_idx]
        filename: str = shard_fns[local_i] if shard_fns is not None else ""

        # Load and resize mask; zeros sentinel if not found (collation-safe).
        # F2 fast-path: skip NFS lookup+load when mask_required=False (warmup phase).
        mask_2d = torch.zeros(self._image_size, self._image_size, dtype=torch.float32)
        mask_valid = False
        if self._mask_required and filename:
            npz_path = _find_mask_npz(self._mask_root, filename)
            if npz_path is not None:
                mask_2d = _load_mask_resize(npz_path, self._image_size)  # [H, W] float32
                mask_valid = True
        # Synchronized hflip for mask (latent already loaded from latents_flip key above).
        if do_flip and mask_valid:
            mask_2d = mask_2d.flip(-1)                 # [H, W] horizontal flip

        # F2 fast-path: skip canny when canny_required=False (warmup/mask phase).
        if self._canny_required:
            canny_2d, canny_valid = _load_canny_from_index(
                self._canny_index, filename, self._image_size)
        else:
            canny_2d = torch.zeros(self._image_size, self._image_size, dtype=torch.float32)
            canny_valid = False
        if do_flip and canny_valid:
            canny_2d = canny_2d.flip(-1)

        # Load RGB image for patch conditioning; zeros sentinel if unavailable.
        image_3ch = torch.zeros(3, self._image_size, self._image_size, dtype=torch.float32)
        image_valid = False
        if self._load_image and self._imagenet_root is not None and filename:
            jpeg_path = self._imagenet_root / "train" / filename
            if jpeg_path.exists():
                try:
                    image_3ch = self._load_image_3ch(jpeg_path, do_flip)
                    image_valid = True
                except Exception:
                    pass  # sentinel already set

        # Random patch token (from flipped-image token grid; no idx flip needed — consistent).
        patch_tok = int(
            torch.randint(0, _TOKEN_GRID, (1,)).item() * _TOKEN_GRID
            + torch.randint(0, _TOKEN_GRID, (1,)).item()
        )

        return {
            "latent": latent,
            "label": label,
            "filename": filename,              # "" if no sidecar
            "mask_2d": mask_2d,                # zeros if mask not found
            "mask_valid": mask_valid,           # True only when real mask loaded
            "image_3ch": image_3ch,            # zeros if not loaded
            "image_valid": image_valid,         # True only when real image loaded
            "patch_tok": patch_tok,
            "canny_2d": canny_2d,               # zeros if canny not loaded
            "canny_valid": canny_valid,          # True only when real canny loaded
        }


# ── Smoke test ────────────────────────────────────────────────────────────────

