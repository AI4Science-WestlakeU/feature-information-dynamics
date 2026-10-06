"""Raw image dataset paired with sharded mask + canny safetensors.

For RAE and Pixel reps that operate on raw images (not latents), this dataset
provides the mask + canny conditioning IO acceleration via sharded safetensors
mmap while loading raw images from an ImageFolder-style ImageNet directory.

Shard layout (produced by build_mask_shards.py / build_canny_shards.py):
    canny_shards/:
        cannys_rank{RR}_shard{SSS}.safetensors  keys: cannys [N,256,256] uint8,
                                                       canny_valid [N] uint8
        cannys_rank{RR}_shard{SSS}.json         {"filenames": ["synset/stem.JPEG", ...]}

    mask_shards/:
        masks_rank{RR}_shard{SSS}.safetensors   keys: masks [N,256,256] uint8,
                                                       mask_valid [N] uint8
        (no JSON sidecar — filename order is identical to canny sidecar)

Usage::
    from feature_information_dynamics.data.raw_image_mask_canny_dataset import RawImageMaskCannyDataset
    ds = RawImageMaskCannyDataset(
        imagenet_root="DATA/imagenet",
        mask_shard_dir="DATA/mask_shards",
        canny_shard_dir="DATA/canny_shards",
        paired_whitelist_json="DATA/paired_train_whitelist.json",
        is_val=False,
    )
    image, mask, canny, label = ds[0]
    # image: [3,256,256] float32 in [-1,1]
    # mask:  [256,256]   float32 in [0,1]
    # canny: [256,256]   float32 in [0,1]
    # label: int in [0,999]
"""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Optional

import torch
import torchvision.transforms as T
from PIL import Image
from safetensors import safe_open
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

# Image normalization: mean=0.5, std=0.5 per channel → maps [0,1] to [-1,1]
_NORM_MEAN = (0.5, 0.5, 0.5)
_NORM_STD = (0.5, 0.5, 0.5)


# ── Lookup table type ─────────────────────────────────────────────────────────

# stem = "synset/filename.JPEG" (canonical form, no leading slash)
# StemLookup: stem → (shard_path_str, slot_idx)
StemLookup = dict[str, tuple[str, int]]


# ── Transform builder ─────────────────────────────────────────────────────────

def _build_image_transform(image_size: int) -> T.Compose:
    """Build center-crop + ToTensor + normalize pipeline.

    Preserves the executed historical transform:
      - Resize the shorter edge using torchvision's default bilinear
        interpolation, then center crop. This is not the ADM crop helper.
      - ToTensor: [H,W,3] uint8 → [3,H,W] float32 in [0,1]
      - Normalize: [0,1] → [-1,1] with mean=std=0.5 per channel

    Args:
        image_size: Target spatial size after crop (e.g. 256).

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    return T.Compose([
        T.Resize(image_size),          # scale shorter edge with default bilinear interpolation
        T.CenterCrop(image_size),      # center crop to image_size × image_size
        T.ToTensor(),                  # [H,W,3] uint8 → [3,H,W] float32 in [0,1]
        T.Normalize(_NORM_MEAN, _NORM_STD),  # [0,1] → [-1,1]
    ])


# ── Shard index builder ───────────────────────────────────────────────────────

def _build_shard_lookup(shard_dir: Path, prefix: str) -> StemLookup:
    """Build stem → (shard_path, slot_idx) lookup from canny sidecar JSONs.

    Iterates over all ``{prefix}_rank*_shard*.json`` files in shard_dir.
    Each sidecar JSON has the schema ``{"filenames": ["synset/stem.JPEG", ...]}``.

    Args:
        shard_dir: Directory containing sidecar ``.json`` files.
        prefix:    Filename prefix, e.g. ``"cannys"`` or ``"masks"``.

    Returns:
        Dict mapping canonical ``"synset/stem.JPEG"`` to
        ``(safetensors_path_str, slot_idx)``.

    Raises:
        FileNotFoundError: If no sidecar JSON files are found.
    """
    json_paths = sorted(shard_dir.glob(f"{prefix}_rank*_shard*.json"))
    if not json_paths:
        raise FileNotFoundError(
            f"No {prefix}_rank*_shard*.json sidecar files found in {shard_dir}"
        )

    lookup: StemLookup = {}
    for json_path in json_paths:
        safetensors_path = str(json_path.with_suffix(".safetensors"))
        with open(json_path) as fh:
            data = json.load(fh)
        filenames: list[str] = data["filenames"]
        for slot_idx, fn in enumerate(filenames):
            # Deduplicate: first occurrence wins (matches WhitelistedImgLatentDataset)
            if fn not in lookup:
                lookup[fn] = (safetensors_path, slot_idx)

    log.info(
        "[_build_shard_lookup] prefix=%s  shards=%d  unique_stems=%d",
        prefix, len(json_paths), len(lookup),
    )
    return lookup


def _build_mask_lookup_from_canny(
    canny_lookup: StemLookup,
    mask_shard_dir: Path,
) -> StemLookup:
    """Derive mask lookup from canny lookup by rewriting shard filenames.

    Mask shards have no JSON sidecar but are 1:1 aligned with canny shards
    (same rank/shard numbering, same slot ordering).  Replace ``cannys_`` with
    ``masks_`` in the shard path to get the corresponding mask shard.

    Args:
        canny_lookup: Stem → (canny_shard_path, slot_idx) mapping.
        mask_shard_dir: Directory containing mask safetensors shards.

    Returns:
        Stem → (mask_shard_path, slot_idx) mapping.

    Raises:
        FileNotFoundError: If any derived mask shard path does not exist.
    """
    mask_lookup: StemLookup = {}
    missing: list[str] = []

    for fn, (canny_path, slot_idx) in canny_lookup.items():
        canny_name = Path(canny_path).name           # cannys_rank00_shard000.safetensors
        mask_name = canny_name.replace("cannys_", "masks_", 1)
        mask_path = str(mask_shard_dir / mask_name)
        if not os.path.exists(mask_path):
            missing.append(mask_path)
            continue
        mask_lookup[fn] = (mask_path, slot_idx)

    if missing:
        unique_missing = sorted(set(missing))
        raise FileNotFoundError(
            f"Mask shards not found for {len(unique_missing)} path(s): "
            f"{unique_missing[:3]}{'...' if len(unique_missing) > 3 else ''}"
        )

    log.info(
        "[_build_mask_lookup_from_canny] derived %d mask shard entries",
        len(mask_lookup),
    )
    return mask_lookup


# ── Label table builder ───────────────────────────────────────────────────────

def _build_synset_to_class(imagenet_root: str) -> dict[str, int]:
    """Build synset → ImageNet canonical class index (0-999).

    Mirrors torchvision.ImageFolder: alphabetical sort of all synset folders
    under ``{imagenet_root}/train/``.  The full 1000-class ordering is used so
    class indices are compatible with pretrained embeddings.

    Args:
        imagenet_root: Path to ImageNet root (contains ``train/`` subdirectory).

    Returns:
        Dict mapping synset string to int index in [0, 999].
    """
    train_root = os.path.join(imagenet_root, "train")
    all_synsets: list[str] = sorted(
        d for d in os.listdir(train_root)
        if os.path.isdir(os.path.join(train_root, d))
    )
    return {s: i for i, s in enumerate(all_synsets)}


# ── Dataset ───────────────────────────────────────────────────────────────────

class RawImageMaskCannyDataset(Dataset):
    """Raw image dataset paired with sharded mask + canny conditioning.

    Loads raw images from an ImageFolder-style ImageNet directory and paired
    mask / canny tensors from sharded safetensors via mmap per ``__getitem__``.
    Only entries present in both mask AND canny shards are returned.

    Returns:
        image: [3, 256, 256] float32 tensor in [-1, 1]
        mask:  [256, 256]    float32 tensor in [0, 1]  (zeros if mask_valid==0)
        canny: [256, 256]    float32 tensor in [0, 1]  (zeros if canny_valid==0)
        label: int class index in [0, 999]
    """

    def __init__(
        self,
        imagenet_root: str,
        mask_shard_dir: str,
        canny_shard_dir: str,
        paired_whitelist_json: str,
        image_size: int = 256,
        is_val: bool = False,
        load_mask: bool = True,
        load_canny: bool = True,
    ) -> None:
        """Initialize dataset.

        Args:
            imagenet_root:        Path to ImageNet root (contains ``train/``).
            mask_shard_dir:       Directory containing ``masks_rank*_shard*.safetensors``.
            canny_shard_dir:      Directory containing ``cannys_rank*_shard*.safetensors``
                                  and ``cannys_rank*_shard*.json`` sidecar files.
            paired_whitelist_json: Path to JSON ``{"filenames": ["synset/stem.JPEG", ...]}``.
            image_size:           Spatial resolution after center crop. Default 256.
            is_val:               If True, disable random hflip for deterministic eval.
            load_mask:            Read mask tensors.  If False, return an all-zero
                                  placeholder of the same shape.
            load_canny:           Read canny tensors.  If False, return an all-zero
                                  placeholder of the same shape.
        """
        super().__init__()
        self._imagenet_root = Path(imagenet_root)
        self._image_size = image_size
        self._is_val = is_val
        self._load_mask_enabled = load_mask
        self._load_canny_enabled = load_canny
        self._transform = _build_image_transform(image_size)

        # Step 1: load whitelist
        with open(paired_whitelist_json) as fh:
            wl_data = json.load(fh)
        whitelist_fns: list[str] = wl_data["filenames"]
        whitelist_set: set[str] = set(whitelist_fns)

        # Step 2: build canny lookup (has sidecar JSONs)
        canny_lookup = _build_shard_lookup(
            Path(canny_shard_dir), prefix="cannys"
        )

        # Step 3: derive mask lookup (no sidecar; same slot order as canny)
        mask_lookup = _build_mask_lookup_from_canny(
            canny_lookup, Path(mask_shard_dir)
        )

        # Step 4: intersect whitelist with entries in both lookups
        self._whitelist_filtered: list[str] = [
            fn for fn in whitelist_fns
            if fn in whitelist_set
            and fn in canny_lookup
            and fn in mask_lookup
        ]

        # Step 5: store lookups (filtered to whitelist members only for memory)
        self._canny_lookup: StemLookup = {
            fn: canny_lookup[fn] for fn in self._whitelist_filtered
        }
        self._mask_lookup: StemLookup = {
            fn: mask_lookup[fn] for fn in self._whitelist_filtered
        }

        # Step 6: build label table
        self._synset_to_class: dict[str, int] = _build_synset_to_class(imagenet_root)

        log.info(
            "[RawImageMaskCannyDataset] whitelist=%d  paired=%d  is_val=%s",
            len(whitelist_fns), len(self._whitelist_filtered), is_val,
        )

    def __len__(self) -> int:
        """Return number of samples in the filtered dataset."""
        return len(self._whitelist_filtered)

    def _load_mask(self, fn: str) -> torch.Tensor:
        """Load one mask slice from sharded safetensors.

        Args:
            fn: Canonical filename ``synset/stem.JPEG``.

        Returns:
            [256, 256] float32 tensor in [0, 1].
        """
        shard_path, slot_idx = self._mask_lookup[fn]
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            mask_u8 = f.get_slice("masks")[slot_idx]   # [256, 256] uint8 (0/1 binary)
        return mask_u8.float()                          # [256, 256] float32 in [0, 1]

    def _load_canny(self, fn: str) -> torch.Tensor:
        """Load one canny slice from sharded safetensors.

        Args:
            fn: Canonical filename ``synset/stem.JPEG``.

        Returns:
            [256, 256] float32 tensor in [0, 1].
        """
        shard_path, slot_idx = self._canny_lookup[fn]
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            canny_u8 = f.get_slice("cannys")[slot_idx]  # [256, 256] uint8 (0/1 binary)
        return canny_u8.float()                          # [256, 256] float32 in [0, 1]

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Return one (image, mask, canny, label) sample.

        Applies synchronized random horizontal flip (train only) so that
        image, mask, and canny remain spatially aligned.

        Args:
            idx: Dataset index in [0, len(self)).

        Returns:
            image: [3, H, W] float32 in [-1, 1]
            mask:  [H, W]    float32 in [0, 1]
            canny: [H, W]    float32 in [0, 1]
            label: int class index in [0, 999]
        """
        fn = self._whitelist_filtered[idx]   # "synset/stem.JPEG"

        # Load raw image and apply transform
        img_path = self._imagenet_root / "train" / fn
        pil_img = Image.open(img_path).convert("RGB")
        image: torch.Tensor = self._transform(pil_img)   # [3, H, W] float32 in [-1,1]

        # Do not mmap condition tensors which this training phase never uses.
        # The all-zero placeholders preserve the batch interface exactly.
        mask = (
            self._load_mask(fn)
            if self._load_mask_enabled
            else torch.zeros((self._image_size, self._image_size), dtype=torch.float32)
        )
        canny = (
            self._load_canny(fn)
            if self._load_canny_enabled
            else torch.zeros((self._image_size, self._image_size), dtype=torch.float32)
        )

        # Synchronized horizontal flip (train only)
        if not self._is_val and random.random() < 0.5:
            image = torch.flip(image, dims=[-1])   # [3, H, W] hflip
            mask = torch.flip(mask, dims=[-1])     # [H, W] hflip
            canny = torch.flip(canny, dims=[-1])   # [H, W] hflip

        # Class label: synset is first path component
        synset = fn.split("/")[0]
        label: int = self._synset_to_class[synset]

        return image, mask, canny, label


# ── Self-test ─────────────────────────────────────────────────────────────────

