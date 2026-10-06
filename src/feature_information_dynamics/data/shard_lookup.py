from pathlib import Path
from typing import Optional
import json
import logging
import os
import torch
from safetensors import safe_open
StemLookup = dict[str, tuple[str, int]]
log = logging.getLogger(__name__)

def _build_canny_shard_lookup(canny_shard_dir: Path) -> StemLookup:
    """Build filename → (safetensors_path, slot_idx) from canny JSON sidecars.

    Canny shards: cannys_rank{RR}_shard{SSS}.json  {"filenames": ["synset/stem.JPEG", ...]}

    Args:
        canny_shard_dir: Directory containing ``cannys_rank*_shard*.json`` files.

    Returns:
        Dict mapping ``"synset/stem.JPEG"`` to ``(shard_path, slot_idx)``.

    Raises:
        FileNotFoundError: If no canny JSON sidecars found.
    """
    json_paths = sorted(canny_shard_dir.glob('cannys_rank*_shard*.json'))
    if not json_paths:
        raise FileNotFoundError(f'No cannys_rank*_shard*.json sidecar files found in {canny_shard_dir}')
    lookup: StemLookup = {}
    for json_path in json_paths:
        safetensors_path = str(json_path.with_suffix('.safetensors'))
        with open(json_path) as fh:
            data = json.load(fh)
        filenames: list[str] = data['filenames']
        for slot_idx, fn in enumerate(filenames):
            if fn not in lookup:
                lookup[fn] = (safetensors_path, slot_idx)
    return lookup

def _derive_mask_lookup(canny_lookup: StemLookup, mask_shard_dir: Path) -> StemLookup:
    """Derive mask lookup from canny lookup by rewriting shard filenames.

    Mask shards have no JSON sidecar; they are 1:1 aligned with canny shards.
    Replace ``cannys_`` → ``masks_`` in the shard path.

    Args:
        canny_lookup:   Stem → (canny_shard_path, slot_idx) mapping.
        mask_shard_dir: Directory containing mask safetensors shards.

    Returns:
        Stem → (mask_shard_path, slot_idx) mapping.

    Raises:
        FileNotFoundError: If any derived mask shard path does not exist.
    """
    mask_lookup: StemLookup = {}
    missing: list[str] = []
    for fn, (canny_path, slot_idx) in canny_lookup.items():
        canny_name = Path(canny_path).name
        mask_name = canny_name.replace('cannys_', 'masks_', 1)
        mask_path = str(mask_shard_dir / mask_name)
        if not os.path.exists(mask_path):
            missing.append(mask_path)
            continue
        mask_lookup[fn] = (mask_path, slot_idx)
    if missing:
        unique_missing = sorted(set(missing))
        raise FileNotFoundError(f"Mask shards not found ({len(unique_missing)} path(s)): {unique_missing[:3]}{('...' if len(unique_missing) > 3 else '')}")
    return mask_lookup

def build_mask_canny_shard_lookups(mask_shard_dir: str, canny_shard_dir: str) -> tuple[StemLookup, StemLookup]:
    """Build both mask and canny lookups from canny sidecars.

    Args:
        mask_shard_dir:  Directory with masks_rank*_shard*.safetensors.
        canny_shard_dir: Directory with cannys_rank*_shard*.json + .safetensors.

    Returns:
        (mask_lookup, canny_lookup) both mapping
        ``"synset/stem.JPEG"`` to ``(shard_path, slot_idx)``.
    """
    canny_lookup = _build_canny_shard_lookup(Path(canny_shard_dir))
    mask_lookup = _derive_mask_lookup(canny_lookup, Path(mask_shard_dir))
    return (mask_lookup, canny_lookup)

def load_mask_from_shard(mask_lookup: StemLookup, filename: str) -> tuple[torch.Tensor, bool]:
    """Load one mask slice from sharded safetensors.

    Args:
        mask_lookup: Stem → (shard_path, slot_idx) mapping.
        filename:    Canonical ``"synset/stem.JPEG"`` key.

    Returns:
        (mask [256, 256] float32 in [0,1], valid bool).
        Returns (zeros, False) when filename not in lookup.
    """
    if filename not in mask_lookup:
        return (torch.zeros(256, 256, dtype=torch.float32), False)
    shard_path, slot_idx = mask_lookup[filename]
    with safe_open(shard_path, framework='pt', device='cpu') as f:
        mask_u8 = f.get_slice('masks')[slot_idx]
    return (mask_u8.float(), True)

def load_canny_from_shard(canny_lookup: StemLookup, filename: str) -> tuple[torch.Tensor, bool]:
    """Load one canny slice from sharded safetensors.

    Args:
        canny_lookup: Stem → (shard_path, slot_idx) mapping.
        filename:     Canonical ``"synset/stem.JPEG"`` key.

    Returns:
        (canny [256, 256] float32 in [0,1], valid bool).
        Returns (zeros, False) when filename not in lookup.
    """
    if filename not in canny_lookup:
        return (torch.zeros(256, 256, dtype=torch.float32), False)
    shard_path, slot_idx = canny_lookup[filename]
    with safe_open(shard_path, framework='pt', device='cpu') as f:
        canny_u8 = f.get_slice('cannys')[slot_idx]
    return (canny_u8.float(), True)

def _rel_path(abs_path: str, root: str) -> str:
    """Extract 'synset/filename.JPEG' relative path from absolute path.

    Args:
        abs_path: Absolute path to image file.
        root:     Root directory to compute relative path from.

    Returns:
        Relative path string ``synset/filename.JPEG``.
    """
    try:
        return str(Path(abs_path).relative_to(root))
    except ValueError:
        return Path(abs_path).name
def build_mask_shard_lookup(mask_shard_dir: str, canny_shard_dir: str) -> StemLookup:
    """Build canonical mask lookup from canny sidecars + mask shard dir.

    Args:
        mask_shard_dir:  Directory with masks_rank*_shard*.safetensors.
        canny_shard_dir: Directory with cannys_rank*_shard*.json + .safetensors.

    Returns:
        Dict mapping ``"synset/stem.JPEG"`` to ``(mask_shard_path, slot_idx)``.
    """
    canny_lookup = _build_canny_shard_lookup(Path(canny_shard_dir))
    mask_lookup = _derive_mask_lookup(canny_lookup, Path(mask_shard_dir))
    return mask_lookup
