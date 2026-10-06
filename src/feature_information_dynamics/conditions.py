"""Portable deterministic RGB/mask/masked-Canny preparation; no model inference."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .bundle import sha256, write_json
from .suite import fingerprint


def center_crop(array, nearest=False):
    """Follow the 256-pixel ImageNet crop geometry, preserving binary conditions."""
    picture = Image.fromarray(array)
    while min(picture.size) >= 512:
        picture = picture.resize(tuple(v//2 for v in picture.size),
                                 Image.Resampling.NEAREST if nearest else Image.Resampling.BOX)
    scale = 256/min(picture.size)
    picture = picture.resize(tuple(round(v*scale) for v in picture.size),
                             Image.Resampling.NEAREST if nearest else Image.Resampling.BICUBIC)
    value = np.asarray(picture)
    y, x = (value.shape[0]-256)//2, (value.shape[1]-256)//2
    return value[y:y+256, x:x+256]


def prepare(manifest, output):
    """Save numeric condition artifacts; reject missing/corrupt masks explicitly."""
    if output.exists():
        raise FileExistsError(output)
    records = json.loads(manifest.read_text(encoding="utf-8"))
    if not records or len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("Need nonempty unique sample IDs")
    output.mkdir(parents=True)
    rows = []
    for i, record in enumerate(records):
        rgb_path, mask_path = Path(record["rgb_path"]), Path(record["mask_path"])
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        with np.load(mask_path, allow_pickle=False) as data:
            if "mask" in data:
                mask = np.asarray(data["mask"], dtype=bool)
            elif "masks" in data:
                mask = np.asarray(data["masks"], dtype=bool)
                if mask.ndim == 3:
                    mask = mask.any(axis=0)
            else:
                raise ValueError(f"Missing mask array for {record['sample_id']}")
        if mask.ndim != 2 or min(mask.shape) < 1:
            raise ValueError(f"Invalid mask for {record['sample_id']}")
        resized = mask.shape != rgb.shape[:2]
        if resized:
            mask = np.asarray(Image.fromarray(mask.astype(np.uint8)*255).resize(
                (rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST)) > 0
        masked_rgb = rgb.copy()
        masked_rgb[~mask] = 0
        canny = cv2.Canny(cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2GRAY), 100, 200)
        relative = f"sample_{i:06d}.npz"
        rgb_crop = center_crop(rgb)
        mask_crop = center_crop(mask.astype(np.uint8), nearest=True)
        canny_crop = (center_crop(canny, nearest=True) > 0).astype(np.uint8)
        np.savez_compressed(output/relative, rgb=rgb_crop, mask=mask_crop, masked_canny=canny_crop)
        rows.append({"sample_id": record["sample_id"], "class_id": record["class_id"],
                     "rgb_sha256": sha256(rgb_path), "mask_source_sha256": sha256(mask_path),
                     "mask_sha256": fingerprint(mask_crop.tolist()),
                     "masked_canny_sha256": fingerprint(canny_crop.tolist()),
                     "artifact": relative, "artifact_sha256": sha256(output/relative),
                     "mask_resized_to_raw_rgb": resized, "mask_empty": not bool(mask.any()), "mask_source": record.get("mask_source", "provided")})
    write_json(output/"samples.json", rows)
    report = {"samples": len(rows), "samples_sha256": fingerprint(rows),
              "definition": "raw_rgb_masked_then_gray_canny_100_200_then_nearest_center_crop_256",
              "mask_fallback": "error; never substitute all-ones mask",
              "opencv": cv2.__version__, "numpy": np.__version__, "code_sha256": sha256(Path(__file__))}
    write_json(output/"preparation.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest, args.output)))
