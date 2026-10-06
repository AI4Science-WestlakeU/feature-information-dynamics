"""Inventory an existing prepared-data directory; never extract or move data."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from feature_information_dynamics.data.releases import _semantic


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as incoming:
        for block in iter(lambda: incoming.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def build(data_root, release_id, output):
    root = Path(data_root).resolve()
    resources = {
        "mask_shard_dir": "mask_shards", "canny_shard_dir": "canny_shards",
        "paired_train_whitelist": "paired/paired_train_whitelist.json",
        "paired_val_whitelist": "paired/paired_val_whitelist.json",
        "class_ids": "class_ids.json",
    }
    data = {"schema_version": 1, "release_id": release_id, "resources": resources,
            "validation": {}, "files": []}
    common = [root / resources[key] for key in
              ["paired_train_whitelist", "paired_val_whitelist", "class_ids"]]
    if (root / "sr95.json").exists():
        common.append(root / "sr95.json")
    for folder, pattern in [("mask_shards", "masks_rank*_shard*.safetensors"),
                            ("canny_shards", "cannys_rank*_shard*.safetensors"),
                            ("canny_shards", "cannys_rank*_shard*.json")]:
        matched = sorted((root / folder).glob(pattern))
        if not matched:
            raise ValueError("No prepared files match " + folder + "/" + pattern)
        common.extend(matched)
    data["validation"] = _semantic(data, root)
    groups = [("common", common)]
    latents = root / "vavae_latents"
    if latents.exists():
        tensor_files = sorted(latents.glob("latents_rank*_shard*.safetensors"))
        if not tensor_files:
            raise ValueError("VAVAE directory contains no latent shards")
        latent_files = [*tensor_files, *(p.with_suffix(".json") for p in tensor_files),
                        latents / "latents_stats.pt"]
        groups.append(("vavae", latent_files))
        resources["vavae_latents"] = "vavae_latents"
    for component, files in groups:
        for path in files:
            if not path.resolve().is_relative_to(root) or not path.is_file():
                raise ValueError("Missing or escaping release file: " + str(path))
            relative = path.relative_to(root).as_posix()
            data["files"].append({"path": relative, "source_path": relative,
                                  "component": component, "size_bytes": path.stat().st_size,
                                  "sha256": digest(path)})
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    result = build(arguments.data_root, arguments.release_id, arguments.output)
    print(json.dumps({"release_id": result["release_id"], "files": len(result["files"]),
                      "validation": result["validation"]}, indent=2))
