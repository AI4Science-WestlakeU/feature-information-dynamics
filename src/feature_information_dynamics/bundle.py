"""Portable, hash-checked dense measurement bundles (no pickle)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Write readable JSON and reject non-finite numbers."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def write_bundle(
    directory: Path, errors: np.ndarray, metadata: dict, samples: list[dict]
) -> None:
    """Persist errors with axes [subset, snr, noise_repeat, sample]."""
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty bundle: {directory}")
    validate(errors, metadata, samples)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "errors.npy", errors, allow_pickle=False)
    write_json(directory / "metadata.json", metadata)
    write_json(directory / "samples.json", samples)
    files = {name: {"sha256": sha256(directory / name),
                    "bytes": (directory / name).stat().st_size}
             for name in ("errors.npy", "metadata.json", "samples.json")}
    write_json(directory / "bundle.json", {"schema_version": "0.1", "files": files})


def validate(errors: np.ndarray, metadata: dict, samples: list[dict]) -> None:
    """Check paired axes, complete conditions, units, and finite raw errors."""
    features = metadata["features"]
    subsets = metadata["subsets"]
    if not features or len(features) != len(set(features)):
        raise ValueError("Features must be unique and nonempty")
    keys = [frozenset(s) for s in subsets]
    expected = {frozenset(f for i, f in enumerate(features) if bit & (1 << i))
                for bit in range(2 ** len(features))}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Need every condition subset exactly once")
    if any(len(s) != len(set(s)) for s in subsets):
        raise ValueError("Duplicate feature in a subset")
    grid = np.asarray(metadata["log10_snr"], dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all():
        raise ValueError("Need at least two finite SNR coordinates")
    if not np.all(np.diff(grid) > 0):
        raise ValueError("SNR grid must be strictly increasing")
    if metadata.get("error_reduction") != "sum" or metadata["target_dimension"] <= 0:
        raise ValueError("Information analysis requires vector squared-error sums")
    ids = [s["sample_id"] for s in samples]
    if len(ids) < 2 or len(ids) != len(set(ids)):
        raise ValueError("Need unique paired sample IDs")
    if errors.ndim != 4 or errors.shape[0:2] != (len(keys), len(grid)):
        raise ValueError("Expected [subset, snr, noise_repeat, sample]")
    if errors.shape[2] < 1 or errors.shape[3] != len(ids):
        raise ValueError("Error axes do not match the sample manifest")
    if errors.dtype.kind != "f" or not np.isfinite(errors).all() or (errors < 0).any():
        raise ValueError("Raw squared errors must be finite nonnegative floats")


def load_bundle(directory: Path) -> tuple[np.ndarray, dict, list[dict]]:
    """Verify all files before loading measurements."""
    manifest = json.loads((directory / "bundle.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != "0.1":
        raise ValueError("Unsupported bundle schema")
    required = {"errors.npy", "metadata.json", "samples.json"}
    if set(manifest["files"]) != required:
        raise ValueError("Unexpected or missing bundle files")
    for name, record in manifest["files"].items():
        path = directory / name
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Integrity check failed: {name}")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    samples = json.loads((directory / "samples.json").read_text(encoding="utf-8"))
    errors = np.load(directory / "errors.npy", allow_pickle=False)
    validate(errors, metadata, samples)
    return errors, metadata, samples
