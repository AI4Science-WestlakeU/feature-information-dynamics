"""Download and validate official MNIST IDX archives without torchvision."""

from __future__ import annotations

import gzip
import hashlib
import struct
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from .bundle import sha256

ARCHIVES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def load_mnist(root: Path, download: bool) -> tuple[dict, dict]:
    """Read exact official MNIST files and return source hashes."""
    root.mkdir(parents=True, exist_ok=True)
    arrays, hashes = {}, {}
    for filename, expected in ARCHIVES.items():
        path = root / filename
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"Missing {path}; pass --download")
            with urlopen(f"https://ossci-datasets.s3.amazonaws.com/mnist/{filename}", timeout=60) as response:
                payload = response.read()
            if hashlib.md5(payload).hexdigest() != expected:
                raise ValueError(f"Download checksum mismatch: {filename}")
            temporary = path.with_suffix(path.suffix + ".part")
            temporary.write_bytes(payload)
            temporary.replace(path)
        compressed = path.read_bytes()
        if hashlib.md5(compressed).hexdigest() != expected:
            raise ValueError(f"MNIST source checksum mismatch: {filename}")
        hashes[filename] = sha256(path)
        payload = gzip.decompress(compressed)
        magic, count = struct.unpack(">II", payload[:8])
        if "images" in filename:
            if magic != 2051 or struct.unpack(">II", payload[8:16]) != (28, 28):
                raise ValueError("Invalid image IDX header")
            arrays[filename] = np.frombuffer(payload[16:], np.uint8).reshape(count, 784).astype(np.float32) / 127.5 - 1
        else:
            if magic != 2049:
                raise ValueError("Invalid label IDX header")
            arrays[filename] = np.frombuffer(payload[8:], np.uint8).reshape(count).astype(np.int64)
    return arrays, hashes


def balanced_indices(labels: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Select balanced classes in reproducible class-major order."""
    if count <= 0 or count % 10:
        raise ValueError("MNIST split sizes must be positive multiples of ten")
    rng = np.random.default_rng(seed)
    groups = [rng.permutation(np.flatnonzero(labels == k)) for k in range(10)]
    if any(len(group) < count // 10 for group in groups):
        raise ValueError("Not enough examples in one class")
    return np.concatenate([group[:count // 10] for group in groups])
