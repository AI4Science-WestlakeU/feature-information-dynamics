"""Sample-ID noise independent of evaluation batch size and sharding."""

import hashlib
import re

import numpy as np


def stable_noise(ids: list[str], dimension: int, seed: int) -> np.ndarray:
    """Generate PCG64 float32 Gaussian vectors, fixed for each ID and stream."""
    if dimension < 1:
        raise ValueError("Noise dimension must be positive")
    result = []
    for sample_id in ids:
        digest = hashlib.sha256(f"fid-noise-v1:{seed}:{sample_id}".encode()).digest()
        rng = np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], "little")))
        result.append(rng.standard_normal(dimension).astype(np.float32))
    return np.stack(result)


TORCH_RANK_ALGORITHMS = {
    "torch_rank_seed_plus_100003_v1",  # Pixel evaluator
    "torch_rank_polynomial_seed_v1",   # latent evaluators / eval_lib_mmse.py
}


def validate_noise_identity(observation: dict) -> None:
    """Validate the declared generator; legacy Torch draws depend on sharding.

    Torch/CUDA version, ordered rank shards, tensor shape and RNG consumption
    belong to the execution identity. A numeric seed alone cannot reproduce them.
    """
    algorithm = observation.get("noise_algorithm")
    if algorithm == "sha256_sample_id_pcg64_float64_to_float32_v1":
        return
    if algorithm not in TORCH_RANK_ALGORITHMS:
        raise ValueError("Unknown declared noise algorithm")
    identity = observation.get("noise_execution", {})
    world_size = identity.get("world_size")
    if type(world_size) is not int or world_size < 1:
        raise ValueError("Torch noise requires a positive world_size")
    if identity.get("ranks") != list(range(world_size)):
        raise ValueError("Torch noise requires every participating rank")
    for key in ("ordered_rank_samples_sha256", "rng_draw_schedule_sha256"):
        if not isinstance(identity.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", identity[key]) is None:
            raise ValueError(f"Torch noise requires {key}")
    for key in ("torch_version", "device", "dtype"):
        if not isinstance(identity.get(key), str) or not identity[key].strip():
            raise ValueError(f"Torch noise requires {key}")


def rank_noise_seed(seed: int, rank: int, algorithm: str, t_idx: int = 0,
                    sample_idx: int = 0) -> int:
    """Exact did_q01 seed rules; draw tensors using the recorded Torch runtime."""
    if any(type(v) is not int or v < 0 for v in (seed, rank, t_idx, sample_idx)):
        raise ValueError("Seed and rank/grid/sample indices must be nonnegative integers")
    if algorithm == "torch_rank_seed_plus_100003_v1":
        return seed + rank * 100003
    if algorithm == "torch_rank_polynomial_seed_v1":
        return (seed * 2654435761 + rank * 40503 + t_idx * 12347 + sample_idx) & ((1 << 63) - 1)
    raise ValueError("Unknown Torch rank seed rule")
