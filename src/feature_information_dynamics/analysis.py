"""Exact condition decompositions and finite log-SNR integration."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .bundle import load_bundle, write_json


def weights(features: list[str], subsets: list[list[str]]) -> tuple[np.ndarray, np.ndarray]:
    """Return classical-chain and exact Shapley linear risk contrasts."""
    lookup = {frozenset(s): i for i, s in enumerate(subsets)}
    count = len(features)
    chain = np.zeros((count, len(subsets)))
    shapley = np.zeros_like(chain)
    previous = frozenset()
    for i, feature in enumerate(features):
        following = previous | {feature}
        chain[i, lookup[previous]], chain[i, lookup[following]] = 1, -1
        previous = following
        for subset, index in lookup.items():
            if feature in subset:
                continue
            k = len(subset)
            weight = math.factorial(k) * math.factorial(count - k - 1) / math.factorial(count)
            shapley[i, index] += weight
            shapley[i, lookup[subset | {feature}]] -= weight
    return chain, shapley


def cumulative_integral(density: np.ndarray, log10_snr: np.ndarray) -> np.ndarray:
    """Integrate density per ln(gamma), in nats, on a finite log10 grid."""
    areas = (density[..., 1:] + density[..., :-1]) / 2
    areas = areas * np.diff(log10_snr) * np.log(10)
    return np.concatenate([np.zeros_like(density[..., :1]), np.cumsum(areas, axis=-1)], axis=-1)


def analyze(directory: Path, output: Path, bootstrap: int = 200) -> dict:
    """Analyze one complete bundle with paired, class-stratified bootstrap."""
    if bootstrap < 20:
        raise ValueError("Use at least 20 bootstrap draws")
    errors, meta, samples = load_bundle(directory)
    output.mkdir(parents=True, exist_ok=True)
    grid = np.asarray(meta["log10_snr"], dtype=np.float64)
    per_image = errors.astype(np.float64).mean(axis=2)
    risk = per_image.mean(axis=-1)
    chain, shapley = weights(meta["features"], meta["subsets"])
    factor = 0.5 * 10 ** grid
    labels = np.asarray([str(s.get("class_id", "all")) for s in samples])
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    rng = np.random.default_rng(20260910)
    result = {"representation": meta["representation"], "features": meta["features"],
              "log10_snr": grid.tolist(), "risk_sum": risk.tolist(),
              "interpretation": "Finite-estimator risk; not certified Bayes MMSE",
              "interval_scope": "Pointwise 95% paired class-stratified image bootstrap; fixed checkpoints and noise realizations; not simultaneous or training-seed uncertainty",
              "bootstrap_draws": bootstrap, "bootstrap_seed": 20260910,
              "finite_integration_bounds_log10_snr": [float(grid[0]), float(grid[-1])]}
    lookup = {frozenset(s): i for i, s in enumerate(meta["subsets"])}
    edges = []
    for subset, index in lookup.items():
        for f in meta["features"]:
            if f not in subset:
                gap = risk[index] - risk[lookup[subset | {f}]]
                edges.append({"base": sorted(subset), "added": f, "gap": gap.tolist(),
                              "negative_coordinates": np.flatnonzero(gap < 0).tolist()})
    result["edges"] = edges
    # Reuse bootstrap indices for all conditions and both decompositions.
    bootstrap_risk = np.stack([per_image[..., np.concatenate(
        [rng.choice(g, size=len(g), replace=True) for g in groups])].mean(axis=-1)
        for _ in range(bootstrap)])
    for name, matrix in (("chain", chain), ("shapley", shapley)):
        gap = matrix @ risk
        density = gap * factor
        boot_density = np.einsum("fs,bsg->bfg", matrix, bootstrap_risk) * factor
        interval = np.quantile(boot_density, [0.025, 0.975], axis=0)
        cumulative = cumulative_integral(density, grid)
        peak_index = np.argmax(density, axis=1)
        result[name] = {"gap": gap.tolist(), "density_per_ln_snr": density.tolist(),
                        "pointwise_lower": interval[0].tolist(),
                        "pointwise_upper": interval[1].tolist(),
                        "cumulative_nats": cumulative.tolist(),
                        "finite_integral_nats": cumulative[:, -1].tolist(),
                        "observed_peak_log10_snr": grid[peak_index].tolist(),
                        "peak_at_boundary": ((peak_index == 0) | (peak_index == len(grid)-1)).tolist()}
    total_gap = risk[lookup[frozenset()]] - risk[lookup[frozenset(meta["features"])]]
    result["shapley_efficiency_max_abs_error"] = float(
        np.max(np.abs((shapley @ risk).sum(axis=0) - total_gap)))
    write_json(output / "summary.json", result)
    plot(result, output / "information.png")
    return result


def plot(result: dict, path: Path) -> None:
    """Plot raw signed density, intervals, and finite cumulative integrals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = result["log10_snr"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for row, name in enumerate(("chain", "shapley")):
        for i, feature in enumerate(result["features"]):
            values = result[name]
            axes[row, 0].plot(grid, values["density_per_ln_snr"][i], "o-", label=feature)
            if "pointwise_lower" in values:
                axes[row, 0].fill_between(grid, values["pointwise_lower"][i], values["pointwise_upper"][i], alpha=0.15)
            axes[row, 1].plot(grid, values["cumulative_nats"][i], "o-", label=feature)
        for col in range(2):
            axes[row, col].axhline(0, color="gray", linewidth=0.7)
            axes[row, col].set_xlabel("log10 SNR")
            axes[row, col].legend()
        axes[row, 0].set_title(f"{name}: signed density per ln SNR")
        axes[row, 1].set_title(f"{name}: finite cumulative integral (nats)")
    fig.suptitle(f"{result['representation']} — empirical risk differences")
    fig.savefig(path, dpi=150)
    plt.close(fig)
