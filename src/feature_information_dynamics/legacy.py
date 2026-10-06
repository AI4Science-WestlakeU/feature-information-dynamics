"""Explicitly labeled analysis of historical aggregate curves without fake CIs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .analysis import cumulative_integral, plot, weights
from .bundle import sha256, write_json


def analyze_curves(source: Path, output: Path) -> list[dict]:
    """Recompute legacy eight-condition point estimates, never paired statistics."""
    data = json.loads(source.read_text(encoding="utf-8"))
    expected = source.with_suffix(".sha256").read_text().strip()
    if sha256(source) != expected:
        raise ValueError("Historical aggregate data hash mismatch")
    if data["schema"] != "legacy-aggregate-v1":
        raise ValueError("Not a supported legacy aggregate artifact")
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for rep in data["representations"]:
        features, subsets = rep["features"], rep["subsets"]
        if features != ["class", "mask", "canny"]:
            raise ValueError("Expected classical three-feature definition")
        if {frozenset(s) for s in subsets} != {frozenset(f for i, f in enumerate(features) if bit & (1 << i)) for bit in range(8)} or len(subsets) != 8:
            raise ValueError("Historical artifact lacks the eight unique conditions")
        grid = np.asarray(rep["log10_snr"], dtype=np.float64)
        risk = np.asarray(rep["risk_sum"], dtype=np.float64)
        if risk.shape != (8, len(grid)) or len(grid) < 2 or not np.isfinite(risk).all() or (risk < 0).any() or not np.isfinite(grid).all() or not (np.diff(grid) > 0).all():
            raise ValueError("Invalid historical curve data")
        chain, shapley = weights(features, subsets)
        result = {"representation": rep["name"] + " (HISTORICAL AGGREGATES)",
                  "features": features, "log10_snr": grid.tolist(),
                  "risk_sum": risk.tolist(), "source_sha256": sha256(source),
                  "warning": "Historical protocols differ; no per-image arrays, no CIs, no certification of the current paper or unified remeasurement.",
                  "provenance": rep["provenance"]}
        for name, matrix in (("chain", chain), ("shapley", shapley)):
            gap = matrix @ risk
            density = gap * (0.5 * 10**grid)
            cumulative = cumulative_integral(density, grid)
            result[name] = {"gap": gap.tolist(), "density_per_ln_snr": density.tolist(),
                            "cumulative_nats": cumulative.tolist(),
                            "finite_integral_nats": cumulative[:, -1].tolist()}
        name = rep["name"]
        if name not in ("pixel", "rae", "sdvae", "vavae"):
            raise ValueError("Unexpected representation name")
        write_json(output / f"{name}_summary.json", result)
        plot(result, output / f"{name}_information.png")
        reports.append({"representation": name, "status": "legacy_aggregate_only"})
    write_json(output / "index.json", reports)
    return reports
