"""Frozen, previously unmeasured MNIST confirmation with explicit 10% acceptance."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch

from .analysis import analyze, cumulative_integral
from .bundle import sha256, write_bundle, write_json
from .mnist import evaluate
from .mnist_calibrated import SharedDenoiser
from .mnist_data import balanced_indices, load_mnist


def prepare(data_root: Path, excluded: list[Path], output: Path) -> dict:
    """Freeze 5000 balanced test IDs, excluding both previously evaluated demos."""
    if output.exists():
        raise FileExistsError(output)
    arrays, hashes = load_mnist(data_root, False)
    prior_ids = set()
    for path in excluded:
        prior_ids.update(row["sample_id"] for row in json.loads(path.read_text()))
    labels = arrays["t10k-labels-idx1-ubyte.gz"]
    available = np.asarray([i for i in range(len(labels)) if f"test:{i}" not in prior_ids])
    selected = available[balanced_indices(labels[available], 5000, 20260929)]
    result = {"samples": [{"sample_id": f"test:{i}", "class_id": int(labels[i])} for i in selected],
              "excluded_id_count": len(prior_ids), "overlap_with_excluded": 0,
              "source_sha256": hashes, "selection_seed": 20260929,
              "noise_seeds": [20260930, 20261001],
              "relative_error_limit": .1, "reference_nats": float(np.log(10)),
              "criterion": "Absolute relative error of unmodified finite signed integral vs ln(10) <= 10%; report image bootstrap CI and each noise repeat separately"}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return {"samples": 5000, "excluded": len(prior_ids), "sha256": sha256(output)}


def freeze(run_root: Path, manifest: Path, output: Path) -> dict:
    """Bind one validation-selected checkpoint to the existing test protocol."""
    if output.exists():
        raise FileExistsError(output)
    state = torch.load(run_root/"selected.pt", weights_only=True, map_location="cpu")
    result = {"checkpoint_sha256": sha256(run_root/"selected.pt"), "selected_step": state["step"],
              "config": state["config"], "manifest_sha256": sha256(manifest),
              "selection": "Lowest mean validation reconstruction risk; not test-selected",
              "test_results_seen": False, "weight_state": state.get("weight_state", "raw"),
              "initialization": json.loads((run_root/"initialization.json").read_text())
                  if (run_root/"initialization.json").exists() else None}
    write_json(output, result)
    return result


def prepare_full(data_root: Path, output: Path, relative_error_limit: float = .2) -> dict:
    """Freeze all official test IDs and their empirical class entropy before inference."""
    if output.exists():
        raise FileExistsError(output)
    if not 0 < relative_error_limit < 1:
        raise ValueError("Relative error limit must lie between zero and one")
    arrays, hashes = load_mnist(data_root, False)
    labels = arrays["t10k-labels-idx1-ubyte.gz"]
    probabilities = np.bincount(labels, minlength=10) / len(labels)
    positive = probabilities[probabilities > 0]
    entropy = float(-np.sum(positive * np.log(positive)))
    result = {"samples": [{"sample_id": f"test:{i}", "class_id": int(y)}
                          for i, y in enumerate(labels)],
              "source_sha256": hashes, "noise_seeds": [20261004, 20261005],
              "relative_error_limit": relative_error_limit, "reference_nats": entropy,
              "class_probabilities": probabilities.tolist(), "excluded_id_count": 0,
              "scope": "All official test images; includes previously used demo IDs, not a pristine unseen test set",
              "criterion": "Absolute relative error of raw finite signed integral versus test label entropy"}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return {"samples": len(labels), "reference_nats": entropy, "sha256": sha256(output)}


def confirm(run_root: Path, data_root: Path, manifest_path: Path, frozen_path: Path,
            output: Path, device_name: str) -> dict:
    """Run the pre-frozen test, saving all signed data and acceptance evidence."""
    if output.exists():
        raise FileExistsError(output)
    frozen = json.loads(frozen_path.read_text())
    if sha256(run_root/"selected.pt") != frozen["checkpoint_sha256"] or sha256(manifest_path) != frozen["manifest_sha256"]:
        raise ValueError("Checkpoint or test protocol changed after freezing")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text())
    arrays, hashes = load_mnist(data_root, False)
    if hashes != manifest["source_sha256"]:
        raise ValueError("Confirmation dataset hashes differ")
    ids = [row["sample_id"] for row in manifest["samples"]]
    indices = np.asarray([int(sample_id.split(":")[1]) for sample_id in ids])
    images = arrays["t10k-images-idx3-ubyte.gz"][indices]
    labels = arrays["t10k-labels-idx1-ubyte.gz"][indices]
    device = torch.device(device_name)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    state = torch.load(run_root/"selected.pt", weights_only=True, map_location=device)
    config = state["config"]
    if config != frozen["config"]:
        raise ValueError("Frozen architecture/config mismatch")
    model = SharedDenoiser(config["channels"]).to(device)
    model.load_state_dict(state["model"])
    started = time.perf_counter()
    all_errors = []
    grid = np.asarray(config["log10_snr"], dtype=np.float64)
    for seed in manifest["noise_seeds"]:
        all_errors.append(np.stack([evaluate(model, images, labels, ids, grid.tolist(), cond,
                                              device, config["eval_batch_size"], seed)
                                    for cond in (False, True)]))
        print(f"confirmation noise {seed} complete", flush=True)
    errors = np.stack(all_errors, axis=2)
    metadata = {"features": ["class"], "subsets": [[], ["class"]],
                "representation": "MNIST shared conditional U-Net confirmation",
                "log10_snr": grid.tolist(), "target_dimension": 784, "error_reduction": "sum",
                "checkpoint_sha256": frozen["checkpoint_sha256"], "selected_step": state["step"],
                "manifest_sha256": sha256(manifest_path), "frozen_config": config,
                "weight_state": frozen["weight_state"], "initialization": frozen["initialization"],
                "noise": {"seeds": manifest["noise_seeds"], "algorithm": "SHA256 per ID, PCG64 float64 then float32"},
                "source_sha256": hashes, "pixel_scale": "uint8/127.5-1"}
    write_bundle(output/"bundle", errors, metadata, manifest["samples"])
    summary = analyze(output/"bundle", output/"analysis", bootstrap=1000)
    # Integrate each paired observation first, then bootstrap whole images.
    differences = errors[0].astype(np.float64)-errors[1].astype(np.float64)
    density = differences.transpose(1, 2, 0) * (.5*10**grid)
    per_observation = cumulative_integral(density, grid)[..., -1]
    per_image = per_observation.mean(axis=0)
    estimate = float(per_image.mean())
    rng = np.random.default_rng(20260929)
    groups = [np.flatnonzero(labels == k) for k in range(10)]
    replicates = [np.mean(per_image[np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])])
                  for _ in range(2000)]
    ci = np.quantile(replicates, [.025, .975])
    reference = manifest["reference_nats"]
    relative = abs(estimate-reference)/reference
    half_grid = grid[::2]
    half_density = density[..., ::2].mean(axis=(0, 1))
    coarse = float(cumulative_integral(half_density, half_grid)[-1])
    limit = manifest["relative_error_limit"]
    report = {"passed": bool(relative <= limit),
              "relative_error_limit": limit,
              "estimate_nats": estimate, "reference_nats": reference,
              "relative_error": relative, "paired_image_bootstrap_95ci": ci.tolist(),
              "ci_entirely_within_10_percent": bool(ci[0] >= .9*reference and ci[1] <= 1.1*reference),
              "ci_entirely_within_acceptance_band": bool(ci[0] >= (1-limit)*reference and ci[1] <= (1+limit)*reference),
              "per_noise_repeat_nats": per_observation.mean(axis=1).tolist(),
              "coarser_grid_integral_nats": coarse, "coarser_grid_change_nats": coarse-estimate,
              "sample_count": len(ids), "noise_repeat_count": len(all_errors),
              "excluded_prior_demo_ids": manifest["excluded_id_count"],
              "log10_snr_bounds": [float(grid[0]), float(grid[-1])],
              "selected_step": state["step"], "checkpoint_sha256": frozen["checkpoint_sha256"],
              "seconds": time.perf_counter()-started, "device": str(device),
              "torch": str(torch.__version__), "numpy": np.__version__,
              "measurement_precision": "FP32; TF32 disabled for matmul and cuDNN convolutions",
              "scope": "A frozen neural estimator and finite integral; agreement with class entropy is not proof of pointwise Bayes MMSE"}
    np.testing.assert_allclose(estimate, summary["chain"]["finite_integral_nats"][0], atol=1e-10)
    write_json(output/"acceptance.json", report)
    return report


def main() -> None:
    """Expose separate preparation, freezing and confirmation actions."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "prepare-full", "freeze", "confirm"])
    parser.add_argument("--relative-error-limit", type=float, default=.2)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--exclude-samples", type=Path, nargs="*", default=[])
    parser.add_argument("--run", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.data, args.exclude_samples, args.output)
    elif args.action == "prepare-full":
        result = prepare_full(args.data, args.output, args.relative_error_limit)
    elif args.action == "freeze":
        result = freeze(args.run, args.manifest, args.output)
    else:
        result = confirm(args.run, args.data, args.manifest, args.frozen, args.output, args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
