"""Independently reload a frozen neural checkpoint and verify bundled errors."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from feature_information_dynamics.bundle import load_bundle, sha256, write_json
from feature_information_dynamics.mnist import evaluate
from feature_information_dynamics.mnist_calibrated import SharedDenoiser
from feature_information_dynamics.mnist_data import load_mnist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=37)
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    errors, metadata, samples = load_bundle(args.bundle)
    if sha256(args.checkpoint) != metadata["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash differs")
    arrays, hashes = load_mnist(args.data, False)
    if hashes != metadata["source_sha256"]:
        raise ValueError("Source hashes differ")
    # Spread the check over the full manifest, including all classes.
    positions = np.linspace(0, len(samples)-1, min(args.samples, len(samples)), dtype=int)
    ids = [samples[i]["sample_id"] for i in positions]
    indices = [int(value.split(":")[1]) for value in ids]
    images = arrays["t10k-images-idx3-ubyte.gz"][indices]
    labels = arrays["t10k-labels-idx1-ubyte.gz"][indices]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, weights_only=True, map_location=device)
    model = SharedDenoiser(state["config"]["channels"]).to(device)
    model.load_state_dict(state["model"])
    started = time.perf_counter()
    repeats = []
    for seed in metadata["noise"]["seeds"]:
        repeats.append(np.stack([evaluate(model, images, labels, ids, metadata["log10_snr"],
                                          cond, device, args.batch_size, seed)
                                 for cond in (False, True)]))
    actual = np.stack(repeats, axis=2)
    expected = errors[..., positions]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    report = {"passed": True, "samples": len(ids), "batch_size": args.batch_size,
              "device": str(device), "checkpoint_sha256": sha256(args.checkpoint),
              "max_absolute_sse_difference": float(np.abs(actual-expected).max()),
              "rtol": 1e-5, "atol": 1e-5, "seconds": time.perf_counter()-started,
              "sample_ids": ids, "torch": str(torch.__version__)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
