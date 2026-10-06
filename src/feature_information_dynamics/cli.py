"""Public command-line entry points; analysis does not import torch."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Dispatch CPU analysis or optional neural MNIST workflows."""
    parser = argparse.ArgumentParser(prog="feature-information")
    commands = parser.add_subparsers(dest="command", required=True)
    weights = commands.add_parser("weights", help="Download official initialization weights before training")
    weights.add_argument("arguments", nargs=argparse.REMAINDER)
    unified = commands.add_parser("reproduce", help="Validate and reproduce one unified four-representation suite")
    unified.add_argument("--suite", type=Path, required=True)
    unified.add_argument("--output", type=Path, required=True)
    unified.add_argument("--bootstrap", type=int, default=1000)
    analysis = commands.add_parser("analyze", help="Recompute signed chain and Shapley from paired errors")
    analysis.add_argument("--bundle", type=Path, required=True)
    analysis.add_argument("--output", type=Path, required=True)
    analysis.add_argument("--bootstrap", type=int, default=200)
    legacy = commands.add_parser("analyze-curves", help="Historical aggregate curves only; no confidence intervals")
    legacy.add_argument("--source", type=Path, required=True)
    legacy.add_argument("--output", type=Path, required=True)
    mnist = commands.add_parser("mnist", help="Train the bounded MNIST label teaching example")
    mnist.add_argument("--config", type=Path, required=True)
    mnist.add_argument("--data", type=Path, default=Path("data/mnist"))
    mnist.add_argument("--output", type=Path, required=True)
    mnist.add_argument("--device", default="auto")
    mnist.add_argument("--download", action="store_true")
    mnist.add_argument("--resume", action="store_true")
    verify = commands.add_parser("mnist-evaluate", help="Reload selected weights and verify paired errors")
    verify.add_argument("--run", type=Path, required=True)
    verify.add_argument("--data", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--device", default="cpu")
    verify.add_argument("--batch-size", type=int, default=128)
    reference = commands.add_parser("mnist-reference", help="Exact posterior on a finite MNIST prior; not population MNIST")
    reference.add_argument("--data", type=Path, required=True)
    reference.add_argument("--output", type=Path, required=True)
    reference.add_argument("--device", default="cpu")
    reference.add_argument("--support-size", type=int, default=1000)
    reference.add_argument("--observations", type=int, default=2000)
    calibration = commands.add_parser("mnist-shared-train", help="Train shared conditional U-Net; validation diagnostics only")
    calibration.add_argument("--config", type=Path, required=True)
    calibration.add_argument("--data", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True)
    calibration.add_argument("--device", default="cpu")
    calibration.add_argument("--resume", action="store_true")
    calibration.add_argument("--download", action="store_true")
    calibration.add_argument("--init-checkpoint", type=Path)
    for representation in ("pixel", "rae", "sdvae", "vavae"):
        experiment = commands.add_parser(representation, help=f"Run {representation} training or evaluation")
        actions = ["train", "eval", "cache"] if representation == "vavae" else ["train", "eval"]
        experiment.add_argument("action", choices=actions)
        experiment.add_argument("arguments", nargs=argparse.REMAINDER, help="Upstream source options, then -- and workflow arguments")
    released = commands.add_parser("data", help="Download, verify and configure prepared experimental data")
    released.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "weights":
        from .weights import main as prepare_weights
        prepare_weights(args.arguments)
        return
    if args.command == "data":
        from .data.releases import main as release_data
        release_data(args.arguments)
        return
    if args.command in {"pixel", "rae", "sdvae", "vavae"}:
        forwarded = args.arguments
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]
        if args.action == "train":
            from .weights import check_training_checkpoint
            try:
                check_training_checkpoint(forwarded)
            except ValueError as error:
                parser.error(str(error))
        if args.command == "pixel":
            module = "feature_information_dynamics.pixel." + ("train" if args.action == "train" else "evaluate")
            raise SystemExit(subprocess.call([sys.executable, "-m", module, *forwarded]))
        from .workflows import main as launch_workflow
        raise SystemExit(launch_workflow([args.command, args.action, *forwarded]))
    if args.command == "reproduce":
        from .suite import reproduce
        print(json.dumps(reproduce(args.suite, args.output, args.bootstrap)))
    elif args.command == "analyze":
        from .analysis import analyze
        result = analyze(args.bundle, args.output, args.bootstrap)
        print(json.dumps({"integral_nats": result["chain"]["finite_integral_nats"]}))
    elif args.command == "analyze-curves":
        from .legacy import analyze_curves
        print(json.dumps(analyze_curves(args.source, args.output)))
    elif args.command == "mnist-reference":
        from .mnist_reference import run_reference
        run_reference(args.data, args.output, args.device, args.support_size, args.observations)
    elif args.command == "mnist-shared-train":
        from .mnist_calibrated import train
        train(args.config, args.data, args.output, args.device, args.resume, args.init_checkpoint, args.download)
    else:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        from .mnist import re_evaluate, run
        if args.command == "mnist":
            result = run(args.config, args.data, args.output, args.device, args.download, args.resume)
        else:
            if args.batch_size < 1:
                parser.error("batch size must be positive")
            result = re_evaluate(args.run, args.data, args.output, args.batch_size, args.device)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
