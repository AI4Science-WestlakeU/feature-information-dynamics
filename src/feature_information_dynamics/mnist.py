"""Bounded two-denoiser MNIST teaching experiment and checkpoint re-evaluation."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .analysis import analyze
from .bundle import sha256, write_bundle, write_json
from .mnist_data import balanced_indices, load_mnist
from .noise import stable_noise


class Denoiser(nn.Module):
    """Small residual MLP with bounded clean-end preconditioning."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(784 + 10 + 3, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 784))

    def forward(self, noisy: torch.Tensor, t: torch.Tensor, labels: torch.Tensor,
                conditional: bool) -> torch.Tensor:
        """Predict X; both arms have exactly the same parameter count."""
        variance = t.square() + (1-t).square()
        one_hot = torch.nn.functional.one_hot(labels, 10).to(noisy.dtype)
        if not conditional:
            one_hot = torch.zeros_like(one_hot)
        time_features = torch.cat([t, torch.sin(torch.pi*t), torch.cos(torch.pi*t)], dim=1)
        inputs = torch.cat([noisy / variance.sqrt(), one_hot, time_features], dim=1)
        skip = t / variance
        out = (1-t) / variance.sqrt()
        return skip * noisy + out * self.net(inputs)


@torch.no_grad()
def evaluate(model: Denoiser, images: np.ndarray, labels: np.ndarray,
             ids: list[str], grid: list[float], conditional: bool,
             device: torch.device, batch_size: int, seed: int) -> np.ndarray:
    """Evaluate same images/noise for both arms, with no clipping or guidance."""
    model.eval()
    noise = stable_noise(ids, 784, seed)
    result = np.empty((len(grid), len(images)), dtype=np.float32)
    for j, lg in enumerate(grid):
        t_value = 1 / (1 + 10 ** (-lg / 2))
        for start in range(0, len(images), batch_size):
            end = start + batch_size
            x = torch.from_numpy(images[start:end]).to(device)
            y = torch.from_numpy(labels[start:end]).to(device)
            eps = torch.from_numpy(noise[start:end]).to(device)
            t = torch.full((len(x), 1), t_value, device=device)
            predicted = model(t*x + (1-t)*eps, t, y, conditional)
            error = (predicted-x).double().square().sum(dim=1)
            result[j, start:end] = error.cpu().numpy()
    return result


def atomic_save(path: Path, payload: dict) -> None:
    """Replace only this run's checkpoint after a complete write."""
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_arm(config: dict, output: Path, conditional: bool, train: tuple,
              selection: tuple, device: torch.device, resume: bool) -> tuple[Denoiser, dict]:
    """Train one finite arm; only the training-held-out split selects weights."""
    arm = "class" if conditional else "empty"
    checkpoint = output / f"{arm}_last.pt"
    selected = output / f"{arm}_selected.pt"
    torch.manual_seed(config["seed"])
    model = Denoiser(config["hidden"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    stream = torch.Generator().manual_seed(config["seed"] + 1)
    first_step, best, history = 0, float("inf"), []
    if checkpoint.exists():
        if not resume:
            raise FileExistsError("Checkpoint exists; use --resume or a new output")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["config"] != config:
            raise ValueError("Resume configuration differs from frozen run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        stream.set_state(state["stream"])
        first_step, best, history = state["step"], state["best"], state["history"]
    images, labels = (torch.from_numpy(a) for a in train)
    grid = config["log10_snr"]
    started = time.perf_counter()
    for step in range(first_step + 1, config["steps"] + 1):
        model.train()
        index = torch.randint(len(images), (config["batch_size"],), generator=stream)
        x, y = images[index].to(device), labels[index].to(device)
        lg = grid[0] + (grid[-1]-grid[0]) * torch.rand((len(x), 1), generator=stream)
        t = (1 / (1 + 10 ** (-lg / 2))).to(device)
        noise = torch.randn(x.shape, generator=stream).to(device)
        loss = (model(t*x + (1-t)*noise, t, y, conditional)-x).square().mean()
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss at {arm} step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % config["validate_every"] == 0 or step == config["steps"]:
            values = evaluate(model, *selection, grid, conditional, device,
                              config["eval_batch_size"], config["seed"] + 2)
            metric = float(values.mean(dtype=np.float64))
            history.append({"step": step, "selection_risk_sum": metric,
                            "train_loss_per_element": float(loss.detach())})
            if metric < best:
                best = metric
                atomic_save(selected, {"model": model.state_dict(), "step": step,
                                       "config": config, "conditional": conditional})
            atomic_save(checkpoint, {"model": model.state_dict(),
                                    "optimizer": optimizer.state_dict(),
                                    "stream": stream.get_state(), "step": step,
                                    "config": config, "best": best, "history": history})
            print(f"{arm} {step}/{config['steps']} selection SSE={metric:.4f}", flush=True)
    state = torch.load(selected, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    return model, {"selected_step": state["step"], "selection_history": history,
                   "current_invocation_seconds": time.perf_counter()-started,
                   "selected_checkpoint_sha256": sha256(selected)}


def run(config_path: Path, data_root: Path, output: Path, device_name: str,
        download: bool, resume: bool = False) -> dict:
    """Execute paired train/select/test workflow using official MNIST files."""
    config = json.loads(config_path.read_text())
    for key in ("steps", "validate_every", "hidden", "batch_size", "eval_batch_size"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"Invalid {key}")
    for key in ("train_samples", "selection_samples", "test_samples"):
        if not isinstance(config[key], int) or config[key] <= 0 or config[key] % 10:
            raise ValueError(f"{key} must be a positive multiple of ten")
    if not np.isfinite(config["lr"]) or config["lr"] <= 0:
        raise ValueError("Learning rate must be finite and positive")
    if len(config["log10_snr"]) < 2 or not np.all(np.diff(config["log10_snr"]) > 0):
        raise ValueError("Invalid SNR grid")
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("Use a new output directory, or --resume")
    if (output / "DONE.json").exists():
        raise FileExistsError("Run is complete; use analyze or mnist-evaluate instead")
    output.mkdir(parents=True, exist_ok=True)
    frozen_path = output / "config.json"
    if frozen_path.exists() and json.loads(frozen_path.read_text()) != config:
        raise ValueError("Frozen run config differs")
    write_json(frozen_path, config)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available()
                          else "cpu" if device_name == "auto" else device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    arrays, hashes = load_mnist(data_root, download)
    x, y = arrays["train-images-idx3-ubyte.gz"], arrays["train-labels-idx1-ubyte.gz"]
    all_indices = balanced_indices(y, config["train_samples"] + config["selection_samples"], config["seed"])
    # Within every class, reserve the final selection quota.
    grouped = all_indices.reshape(10, -1)
    n_train = config["train_samples"] // 10
    training = grouped[:, :n_train].ravel()
    validation = grouped[:, n_train:].ravel()
    xt, yt = arrays["t10k-images-idx3-ubyte.gz"], arrays["t10k-labels-idx1-ubyte.gz"]
    testing = balanced_indices(yt, config["test_samples"], config["seed"] + 3)
    selection = (x[validation], y[validation], [f"train:{i}" for i in validation])
    samples = [{"sample_id": f"test:{i}", "class_id": int(yt[i])} for i in testing]
    write_json(output / "split_ids.json", {"training": training.tolist(),
               "selection": validation.tolist(), "test": testing.tolist(), "source_sha256": hashes})
    errors, provenance = [], {}
    for conditional in (False, True):
        model, record = train_arm(config, output, conditional, (x[training], y[training]),
                                  selection, device, resume)
        errors.append(evaluate(model, xt[testing], yt[testing], [s["sample_id"] for s in samples],
                               config["log10_snr"], conditional, device,
                               config["eval_batch_size"], config["seed"]+4))
        provenance["class" if conditional else "empty"] = record
    metadata = {"features": ["class"], "subsets": [[], ["class"]],
                "representation": "MNIST label teaching experiment",
                "target_dimension": 784, "error_reduction": "sum",
                "log10_snr": config["log10_snr"], "pixel_scale": "uint8/127.5-1",
                "noise": {"algorithm": "SHA256 seed + NumPy PCG64 per sample ID; same noise across SNR", "seed": config["seed"]+4},
                "source_sha256": hashes, "checkpoints": provenance,
                "note": "Small shared-SNR neural estimators; not the paper spectral experiment or certified Bayes MMSE"}
    # Keep a completed measurement immutable; an interrupted postprocessing can resume.
    bundle_path = output / "bundle"
    if not bundle_path.exists():
        write_bundle(bundle_path, np.stack(errors)[:, :, None, :], metadata, samples)
    summary = analyze(bundle_path, output / "analysis", config["bootstrap"])
    record = {"seconds": time.perf_counter()-started, "device": str(device),
              "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
              "torch": str(torch.__version__), "numpy": np.__version__, "python": platform.python_version(),
              "config_sha256": sha256(frozen_path), "label_entropy_nats": float(np.log(10)),
              "estimated_finite_integral_nats": summary["chain"]["finite_integral_nats"],
              "parameters_per_model": sum(p.numel() for p in model.parameters())}
    write_json(output / "DONE.json", record)
    return record


def re_evaluate(run_root: Path, data_root: Path, output: Path, batch_size: int,
                device_name: str = "cpu") -> dict:
    """Reload only selected weights and independently recompute published errors."""
    from .bundle import load_bundle
    reference, meta, samples = load_bundle(run_root / "bundle")
    arrays, hashes = load_mnist(data_root, False)
    if hashes != meta["source_sha256"]:
        raise ValueError("Dataset differs from measurement source")
    config = json.loads((run_root / "config.json").read_text())
    device = torch.device(device_name)
    torch.set_num_threads(2)
    indices = np.asarray([int(s["sample_id"].split(":")[1]) for s in samples])
    values = []
    for conditional, arm in ((False, "empty"), (True, "class")):
        checkpoint = run_root / f"{arm}_selected.pt"
        if sha256(checkpoint) != meta["checkpoints"][arm]["selected_checkpoint_sha256"]:
            raise ValueError("Selected checkpoint hash mismatch")
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model = Denoiser(config["hidden"]).to(device)
        model.load_state_dict(state["model"])
        values.append(evaluate(model, arrays["t10k-images-idx3-ubyte.gz"][indices],
                               arrays["t10k-labels-idx1-ubyte.gz"][indices],
                               [s["sample_id"] for s in samples], config["log10_snr"],
                               conditional, device, batch_size, config["seed"]+4))
    actual = np.stack(values)[:, :, None, :]
    difference = np.abs(actual-reference)
    report = {"passed": bool(np.allclose(actual, reference, rtol=1e-5, atol=1e-5)),
              "max_abs_error_sum_difference": float(difference.max()),
              "rtol": 1e-5, "atol": 1e-5, "device": str(device), "batch_size": batch_size}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "verification.json", report)
    if not report["passed"]:
        raise ValueError(f"Checkpoint re-evaluation failed: {report}")
    return report
