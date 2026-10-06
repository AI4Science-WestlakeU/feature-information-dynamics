"""Shared conditional U-Net training, with validation-only calibration diagnostics."""

from __future__ import annotations

import json
import copy
import math
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .analysis import cumulative_integral
from .bundle import sha256, write_json
from .mnist import atomic_save, evaluate
from .mnist_data import balanced_indices, load_mnist


def load_training_state(path: Path, device: torch.device) -> dict:
    """Read our early states with a narrowly allowlisted NumPy float scalar."""
    try:
        state = torch.load(path, weights_only=True, map_location=device)
    except pickle.UnpicklingError:
        # Compatibility only for the first development run's NumPy 2 scalar.
        with torch.serialization.safe_globals([np._core.multiarray.scalar, np.dtype,
                                               type(np.dtype("float64"))]):
            state = torch.load(path, weights_only=True, map_location=device)
    for record in state.get("history", []):
        for key, value in record.items():
            if isinstance(value, np.generic):
                record[key] = value.item()
    return state


class Block(nn.Module):
    """Residual convolution block with time and optional class modulation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.condition = nn.Linear(96, 2*channels)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Apply a residual update using scale and shift modulation."""
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.condition(F.silu(condition)).chunk(2, dim=1)
        h = self.norm2(h) * (1+scale[:, :, None, None]) + shift[:, :, None, None]
        return (x + self.conv2(F.silu(h))) / math.sqrt(2)


class SharedDenoiser(nn.Module):
    """Shared weights make the label and null predictions comparable."""

    def __init__(self, channels: int = 24) -> None:
        super().__init__()
        c = channels
        self.time = nn.Sequential(nn.Linear(16, 96), nn.SiLU(), nn.Linear(96, 96))
        self.label = nn.Embedding(11, 96, padding_idx=10)
        self.stem = nn.Conv2d(1, c, 3, padding=1)
        self.b1, self.b2 = Block(c), Block(2*c)
        self.down1 = nn.Conv2d(c, 2*c, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(2*c, 4*c, 3, stride=2, padding=1)
        self.mid1, self.mid2 = Block(4*c), Block(4*c)
        self.merge1, self.merge2 = nn.Conv2d(6*c, 2*c, 1), nn.Conv2d(3*c, c, 1)
        self.up1, self.up2 = Block(2*c), Block(c)
        self.final = nn.Conv2d(c, 1, 3, padding=1)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self, noisy: torch.Tensor, t: torch.Tensor, labels: torch.Tensor,
                conditional: bool | torch.Tensor) -> torch.Tensor:
        """Return a preconditioned clean-image estimate without guidance."""
        sigma_data = .5
        variance = (sigma_data*t).square() + (1-t).square()
        log_snr = 2*(t.clamp_min(1e-8).log()-(1-t).clamp_min(1e-8).log())
        frequencies = torch.exp(torch.linspace(0, math.log(16), 8, device=t.device))
        angles = log_snr * frequencies[None, :] / 4
        embedding = self.time(torch.cat([angles.sin(), angles.cos()], dim=1))
        if isinstance(conditional, bool):
            ids = labels if conditional else torch.full_like(labels, 10)
        else:
            ids = torch.where(conditional, labels, 10)
        embedding = embedding + self.label(ids)
        h1 = self.b1(self.stem((noisy / variance.sqrt()).reshape(-1, 1, 28, 28)), embedding)
        h2 = self.b2(self.down1(h1), embedding)
        h = self.mid2(self.mid1(self.down2(h2), embedding), embedding)
        h = self.up1(self.merge1(torch.cat([F.interpolate(h, size=(14, 14)), h2], dim=1)), embedding)
        h = self.up2(self.merge2(torch.cat([F.interpolate(h, size=(28, 28)), h1], dim=1)), embedding)
        skip = sigma_data**2*t/variance
        out = sigma_data*(1-t)/variance.sqrt()
        return skip*noisy + out*self.final(F.silu(h)).flatten(1)


def train(config_path: Path, data_root: Path, output: Path, device_name: str,
          resume: bool = False, initial_checkpoint: Path | None = None,
          download: bool = False) -> dict:
    """Train against a selection split only; never load test images for metrics."""
    config = json.loads(config_path.read_text())
    if output.exists() and not resume:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"config.json").exists() and json.loads((output/"config.json").read_text()) != config:
        raise ValueError("Resume config mismatch")
    write_json(output/"config.json", config)
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(device_name)
    arrays, hashes = load_mnist(data_root, download)
    images, labels = arrays["train-images-idx3-ubyte.gz"], arrays["train-labels-idx1-ubyte.gz"]
    if config.get("split_policy") == "all_training_random_holdout":
        if config["train_samples"] + config["selection_samples"] != len(images):
            raise ValueError("Full-data split must cover every official training image")
        indices = np.random.default_rng(config["seed"]).permutation(len(images))
        training = indices[:config["train_samples"]]
        selection = indices[config["train_samples"]:]
        if config["diagnostic_samples"] != len(selection):
            raise ValueError("Full-data protocol evaluates the complete holdout")
        diagnostic = selection
    else:
        indices = balanced_indices(labels, config["train_samples"]+config["selection_samples"], config["seed"])
        grouped = indices.reshape(10, -1)
        split = config["train_samples"]//10
        training, selection = grouped[:, :split].ravel(), grouped[:, split:].ravel()
        diagnostic = grouped[:, split:split+config["diagnostic_samples"]//10].ravel()
    write_json(output/"split_ids.json", {"training": training.tolist(), "selection": selection.tolist(),
                                        "diagnostic": diagnostic.tolist(), "source_sha256": hashes})
    x_all = torch.from_numpy(images[training]).to(device)
    y_all = torch.from_numpy(labels[training]).to(device)
    model = SharedDenoiser(config["channels"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0)
    stream = torch.Generator(device=device).manual_seed(config["seed"]+1)
    initial_step, history, best = 0, [], float("inf")
    if initial_checkpoint is not None and not (resume and (output/"last.pt").exists()):
        parent = load_training_state(initial_checkpoint, device)
        if parent["config"]["channels"] != config["channels"]:
            raise ValueError("Initialization architecture differs")
        model.load_state_dict(parent["model"])
        optimizer.load_state_dict(parent["optimizer"])
        stream.set_state(parent["rng"].cpu())
        write_json(output/"initialization.json", {"checkpoint_sha256": sha256(initial_checkpoint),
                                                   "parent_step": parent["step"], "parent_config": parent["config"],
                                                   "optimizer_and_rng_restored": True})
    ema = copy.deepcopy(model).eval() if config.get("ema_decay") else None
    if resume and (output/"last.pt").exists():
        state = load_training_state(output/"last.pt", device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        stream.set_state(state["rng"].cpu())
        initial_step, history, best = state["step"], state["history"], state["best"]
        if ema is not None:
            ema.load_state_dict(state["ema"])
    grid = config["log10_snr"]
    started = time.perf_counter()
    for step in range(initial_step+1, config["steps"]+1):
        model.train()
        if config.get("lr_schedule") == "cosine":
            fraction = (step-1)/max(1, config["steps"]-1)
            lr = config["min_lr"] + .5*(config["lr"]-config["min_lr"])*(1+math.cos(math.pi*fraction))
        else:
            lr = config["lr"]
        for group in optimizer.param_groups:
            group["lr"] = lr
        n = config["batch_size"]//2
        sampled = torch.randint(len(x_all), (n,), generator=stream, device=device)
        x, y = x_all[sampled], y_all[sampled]
        log_snr = grid[0]+(grid[-1]-grid[0])*torch.rand((n, 1), generator=stream, device=device)
        t = 1/(1+10**(-log_snr/2))
        noise = torch.randn(x.shape, generator=stream, device=device)
        observation = t*x+(1-t)*noise
        pred = model(torch.cat([observation]*2), torch.cat([t]*2), torch.cat([y]*2),
                     torch.arange(2*n, device=device) >= n)
        squared = (pred-torch.cat([x]*2)).square()
        if config.get("loss_weighting", "x_mse") == "edm":
            variance = (.5*t).square()+(1-t).square()
            output_variance = .25*(1-t).square()/variance
            squared = squared / torch.cat([output_variance]*2)
        loss = squared.mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if ema is not None:
            with torch.no_grad():
                for average, current in zip(ema.parameters(), model.parameters()):
                    average.lerp_(current, 1-config["ema_decay"])
        if step % config["validate_every"] == 0 or step == config["steps"]:
            measured_model = ema if ema is not None else model
            kwargs = (measured_model, images[diagnostic], labels[diagnostic], [f"train:{i}" for i in diagnostic], grid)
            curves = np.stack([evaluate(*kwargs, cond, device, config["eval_batch_size"], config["seed"]+2)
                               for cond in (False, True)])
            score = float(curves.mean(dtype=np.float64))
            density = (curves[0].mean(axis=-1, dtype=np.float64)-curves[1].mean(axis=-1, dtype=np.float64))*.5*10.0**np.asarray(grid, dtype=np.float64)
            integral = float(cumulative_integral(density, np.asarray(grid))[-1])
            record = {"step": step, "validation_risk_sum": score, "validation_integral_nats": integral,
                      "relative_entropy_error": float(abs(integral-np.log(10))/np.log(10)),
                      "elapsed_seconds": time.perf_counter()-started, "lr": lr,
                      "weight_state": "ema" if ema is not None else "raw"}
            history.append(record)
            np.save(output/f"validation_step_{step}.npy", curves, allow_pickle=False)
            if score < best:
                best = score
                atomic_save(output/"selected.pt", {"model": measured_model.state_dict(), "config": config, "step": step,
                                                   "weight_state": record["weight_state"]})
            atomic_save(output/"last.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                          "rng": stream.get_state(), "config": config, "step": step,
                                          "history": history, "best": best,
                                          "ema": ema.state_dict() if ema is not None else None})
            write_json(output/"history.json", history)
            print(json.dumps(record), flush=True)
    report = {"config_sha256": sha256(output/"config.json"), "selected_sha256": sha256(output/"selected.pt"),
              "history": history, "test_evaluated": False, "parameters": sum(p.numel() for p in model.parameters())}
    write_json(output/"TRAIN_DONE.json", report)
    return report
