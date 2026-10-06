"""Exact finite-MNIST-prior posteriors with Monte Carlo observation integration."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch

from .analysis import cumulative_integral
from .bundle import write_json
from .mnist_data import balanced_indices, load_mnist


def run_reference(data_root: Path, output: Path, device_name: str,
                  support_size: int = 1000, observations: int = 2000) -> dict:
    """Compare I-MMSE quadrature to entropy of exact class posteriors.

    The data distribution is explicitly uniform on the selected training images,
    not the unknown population distribution of handwritten digits. No fitted
    classifier or learned denoiser is used. The finite support is public by ID.
    """
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new reference output directory")
    if observations < 20:
        raise ValueError("Need at least twenty observation draws")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(device_name)
    started = time.perf_counter()
    arrays, hashes = load_mnist(data_root, False)
    labels = arrays["train-labels-idx1-ubyte.gz"]
    support_ids = balanced_indices(labels, support_size, 20260910)
    support = arrays["train-images-idx3-ubyte.gz"][support_ids]
    rng = np.random.default_rng(20260912)
    indices = rng.integers(0, support_size, observations)
    noise = rng.standard_normal((observations, 784)).astype(np.float32)
    # float64 avoids cancellation in small posterior-variance differences.
    points = torch.from_numpy(support).to(device, torch.float64)
    square_norm = points.square().sum(dim=1)
    grid = np.linspace(-6, 1, 29)
    density, direct_mi, risks = [], [], []
    per_class = support_size // 10
    with torch.no_grad():
        for lg in grid:
            gamma = float(10**lg)
            row_density, row_entropy, row_risks = [], [], []
            for start in range(0, observations, 100):
                x = points[torch.as_tensor(indices[start:start+100], device=device)]
                eps = torch.from_numpy(noise[start:start+100]).to(device, torch.float64)
                observation = gamma**.5 * x + eps
                # Omit a row-constant distance term before softmax.
                logits = gamma**.5 * (observation @ points.T) - .5*gamma*square_norm
                posterior = logits.softmax(dim=1)
                mean = posterior @ points
                unconditional_variance = (posterior @ square_norm) - mean.square().sum(dim=1)
                between = torch.zeros(len(x), dtype=torch.float64, device=device)
                entropy = torch.zeros_like(between)
                for k in range(10):
                    sl = slice(k*per_class, (k+1)*per_class)
                    mass = posterior[:, sl].sum(dim=1)
                    # Direct conditional softmax remains stable when class mass underflows.
                    class_mean = logits[:, sl].softmax(dim=1) @ points[sl]
                    between += mass * (class_mean-mean).square().sum(dim=1)
                    entropy -= mass * mass.clamp_min(1e-300).log()
                row_density.extend((.5*gamma*between).cpu().tolist())
                row_entropy.extend(entropy.cpu().tolist())
                row_risks.extend(torch.stack([unconditional_variance,
                                             unconditional_variance-between], dim=1).cpu().tolist())
            density.append(float(np.mean(row_density)))
            direct_mi.append(float(np.log(10)-np.mean(row_entropy)))
            risks.append(np.mean(row_risks, axis=0).tolist())
    cumulative = cumulative_integral(np.asarray(density)[None, :], grid)[0]
    entropy_increment = direct_mi[-1]-direct_mi[0]
    result = {"definition": "Uniform finite prior on balanced MNIST training images; exact posterior, Monte Carlo observations and finite-grid quadrature",
              "population_mnist_claim": False, "support_size": support_size,
              "observations": observations, "support_ids": support_ids.tolist(),
              "source_sha256": hashes, "support_seed": 20260910, "observation_seed": 20260912,
              "log10_snr": grid.tolist(), "density_per_ln_snr": density,
              "posterior_risks_sum": risks, "direct_posterior_mi_nats": direct_mi,
              "cumulative_nats": cumulative.tolist(), "label_entropy_nats": float(np.log(10)),
              "finite_integral_nats": float(cumulative[-1]),
              "direct_finite_information_increment_nats": entropy_increment,
              "quadrature_minus_direct_nats": float(cumulative[-1]-entropy_increment),
              "seconds": time.perf_counter()-started, "device": str(device),
              "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
              "torch": str(torch.__version__), "numpy": np.__version__}
    write_json(output / "reference.json", result)
    plot_reference(result, output / "reference.png")
    print(f"finite-prior reference integral={cumulative[-1]:.6f}; direct increment={entropy_increment:.6f}", flush=True)
    return result


def plot_reference(result: dict, path: Path) -> None:
    """Plot a separate, explicitly labeled finite-prior reference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = result["log10_snr"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(grid, result["density_per_ln_snr"], "o-")
    axes[0].set_title("Exact posterior: density per ln SNR")
    axes[1].plot(grid, result["direct_posterior_mi_nats"], label="Direct posterior MI")
    # Start at the same finite lower limit rather than silently claiming I(0)=0.
    axes[1].plot(grid, np.asarray(result["cumulative_nats"])+result["direct_posterior_mi_nats"][0], "--", label="I-MMSE integral + lower endpoint")
    axes[1].axhline(result["label_entropy_nats"], color="gray", linestyle=":", label="H(class)=ln(10)")
    axes[1].set_title("Information (nats)")
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("log10 SNR")
    fig.suptitle("Finite MNIST prior — not population-MNIST MMSE")
    fig.savefig(path, dpi=150)
    plt.close(fig)
