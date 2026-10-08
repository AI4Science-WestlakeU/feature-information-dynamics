"""Build the MNIST webpage demo from the same verified measurements as the guide."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from feature_information_dynamics.mnist_data import load_mnist
from feature_information_dynamics.mnist_calibrated import SharedDenoiser
from feature_information_dynamics.noise import stable_noise
from feature_information_dynamics.bundle import sha256, load_bundle

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "examples" / "mnist_class"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
plt.rcParams.update({"figure.dpi": 110, "font.size": 11})
asset = json.loads((ASSETS / "checkpoint.json").read_text())
checkpoint = ASSETS / "selected.pt"
if sha256(checkpoint) != asset["sha256"]: raise ValueError("Tutorial checkpoint checksum mismatch")
state = torch.load(checkpoint, map_location="cpu", weights_only=True)
model = SharedDenoiser(state["config"]["channels"]).to(device)
model.load_state_dict(state["model"])
model.eval()
arrays, _ = load_mnist(ROOT / "data" / "mnist", download=True)
images = arrays["t10k-images-idx3-ubyte.gz"]
labels = arrays["t10k-labels-idx1-ubyte.gz"]

DIGIT = 3
EXAMPLE = 0
errors, measured_metadata, measured_samples = load_bundle(ASSETS / "measurements")
assert measured_metadata["checkpoint_sha256"] == sha256(checkpoint)
assert [s["sample_id"] for s in measured_samples] == [f"test:{i}" for i in range(len(images))]
assert [s["class_id"] for s in measured_samples] == labels.tolist()
grid = np.asarray(measured_metadata["log10_snr"])
seeds = measured_metadata["noise"]["seeds"]
print(f"Loaded measured errors with axes (mode, SNR, noise, image): {errors.shape}")
risks = errors.astype(np.float64).mean(axis=(2, 3))
gap = risks[0] - risks[1]
density = .5 * 10.0**grid * gap
progress_density = np.maximum(density, 0.)
positive_cumulative = np.r_[0., np.cumsum(.5*(progress_density[1:]+progress_density[:-1])*np.diff(grid)*np.log(10))]
progress = positive_cumulative / positive_cumulative[-1]
cumulative = np.r_[0., np.cumsum(.5*(density[1:]+density[:-1])*np.diff(grid)*np.log(10))]

@torch.no_grad()
def prepare_digit(digit=7, example=0, levels=None):
    position = np.flatnonzero(labels == digit)[example]
    x = torch.from_numpy(images[position:position+1]).to(device)
    y = torch.tensor([digit], device=device)
    eps = torch.randn((1, 784), generator=torch.Generator().manual_seed(20261006)).to(device)
    frames = []
    for log_snr in visual_grid if levels is None else levels:
        t = torch.full((1, 1), 1/(1 + 10**(-log_snr/2)), device=device)
        noisy = t*x + (1-t)*eps
        frames.append(torch.cat([noisy, model(noisy,t,y,False), model(noisy,t,y,True)]).cpu().numpy().reshape(3,28,28))
    return {"digit": digit, "example": example, "clean": x.cpu().numpy().reshape(28,28), "frames": np.asarray(frames)}


preview_dir = ROOT / "runs/mnist_webpage"
preview_dir.mkdir(parents=True, exist_ok=True)
visual_grid = np.linspace(-2, 2, 10)
samples = [prepare_digit(digit, EXAMPLE if digit == DIGIT else 0)
           for digit in dict.fromkeys((7, 3, 8, DIGIT))]
raw_progress = cumulative / cumulative[-1]

stage_targets = np.arange(.05, 1., .10)
stage_levels = []
for target in stage_targets:
    j = np.flatnonzero((progress[:-1] < target) & (progress[1:] >= target))[0]
    stage_levels.append(np.interp(target, progress[j:j+2], grid[j:j+2]))
stage_levels = np.asarray(stage_levels)
interactive_levels = np.unique(np.r_[grid, stage_levels])
interactive_samples = [prepare_digit(s["digit"], s["example"], interactive_levels) for s in samples]
stage_indices = np.array([np.argmin(abs(interactive_levels-level)) for level in stage_levels])
np.savez_compressed(preview_dir / "preview_data.npz", visual_grid=visual_grid,
    grid=grid, risks=risks, gap=gap, density=density, digits=np.array([s["digit"] for s in samples]),
    clean=np.stack([s["clean"] for s in samples]),
    frames=np.stack([s["frames"] for s in samples]), progress=progress,
    raw_progress=raw_progress, progress_density=progress_density,
    stage_targets=stage_targets, stage_levels=stage_levels,
    interactive_levels=interactive_levels,
    interactive_frames=np.stack([s["frames"] for s in interactive_samples]),
    stage_frames=np.stack([s["frames"][stage_indices] for s in interactive_samples]))
print(f"Exported measured MNIST display arrays to {preview_dir / 'preview_data.npz'}")

