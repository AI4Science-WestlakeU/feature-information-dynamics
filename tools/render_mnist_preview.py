"""Render raw density and monotone class progress; retain signed integral data."""
from pathlib import Path
import base64
import io
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt

ROOT = globals().get("ROOT", Path(__file__).resolve().parents[1])
assets = ROOT / "runs/mnist_webpage"
data = np.load(assets / "preview_data.npz")
grid, density, progress = data["grid"], data["density"], data["progress"]
levels, targets = data["stage_levels"], data["stage_targets"]
raw = np.r_[0., np.cumsum(.5*(density[1:]+density[:-1])*np.diff(grid)*np.log(10))]
np.testing.assert_allclose(data["raw_progress"], raw/raw[-1])
np.testing.assert_allclose(data["progress_density"], np.maximum(density, 0.))
positive = data["progress_density"]
area = np.r_[0., np.cumsum(.5*(positive[1:]+positive[:-1])*np.diff(grid)*np.log(10))]
np.testing.assert_allclose(progress, area/area[-1])
assert np.all(np.diff(progress) >= 0) and progress[0] == 0 and progress[-1] == 1
np.testing.assert_allclose(np.interp(levels, grid, progress), targets)

def png(array):
    buffer = io.BytesIO()
    plt.imsave(buffer,array,cmap="gray",vmin=-1,vmax=1,format="png")
    return "data:image/png;base64,"+base64.b64encode(buffer.getvalue()).decode()

samples = [{"digit": int(d), "clean": png(c), "frames": [[png(p) for p in f] for f in fs]}
           for d,c,fs in zip(data["digits"],data["clean"],data["interactive_frames"])]
payload = json.dumps({"levels":data["interactive_levels"].tolist(),"targets":targets.tolist(),
                     "stageLevels":levels.tolist(),"grid":grid.tolist(),"density":density.tolist(),
                     "risks":data["risks"].tolist(),"gap":data["gap"].tolist(),
                     "progress":progress.tolist(),"rawProgress":data["raw_progress"].tolist(),
                     "progressDensity":positive.tolist(),"samples":samples})
template = (ROOT / "tools/mnist_preview.html").read_text(encoding="utf-8")
target = ROOT / "docs/preview/index.html"
target.parent.mkdir(parents=True,exist_ok=True)
shutil.copytree(ROOT / "docs/vendor", target.parent / "vendor", dirs_exist_ok=True)
target.write_text(template.replace("/*DATA*/",payload),encoding="utf-8")
plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False})
for sample_id,digit in enumerate(data["digits"]):
    if not globals().get("SHOW_FIGURES", True):
        break
    if digit != globals().get("DISPLAY_DIGIT", 3):
        continue
    frames = data["stage_frames"][sample_id]
    fig, axes = plt.subplots(3,10,figsize=(16,5))
    for col,(fraction,level) in enumerate(zip(targets,levels)):
        for row in range(3):
            axes[row,col].imshow(frames[col,row],cmap="gray",vmin=-1,vmax=1)
            axes[row,col].set_xticks([]); axes[row,col].set_yticks([])
        axes[0,col].set_title(f"{fraction:.0%}\nlog SNR {level:.2f}",fontsize=10)
    for row,name in enumerate(["Noisy","Unconditional",f"Class {digit}"]):axes[row,0].set_ylabel(name)
    fig.suptitle(f"Digit {digit} | class information accumulation",fontsize=14)
    fig.tight_layout();fig.savefig(assets/f"progress_sheet_{digit}.png",bbox_inches="tight");plt.show()

    fig = plt.figure(figsize=(11,6))
    chosen = 4
    pictures = [data["clean"][sample_id],*frames[chosen]]
    for j,(picture,title) in enumerate(zip(pictures,["Clean reference","Noisy observation","Unconditional",f"Class {digit}"])):
        ax=fig.add_subplot(2,4,j+1);ax.imshow(picture,cmap="gray",vmin=-1,vmax=1)
        ax.set_title(title);ax.axis("off")
    for j,(curve,color,title,ylabel) in enumerate([
        (positive,"#087f8c","Where information accumulates","Density per ln SNR"),
        (area,"#ba6515","Accumulated class information (clipped estimate)","nats")]):
        ax=fig.add_subplot(2,2,3+j);ax.plot(grid,curve,color=color,linewidth=2)
        ax.axvline(levels[chosen],color="#303847",linewidth=1,linestyle="--")
        ax.plot(levels[chosen],np.interp(levels[chosen],grid,curve),'o',color=color)
        ax.set(xlabel="log10 SNR",ylabel=ylabel,title=title);ax.grid(alpha=.15)
    fig.suptitle(f"Digit {digit} | {targets[chosen]:.0%} of measured information | log10 SNR {levels[chosen]:.2f}")
    fig.tight_layout();fig.savefig(assets/f"linked_panel_{digit}.png",bbox_inches="tight");plt.show()
print("Offline interactive preview and two selected-class progress-layout figures saved.")
