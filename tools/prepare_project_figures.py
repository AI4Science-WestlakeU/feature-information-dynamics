"""Render exactly the PDFs referenced by the manuscript; preserve curve pixels."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

parser=argparse.ArgumentParser()
parser.add_argument("--paper-directory",type=Path,required=True)
parser.add_argument("--renderer",default="pdftoppm")
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
assets=root/"docs/project/assets"
assets.mkdir(parents=True,exist_ok=True)
sources={"spectral-mnist":"mnist_freq_panorama_paper.pdf",
         "spectral-cifar":"cifar10_freq_panorama_paper.pdf",
         "convergence":"repr_unconditional_fid.pdf",
         "representations":"repr_layer_curves.pdf"}
provenance={"source_rule":"PDFs actually referenced by experiments.tex, not same-name historical PNGs",
            "sources":{},"crops":{}}
for destination,name in sources.items():
    source=args.paper_directory/"figures"/name
    subprocess.run([args.renderer,"-png","-singlefile","-scale-to","2200",
                    str(source),str(assets/destination)],check=True)
    provenance["sources"][destination+".png"]={"paper_file":"figures/"+name,
                                              "sha256":hashlib.sha256(source.read_bytes()).hexdigest()}

image=Image.open(assets/"representations.png").convert("RGB")
array=np.asarray(image)
black=(array<70).all(axis=2)
counts=black[:,int(image.width*.12):int(image.width*.98)].sum(axis=1)
rows=np.flatnonzero(counts>image.width*.78)
groups=np.split(rows,np.flatnonzero(np.diff(rows)>1)+1)
assert len(groups)==8,"Expected four panels with black top and bottom borders"
borders=[int(g[0]) for g in groups]
left=int(image.width*.05)
axis_top=int(groups[-1][-1])+3
axis=image.crop((left,axis_top,image.width,image.height))
for i,name in enumerate(["pixel","sdvae","vavae","rae"]):
    top=max(0,borders[2*i]-int(image.height*.032))
    bottom=int(groups[2*i+1][-1])+3
    panel=image.crop((left,top,image.width,bottom))
    canvas=Image.new("RGB",(panel.width,panel.height+axis.height),"white")
    canvas.paste(panel,(0,0));canvas.paste(axis,(0,panel.height))
    canvas.save(assets/f"representation-{name}.png")
    provenance["crops"][f"representation-{name}.png"]={"panel":[left,top,image.width,bottom],
        "shared_axis":[left,axis_top,image.width,image.height],
        "operation":"Original panel and original shared x axis concatenated; curves unchanged"}
target=root/"docs/project/figure_sources.json"
target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(json.dumps(provenance,indent=2)+"\n",encoding="utf-8")
print("Rendered four manuscript PDFs and four lossless panel crops")
