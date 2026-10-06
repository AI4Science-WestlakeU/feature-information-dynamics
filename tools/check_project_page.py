"""Check project-page anchors, local assets and asset hashes with the stdlib."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
from urllib.request import urlopen
import argparse
import hashlib
import json
import re

import numpy as np

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--url", help="Optional local HTTP origin to check actual asset responses")
args = parser.parse_args()
page = root / "docs/project"

class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.ids=[]; self.references=[]
    def handle_starttag(self,tag,attrs):
        attributes=dict(attrs)
        if "id" in attributes: self.ids.append(attributes["id"])
        for key in ("href","src"):
            if key in attributes: self.references.append(attributes[key])

document=Links();document.feed((page/"index.html").read_text(encoding="utf-8"))
assert len(document.ids)==len(set(document.ids)),"Duplicate IDs"
local=set();anchors=set()
for ref in document.references:
    split=urlsplit(ref)
    if split.scheme: continue
    if not split.path:
        assert split.fragment in document.ids,ref
        anchors.add(split.fragment)
    else:
        path=(page/unquote(split.path)).resolve()
        assert path.is_relative_to(page.resolve()),ref
        assert path.is_file(),ref
        local.add(split.path)
manifest=json.loads((page/"assets_manifest.json").read_text(encoding="utf-8"))
for item in manifest["assets"]:
    path=page/item["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest()==item["sha256"],item["file"]
http=[]
if args.url:
    for ref in sorted(local):
        with urlopen(args.url.rstrip("/")+"/"+ref,timeout=10) as response:
            assert response.status==200
            assert len(response.read())==(page/ref).stat().st_size
            http.append(ref)
report={"unique_ids":len(document.ids),"anchors":sorted(anchors),
        "local_assets":sorted(local),"manifest_hashes_checked":len(manifest["assets"]),
        "http_200_assets":http,"passed":True}
export = np.load(root / "runs/mnist_webpage/preview_data.npz")
from feature_information_dynamics.bundle import load_bundle
measured_errors, measured_metadata, _ = load_bundle(root / "examples/mnist_class/measurements")
losses = measured_errors.astype(np.float64).mean(axis=(2, 3))
np.testing.assert_array_equal(export["grid"], measured_metadata["log10_snr"])
np.testing.assert_array_equal(export["risks"], losses)
np.testing.assert_array_equal(export["gap"], losses[0] - losses[1])
np.testing.assert_array_equal(export["density"], .5 * 10.0**export["grid"] * export["gap"])

for filename in ("docs/preview/index.html", "docs/project/index.html"):
    html = (root / filename).read_text(encoding="utf-8")
    payload = json.loads(re.search(r"const data=(.*?);\s*\n", html, re.S).group(1))
    for key, source in {
        "grid": "grid", "risks": "risks", "gap": "gap", "density": "density",
        "progressDensity": "progress_density", "progress": "progress",
        "rawProgress": "raw_progress", "levels": "interactive_levels",
        "stageLevels": "stage_levels",
    }.items():
        np.testing.assert_array_equal(payload[key], export[source])
    assert "chart('plot',data.progressDensity" in html
    assert "data.progressDensity[i-1]+data.progressDensity[i]" in html
    assert "neural-denoiser fitting error" in html
np.testing.assert_array_equal(export["progress_density"], np.maximum(export["density"], 0))
area = np.r_[0., np.cumsum(.5 * (export["progress_density"][1:]
    + export["progress_density"][:-1]) * np.diff(export["grid"]) * np.log(10))]
np.testing.assert_allclose(export["progress"], area / area[-1])
assert (root / "mnist_information_concepts.ipynb").read_bytes() == (
    page / "assets/mnist_information_concepts.ipynb").read_bytes()
report["mnist_semantics"] = {
    "numeric_arrays_match_shared_verified_measurements": True,
    "density": "clip negative fitting-error estimates to zero",
    "accumulation": "trapezoidal clipped finite-range integral in nats",
    "curve_population": "all 10000 test images; selected class changes images only",
    "notebook_download_identical": True,
    "display_total_nats": float(area[-1]),
}
destination=root/"docs/evidence/project_page/link_check.json"
destination.parent.mkdir(parents=True,exist_ok=True)
destination.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report))
