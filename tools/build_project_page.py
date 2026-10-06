"""Assemble the local project page from the verified demo and paper assets."""
from pathlib import Path
import hashlib
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
preview = (ROOT / "docs/preview/index.html").read_text(encoding="utf-8")
style = re.search(r"<style>(.*?)</style>", preview, re.S).group(1)
start = preview.index('<section class="panel" aria-label="Interactive denoising demonstration">')
end = preview.index('<section class="panel explanation">', start)
demo = preview[start:end]
script = re.search(r"<script>(.*?)</script>", preview, re.S).group(1)
template = (ROOT / "tools/project_page.html").read_text(encoding="utf-8")
html = template.replace("/*DEMO_STYLE*/",style).replace("<!--DEMO-->",demo).replace("/*DEMO_SCRIPT*/",script)
math_assets = '\n'.join(re.findall(r'<(?:link|script)\b[^>]*(?:href|src)="vendor/[^>]+>(?:</script>)?', preview))
html = html.replace('</head>', math_assets + '\n</head>')
html = html.replace('loading="lazy"' , 'decoding="async"')
html = html.replace('−5 · mostly noise', '−5 · noise').replace('3 · mostly signal', '3 · signal')
destination = ROOT / "docs/project"
destination.mkdir(parents=True,exist_ok=True)
shutil.copytree(ROOT / "docs/vendor", destination / "vendor", dirs_exist_ok=True)
shutil.copy2(ROOT / "mnist_information_concepts.ipynb", destination / "assets/mnist_information_concepts.ipynb")
(destination / "index.html").write_text(html,encoding="utf-8")
assets = [{"file":p.relative_to(destination).as_posix(),"bytes":p.stat().st_size,
           "sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
          for p in sorted([*(destination / "assets").rglob("*"), *(destination / "vendor").rglob("*")]) if p.is_file()]
(destination / "assets_manifest.json").write_text(json.dumps({"assets":assets,
    "demo_source":"../preview/index.html",
    "paper_source":"did_paper/paper/output/pdf/camera_ready_final_style_redline_20260929.pdf",
    "reference_design":"https://hongcanguo.github.io/Cola-DLM/"},indent=2)+"\n",encoding="utf-8")
print(f"Built {destination / 'index.html'} ({len(html.encode('utf-8'))} bytes)")
