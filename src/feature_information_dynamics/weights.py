"""Download a specified upstream checkpoint before starting an experiment."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import urllib.request

DEFAULT_URLS = {
    "pixel": "https://www.dropbox.com/scl/fo/3ken1avtsd81ip67b9qpi/AGEqmJoyiacjYKSG_DnOX-c/jit-l-16/checkpoint-last.pth?rlkey=14gjrblmljewpl6ygxzlr3njm&dl=1",
    "rae": "https://huggingface.co/nyu-visionx/RAE-collections/resolve/1be4f03273523431f099a934da4cf1940dc6039f/DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-XL/stage2_model.pt",
    "sdvae": "https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=1",
    "vavae": "https://huggingface.co/hustvl/lightningdit-xl-imagenet256-800ep/resolve/main/lightningdit-xl-imagenet256-800ep.pt",
}
RAE_SHA256 = "fa5e0b0d4b1977a59908a87ec4c3c8a67ba2372f570028924fac24850f9458ed"
PIXEL_SHA256 = "5daaffa1eb733c55518eac10b609483f9e0ff454a065947c98a8685459852de7"


def check_training_checkpoint(arguments: list[str]) -> None:
    """Check public training launches before importing any GPU dependencies."""
    if "--help" in arguments or "-h" in arguments:
        return
    config = None
    for index, argument in enumerate(arguments):
        if argument == "--config" and index + 1 < len(arguments):
            config = arguments[index + 1]
        elif argument.startswith("--config="):
            config = argument.split("=", 1)[1]
    if not config:
        raise ValueError("Formal training requires an explicit --config with train.weight_init; see docs/weights.md.")
    import yaml
    recipe = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    parent = recipe.get("train", {}).get("weight_init")
    if not parent or "YOUR_PATH" in str(parent) or not Path(parent).is_file():
        raise ValueError("Missing train.weight_init checkpoint. Download official weights for warmup, or supply the preceding stage's checkpoint for mask/Canny; see docs/weights.md.")


def download(url: str, output: Path, expected_sha256: str | None = None) -> Path:
    """Publish only a complete response; never overwrite an existing checkpoint."""
    if not url.startswith("https://"):
        raise ValueError("Use an HTTPS direct checkpoint URL.")
    if output.exists():
        raise FileExistsError(f"Already exists; use that checkpoint or a new destination: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    try:
        with temporary.open("xb") as target:
            with urllib.request.urlopen(url, timeout=60) as response:
                content_type = response.headers.get("Content-Type", "").lower()
                if "text/html" in content_type:
                    raise ValueError("URL returned a webpage; provide a direct checkpoint download link.")
                shutil.copyfileobj(response, target)
                expected = response.headers.get("Content-Length")
                if target.tell() == 0 or (expected and target.tell() != int(expected)):
                    raise ValueError("Incomplete checkpoint download.")
        if expected_sha256:
            digest = hashlib.sha256()
            with temporary.open("rb") as downloaded:
                for block in iter(lambda: downloaded.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected_sha256:
                raise ValueError("Downloaded checkpoint SHA256 does not match the selected official version.")
        # Hard link is atomic and fails if another process created the destination.
        os.link(temporary, output)
    finally:
        # If exclusive open failed, the partial file belongs to another process.
        if 'target' in locals():
            temporary.unlink(missing_ok=True)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("representation", choices=["pixel", "rae", "sdvae", "vavae"])
    parser.add_argument("--url", help="Override the official checkpoint URL")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    url = args.url or DEFAULT_URLS.get(args.representation)
    if not url:
        parser.error("Provide --url for the exact official checkpoint selected for this experiment; see docs/weights.md.")
    checksum = {"rae": RAE_SHA256, "pixel": PIXEL_SHA256}.get(args.representation) if not args.url else None
    print(download(url, args.output, checksum))
