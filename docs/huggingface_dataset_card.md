---
language:
- en
task_categories:
- image-segmentation
- image-to-image
size_categories:
- 100K<n<1M
---

# Feature Information Dynamics: prepared ImageNet conditions

Prepared spatial conditions and paired sample metadata for Feature Information
Dynamics in Diffusion. The common component contains segmentation masks,
masked-Canny maps, filename sidecars, paired train/validation lists and a full
ImageNet class-ID mapping. It contains no RGB images or VAVAE latent arrays.
The optional VAVAE component contains sampled original/flip latents, globally
correct labels, filename sidecars and channel mean/std statistics.

Common is published. The optional VAVAE payload upload is still in progress;
its entries will be added to the manifest after remote identity checks.

## Usage

Obtain ImageNet training images separately and retain all 1000 class directories.
Install the companion Feature Information Dynamics repository, then use:

```sh
python -m pip install -e '.[data]'
feature-information data download --manifest https://huggingface.co/datasets/AI4Science-WestlakeU/feature-information-dynamics/resolve/main/release.json --output data/prepared
```

The release manifest is published after its payloads have been uploaded and verified. Common is installed by default;
add `--component vavae` for the offline VAVAE cache. Pixel, RAE and online SDVAE
do not require the VAVAE component. Follow the code repository's `docs/data.md`
for weight resources, configuration generation and stage-chain execution.

## Files and identity

Each release file has a relative installed path, byte count and SHA256. Compressed
payloads additionally identify the source path, compressed byte count and SHA256.
The installer restores the original tensor bytes. Keep the supplied filename order
and global class IDs; do not renumber retained SR95 classes or substitute a random
image-level split.

The common component has 196 payloads: 3,344,352,304 download bytes and
85,089,831,558 installed bytes. VAVAE has 129 payloads: 39,500,888,780
download bytes and 42,548,211,383 installed bytes. Actual payload sizes and split counts must be taken from the
published release manifest, not these rounded historical totals. The release contains 547448 training and 66685 validation images (614133 unique).
It retains the 980 invalid mask sentinels and removes the validation occurrence
of the historical one-image train/validation overlap; the manifest records this correction.

## Attribution

The images originate from ImageNet. Masks were generated using SAM3.1;
VAVAE encoding follows LightningDiT. Please cite the original dataset/models
and Feature Information Dynamics when using this release. The companion code's
MIT license does not replace upstream dataset/model terms.

## Release verification

Each release records original and compressed SHA256 checksums and exact split
counts. The installer validates those identities and spatial/latent alignment.
Repository code checks and bounded real-data reader checks are documented in the
companion code repository. They do not claim a new full training rerun.
