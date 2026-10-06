# Run with prepared data

Start with prepared masks, masked-Canny and the paired image lists. SAM annotation,
mask extraction and VAVAE encoding are not prerequisites for this workflow.

Prepared data is hosted on [Hugging Face](https://huggingface.co/datasets/AI4Science-WestlakeU/feature-information-dynamics), in the organization's [feature-information-dynamics collection](https://huggingface.co/collections/AI4Science-WestlakeU/feature-information-dynamics-6ac4bfa34224063cb603a7dc). The common component and its verified manifest are published. See the dataset page and release manifest for current availability of the optional VAVAE component; it is published after payload verification. No ImageNet-derived arrays are included in the Git repository.

## What to download

| Component | Contents | Used by |
| --- | --- | --- |
| `common` (default) | Mask/Canny shards and filename sidecars, paired train/validation lists, full 1000-class mapping and release hashes | All four representations |
| `vavae` (optional) | Original/flip sampled latent shards, filename sidecars, labels and channel statistics | VAVAE only |

Keep your ImageNet training images under `/data/imagenet/train/SYNSET/FILE.JPEG`.
Keep all 1000 class directories so the class IDs retain the original ordering.
Pixel, RAE and SDVAE encode/use those RGB images online. VAVAE training reads its
latent component. Evaluation may also need the original images for sample/label
selection, so keep the same ImageNet installation.

Historical uncompressed sizes are approximately 85.1 GB for mask/Canny tensors
and 42.5 GB for VAVAE tensors. Metadata adds roughly 41 MB to common. Compression
reduces download traffic but not the installed tensor sizes. The common download
is approximately 3.34 GB; the optional VAVAE download is approximately 39.50 GB. Gzip installation also needs
space for one downloaded shard and its temporary decompressed copy.

## Download and verify

```sh
python -m pip install -e '.[data]'
feature-information data download --manifest https://huggingface.co/datasets/AI4Science-WestlakeU/feature-information-dynamics/resolve/main/release.json --output data/prepared
```

The default downloads only `common`. Files are installed atomically; rerunning
reuses files with matching size and SHA256. A gzip payload has separate download
and installed hashes, and is decompressed into the original safetensors format.
To add VAVAE later:

```sh
feature-information data download --manifest https://huggingface.co/datasets/AI4Science-WestlakeU/feature-information-dynamics/resolve/main/release.json --output data/prepared --component vavae
feature-information data verify --manifest data/prepared/release.json --output data/prepared --component all
```

If files were downloaded by another tool, place them in the layout below and run
`verify` against their release manifest. A local payload source can also be used:
`--manifest /downloads/release.json --source-base /downloads --output data/prepared`.

```text
prepared/
    release.json
    paired/
        class_ids.json          full synset -> global class ID mapping
        sr95.json
        train.json
        val.json
    mask_shards/masks_rankRR_shardSSS.safetensors
    canny_shards/cannys_rankRR_shardSSS.safetensors
    canny_shards/cannys_rankRR_shardSSS.json
    vavae_latents/              optional
        latents_rankRR_shardSSS.safetensors
        latents_rankRR_shardSSS.json
        latents_stats.pt
```

Verification checks filenames, row alignment, class identities, split separation
and declared invalid-map counts, in addition to hashes. Historical mask shards
need no own sidecar or embedded labels: the matching Canny sidecar and full class
mapping supply that information. Invalid mask sentinels are retained and reported;
they are not silently removed from the experiment population. Historical paired
lists contained one overlapping image. This release keeps its training occurrence
and explicitly removes its validation occurrence: 547448 training and 66685
validation images (614133 unique images). The manifest records this correction.

## Configure once, then train the stage chain

Install `.[models]` and the chosen pinned upstream dependency stack as described in
[experiments](experiments.md). Download that project's official pretrained weights.
Copy [resources.example.json](../configs/resources.example.json) to a local file,
then fill your data, source, weight and output paths. RAE additionally needs a
copy of [model.yaml](../configs/rae/model.yaml) with its stage-1 resource paths filled:
DINOv2 encoder directory, decoder config/weights and feature normalization stats.

```sh
feature-information data configure --manifest data/prepared/release.json --data-root data/prepared --resources resources.local.json --configs configs/pixel/warmup.yaml configs/pixel/mask.yaml configs/pixel/canny.yaml --output runs/configured
```

List every desired stage in one invocation: common data is verified once per
call. Configure other representations by replacing the `configs/pixel/...` files
with their `rae`, `sdvae` or `vavae` counterparts. The command writes new YAMLs;
it changes resource paths, not the training recipe. It reports unfilled resources.
A populated future parent path does not mean its checkpoint already exists.
`fid_reference_file` is an unused historical field in these trainers and need not
be supplied to run the chain.

```sh
accelerate launch --multi_gpu --num_processes 8 --mixed_precision bf16 --module feature_information_dynamics.pixel.train --config runs/configured/pixel/warmup.yaml
accelerate launch --multi_gpu --num_processes 8 --mixed_precision bf16 --module feature_information_dynamics.pixel.train --config runs/configured/pixel/mask.yaml
accelerate launch --multi_gpu --num_processes 8 --mixed_precision bf16 --module feature_information_dynamics.pixel.train --config runs/configured/pixel/canny.yaml
```

Finish warmup before mask, and mask before Canny. The next stage's `weight_init`
must point to the preceding stage's actual `checkpoints/best.pt`, including its
experiment subdirectory. The example launch uses eight GPUs. The generated configuration retains the
original global batch size and gradient accumulation; provide enough GPU memory
for its per-device batch rather than silently shrinking the experiment budget.
RAE/SDVAE/VAVAE use the same sequence with their launcher source argument, e.g.:

```sh
accelerate launch --multi_gpu --num_processes 8 --mixed_precision bf16 --module feature_information_dynamics.workflows sdvae train --upstream-source /opt/SiT --in-process -- --config runs/configured/sdvae/warmup.yaml
```

See [experiments](experiments.md) for model pins and evaluation commands, and
[validation](validation.md) for the actual checked scope. The historical RGB reader
was tested on real prepared data; this is not a claim that all full training runs
have been repeated in a fresh downloaded installation.

## Maintainer: publish an existing data layout

`tools/build_data_release_manifest.py` inventories an already prepared directory
with the layout above, checks paired alignment and records per-file hashes. It
does not annotate images, encode latents, move files or upload anything:

```sh
python tools/build_data_release_manifest.py --data-root /existing/prepared --release-id imagenet-sr95-v1 --output /existing/prepared/release.json
```

The source lists must already be unique and disjoint. Missing conditions or an
undeclared class mapping are errors. Publish this manifest alongside its relative
file paths, then insert its real URL in the reader instructions. Plain files are
supported; compressed delivery additionally declares `encoding: gzip`, source path,
compressed byte count and SHA256. Installed array bytes remain unchanged.
