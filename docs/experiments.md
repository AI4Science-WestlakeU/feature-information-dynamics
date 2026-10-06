# Representation experiments

Use the notebook for the gentle introduction; use these workflows for the
large-model experiments. The code follows did_q01's actual experiment logic,
with the approved Pixel image-token correction.

## Layout

```text
src/feature_information_dynamics/
    pixel/     model.py, train.py, evaluate.py
    rae/       train.py, evaluate.py
    sdvae/     train.py, evaluate.py
    vavae/     train.py, evaluate.py, extract_latents.py
    data/      shared paired-image and cached-latent loaders
    evaluation.py
    analysis.py
configs/
    pixel/     warmup.yaml, mask.yaml, canny.yaml, fixed_snr.yaml
    rae/       warmup.yaml, mask.yaml, canny.yaml
    sdvae/     warmup.yaml, mask.yaml, canny.yaml
    vavae/     encoder.yaml, warmup.yaml, mask.yaml, canny.yaml
```

The representation-specific conditional models are defined in their training
module, except Pixel's adapter in model.py. Evaluation uses the same model.
workflows.py isolates overlapping upstream module names in separate processes.
It is a small launcher, not a shared training algorithm.

## Scientific routes

| Representation | Input to training | Encoding and objective |
| --- | --- | --- |
| Pixel | RGB images | JiT clean-image output; historical velocity loss or the explicitly selected fixed-time clean-X branch |
| RAE | RGB images | Frozen stage-1 encoder online; native velocity transport |
| SDVAE | RGB images | May online posterior.sample × 0.18215 once; no extra z-score; native SiT velocity transport |
| VAVAE | Pre-extracted latents | Original/flip posterior samples; channel z-score; native LightningDiT velocity/cosine transport |

Only VAVAE has a cache command. SDVAE's April offline experiments remain
historical evidence, not a supported alternate public training route.
See [SDVAE history](sdvae_cache_origin.md).

## Sources and inputs

```sh
pip install -e ".[models]"
```

Install the selected upstream project's dependencies too. Use these source pins:

| Source | Commit | Required patch |
| --- | --- | --- |
| [JiT](https://github.com/LTH14/JiT) | cbc743a2ada5e9762697da2c83f8c4f8379e8c17 | [jit.patch](../patches/upstream/jit.patch) |
| [RAE](https://github.com/bytetriper/RAE) | a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041 | None |
| [SiT](https://github.com/willisma/SiT) | cbde832a40b153ccc79603412409da9c9b0c568c | None |
| [LightningDiT](https://github.com/hustvl/LightningDiT) | f315f25b6aaad600b4d8e50a8167ce06f2e957f3 | [lightningdit-native.patch](../patches/upstream/lightningdit-native.patch) |

From the selected source checkout, apply its patch with:

```sh
git -c core.autocrlf=false apply /path/to/feature-information-dynamics/patches/upstream/PATCH_FILE
```

All four formal experiments start from official pretrained generation checkpoints,
then train warmup → mask → Canny. Follow the [weight preparation guide](weights.md)
to download missing weights and configure the separate representation encoders.
Provide your ImageNet. Download the prepared
mask/Canny shards and paired filename whitelists following the [data guide](data.md),
then configure their paths with `feature-information data configure`. Filename sidecars must align images
across condition and latent shards, and labels retain their ImageNet class IDs.
Public commands do not require did_q01 or private remote paths.

## Stages and training

Fill each configuration's resource paths before running. warmup, mask and Canny
are separate stages; mask loads the appropriate warmup checkpoint, and Canny
loads the mask checkpoint. Do not replace a chained parent with fresh weights.

SDVAE mask uses the actual May online FiLM launch: 64200 steps, early-stop
patience 8 and minimum 30 epochs. Its Canny child uses 64200 steps, patience 8
and minimum 15 epochs. Both disable latent normalization and extra scaling.
Other settings are retained from the original recipe. The Pixel fixed-SNR
configuration is a particular historical child experiment, not a default for
all representations. See [Pixel recipes](../configs/pixel/README.md).

```sh
feature-information pixel train --config configs/pixel/canny.yaml
feature-information rae train --upstream-source /path/to/RAE -- --config configs/rae/warmup.yaml
feature-information sdvae train --upstream-source /path/to/SiT -- --config configs/sdvae/warmup.yaml
feature-information vavae train --upstream-source /path/to/LightningDiT -- --config configs/vavae/warmup.yaml
```

Pixel reads the JiT source path from resources.jit_source in its YAML. Other
representations take --upstream-source before the separator; original trainer
flags follow --. Global batch sizes and distributed settings follow the saved
recipe; changing them changes the experiment.

## VAVAE data

Download the VAVAE component described in the [data guide](data.md). It contains
original/flip sampled latents, globally correct labels, filename sidecars and
channel statistics. No encoder extraction is needed before training. Training
selects an orientation and applies the saved channel statistics; it does not
re-encode or resample the posterior. No 0.18215 scaling is used for VAVAE.

## Evaluate

Each representation retains its original evaluation flags and checkpoint-state
selection. Inspect them with:

```sh
feature-information pixel eval --help
feature-information sdvae eval --upstream-source /path/to/SiT -- --help
feature-information vavae eval --upstream-source /path/to/LightningDiT -- --help
```

All use vector squared-error sums for information analysis. RAE's native time
and velocity sign are converted at the measurement boundary. Evaluating a
clean-feature error does not change training into clean-X prediction. Keep the
checkpoint state, image IDs, noise and time grid consistent across conditions.

## Validation

Source comparison, pinned patch replay and bounded CUDA checks passed. Moving
modules changes imports, not selected training calculations. Current import
checks are recorded separately from older native-layout evidence. Actual full
training, encoder-weight inference on real data, cached normalization on real
images and distributed checkpoint recovery have not been rerun. See
[validation](validation.md) and [layout record](evidence/native_migration/layout.json).
