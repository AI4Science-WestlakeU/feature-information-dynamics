# Feature Information Dynamics in Diffusion

Code for the paper's conditional denoising experiments, with a
[gentle MNIST notebook](mnist_information_concepts.ipynb) explaining the project
page's class-information example. The notebook displays one digit class at a
time and loads a small model trained from scratch plus measured full-test errors.

## Start with the notebook

```sh
python -m pip install -e ".[mnist,notebook]"
jupyter lab mnist_information_concepts.ipynb
```

The notebook downloads MNIST on first use. Its default path runs on CPU without
repeating model training. From-scratch training and independent checkpoint
measurement are linked at the end. The recorded full training took about
31 minutes on one A100; that is not a laptop timing estimate.

The [project page](docs/project/index.html) uses the same measured MNIST errors,
clipped displayed density and finite-range accumulation. This teaching example
is separate from the paper's spectral MNIST experiment.

## Representation experiments

| Representation | Feature preparation |
| --- | --- |
| Pixel / JiT | RGB images |
| RAE | Frozen stage-1 encoder online |
| SDVAE / SiT | May online posterior sampling, multiplied once by 0.18215; no extra z-score |
| VAVAE / LightningDiT | Original/flip posterior samples cached offline, then channel normalization |

Each model retains its own loss, time sampler, optimizer, EMA and checkpoint
chain. Pixel spatial conditions target trailing image tokens after class tokens
are prepended. Source files and configurations are grouped by representation:

```text
src/feature_information_dynamics/
    pixel/     rae/     sdvae/     vavae/
    data/      shared preparation and paired loaders
    evaluation.py
    analysis.py
configs/
    pixel/     rae/     sdvae/     vavae/
```

Start from the [prepared-data guide](docs/data.md): download the common mask/Canny
shards and paired filename lists, then configure their local paths. Pixel, RAE and
SDVAE read your ImageNet RGB online; none requires VAVAE latents or SAM annotation.
VAVAE uses a separate optional latent download. Prepared data is hosted on
[Hugging Face](https://huggingface.co/datasets/AI4Science-WestlakeU/feature-information-dynamics);
the common component is published. See the dataset page for current availability of the optional VAVAE component. The repository does not include the ImageNet-derived arrays.

Then follow the [experiment guide](docs/experiments.md) for exact upstream pins,
patches, weights and stage configurations. For example:

```sh
python -m pip install -e ".[models]"
feature-information sdvae train --upstream-source /path/to/SiT -- --config configs/sdvae/mask.yaml
feature-information vavae train --upstream-source /path/to/LightningDiT -- --config configs/vavae/mask.yaml
```

Fill explicit resource paths first. A mask or Canny child stage requires its
specified parent checkpoint. Official backbone/encoder weights and ImageNet
are obtained separately.

## Measurement and analysis

Errors are vector squared-error sums. Information density is half the SNR times
the conditional-error difference; integration over log10 SNR includes ln(10).
Signed raw estimates are retained. The teaching display clips negative estimates
caused by fitting and sampling error; its raw integral remains a diagnostic.

```sh
python -m pip install -e .
feature-information analyze --bundle examples/mnist_class/measurements --output runs/mnist_analysis
```

See the [bundle format](docs/bundle_format.md) and
[paired four-representation protocol](docs/unified_main_results.md).
Finite-model error estimates do not prove Bayes MMSE convergence.

## Verification and license

```sh
python -m unittest discover -s tests -v
```

[Validation](docs/validation.md) covers source comparisons, real-model bounded
CUDA checks, current command imports and the notebook run from a fresh directory.
Complete production training and real-data distributed recovery were not rerun
from this public layout.

Project-specific code is [MIT licensed](LICENSE). Upstream notices are retained
in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Historical tuning runs, raw
measurement archives, screenshots and temporary launch scripts are excluded
from the public tree; see the [release-layout note](docs/release_layout.md).
