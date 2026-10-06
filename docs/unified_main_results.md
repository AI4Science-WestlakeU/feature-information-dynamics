# Paired measurements across representations

Pixel, RAE, SDVAE and VAVAE retain their own upstream training recipes. The common protocol governs measurement identities, not a shared training objective.

```sh
feature-information reproduce --suite DATA/suite.json --output runs/main_results
```

A suite contains all four representations and all eight subsets of class, mask and masked-Canny conditions. Each raw error tensor has axes `[condition subset, SNR, noise repeat, image]`. Classical-chain and Shapley curves are computed from this same tensor. Signed densities and finite-range integrals are retained without fitted tails or peak normalization.

The command checks file hashes, ordered image and condition identities, training/selection/evaluation splits, SNR grids and noise seeds. Noise is identical across conditions within each representation. Each representation records its encoding policy, native training recipe, checkpoint selection and raw/EMA weight choice. Evaluation uses no classifier-free guidance and measures vector squared-error sums.

See [prepared data](data.md) for paired inputs, [experiments](experiments.md) for training and evaluation, and [bundle format](bundle_format.md) for outputs. Canonical JSON identities use `suite.fingerprint`; stored files use byte-level SHA256.

There is no completed qualifying four-representation measurement suite included in this release. Synthetic tests check acceptance and rejection of metadata combinations; they are not paper results. Native CUDA smoke checks are described in [validation](validation.md). Identity checks do not establish Bayes MMSE convergence.
