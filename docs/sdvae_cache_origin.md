# SDVAE and VAVAE feature preparation

The public SDVAE route follows the May SiT experiment: encode RGB online, sample the posterior, and multiply once by 0.18215. There is no channel z-score. The factor is the Stable Diffusion VAE latent scaling convention, also used by the official SiT implementation.

VAVAE follows [LightningDiT](https://github.com/hustvl/LightningDiT/tree/f315f25b6aaad600b4d8e50a8167ce06f2e957f3): ADM crop and normalize RGB, independently sample original and horizontally flipped images, and store both latent arrays plus labels. The public extractor adds filename sidecars for spatial-condition alignment. Training randomly chooses original/flip caches and applies the official channel mean/std normalization. VAVAE does not use 0.18215.

## Historical SDVAE caches

Earlier research runs also used locally extracted SDVAE caches in LightningDiT-compatible safetensors, with later filename sidecars. They were not established to be a downloadable official SiT cache. This historical route is excluded from the public training entrypoints.

Pre-extraction has public precedents: [FastDiT](https://github.com/chuanyangjin/fast-DiT/blob/af841de919a2f2966a0aa13b2be3a24a8b542b85/extract_features.py) stores sampled SDVAE features, while [REPA](https://github.com/sihyun-yu/REPA/tree/67f714503e3892f993844aab088ffc5791c92613/preprocessing) stores posterior parameters and resamples during training. These differ in posterior randomness and should not be treated as the same encoding policy.

See [experiments](experiments.md) for the supported routes and [prepared data](data.md) for the reader download workflow.
