# Pixel recipes

These configurations preserve executed JiT-L/16 recipes from the research
implementation. Replace `YOUR_PATH` resources before running. They are separate
experiments, not interchangeable training defaults.

- `warmup.yaml`: the production warmup initialized from the official JiT-L/16
  `checkpoint-last.pth`, with class dropout 0.1, a 32100-step ceiling and
  validation-based early stopping. It starts the class-conditioned chain.
- `mask.yaml`: the production class + mask stage, initialized from the warmup
  run's `checkpoints/best.pt`, with class dropout 0, a 32100-step ceiling and
  validation-based early stopping.
- `canny.yaml`: the May 6 large-model chain, initialized from the mask-stage
  checkpoint, with class + mask + Canny active. It retains the original native
  lognormal-time velocity objective, AdamW beta2=0.95, global batch 1024,
  per-step EMA and early stopping. Despite the filename, both spatial conditions
  are active: this follows the actual saved configuration.
- `fixed_snr.yaml`: the September large-model path comparison's fixed-SNR
  child stage, initialized from its raw parent checkpoint. It retains direct
  clean-image MSE at t=0.150979557211, AdamW beta2=0.95, global batch 256,
  per-step EMA, and the 12828-step budget. It does not recreate the parent stage;
  provide that stage's checkpoint before running.

Run `warmup.yaml`, then point `mask.yaml`'s `train.weight_init` at the actual
warmup checkpoint, and finally point `canny.yaml` at the actual mask checkpoint.
Set `resources.jit_source` to the patched JiT source described in
[`../../docs/experiments.md`](../../docs/experiments.md), and fill
the ImageNet root, paired manifests, mask/Canny shard directories and official
checkpoint paths. The historical mask parent was warmup best epoch 3; a rerun
selects its own best validation checkpoint. These are maximum budgets, not
promises of a fixed number of completed training steps.

The warmup and mask recipes were copied from the remote production configs
`m06_pixel_warmup_prod.yaml` and `m06_pixel_mask_prod.yaml`; only source/data/
checkpoint/output paths were replaced. All three stages use native velocity
training, AdamW learning rate 0.0001 and beta2 0.95, and per-step EMA.

The native modules are `feature_information_dynamics.pixel.train` and
`feature_information_dynamics.pixel.evaluate`. Training reads `--config`.
Evaluation retains explicit checkpoint, image-root, paired-whitelist, condition
shard, time-grid and checkpoint-state arguments; `--config_yaml` also supplies
`resources.jit_source`. The top-level command provides the convenient interface.

Input data consists of RGB ImageNet images paired by filename with mask/Canny
safetensor shards. The historical reader applies the same random horizontal flip
to all three training inputs, and disables flips in validation. It preserves the
actual preprocessing implementation, rather than changing it to match old prose.

Training uses the conditional backend's corrected injection into trailing image
tokens after the JiT class tokens are prepended. Loss, sampler, optimizer, EMA,
checkpoint chaining, phase freezing and deterministic validation remain in the
native trainer. Checkpoint evaluation can select either raw model or EMA weights;
its output is a finite-model reconstruction-error estimate.

The existing full fixed-SNR Pixel results and their runner are preserved as a
separate historical experiment. These recipes do not silently adopt that runner's
optimizer, initialization, missing augmentation or noise protocol.
