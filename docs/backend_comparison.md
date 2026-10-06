# Implementation comparison

Baseline: did_q01's executed FiLM-advanced experiments, checked locally and on
the research machine on 2026-10-06. Small-model controlled RAE clean-X experiments
are not the large-model baseline.

## Scientific behavior

| Representation | Preserved behavior | Intentional difference |
| --- | --- | --- |
| Pixel / JiT | Historical data transforms and synchronized flip, time sampler, native velocity or explicit fixed-time clean-X branch, optimizer, EMA, freezing, parent checkpoint initialization and evaluation | Spatial conditions target trailing image tokens after class tokens are prepended; this fixes the approved old offset bug |
| RAE | Frozen online stage-1 encoder, native velocity transport, logit-normal time with native shift, optimizer/EMA, checkpoint and measurement time/sign conventions | Public resource paths and imports |
| SDVAE | Posterior sampling times 0.18215 once, no extra z-score, native SiT velocity transport and condition-stage class/dropout settings | Public resource paths and imports |
| VAVAE | Sampled original/flip cache, channel normalization, native velocity/cosine transport, time sampling and EMA | Public paths/imports; cache loader now honors the explicitly supplied checkpoint instead of silently taking a YAML default |

The public SDVAE route is the May online experiment: sampled posterior times
0.18215 once, no additional normalization. VAVAE retains the original offline
sampled-cache/channel-statistics workflow. April SDVAE offline code is no longer
an active public route; its provenance remains in [cache history](sdvae_cache_origin.md).

Implementations are grouped under [pixel](../src/feature_information_dynamics/pixel/),
[rae](../src/feature_information_dynamics/rae/),
[sdvae](../src/feature_information_dynamics/sdvae/) and
[vavae](../src/feature_information_dynamics/vavae/). Matching configurations are
in [configs](../configs/); see the [experiment guide](experiments.md).
Moving files changes import paths, not selected scientific calculations.

## Code and execution checks

Independent source comparisons check computation ASTs, including data transforms,
losses, initialization, EMA and measurement. They exclude import relocation and
explanatory text. Upstream modifications affecting actual execution are retained
as patches against exact source pins, rather than silently substituting a clean
but different upstream implementation.

Independent source, upstream-patch and reader reviews have passed within the
migration scope. Pixel token alignment passed against real JiT blocks on CUDA.
Pixel, RAE, VAVAE, and both SDVAE routes completed bounded CUDA forward, native
loss, backward, optimizer and EMA checks; SiT used its full XL architecture.
Before route selection, eleven latent train/cache/evaluation entrypoints imported successfully with the
published LightningDiT patch replayed against its official pin. See
[migration evidence](evidence/native_migration/). Full large-model training,
real-data distributed resume and a complete four-representation remeasurement
have not been performed by this migration. After route selection and directory
reorganization, current CLI import checks are recorded in
[layout_imports.json](evidence/native_migration/layout_imports.json).

The suite validator now accepts valid step-0 checkpoints, repeated checkpoint use
across SNR, parent-checkpoint lineage and declared historical Torch-noise protocols.
Saved empty masks remain zero masks. Error units remain vector squared-error sums;
information density is 0.5 gamma times the error difference, with ln(10) when
integrating a log10 SNR grid. Signed raw estimates remain available.

## Historical alternate experiments

Historical fixed-SNR Pixel tuning results used a different batch/optimizer/
initialization/EMA/augmentation recipe. Their runners and raw archives are
excluded from the public tree and retained locally; they are not the public
representation entry. No historical result was relabelled as a new execution.

The old optional RAE cache extractor was not used by the actual online large-model
route and compressed class labels after filtering. It is not part of the native
public workflow. No change to RAE's actual online training is needed for that issue.

No Git commit, push or publication has been performed.
