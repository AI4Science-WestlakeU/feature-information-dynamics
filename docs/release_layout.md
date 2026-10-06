# Public repository contents

The candidate tree was reduced from approximately 1,078 files / 248 MB to 351 files / 23.24 MiB including the prepared-data reader tools. Counts exclude ignored environments, datasets, build products and local archives.

| Keep public | Purpose |
| --- | --- |
| Root notebook and `examples/mnist_class` | Runnable gentle guide, selected teaching checkpoint and measured errors; about 13.2 MiB for the checkpoint/error array |
| Representation source, configurations and upstream patches | Actual training/evaluation routes, grouped by Pixel, RAE, SDVAE and VAVAE |
| Prepared-data download, verification and configuration | Consume existing masks, masked-Canny, paired inputs and optional VAVAE latents |
| `docs/project`, preview generator and vendor assets | Project page, figure provenance and reproducible MNIST display |
| Tests and compact implementation evidence | Check scientific parity, interfaces and numerical semantics |
| MIT license and upstream notices | Project license and retained third-party attribution |

Old Pixel grids, repeated measurement exports, source snapshots, notebook tuning runs, pilot launchers and webpage review screenshots were removed from the public tree. Sixty-seven paths (223.88 MB) were moved to the ignored `.local_archive/release_cleanup`, with an inventory for recovery. No Git history was rewritten.

Raw ImageNet, generated masks, latent caches and training runs belong under ignored `data/` and `runs/`. Virtual environments, build products and local archives are ignored. The small teaching checkpoint is an explicit exception to the model-file ignore rule.

Keep the teaching checkpoint and errors for a fast first run. Keep the webpage preview because the project-page build consumes it. The full raw-image
preparation pipeline is now archived locally; readers start with existing prepared data. Avoid restoring historical development archives into the public tree; retain only new evidence that explains current behavior or a reproducible result.

