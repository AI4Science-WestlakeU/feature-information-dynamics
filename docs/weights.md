# Pretrained weights for the formal experiments

All four formal ImageNet experiments fine-tune an official pretrained generation
model. They do not train the backbone from scratch. The MNIST teaching notebook
is a separate, small experiment that trains its own denoisers.

The initialization chain is **official checkpoint → warmup → mask → Canny**.
New condition modules are initialized separately. Set `train.weight_init` to the
official checkpoint for warmup, the warmup checkpoint for mask, and the mask
checkpoint for Canny. `train.ckpt` is for resuming a run, not choosing this parent.
The public training command checks that `train.weight_init` exists before launching
the trainer; missing weights produce an error instead of starting from scratch.

## Download before training

Already have the correct weights? Reuse them and fill `configs/resources.example.json`.
Otherwise install the package and download to your model directory:

```sh
feature-information weights sdvae --output models/SiT-XL-2-256.pt
feature-information weights vavae --output models/lightningdit-xl-imagenet256-800ep.pt
feature-information weights rae --output models/RAE/stage2_model.pt
feature-information weights pixel --output models/JiT/checkpoint-last.pth
```

These download the official SiT-XL/2 256 and LightningDiT-XL 800-epoch generation
checkpoints. They can be large; download is an explicit preparation step. Existing
files are preserved. A failed transfer does not leave a checkpoint at the final path;
rerun the command to restart the download.

RAE uses the exact upstream path recorded in the historical warmup configuration:
`DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-XL/stage2_model.pt` in
`nyu-visionx/RAE-collections`, pinned at revision
`1be4f03273523431f099a934da4cf1940dc6039f`. This is the `DiTDH-XL`
release, not `DiTDH-XL_ep20` or `DiTDH-XL_ep80`. The default RAE download
verifies SHA256 `fa5e0b0d4b1977a59908a87ec4c3c8a67ba2372f570028924fac24850f9458ed`
before making the checkpoint available (3,355,251,584 bytes).
The actual historical server checkpoint was hashed and matched this official
SHA256 exactly; this identifies the file independently of its local name.

Pixel defaults to the exact **JiT-L/16, 256×256**
[checkpoint file](https://www.dropbox.com/scl/fo/3ken1avtsd81ip67b9qpi/AGEqmJoyiacjYKSG_DnOX-c/jit-l-16/checkpoint-last.pth?rlkey=14gjrblmljewpl6ygxzlr3njm&dl=0)
inside the official directory linked by the [JiT repository](https://github.com/LTH14/JiT).
The download URL targets that file directly, not the entire folder.
Its reported size is 5,510,063,042 bytes. A bounded 16-byte range request verified
that the link returns a binary PyTorch ZIP container, without downloading the model.
The historical server checkpoint has the same byte count and SHA256
`5daaffa1eb733c55518eac10b609483f9e0ff454a065947c98a8685459852de7`.
The default Pixel download checks that hash before publishing the file. This
release preparation did not download the full upstream Pixel file again.

For a custom `--url`, the downloader retrieves the supplied file without the
default Pixel/RAE identity checks. Record the selected upstream URL and retain the
correct version when reproducing an existing experiment.

## Representation encoders are separate

- **Pixel:** no pretrained representation encoder is needed.
- **SDVAE:** the online encoder requires `stabilityai/sd-vae-ft-ema`. With the
  Hugging Face CLI, run `hf download stabilityai/sd-vae-ft-ema --local-dir models/sd-vae-ft-ema`
  and point `data.vae_path` at it.
- **RAE:** fill the encoder, decoder and normalization paths in
  [model.yaml](../configs/rae/model.yaml). DINOv2 uses
  `facebook/dinov2-with-registers-base`; the matching decoder and statistics are
  in `nyu-visionx/RAE-collections` under `decoders/dinov2/wReg_base/ViTXL_n08/model.pt`
  and `stats/dinov2/wReg_base/imagenet1k/stat.pt`. Preserve those relative paths
  when downloading. The decoder architecture configuration comes from the RAE source.
- **VAVAE:** prepared latent data already incorporates the pretrained encoder.
  Training from those data needs the LightningDiT checkpoint, not a new encoder
  download. Extracting your own cache additionally needs
  [the official VAVAE tokenizer](https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/tree/main).

After preparing weights, follow [the experiment guide](experiments.md) to configure
data and resource paths and run the stages in order.
