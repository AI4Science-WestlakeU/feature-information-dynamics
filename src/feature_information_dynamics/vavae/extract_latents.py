"""Extract VAVAE posterior samples for offline paired-condition training.

Original and flipped images are encoded separately. Each shard saves latents,
latents_flip and labels, plus a JSON filename sidecar for condition alignment.
Channel statistics are saved separately and applied by the training loader.
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
_VAVAE_LATENT_C: int = 32
_VAVAE_DOWNSAMPLE: int = 16

def build_vavae(vae_path: str, device: torch.device) -> object:
    """Load VA_VAE (f16d32) encoder.

    VA_VAE is NOT an nn.Module — it is a plain Python class that internally
    constructs self.model = AutoencoderKL(...).cuda().eval().  Do NOT call
    .to(device), .eval(), or .parameters() on the VA_VAE wrapper itself.
    Call vae.encode_images(images) for encoding (images must be on CUDA).
    Ensure torch.cuda.set_device(local_rank) is called BEFORE building the VAE
    so the internal .cuda() call lands on the correct GPU.

    Args:
        vae_path: yaml config path (preferred) or .pt checkpoint path.
                  When .pt, a sibling config.yaml or LDiT tokenizer/configs/
                  vavae_f16d32.yaml must exist for architecture config.
        device:   used for set_device (local_rank); VA_VAE handles GPU internally.

    Returns:
        VA_VAE instance (not nn.Module).
    """
    from tokenizer.vavae import VA_VAE
    import tokenizer as _tok_pkg
    if vae_path.endswith('.yaml'):
        cfg_path = vae_path
    else:
        candidates = [Path(vae_path).parent / 'config.yaml', Path(_tok_pkg.__file__).parent / 'configs' / 'vavae_f16d32.yaml']
        found = next((p for p in candidates if p.exists()), None)
        if found is None:
            raise FileNotFoundError(f'Cannot find vavae config for {vae_path}. Pass the yaml path as --vae_path or place config.yaml next to .pt.')
        from omegaconf import OmegaConf
        import tempfile
        config = OmegaConf.load(found)
        config.ckpt_path = str(Path(vae_path).resolve())
        with tempfile.TemporaryDirectory(prefix="fid-vavae-config-") as directory:
            cfg_path = str(Path(directory) / "config.yaml")
            OmegaConf.save(config, cfg_path)
            return VA_VAE(cfg_path)
    return VA_VAE(cfg_path)


@torch.no_grad()
def encode_vavae(vae: object, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Encode [B,3,H,W] → [B,32,H/16,W/16] VAVAE latents (raw, unnormalized).

    VA_VAE.encode_images expects images on CUDA (internal model is hardcoded .cuda()).
    device arg is provided for API consistency but VA_VAE handles device internally.

    Args:
        vae:    VA_VAE instance (not nn.Module).
        images: [B, 3, H, W] float32 in [-1, 1].
        device: compute device (images moved here before passing to VA_VAE).

    Returns:
        [B, 32, H//16, W//16] float32 on CPU.
    """
    return vae.encode_images(images.to(device)).float().cpu()


class ImageFolderWithFilenames(ImageFolder):
    """ImageFolder subclass that exposes the relative path of each sample.

    Returns (image, class_idx, relative_filename) where relative_filename
    is "{synset}/{basename}" e.g. "n01440764/n01440764_10026.JPEG".
    """

    def __init__(self, root: str, transform: object=None) -> None:
        """Initialize dataset with root and transform."""
        super().__init__(root, transform=transform)

    def __getitem__(self, index: int) -> tuple[object, int, str]:
        """Return (image, class_idx, relative_filename)."""
        path, class_idx = self.samples[index]
        rel = '/'.join(Path(path).parts[-2:])
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return (img, class_idx, rel)

def save_shard(output_dir: str, rank: int, shard_idx: int, latents: torch.Tensor, latents_flip: torch.Tensor, labels: torch.Tensor, filenames: list[str], vae_tag: str) -> None:
    """Save one shard: safetensors for tensors + JSON sidecar for filenames.

    Args:
        output_dir:   directory to write shard files.
        rank:         DDP rank (used in filename).
        shard_idx:    shard counter.
        latents:      [N, C, H, W] float32.
        latents_flip: [N, C, H, W] float32.
        labels:       [N] int64.
        filenames:    list of N relative filename strings.
        vae_tag:      "vavae" or "sdvae" (for shard metadata).
    """
    stem = f'latents_rank{rank:02d}_shard{shard_idx:03d}'
    st_path = os.path.join(output_dir, f'{stem}.safetensors')
    json_path = os.path.join(output_dir, f'{stem}.json')
    save_file({'latents': latents, 'latents_flip': latents_flip, 'labels': labels}, st_path, metadata={'n': str(latents.shape[0]), 'vae': vae_tag})
    with open(json_path, 'w') as f:
        json.dump({'filenames': filenames}, f, separators=(',', ':'))

def compute_and_save_stats(output_dir: str, latent_c: int) -> None:
    """Compute per-channel mean/std from all shards; save latents_stats.pt.

    Args:
        output_dir: directory containing latents_*.safetensors shards.
        latent_c:   number of latent channels.
    """
    import glob as glob_mod
    from safetensors import safe_open
    shard_files = sorted(glob_mod.glob(os.path.join(output_dir, 'latents_rank*.safetensors')))
    if not shard_files:
        log.warning('No shards found for stats computation')
        return
    total_sum = torch.zeros(latent_c, dtype=torch.float64)
    total_sq = torch.zeros(latent_c, dtype=torch.float64)
    total_px = 0
    for sf in shard_files:
        with safe_open(sf, framework='pt', device='cpu') as f:
            lat = f.get_slice('latents')
            n, c, h, w = lat.get_shape()
            for s in range(0, n, 256):
                chunk = lat[s:min(s + 256, n)].to(torch.float64)
                total_sum += chunk.sum(dim=(0, 2, 3))
                total_sq += (chunk ** 2).sum(dim=(0, 2, 3))
                total_px += chunk.shape[0] * h * w
    mean = (total_sum / total_px).float().view(1, latent_c, 1, 1)
    std = (total_sq / total_px - (total_sum / total_px) ** 2).clamp_min(1e-10).sqrt().float().view(1, latent_c, 1, 1)
    torch.save({'mean': mean, 'std': std}, os.path.join(output_dir, 'latents_stats.pt'))
    log.info('Stats saved. mean=%s std=%s', mean.view(-1).tolist(), std.view(-1).tolist())

def main() -> None:
    """CLI entry point for latent extraction with filenames."""
    p = argparse.ArgumentParser(description='Extract VAE latents with filenames (B path)')
    p.add_argument('--encoder', choices=['vavae'], default='vavae')
    p.add_argument('--data_root', required=True, help='ImageNet train dir')
    p.add_argument('--whitelist_json', required=True, help='SR95 whitelist JSON (synset list or {synset: ...} dict)')
    p.add_argument('--vae_path', required=True, help='VAVAE architecture YAML with checkpoint path, or .pt checkpoint.')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--image_size', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--shard_size', type=int, default=10240, help='Samples per shard per rank.')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    try:
        dist.init_process_group('nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    except Exception:
        rank = 0
        world_size = 1
        device = torch.device('cuda:0')
    torch.manual_seed(args.seed + rank)
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        log.info('encoder=%s  data_root=%s  output_dir=%s', args.encoder, args.data_root, args.output_dir)
    with open(args.whitelist_json) as f:
        wl = json.load(f)
    if isinstance(wl, dict):
        whitelist_synsets = set(wl.get('synsets', list(wl.keys())))
    else:
        whitelist_synsets = set(wl)
    vae = build_vavae(args.vae_path, device)
    encode_fn = encode_vavae
    latent_c = _VAVAE_LATENT_C
    import torch.nn as _nn
    _vae_module = vae.model if hasattr(vae, 'model') else vae
    if isinstance(_vae_module, _nn.Module):
        _vae_norm = next(_vae_module.parameters()).norm().item()
        if _vae_norm < 0.01:
            raise RuntimeError(f'VAE first-param norm={_vae_norm:.4f} — likely random-init. Check --vae_path={args.vae_path} and config/ckpt loading.')
        if rank == 0:
            log.info('VAE loaded OK: first-param norm=%.4f', _vae_norm)
    elif rank == 0:
        log.info('VAE loaded OK (non-Module wrapper, weight check skipped)')
    tfm = transforms.Compose([transforms.Resize(args.image_size), transforms.CenterCrop(args.image_size), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    tfm_flip = transforms.Compose([transforms.Resize(args.image_size), transforms.CenterCrop(args.image_size), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    ds_orig = ImageFolderWithFilenames(args.data_root, transform=tfm)
    ds_flip = ImageFolderWithFilenames(args.data_root, transform=tfm_flip)
    if set(ds_orig.classes).issubset({'train', 'val', 'test'}) or len(ds_orig.classes) < 100:
        raise RuntimeError(f'data_root must point at the ImageNet split directory containing synset subdirs, e.g. <data>/imagenet/train. Got {args.data_root!r} with classes={ds_orig.classes[:10]!r} (n={len(ds_orig.classes)}).')
    assert ds_orig.samples == ds_flip.samples, 'ImageFolder sample order differs between orig and flip datasets — non-deterministic filesystem? Cannot guarantee latent/filename alignment.'
    valid_idx = [i for i, (path, _) in enumerate(ds_orig.samples) if Path(path).parent.name in whitelist_synsets]
    retained_labels = {int(ds_orig.samples[i][1]) for i in valid_idx}
    if len(retained_labels) < min(100, len(whitelist_synsets)):
        raise RuntimeError(f'SR95 retained too few distinct ImageFolder labels ({len(retained_labels)} for {len(valid_idx)} samples). This usually means --data_root was pointed at <data>/imagenet instead of <data>/imagenet/train, collapsing labels to train/val.')
    if rank == 0:
        log.info('SR95 filter: %d / %d samples retained, distinct labels=%d', len(valid_idx), len(ds_orig), len(retained_labels))
    from torch.utils.data import Subset
    ds_orig = Subset(ds_orig, valid_idx)
    ds_flip = Subset(ds_flip, valid_idx)
    sampler = DistributedSampler(ds_orig, num_replicas=world_size, rank=rank, shuffle=False, seed=args.seed)
    sampler_flip = DistributedSampler(ds_flip, num_replicas=world_size, rank=rank, shuffle=False, seed=args.seed)
    loader = DataLoader(ds_orig, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    loader_flip = DataLoader(ds_flip, batch_size=args.batch_size, sampler=sampler_flip, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    if args.shard_size % args.batch_size != 0:
        raise ValueError(f'shard_size ({args.shard_size}) must be a multiple of batch_size ({args.batch_size}) to keep latent/filename alignment exact.')
    shard_batch_count = args.shard_size // args.batch_size
    lat_buf: list[torch.Tensor] = []
    flip_buf: list[torch.Tensor] = []
    lbl_buf: list[torch.Tensor] = []
    fn_buf: list[str] = []
    shard_idx = 0
    processed = 0
    for (x, y, fns), (x_f, _, _) in zip(loader, loader_flip):
        lat_buf.append(encode_fn(vae, x, device))
        flip_buf.append(encode_fn(vae, x_f, device))
        lbl_buf.append(y)
        fn_buf.extend(fns)
        processed += x.shape[0]
        if processed % (args.batch_size * 10) == 0 and rank == 0:
            log.info('%s rank0 %d / ~%d samples', datetime.now().strftime('%H:%M:%S'), processed, len(ds_orig) // world_size)
        if len(lat_buf) >= shard_batch_count:
            save_shard(args.output_dir, rank, shard_idx, torch.cat(lat_buf[:shard_batch_count]), torch.cat(flip_buf[:shard_batch_count]), torch.cat(lbl_buf[:shard_batch_count]).to(torch.int64), fn_buf[:args.shard_size], args.encoder)
            if rank == 0:
                log.info('Saved shard %d (%d samples)', shard_idx, args.shard_size)
            lat_buf = lat_buf[shard_batch_count:]
            flip_buf = flip_buf[shard_batch_count:]
            lbl_buf = lbl_buf[shard_batch_count:]
            fn_buf = fn_buf[args.shard_size:]
            shard_idx += 1
    if fn_buf:
        save_shard(args.output_dir, rank, shard_idx, torch.cat(lat_buf), torch.cat(flip_buf), torch.cat(lbl_buf).to(torch.int64), fn_buf, args.encoder)
        if rank == 0:
            log.info('Saved remainder shard %d (%d samples)', shard_idx, len(fn_buf))
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if rank == 0:
        try:
            compute_and_save_stats(args.output_dir, latent_c)
        except Exception as exc:
            log.error('stats computation failed (non-fatal): %s', exc)
        log.info('Extraction complete. Output: %s', args.output_dir)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
if __name__ == '__main__':
    main()
