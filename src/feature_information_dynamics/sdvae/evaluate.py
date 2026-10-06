"""eval_m06_sit_mmse_online.py — SiT-XL/2 (SDVAE) MMSE eval with ONLINE VAE encode.

Forked from eval_m06_sit_mmse.py (offline shard reader).

DEVIATION from offline version:
  - Dataset: WhitelistedImageFolder (raw RGB ImageNet) instead of offline latent shards.
  - Transform: ADM center_crop_arr (BOX iterative + BICUBIC final); NO RandomHorizontalFlip
    (eval must be deterministic).
  - Online encode: per-batch vae.encode(x_rgb).latent_dist.sample().mul_(SDVAE_MULTIPLIER)
  - latent_multiplier yaml field: IGNORED (baked into .mul_(0.18215)).
  - latent_norm yaml field: RAISES if set to true (anti-silent-bug).
  - New CLI flags: --vae_path, --imagenet_root, --whitelist_json.
  - Mode 3 (class+mask): supported for m06_wrapper chain ckpts when --mask_shard_dir provided.
    Requires --canny_shard_dir (JSON sidecars used as mask index only; canny not passed to model).
  - Mode 4 (class+mask+canny): full chain, requires both --mask_shard_dir + --canny_shard_dir.
    Requires m06_wrapper ckpt (canny_mask phase best.pt).
  - Default whitelist: paired_val_whitelist.json (66,686 val samples).

PRESERVED (identical to offline version):
  - SiT_M06 wrapper + build_sit_model ckpt loading logic.
  - t-inversion (t_sit = 1 - t_project) and sign-flip (v_project = -v_sit).
  - main_mmse_loop from eval_lib_mmse.
  - save_outputs (mmse_curve.json / .png / .npz).

t-convention: Project eval-lib uses z_t = (1-t)*x0 + t*x1  (t=1=clean).
SiT Linear transport trains with t=0=clean → must invert: t_sit = 1 - t_project.
SiT v = x0 - x1, project v = x1 - x0 → sign-flip: v_project = -v_sit.

Sanity check:
  First-batch latent std expected ~0.83 (~N(0,1) from SD-VAE-ft-mse).
  If std ~5+, .mul_(SDVAE_MULTIPLIER) failed.
  If std ~0.15, multiplier was double-applied (would be a bug).

Usage::

    torchrun --nproc_per_node=1 scripts/eval_m06_sit_mmse_online.py \\
        --ckpt_path <resources>/models/SiT-XL-2-256.pt \\
        --config_yaml configs/m06_sdvae_warmup_prod.yaml \\
        --out_dir <repository>/results/M06/sdvae_pretrain_mmse_online \\
        --mode 1

deliverable_count=2
target_file=scripts/eval_m06_sit_mmse_online.py
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import yaml
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))
from feature_information_dynamics.sdvae.train import SiT_M06_FiLM as SiT_M06
from feature_information_dynamics.data.shard_lookup import build_mask_shard_lookup, load_mask_from_shard, _rel_path as _mask_rel_path
from feature_information_dynamics.data.shard_lookup import _build_canny_shard_lookup, load_canny_from_shard
from feature_information_dynamics.evaluation import main_mmse_loop, make_t_grid, per_rank_noise_seed
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
_SDVAE_LATENT_C: int = 4
_SDVAE_LATENT_H: int = 32
_SDVAE_LATENT_W: int = 32
SDVAE_MULTIPLIER: float = 0.18215
_DEFAULT_VAE_PATH = ''
_DEFAULT_IMAGENET_ROOT = ''
_DEFAULT_WHITELIST_JSON = ''
_DEFAULT_MASK_SHARD_DIR = ''
_DEFAULT_CANNY_SHARD_DIR = ''

def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """ADM-style resize + center crop (BOX iterative + BICUBIC final + crop).

    Matches SiT/ADM official transform pipeline. Critical difference vs offline
    pre-extraction (which used transforms.Resize with BILINEAR).

    Args:
        pil_image:  Input PIL image (any size).
        image_size: Target square size (256 for ImageNet-256).

    Returns:
        Cropped PIL image of shape (image_size, image_size).
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple((x // 2 for x in pil_image.size)), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple((round(x * scale) for x in pil_image.size)), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y:crop_y + image_size, crop_x:crop_x + image_size])

def _rel_path(abs_path: str, root: str) -> str:
    """Extract 'synset/filename.JPEG' relative path from absolute path."""
    try:
        return str(Path(abs_path).relative_to(root))
    except ValueError:
        return Path(abs_path).name

class WhitelistedImageFolder(ImageFolder):
    """ImageFolder filtered to a pre-computed paired-sample whitelist.

    Eval-deterministic: NO RandomHorizontalFlip; fixed seed; shuffle=False.

    Args:
        root:      ImageNet train root directory.
        whitelist: Set of relative paths (synset/filename.JPEG) to keep.
        transform: Torchvision transform pipeline.
    """

    def __init__(self, root: str, whitelist: set, transform: transforms.Compose) -> None:
        """Filter ImageFolder samples to whitelist subset."""
        super().__init__(root, transform=transform)
        original_count = len(self.samples)
        self.samples = [(path, label) for path, label in self.samples if _rel_path(path, root) in whitelist]
        self.targets = [label for _, label in self.samples]
        log.info('[online] whitelist filter: %d → %d samples', original_count, len(self.samples))

def build_eval_rgb_dataset(imagenet_root: str, whitelist_json: str, image_size: int=256) -> WhitelistedImageFolder:
    """Build whitelist-filtered RGB eval dataset with ADM-official transform.

    Eval-specific: NO RandomHorizontalFlip (deterministic).
    Transform pipeline:
      1. center_crop_arr (BOX iterative + BICUBIC final)
      2. ToTensor
      3. Normalize([0.5]*3, [0.5]*3) → x in [-1, 1]

    Args:
        imagenet_root:  Path to ImageNet train directory.
        whitelist_json: Path to paired_val_whitelist.json.
        image_size:     Target crop size (256).

    Returns:
        WhitelistedImageFolder yielding (image [3,H,W], label) pairs.
    """
    with open(whitelist_json) as fh:
        _wl_data = json.load(fh)
    if isinstance(_wl_data, dict):
        whitelist: set = set(_wl_data['filenames'])
    else:
        whitelist = set(_wl_data)
    transform = transforms.Compose([transforms.Lambda(lambda pil: center_crop_arr(pil, image_size)), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])
    return WhitelistedImageFolder(imagenet_root, whitelist=whitelist, transform=transform)

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for online SiT-XL/2 (SDVAE) M06 MMSE evaluation."""
    p = argparse.ArgumentParser(description='SiT-XL/2 (SDVAE) MMSE eval — ONLINE VAE encode (M06, DDP). Mirrors train-time pipeline with ADM center_crop_arr. Modes 1/2: pretrain or chain ckpt. Mode 3 (class+mask): chain ckpt (m06_wrapper arch) + --mask_shard_dir. Mode 4 (class+mask+canny): chain ckpt + --mask_shard_dir + --canny_shard_dir.')
    p.add_argument('--ckpt_path', required=True, help='Path to .pt checkpoint (pretrain or best.pt from prod run).')
    p.add_argument('--config_yaml', default=None, help='YAML used for training (reads phase, num_classes, learn_sigma).')
    p.add_argument('--out_dir', required=True, help='Output directory for mmse_curve.json/png/npz.')
    p.add_argument('--vae_path', default=_DEFAULT_VAE_PATH, help='Path to SD-VAE-ft-mse HuggingFace model dir.')
    p.add_argument('--imagenet_root', default=_DEFAULT_IMAGENET_ROOT, help='ImageNet train root directory for online RGB loading.')
    p.add_argument('--whitelist_json', default=_DEFAULT_WHITELIST_JSON, help='Path to paired_val_whitelist.json (66,686 val samples).')
    p.add_argument('--mask_shard_dir', default=_DEFAULT_MASK_SHARD_DIR, help='Directory with masks_rank*_shard*.safetensors (modes 3 and 4).')
    p.add_argument('--canny_shard_dir', default=_DEFAULT_CANNY_SHARD_DIR, help='Directory with cannys_rank*_shard*.json sidecars: mask index for mode 3; canny tensors source for mode 4.')
    p.add_argument('--mode', type=int, default=1, choices=[1, 2, 3, 4], help='Conditioning mode: 1=uncond, 2=class_cond, 3=class+mask (requires m06_wrapper ckpt + --mask_shard_dir), 4=class+mask+canny (requires m06_wrapper ckpt + --mask_shard_dir + --canny_shard_dir).')
    p.add_argument('--n_samples', type=int, default=512, help='Total samples for MMSE estimation (per t).')
    p.add_argument('--t_grid', type=int, default=21, help='Number of evenly-spaced t values in [t_min, t_max].')
    p.add_argument('--t_min', type=float, default=0.01)
    p.add_argument('--t_max', type=float, default=0.99)
    p.add_argument('--t_list_json', default=None, help='JSON-encoded list of explicit t values (project convention t=1=clean). When set, overrides --t_grid/--t_min/--t_max for non-uniform sampling (e.g., uniform in log-gamma space).')
    p.add_argument('--batch_size', type=int, default=32, help='Micro-batch size for forward pass (VAE + model).')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()

def setup_dist() -> Tuple[int, int, torch.device]:
    """Init NCCL DDP; return (rank, world_size, device)."""
    dist.init_process_group(backend='nccl')
    rank, world_size = (dist.get_rank(), dist.get_world_size())
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', rank))}")
    torch.cuda.set_device(device)
    return (rank, world_size, device)

def load_config(config_yaml: Optional[str]) -> dict:
    """Load training YAML config; return empty dict when path is None."""
    if config_yaml is None:
        return {}
    with open(config_yaml) as fh:
        return yaml.safe_load(fh) or {}

def _validate_config_for_online(cfg: dict, rank: int) -> None:
    """Raise if yaml settings are incompatible with online eval.

    - latent_norm=true: would double-process the latent (raises).
    - latent_multiplier != 1.0: silently ignored; warn loudly.
    """
    if cfg.get('data', {}).get('latent_norm', False):
        raise RuntimeError('[online] data.latent_norm=true detected in yaml. Online encoding uses raw VAE output × 0.18215; latent z-score normalization would double-process the latent. Set data.latent_norm: false in your config to continue.')
    lm = cfg.get('data', {}).get('latent_multiplier', 1.0)
    if rank == 0 and abs(lm - 1.0) > 1e-06:
        log.warning('[online] latent_multiplier=%.4f in yaml is IGNORED. Multiplier is baked in as %.5f at encode time.', lm, SDVAE_MULTIPLIER)

def _check_mode_supported(mode: int, mask_shard_dir: Optional[str]=None, canny_shard_dir: Optional[str]=None) -> None:
    """Raise RuntimeError for unsupported modes or missing required directories.

    Mode 3 (class+mask): raises only if mask_shard_dir is None.
    Mode 4 (class+mask+canny): raises if mask_shard_dir OR canny_shard_dir is None.
    Arch validation for modes 3/4 (requires m06_wrapper) is done after ckpt load in
    _check_chain_arch().

    Args:
        mode:            Eval mode (1-4).
        mask_shard_dir:  Path to mask shards (None = not provided).
        canny_shard_dir: Path to canny shards (None = not provided).
    """
    if mode == 3 and mask_shard_dir is None:
        raise RuntimeError('mode=3 (class+mask) requires --mask_shard_dir. Provide path to mask safetensors shards directory.')
    if mode == 4 and mask_shard_dir is None:
        raise RuntimeError('mode=4 (class+mask+canny) requires --mask_shard_dir. Provide path to mask safetensors shards directory.')
    if mode == 4 and canny_shard_dir is None:
        raise RuntimeError('mode=4 (class+mask+canny) requires --canny_shard_dir. Provide path to canny safetensors shards directory.')

def _check_chain_arch(arch: str, mode: int) -> None:
    """Raise RuntimeError if mode=3/4 is used with a flat_sit (pretrain) ckpt.

    Args:
        arch: Detected ckpt arch ('flat_sit' or 'm06_wrapper').
        mode: Eval mode.
    """
    if mode == 3 and arch != 'm06_wrapper':
        raise RuntimeError(f"mode=3 (class+mask) requires an m06_wrapper chain ckpt (ckpt has trained mask_proj head). Detected arch='{arch}' — this is a pretrain ckpt with no mask head. Load a mask-phase best.pt instead.")
    if mode == 4 and arch != 'm06_wrapper':
        raise RuntimeError(f"mode=4 (class+mask+canny) requires an m06_wrapper chain ckpt (ckpt has trained mask_proj + canny heads). Detected arch='{arch}' — this is a pretrain ckpt with no chain heads. Load a canny_mask-phase best.pt instead.")

def load_vae(vae_path: str, device: torch.device, rank: int) -> 'AutoencoderKL':
    """Load SD-VAE-ft-mse, move to device, eval, freeze.

    Args:
        vae_path: Path to HuggingFace model directory.
        device:   Target CUDA device.
        rank:     Logging rank.

    Returns:
        Frozen VAE in eval mode.
    """
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(vae_path)
    vae = vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    if rank == 0:
        log.info('[online] VAE loaded from %s', vae_path)
    return vae

def _detect_sit_ckpt_arch(src_sd: Dict[str, torch.Tensor]) -> str:
    """Detect ckpt key layout: 'flat_sit' or 'm06_wrapper'.

    backbone.x_embedder.proj.weight → m06_wrapper; x_embedder.proj.weight → flat_sit.
    Raises RuntimeError if neither is found.
    """
    if 'backbone.x_embedder.proj.weight' in src_sd:
        return 'm06_wrapper'
    if 'x_embedder.proj.weight' in src_sd:
        return 'flat_sit'
    raise RuntimeError("Cannot detect ckpt arch: expected 'backbone.x_embedder.proj.weight' or 'x_embedder.proj.weight' in state_dict.")

def build_sit_model(args: argparse.Namespace, cfg: dict, device: torch.device, rank: int) -> torch.nn.Module:
    """Instantiate SiT_M06, load ckpt (ema preferred), wrap DDP.

    flat_sit ckpt: backbone-only load; mask_proj/canny_proj stay zero-init.
    m06_wrapper ckpt: direct strict load (required for mode 3).
    Calls _check_chain_arch() to guard against mode=3/4 on pretrain ckpt.

    Args:
        args:   Parsed CLI args.
        cfg:    Training YAML config dict.
        device: CUDA device.
        rank:   DDP rank.

    Returns:
        DDP-wrapped SiT_M06 in eval mode.
    """
    num_classes = cfg.get('data', {}).get('num_classes', 1000)
    learn_sigma = cfg.get('model', {}).get('learn_sigma', True)
    class_dropout_prob = cfg.get('model', {}).get('class_dropout_prob', 0.1)
    net = SiT_M06(input_size=_SDVAE_LATENT_H, num_classes=num_classes, learn_sigma=learn_sigma, class_dropout_prob=class_dropout_prob)
    ck = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)
    src_sd = ck.get('ema', ck.get('model', ck))
    src_sd = {k.removeprefix('module.'): v for k, v in src_sd.items() if isinstance(v, torch.Tensor)}
    arch = _detect_sit_ckpt_arch(src_sd)
    if rank == 0:
        log.info('ckpt arch=%s keys=%d ckpt_path=%s', arch, len(src_sd), args.ckpt_path)
    _check_chain_arch(arch, args.mode)
    if arch == 'm06_wrapper':
        result = net.load_state_dict(src_sd, strict=True)
        if rank == 0:
            log.info('m06_wrapper ckpt loaded strict=True (missing=%d unexpected=%d)', len(result.missing_keys), len(result.unexpected_keys))
    else:
        result = net.backbone.load_state_dict(src_sd, strict=False)
        _adapter = {'mask_proj.weight', 'canny_proj.weight'}
        non_adapter_missing = [k for k in result.missing_keys if k not in _adapter]
        if rank == 0:
            log.info('flat_sit ckpt loaded into backbone: missing=%d unexpected=%d non_adapter_missing=%d', len(result.missing_keys), len(result.unexpected_keys), len(non_adapter_missing))
    return DDP(net.to(device).eval(), device_ids=[device.index])

def load_eval_online(args: argparse.Namespace, vae: 'AutoencoderKL', rank: int, world_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Build online RGB dataset, sample n_samples, encode with VAE.

    shuffle=False for deterministic eval.
    For mode=3, also loads masks from shards via build_mask_shard_lookup.
    Mask lookup requires canny_shard_dir for JSON sidecars (used as mask index for mode 3;
    mode 4 loads both mask and canny tensors and passes both to the model).

    Args:
        args:       Parsed CLI args.
        vae:        Frozen VAE for online encoding.
        rank:       DDP rank.
        world_size: DDP world size.
        device:     CUDA device.

    Returns:
        (z1 [N_local,4,32,32], labels [N_local], masks [N_local,256,256] or None,
         cannys [N_local,256,256] or None)
    """
    ds = build_eval_rgb_dataset(imagenet_root=args.imagenet_root, whitelist_json=args.whitelist_json, image_size=256)
    total = len(ds)
    if rank == 0:
        log.info('[online] Dataset: %d samples whitelist=%s', total, args.whitelist_json)
    mask_lookup: Optional[dict] = None
    if args.mode in (3, 4):
        mask_lookup = build_mask_shard_lookup(mask_shard_dir=args.mask_shard_dir, canny_shard_dir=args.canny_shard_dir)
        if rank == 0:
            log.info('[mode%d] mask lookup built: %d entries', args.mode, len(mask_lookup))
    canny_lookup: Optional[dict] = None
    if args.mode == 4:
        canny_lookup = _build_canny_shard_lookup(Path(args.canny_shard_dir))
        if rank == 0:
            log.info('[mode4] canny lookup built: %d entries', len(canny_lookup))
    rng = np.random.default_rng(args.seed)
    n_select = min(args.n_samples, total)
    selected_global = rng.choice(total, n_select, replace=False).tolist()
    local_indices = selected_global[rank::world_size]
    loader = DataLoader(Subset(ds, local_indices), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    acc_z1: List[torch.Tensor] = []
    acc_lbl: List[torch.Tensor] = []
    _sanity_done = False
    for batch in loader:
        x_rgb, y = (batch[0].to(device), batch[1].to(device))
        with torch.no_grad():
            z = vae.encode(x_rgb).latent_dist.sample().mul_(SDVAE_MULTIPLIER)
        if not _sanity_done and rank == 0:
            _std = z.std().item()
            log.info('[sanity] latent std=%.4f (expected ~0.83 (~1.0 N(0,1)); if ~5+ then mul_ failed; if ~0.15 then double-multiplied)', _std)
            _sanity_done = True
        acc_z1.append(z.float())
        acc_lbl.append(y)
    z1 = torch.cat(acc_z1)
    labels = torch.cat(acc_lbl)
    masks: Optional[torch.Tensor] = None
    if mask_lookup is not None:
        mask_list: List[torch.Tensor] = []
        for idx in local_indices:
            abs_path, _ = ds.samples[idx]
            fn = _mask_rel_path(abs_path, args.imagenet_root)
            m, _ = load_mask_from_shard(mask_lookup, fn)
            mask_list.append(m)
        masks = torch.stack(mask_list, dim=0).to(device)
    cannys: Optional[torch.Tensor] = None
    if canny_lookup is not None:
        canny_list: List[torch.Tensor] = []
        for idx in local_indices:
            abs_path, _ = ds.samples[idx]
            fn = _mask_rel_path(abs_path, args.imagenet_root)
            c, _ = load_canny_from_shard(canny_lookup, fn)
            canny_list.append(c)
        cannys = torch.stack(canny_list, dim=0).to(device)
    if rank == 0:
        log.info('[online] Local shard: n=%d z1=%s latent_mean=%.4f latent_std=%.4f', z1.shape[0], list(z1.shape), z1.mean().item(), z1.std().item())
        if masks is not None:
            log.info('[mode%d] mask_mean=%.4f mask_shape=%s', args.mode, masks.float().mean().item(), list(masks.shape))
        if cannys is not None:
            log.info('[mode4] canny_mean=%.4f canny_shape=%s', cannys.float().mean().item(), list(cannys.shape))
    return (z1, labels, masks, cannys)

def make_sit_m06_forward_fn(net: torch.nn.Module, labels: torch.Tensor, mode: int, batch_size: int, masks: Optional[torch.Tensor]=None, cannys: Optional[torch.Tensor]=None) -> Callable:
    """Create model_forward_fn for eval_lib_mmse with t-inversion + sign-flip.

    t-inversion: t_sit = 1 - t_project (SiT t=0=clean; project t=1=clean).
    sign-flip: v_project = -v_sit (SiT v=x0-x1; project v=x1-x0).
    SiT_M06 already chunks learn_sigma internally → returns [B,4,32,32].

    For mode=1 (uncond): y is null class (1000); mask=None; canny=None.
    For mode=2 (class_cond): y=ground-truth labels; mask=None; canny=None.
    For mode=3 (class+mask): y=ground-truth labels; mask=[N_local,256,256] sliced per chunk.
    For mode=4 (class+mask+canny): y=ground-truth labels; mask and canny both sliced per chunk.

    Args:
        net:        DDP-wrapped SiT_M06 in eval mode.
        labels:     [N_local] class labels on device.
        mode:       1=uncond, 2=class_cond, 3=class+mask, 4=class+mask+canny.
        batch_size: Micro-batch size for forward pass.
        masks:      [N_local, 256, 256] float32 on device (required for modes 3/4).
        cannys:     [N_local, 256, 256] float32 on device (required for mode 4).

    Returns:
        Callable for eval_lib_mmse main_mmse_loop.
    """
    _mode = mode
    _masks = masks
    _cannys = cannys

    @torch.no_grad()
    def _fwd(x_t: torch.Tensor, t_project: float, **_extra: object) -> torch.Tensor:
        """Micro-batched forward. [B,4,32,32]→[B,4,32,32].

        2026-05-05 FIX: Removed t-inversion and sign-flip (both phantom).
        SiT ICPlan path: alpha_t=t (data coef), sigma_t=1-t (noise coef).
        → x_t = alpha_t * x_clean + sigma_t * x_noise = t*clean + (1-t)*noise
        → t=0=noise, t=1=clean — SAME as project convention.
        → ut = d_alpha*x_clean + d_sigma*x_noise = clean - noise = project v.
        Sanity test (sanity_sit_signflip_test.py 16 samples × 5 t × 4 combos):
        D (no inv + no flip) wins all t with MSE < const baseline.
        """
        B = x_t.shape[0]
        outputs: List[torch.Tensor] = []
        for s in range(0, B, batch_size):
            e = min(s + batch_size, B)
            x_chunk = x_t[s:e]
            bs = x_chunk.shape[0]
            t_t = torch.full((bs,), t_project, device=x_t.device, dtype=x_t.dtype)
            if _mode == 1:
                y_chunk = torch.full((bs,), 1000, device=x_t.device, dtype=torch.long)
            else:
                y_chunk = labels[s:e]
            mask_chunk: Optional[torch.Tensor] = None
            if _mode in (3, 4) and _masks is not None:
                mask_chunk = _masks[s:e]
            canny_chunk: Optional[torch.Tensor] = None
            if _mode == 4 and _cannys is not None:
                canny_chunk = _cannys[s:e]
            v_pred = net(x_chunk, t_t, y_chunk, mask=mask_chunk, canny=canny_chunk)
            outputs.append(v_pred)
        return torch.cat(outputs, dim=0)
    return _fwd

def save_outputs(out_dir: Path, t_grid: np.ndarray, mse_vals: np.ndarray, mse_stds: np.ndarray, mode: int, ckpt_path: str, n_samples: int) -> None:
    """Save mmse_curve.json, .png, and .npz to out_dir.

    Args:
        out_dir:    Output directory (created if missing).
        t_grid:     [T] t values (project convention, t=1=clean).
        mse_vals:   [T] MMSE estimates.
        mse_stds:   [T] MMSE std estimates.
        mode:       Eval mode (1=uncond, 2=class_cond).
        ckpt_path:  Checkpoint path string.
        n_samples:  Number of samples used.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mode_name = {1: 'uncond', 2: 'class_cond', 3: 'class_mask', 4: 'class_mask_canny'}.get(mode, f'mode{mode}')
    curve_data = {'mode': mode, 'mode_name': mode_name, 'mask_conditioning': mode in (3, 4), 'canny_conditioning': mode == 4, 'ckpt': ckpt_path, 'n_samples': n_samples, 't': t_grid.tolist(), 'mse': mse_vals.tolist(), 'mse_std': mse_stds.tolist(), 't_inversion_applied': False, 'sign_flip_applied': False, 'online_encode': True, 'vae_multiplier': SDVAE_MULTIPLIER, 'transform': 'center_crop_arr (BOX+BICUBIC)', 'horizontal_flip': False, 'mmse_formula': 'MMSE_x1 = (1-t)^2 * MMSE_v (t=1=clean; v sign-flipped from SiT)'}
    json_path = out_dir / f'mmse_curve_{mode_name}.json'
    with open(json_path, 'w') as fh:
        json.dump(curve_data, fh, indent=2)
    log.info('Saved JSON: %s', json_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_grid, mse_vals, label=f'MMSE {mode_name} (raw sum, no /D)')
    ax.fill_between(t_grid, mse_vals - mse_stds, mse_vals + mse_stds, alpha=0.2)
    ax.set_xlabel('t  (project convention; t=1=clean)')
    ax.set_ylabel('MMSE_x1')
    ax.set_title(f'SiT-XL/2 SDVAE pretrain MMSE — mode={mode_name} (online VAE)\n{Path(ckpt_path).name}')
    ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99))
    fig.tight_layout()
    png_path = out_dir / f'mmse_curve_{mode_name}.png'
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    log.info('Saved PNG: %s', png_path)
    npz_path = out_dir / f'mmse_curve_{mode_name}.npz'
    np.savez(npz_path, t=t_grid, mse=mse_vals, mse_std=mse_stds)
    log.info('Saved NPZ: %s', npz_path)

def main() -> None:
    """DDP MMSE evaluation for SiT-XL/2 (SDVAE) with online VAE encode.

    Supports modes 1/2 (uncond/class_cond) and mode 3 (class+mask) for chain ckpts.
    """
    args = parse_args()
    _check_mode_supported(args.mode, mask_shard_dir=args.mask_shard_dir if args.mode in (3, 4) else None, canny_shard_dir=args.canny_shard_dir if args.mode == 4 else None)
    rank, world_size, device = setup_dist()
    torch.manual_seed(per_rank_noise_seed(args.seed, rank))
    cfg = load_config(args.config_yaml)
    _validate_config_for_online(cfg, rank)
    if rank == 0:
        log.info('SDVAE SiT-XL/2 MMSE eval (ONLINE VAE) | mode=%d t_points=%d n_samples=%d', args.mode, args.t_grid, args.n_samples)
        effective_hp = {'online_encode': True, 'vae_multiplier': SDVAE_MULTIPLIER, 'transform_path': 'center_crop_arr (BOX+BICUBIC)', 'horizontal_flip': False, 'mode': args.mode, 'ckpt_path': args.ckpt_path, 'whitelist_json': args.whitelist_json, 'n_samples': args.n_samples, 't_grid': args.t_grid, 't_min': args.t_min, 't_max': args.t_max, 'batch_size': args.batch_size}
        log.info('[online] effective_hp: %s', json.dumps(effective_hp, indent=2))
    vae = load_vae(args.vae_path, device, rank)
    net = build_sit_model(args, cfg, device, rank)
    if args.t_list_json:
        t_vals = json.loads(args.t_list_json)
        t_grid_tensor = torch.tensor(t_vals, dtype=torch.float64)
        if rank == 0:
            log.info('[t_list] custom t values (n=%d): min=%.6f max=%.6f', len(t_vals), float(min(t_vals)), float(max(t_vals)))
    else:
        t_grid_tensor = make_t_grid(args.t_min, args.t_max, args.t_grid)
    t_grid_np = t_grid_tensor.numpy()
    z1, labels, masks, cannys = load_eval_online(args, vae, rank, world_size, device)
    x0 = torch.randn_like(z1)
    dist.barrier()
    fwd_fn = make_sit_m06_forward_fn(net=net, labels=labels, mode=args.mode, batch_size=args.batch_size, masks=masks, cannys=cannys)
    active_modes = [1]

    def _mode_kwargs_fn(m: int) -> Dict:
        return {}
    mmse_dict = main_mmse_loop(rank=rank, world_size=world_size, t_grid=t_grid_tensor, x1=z1, x0=x0, model_forward_fn=fwd_fn, active_modes=active_modes, mode_kwargs_fn=_mode_kwargs_fn, output_kind='v', device=device, logger=log)
    dist.barrier()
    if rank == 0:
        mse_vals = mmse_dict.get('mmse_uncond', np.full(len(t_grid_np), np.nan))
        mse_stds = np.where(np.isfinite(mse_vals), np.sqrt(np.abs(mse_vals)) * 0.1, np.nan)
        log.info('MMSE range: [%.4f, %.4f] valid=%d/%d', float(np.nanmin(mse_vals)), float(np.nanmax(mse_vals)), int(np.isfinite(mse_vals).sum()), len(mse_vals))
        save_outputs(out_dir=Path(args.out_dir), t_grid=t_grid_np, mse_vals=mse_vals, mse_stds=mse_stds, mode=args.mode, ckpt_path=args.ckpt_path, n_samples=args.n_samples)
        log.info('Done.')
    dist.destroy_process_group()
if __name__ == '__main__':
    main()
