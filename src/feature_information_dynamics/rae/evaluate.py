"""eval_m06_lib_canny_mmse.py — RAE/DiTwDDTHead (RAE_M06) MMSE eval per t for M06 canny conditioning.

t-convention (CRITICAL — preserved from eval_m06_lib_mmse.py):
  RAE upstream training: t=0=clean, ut=x0-x1 (noise to clean direction).
  Project eval_lib:      t=1=clean, v=x1-x0 (clean to noise direction).
  This wrapper MUST:
    1. Pass t_rae = 1 - t_project to RAE_M06.forward (t-axis flip).
    2. Return -v_pred_rae to flip velocity sign to project convention.

Latent shape: [B, 768, 16, 16] (DINOv2-B RAE, 768-dim, 16x16 spatial grid).

EDIT A: _RAE_M06_CannyAdapter.forward — canny_per_token replaces patch_per_token,
        passed as canny= kwarg to RAE_M06 (raw [B, 256, 256] 2D map; RAE_M06's
        Conv2d canny_proj_x patchifies internally — no per-token reshape here).
EDIT B: mode dispatch table — mode 4 = class+mask+canny (was warmup/mask/canny phase-only).
EDIT C: dataset = RawImageMaskCannyDataset (RAE convention; raw images +
        sharded mask/canny safetensors). Project convention: VAVAE/SDVAE use
        MaskedCannyShardedDataset(phase="canny_mask") for precomputed latents,
        RAE/Pixel use RawImageMaskCannyDataset because RAE latents are produced
        on-the-fly by stage1 encoder. The conditioning semantics (mask + canny
        both ON) match SDVAE's phase="canny_mask" exactly.

Compared to eval_m06_lib_mmse.py (legacy single-phase eval):
  - Supports explicit conditional-mode eval
    (uncond/class/class+mask/class+mask+canny/mask-only)
    matching eval_m06_sit_canny_mmse.py / eval_m06_vavae_canny_mmse.py for
    apples-to-apples comparison across backbones.
  - Output prefix canny_mmse_FIXED matches SDVAE/VAVAE naming convention.
  - DDP-enabled (multi-GPU eval); legacy lib_mmse was single-GPU.
  - t-flip + sign-flip preserved from parent eval_m06_lib_mmse.py.

Usage::

    torchrun --nproc_per_node=8 --master_port=29735 \\
        scripts/eval_m06_lib_canny_mmse.py \\
        --ckpt_path <repository>/results/M06/rae_canny_mask/best.pt \\
        --config_yaml <repository>/configs/m06_rae_canny_prod.yaml \\
        --paired_val_whitelist <repository>/configs/paired_val_whitelist.json \\
        --mask_shard_dir <repository>/data/mask_shards \\
        --canny_shard_dir <repository>/data/canny_shards \\
        --imagenet_root <data>/imagenet \\
        --modes 1,2,3,4 --n_per_class 16 --t_points 64 --seed 42 \\
        --out <repository>/results/M06/rae_canny_mmse_FIXED.pt \\
        --manifest_out <repository>/results/M06/rae_canny_mmse_FIXED_manifest.json \\
        --per_sample_out <repository>/results/M06/rae_canny_mmse_FIXED_per_sample.pt

deliverable_count=1
target_file=scripts/eval_m06_lib_canny_mmse.py
"""
from __future__ import annotations
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.distributed as dist
import yaml
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
sys.path.insert(0, str(_SCRIPT_DIR))
from feature_information_dynamics.data.raw_image_mask_canny_dataset import RawImageMaskCannyDataset
from feature_information_dynamics.rae.train import RAE_M06, build_rae_stage1
from feature_information_dynamics.evaluation import MODE_KEYS, _T_EPS_DEFAULT, check_loss_order, jit_weighted_loss, main_mmse_loop, make_t_grid, per_rank_noise_seed, save_mmse_pt, save_per_sample_errors_pt, write_manifest_jsonl
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
_RAE_LATENT_C: int = 768
_RAE_LATENT_H: int = 16
_RAE_LATENT_W: int = 16
_DEFAULT_IMAGENET_ROOT = ''
NULL_CLASS_RAE: int = 1000

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for RAE_M06 MMSE canny-aware evaluation."""
    p = argparse.ArgumentParser(description='RAE/DiTwDDTHead (RAE_M06) MMSE eval — M06 canny conditioning (DDP)')
    p.add_argument('--ckpt_path', required=True, help="M06 RAE checkpoint (best.pt); prefers 'ema' key.")
    p.add_argument('--config_yaml', required=True, help='Training YAML with rae_config_yaml field for stage1 path.')
    p.add_argument('--paired_val_whitelist', required=True, help='Path to paired-val filename whitelist JSON for RawImageMaskCannyDataset.')
    p.add_argument('--mask_shard_dir', required=True, help='Directory with sharded mask safetensors.')
    p.add_argument('--canny_shard_dir', required=True, help='Directory with sharded canny safetensors.')
    p.add_argument('--imagenet_root', default=_DEFAULT_IMAGENET_ROOT, help='Raw ImageNet root for online RAE stage1 encoding.')
    p.add_argument('--out', required=True, help='Output .pt path (prefix: canny_mmse_FIXED).')
    p.add_argument('--t_min', type=float, default=0.01)
    p.add_argument('--t_max', type=float, default=0.99)
    p.add_argument('--t_points', type=int, default=64)
    p.add_argument('--t_list_json', default=None, help='JSON-encoded list of explicit t values (project convention t=1=clean). When set, overrides --t_points/--t_min/--t_max.')
    p.add_argument('--t_eps', type=float, default=_T_EPS_DEFAULT)
    p.add_argument('--n_per_class', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data_seed', type=int, default=None, help='Validation-subset seed; defaults to --seed for backward compatibility.')
    p.add_argument('--noise_seed', type=int, default=None, help='Diffusion-noise seed; defaults to --seed for backward compatibility.')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--micro_bs', type=int, default=8, help='Micro-batch for chunked forward; RAE 768ch heaviest, default 8.')
    p.add_argument('--modes', type=str, default='1,2,3,4', help='Modes 1-8: uncond, class, class+mask, full, mask, canny, class+canny, mask+canny')
    p.add_argument('--manifest_out', default=None, help='Optional JSONL output listing evaluated sample ids/classes.')
    p.add_argument('--per_sample_out', default=None, help='Optional .pt output with per-sample MMSE_x1 errors for bootstrap CI.')
    return p.parse_args()

def setup_dist() -> Tuple[int, int, torch.device]:
    """Init NCCL DDP; return (rank, world_size, device)."""
    dist.init_process_group(backend='nccl')
    rank, world_size = (dist.get_rank(), dist.get_world_size())
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', rank))}")
    torch.cuda.set_device(device)
    return (rank, world_size, device)

def _load_rae_stage1_from_config(config_yaml: str, device: torch.device) -> torch.nn.Module:
    """Load frozen RAE stage1 encoder from training config yaml."""
    with open(config_yaml) as f:
        cfg = yaml.safe_load(f)
    rae_config_yaml = cfg.get('model', {}).get('rae_config_yaml')
    if not rae_config_yaml:
        raise KeyError(f"'model.rae_config_yaml' not found in {config_yaml}; needed to instantiate RAE stage1 encoder.")
    log.info('Loading RAE stage1 from: %s', rae_config_yaml)
    return build_rae_stage1(rae_config_yaml, device)

def build_rae_model(args: argparse.Namespace, device: torch.device, rank: int) -> torch.nn.Module:
    """Instantiate RAE_M06 from OmegaConf yaml, load M06 ckpt, wrap DDP.

    Allowed missing keys: canny_proj_x.* (pre-canny-phase ckpts; zeros-init kept).

    Args:
        args:   parsed CLI namespace with .ckpt_path / .config_yaml.
        device: target CUDA device.
        rank:   DDP rank for logging.

    Returns:
        DDP-wrapped RAE_M06 in eval mode.
    """
    with open(args.config_yaml) as f:
        cfg = yaml.safe_load(f)
    rae_config_yaml = cfg.get('model', {}).get('rae_config_yaml')
    if not rae_config_yaml:
        raise KeyError(f"'model.rae_config_yaml' not found in {args.config_yaml}")
    cfg_all = OmegaConf.load(rae_config_yaml)
    model_params = OmegaConf.to_container(cfg_all.stage_2.params, resolve=True)
    for _k in ('num_classes', 'class_dropout_prob'):
        if _k in cfg.get('model', {}):
            model_params[_k] = cfg['model'][_k]
    raw = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)
    src_sd = None
    for key in ('ema', 'model'):
        if isinstance(raw, dict) and key in raw:
            src_sd = raw[key]
            if rank == 0:
                log.info("Loading RAE_M06 from ckpt key='%s'", key)
            break
    if src_sd is None:
        src_sd = raw
        if rank == 0:
            log.info('Loading RAE_M06 from flat ckpt')
    src_sd = {k.removeprefix('module.').removeprefix('net.'): v for k, v in src_sd.items() if isinstance(v, torch.Tensor)}
    for _y_key in ('backbone.y_embedder.embedding_table.weight', 'y_embedder.embedding_table.weight'):
        if _y_key in src_sd:
            _ckpt_n_slots = src_sd[_y_key].shape[0]
            _class_dropout = float(model_params.get('class_dropout_prob', 0.0))
            _offset = 1 if _class_dropout > 0 else 0
            _detected = _ckpt_n_slots - _offset
            _cur = model_params.get('num_classes', 1000)
            if _detected != _cur:
                if rank == 0:
                    log.warning('num_classes override (chained mismatch): cfg=%d → ckpt=%d (slots=%d)', _cur, _detected, _ckpt_n_slots)
                model_params['num_classes'] = _detected
            break
    net = RAE_M06(**model_params)
    _allowed_missing: set[str] = {'canny_proj_x.weight', 'mask_gates', 'canny_gates'}
    result = net.load_state_dict(src_sd, strict=False)
    _bad_missing = [k for k in result.missing_keys if k not in _allowed_missing]
    if rank == 0:
        log.info('RAE_M06 ckpt loaded strict=False (missing=%d unexpected=%d bad_missing=%d)', len(result.missing_keys), len(result.unexpected_keys), len(_bad_missing))
    if _bad_missing:
        raise RuntimeError(f'RAE_M06 ckpt missing unexpected keys: {_bad_missing}')
    with torch.no_grad():
        if 'mask_gates' in result.missing_keys:
            net.mask_gates.zero_()
        if 'canny_gates' in result.missing_keys:
            net.canny_gates.zero_()
    return DDP(net.to(device).eval(), device_ids=[device.index])

def load_eval_subset(args: argparse.Namespace, rank: int, world_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build RAE-convention canny-aware dataset; return (images_raw, masks, cannys, labels, sample_ids).

    Uses RawImageMaskCannyDataset (is_val=True) which returns (image, mask, canny, label)
    4-tuple — RAE counterpart of MaskedCannyShardedDataset(phase="canny_mask") used by
    VAVAE/SDVAE evals. Both mask and canny are loaded from sharded safetensors.

    Returns:
        images_raw: [N, 3, 256, 256] float32 in [-1,1]
        masks:      [N, 256, 256]    float32 in [0,1]
        cannys:     [N, 256, 256]    float32 in [0,1]
        labels:     [N]              long
        sample_ids: [N]              long (local indices for manifest tracing)
    """
    ds = RawImageMaskCannyDataset(imagenet_root=args.imagenet_root, mask_shard_dir=args.mask_shard_dir, canny_shard_dir=args.canny_shard_dir, paired_whitelist_json=args.paired_val_whitelist, is_val=True)
    data_seed = args.seed if args.data_seed is None else args.data_seed
    rng = np.random.default_rng(data_seed)
    cls_to_idx: Dict[int, List[int]] = {}
    for i in range(len(ds)):
        fn = ds._whitelist_filtered[i]
        synset = fn.split('/')[0]
        cls = int(ds._synset_to_class[synset])
        cls_to_idx.setdefault(cls, []).append(i)
    wl_cls = sorted(cls_to_idx.keys())
    if rank == 0:
        log.info('Dataset: %d image samples, %d classes, phase=canny_mask, is_val=True', len(ds), len(wl_cls))
    selected: List[int] = []
    for cls in wl_cls:
        idxs = cls_to_idx[cls]
        selected.extend(rng.choice(idxs, min(args.n_per_class, len(idxs)), replace=False).tolist())
    local_indices = selected[rank::world_size]
    acc: Dict[str, List] = {'img': [], 'mask': [], 'canny': [], 'lbl': []}
    for batch in DataLoader(Subset(ds, local_indices), batch_size=args.micro_bs, shuffle=False, num_workers=args.num_workers, pin_memory=True):
        image, mask_2d, canny_2d, label = batch
        acc['img'].append(image.cpu())
        acc['mask'].append(mask_2d.cpu())
        acc['canny'].append(canny_2d.cpu())
        acc['lbl'].append(torch.as_tensor(label, dtype=torch.long).cpu())
    return (torch.cat(acc['img']).to(device), torch.cat(acc['mask']).to(device), torch.cat(acc['canny']).to(device), torch.cat(acc['lbl']).to(device), torch.tensor(local_indices, dtype=torch.long, device=device))

class _RAE_M06_CannyAdapter(torch.nn.Module):
    """Adapter wrapping RAE_M06 DDP for canny-aware 5-arg eval interface.

    RAE_M06.forward(x, t, y, s=None, mask=None, canny=None)

    Eval interface: forward(x, t, y, mask_per_token, canny_per_token).
    Both mask and canny are raw 2D maps [B, 256, 256] (RAE_M06's Conv2d
    mask_proj_x / canny_proj_x patchifies internally — no per-token reshape).
    Kwarg names are kept canonical (canny_per_token) for cross-backbone
    compatibility with eval_m06_sit_canny_mmse.py / eval_m06_vavae_canny_mmse.py.
    """

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, mask_per_token: Optional[torch.Tensor]=None, canny_per_token: Optional[torch.Tensor]=None) -> torch.Tensor:
        """Delegate to RAE_M06 passing mask and canny as 2D kwargs."""
        return self.inner(x, t, y, mask=mask_per_token, canny=canny_per_token)

def make_mode_kwargs_fn(labels: torch.Tensor, mask_2d: torch.Tensor, canny_2d: torch.Tensor, device: torch.device) -> Callable[[int], Dict]:
    """Closure: mode_idx -> forward kwargs for _RAE_M06_CannyAdapter.

    Modes:
        1 = uncond           (null class, zero mask, zero canny)
        2 = class            (real class, zero mask, zero canny)
        3 = class+mask       (real class, real mask, zero canny)
        4 = class+mask+canny (real class, real mask, real canny)
        5 = mask-only       (null class, real mask, zero canny)
        6 = canny-only      (null class, zero mask, real canny)
        7 = class+canny     (real class, zero mask, real canny)
        8 = mask+canny      (null class, real mask, real canny)
    """
    N = labels.shape[0]
    null_y = torch.full((N,), NULL_CLASS_RAE, dtype=torch.long, device=device)
    zero_mask = torch.zeros(N, 256, 256, device=device)
    zero_canny = torch.zeros(N, 256, 256, device=device)
    _CFG: Dict[int, Tuple] = {1: (null_y, zero_mask, zero_canny), 2: (labels, zero_mask, zero_canny), 3: (labels, mask_2d, zero_canny), 4: (labels, mask_2d, canny_2d), 5: (null_y, mask_2d, zero_canny), 6: (null_y, zero_mask, canny_2d), 7: (labels, zero_mask, canny_2d), 8: (null_y, mask_2d, canny_2d)}

    def _fn(m: int) -> Dict:
        y, mp, cp = _CFG[m]
        return {'y': y, 'mask_per_token': mp, 'canny_per_token': cp}
    return _fn

def make_rae_model_forward_fn(net: torch.nn.Module, micro_bs: int=8) -> Callable:
    """Wrap DDP RAE_M06 for eval_lib_mmse interface.

    CRITICAL — RAE t-convention adaptation:
      t-flip: t_rae = 1 - t_project (RAE t=0=clean; project t=1=clean).
      sign-flip: v_project = -v_rae (RAE ut=x0-x1; project v=x1-x0).
    Output: full [B, 768, 16, 16] velocity in project convention.
    Processes in micro-batches to avoid OOM on 768ch latents.
    """
    adapter = _RAE_M06_CannyAdapter(net)

    def _fwd(x_t: torch.Tensor, t_project: float, y: torch.Tensor, mask_per_token: Optional[torch.Tensor]=None, canny_per_token: Optional[torch.Tensor]=None) -> torch.Tensor:
        """RAE canny adapter: t-flip + sign-flip. [B,768,16,16]→[B,768,16,16]."""
        t_rae_val = 1.0 - t_project
        B = x_t.shape[0]
        outputs: List[torch.Tensor] = []
        for s_idx in range(0, B, micro_bs):
            e_idx = min(s_idx + micro_bs, B)
            x_chunk = x_t[s_idx:e_idx]
            y_chunk = y[s_idx:e_idx]
            mask_chunk = mask_per_token[s_idx:e_idx] if mask_per_token is not None else None
            canny_chunk = canny_per_token[s_idx:e_idx] if canny_per_token is not None else None
            bs = x_chunk.shape[0]
            t_rae = torch.full((bs,), t_rae_val, device=x_t.device, dtype=x_t.dtype)
            v_rae = adapter(x_chunk, t_rae, y_chunk, mask_chunk, canny_chunk)
            v_project = -v_rae
            outputs.append(v_project)
        return torch.cat(outputs, dim=0)
    return _fwd

def main() -> None:
    """DDP MMSE evaluation for RAE_M06 (DiTwDDTHead) M06 canny conditioning."""
    args = parse_args()
    try:
        active_modes = sorted({int(m.strip()) for m in args.modes.split(',')})
    except ValueError as exc:
        raise ValueError(f'--modes must be comma-separated ints: {args.modes!r}') from exc
    if (bad := [m for m in active_modes if m not in range(1, 9)]):
        raise ValueError(f'Invalid mode(s) {bad}; valid range 1-8')
    rank, world_size, device = setup_dist()
    data_seed = args.seed if args.data_seed is None else args.data_seed
    noise_seed = args.seed if args.noise_seed is None else args.noise_seed
    torch.manual_seed(per_rank_noise_seed(noise_seed, rank))
    if rank == 0:
        log.info('RAE canny eval | modes=%s t_eps=%.3f ckpt=%s', active_modes, args.t_eps, args.ckpt_path)
    net = build_rae_model(args, device, rank)
    stage1 = _load_rae_stage1_from_config(args.config_yaml, device)
    if args.t_list_json:
        import json as _json
        t_vals = _json.loads(args.t_list_json)
        t_grid = torch.tensor(t_vals, dtype=torch.float64)
        if rank == 0:
            log.info('[t_list] custom t (n=%d): min=%.6f max=%.6f', len(t_vals), float(min(t_vals)), float(max(t_vals)))
    else:
        t_grid = make_t_grid(args.t_min, args.t_max, args.t_points)
    images_raw, masks, cannys, labels, sample_ids = load_eval_subset(args, rank, world_size, device)
    N = images_raw.shape[0]
    if rank == 0:
        log.info('Online encoding %d images (rank=%d) with RAE stage1...', N, rank)
    x_clean_tok_list: List[torch.Tensor] = []
    with torch.no_grad():
        for s in range(0, N, args.micro_bs):
            e = min(s + args.micro_bs, N)
            enc = stage1.encode(images_raw[s:e])
            x_clean_tok_list.append(enc)
    z1 = torch.cat(x_clean_tok_list, dim=0)
    if rank == 0:
        log.info('Encoded latent shape: %s', tuple(z1.shape))
    del images_raw, x_clean_tok_list
    torch.cuda.empty_cache()
    x0 = torch.randn_like(z1)
    dist.barrier()
    per_sample: Optional[Dict[str, object]] = {} if args.per_sample_out or args.manifest_out else None
    mmse_dict = main_mmse_loop(rank=rank, world_size=world_size, t_grid=t_grid, x1=z1, x0=x0, model_forward_fn=make_rae_model_forward_fn(net, args.micro_bs), active_modes=active_modes, mode_kwargs_fn=make_mode_kwargs_fn(labels, masks, cannys, device), t_eps=args.t_eps, output_kind='v', device=device, logger=log, sample_ids=sample_ids, labels=labels, per_sample_out=per_sample)
    dist.barrier()
    if rank == 0:
        jit_losses: Optional[Dict[str, float]] = None
        order_ok: Optional[bool] = None
        if set(active_modes) == {1, 2, 3, 4}:
            jit_losses = {k: jit_weighted_loss(t_grid, mmse_dict[k]) for k in MODE_KEYS}
            order_ok = check_loss_order({k.removeprefix('mmse_'): v for k, v in jit_losses.items()})
        save_mmse_pt(out_path=Path(args.out), t_grid=t_grid, mmse_dict=mmse_dict, jit_weighted_losses=jit_losses, loss_order_ok=order_ok, meta={'checkpoint': args.ckpt_path, 'checkpoint_kind': 'rae_m06', 'config_yaml': args.config_yaml, 'imagenet_root': args.imagenet_root, 'mask_shard_dir': args.mask_shard_dir, 'canny_shard_dir': args.canny_shard_dir, 'active_modes': active_modes, 't_eps': args.t_eps, 'n_per_class': args.n_per_class, 'seed': args.seed, 'data_seed': data_seed, 'noise_seed': noise_seed, 't_min': args.t_min, 't_max': args.t_max, 't_points': args.t_points, 'world_size': world_size, 'output_kind': 'v', 'micro_bs': args.micro_bs, 'latent_channels': _RAE_LATENT_C, 'latent_source': 'RawImageMaskCannyDataset+stage1_online_encode', 'phase': 'canny_mask', 'split': 'val', 't_inversion_applied': True, 'sign_flip_applied': True, 'channel_slice_applied': False, 'model': 'RAE_M06', 'modes_description': {'1': 'uncond', '2': 'class', '3': 'class+mask', '4': 'class+mask+canny', '5': 'mask-only', '6': 'canny-only', '7': 'class+canny', '8': 'mask+canny'}, 'mmse_formula': 'MMSE_x1 = (1-t)^2 * MMSE_v (t=1=clean; t_rae=1-t, v=-v_rae; full 768ch)', 'rae_t_note': 't_rae=1-t_project; v_project=-v_rae (see project_rae_t_convention)'})
        manifest_meta = {'checkpoint': args.ckpt_path, 'active_modes': active_modes, 'seed': args.seed, 'data_seed': data_seed, 'noise_seed': noise_seed, 'n_per_class': args.n_per_class, 't_points': args.t_points, 'world_size': world_size, 'latent_source': 'RawImageMaskCannyDataset+stage1_online_encode'}
        if args.per_sample_out and per_sample is not None:
            save_per_sample_errors_pt(Path(args.per_sample_out), t_grid=t_grid, per_sample=per_sample, meta=manifest_meta)
        if args.manifest_out and per_sample is not None:
            write_manifest_jsonl(Path(args.manifest_out), sample_ids=per_sample['sample_ids'].tolist(), labels=per_sample['labels'].tolist(), meta=manifest_meta)
        for key in MODE_KEYS:
            arr = mmse_dict.get(key)
            if arr is not None and np.isfinite(arr).any():
                v = arr[np.isfinite(arr)]
                log.info('%s range=[%.4f, %.4f] valid=%d/%d', key, v.min(), v.max(), len(v), len(arr))
        log.info('Done.')
    dist.destroy_process_group()
if __name__ == '__main__':
    main()
