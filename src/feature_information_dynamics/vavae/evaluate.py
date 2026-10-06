"""eval_m06_vavae_canny_mmse.py — LightningDiT-XL/1 (VAVAE) MMSE eval per t for M06 canny conditioning.

t-convention: LightningDiT Linear path x_t = t*x1 + (1-t)*eps (t=1=clean).
Matches project eval-lib convention — NO t inversion or sign flip needed
(unlike SiT which requires t_sit=1-t and v=-v_sit[:4]).

Output: [B, 32, 16, 16] velocity prediction (full 32 channels, no slice).

EDIT A: _M06UpstreamAdapter.forward — canny_per_token replaces patch_per_token,
        passed as canny= kwarg to LightningDiT_M06.
EDIT B: mode dispatch table — mode 4 = class+mask+canny (was class+mask+patch).
EDIT C: dataset swap — MaskedCannyShardedDataset replaces PairedLatentMaskDataset.

Compared to eval_m06_vavae_mmse.py (legacy patch eval):
  - mode3 ≡ mode4 bug is fixed: canny now actively enters mode-4 forward.
  - Outputs prefixed canny_v4_mmse_modes1234_t64_n16_FIXED to distinguish from buggy run.

Usage::

    torchrun --nproc_per_node=8 --master_port=29733 \\
        scripts/eval_m06_vavae_canny_mmse.py \\
        --ckpt /path/to/best.pt \\
        --modes 1,2,3,4 --n_per_class 16 --t_points 64 --seed 42 \\
        --latent_root <data>/imagenet_latents_f16d32_sr95_v2_labels_fixed/vavae_f16d32_panjiashu/imagenet_train_256 \\
        --mask_shard_dir <repository>/data/mask_shards \\
        --canny_shard_dir <repository>/data/canny_shards \\
        --paired_val_whitelist <repository>/configs/paired_val_whitelist.json \\
        --imagenet_root <data>/imagenet \\
        --out <repository>/results/M06/eval_run/canny_v4_mmse_FIXED.jsonl \\
        --manifest_out <repository>/results/M06/eval_run/canny_v4_mmse_FIXED_manifest.json \\
        --per_sample_out <repository>/results/M06/eval_run/canny_v4_mmse_FIXED_per_sample.jsonl

deliverable_count=1
target_file=scripts/eval_m06_vavae_canny_mmse.py
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
sys.path.insert(0, str(_SCRIPT_DIR))
from feature_information_dynamics.vavae.legacy_adapter import LightningDiT_MaskPatch, NULL_CLASS_LDIT, LDIT_LATENT_C
from feature_information_dynamics.data.train_lib_mask_patch import NUM_TOKENS, build_mask_per_token, build_canny_per_token, load_paper_checkpoint
from feature_information_dynamics.vavae.train import VAVAE_M06_FiLM as LightningDiT_M06
from feature_information_dynamics.data.sharded_mask_canny_dataset import MaskedCannyShardedDataset
from feature_information_dynamics.evaluation import MODE_KEYS, _T_EPS_DEFAULT, check_loss_order, jit_weighted_loss, main_mmse_loop, make_t_grid, per_rank_noise_seed, save_mmse_pt, save_per_sample_errors_pt, write_manifest_jsonl
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
_IMG_SIZE: int = 256
_DEFAULT_LATENT_ROOT = ''

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for LightningDiT-XL/1 (VAVAE) MMSE eval with canny conditioning."""
    p = argparse.ArgumentParser(description='LightningDiT-XL/1 (VAVAE) MMSE eval — M06 canny conditioning (DDP)')
    p.add_argument('--mask_shard_dir', required=True, help='Directory containing masks_rank*.safetensors shards.')
    p.add_argument('--canny_shard_dir', required=True, help='Directory containing cannys_rank*.safetensors shards.')
    p.add_argument('--paired_val_whitelist', default=None, help='Path to paired-val filename whitelist JSON for MaskedCannyShardedDataset.')
    p.add_argument('--out', required=True, help='Output .jsonl path (prefix: canny_v4_mmse_modes1234_t64_n16_FIXED).')
    p.add_argument('--ckpt', default=None, help='M06 upstream checkpoint (train_m06_upstream.py).')
    p.add_argument('--paper_ckpt', default=None, help='Official LightningDiT checkpoint; loaded into backbone only.')
    p.add_argument('--latent_root', default=_DEFAULT_LATENT_ROOT, help='Training latent shard root.')
    p.add_argument('--imagenet_root', default='', help='ImageNet root (kept for CLI compat; not used in canny eval).')
    p.add_argument('--use_hflip', action=argparse.BooleanOptionalAction, default=False, help='Synchronized hflip; default False for deterministic canny eval.')
    p.add_argument('--t_min', type=float, default=0.01)
    p.add_argument('--t_max', type=float, default=0.99)
    p.add_argument('--t_points', type=int, default=256)
    p.add_argument('--t_list_json', default=None, help='JSON-encoded list of explicit t values (overrides t_min/t_max/t_points).')
    p.add_argument('--t_eps', type=float, default=_T_EPS_DEFAULT)
    p.add_argument('--n_per_class', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--micro_bs', type=int, default=16, help='Micro-batch for chunked forward; VAVAE 32ch larger than SiT 4ch.')
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

def _detect_ckpt_arch(src_sd: Dict[str, torch.Tensor]) -> str:
    """Detect ckpt architecture from state_dict shapes.

    Routing rules (positive checks, raise on neither-match):
        Conv2d mask_proj (ndim==4) AND Conv2d x_embedder (ndim==4) → "m06_upstream"
        Linear mask_proj (ndim==2)                                 → "wrapper"
    """
    mask_w = src_sd.get('mask_proj.weight')
    if mask_w is None:
        mask_w = src_sd.get('mask_proj_x.weight')
    x_emb_w = src_sd.get('x_embedder.proj.weight')
    if mask_w is None:
        raise RuntimeError("ckpt missing 'mask_proj.weight' / 'mask_proj_x.weight' — cannot route")
    if mask_w.ndim == 4 and x_emb_w is not None and (x_emb_w.ndim == 4):
        return 'm06_upstream'
    if mask_w.ndim == 2:
        return 'wrapper'
    raise RuntimeError(f'unrecognized mask_proj shape ndim={mask_w.ndim}; expected 2 (Linear) or 4 (Conv2d)')

class _M06UpstreamAdapter(torch.nn.Module):
    """Adapter wrapping LightningDiT_M06 for canny-aware 5-arg eval interface.

    LightningDiT_M06.forward(x, t, y, mask=None, canny=None)

    Eval interface: forward(x, t, y, mask_per_token, canny_per_token)
    Both mask and canny are in per-token format [B, 256, 256] and passed
    as keyword args so LightningDiT_M06 can accept either or both as None.
    """

    def __init__(self, inner: LightningDiT_M06) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, mask_per_token: Optional[torch.Tensor]=None, canny_per_token: Optional[torch.Tensor]=None) -> torch.Tensor:
        """Delegate to LightningDiT_M06 passing mask and canny as kwargs."""
        return self.inner(x, t, y, mask=mask_per_token, canny=canny_per_token)

def _build_m06_upstream_net(src_sd: Dict[str, torch.Tensor]) -> torch.nn.Module:
    """Build LightningDiT_M06 matching the ckpt's architecture, wrapped for eval compat.

    Infers input_size and in_channels from ckpt state_dict shapes.
    LightningDiT_M06 derives mask_proj kernel adaptively from num_patches.
    """
    x_emb_w = src_sd['x_embedder.proj.weight']
    in_channels = int(x_emb_w.shape[1])
    patch_size = int(x_emb_w.shape[2])
    pos_emb = src_sd['pos_embed']
    num_tokens = int(pos_emb.shape[1])
    tokens_per_side = int(round(num_tokens ** 0.5))
    input_size = tokens_per_side * patch_size
    num_classes = 1000
    y_w = src_sd.get('y_embedder.embedding_table.weight')
    if y_w is not None:
        num_classes = int(y_w.shape[0]) - 1
    inner = LightningDiT_M06(input_size=input_size, patch_size=patch_size, in_channels=in_channels, num_classes=num_classes, use_swiglu=True, use_rope=True, use_rmsnorm=True)
    return _M06UpstreamAdapter(inner)

def build_vavae_model(args: argparse.Namespace, device: torch.device, rank: int) -> torch.nn.Module:
    """Instantiate matching model, load eval ckpt (prefers 'ema'), wrap DDP.

    Architecture is auto-detected from ckpt shapes:
      - Conv2d mask_proj → LightningDiT_M06  (train_m06_upstream.py fast-pipeline ckpts)
      - Linear/other     → LightningDiT_MaskPatch  (legacy wrapper ckpts — unchanged path)

    Args:
        args:   parsed CLI namespace with .ckpt / .paper_ckpt fields.
        device: target CUDA device.
        rank:   DDP rank for logging.

    Returns:
        DDP-wrapped model in eval mode.
    """
    if (args.ckpt is None) == (args.paper_ckpt is None):
        raise ValueError('Pass exactly one of --ckpt or --paper_ckpt.')
    if args.paper_ckpt:
        net = LightningDiT_MaskPatch()
        info = load_paper_checkpoint(net.backbone, args.paper_ckpt, adapter_prefixes=())
        if rank == 0:
            log.info('vavae paper ckpt loaded into backbone: key=%s missing=%d unexpected=%d', info['key_used'], len(info['missing_keys']), len(info['unexpected_keys']))
        return DDP(net.to(device).eval(), device_ids=[device.index])
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    src_sd = ck.get('ema', ck.get('model', ck))
    src_sd = {k.removeprefix('module.'): v for k, v in src_sd.items() if isinstance(v, torch.Tensor)}
    arch = _detect_ckpt_arch(src_sd)
    if rank == 0:
        log.info('ckpt arch detected: %s (mask_proj.weight ndim=%s)', arch, src_sd.get('mask_proj.weight', torch.tensor([])).ndim)
    if arch == 'm06_upstream':
        net = _build_m06_upstream_net(src_sd)
        result = net.inner.load_state_dict(src_sd, strict=False)
        _allowed_missing: set[str] = {'canny_proj.weight', 'canny_proj.bias', 'mask_proj_x.weight', 'mask_proj_x.bias', 'canny_proj_x.weight', 'canny_proj_x.bias', 'mask_gates', 'canny_gates'}
        _bad_missing = [k for k in result.missing_keys if k not in _allowed_missing]
        if _bad_missing:
            raise RuntimeError(f'm06_upstream ckpt missing unexpected keys: {_bad_missing}')
        if rank == 0:
            log.info('m06_upstream ckpt loaded strict=False (missing=%d unexpected=%d bad_missing=%d)', len(result.missing_keys), len(result.unexpected_keys), len(_bad_missing))
    else:
        net = LightningDiT_MaskPatch()
        result = net.load_state_dict(src_sd, strict=False)
        if rank == 0:
            adapter_pfx = ('mask_proj', 'patch_proj')
            non_adapter_missing = [k for k in result.missing_keys if not any((k.startswith(p) for p in adapter_pfx))]
            log.info('vavae wrapper ckpt loaded: missing=%d unexpected=%d non-adapter-missing=%d', len(result.missing_keys), len(result.unexpected_keys), len(non_adapter_missing))
    return DDP(net.to(device).eval(), device_ids=[device.index])

def load_eval_shard(args: argparse.Namespace, rank: int, world_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build canny-aware dataset; return (z1, mask_pt, canny_pt, labels, sample_ids).

    Uses MaskedCannyShardedDataset with phase='canny_mask' so both mask AND
    canny shards are loaded.  Dataset returns (latent, label, mask, canny) 4-tuple.

    Returns:
        z1        : [N, 32, 16, 16] VAVAE latents (clean, training-normalized)
        mask_pt   : [N, 256, 256] per-token mask float32
        canny_pt  : [N, 256, 256] per-token canny float32
        labels    : [N] long
        sample_ids: [N] long
    """
    _is_val = bool(args.paired_val_whitelist)
    if rank == 0 and _is_val:
        log.info('[paired-val mode] using paired_val_whitelist=%s is_val=True', args.paired_val_whitelist)
    ds = MaskedCannyShardedDataset(data_dir=args.latent_root, mask_shard_dir=args.mask_shard_dir, canny_shard_dir=args.canny_shard_dir, latent_norm=True, latent_multiplier=1.0, split='val' if _is_val else 'train', paired_whitelist_json=args.paired_val_whitelist, phase='canny_mask', is_val=_is_val)
    rng = np.random.default_rng(args.seed)
    cls_to_idx: Dict[int, List[int]] = {}
    for i in range(len(ds)):
        global_idx = ds.valid_indices[i]
        info = ds._full_map[global_idx]
        with __import__('safetensors').safe_open(info['safe_file'], framework='pt', device='cpu') as f:
            cls = int(f.get_slice('labels')[info['idx_in_file']].item())
        cls_to_idx.setdefault(cls, []).append(i)
    wl_cls = sorted(cls_to_idx.keys())
    if rank == 0:
        log.info('Dataset: %d latent samples, %d classes, phase=canny_mask, is_val=%s', len(ds), len(wl_cls), _is_val)
    selected: List[int] = []
    for cls in wl_cls:
        idxs = cls_to_idx[cls]
        selected.extend(rng.choice(idxs, min(args.n_per_class, len(idxs)), replace=False).tolist())
    local_indices = selected[rank::world_size]
    acc: Dict[str, List] = {'z1': [], 'mask_pt': [], 'canny_pt': [], 'lbl': []}
    for batch in DataLoader(Subset(ds, local_indices), batch_size=16, shuffle=False, num_workers=args.num_workers, pin_memory=True):
        latent, cls, mask_2d, canny_2d = batch
        z = latent.to(device).float()
        mask_2d = mask_2d.to(device)
        canny_2d = canny_2d.to(device)
        acc['z1'].append(z.cpu())
        acc['mask_pt'].append(mask_2d.cpu())
        acc['canny_pt'].append(canny_2d.cpu())
        acc['lbl'].append(cls)
    return (torch.cat(acc['z1']).to(device), torch.cat(acc['mask_pt']).to(device), torch.cat(acc['canny_pt']).to(device), torch.cat(acc['lbl']).to(device), torch.tensor(local_indices, dtype=torch.long, device=device))

def make_mode_kwargs_fn(labels: torch.Tensor, mask_pt: torch.Tensor, canny_pt: torch.Tensor, device: torch.device) -> Callable[[int], Dict]:
    """Closure: mode_idx -> forward kwargs for _M06UpstreamAdapter.

    Modes:
        1 = uncond           (null class, zero mask, zero canny)
        2 = class            (real class, zero mask, zero canny)
        3 = class+mask       (real class, real mask, zero canny)
        4 = class+mask+canny (real class, real mask, real canny)  ← EDIT B
        5 = mask             (null class, real mask, zero canny)
        6 = canny            (null class, zero mask, real canny)
        7 = class+canny      (real class, zero mask, real canny)
        8 = mask+canny       (null class, real mask, real canny)
    """
    N = labels.shape[0]
    null_y = torch.full((N,), NULL_CLASS_LDIT, dtype=torch.long, device=device)
    zero_mask = torch.zeros(N, NUM_TOKENS, NUM_TOKENS, device=device)
    zero_canny = torch.zeros(N, NUM_TOKENS, NUM_TOKENS, device=device)
    _CFG: Dict[int, Tuple] = {1: (null_y, zero_mask, zero_canny), 2: (labels, zero_mask, zero_canny), 3: (labels, mask_pt, zero_canny), 4: (labels, mask_pt, canny_pt), 5: (null_y, mask_pt, zero_canny), 6: (null_y, zero_mask, canny_pt), 7: (labels, zero_mask, canny_pt), 8: (null_y, mask_pt, canny_pt)}

    def _fn(m: int) -> Dict:
        y, mp, cp = _CFG[m]
        return {'y': y, 'mask_per_token': mp, 'canny_per_token': cp}
    return _fn

def make_model_forward_fn(net: torch.nn.Module, micro_bs: int=16) -> Callable:
    """Wrap DDP _M06UpstreamAdapter for eval_lib_mmse interface.

    NO t inversion (t=1=clean matches project), NO sign flip, NO first-4ch slice.
    Output is full [B, 32, 16, 16] velocity prediction.
    Processes in micro-batches to avoid OOM on 32ch latents.
    """

    def _fwd(x_t: torch.Tensor, t: float, y: torch.Tensor, mask_per_token: Optional[torch.Tensor]=None, canny_per_token: Optional[torch.Tensor]=None) -> torch.Tensor:
        """VAVAE canny adapter: direct forward, no t inversion. [B,32,16,16] → [B,32,16,16]."""
        B = x_t.shape[0]
        outputs = []
        for s_idx in range(0, B, micro_bs):
            e_idx = min(s_idx + micro_bs, B)
            x_chunk = x_t[s_idx:e_idx]
            y_chunk = y[s_idx:e_idx]
            mask_chunk = mask_per_token[s_idx:e_idx] if mask_per_token is not None else None
            canny_chunk = canny_per_token[s_idx:e_idx] if canny_per_token is not None else None
            chunk_B = x_chunk.shape[0]
            t_t = torch.full((chunk_B,), t, device=x_t.device, dtype=x_t.dtype)
            v_pred = net(x_chunk, t_t, y_chunk, mask_chunk, canny_chunk)
            outputs.append(v_pred)
        return torch.cat(outputs, dim=0)
    return _fwd

def main() -> None:
    """DDP MMSE evaluation for LightningDiT-XL/1 (VAVAE) M06 canny conditioning."""
    args = parse_args()
    try:
        active_modes = sorted({int(m.strip()) for m in args.modes.split(',')})
    except ValueError as exc:
        raise ValueError(f'--modes must be comma-separated ints: {args.modes!r}') from exc
    if (bad := [m for m in active_modes if m not in range(1, 9)]):
        raise ValueError(f'Invalid mode(s) {bad}; valid range 1-8')
    rank, world_size, device = setup_dist()
    torch.manual_seed(per_rank_noise_seed(args.seed, rank))
    if rank == 0:
        ckpt_label = args.ckpt or args.paper_ckpt
        log.info('VAVAE canny eval | modes=%s t_eps=%.3f ckpt=%s', active_modes, args.t_eps, ckpt_label)
    net = build_vavae_model(args, device, rank)
    if args.t_list_json:
        import json as _json
        _t_vals = _json.loads(args.t_list_json)
        t_grid = torch.tensor(_t_vals, dtype=torch.float64)
        if rank == 0:
            log.info('[t_list] custom t (n=%d): min=%.6f max=%.6f', len(_t_vals), float(min(_t_vals)), float(max(_t_vals)))
    else:
        t_grid = make_t_grid(args.t_min, args.t_max, args.t_points)
    z1, mask_pt, canny_pt, labels, sample_ids = load_eval_shard(args, rank, world_size, device)
    x0 = torch.randn_like(z1)
    dist.barrier()
    per_sample: Optional[Dict[str, object]] = {} if args.per_sample_out or args.manifest_out else None
    mmse_dict = main_mmse_loop(rank=rank, world_size=world_size, t_grid=t_grid, x1=z1, x0=x0, model_forward_fn=make_model_forward_fn(net, args.micro_bs), active_modes=active_modes, mode_kwargs_fn=make_mode_kwargs_fn(labels, mask_pt, canny_pt, device), t_eps=args.t_eps, output_kind='v', device=device, logger=log, sample_ids=sample_ids, labels=labels, per_sample_out=per_sample)
    dist.barrier()
    if rank == 0:
        jit_losses: Optional[Dict[str, float]] = None
        order_ok: Optional[bool] = None
        if set(active_modes) == {1, 2, 3, 4}:
            jit_losses = {k: jit_weighted_loss(t_grid, mmse_dict[k]) for k in MODE_KEYS}
            order_ok = check_loss_order({k.removeprefix('mmse_'): v for k, v in jit_losses.items()})
        save_mmse_pt(out_path=Path(args.out), t_grid=t_grid, mmse_dict=mmse_dict, jit_weighted_losses=jit_losses, loss_order_ok=order_ok, meta={'checkpoint': args.ckpt or args.paper_ckpt, 'checkpoint_kind': 'paper' if args.paper_ckpt else 'm06_upstream', 'latent_root': args.latent_root, 'mask_shard_dir': args.mask_shard_dir, 'canny_shard_dir': args.canny_shard_dir, 'imagenet_root': args.imagenet_root, 'active_modes': active_modes, 't_eps': args.t_eps, 'n_per_class': args.n_per_class, 'seed': args.seed, 't_min': args.t_min, 't_max': args.t_max, 't_points': args.t_points, 'world_size': world_size, 'output_kind': 'v', 'micro_bs': args.micro_bs, 'latent_channels': LDIT_LATENT_C, 'latent_source': 'MaskedCannyShardedDataset', 'latent_norm': True, 'latent_multiplier': 1.0, 'phase': 'canny_mask', 'split': 'val' if args.paired_val_whitelist else 'train', 't_inversion_applied': False, 'sign_flip_applied': False, 'channel_slice_applied': False, 'model': 'LightningDiT_M06', 'modes_description': {'1': 'uncond', '2': 'class', '3': 'class+mask', '4': 'class+mask+canny', '5': 'mask', '6': 'canny', '7': 'class+canny', '8': 'mask+canny'}, 'mmse_formula': 'MMSE_x1 = (1-t)^2 * MMSE_v (t=1=clean, no inversion; v=[B,32,16,16] full output)'})
        manifest_meta = {'checkpoint': args.ckpt or args.paper_ckpt, 'active_modes': active_modes, 'seed': args.seed, 'n_per_class': args.n_per_class, 't_points': args.t_points, 'world_size': world_size, 'latent_source': 'MaskedCannyShardedDataset'}
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
