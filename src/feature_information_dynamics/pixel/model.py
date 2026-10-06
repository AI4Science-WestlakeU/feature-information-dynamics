"""Adapted from M06 FiLM implementation; native upstream architecture required.

Pixel injection targets trailing spatial tokens, after prepended class tokens.
"""
from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
from model_jit import JiT_models

class JiT_M06_FiLM(nn.Module):

    def __init__(self, input_size: int=256, num_classes: int=1000, use_checkpoint: bool=False, class_dropout_prob: float=0.1, model_name: str='JiT-L/16') -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
        if model_name not in JiT_models:
            raise ValueError(f'Unsupported JiT model {model_name!r}; available={sorted(JiT_models)}')
        self.model_name = model_name
        self.backbone = JiT_models[model_name](input_size=input_size, in_channels=3, num_classes=num_classes, mask_cond=False, patch_cond=False)
        hidden_size = int(self.backbone.hidden_size)
        patch_size = int(self.backbone.patch_size)
        self.mask_proj_x = nn.Conv2d(1, hidden_size, kernel_size=patch_size, stride=patch_size, bias=False)
        self.canny_proj_x = nn.Conv2d(1, hidden_size, kernel_size=patch_size, stride=patch_size, bias=False)
        nn.init.kaiming_uniform_(self.mask_proj_x.weight, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.canny_proj_x.weight, a=5 ** 0.5)
        num_blocks: int = len(self.backbone.blocks)
        self.mask_gates = nn.Parameter(torch.zeros(num_blocks))
        self.canny_gates = nn.Parameter(torch.zeros(num_blocks))
        self.mask_null_embed = nn.Parameter(torch.zeros(hidden_size))
        self.canny_null_embed = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor]=None, canny: Optional[torch.Tensor]=None, mask_drop: Optional[torch.Tensor]=None, canny_drop: Optional[torch.Tensor]=None) -> torch.Tensor:
        mask_full = None
        canny_full = None
        if mask is not None:
            m = mask if mask.dtype.is_floating_point else mask.float()
            if m.dim() == 3:
                m = m.unsqueeze(1)
            mask_full = self.mask_proj_x(m).flatten(2).transpose(1, 2)
            if mask_drop is not None:
                B_ = mask_full.shape[0]
                null_exp = self.mask_null_embed.view(1, 1, -1).expand_as(mask_full)
                drop_b = mask_drop.view(B_, 1, 1).bool()
                mask_full = torch.where(drop_b, null_exp, mask_full)
        if canny is not None:
            ce = canny if canny.dtype.is_floating_point else canny.float()
            if ce.dim() == 3:
                ce = ce.unsqueeze(1)
            canny_full = self.canny_proj_x(ce).flatten(2).transpose(1, 2)
            if canny_drop is not None:
                B_ = canny_full.shape[0]
                null_exp = self.canny_null_embed.view(1, 1, -1).expand_as(canny_full)
                drop_b = canny_drop.view(B_, 1, 1).bool()
                canny_full = torch.where(drop_b, null_exp, canny_full)
        if mask_full is None and canny_full is None:
            return self.backbone(x, t, y)
        handles: list[torch.utils.hooks.RemovableHook] = []
        for i, block in enumerate(self.backbone.blocks):
            g_mask = self.mask_gates[i]
            g_canny = self.canny_gates[i]
            _mf = mask_full
            _cf = canny_full

            def _make_hook(gm: torch.Tensor, gc: torch.Tensor, mf: Optional[torch.Tensor], cf: Optional[torch.Tensor]):

                def _hook(module: nn.Module, inputs: tuple) -> tuple:
                    if not inputs or not isinstance(inputs[0], torch.Tensor):
                        return inputs
                    tok = inputs[0]
                    N_patch = mf.shape[1] if mf is not None else cf.shape[1]
                    delta = torch.zeros_like(tok)
                    if mf is not None:
                        delta[:, -N_patch:, :] = delta[:, -N_patch:, :] + gm * mf
                    if cf is not None:
                        delta[:, -N_patch:, :] = delta[:, -N_patch:, :] + gc * cf
                    return (tok + delta,) + inputs[1:]
                return _hook
            h = block.register_forward_pre_hook(_make_hook(g_mask, g_canny, _mf, _cf))
            handles.append(h)
        try:
            out = self.backbone(x, t, y)
        finally:
            for h in handles:
                h.remove()
        return out
