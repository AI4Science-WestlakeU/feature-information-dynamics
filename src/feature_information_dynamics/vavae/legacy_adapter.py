from typing import Optional
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from models.lightningdit import LightningDiT
from feature_information_dynamics.data.train_lib_mask_patch import NUM_TOKENS, DINO_DIM, set_backbone_class_dropout, build_mask_per_token, build_canny_per_token
LDIT_HIDDEN: int = 1152
LDIT_LATENT_C: int = 32
LDIT_LATENT_H: int = 16
NULL_CLASS_LDIT: int = 1000

class LightningDiT_MaskPatch(LightningDiT):
    """LightningDiT-XL/1 with zero-init mask+patch+canny per-token adapters.

    Directly inherits LightningDiT (no self.backbone wrapper) to match upstream
    forward-path overhead. Adapters injected into x_tok before blocks when any
    conditioning is present; skipped entirely when all conditioning is None (F1).
    t follows project convention (t=1=clean) — no inversion needed.
    """

    def __init__(self, input_size: int=LDIT_LATENT_H, in_channels: int=LDIT_LATENT_C, num_classes: int=1000, use_checkpoint: bool=True) -> None:
        """Build LightningDiT-XL/1 base + 3 zero-init adapter projections."""
        super().__init__(input_size=input_size, patch_size=1, in_channels=in_channels, hidden_size=1152, depth=28, num_heads=16, num_classes=num_classes, use_qknorm=False, use_swiglu=True, use_rope=True, use_rmsnorm=True, wo_shift=False, use_checkpoint=use_checkpoint)
        self.mask_proj = nn.Linear(NUM_TOKENS, LDIT_HIDDEN, bias=True)
        self.patch_proj = nn.Linear(DINO_DIM, LDIT_HIDDEN, bias=True)
        self.canny_proj = nn.Linear(NUM_TOKENS, LDIT_HIDDEN, bias=True)
        for layer in (self.mask_proj, self.patch_proj, self.canny_proj):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        set_backbone_class_dropout(self, 0.0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, mask_per_token: Optional[torch.Tensor]=None, patch_per_token: Optional[torch.Tensor]=None, canny_per_token: Optional[torch.Tensor]=None) -> torch.Tensor:
        """Forward with optional per-token mask/patch/canny conditioning.

        Args:
            x              : [B, 32, 16, 16] noisy VAVAE latent
            t              : [B] project timestep (t=1=clean)
            y              : [B] class label (NULL_CLASS_LDIT=1000 for uncond)
            mask_per_token : [B, 256, 256] or None
            patch_per_token: [B, 256, 768] or None
            canny_per_token: [B, 256, 256] or None

        Returns:
            [B, 32, 16, 16] velocity prediction
        """
        x_tok = self.x_embedder(x) + self.pos_embed
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y, self.training)
        c = t_emb + y_emb
        if mask_per_token is not None or patch_per_token is not None or canny_per_token is not None:
            delta = torch.zeros(x.shape[0], NUM_TOKENS, LDIT_HIDDEN, device=x.device, dtype=x.dtype)
            if mask_per_token is not None:
                delta = delta + self.mask_proj(mask_per_token.to(x.dtype))
            if patch_per_token is not None:
                delta = delta + self.patch_proj(patch_per_token.to(x.dtype))
            if canny_per_token is not None:
                delta = delta + self.canny_proj(canny_per_token.to(x.dtype))
            x_tok = x_tok + delta
        for block in self.blocks:
            if self.use_checkpoint:
                x_tok = checkpoint(block, x_tok, c, self.feat_rope, use_reentrant=False)
            else:
                x_tok = block(x_tok, c, self.feat_rope)
        out = self.final_layer(x_tok, c)
        out = self.unpatchify(out)
        if self.learn_sigma:
            out, _ = out.chunk(2, dim=1)
        return out
