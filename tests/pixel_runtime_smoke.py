"""Opt-in bounded real-JiT smoke; run with --jit-source and optional --device.

Uses random tiny weights, never the training entrypoint or user datasets.
"""
import argparse
import ast
import copy
import json
import importlib.util
import math
import os
from collections import OrderedDict
from pathlib import Path
import sys
from typing import Optional

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jit-source", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.jit_source).resolve()))
    import torch
    from torch import nn
    import model_jit
    from feature_information_dynamics.pixel.model import JiT_M06_FiLM

    model_jit.JiT_models["audit-tiny"] = lambda **kw: model_jit.JiT(
        patch_size=4, hidden_size=64, depth=3, num_heads=4, bottleneck_dim=32,
        in_context_len=2, in_context_start=1, **kw)
    torch.manual_seed(13)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    model = JiT_M06_FiLM(input_size=16, model_name="audit-tiny").to(device)
    # Some historical upstream RoPE tables are ordinary CUDA attributes.
    for module in model.modules():
        for name in ("freqs_cos", "freqs_sin"):
            if hasattr(module, name):
                setattr(module, name, getattr(module, name).to(device))
    x = torch.randn(2, 3, 16, 16, device=device)
    y = torch.tensor([2, 3], device=device)
    mask = torch.ones(2, 16, 16, device=device)
    canny = torch.zeros_like(mask)
    with torch.no_grad():
        t = torch.full((2,), .3, device=device)
        assert torch.equal(model(x, t, y, mask=mask, canny=canny), model.backbone(x, t, y))
        before, after, observers = {}, {}, []
        for i, block in enumerate(model.backbone.blocks):
            def capture_before(module, values, index=i):
                before[index] = values[0].clone()
            def capture_after(module, values, output, index=i):
                after[index] = values[0].clone()
            observers.append(block.register_forward_pre_hook(capture_before))
            observers.append(block.register_forward_hook(capture_after))
        model.mask_gates.fill_(1.)
        model(x, t, y, mask=mask)
        expected = model.mask_proj_x(mask.unsqueeze(1)).flatten(2).transpose(1, 2)
        for i in before:
            delta = after[i] - before[i]
            assert torch.allclose(delta[:, -16:], expected, atol=1e-6)
            if i >= 1:
                assert torch.equal(delta[:, :2], torch.zeros_like(delta[:, :2]))
        for observer in observers:
            observer.remove()
        assert not any(block._forward_pre_hooks for block in model.backbone.blocks)
        model.mask_gates.zero_()
        model.backbone.final_layer.linear.weight.normal_(std=.01)
    # Execute the exact migrated scientific functions without trainer I/O imports.
    source = Path(importlib.util.find_spec("feature_information_dynamics.pixel.train").origin)
    names = {"_sample_t_lognormal", "_sample_t_fixed", "_sample_t_full_snr_mixture",
             "_compute_velocity_loss", "update_ema"}
    tree = ast.parse(source.read_text(encoding="utf-8"))
    scope = dict(torch=torch, nn=nn, math=math, Optional=Optional, OrderedDict=OrderedDict)
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                                 and n.name in names], type_ignores=[]), str(source), "exec"), scope)
    ema = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    losses = {}
    for name, sampler in (("native", None), ("fixed", {"mode": "fixed_t", "fixed_t": .3})):
        optimizer.zero_grad()
        loss = scope["_compute_velocity_loss"](model, x, y, device, -.8, .8, .05,
                                                 mask=mask, canny=canny, sampler_config=sampler)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        losses[name] = float(loss.detach())
        optimizer.step()
        scope["update_ema"](ema, model)
    assert all(torch.isfinite(p).all() for p in ema.parameters())
    print(json.dumps({"status": "pass", "device": str(device), "torch": torch.__version__,
                      "zero_gate_identity": True, "loss_backward_optimizer_ema": True,
                      "real_tail_token_alignment_and_hook_cleanup": True,
                      "losses": losses, "shape": list(x.shape), "tiny_random_weights": True}))


if __name__ == "__main__":
    main()
