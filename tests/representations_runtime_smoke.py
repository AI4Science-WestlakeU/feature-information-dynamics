"""Opt-in native latent-model smoke on synthetic latents, without training I/O.

Each invocation isolates upstream modules. SDVAE uses random full XL weights;
RAE and VAVAE use explicitly reduced native constructor dimensions.
"""
import argparse
import copy
import importlib
import json
import os
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("representation", choices=["rae", "sdvae", "vavae"])
    p.add_argument("--upstream-source", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--config", required=True, help="Native YAML supplying transport and optimizer settings")
    args = p.parse_args()
    from feature_information_dynamics.workflows import source_paths, MODULES, PREFIX
    sys.path[:0] = source_paths(args.representation, "train", args.upstream_source)
    import torch
    import yaml
    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    torch.manual_seed(71)
    torch.set_num_threads(2)
    m = importlib.import_module(PREFIX + MODULES[(args.representation, "train")])
    device = torch.device(args.device)
    if args.representation == "rae":
        spec = dict(input_size=16, patch_size=1, in_channels=768, hidden_size=[64, 128],
                    depth=[2, 1], num_heads=[4, 4], class_dropout_prob=0., num_classes=1001)
        model = m.RAE_M06(**spec)
        shape = [1, 768, 16, 16]
    elif args.representation == "vavae":
        spec = dict(input_size=16, patch_size=1, in_channels=32, hidden_size=64,
                    depth=2, num_heads=4, class_dropout_prob=0., num_classes=1001,
                    learn_sigma=False, use_checkpoint=False)
        model = m.VAVAE_M06_FiLM(**spec)
        shape = [1, 32, 16, 16]
    else:
        spec = dict(input_size=32, num_classes=config["data"]["num_classes"],
                    learn_sigma=config["model"].get("learn_sigma", True),
                    use_checkpoint=False,
                    class_dropout_prob=config["model"].get("class_dropout_prob", .1))
        model = m.SiT_M06_FiLM(**spec)
        shape = [1, 4, 32, 32]
    model = model.to(device)
    ema = copy.deepcopy(model).requires_grad_(False)
    x = torch.randn(shape, device=device)
    y = torch.tensor([3], device=device)
    conditions = dict(y=y, mask=torch.ones(1, 256, 256, device=device),
                      canny=torch.zeros(1, 256, 256, device=device))
    transport_spec = config["transport"]
    transport = m.create_transport(**transport_spec)
    opt_config = config["optimizer"]
    optimizer_spec = dict(lr=opt_config["lr"], betas=(.9, opt_config["beta2"]), weight_decay=0.)
    adapter_mult = opt_config.get("adapter_lr_multiplier", 1.)
    groups = [[], []]
    for name, value in model.named_parameters():
        is_adapter = any(k in name for k in ("mask_proj_x", "canny_proj_x", "mask_gates", "canny_gates"))
        groups[int(is_adapter)].append(value)
    optimizer = torch.optim.AdamW(
        [{"params": groups[0], "lr": optimizer_spec["lr"]},
         {"params": groups[1], "lr": optimizer_spec["lr"] * adapter_mult}],
        betas=optimizer_spec["betas"], weight_decay=0.,
        fused=args.representation in ("rae", "sdvae") and opt_config.get("fused_adamw", True))
    optimizer.zero_grad()
    loss_dict = transport.training_losses(model, x, conditions)
    loss = loss_dict["loss"].mean()
    if args.representation == "vavae":
        loss = loss + loss_dict["cos_loss"].mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
    if "max_grad_norm" in opt_config:
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt_config["max_grad_norm"])
    optimizer.step()
    m.update_ema(ema, model)
    assert all(torch.isfinite(v).all() for v in ema.parameters())
    print(json.dumps(dict(status="pass", representation=args.representation,
                          device=str(device), torch=torch.__version__, shape=shape,
                          constructor=spec, transport=transport_spec, optimizer=optimizer_spec,
                          loss=float(loss.detach()), loss_keys=list(loss_dict),
                          backward_optimizer_ema=True, pretrained_weights=False,
                          cuda_peak_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                          adapter_lr_multiplier=adapter_mult,
                          limits="Synthetic latent inputs; no encoder, real data, checkpoint, phase freezing or distributed resume.")))


if __name__ == "__main__":
    main()
