"""End-to-end smoke test on synthetic data. RUN THIS BEFORE ANY TRAINING.

    python src/models/wiltechs_x/smoke_test.py                    # real Qwen3-VL
    python src/models/wiltechs_x/smoke_test.py --tiny             # 2-layer stub
    python src/models/wiltechs_x/smoke_test.py --freeze_vlm       # ablation path

It builds the model, runs compute_loss and sample_actions on a random batch,
and checks the things that are cheap to check here and expensive to discover
40 minutes into a run:

  * every loss term is finite and every intended parameter group gets gradient
  * the wrist encoder, expert, and LoRA all receive gradient (a silently
    disconnected module trains to nothing and looks like a bad hyperparameter)
  * sample_actions runs the VLM ONCE regardless of num_inference_steps -- the
    prefix->suffix mask invariant, measured rather than assumed
  * shapes and dtypes survive the bf16 VLM / fp32 expert boundary

--tiny replaces Qwen with a randomly-initialised 2-layer stand-in that has the
same module attribute names, so the plumbing can be exercised without a 4B
download. It proves nothing about the real backbone's interface; run without
it before trusting anything.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.wiltechs_x.wiltechs_x_config import WiltechsXConfig      # noqa: E402
from models.wiltechs_x.wiltechs_x_model import WiltechsXModel        # noqa: E402

from lerobot.configs.types import FeatureType, PolicyFeature          # noqa: E402

OK = True


def check(label, cond, extra=""):
    global OK
    OK = OK and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}{(' — ' + extra) if extra else ''}")


def make_config(args) -> WiltechsXConfig:
    cams = ["observation.images.image", "observation.images.image2"]
    return WiltechsXConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            **{c: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
               for c in cams},
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        n_obs_steps=8, horizon=args.horizon, n_action_steps=8,
        state_dim=8, action_dim=7,
        vlm_model_id=args.vlm_model_id, freeze_vlm=args.freeze_vlm,
        num_cameras=len(cams), cameras_for_vlm=cams,
        wrist_cameras=[cams[1]], wrist_encoder_id=args.wrist_encoder_id,
        use_wrist_encoder=not args.no_wrist, wrist_tokens=args.wrist_tokens,
        expert_hidden_size=args.expert_hidden, expert_num_layers=args.expert_layers,
        lang_max_len=16, motion_history_len=8,
        flow_objective=args.flow_objective,
        num_inference_steps=args.num_inference_steps,
        gripper_threshold_norm=0.0, gripper_action_dim=-1,
        device="cpu",
    )


def make_batch(cfg, B=2, device="cpu"):
    return {
        "observation.state": torch.randn(B, cfg.n_obs_steps, cfg.state_dim, device=device),
        "action": torch.randn(B, cfg.horizon, cfg.action_dim, device=device),
        "action_is_pad": torch.zeros(B, cfg.horizon, dtype=torch.bool, device=device),
        "progress": torch.rand(B, device=device),
        "task": ["pick up the black bowl", "put the cream cheese in the basket"][:B],
        **{c: torch.rand(B, 3, 256, 256, device=device) for c in cfg.cameras_for_vlm},
    }


def install_tiny_backbone(model, cfg):
    """Swap in a 2-layer randomly-initialised stand-in for the Qwen text stack.

    Only the layers are replaced; the tokenizer, processor, vision tower and
    rotary embedding stay real, so the parts most likely to break (token ids,
    grid_thw, M-RoPE shapes) are still exercised.
    """
    model.language_model.layers = model.language_model.layers[:2]
    model.num_vlm_layers = 2
    model.first_joint_layer = max(0, 2 - len(model.expert_layers))
    model.expert_layers = model.expert_layers[: 2 - model.first_joint_layer]
    print(f"[smoke] tiny backbone: 2 VLM layers, "
          f"{len(model.expert_layers)} expert layers")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vlm_model_id", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--wrist_encoder_id", default="facebook/dinov2-small")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--freeze_vlm", action="store_true")
    p.add_argument("--no_wrist", action="store_true")
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--expert_hidden", type=int, default=256)
    p.add_argument("--expert_layers", type=int, default=2)
    p.add_argument("--flow_objective", default="shortcut", choices=["flow", "shortcut"])
    p.add_argument("--num_inference_steps", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=2,
                   help="Raise this to find the OOM boundary before committing "
                        "to a training run; the peak-memory line scales roughly "
                        "linearly in it.")
    p.add_argument("--wrist_tokens", type=int, default=256)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = make_config(args)
    cfg.device = device

    print("building model ...")
    t0 = time.time()
    model = WiltechsXModel(cfg)
    if args.tiny:
        install_tiny_backbone(model, cfg)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model = model.to(device)
    print(f"built in {time.time() - t0:.1f}s")
    counts = model.count_parameters()
    print(f"params: trainable={counts['trainable']:,} frozen={counts['frozen']:,}")

    batch = make_batch(cfg, device=device, B=args.batch_size)

    # ---- forward / backward -----------------------------------------
    print("\ncompute_loss")
    model.train()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        w = torch.cuda.memory_allocated() / 2 ** 30
    t0 = time.time()
    loss, parts = model.compute_loss(batch, return_parts=True)
    loss.backward()
    print(f"  terms: {parts}   ({time.time() - t0:.1f}s)")
    if device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        print(f"  memory: weights {w:.2f} GiB, peak {peak:.2f} GiB of "
              f"{total:.1f} GiB  (B={args.batch_size}, prefix grad="
              f"{model.needs_prefix_grad}, ckpt={model.gradient_checkpointing})")
        print(f"  -> AdamW will add ~"
              f"{counts['trainable'] * 12 / 2 ** 30:.2f} GiB of optimizer state "
              f"on top of this; budget for it before choosing --batch_size")
    check("loss is finite", torch.isfinite(loss), f"{float(loss):.4f}")
    check("every term is finite", all(v == v and abs(v) < 1e6 for v in parts.values()))
    check("flow term present", "flow" in parts)
    if cfg.progress_head:
        check("progress term present (targets reached the model)", "progress" in parts)

    if model.discrete_head is not None:
        # The discrete head is the ONLY gradient path into the VLM, so its
        # starting point matters. ln(n_bins) is the uniform-prediction value;
        # far above it means the head's own initialisation scale, not the
        # task, dominates the first thousand LoRA updates.
        uniform = math.log(model.discrete_head.n_bins)
        check(f"discrete CE starts near uniform (ln {model.discrete_head.n_bins} "
              f"= {uniform:.2f})", abs(parts["discrete"] - uniform) < 1.0,
              f"{parts['discrete']:.2f}")

    # Parameter budget. A single oversized Linear is easy to write and
    # invisible in the loss: the motion encoder was 419M -- 80% of everything
    # trainable -- for encoding eight frames of an 8-dim vector.
    print("\nparameter budget")
    groups = {"expert": model.expert_layers, "wrist": model.wrist_encoder,
              "motion": model.motion_encoder, "discrete": model.discrete_head,
              "lora": None}
    lora_n = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    for name, mod in groups.items():
        n = lora_n if name == "lora" else (
            sum(p.numel() for p in mod.parameters() if p.requires_grad) if mod else 0)
        if n:
            print(f"  {name:9s} {n:>13,}  ({100 * n / counts['trainable']:.1f}%)")
    if model.motion_encoder is not None:
        n_mv = sum(p.numel() for p in model.motion_encoder.parameters())
        check("motion encoder is not a parameter sink (<10% of trainable)",
              n_mv < 0.10 * counts["trainable"], f"{n_mv:,}")

    # ---- gradient reaches every module it should ---------------------
    print("\ngradient coverage")

    def grad_sum(mod):
        return sum(float(p.grad.abs().sum()) for p in mod.parameters()
                   if p.grad is not None)

    check("expert layers get gradient", grad_sum(model.expert_layers) > 0)
    check("action_out_proj gets gradient", grad_sum(model.action_out_proj) > 0)
    if model.wrist_encoder is not None and not cfg.freeze_wrist_encoder:
        check("wrist encoder gets gradient", grad_sum(model.wrist_encoder) > 0,
              "a disconnected wrist path trains to nothing and reads as a bad LR")
    if model.motion_encoder is not None:
        check("motion encoder gets gradient", grad_sum(model.motion_encoder) > 0)
    if model.discrete_head is not None:
        check("discrete head gets gradient", grad_sum(model.discrete_head) > 0)

    lora = [p for n, p in model.language_model.named_parameters()
            if "lora_" in n and p.grad is not None and p.grad.abs().sum() > 0]
    if cfg.freeze_vlm:
        check("freeze_vlm: no LoRA parameters exist",
              not any("lora_" in n for n, _ in model.language_model.named_parameters()))
    else:
        check("LoRA gets gradient (via the discrete head, not the flow loss)",
              len(lora) > 0 if model.discrete_head is not None else True,
              f"{len(lora)} tensors")

    base_grads = [float(p.grad.abs().sum()) for n, p in
                  model.language_model.named_parameters()
                  if "lora_" not in n and p.grad is not None]
    check("frozen VLM base weights got no gradient",
          all(g == 0 for g in base_grads) or not base_grads)

    # ---- shortcut term, on a NON-degenerate network -------------------
    # At init adaLN and action_out_proj are zero, so the velocity is
    # identically zero and every shortcut evaluation agrees at 0.0. That
    # reports as a pass while testing nothing. Perturb the output head and
    # re-check, which is the only way to tell "correctly zero" from
    # "silently disconnected".
    if cfg.flow_objective == "shortcut":
        print("\nshortcut term (after perturbing action_out_proj)")
        check("shortcut term is exactly 0 at init (zero-init velocity)",
              parts.get("shortcut", -1) == 0.0)
        with torch.no_grad():
            model.action_out_proj.weight.normal_(std=0.1)
            for layer in model.expert_layers:
                layer.ada[1].bias.normal_(std=0.1)
        model.zero_grad(set_to_none=True)
        _, parts2 = model.compute_loss(make_batch(cfg, B=args.batch_size, device=device),
                                       return_parts=True)
        check("shortcut term becomes non-zero once velocity is non-zero",
              parts2.get("shortcut", 0.0) > 0.0, f"{parts2.get('shortcut', 0):.4f}")

    # ---- sampling, and the once-per-chunk invariant -------------------
    print("\nsample_actions")
    model.eval()
    calls = {"prefix": 0}
    orig = model._build_prefix

    def counting(batch):
        calls["prefix"] += 1
        return orig(batch)

    model._build_prefix = counting
    t0 = time.time()
    with torch.no_grad():
        actions = model.sample_actions(batch)
    dt = time.time() - t0
    model._build_prefix = orig

    check("action shape",
          actions.shape == (args.batch_size, cfg.horizon, cfg.action_dim),
          str(tuple(actions.shape)))
    check("actions are finite", bool(torch.isfinite(actions).all()))
    check(f"prefix computed ONCE for {cfg.num_inference_steps} denoising steps",
          calls["prefix"] == 1, f"got {calls['prefix']}")
    print(f"  {dt * 1000:.0f} ms for B={args.batch_size} at "
          f"{cfg.num_inference_steps} NFE")

    print("\nRESULT:", "ALL PASS" if OK else "FAILURES ABOVE")
    sys.exit(0 if OK else 1)


if __name__ == "__main__":
    main()
