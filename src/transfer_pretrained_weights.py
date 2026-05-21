"""
Initialize an `InterleavedFlowMatchingPolicy` with pretrained weights, then
save as a checkpoint that `train_libero_interleaved.py --resume_from_checkpoint`
can pick up.

Two modes:

  --mode vlm_init         (RELIABLE — recommended)
      Copy SmolVLM2's own frozen text layer weights into the parallel expert
      layers. Expert starts as a 1:1 trainable copy of the frozen VLM stack,
      then learns to specialise for action prediction. Always works because
      our expert was deliberately shaped (RMSNorm + GQA + SwiGLU + same dim)
      to match SmolLM2's decoder layers.

  --mode smolvla          (EXPERIMENTAL — best-effort)
      Load a SmolVLA pretrained checkpoint, attempt to map its expert layers
      onto ours. Prints what was matched, what was skipped, what was shape-
      mismatched. **No guarantee** the key layout matches what we expect —
      run once, inspect the report, iterate. If transfer mostly fails, fall
      back to `vlm_init` (still gives most of the warm-start benefit).

Why bother with `vlm_init` at all when SmolVLA is the "right" answer?
  - SmolVLA expert hidden dim may not match ours (we forced d_model=960 for
    clean joint attention; SmolVLA's expert may be smaller, breaking direct
    transfer). `vlm_init` sidesteps the entire dim issue.
  - SmolVLA's exact module names depend on lerobot version. `vlm_init` only
    depends on SmolVLM2's structure which is stable in `transformers`.

Output: directory containing `model.safetensors` + `config.json` +
preprocessor / postprocessor files. Use it as
  `--resume_from_checkpoint <output_dir>` for training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from safetensors.torch import load_file as load_safetensors

from models.interleaved_flow_matching.interleaved_flow_matching_config import InterleavedFlowMatchingConfig
from models.interleaved_flow_matching.interleaved_flow_matching_policy import InterleavedFlowMatchingPolicy
from models.interleaved_flow_matching.processor_interleaved_flow_matching import make_pre_post_processors


# ---------------------------------------------------------------------------
# VLM warm init — copy SmolLM2's own layer weights into the expert
# ---------------------------------------------------------------------------

def vlm_warm_init(model, init_scale: float = 1.0) -> dict:
    """
    Copy each VLM text layer's weights to the parallel expert layer.

    Shape match is guaranteed because:
      - our `d_model` is forced to VLM hidden in InterleavedFlowMatchingTransformer.__init__
      - ExpertProjections mirrors LlamaDecoderLayer attribute names and dims

    The o_proj and mlp.down_proj weights are scaled by `init_scale` so the
    expert's contribution to the residual stream doesn't immediately double
    the frozen VLM's activations. init_scale=1.0 means full copy (expert
    behaves like VLM on step 0); init_scale=0.1 makes it a small perturbation
    on top of frozen VLM behaviour (closer to the zero-init default).

    Returns a stats dict for logging.
    """
    stats = {"layers": 0, "params_copied": 0}

    vlm_layers = model.text_model.layers
    exp_layers = model.expert_layers
    assert len(vlm_layers) == len(exp_layers), \
        f"VLM has {len(vlm_layers)} layers but expert has {len(exp_layers)}"

    for i, (vlm_layer, exp_layer) in enumerate(zip(vlm_layers, exp_layers)):
        # Attention QKV (cast in case VLM is bfloat16 and expert is fp32)
        exp_layer.q_proj.weight.data.copy_(vlm_layer.self_attn.q_proj.weight.data.to(exp_layer.q_proj.weight.dtype))
        exp_layer.k_proj.weight.data.copy_(vlm_layer.self_attn.k_proj.weight.data.to(exp_layer.k_proj.weight.dtype))
        exp_layer.v_proj.weight.data.copy_(vlm_layer.self_attn.v_proj.weight.data.to(exp_layer.v_proj.weight.dtype))

        # Output projection — scale to control residual blast radius at step 0
        o_w = vlm_layer.self_attn.o_proj.weight.data.to(exp_layer.o_proj.weight.dtype)
        exp_layer.o_proj.weight.data.copy_(o_w * init_scale)

        # Norms (single learnable scale vector)
        exp_layer.input_layernorm.weight.data.copy_(vlm_layer.input_layernorm.weight.data.to(exp_layer.input_layernorm.weight.dtype))
        exp_layer.post_attention_layernorm.weight.data.copy_(vlm_layer.post_attention_layernorm.weight.data.to(exp_layer.post_attention_layernorm.weight.dtype))

        # FFN (SwiGLU)
        exp_layer.mlp.gate_proj.weight.data.copy_(vlm_layer.mlp.gate_proj.weight.data.to(exp_layer.mlp.gate_proj.weight.dtype))
        exp_layer.mlp.up_proj.weight.data.copy_(vlm_layer.mlp.up_proj.weight.data.to(exp_layer.mlp.up_proj.weight.dtype))
        dn_w = vlm_layer.mlp.down_proj.weight.data.to(exp_layer.mlp.down_proj.weight.dtype)
        exp_layer.mlp.down_proj.weight.data.copy_(dn_w * init_scale)

        stats["layers"] += 1
        stats["params_copied"] += sum(
            p.numel() for p in exp_layer.parameters() if p.requires_grad
        )

    return stats


# ---------------------------------------------------------------------------
# SmolVLA best-effort transfer
# ---------------------------------------------------------------------------

# Map our expert sub-module suffix → list of regex patterns that match the
# corresponding SmolVLA key. We try each pattern; first hit wins.
# Add patterns here if the printed key dump shows a different naming convention.
_SMOLVLA_PATTERNS = {
    "q_proj.weight":   [r"expert.*\.layers\.{i}\.self_attn\.q_proj\.weight",
                        r"action_expert.*\.layers\.{i}\.self_attn\.q_proj\.weight",
                        r"model\.layers\.{i}\.self_attn\.q_proj\.weight"],
    "k_proj.weight":   [r"expert.*\.layers\.{i}\.self_attn\.k_proj\.weight",
                        r"action_expert.*\.layers\.{i}\.self_attn\.k_proj\.weight",
                        r"model\.layers\.{i}\.self_attn\.k_proj\.weight"],
    "v_proj.weight":   [r"expert.*\.layers\.{i}\.self_attn\.v_proj\.weight",
                        r"action_expert.*\.layers\.{i}\.self_attn\.v_proj\.weight",
                        r"model\.layers\.{i}\.self_attn\.v_proj\.weight"],
    "o_proj.weight":   [r"expert.*\.layers\.{i}\.self_attn\.o_proj\.weight",
                        r"action_expert.*\.layers\.{i}\.self_attn\.o_proj\.weight",
                        r"model\.layers\.{i}\.self_attn\.o_proj\.weight"],
    "input_layernorm.weight":         [r"expert.*\.layers\.{i}\.input_layernorm\.weight",
                                        r"action_expert.*\.layers\.{i}\.input_layernorm\.weight"],
    "post_attention_layernorm.weight": [r"expert.*\.layers\.{i}\.post_attention_layernorm\.weight",
                                         r"action_expert.*\.layers\.{i}\.post_attention_layernorm\.weight"],
    "mlp.gate_proj.weight": [r"expert.*\.layers\.{i}\.mlp\.gate_proj\.weight",
                              r"action_expert.*\.layers\.{i}\.mlp\.gate_proj\.weight"],
    "mlp.up_proj.weight":   [r"expert.*\.layers\.{i}\.mlp\.up_proj\.weight",
                              r"action_expert.*\.layers\.{i}\.mlp\.up_proj\.weight"],
    "mlp.down_proj.weight": [r"expert.*\.layers\.{i}\.mlp\.down_proj\.weight",
                              r"action_expert.*\.layers\.{i}\.mlp\.down_proj\.weight"],
}


def _find_smolvla_key(smolvla_keys: list[str], layer_idx: int, suffix: str) -> Optional[str]:
    """Try the registered regex patterns for a given (layer, suffix), return first match."""
    import re
    for pat in _SMOLVLA_PATTERNS.get(suffix, []):
        rx = re.compile(pat.format(i=layer_idx))
        for k in smolvla_keys:
            if rx.search(k):
                return k
    return None


def smolvla_transfer(model, smolvla_state: dict[str, torch.Tensor]) -> dict:
    """Best-effort: walk our expert layers and copy whatever shape-matches."""
    stats = {"matched": 0, "shape_mismatch": 0, "unfound": 0, "details": []}
    smolvla_keys = list(smolvla_state.keys())

    # Quick sanity print so we can spot if the keys look totally unlike Llama
    print("\nFirst 30 SmolVLA keys (for naming-convention sanity check):")
    for k in smolvla_keys[:30]:
        print(f"  {k}  shape={tuple(smolvla_state[k].shape)}")
    print(f"  ... ({len(smolvla_keys)} keys total)\n")

    for i, exp_layer in enumerate(model.expert_layers):
        for suffix in _SMOLVLA_PATTERNS.keys():
            src_key = _find_smolvla_key(smolvla_keys, i, suffix)
            target = _resolve_target(exp_layer, suffix)
            if src_key is None:
                stats["unfound"] += 1
                stats["details"].append(f"L{i:02d} {suffix:35s} : NOT FOUND in SmolVLA")
                continue
            src_w = smolvla_state[src_key]
            if target.shape != src_w.shape:
                stats["shape_mismatch"] += 1
                stats["details"].append(
                    f"L{i:02d} {suffix:35s} : shape {tuple(src_w.shape)} != {tuple(target.shape)} (skipped)"
                )
                continue
            target.data.copy_(src_w.to(target.dtype))
            stats["matched"] += 1

    return stats


def _resolve_target(exp_layer, suffix: str) -> torch.Tensor:
    """Walk suffix path on the expert layer to get the parameter."""
    obj = exp_layer
    for part in suffix.split("."):
        obj = getattr(obj, part)
    return obj


def load_smolvla_state(smolvla_id: str, cache_dir: Optional[str] = None) -> dict[str, torch.Tensor]:
    """Download a SmolVLA checkpoint from HF and return its raw state_dict."""
    import huggingface_hub
    path = Path(huggingface_hub.snapshot_download(smolvla_id, cache_dir=cache_dir))
    candidates = list(path.glob("*.safetensors"))
    if not candidates:
        candidates = list(path.glob("**/*.safetensors"))
    if not candidates:
        raise FileNotFoundError(
            f"No .safetensors found under {path}. "
            f"SmolVLA may be stored differently — check the repo: https://huggingface.co/{smolvla_id}"
        )
    state: dict[str, torch.Tensor] = {}
    for ckpt_file in candidates:
        state.update(load_safetensors(str(ckpt_file)))
    print(f"Loaded {len(state)} tensors from {len(candidates)} file(s) in {path}")
    return state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_config(dataset_id: str) -> tuple[InterleavedFlowMatchingConfig, Any, str, str]:
    """Build a config from dataset metadata."""
    md = LeRobotDatasetMetadata(dataset_id, force_cache_sync=True, revision="main")
    features = dataset_to_policy_features(md.features)
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}
    if not output_features:
        raise ValueError("No action features found in dataset.")

    camera_keys = sorted([k for k, ft in input_features.items() if ft.type is FeatureType.VISUAL])
    state_key = next((k for k in ("observation.state", "state") if k in input_features), None)
    if state_key is None:
        raise ValueError(f"No state key in input_features: {list(input_features.keys())}")
    action_key = next(iter(output_features.keys()))

    cfg = InterleavedFlowMatchingConfig(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=2,
        horizon=64,
        n_action_steps=64,
        state_dim=input_features[state_key].shape[-1],
        action_dim=output_features[action_key].shape[-1],
        num_vlm_layers=16,
        num_cameras=len(camera_keys),
        cameras_for_vision_state_concat=camera_keys,
        action_dim_weights=[1.0] * output_features[action_key].shape[-1],
        pos_decay_lambda=0.0,
        num_latent_tokens=8,
        vlm_attends_to_expert=True,
        vision_lora_num_layers=0,
    )
    return cfg, md, state_key, action_key


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["vlm_init", "smolvla"], default="vlm_init",
                        help="vlm_init: reliable warm start from SmolVLM2 text layers. "
                             "smolvla: best-effort transfer from a SmolVLA checkpoint.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save the initialised checkpoint. "
                             "Use this with train_libero_interleaved.py --resume_from_checkpoint")
    parser.add_argument("--dataset_id", type=str, default="lerobot/libero",
                        help="Dataset for metadata (state_dim, action_dim, image features).")
    parser.add_argument("--smolvla_id", type=str, default="lerobot/smolvla_base",
                        help="HF model id for SmolVLA (only used with --mode smolvla).")
    parser.add_argument("--init_scale", type=float, default=1.0,
                        help="(vlm_init only) Scale factor for o_proj and mlp.down_proj. "
                             "1.0 = full copy (expert as second VLM); 0.1 = small perturbation.")
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Mode: {args.mode}")
    print(f"Dataset for config: {args.dataset_id}")
    print(f"Output dir: {args.output_dir}\n")

    # ---- Build config + instantiate model ----
    cfg, dataset_metadata, state_key, action_key = build_config(args.dataset_id)
    print(f"State key: {state_key}  (dim={cfg.state_dim})")
    print(f"Action key: {action_key}  (dim={cfg.action_dim})")

    print("\nInstantiating InterleavedFlowMatchingPolicy (loads frozen SmolVLM2)...")
    policy = InterleavedFlowMatchingPolicy(cfg)

    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in policy.parameters() if not p.requires_grad)
    print(f"Trainable: {n_train:,}   Frozen: {n_frozen:,}")

    # ---- Apply init ----
    if args.mode == "vlm_init":
        print(f"\n[vlm_init] Copying VLM text-layer weights to expert layers (init_scale={args.init_scale})...")
        stats = vlm_warm_init(policy.model, init_scale=args.init_scale)
        print(f"[vlm_init] Done. Initialised {stats['layers']} expert layers, "
              f"{stats['params_copied']:,} trainable params overwritten.")
        print("[vlm_init] Note: latent_embs / robot_visual_encoder / action_in_proj / "
              "action_out_proj / state_encoder remain at their constructor defaults.")
    elif args.mode == "smolvla":
        print(f"\n[smolvla] Loading SmolVLA checkpoint from {args.smolvla_id}...")
        smolvla_state = load_smolvla_state(args.smolvla_id, cache_dir=args.cache_dir)
        print("[smolvla] Attempting expert layer transfer...")
        stats = smolvla_transfer(policy.model, smolvla_state)
        print("\n[smolvla] Transfer summary:")
        print(f"  Matched:        {stats['matched']}")
        print(f"  Shape mismatch: {stats['shape_mismatch']}")
        print(f"  Not found:      {stats['unfound']}")
        if stats["matched"] == 0:
            print("\n[smolvla] WARNING: nothing matched. Either SmolVLA's key layout doesn't follow "
                  "the patterns in _SMOLVLA_PATTERNS, or shapes mismatch. Inspect the printed key "
                  "dump above and either: (a) extend _SMOLVLA_PATTERNS with the correct regex, or "
                  "(b) re-run with --mode vlm_init.")
        else:
            print(f"\n[smolvla] Per-layer details (first 20):")
            for line in stats["details"][:20]:
                print(f"  {line}")
            if len(stats["details"]) > 20:
                print(f"  ... ({len(stats['details']) - 20} more)")

    # ---- Save in resume-compatible format ----
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to {out} ...")
    # Mark step 0 so the resume scheduler doesn't fast-forward past warmup.
    policy.config.training_step = 0
    policy.config.training_epoch = 0
    policy.config.optimizer_lr = cfg.optimizer_lr
    policy.config.current_lr = cfg.optimizer_lr
    policy.config.training_steps_total = 0  # unset → training script falls back to its own default

    policy.save_pretrained(out)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)
    preprocessor.save_pretrained(out)
    postprocessor.save_pretrained(out)
    print("Done.")
    print(f"\nNext step:")
    print(f"  python src/train_libero_interleaved.py \\")
    print(f"      --output_dir ./outputs/<run_name> \\")
    print(f"      --dataset_id {args.dataset_id} \\")
    print(f"      --resume_from_checkpoint {out}\n")


if __name__ == "__main__":
    main()
