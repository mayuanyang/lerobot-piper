"""
Training script for WiltechsMoE (Qwen3-VL-4B encoder + N-expert MoE decoder flow matching).

Mirrors `train_wiltechs_vla.py`'s data path: train on ONE OR MORE explicit LeRobot v3
datasets passed via `--dataset_id`. Multiple datasets are concatenated and
assumed HOMOGENEOUS (same robot / cameras / state+action dims / fps) — e.g.
several piper sets — and their normalization stats are aggregated. There is NO
community-hub discovery, version filtering, allowlist/denylist, or canonical-
schema projection here; the model's input/output features come straight from the
dataset schema. For mixed-robot community pretraining use `train_community.py`
(the canonical multi-robot DatasetAdapter path) instead.

Usage:
    # Single dataset
    python src/train_wiltechs_moe.py \
        --output_dir outputs/train/wiltechs_moe_piper \
        --dataset_id ISdept/piper_arm \
        --batch_size 16 \
        --training_steps 300000

    # Concatenate several homogeneous datasets
    python src/train_wiltechs_moe.py \
        --output_dir outputs/train/wiltechs_moe_piper \
        --dataset_id ISdept/piper_arm ISdept/piper_arm_v2 \
        --batch_size 16

    # Resume from a checkpoint
    python src/train_wiltechs_moe.py \
        --output_dir outputs/train/wiltechs_moe_piper \
        --dataset_id ISdept/piper_arm \
        --resume_from_checkpoint outputs/train/wiltechs_moe_piper/checkpoint-50000
"""

from __future__ import annotations

import json
import math
import argparse
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from tqdm import tqdm
import huggingface_hub
from safetensors.torch import load_file as load_safetensors

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.datasets.compute_stats import aggregate_stats

from models.wiltechs_moe.wiltechs_moe_config import WiltechsMoEConfig
from models.wiltechs_moe.wiltechs_moe_policy import WiltechsMoEPolicy
from models.wiltechs_moe.processor_wiltechs_moe import make_pre_post_processors
from models.wiltechs_vla.wiltechs_vla_model import (
    preprocess_camera_to_pixels, vlm_pixels_key, vlm_grid_key, _VLM_PIX_PREFIX,
    format_xattn,
)

from torchvision.transforms import v2
from transformers import get_cosine_schedule_with_warmup

# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True


# ---------------------------------------------------------------------------
# Optimizer factory — optional 8-bit Adam (bitsandbytes) to cut optimizer
# state memory ~4× (fp32 m+v → int8 m+v). The big MoE expert stack dominates GPU
# memory via its Adam state, so this is the main lever on small GPUs.
# ---------------------------------------------------------------------------
def make_optimizer(params, lr, weight_decay, use_8bit: bool):
    if use_8bit:
        try:
            import bitsandbytes as bnb
            print("Using 8-bit Adam (bitsandbytes) — optimizer state in int8.")
            return bnb.optim.Adam8bit(params, lr=lr, weight_decay=weight_decay)
        except ImportError:
            print("[WARN] --use_8bit_adam set but bitsandbytes not installed; "
                  "falling back to fp32 Adam. `pip install bitsandbytes` to enable.")
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
def get_augmentations(translate: float = 0.0, scale_jitter: float = 0.0):
    """Photometric augmentation always; geometric augmentation opt-in.

    Geometric augmentation moves the objects in the frame but NOT the action
    label, so it teaches "position does not change the action" -- the exact
    invariance a spatial-referring task must not have. On a 256px LIBERO frame
    the old translate=0.03 was +-7.7px against a ~19px ramekin-to-bowl
    separation: 40% of the distance the policy is being asked to resolve. The
    LIBERO camera is fixed and eval uses the same viewpoint as training, so
    there is no viewpoint robustness to buy in exchange.

    Colour/blur do not move anything and stay on.

    Not measured against a controlled A/B -- the mechanism is clear but the
    magnitude of the effect is not. Pass --image_aug_translate 0.03 to restore
    the previous behaviour.
    """
    tfs = []
    if translate > 0 or scale_jitter > 0:
        tfs.append(v2.RandomAffine(
            degrees=0,
            translate=(translate, translate) if translate > 0 else (0.0, 0.0),
            scale=(1 - scale_jitter, 1 + scale_jitter) if scale_jitter > 0 else None,
            fill=0))
    tfs.append(v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08))
    tfs.append(v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))], p=0.3))
    return v2.Compose(tfs)


def apply_image_augmentations(batch: dict, camera_keys: list[str], transform) -> dict:
    present_keys = [k for k in camera_keys if k in batch and isinstance(batch[k], torch.Tensor)]
    if not present_keys:
        return batch
    B = batch[present_keys[0]].shape[0]
    for b in range(B):
        sample_img = batch[present_keys[0]][b]
        has_time_dim = sample_img.dim() == 4  # (T, C, H, W)
        if has_time_dim:
            T = sample_img.shape[0]
            stacked = torch.cat([batch[k][b] for k in present_keys], dim=0)
            stacked_aug = transform(stacked)
            for i, k in enumerate(present_keys):
                batch[k][b] = stacked_aug[i * T : (i + 1) * T]
        else:
            stacked = torch.stack([batch[k][b] for k in present_keys], dim=0)
            stacked_aug = transform(stacked)
            for i, k in enumerate(present_keys):
                batch[k][b] = stacked_aug[i]
    return batch


def apply_joint_augmentations(batch: dict, state_key: str) -> dict:
    if torch.rand(1).item() > 0.5:
        if state_key in batch:
            noise = torch.randn_like(batch[state_key]) * 0.02
            batch[state_key] = batch[state_key] + noise
    return batch


# ---------------------------------------------------------------------------
# Move the Qwen image preprocessing (and image augmentation) into the DataLoader
# workers so it runs in parallel and overlaps with GPU compute, instead of on
# the critical path inside _encode_images every step.
# ---------------------------------------------------------------------------
class VLMImagePreprocDataset(torch.utils.data.Dataset):
    """Wraps a dataset; per sample it (optionally) augments the camera images
    with cross-camera-consistent transforms, then runs the Qwen image_processor
    to produce pixel_values / grid_thw, stored under vlm_pixels_key()/
    vlm_grid_key(). The (augmented) raw camera tensors are kept too — the robot
    CNN still consumes them. Assumes a uniform camera resolution (true for a
    single homogeneous dataset) so pixel_values collate cleanly to (B, P, dim)."""

    def __init__(self, dataset, image_processor, camera_keys, augment=None,
                 cam_target_sizes=None):
        self.dataset = dataset
        self.image_processor = image_processor
        self.camera_keys = list(camera_keys)
        self.augment = augment  # torchvision transform, or None (e.g. eval)
        # {cam_key: square input side length}; 0/missing = processor default.
        # Must match the model's cam_target_size() or the two paths would build
        # different grids depending on --preprocess_in_workers.
        self.cam_target_sizes = dict(cam_target_sizes or {})

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        present = [k for k in self.camera_keys
                   if k in sample and isinstance(sample[k], torch.Tensor)]
        if not present:
            return sample

        imgs3, had_t = [], []
        for k in present:
            v = sample[k]
            had_t.append(v.dim() == 4)
            imgs3.append(v[0] if v.dim() == 4 else v)

        if self.augment is not None:
            stacked = self.augment(torch.stack(imgs3, dim=0))
            imgs3 = [stacked[i] for i in range(len(present))]
            for i, k in enumerate(present):
                sample[k] = imgs3[i].unsqueeze(0) if had_t[i] else imgs3[i]

        for i, k in enumerate(present):
            pv, thw = preprocess_camera_to_pixels(
                self.image_processor, imgs3[i],
                target_size=self.cam_target_sizes.get(k, 0))
            sample[vlm_pixels_key(k)] = pv         # (P, dim)
            sample[vlm_grid_key(k)] = thw[0]       # (3,)
        return sample


# ---------------------------------------------------------------------------
# Gradient analysis
# ---------------------------------------------------------------------------
def _log_gradient_analysis(policy, step: int) -> None:
    print(f"\n--- Gradient Analysis at Step {step} ---")

    def _grad_stats(prefix: str):
        total, g_norm_sq, count = 0.0, 0.0, 0
        for name, param in policy.model.named_parameters():
            if param.requires_grad and prefix in name and param.grad is not None:
                total += param.grad.abs().mean().item() * param.numel()
                g_norm_sq += param.grad.norm().item() ** 2
                count += param.numel()
        if count == 0:
            return None, None, 0
        return (total / count, (g_norm_sq ** 0.5) / (count ** 0.5), count)

    for label, prefix in [
        ("Robot CNN",      "robot_visual_encoder"),
        ("State Enc",      "state_encoder"),
        ("Sink Token",     "sink_token"),
        ("Action Pos Emb", "action_pos_emb"),
        ("Time Embedder",  "time_embedder"),
        ("Experts",        "experts"),
        ("Router",         "router"),
        ("Action In/Out",  "action_"),
        ("Final Norm",     "final_norm"),
        ("Thought QFormer", "thought_qformer"),
    ]:
        grad, rms_pp, n = _grad_stats(prefix)
        if grad is not None:
            print(f"  {label:14s} - Avg Abs Grad: {grad:.6f}   RMS/param: {rms_pp:.2e} "
                  f"({n} params)")
        else:
            print(f"  {label:14s} - no grad")

    # Robot CNN grid positional encoding. The gate is a single scalar starting
    # at exactly 0, so "pos share" is 0.0% at init by construction and any
    # nonzero value is the model actively choosing to use grid position. If it
    # stays flat near zero while --robot_cnn_fine_tokens is raised, the extra
    # tokens are resolution the DiT cannot localise and the sequence cost is
    # being paid for nothing.
    _gate = getattr(policy.model, "robot_pos_gate", None)
    if _gate is not None:
        _g = float(_gate.detach())
        _gg = float(_gate.grad.abs().mean()) if _gate.grad is not None else float("nan")
        _prms = getattr(policy.model, "_last_robot_pos_rms", 0.0)
        _trms = getattr(policy.model, "_last_robot_tok_rms", 0.0)
        _share = (_prms / _trms * 100.0) if _trms else 0.0
        print(f"  Robot pos gate  : gate={_g:+.5f}  grad={_gg:.2e}   "
              f"pos RMS {_prms:.4f} / token RMS {_trms:.4f} = {_share:.1f}% of magnitude")

    # MoE router usage — show expert load balancing
    usage = getattr(policy.model, "_last_router_usage", None)
    if usage is not None:
        usage_cpu = usage.detach().cpu()
        cells = "  ".join(f"E{i}={v*100:5.1f}%" for i, v in enumerate(usage_cpu.tolist()))
        cv_sq = (usage_cpu.std() / usage_cpu.mean().clamp(min=1e-8)).pow(2).item()
        print(f"  Router usage    : {cells}    CV²={cv_sq:.4f}")
        # Per-sample shape. CV² is a batch mean and reads the same whether the
        # router specialises per sample or has collapsed to uniform-for-everything
        # (which satisfies the balance penalty for free and makes the MoE a fixed
        # average). max_w at 1/E and entropy at ln(E) is that collapse.
        #
        # Both come from the router's pre-noise weights and from the main
        # forward only (not the contrastive negative), so they describe
        # inference-time routing and the uniform references below are the right
        # comparison. See FINDINGS.md -- reading them off the noisy logits, as
        # this did before 2026-08-04, put the collapse floor at 0.388/1.301.
        mw = getattr(policy.model, "_last_router_max_w", None)
        ent = getattr(policy.model, "_last_router_entropy", None)
        if mw is not None and ent is not None:
            E = int(usage_cpu.numel())
            print(f"  Router per-samp : max_w={mw:.3f} (uniform={1/E:.3f})   "
                  f"entropy={ent:.3f} (uniform={math.log(E):.3f})")

    # Thought Q-Former residual gates — the model's OWN "how much do I use the
    # thought tokens" knob (init 0.1 each). Four scalars, no batch noise: a far
    # lower-variance signal than the gradient RMS above, which swings 2-3x
    # between adjacent readings. Growing => the DiT is leaning on the thoughts;
    # pinned near 0.100 after thousands of steps => it is ignoring them, and the
    # module is dead weight regardless of what its gradient looks like.
    qf = getattr(policy.model, "thought_qformer", None)
    gates = getattr(qf, "gates", None) if qf is not None else None
    if gates is not None:
        cells = "  ".join(f"L{i}[ca={g[0].item():+.3f} ffn={g[1].item():+.3f}]"
                          for i, g in enumerate(gates))
        print(f"  Thought gates   : {cells}   (init +0.100)")
    t_rms = getattr(policy.model, "_last_thought_rms", None)
    a_rms = getattr(policy.model, "_last_action_emb_rms", None)
    if t_rms is not None:
        ratio = f"   ratio: {t_rms / a_rms:.3f}" if a_rms else ""
        print(f"  Thought tok RMS : {t_rms:.4f}   action_emb RMS: {a_rms:.4f}{ratio}"
              if a_rms else f"  Thought tok RMS : {t_rms:.4f}")
        # Does that magnitude carry information, or is it a learned constant?
        # The Q-Former output is `queries` (a constant) plus gated corrections,
        # so as the gates decay it collapses toward a big input-independent bias
        # prepended to every expert sequence -- which the RMS line above cannot
        # distinguish from a working thought pathway.
        c_rms = getattr(policy.model, "_last_thought_const_rms", None)
        v_rms = getattr(policy.model, "_last_thought_vary_rms", None)
        q_rms = getattr(policy.model, "_last_thought_query_rms", None)
        B_th = getattr(policy.model, "_last_thought_batch", None)
        if c_rms is not None and v_rms is not None and t_rms:
            # Ceiling is sqrt(1 - 1/B), not 1.0: the batch mean of B independent
            # samples absorbs a 1/sqrt(B) share even when nothing is constant.
            ceil = math.sqrt(max(0.0, 1.0 - 1.0 / B_th)) if B_th else 1.0
            q_txt = f"   queries={q_rms:.4f}" if q_rms is not None else ""
            print(f"  Thought input-dep: vary/total={v_rms / t_rms:.3f} "
                  f"(0.0=learned constant, {ceil:.3f}=fully input-dependent at B={B_th})"
                  f"   const={c_rms:.4f} vary={v_rms:.4f}{q_txt}")

    comps = getattr(policy.model, "_last_loss_components", None)
    if comps is not None:
        main_v = comps.get("main", float("nan"))
        contr_v = comps.get("contrastive", float("nan"))
        bal_v = comps.get("balance", float("nan"))
        cw = getattr(policy.model.config, "contrastive_loss_weight", 0.0)
        bw = getattr(policy.model.config, "router_balance_weight", 0.0)
        parts = [f"main: {main_v:.4f}"]
        if cw > 0.0:
            parts.append(f"contrastive: {contr_v:.4f} (w={cw})")
        if bw > 0.0:
            parts.append(f"balance: {bal_v:.4f} (w={bw})")
        print(f"  Loss components : {'  '.join(parts)}")

    x_stats = getattr(policy.model, "_last_cross_attention_stats", None)
    if x_stats:
        # Router-weighted across experts — the figure comparable with
        # WiltechsVLA's single-stack reading.
        print("  Action→ x-attn  : " + format_xattn(x_stats)
              + "    (router-weighted over experts)")
        per_e = x_stats.get("_per_expert") or []
        if len(per_e) > 1:
            # Each expert reads a different VLM depth band, so a spread here
            # means the bands are being used differently — invisible in the
            # weighted average above.
            labels = x_stats.get("_labels") or [f"E{i}" for i in range(len(per_e))]
            cells = "  ".join(f"{lab}={lang * 100:4.1f}%"
                              for lab, (_v, lang) in zip(labels, per_e))
            print(f"  x-attn lang/exp : {cells}    (language %, shallow→deep)")

    print("--- End Gradient Analysis ---\n")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train(
    output_dir: str,
    dataset_id="ISdept/piper_arm",
    resume_from_checkpoint: Optional[str] = None,
    reset_training_state: bool = False,
    reset_params: Optional[list] = None,
    batch_size: int = 16,
    training_steps: int = 300000,
    gradient_checkpointing: bool = False,
    num_experts: int = 4,
    expert_num_layers: int = 4,
    dit_hidden_size: int = 1280,
    use_8bit_adam: bool = False,
    val_episodes: float = 0.0,
    val_every: int = 500,
    val_seed: int = 42,
    val_max_batches: int = 8,
    max_episode_index: Optional[int] = None,
    lock_joint_index: Optional[int] = None,
    contrastive_loss_weight: float = 0.1,
    contrastive_margin: float = 0.05,
    contrastive_hard_negatives: bool = False,
    vision_kv_dropout_prob: float = 0.0,
    use_chat_template: bool = False,
    chat_directive: str = "",
    use_descriptive_objects: bool = False,
    text_first: bool = True,
    vision_input_size: int = 0,
    vision_hires_cameras: Optional[list] = None,
    n_action_steps: int = 4,
    vlm_model_id: str = "",
    robot_encoder_tokens: int = 16,
    robot_encoder_input_size: int = 224,
    robot_cnn_fine_cameras: Optional[list] = None,
    robot_cnn_fine_tokens: int = 0,
    noise_temporal_correlation: float = 0.0,
    vision_dropout_prob: float = 0.3,
    vision_dropout_start: float = -1.0,
    vision_dropout_anneal_steps: int = 0,
    robot_cnn_cameras: Optional[list] = None,
    robot_cnn_wrist_only: bool = False,
    use_robot_cnn: bool = True,
    preprocess_in_workers: bool = False,
    router_temperature: float = 1.0,
    router_balance_weight: float = 0.1,
    router_top_k: int = 0,
    vlm_capture_layers: Optional[list] = None,
    num_thought_tokens: int = 8,
    thought_qformer_layers: int = 2,
    thought_vlm_layer_idx: int = -1,
    thought_consistency_weight: float = 0.0,
    image_aug_translate: float = 0.0,
    image_aug_scale: float = 0.0,
    loss_exec_steps: int = 0,
    future_steps_weight: float = 0.3,
):
    """Train WiltechsMoE on one or more HOMOGENEOUS LeRobot datasets.

    `dataset_id` may be a single id or a list. Multiple datasets are concatenated
    and must share the same robot / cameras / state+action dims / fps; their
    normalization stats are aggregated. For mixed-robot data use
    `train_community.py` instead.
    """
    dataset_ids = [dataset_id] if isinstance(dataset_id, str) else list(dataset_id)
    if not dataset_ids:
        raise ValueError("At least one dataset_id is required.")

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    progress_update_freq = 200
    checkpoint_freq = 2000
    image_transforms = get_augmentations(image_aug_translate, image_aug_scale)
    if image_aug_translate or image_aug_scale:
        print(f"Image aug: GEOMETRIC ON (translate={image_aug_translate}, "
              f"scale=+-{image_aug_scale}) + colour/blur. Note the action label is "
              f"NOT transformed with the image.")
    else:
        print("Image aug: colour/blur only (no geometric). "
              "Pass --image_aug_translate/--image_aug_scale to re-enable.")

    # ── Load metadata for all datasets; first is the schema reference ────
    metas = {did: LeRobotDatasetMetadata(did, force_cache_sync=True, revision="main")
             for did in dataset_ids}
    ref_meta = metas[dataset_ids[0]]
    features = dataset_to_policy_features(ref_meta.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    if len(output_features) == 0:
        raise ValueError("No output features (actions) found! Check your dataset schema.")

    print('input_features:', input_features)
    print('output_features:', output_features)

    camera_keys = sorted([key for key, ft in input_features.items() if ft.type is FeatureType.VISUAL])
    state_dim = input_features["observation.state"].shape[-1] if "observation.state" in input_features else 7
    action_dim = next(iter(output_features.values())).shape[-1]
    print(f"Detected cameras ({len(camera_keys)}): {camera_keys}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Validate eagerly: a mistyped key would otherwise silently resolve to
    # "no camera gets the higher resolution" and the run would look normal.
    if vision_hires_cameras:
        missing = [c for c in vision_hires_cameras if c not in camera_keys]
        if missing:
            raise ValueError(
                f"--vision_hires_cameras {missing} not in detected cameras {camera_keys}")
    if vision_input_size:
        targeted = list(vision_hires_cameras) if vision_hires_cameras else list(camera_keys)
        print(f"Vision input size {vision_input_size}px -> "
              f"{(vision_input_size // 32) ** 2} tokens/frame for {targeted}; "
              f"other cameras keep the processor default. "
              f"Confirm with the '[wiltechs_moe] vision grid ...' lines below.")

    # ── Sequence-shape constants ─────────────────────────────────────────
    # Defined here rather than with the other training parameters below: the
    # RobotCNN report needs `horizon` to print the DiT sequence length, and a
    # name assigned anywhere in a function is local to the WHOLE function, so
    # reading it before the assignment is an UnboundLocalError, not a fallback.
    obs = 2
    horizon = 64
    # Inference-only: how many steps the action queue pops before replanning.
    # It no longer touches the loss (see compute_loss), so it is safe to set to
    # whatever the eval actually uses. Stored in the checkpoint config, and an
    # eval that forgets to override it inherits THIS value -- at 10Hz, 64 means
    # 6.4s of open-loop execution from a single observation, which is fatal for
    # grasp precision. Default to the value evals really run at.
    n_action_steps = int(n_action_steps)

    # ── Resolve the RobotCNN camera list ────────────────────────────────
    robot_cnn_camera_keys: list[str] = []
    if not use_robot_cnn:
        # No parameter SHAPE changes when this is off -- robot_visual_encoder
        # becomes None, its keys are simply absent, and action_start_idx is
        # derived from the built sequence rather than assumed. So a checkpoint
        # trained WITH the CNN resumes cleanly without it, which makes this a
        # cheap controlled test rather than a from-scratch run.
        print("RobotCNN: DISABLED (--no_robot_cnn). The raw-pixel pathway that "
              "bypasses the VLM is gone; scene information can now only reach the "
              "experts through the VLM KV or the thought tokens. Watch the "
              "'Action-> x-attn' vision share: if it climbs, the CNN was the "
              "shortcut that made VLM vision redundant.")
    elif robot_cnn_cameras:
        missing = [c for c in robot_cnn_cameras if c not in camera_keys]
        if missing:
            raise ValueError(
                f"--robot_cnn_cameras {missing} not in detected cameras {camera_keys}")
        robot_cnn_camera_keys = list(robot_cnn_cameras)
    elif robot_cnn_wrist_only:
        _WRIST_HINTS = ("image2", "wrist", "gripper", "eye_in_hand", "hand")
        robot_cnn_camera_keys = [
            c for c in camera_keys
            if any(h in c.rsplit(".", 1)[-1].lower() for h in _WRIST_HINTS)]
        if not robot_cnn_camera_keys:
            raise ValueError(
                f"--robot_cnn_wrist_only: no wrist-like camera among {camera_keys}. "
                f"Pass --robot_cnn_cameras <key> explicitly.")
    # Cameras that get the DENSE token grid. Explicit list wins; otherwise
    # --robot_cnn_fine_tokens auto-targets the wrist view, which is where
    # contact geometry lives.
    fine_cam_keys: list = []
    if use_robot_cnn and robot_cnn_fine_tokens > 0:
        _pool = robot_cnn_camera_keys or camera_keys
        if robot_cnn_fine_cameras:
            missing = [c for c in robot_cnn_fine_cameras if c not in _pool]
            if missing:
                raise ValueError(
                    f"--robot_cnn_fine_cameras {missing} not among the RobotCNN's "
                    f"cameras {_pool}")
            fine_cam_keys = list(robot_cnn_fine_cameras)
        else:
            _WRIST_HINTS = ("image2", "wrist", "gripper", "eye_in_hand", "hand")
            fine_cam_keys = [
                c for c in _pool
                if any(h in c.rsplit(".", 1)[-1].lower() for h in _WRIST_HINTS)]
            if not fine_cam_keys:
                raise ValueError(
                    f"--robot_cnn_fine_tokens: no wrist-like camera among {_pool}. "
                    f"Pass --robot_cnn_fine_cameras <key> explicitly.")

    if not use_robot_cnn:
        pass  # already reported above
    else:
        if robot_cnn_camera_keys:
            print(f"RobotCNN cameras (wrist-specialized): {robot_cnn_camera_keys}  "
                  f"(VLM still sees all {len(camera_keys)})")
        else:
            print(f"RobotCNN cameras: ALL {camera_keys} (legacy: CNN re-encodes VLM views)")
        # Report the actual spatial granularity, in native px of the source
        # frame. This is the number that was silently 64 while the frozen VLM
        # ran at 32 -- print it so a coarse setting cannot hide again.
        _fmap = robot_encoder_input_size // 16  # ResNet-18 through layer3: stride 16
        _cams = robot_cnn_camera_keys or camera_keys
        _total = 0
        for _c in _cams:
            _n = robot_cnn_fine_tokens if _c in fine_cam_keys else robot_encoder_tokens
            _side = int(_n ** 0.5)
            _total += _n
            print(f"  {_c}: {_side}x{_side}={_n} tok  "
                  f"({256 / _side:.1f} native px/token, feature map {_fmap}x{_fmap})"
                  + ("  <- FINE" if _c in fine_cam_keys else ""))
            if _side > _fmap:
                print(f"    WARNING: {_side}x{_side} exceeds the {_fmap}x{_fmap} feature "
                      f"map -- adaptive pooling will UPSAMPLE and add no information. "
                      f"Raise --robot_encoder_input_size to {_side * 16} or lower the tokens.")
        print(f"  RobotCNN total: {_total} tokens  "
              f"(DiT sequence = 2 + {_total} + {num_thought_tokens} + {horizon} "
              f"= {2 + _total + num_thought_tokens + horizon})")

    # ── Validate the other datasets share the same schema ────────────────
    for did in dataset_ids[1:]:
        f = dataset_to_policy_features(metas[did].features)
        out_f = {k: ft for k, ft in f.items() if ft.type is FeatureType.ACTION}
        in_f = {k: ft for k, ft in f.items() if k not in out_f}
        cks = sorted(k for k, ft in in_f.items() if ft.type is FeatureType.VISUAL)
        sd = in_f["observation.state"].shape[-1] if "observation.state" in in_f else 7
        ad = next(iter(out_f.values())).shape[-1]
        if cks != camera_keys or sd != state_dim or ad != action_dim:
            raise ValueError(
                f"Dataset '{did}' schema differs from '{dataset_ids[0]}':\n"
                f"  cameras {cks} vs {camera_keys}\n"
                f"  state_dim {sd} vs {state_dim}, action_dim {ad} vs {action_dim}\n"
                f"train_wiltechs_moe.py concatenation requires a homogeneous schema. "
                f"For mixed robots use train_community.py."
            )

    # ── Aggregate normalization stats across datasets ────────────────────
    if len(dataset_ids) == 1:
        combined_stats = ref_meta.stats
    else:
        combined_stats = aggregate_stats([metas[did].stats for did in dataset_ids])
        print(f"Aggregated normalization stats across {len(dataset_ids)} datasets.")

    # ── Training parameters ──────────────────────────────────────────────
    # obs / horizon / n_action_steps are set above, before the RobotCNN report.

    # Report the temporal loss weighting explicitly. This block existed and was
    # silently doing nothing: the loss read n_action_steps directly, and with
    # n_action_steps == horizon the `pos_w[n_exec:]` slice was empty, so
    # future_steps_weight applied to no timestep at all. Printing the resulting
    # share of the first few steps makes that visible instead of inferable.
    _n_exec = min(max(1, loss_exec_steps or horizon), horizon)
    _total_w = _n_exec + (horizon - _n_exec) * future_steps_weight
    print(f"Loss horizon weighting: steps 0..{_n_exec - 1} at 1.0, "
          f"{_n_exec}..{horizon - 1} at {future_steps_weight} "
          f"(total weight {_total_w:.1f})")
    for _k in (4, 8):
        _share = min(_k, _n_exec) + max(0, _k - _n_exec) * future_steps_weight
        print(f"  first {_k:>2} steps carry {_share / _total_w * 100:5.1f}% of the "
              f"horizon weight")
    if _n_exec >= horizon:
        print("  NOTE: loss_exec_steps >= horizon, so future_steps_weight is inert "
              "and every step is weighted equally. Pass --loss_exec_steps to "
              "concentrate the gradient on the part an eval actually executes.")

    action_dim_weights = [1.0] * action_dim
    if lock_joint_index is not None and 0 <= lock_joint_index < action_dim:
        action_dim_weights[lock_joint_index] = 0.0
        print(f"Locking action dim {lock_joint_index} (weight=0); "
              f"action_dim_weights={action_dim_weights}")
    else:
        print(f"All {action_dim} action dims weighted equally; "
              f"action_dim_weights={action_dim_weights}")

    # ── Build config ─────────────────────────────────────────────────────
    cfg = WiltechsMoEConfig(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=obs,
        horizon=horizon,
        n_action_steps=n_action_steps,
        state_dim=state_dim,
        action_dim=action_dim,
        num_experts=num_experts,
        expert_num_layers=expert_num_layers,
        dit_hidden_size=dit_hidden_size,
        vlm_model_id=vlm_model_id,
        num_cameras=len(camera_keys),
        cameras_for_vision_state_concat=camera_keys,
        use_robot_cnn=use_robot_cnn,
        robot_cnn_cameras=robot_cnn_camera_keys,
        action_dim_weights=action_dim_weights,
        pos_decay_lambda=0.0,
        loss_exec_steps=loss_exec_steps,
        future_steps_weight=future_steps_weight,
        contrastive_loss_weight=contrastive_loss_weight,
        contrastive_margin=contrastive_margin,
        contrastive_hard_negatives=contrastive_hard_negatives,
        vision_kv_dropout_prob=vision_kv_dropout_prob,
        vision_dropout_prob=vision_dropout_prob,
        use_chat_template=use_chat_template,
        chat_directive=chat_directive,
        use_descriptive_objects=use_descriptive_objects,
        text_first=text_first,
        vision_input_size=vision_input_size,
        vision_hires_cameras=list(vision_hires_cameras or []),
        robot_encoder_tokens=robot_encoder_tokens,
        robot_encoder_input_size=robot_encoder_input_size,
        robot_cnn_fine_cameras=fine_cam_keys,
        robot_cnn_fine_tokens=robot_cnn_fine_tokens,
        noise_temporal_correlation=noise_temporal_correlation,
        router_temperature=router_temperature,
        router_balance_weight=router_balance_weight,
        router_top_k=router_top_k,
        vlm_capture_layers=vlm_capture_layers if vlm_capture_layers else [],
        num_thought_tokens=num_thought_tokens,
        thought_qformer_layers=thought_qformer_layers,
        thought_vlm_layer_idx=thought_vlm_layer_idx,
        thought_consistency_weight=thought_consistency_weight,
    )

    # ── Model setup ──────────────────────────────────────────────────────
    if resume_from_checkpoint is not None:
        print(f"\nResuming training from checkpoint: {resume_from_checkpoint}")
        policy = WiltechsMoEPolicy(cfg)
        ckpt_path = Path(resume_from_checkpoint)
        local_ckpt = ckpt_path if ckpt_path.exists() else Path(
            huggingface_hub.snapshot_download(resume_from_checkpoint)
        )
        model_file = local_ckpt / "model.safetensors"
        if not model_file.exists():
            candidates = list(local_ckpt.glob("*.safetensors"))
            if not candidates:
                raise FileNotFoundError(f"No .safetensors found in {local_ckpt}")
            model_file = candidates[0]

        step, epoch = 0, 0
        saved_cfg_json: dict = {}
        for cfg_name in ("config.json", "pretrained_config.json"):
            cfg_file = local_ckpt / cfg_name
            if cfg_file.exists():
                with open(cfg_file) as f:
                    saved_cfg_json = json.load(f)
                break
        if reset_training_state:
            # FINETUNE, not continuation. Without this the counters below come
            # back from the checkpoint, and a checkpoint from a COMPLETED run
            # carries step == training_steps_total -- the training loop's
            # `if step >= training_steps` then fires on the very first batch and
            # the "finetune" silently trains for zero steps.
            print(f"--reset_training_state: starting a NEW schedule at step 0 for "
                  f"{training_steps} steps (checkpoint reported step="
                  f"{saved_cfg_json.get('training_step', 0)}). Optimizer state and "
                  f"the saved LR schedule are discarded; weights are kept.")
        else:
            if saved_cfg_json:
                step = saved_cfg_json.get("training_step", 0)
                epoch = saved_cfg_json.get("training_epoch", 0)
                saved_total = saved_cfg_json.get("training_steps_total", 0)
                if saved_total > 0:
                    training_steps = saved_total
                print(f"Read config from checkpoint: step={step}, epoch={epoch}, "
                      f"training_steps_total={training_steps}")
            if step == 0 and local_ckpt.name.startswith("checkpoint-"):
                step = int(local_ckpt.name.split("-")[1])
            if step >= training_steps:
                raise RuntimeError(
                    f"Checkpoint is at step {step} of {training_steps}: the training loop "
                    f"would exit before the first optimizer step. This is what a FINISHED "
                    f"pretraining run looks like -- pass --reset_training_state to start a "
                    f"fresh schedule from these weights, or raise --training_steps to "
                    f"genuinely continue the same run.")
            print(f"Resuming from step {step}, epoch {epoch}")

        ckpt_state = load_safetensors(model_file, device="cpu")
        policy.train()
        cur_state = policy.state_dict()
        # NEVER restore the frozen VLM from the checkpoint. save_pretrained writes
        # the full state_dict -- requires_grad=False does not keep a module out of
        # it -- so every checkpoint carries a copy of the encoder it was trained
        # against. The shape filter cannot catch this: a LoRA-merged encoder has
        # the identical architecture, so all of its tensors "match" and get
        # overwritten by the stock weights baked into the checkpoint. Combining
        # --vlm_model_id with --resume_from_checkpoint would silently undo the
        # former, with no error and no visible sign.
        #
        # Dropping them is unconditionally correct, not just a fix for that
        # combination: the encoder is always loaded by from_pretrained(model_id)
        # before this point, so the checkpoint's copy is redundant either way.
        VLM_PREFIXES = ("model.vlm_model.", "model.visual.", "model.language_model.")
        vlm_keys = [k for k in ckpt_state if k.startswith(VLM_PREFIXES)]
        filtered = {
            k: v for k, v in ckpt_state.items()
            if k in cur_state and cur_state[k].shape == v.shape
            and not k.startswith(VLM_PREFIXES)
        }
        if vlm_keys:
            _src = vlm_model_id or getattr(type(policy.model), "VLM_MODEL_ID", "the stock hub id")
            print(f"Ignoring {len(vlm_keys)} frozen-VLM tensors in the checkpoint; the "
                  f"encoder comes from {_src}")
        # Deliberately drop transferable-by-shape weights whose MEANING changed.
        # The shape filter above cannot see this: after cross-embodiment
        # pretraining, action_pos_emb has the same (1, horizon, dit_hidden) shape
        # but encodes a different physical duration per step whenever the two fps
        # differ (64 frames is 2.1 s at 30 fps and 6.4 s at 10 fps), and
        # action_in_proj / action_out_proj are calibrated to the pretraining
        # dataset's normalisation statistics.
        if reset_params:
            dropped = [k for k in filtered if any(p in k for p in reset_params)]
            for k in dropped:
                del filtered[k]
            print(f"--reset_params {list(reset_params)}: dropped {len(dropped)} tensors, "
                  f"they will re-initialise: {dropped[:6]}")
            if not dropped:
                print(f"[WARN] --reset_params matched NOTHING. Prefixes are matched as "
                      f"substrings of parameter names, e.g. 'action_pos_emb', "
                      f"'action_in_proj', 'state_encoder'.")
        skipped = [k for k in ckpt_state if k not in filtered]
        missing = [k for k in cur_state if k not in ckpt_state]
        if skipped:
            print(f"Skipped {len(skipped)} keys (shape mismatch/removed): {skipped[:5]}")
        if missing:
            print(f"Missing {len(missing)} keys (will use init values): {missing[:5]}")
        policy.load_state_dict(filtered, strict=False)
        n_loaded, n_total = len(filtered), len(cur_state)
        del ckpt_state, filtered
        policy.to(device)
        print(f"Loaded {n_loaded}/{n_total} model keys (on CPU), moved policy to {device}")

        preprocessor, postprocessor = make_pre_post_processors(
            policy.config, dataset_stats=combined_stats,
        )

        base_lr = cfg.optimizer_lr
        resume_warmup = (cfg.scheduler_warmup_steps if reset_training_state
                         else saved_cfg_json.get("scheduler_warmup_steps",
                                                 cfg.scheduler_warmup_steps))
        print(f"Scheduler base (peak) LR: {base_lr:.2e}  (warmup {resume_warmup}, "
              + ("fresh schedule from step 0)" if reset_training_state
                 else f"decay rebuilt by fast-forwarding to step {step})"))

        trainable_params = [p for p in policy.model.parameters() if p.requires_grad]
        optimizer = make_optimizer(trainable_params, base_lr, cfg.optimizer_weight_decay, use_8bit_adam)
        opt_state_path = local_ckpt / "optimizer_state.pth"
        # Adam's moment estimates describe the PRETRAINING data distribution. On
        # a finetune they point the first few hundred steps in the wrong
        # direction, and the second-moment scaling is calibrated to gradients
        # that no longer occur.
        if reset_training_state and opt_state_path.exists():
            print("--reset_training_state: skipping optimizer_state.pth (Adam moments "
                  "belong to the previous data distribution).")
        elif opt_state_path.exists():
            try:
                opt_sd = torch.load(opt_state_path, map_location="cpu")
                optimizer.load_state_dict(opt_sd)
                del opt_sd
                for pg in optimizer.param_groups:
                    pg["lr"] = base_lr
                    pg["initial_lr"] = base_lr
                print(f"Optimizer state loaded (via CPU). Scheduler base LR set to peak {base_lr:.2e}")
            except (ValueError, RuntimeError) as e:
                print(f"Skipping optimizer state — {e}")
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=resume_warmup, num_training_steps=training_steps,
        )
        for _ in range(step):
            scheduler.step()
        print(f"Scheduler fast-forwarded to step {step}, LR={optimizer.param_groups[0]['lr']:.2e}")
    else:
        policy = WiltechsMoEPolicy(cfg)
        policy.train()
        policy.to(device)
        preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=combined_stats)
        step, epoch = 0, 0
        trainable_params = [p for p in policy.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in trainable_params)
        n_frozen = sum(p.numel() for p in policy.parameters() if not p.requires_grad)
        print(f"Total trainable parameters: {n_trainable:,}  (frozen: {n_frozen:,})")
        optimizer = make_optimizer(trainable_params, cfg.optimizer_lr, cfg.optimizer_weight_decay, use_8bit_adam)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.scheduler_warmup_steps, num_training_steps=training_steps,
        )

    # MoE expert gradient checkpointing
    if gradient_checkpointing and hasattr(policy.model, "gradient_checkpointing_enable"):
        policy.model.gradient_checkpointing_enable()

    if isinstance(preprocessor, torch.nn.Module):
        preprocessor.to(device)

    # ── Dataset setup ────────────────────────────────────────────────────
    fps = int(getattr(ref_meta, "fps", 30) or 30)
    for did in dataset_ids[1:]:
        f2 = int(getattr(metas[did], "fps", fps) or fps)
        if f2 != fps:
            raise ValueError(
                f"Dataset '{did}' fps={f2} differs from '{dataset_ids[0]}' fps={fps}. "
                f"Resample to a common fps before mixing."
            )
    frame_time = 1 / fps
    print(f"Dataset fps: {fps} (frame_time={frame_time:.4f}s)")

    obs_temporal_window = [-i * frame_time for i in range(obs)][::-1]
    action_temporal_window = [i * frame_time for i in range(horizon)]
    delta_timestamps = {
        "observation.state": obs_temporal_window,
        "action": action_temporal_window,
        **{key: [0.0] for key in camera_keys},
    }
    tolerance_s = max(0.005, frame_time / 2)

    sub_datasets = []
    ep_from: list[int] = []
    ep_to: list[int] = []
    # Stratification label per episode, for the validation split below. Task
    # rather than suite: a LIBERO suite is a bundle of ~10 tasks, so covering
    # every task covers every suite, and it additionally rules out a val set
    # that samples a suite but misses the tasks in it that the model is worst at.
    ep_group: list[str] = []
    no_task_col: list[str] = []
    offset = 0
    first_root = None
    for did in dataset_ids:
        ds = LeRobotDataset(
            did, delta_timestamps=delta_timestamps,
            force_cache_sync=True, revision="main", tolerance_s=tolerance_s,
        )
        if first_root is None:
            first_root = ds.root
        ep_ids = np.array(ds.hf_dataset["episode_index"])
        try:
            task_ids = np.array(ds.hf_dataset["task_index"])
        except (KeyError, ValueError, TypeError):
            task_ids = None
            no_task_col.append(did)
        changes = np.where(np.diff(ep_ids) != 0)[0] + 1
        starts = np.concatenate([[0], changes])
        ends = np.concatenate([changes, [len(ep_ids)]])
        kept = 0
        for s, e in zip(starts, ends):
            if max_episode_index is not None and int(ep_ids[s]) > max_episode_index:
                continue
            ep_from.append(offset + int(s))
            ep_to.append(offset + int(e))
            # Prefixed by dataset id so a multi-dataset run stratifies across
            # datasets too -- task_index restarts at 0 in each of them.
            ep_group.append(f"{did}#{int(task_ids[s])}" if task_ids is not None else did)
            kept += 1
        suffix = f" (<= ep {max_episode_index})" if max_episode_index is not None else ""
        print(f"  {did}: {len(ds)} frames, {kept} episodes{suffix}")
        sub_datasets.append(ds)
        offset += len(ds)

    dataset = ConcatDataset(sub_datasets)
    print(f"Combined dataset: {len(dataset)} frames, {len(ep_from)} episodes "
          f"across {len(sub_datasets)} dataset(s)")

    # ── Held-out validation split, BY EPISODE ────────────────────────────
    # Never by frame. Consecutive frames are 0.1 s apart and are near copies of
    # each other, so a frame split puts a sample's own neighbours on the other
    # side of the line: val loss then tracks train loss forever and detects
    # nothing, which is exactly the failure mode a val loss exists to catch.
    # (Same trap as the CV folds in kv_grounding_probe.py, which had to be
    # grouped by layout rather than by row.)
    val_ep: list[int] = []
    val_indices: list[int] = []
    if val_episodes > 0:
        n_ep = len(ep_from)
        n_val = int(round(n_ep * val_episodes)) if val_episodes < 1 else int(val_episodes)
        n_val = max(1, min(n_val, n_ep - 1))
        rng = np.random.default_rng(val_seed)
        # STRATIFIED by ep_group (dataset#task). A flat draw over all episodes
        # is only stratified in expectation: at 40 LIBERO tasks and n_val=34 the
        # odds that some task lands zero val episodes are high, and the val loss
        # then silently stops covering it. Proportional allocation with
        # largest-remainder, floored at one episode per group, capped so no
        # group loses all of its episodes to validation.
        by_group: dict[str, list[int]] = {}
        for i, g in enumerate(ep_group):
            by_group.setdefault(g, []).append(i)
        keys = sorted(by_group)
        exact = {k: n_val * len(by_group[k]) / n_ep for k in keys}
        alloc = {k: int(exact[k]) for k in keys}
        for k in sorted(keys, key=lambda k: exact[k] - alloc[k],
                        reverse=True)[:n_val - sum(alloc.values())]:
            alloc[k] += 1
        for k in keys:  # every group represented, funded by the largest donor
            if alloc[k] == 0:
                donor = max(keys, key=lambda j: alloc[j])
                if alloc[donor] > 1:
                    alloc[donor] -= 1
                    alloc[k] = 1
        for k in keys:  # never hold out a whole group
            alloc[k] = min(alloc[k], max(0, len(by_group[k]) - 1))
        val_ep = sorted(int(i) for k in keys
                        for i in rng.choice(by_group[k], size=alloc[k], replace=False))
        empty = [k for k in keys if alloc[k] == 0]
        print(f"Stratified over {len(keys)} groups (dataset#task): "
              f"{min(alloc.values())}-{max(alloc.values())} val episodes each"
              + (f"  *** {len(empty)} group(s) got NONE -- raise --val_episodes to "
                 f"at least {len(keys)}: {empty[:5]} ***" if empty else ""))
        if no_task_col:
            print(f"  [WARN] no 'task_index' column in {no_task_col}: those datasets "
                  f"are ONE group, so the split is not stratified by task inside them "
                  f"and a task can end up with no held-out episodes.")
        val_set = set(val_ep)
        for i in val_ep:
            val_indices.extend(range(ep_from[i], ep_to[i]))
        # The train sampler must see ONLY the training episodes -- dropping the
        # val frames from the val loader alone would leave them in the training
        # stream and the split would measure nothing.
        ep_from_tr = [f for i, f in enumerate(ep_from) if i not in val_set]
        ep_to_tr = [t for i, t in enumerate(ep_to) if i not in val_set]
        # Fingerprint of the split itself. Comparing two runs (e.g. dit_hidden
        # 1280 vs 2560) is only valid if they held out the SAME episodes; a
        # different --max_episode_index or dataset order silently reshuffles it,
        # and two val curves over different data are not comparable.
        fp = hashlib.sha1(",".join(map(str, val_ep)).encode()).hexdigest()[:8]
        print(f"Validation split: {len(val_ep)}/{n_ep} episodes held out "
              f"({len(val_indices)} frames), {len(ep_from_tr)} episodes "
              f"({sum(ep_to_tr) - sum(ep_from_tr)} frames) train. "
              f"seed={val_seed} fingerprint={fp}")
        if len(val_indices) < batch_size:
            raise ValueError(
                f"--val_episodes {val_episodes} holds out {len(val_indices)} frames, "
                f"fewer than one batch ({batch_size}). Raise --val_episodes.")
    else:
        ep_from_tr, ep_to_tr = ep_from, ep_to
        print("Validation: DISABLED (--val_episodes 0). Training loss alone cannot "
              "separate a better model from a better-memorised one.")

    if preprocess_in_workers:
        dataset = VLMImagePreprocDataset(
            dataset, policy.model.processor.image_processor, camera_keys,
            augment=image_transforms,
            cam_target_sizes={k: policy.model.cam_target_size(k) for k in camera_keys},
        )
        print("Image preprocessing moved into DataLoader workers "
              "(augment + Qwen image_processor per-sample, parallel + overlapped).")

    # task_index → description. Primary source is ref_meta.tasks, which
    # LeRobotDatasetMetadata already loaded from meta/tasks.parquet during
    # construction; it is a DataFrame INDEXED BY THE TASK STRING with a
    # `task_index` column. Re-reading the parquet by hand (the old path) failed
    # silently in train_wiltechs_vla.py: the map stayed empty, no batch got a
    # `task_description`, and the model fell through to batch["task"] — which
    # holds the task INDEX, not the text. Vision-only training, nothing in the
    # logs. Same code was here, so it is fixed the same way.
    task_idx_to_description: dict[int, str] = {}
    try:
        tasks_df = getattr(ref_meta, "tasks", None)
        if tasks_df is not None and "task_index" in getattr(tasks_df, "columns", []):
            if "task" in tasks_df.columns:
                task_idx_to_description = {
                    int(r["task_index"]): str(r["task"]) for _, r in tasks_df.iterrows()}
            else:
                task_idx_to_description = {
                    int(r["task_index"]): str(k) for k, r in tasks_df.iterrows()}
        else:
            p = first_root / "meta" / "tasks.parquet"
            if p.exists():
                df = pd.read_parquet(p)
                if "task_index" in df.columns:
                    task_idx_to_description = {
                        int(r["task_index"]): str(r["task"] if "task" in df.columns else k)
                        for k, r in df.iterrows()}
    except Exception as e:
        print(f"Warning: could not build the task_index → description map: {e}")

    if task_idx_to_description:
        print(f"Loaded {len(task_idx_to_description)} task descriptions "
              f"(e.g. 0 -> {task_idx_to_description.get(0, '?')!r})")
    else:
        raise RuntimeError(
            f"Could not build a task_index → description map for "
            f"'{dataset_ids[0]}'. ref_meta.tasks="
            f"{type(getattr(ref_meta, 'tasks', None)).__name__}, columns="
            f"{list(getattr(getattr(ref_meta, 'tasks', None), 'columns', []))}. "
            f"Without it no instruction reaches the VLM and the run is "
            f"vision-only. Inspect ref_meta.tasks before rerunning."
        )

    sampler = EpisodeAwareSampler(
        dataset_from_indices=ep_from_tr,
        dataset_to_indices=ep_to_tr,
        drop_n_first_frames=0,
        drop_n_last_frames=0,
        shuffle=True,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=8,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )
    print(f"\nDataLoader: {len(dataloader)} batches/epoch, batch_size={batch_size}")

    # Fixed frame subset, fixed order, no shuffling: the val loss has to move
    # only because the MODEL moved. Capped at --val_max_batches so a validation
    # pass stays a rounding error against --val_every training steps.
    val_loader = None
    if val_indices:
        cap = val_max_batches * batch_size
        if len(val_indices) > cap:
            val_indices = np.random.default_rng(val_seed + 1).choice(
                val_indices, size=cap, replace=False).tolist()
            val_indices.sort()
        # augment=None: the val loss must not include augmentation noise, or it
        # measures the augmentation as much as the model. The training wrapper
        # above keeps its own augment=image_transforms.
        val_base = dataset
        if preprocess_in_workers:
            val_base = VLMImagePreprocDataset(
                dataset.dataset, policy.model.processor.image_processor, camera_keys,
                augment=None,
                cam_target_sizes={k: policy.model.cam_target_size(k) for k in camera_keys},
            )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(val_base, val_indices),
            num_workers=2, batch_size=batch_size, shuffle=False,
            pin_memory=device.type != "cpu", drop_last=True,
        )
        print(f"Validation loader: {len(val_loader)} batches x {batch_size} "
              f"= {len(val_loader) * batch_size} frames, every {val_every} steps")

    # ── Training loop ────────────────────────────────────────────────────
    print(f"\nStarting training loop ({training_steps} steps, batch_size={batch_size})...")
    print(f"MoE config: {num_experts} experts x {expert_num_layers} layers each "
          f"(dit_hidden={dit_hidden_size})")
    done = False
    lang_check_done = False

    def _prep_batch(batch, augment: bool):
        """Device move + instruction resolution + preprocessor.

        Shared by training and validation so the two cannot drift apart -- a val
        batch built even slightly differently from a train batch makes the gap
        between the two losses report the difference in PREPARATION, which is
        indistinguishable from the overfitting it is supposed to detect.
        `augment` is the one deliberate difference."""
        nonlocal lang_check_done
        for key in list(batch.keys()):
            if isinstance(batch[key], torch.Tensor) and not key.startswith(_VLM_PIX_PREFIX):
                batch[key] = batch[key].to(device, non_blocking=True)

        # Preferred: the batch already carries the instruction TEXT.
        if ("task" in batch and isinstance(batch["task"], (list, tuple))
                and all(isinstance(t, str) for t in batch["task"])):
            batch["task_description"] = batch["task"]
        else:
            # Otherwise resolve an INDEX through the map. The index can
            # arrive under either key: some datasets expose "task_index",
            # others put the integer in "task" itself (which then collates
            # to a (B,) tensor and is NOT the text, however it is named).
            idx_src = None
            for k in ("task_index", "task"):
                v = batch.get(k)
                if isinstance(v, torch.Tensor):
                    idx_src = v[:, 0] if v.dim() > 1 else v
                    break
            if idx_src is not None:
                batch["task_description"] = [
                    task_idx_to_description.get(int(ti), "") for ti in idx_src
                ]
                n_unmapped = sum(1 for d in batch["task_description"] if not d)
                if n_unmapped and not lang_check_done:
                    print(f"WARNING: {n_unmapped}/{len(idx_src)} task indices are "
                          f"not in the description map (known: "
                          f"{sorted(task_idx_to_description)[:10]}). Those "
                          f"samples train with an EMPTY instruction.")

        if augment:
            if not preprocess_in_workers:
                present_cams = [c for c in camera_keys if c in batch]
                batch = apply_image_augmentations(batch, present_cams, image_transforms)
            if "observation.state" in batch:
                batch = apply_joint_augmentations(batch, "observation.state")

        vlm_pix = {k: batch.pop(k) for k in list(batch) if k.startswith(_VLM_PIX_PREFIX)}
        batch = preprocessor(batch)
        batch.update(vlm_pix)
        return batch

    def _autocast():
        return (torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                if device.type == "cuda"
                else torch.autocast(device_type="cpu", enabled=False))

    @torch.no_grad()
    def _validate():
        """Mean MAIN loss over the held-out episodes. Lower than the train loss
        is normal (no augmentation, no dropout); what matters is the GAP and
        whether the val curve turns up while the train curve keeps falling.

        Two things are pinned so the number moves only when the model does:
          * the frame subset and its order (fixed Subset, shuffle=False)
          * the flow-matching draws -- sample_time and sample_noise are random
            per call, and at these batch sizes their variance swamps the model
            improvement between two validations. Re-seeding makes every pass
            score the SAME (t, noise) pairs, then restores the training RNG so
            the run is otherwise byte-identical to one without validation.
        """
        was_training = policy.training
        policy.eval()   # kills dropout, vision-KV dropout, router noise, and the
                        # contrastive branch -- all gated on self.training
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(val_seed)
        tot, nb = 0.0, 0
        try:
            for vb in val_loader:
                vb = _prep_batch(vb, augment=False)
                with _autocast():
                    policy.model.compute_loss(vb)
                tot += float(policy.model._last_loss_components["main"])
                nb += 1
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            if was_training:
                policy.train()
        return tot / max(1, nb)

    prog_bar = tqdm(total=training_steps, desc="Training Progress", initial=step)
    val_hist: list[tuple[int, float]] = []

    while not done:
        epoch += 1
        for batch in dataloader:
            batch = _prep_batch(batch, augment=True)

            if not lang_check_done:
                lang_check_done = True
                descs = batch.get("task_description")
                if descs is None:
                    descs = batch.get("task")
                if descs is None:
                    print("WARNING: neither 'task_description' nor 'task' in "
                          "batch after preprocessor — model will train VISION-ONLY.")
                else:
                    n_nonempty = sum(1 for d in descs if d)
                    print(f"\n--- LANG CHECK (step 0) ---")
                    print(f"  key present: task_description={'task_description' in batch}, "
                          f"task={'task' in batch}")
                    print(f"  non-empty descriptions: {n_nonempty}/{len(descs)}")
                    print(f"  examples: {list(descs[:2])}")
                    try:
                        tok = policy.model.processor.tokenizer(
                            list(descs), return_tensors="pt", padding=True,
                            truncation=True, add_special_tokens=True,
                        )
                        L_lang = int(tok["input_ids"].shape[1])
                        print(f"  tokenized L_lang (padded): {L_lang}")
                    except Exception as e:
                        print(f"  (could not tokenize to report L_lang: {e})")
                    if n_nonempty == 0:
                        print("WARNING: ALL descriptions empty — language is NOT reaching "
                              "the model. Fix data flow before tuning contrastive loss.")
                    print("--- end LANG CHECK ---\n")

            if step % progress_update_freq == 0:
                policy.model._capture_attention_stats = True

            if vision_dropout_start >= 0.0 and vision_dropout_anneal_steps > 0:
                frac = min(step / vision_dropout_anneal_steps, 1.0)
                vp_now = vision_dropout_start + (vision_dropout_prob - vision_dropout_start) * frac
                policy.model.config.vision_dropout_prob = float(vp_now)

            with _autocast():
                loss, _ = policy.forward(batch)

            loss.backward()

            if step % progress_update_freq == 0:
                _log_gradient_analysis(policy, step)

            trainable_params = [p for p in policy.parameters() if p.requires_grad]
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if step % progress_update_freq == 0:
                lr = optimizer.param_groups[0]["lr"]
                prog_bar.set_description(f"Epoch {epoch}, Step {step}")
                prog_bar.set_postfix({
                    "loss": f"{loss.item():.3f}",
                    "lr": f"{lr:.2e}",
                    "grad_norm": f"{grad_norm:.2f}",
                })

            # After the optimizer step and after _log_gradient_analysis: a
            # validation forward overwrites every _last_* diagnostic on the
            # model (router usage, thought RMS, cross-attn shares), so running
            # it earlier would make the gradient report describe the HELD-OUT
            # batch while labelling it as the training step.
            if val_loader is not None and step > 0 and step % val_every == 0:
                # BEFORE _validate: its forwards overwrite _last_loss_components,
                # so reading the train loss afterwards would report the last VAL
                # batch and print a gap of ~0 no matter how bad the fit is.
                tr = float(policy.model._last_loss_components["main"])
                v = _validate()
                val_hist.append((step, v))
                trend = ""
                if len(val_hist) >= 2:
                    d = v - val_hist[-2][1]
                    trend = f" ({d:+.4f} vs step {val_hist[-2][0]})"
                    if d > 0 and len(val_hist) >= 3 and val_hist[-2][1] > val_hist[-3][1]:
                        trend += "  *** val rising 2x -- overfitting ***"
                print(f"\n[val] step {step}: val_main={v:.4f}  train_main={tr:.4f}  "
                      f"gap={v - tr:+.4f}{trend}")

            if step > 0 and step % checkpoint_freq == 0:
                checkpoint_dir = output_directory / f"checkpoint-{step}"
                checkpoint_dir.mkdir(exist_ok=True)
                policy.config.training_step = step
                policy.config.training_epoch = epoch
                policy.config.optimizer_lr = optimizer.param_groups[0]["lr"]
                policy.config.current_lr = optimizer.param_groups[0]["lr"]
                policy.config.training_steps_total = training_steps
                policy.save_pretrained(checkpoint_dir)
                torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer_state.pth")
                preprocessor.save_pretrained(checkpoint_dir)
                postprocessor.save_pretrained(checkpoint_dir)
                print(f"\nCheckpoint saved at step {step}")

            step += 1
            if step % progress_update_freq == 0 or step >= training_steps:
                prog_bar.update(progress_update_freq)

            if step >= training_steps:
                done = True
                prog_bar.close()
                break

    prog_bar.close()

    # ── Final save ───────────────────────────────────────────────────────
    policy.config.training_step = step
    policy.config.training_epoch = epoch
    policy.config.optimizer_lr = optimizer.param_groups[0]["lr"]
    policy.config.current_lr = optimizer.param_groups[0]["lr"]
    policy.config.training_steps_total = training_steps
    policy.save_pretrained(output_directory)
    torch.save(optimizer.state_dict(), output_directory / "optimizer_state.pth")
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)
    print(f"\nTraining complete. Model saved to {output_directory}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train WiltechsMoE on one or more homogeneous LeRobot datasets.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints")
    parser.add_argument("--dataset_id", type=str, nargs="+", default=["ISdept/piper_arm"],
                        help="One or more LeRobot dataset ids. Multiple are concatenated and "
                             "must share a homogeneous schema (same robot/cameras/dims/fps); "
                             "their normalization stats are aggregated.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from a checkpoint")
    parser.add_argument("--reset_training_state", action="store_true",
                        help="Load the checkpoint's WEIGHTS but start a new run: step 0, a "
                             "fresh warmup+cosine over --training_steps, and no optimizer "
                             "state. Required to finetune from a FINISHED run -- such a "
                             "checkpoint reports step == training_steps_total, which without "
                             "this flag makes the loop exit before the first optimizer step.")
    parser.add_argument("--reset_params", type=str, nargs="+", default=None,
                        help="Parameter-name substrings to DROP from the checkpoint so they "
                             "re-initialise, e.g. 'action_pos_emb action_in_proj "
                             "action_out_proj'. Use when a tensor transfers by shape but not "
                             "by meaning: action_pos_emb indexes frames, so the same horizon "
                             "spans 2.1s at 30fps and 6.4s at 10fps, and the action "
                             "projections are calibrated to the pretraining normalisation.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (Qwen3-VL-4B backbone is memory-heavy; 8-24).")
    parser.add_argument("--training_steps", type=int, default=300000, help="Total training steps")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Recompute MoE expert layer activations in backward to save GPU memory "
                             "(trades extra forward compute; frozen VLM is unaffected). "
                             "Recommended when using the contrastive loss, which runs a 2nd MoE forward.")
    parser.add_argument("--num_experts", type=int, default=4,
                        help="Number of MoE expert decoders. Each expert cross-attends to a "
                             "different block of VLM KV cache layers. More experts = more "
                             "diversity but more parameters (each expert ~496M at 4 layers).")
    parser.add_argument("--expert_num_layers", type=int, default=4,
                        help="DiT layers per expert. Total VLM capture layers = "
                             "num_experts x expert_num_layers (must be <= 36 for Qwen3-VL-4B). "
                             "Default 4x4=16 layers captured from 36 total.")
    parser.add_argument("--dit_hidden_size", type=int, default=1280,
                        help="Expert decoder width. 0 = match VLM hidden size (2560). "
                             "Set a smaller multiple of the VLM head_dim (e.g. 1280) to shrink "
                             "expert self-attn/FFN/adaLN (quadratic param savings). "
                             "Cross-attention is bridged back up to the frozen VLM KV.")
    parser.add_argument("--use_8bit_adam", action="store_true",
                        help="Use bitsandbytes 8-bit Adam (int8 optimizer state) instead of fp32 Adam, "
                             "cutting optimizer memory ~4x. Requires bitsandbytes + CUDA.")
    parser.add_argument("--val_episodes", type=float, default=0.0,
                        help="Held-out EPISODES for validation: <1 a fraction, >=1 a count, "
                             "0 (default) disables. The split is on EPISODES, never frames -- "
                             "consecutive frames are 0.1s apart and near copies, so a frame "
                             "split leaves a sample's own neighbours on the other side and the "
                             "val loss tracks the train loss forever, detecting nothing. "
                             "Without this, training loss alone cannot tell a better model from "
                             "a better-memorised one, which is the whole question when "
                             "comparing --dit_hidden_size settings.")
    parser.add_argument("--val_every", type=int, default=500,
                        help="Steps between validation passes.")
    parser.add_argument("--val_seed", type=int, default=42,
                        help="Seeds BOTH the episode split and the flow-matching (t, noise) "
                             "draws used to score it. Two runs being compared must share this "
                             "value or they are scored on different data with different noise; "
                             "the startup line prints a fingerprint of the resulting split.")
    parser.add_argument("--val_max_batches", type=int, default=8,
                        help="Cap on validation batches per pass, so a pass stays cheap "
                             "relative to --val_every training steps.")
    parser.add_argument("--max_episode_index", type=int, default=None,
                        help="Filter to episodes with index <= this value.")
    parser.add_argument("--lock_joint_index", type=int, default=None,
                        help="Action dim with weight 0 (e.g. piper_arm joint 4 = index 3). "
                             "Omit / pass -1 to weight all dims.")
    parser.add_argument("--contrastive_loss_weight", type=float, default=0.1,
                        help="Weight for the language-permute contrastive loss (default: 0.1).")
    parser.add_argument("--contrastive_margin", type=float, default=0.05,
                        help="Hinge margin on MSE between v_t and v_wrong (default: 0.05).")
    parser.add_argument("--contrastive_hard_negatives", action="store_true",
                        help="Pair each sample with its hardest in-batch negative.")
    parser.add_argument("--vision_kv_dropout_prob", type=float, default=0.0,
                        help="Training-time dropout on the VLM VISION positions of the "
                             "cross-attn memory (language is never dropped).")
    parser.add_argument("--vision_dropout_prob", type=float, default=0.3,
                        help="RobotCNN token dropout (regularizer). Default 0.3.")
    parser.add_argument("--vision_dropout_start", type=float, default=-1.0,
                        help="CURRICULUM: starting (high) RobotCNN dropout at step 0, annealed "
                             "linearly down to --vision_dropout_prob.")
    parser.add_argument("--vision_dropout_anneal_steps", type=int, default=0,
                        help="Steps to anneal RobotCNN dropout. 0 disables the schedule.")
    parser.add_argument("--use_chat_template", action="store_true",
                        help="Wrap the VLM input as a Qwen ChatML turn.")
    parser.add_argument("--chat_directive", type=str, default="",
                        help="Optional short directive prepended to the task inside the user turn.")
    parser.add_argument("--use_descriptive_objects", action="store_true",
                        help="Rewrite ambiguous object/region names into visually-groundable "
                             "descriptions via task_rewrites.py.")
    parser.add_argument("--vision_input_size", type=int, default=0,
                        help="Square side length (px) fed to the Qwen image processor. "
                             "Qwen3-VL emits one merged vision token per 32x32 input px, so "
                             "256->8x8=64 tok/cam, 512->16x16=256, 1024->32x32=1024. "
                             "0 (default) = processor smart-resize defaults. NOTE L_vlm is also "
                             "the K/V length of every expert's cross-attention, so the cost "
                             "multiplies through num_experts x expert_num_layers.")
    parser.add_argument("--vision_hires_cameras", type=str, nargs="+", default=None,
                        help="Camera key(s) that get --vision_input_size; the rest keep the "
                             "processor default. Empty = all cameras. Restricting this to the "
                             "third-person view roughly halves the added cost, since the "
                             "relations that need the resolution are not resolvable at the "
                             "wrist camera's scale anyway.")
    parser.add_argument("--text_last", dest="text_first", action="store_false", default=True,
                        help="Legacy VLM layout: instruction AFTER the images. Under the VLM's "
                             "causal mask this leaves every vision KV language-blind. Default is "
                             "text-first (instruction before images).")
    parser.add_argument("--robot_encoder_tokens", type=int, default=16,
                        help="Robot CNN tokens per camera. Must be a perfect square. Default: 16 (4x4).")
    parser.add_argument("--vlm_model_id", type=str, default="",
                        help="Local dir holding a LoRA-merged Qwen3-VL from "
                             "lora_finetune_qwen.py --merge_and_save. Empty = stock "
                             "hub weights. This moves every KV cache the experts "
                             "cross-attend to, so a checkpoint trained against the "
                             "stock encoder does not warm-start cleanly.")
    parser.add_argument("--n_action_steps", type=int, default=4,
                        help="Steps the action queue executes per replan at INFERENCE. Purely "
                             "bookkeeping for training -- it no longer feeds the loss boundary "
                             "(--loss_exec_steps owns that now), so changing it cannot alter "
                             "what is optimised. It IS written into the checkpoint config, so "
                             "an eval that does not override it inherits this value: the old "
                             "hardcoded 64 meant 6.4s of open-loop execution at 10Hz. Default 4 "
                             "matches how evals are actually run.")
    parser.add_argument("--robot_encoder_input_size", type=int, default=224,
                        help="Square resolution the RobotCNN resizes to. Was a config field "
                             "that the training script NEVER passed, so it has been pinned at "
                             "224 in every run to date -- a pointless 256->224 downsample of "
                             "a native 256px LIBERO frame. ResNet-18 through layer3 has stride "
                             "16, so this gives an input/16 feature map: 224->14x14, 256->16x16. "
                             "256 is the natural setting (no resampling, 16x16 grid).")
    parser.add_argument("--robot_cnn_fine_tokens", type=int, default=0,
                        help="Dense token grid for the wrist camera only (perfect square). "
                             "Other cameras keep --robot_encoder_tokens. 0 disables. The "
                             "default 4x4=16 grid pools a 14x14 feature map down to 16 tokens "
                             "covering 64 native px each -- COARSER than the frozen VLM's 32px "
                             "merged tokens, which is the opposite of this encoder's purpose. "
                             "At --robot_encoder_input_size 256: 64=32px, 100=25.6px, "
                             "144=21.3px, 256=16px (ceiling). Costs DiT sequence length, which "
                             "is quadratic in self-attn and multiplies through "
                             "num_experts x expert_num_layers.")
    parser.add_argument("--robot_cnn_fine_cameras", type=str, nargs="+", default=None,
                        help="Explicit camera key(s) that get --robot_cnn_fine_tokens. "
                             "Default: auto-detect the wrist/gripper view.")
    parser.add_argument("--robot_cnn_cameras", type=str, nargs="+", default=None,
                        help="Explicit camera key(s) the trainable RobotCNN ingests.")
    parser.add_argument("--robot_cnn_wrist_only", action="store_true",
                        help="Restrict the RobotCNN to the auto-detected WRIST/gripper camera.")
    parser.add_argument("--no_robot_cnn", dest="use_robot_cnn", action="store_false", default=True,
                        help="Remove the RobotCNN entirely. It is a raw-pixel pathway that "
                             "BYPASSES the VLM, so it can supply close-range visual servoing "
                             "for any task -- which makes the VLM's own vision redundant no "
                             "matter how many tasks are in the training set. Disabling it "
                             "changes no parameter shape (the encoder's keys are simply "
                             "absent), so an existing checkpoint resumes cleanly and this is a "
                             "controlled ablation rather than a from-scratch run.")
    parser.add_argument("--noise_temporal_correlation", type=float, default=0.0,
                        help="AR(1) coefficient correlating the flow-matching source noise "
                             "along the action horizon (0=white; ~0.9=temporally smooth).")
    parser.add_argument("--preprocess_in_workers", action="store_true",
                        help="Run image augmentation + the Qwen image_processor inside the "
                             "DataLoader workers instead of on the critical path.")
    parser.add_argument("--router_temperature", type=float, default=1.0,
                        help="Temperature for the MoE router softmax. Higher = more uniform "
                             "expert usage; lower = more peaked/sparse routing. Default 1.0.")
    parser.add_argument("--router_balance_weight", type=float, default=0.1,
                        help="Weight for the router load-balancing loss (CV^2 of expert usage). "
                             "Prevents expert collapse. 0.1 is the default; 0.01 is too weak and "
                             "allows expert collapse within ~100 steps. 0 disables.")
    parser.add_argument("--router_top_k", type=int, default=0,
                        help="Top-k sparse routing: only the top-k experts get nonzero weight. "
                             "0 (default) = dense (all experts active). Set to e.g. 2 for top-2 "
                             "routing (saves compute with many experts).")
    parser.add_argument("--vlm_capture_layers", type=int, nargs="+", default=None,
                        help="Explicit VLM layer indices to capture KV from. If omitted, "
                             "layers are auto-selected: num_experts x expert_num_layers layers "
                             "uniformly sampled from 0..35.")
    parser.add_argument("--num_thought_tokens", type=int, default=8,
                        help="Number of learned thought tokens for spatial reasoning. 0 disables.")
    parser.add_argument("--thought_qformer_layers", type=int, default=2,
                        help="Number of cross-attention layers in the thought Q-Former.")
    parser.add_argument("--thought_vlm_layer_idx", type=int, default=-1,
                        help="VLM layer to read KV from for thought generation. -1 = deepest captured layer.")
    parser.add_argument("--thought_consistency_weight", type=float, default=0.0,
                        help="Weight for thought consistency loss across denoising timesteps. 0 disables.")
    parser.add_argument("--image_aug_translate", type=float, default=0.0,
                        help="Random image translation as a fraction of frame size. Default 0: "
                             "the action label is not transformed with the image, so this teaches "
                             "position-invariance -- on a 256px LIBERO frame 0.03 is +-7.7px "
                             "against a ~19px ramekin-to-bowl separation. 0.03 restores the "
                             "pre-2026-08 behaviour.")
    parser.add_argument("--loss_exec_steps", type=int, default=0,
                        help="Horizon index where --future_steps_weight kicks in. 0 (default) "
                             "= use n_action_steps, which the training script pins to the full "
                             "horizon of 64 -- so the down-weighting has never actually applied "
                             "and all 64 steps are equal. At 10Hz that is 6.4s predicted from "
                             "one frame, most of it aleatoric, taking most of the gradient, "
                             "while the ~4 steps an eval executes take 6.25%%. Set 8 to give "
                             "those 4 steps ~16%%. UNTESTED: the far horizon is not executed "
                             "but predicting it is a useful auxiliary task, so move this "
                             "gradually rather than to an extreme.")
    parser.add_argument("--future_steps_weight", type=float, default=0.3,
                        help="Loss weight on horizon steps at or beyond --loss_exec_steps. "
                             "Only has any effect once loss_exec_steps < horizon.")
    parser.add_argument("--image_aug_scale", type=float, default=0.0,
                        help="Random image scale jitter (+-this). Default 0, same reasoning as "
                             "--image_aug_translate. 0.05 restores the previous behaviour.")
    args = parser.parse_args()

    _v = args.robot_encoder_tokens
    if int(_v ** 0.5) ** 2 != _v:
        parser.error(f"--robot_encoder_tokens must be a perfect square, got {_v}")
    if args.robot_cnn_fine_tokens:
        _f = args.robot_cnn_fine_tokens
        if int(_f ** 0.5) ** 2 != _f:
            parser.error(f"--robot_cnn_fine_tokens must be a perfect square, got {_f}")
        if int(_f ** 0.5) > args.robot_encoder_input_size // 16:
            parser.error(
                f"--robot_cnn_fine_tokens {_f} ({int(_f ** 0.5)}x{int(_f ** 0.5)}) exceeds the "
                f"{args.robot_encoder_input_size // 16}x{args.robot_encoder_input_size // 16} "
                f"feature map produced by --robot_encoder_input_size "
                f"{args.robot_encoder_input_size}. Adaptive pooling would upsample and add no "
                f"information. Use --robot_encoder_input_size {int(_f ** 0.5) * 16} or higher.")
    if not args.use_robot_cnn and (args.robot_cnn_fine_tokens or args.robot_cnn_fine_cameras):
        parser.error("--no_robot_cnn removes the RobotCNN, so --robot_cnn_fine_tokens / "
                     "--robot_cnn_fine_cameras have nothing to configure. Drop one side.")
    if not args.use_robot_cnn and (args.robot_cnn_cameras or args.robot_cnn_wrist_only):
        # Silently-inert flags are exactly how the vision_dropout_prob confusion
        # happened: a setting that reads as active while doing nothing.
        parser.error("--no_robot_cnn removes the RobotCNN, so --robot_cnn_cameras / "
                     "--robot_cnn_wrist_only have nothing to configure. Drop one side.")
    if args.lock_joint_index is not None and args.lock_joint_index < 0:
        args.lock_joint_index = None

    # Validate MoE layer count
    total_capture = args.num_experts * args.expert_num_layers
    if total_capture > 36:
        parser.error(f"num_experts ({args.num_experts}) x expert_num_layers ({args.expert_num_layers}) "
                     f"= {total_capture} > 36 (Qwen3-VL-4B layer count). Reduce one or both.")
    if args.router_top_k > args.num_experts:
        parser.error(f"--router_top_k ({args.router_top_k}) cannot exceed --num_experts ({args.num_experts}).")
    # An explicit --vlm_capture_layers list is the ONLY way these two can
    # disagree (the auto path derives the count from them). When they do,
    # ExpertDecoder.forward silently cycles its KV blocks with `i % len(...)`:
    # too few captured layers means DiT layers reuse the same VLM layer, too
    # many means the tail is captured and never read. Both run fine and train to
    # a worse model, so fail here instead.
    if args.vlm_capture_layers is not None and len(args.vlm_capture_layers) != total_capture:
        parser.error(
            f"--vlm_capture_layers has {len(args.vlm_capture_layers)} entries but "
            f"num_experts ({args.num_experts}) x expert_num_layers ({args.expert_num_layers}) "
            f"= {total_capture}. Each DiT layer reads exactly one captured VLM layer.")

    train(**vars(args))