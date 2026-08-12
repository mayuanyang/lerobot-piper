"""
Training script for WiltechsVLA (Qwen3-VL-4B encoder-decoder MoT flow matching).

Mirrors `train_wilro.py`'s data path: train on ONE OR MORE explicit LeRobot v3
datasets passed via `--dataset_id`. Multiple datasets are concatenated and
assumed HOMOGENEOUS (same robot / cameras / state+action dims / fps) — e.g.
several piper sets — and their normalization stats are aggregated. There is NO
community-hub discovery, version filtering, allowlist/denylist, or canonical-
schema projection here; the model's input/output features come straight from the
dataset schema. For mixed-robot community pretraining use `train_community.py`
(the canonical multi-robot DatasetAdapter path) instead.

Usage:
    # Single dataset
    python src/train_wiltechs_vla.py \
        --output_dir outputs/train/wiltechs_piper \
        --dataset_id ISdept/piper_arm \
        --batch_size 16 \
        --training_steps 300000

    # Concatenate several homogeneous datasets
    python src/train_wiltechs_vla.py \
        --output_dir outputs/train/wiltechs_piper \
        --dataset_id ISdept/piper_arm ISdept/piper_arm_v2 \
        --batch_size 16

    # Resume from a checkpoint
    python src/train_wiltechs_vla.py \
        --output_dir outputs/train/wiltechs_piper \
        --dataset_id ISdept/piper_arm \
        --resume_from_checkpoint outputs/train/wiltechs_piper/checkpoint-50000
"""

from __future__ import annotations

import json
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

from models.wiltechs_vla.wiltechs_vla_config import WiltechsVLAConfig
from models.wiltechs_vla.wiltechs_vla_policy import WiltechsVLAPolicy
from models.wiltechs_vla.processor_wiltechs_vla import make_pre_post_processors
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
# state memory ~4× (fp32 m+v → int8 m+v). The big DiT stack dominates GPU
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
def get_augmentations():
    spatial = v2.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05), fill=0)
    color = v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08)
    blur = v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))], p=0.3)
    return v2.Compose([spatial, color, blur])


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
        # Per-camera square input size for the Qwen processor, 0 = its default.
        # Must match the model's cam_target_size() or this path and the
        # in-model fallback would build different vision grids for the same
        # frame -- silently, since both are valid inputs downstream.
        self.cam_target_sizes = dict(cam_target_sizes or {})

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        present = [k for k in self.camera_keys
                   if k in sample and isinstance(sample[k], torch.Tensor)]
        if not present:
            return sample

        # Normalize each camera to (3, H, W) for the transform; remember whether
        # it carried a leading T=1 dim so we can restore the original shape.
        imgs3, had_t = [], []
        for k in present:
            v = sample[k]
            had_t.append(v.dim() == 4)
            imgs3.append(v[0] if v.dim() == 4 else v)

        if self.augment is not None:
            # One transform call over the stacked cameras → identical random
            # params across cameras of this sample (cross-camera consistency).
            stacked = self.augment(torch.stack(imgs3, dim=0))
            imgs3 = [stacked[i] for i in range(len(present))]
            for i, k in enumerate(present):
                sample[k] = imgs3[i].unsqueeze(0) if had_t[i] else imgs3[i]

        # Run the (CPU) Qwen image_processor here, in the worker.
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
        """Returns (mean|grad|, grad RMS/param, params with grad, params present).

        The last one separates the two ways a module can report nothing:
        DISABLED (no such parameter — RobotCNN off, latent Q-Former at 0 tokens)
        from BROKEN (parameters exist and received no gradient). The old version
        printed "no grad" for both, so three permanently-dead lines sat in every
        report and the one that would actually matter was camouflaged among them.
        """
        total, g_norm_sq, count, present = 0.0, 0.0, 0, 0
        for name, param in policy.model.named_parameters():
            if not (param.requires_grad and prefix in name):
                continue
            present += param.numel()
            if param.grad is not None:
                total += param.grad.abs().mean().item() * param.numel()
                g_norm_sq += param.grad.norm().item() ** 2
                count += param.numel()
        if count == 0:
            return None, None, 0, present
        # `Avg Abs Grad` = mean|grad| over ALL params — rounds to ~0 for large
        # modules whose gradient is concentrated in a few sub-params (gates,
        # norms), making them look dead. `RMS/param` = grad_L2_norm / sqrt(N) is
        # scale-fair across modules of any size, so it's the honest comparison.
        return (total / count, (g_norm_sq ** 0.5) / (count ** 0.5), count, present)

    for label, prefix in [
        ("State Enc",      "state_encoder"),
        ("Register Tok",   "register_tokens"),
        ("Action Pos Emb", "action_pos_emb"),
        ("Time Embedder",  "time_embedder"),
        ("DiT Layers",     "dit_layers"),
        ("Action In/Out",  "action_"),
        ("Final Norm",     "final_norm"),
        ("Robot CNN",      "robot_visual_encoder"),
        ("Latent QFormer", "latent_qformer"),
    ]:
        grad, rms_pp, n, present = _grad_stats(prefix)
        if grad is not None:
            print(f"  {label:14s} - Avg Abs Grad: {grad:.6f}   RMS/param: {rms_pp:.2e} "
                  f"({n} params)")
        elif present:
            print(f"  {label:14s} - *** {present} params, NO GRAD ***")
        # present == 0 -> module disabled; say nothing.

    reg = getattr(policy.model, "register_tokens", None)
    if reg is not None:
        r = reg.detach().float()                       # (1, R, H)
        rms = r.pow(2).mean().sqrt().item()
        # How much of a register's magnitude is its OWN, as opposed to a direction
        # all R of them share. The registers are pure parameters with no input, so
        # the only thing that can make them 32 tokens rather than 1 is that they
        # point in different directions. If they collapse onto a common vector
        # this ratio falls and the block is one token wearing 32 hats -- the same
        # failure the MoE's thought tokens showed as vary/total decay.
        #
        # At std=0.02 iid init the mean of R samples still absorbs a 1/R share, so
        # the ceiling is sqrt(1 - 1/R), printed alongside rather than assumed 1.0.
        R = r.shape[1]
        spread = (r - r.mean(dim=1, keepdim=True)).pow(2).mean().sqrt().item()
        ceil = (1.0 - 1.0 / R) ** 0.5
        a_rms = getattr(policy.model, "_last_action_emb_rms", None)
        ratio = f"   vs action_emb: {rms / a_rms:.3f}" if a_rms else ""
        g = reg.grad
        gstr = f"{g.norm().item():.2e}" if g is not None else "None"
        print(f"  Register Tok   : RMS {rms:.4f}  distinct {spread / max(rms, 1e-9):.3f} "
              f"(ceiling {ceil:.3f} at R={R}; 0.0 = all identical)  grad_norm {gstr}{ratio}")

    if hasattr(policy.model, "latent_qformer"):
        qf = policy.model.latent_qformer
        w_norm_sq = 0.0
        g_norm_sq = 0.0
        for p in qf.parameters():
            w_norm_sq += p.detach().norm().item() ** 2
            if p.grad is not None:
                g_norm_sq += p.grad.norm().item() ** 2
        # Residual gates start at 0; their growth shows the latents becoming active.
        gate_vals = torch.cat([g.detach().reshape(-1) for g in qf.gates]).abs()
        print(f"  Latent QFormer - weight_norm: {w_norm_sq ** 0.5:.4e}   "
              f"grad_norm: {g_norm_sq ** 0.5:.4e}   gate|mean|: {gate_vals.mean().item():.4e}")

    if hasattr(policy.model, "lang_attn_bias"):
        bias_tensor = policy.model.lang_attn_bias.detach()
        softplus_vals = F.softplus(bias_tensor).cpu()
        grad = policy.model.lang_attn_bias.grad
        grad_norm_str = f"{grad.norm().item():.4e}" if grad is not None else "None"
        sp_str = "[" + " ".join(f"{v:.2f}" for v in softplus_vals.tolist()) + "]"
        print(f"  Lang attn bias - softplus per-layer: {sp_str}")
        print(f"                   min={softplus_vals.min().item():.3f}  "
              f"max={softplus_vals.max().item():.3f}  "
              f"mean={softplus_vals.mean().item():.3f}  grad_norm: {grad_norm_str}")

    if hasattr(policy.model, "lang_adaptor"):
        w_norm = sum(p.detach().norm().item() ** 2 for p in policy.model.lang_adaptor.parameters()) ** 0.5
        g_norm_sq = sum(p.grad.norm().item() ** 2 for p in policy.model.lang_adaptor.parameters() if p.grad is not None) ** 0.5
        print(f"  Lang adaptor   - weight_norm: {w_norm:.4e}   grad_norm: {g_norm_sq:.4e}")

    # Name any DiT params that missed gradient this step — explains drops in
    # the "DiT Layers" param count above (which only counts grad-carrying params).
    none_named = [
        (name, p.numel()) for name, p in policy.model.named_parameters()
        if p.requires_grad and "dit_layers" in name and p.grad is None
    ]
    if none_named:
        n_none = sum(n for _, n in none_named)
        print(f"  [grad=None] {n_none} DiT params across {len(none_named)} tensors, "
              f"e.g.: {[name for name, _ in none_named[:4]]}")

    stats = getattr(policy.model, "_last_attention_stats", None)
    if stats:
        # Match DiT sequence order: [state, register, robot, latent, action]
        order = ["state", "register", "robot", "latent", "action"]
        ordered = [(k, stats[k]) for k in order if k in stats]
        cells = "  ".join(f"{k}={v*100:5.1f}%" for k, v in ordered)
        print(f"  Action→ self-attn : {cells}    (last DiT layer)")

    x_stats = getattr(policy.model, "_last_cross_attention_stats", None)
    if x_stats:
        print("  Action→ x-attn    : " + format_xattn(x_stats)
              + "    (mean over sampled depths)")
        per_d = x_stats.get("_per_expert") or []
        labels = x_stats.get("_labels") or [f"d{i}" for i in range(len(per_d))]
        if len(per_d) > 1:
            # The depth spread is the signal, not the mean: WiltechsMoE at its
            # 92% checkpoint read 55.8% VISION at VLM layer 8 but only 8.6% at
            # layer 35. A shallow band that has gone language-dominated is the
            # thing to catch — that is where spatial grounding lives.
            cells = "  ".join(f"{lab}={lang * 100:4.1f}%"
                              for lab, (_v, lang) in zip(labels, per_d))
            print(f"  x-attn lang/depth : {cells}    (language %, shallow→deep)")

    _cfg = policy.model.config
    _rcnn_on = getattr(_cfg, "use_robot_cnn", True)
    _vdrop = float(getattr(_cfg, "vision_dropout_prob", 0.0))
    _vkvdrop = float(getattr(_cfg, "vision_kv_dropout_prob", 0.0))
    _cnn_cams = getattr(_cfg, "robot_cnn_cameras", None) \
        or getattr(_cfg, "cameras_for_vision_state_concat", [])
    _cnn_cams_short = [c.rsplit(".", 1)[-1] for c in _cnn_cams]
    print(f"  Vision dropout    : robotCNN={_vdrop:.2f} ({'ON' if _rcnn_on else 'OFF'})  "
          f"VLM-vis-KV={_vkvdrop:.2f}    (training-time only; forced 0 at eval/RL)")
    if _rcnn_on:
        print(f"  RobotCNN cameras  : {_cnn_cams_short}  (VLM sees all "
              f"{len(getattr(_cfg, 'cameras_for_vision_state_concat', []))})")

    comps = getattr(policy.model, "_last_loss_components", None)
    cw = getattr(policy.model.config, "contrastive_loss_weight", 0.0)
    if comps is not None and cw > 0.0:
        margin = getattr(policy.model.config, "contrastive_margin", 0.05)
        main_v = comps.get("main", float("nan"))
        contr_v = comps.get("contrastive", float("nan"))
        pct = (contr_v / margin * 100.0) if margin > 0 else float("nan")
        print(f"  Contrastive    - main: {main_v:.4f}   contrastive: {contr_v:.4f} "
              f"({pct:.0f}% of margin {margin:.3f})   weight: {cw}")

    print("--- End Gradient Analysis ---\n")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train(
    output_dir: str,
    dataset_id="ISdept/piper_arm",
    resume_from_checkpoint: Optional[str] = None,
    batch_size: int = 16,
    training_steps: int = 300000,
    reset_lang_params: bool = False,
    gradient_checkpointing: bool = False,
    num_dit_layers: int = 16,
    vlm_capture_layers: Optional[list] = None,
    vlm_capture_mode: str = "last",
    num_register_tokens: int = 32,
    num_latent_tokens: int = 0,
    use_robot_cnn: bool = False,
    vision_input_size: int = 0,
    vision_hires_cameras: Optional[list] = None,
    text_last: bool = False,
    dit_hidden_size: int = 0,
    use_8bit_adam: bool = False,
    max_episode_index: Optional[int] = None,
    lock_joint_index: Optional[int] = None,
    contrastive_loss_weight: float = 0.1,
    contrastive_margin: float = 0.05,
    contrastive_hard_negatives: bool = False,
    vision_kv_dropout_prob: float = 0.0,
    use_chat_template: bool = True,
    val_episodes: float = 0.0,
    val_every: int = 500,
    val_seed: int = 42,
    val_max_batches: int = 8,
    chat_directive: str = "",
    use_descriptive_objects: bool = False,
    robot_encoder_tokens: int = 16,
    noise_temporal_correlation: float = 0.0,
    vision_dropout_prob: float = 0.3,
    vision_dropout_start: float = -1.0,
    vision_dropout_anneal_steps: int = 0,
    robot_cnn_cameras: Optional[list] = None,
    robot_cnn_wrist_only: bool = False,
    preprocess_in_workers: bool = False,
):
    """Train WiltechsVLA on one or more HOMOGENEOUS LeRobot datasets.

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
    checkpoint_freq = 1000
    image_transforms = get_augmentations()

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

    # ── Resolve the RobotCNN camera list (design A: specialize the trainable
    #    CNN to the WRIST view so it complements the frozen VLM instead of
    #    competing with it on the scene/color cameras). ───────────────────────
    robot_cnn_camera_keys: list[str] = []
    if robot_cnn_cameras:
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
    if robot_cnn_camera_keys:
        print(f"RobotCNN cameras (wrist-specialized): {robot_cnn_camera_keys}  "
              f"(VLM still sees all {len(camera_keys)})")
    else:
        print(f"RobotCNN cameras: ALL {camera_keys} (legacy: CNN re-encodes VLM views)")

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
                f"train_wiltechs_vla.py concatenation requires a homogeneous schema. "
                f"For mixed robots use train_community.py."
            )

    # ── Aggregate normalization stats across datasets ────────────────────
    if len(dataset_ids) == 1:
        combined_stats = ref_meta.stats
    else:
        combined_stats = aggregate_stats([metas[did].stats for did in dataset_ids])
        print(f"Aggregated normalization stats across {len(dataset_ids)} datasets.")

    # ── Training parameters ──────────────────────────────────────────────
    obs = 2
    horizon = 64
    n_action_steps = 64

    # action_dim_weights — uniform by default. piper_arm's joint 4 (index 3) is
    # mechanically locked, so pass --lock_joint_index 3 to zero its loss term.
    action_dim_weights = [1.0] * action_dim
    if lock_joint_index is not None and 0 <= lock_joint_index < action_dim:
        action_dim_weights[lock_joint_index] = 0.0
        print(f"Locking action dim {lock_joint_index} (weight=0); "
              f"action_dim_weights={action_dim_weights}")
    else:
        print(f"All {action_dim} action dims weighted equally; "
              f"action_dim_weights={action_dim_weights}")

    # ── Build config ─────────────────────────────────────────────────────
    cfg = WiltechsVLAConfig(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=obs,
        horizon=horizon,
        n_action_steps=n_action_steps,
        state_dim=state_dim,
        action_dim=action_dim,
        num_vlm_layers=num_dit_layers,
        vlm_capture_layers=list(vlm_capture_layers or []),
        vlm_capture_mode=vlm_capture_mode,
        num_register_tokens=num_register_tokens,
        use_robot_cnn=use_robot_cnn,
        dit_hidden_size=dit_hidden_size,
        num_cameras=len(camera_keys),
        cameras_for_vision_state_concat=camera_keys,
        robot_cnn_cameras=robot_cnn_camera_keys,
        vision_input_size=vision_input_size,
        vision_hires_cameras=list(vision_hires_cameras or []),
        text_first=not text_last,
        action_dim_weights=action_dim_weights,
        pos_decay_lambda=0.0,
        num_latent_tokens=num_latent_tokens,
        vlm_attends_to_expert=True,
        contrastive_loss_weight=contrastive_loss_weight,
        contrastive_margin=contrastive_margin,
        contrastive_hard_negatives=contrastive_hard_negatives,
        vision_kv_dropout_prob=vision_kv_dropout_prob,
        vision_dropout_prob=vision_dropout_prob,
        use_chat_template=use_chat_template,
        chat_directive=chat_directive,
        use_descriptive_objects=use_descriptive_objects,
        robot_encoder_tokens=robot_encoder_tokens,
        noise_temporal_correlation=noise_temporal_correlation,
    )

    # ── Model setup ──────────────────────────────────────────────────────
    if resume_from_checkpoint is not None:
        print(f"\nResuming training from checkpoint: {resume_from_checkpoint}")
        policy = WiltechsVLAPolicy(cfg)
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
                step = saved_cfg_json.get("training_step", 0)
                epoch = saved_cfg_json.get("training_epoch", 0)
                saved_total = saved_cfg_json.get("training_steps_total", 0)
                if saved_total > 0:
                    training_steps = saved_total
                print(f"Read config from {cfg_name}: step={step}, epoch={epoch}, "
                      f"training_steps_total={training_steps}")
                break
        if step == 0 and local_ckpt.name.startswith("checkpoint-"):
            step = int(local_ckpt.name.split("-")[1])
        print(f"Resuming from step {step}, epoch {epoch}")

        # Load the checkpoint onto CPU, NOT the GPU. model.safetensors holds the
        # WHOLE policy (frozen 4B VLM + DiT, ~11GB) and the freshly-built `policy`
        # already holds its own copy. Loading the checkpoint straight to CUDA
        # would keep TWO full copies on the GPU at once (~22GB) and OOM a 24GB
        # card at policy.to(device). Load on CPU, copy into the (still-CPU)
        # policy, free the checkpoint, then move the SINGLE policy to the GPU.
        ckpt_state = load_safetensors(model_file, device="cpu")
        policy.train()
        cur_state = policy.state_dict()
        filtered = {
            k: v for k, v in ckpt_state.items()
            if k in cur_state and cur_state[k].shape == v.shape
        }
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

        if reset_lang_params:
            with torch.no_grad():
                if hasattr(policy.model, "lang_attn_bias"):
                    policy.model.lang_attn_bias.zero_()
                    print("Reset lang_attn_bias to zero")
                if hasattr(policy.model, "lang_adaptor"):
                    policy.model.lang_adaptor[1].weight.fill_(1.0)
                    print("Reset lang_adaptor RMSNorm gamma to 1")

        preprocessor, postprocessor = make_pre_post_processors(
            policy.config, dataset_stats=combined_stats,
        )

        # The cosine scheduler's base LR must be the PEAK (pre-decay) value: the
        # decay is reconstructed purely by fast-forwarding scheduler.step() `step`
        # times below. The checkpoint's saved "optimizer_lr" is the ALREADY-DECAYED
        # lr, so using it as the base would double-apply the decay. Use cfg's peak.
        base_lr = cfg.optimizer_lr
        resume_warmup = saved_cfg_json.get("scheduler_warmup_steps", cfg.scheduler_warmup_steps)
        print(f"Scheduler base (peak) LR: {base_lr:.2e}  (decay rebuilt by "
              f"fast-forwarding to step {step})")

        trainable_params = [p for p in policy.model.parameters() if p.requires_grad]
        optimizer = make_optimizer(trainable_params, base_lr, cfg.optimizer_weight_decay, use_8bit_adam)
        opt_state_path = local_ckpt / "optimizer_state.pth"
        if opt_state_path.exists():
            try:
                # Load to CPU first; load_state_dict casts the Adam state to each
                # param's device lazily. Loading straight to CUDA holds the full
                # optimizer state (~GBs) on top of the model and can OOM.
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
        policy = WiltechsVLAPolicy(cfg)
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

    # DiT gradient checkpointing — recompute the trainable DiT layers in
    # backward instead of storing their activations. This is the main lever for
    # the contrastive loss, which runs a second full DiT forward; the frozen VLM
    # is unaffected (it already runs under no_grad).
    if gradient_checkpointing and hasattr(policy.model, "gradient_checkpointing_enable"):
        policy.model.gradient_checkpointing_enable()

    if isinstance(preprocessor, torch.nn.Module):
        preprocessor.to(device)

    # ── Dataset setup ────────────────────────────────────────────────────
    # Read fps from metadata (piper_arm is 30 fps; libero/community are commonly
    # 10 fps). A mismatched frame_time pushes every delta_timestamp outside
    # tolerance_s and the constructor raises. All datasets must share one fps so
    # the action horizon means the same real time everywhere.
    fps = int(getattr(ref_meta, "fps", 30) or 30)
    for did in dataset_ids[1:]:
        f2 = int(getattr(metas[did], "fps", fps) or fps)
        if f2 != fps:
            raise ValueError(
                f"Dataset '{did}' fps={f2} differs from '{dataset_ids[0]}' fps={fps}. "
                f"Resample to a common fps before mixing — the chunk horizon must "
                f"cover the same real time across datasets."
            )
    frame_time = 1 / fps
    print(f"Dataset fps: {fps} (frame_time={frame_time:.4f}s)")

    obs_temporal_window = [-i * frame_time for i in range(obs)][::-1]
    action_temporal_window = [i * frame_time for i in range(horizon)]
    delta_timestamps = {
        "observation.state": obs_temporal_window,
        "action": action_temporal_window,
        # Cameras only need the current frame.
        **{key: [0.0] for key in camera_keys},
    }
    tolerance_s = max(0.005, frame_time / 2)

    # Build each dataset, concatenate, accumulate episode boundaries in the
    # concatenated index space (optionally filtered per-dataset by max_episode_index).
    sub_datasets = []
    ep_from: list[int] = []
    ep_to: list[int] = []
    # Stratification label per episode, for the validation split below. Task
    # rather than suite: a LIBERO suite is a bundle of ~10 tasks, so covering
    # every task covers every suite, and it additionally rules out a val set that
    # samples a suite but misses the tasks in it the model is worst at.
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
    val_ep: list[int] = []
    val_indices: list[int] = []
    if val_episodes > 0:
        n_ep = len(ep_from)
        n_val = int(round(n_ep * val_episodes)) if val_episodes < 1 else int(val_episodes)
        n_val = max(1, min(n_val, n_ep - 1))
        rng = np.random.default_rng(val_seed)
        # STRATIFIED by ep_group (dataset#task). A flat draw is only stratified
        # in expectation: at 40 LIBERO tasks and n_val=34 the odds that some task
        # lands zero val episodes are high, and the val loss silently stops
        # covering it. Proportional allocation with largest-remainder, floored at
        # one per group, capped so no group loses all its episodes.
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
        val_set = set(val_ep)
        for i in val_ep:
            val_indices.extend(range(ep_from[i], ep_to[i]))
        # The train sampler must see ONLY the training episodes -- dropping the
        # val frames from the val loader alone would leave them in the training
        # stream and the split would measure nothing.
        ep_from_tr = [f for i, f in enumerate(ep_from) if i not in val_set]
        ep_to_tr = [t for i, t in enumerate(ep_to) if i not in val_set]
        empty = [k for k in keys if alloc[k] == 0]
        print(f"Stratified over {len(keys)} groups (dataset#task): "
              f"{min(alloc.values())}-{max(alloc.values())} val episodes each"
              + (f"  *** {len(empty)} group(s) got NONE -- raise --val_episodes to "
                 f"at least {len(keys)}: {empty[:5]} ***" if empty else ""))
        if no_task_col:
            print(f"  [WARN] no 'task_index' column in {no_task_col}: those datasets "
                  f"are ONE group, so the split is not stratified by task inside them.")
        # Fingerprint of the split. Comparing two runs is only valid if they held
        # out the SAME episodes; a different --max_episode_index or dataset order
        # silently reshuffles it, and two val curves over different data are not
        # comparable.
        fp = hashlib.sha1(",".join(map(str, val_ep)).encode()).hexdigest()[:8]
        print(f"Validation split: {len(val_ep)}/{n_ep} episodes held out "
              f"({len(val_indices)} frames), {len(ep_from_tr)} episodes train. "
              f"seed={val_seed} fingerprint={fp}")
        if len(val_indices) < batch_size:
            raise ValueError(
                f"--val_episodes {val_episodes} holds out {len(val_indices)} frames, "
                f"fewer than one batch ({batch_size}). Raise --val_episodes.")
    else:
        ep_from_tr, ep_to_tr = ep_from, ep_to
        print("Validation: DISABLED (--val_episodes 0). Training loss alone cannot "
              "separate a better model from a better-memorised one.")

    # Optionally move augmentation + Qwen image preprocessing into the workers.
    if preprocess_in_workers:
        dataset = VLMImagePreprocDataset(
            dataset, policy.model.processor.image_processor, camera_keys,
            augment=image_transforms,
            cam_target_sizes={k: policy.model.cam_target_size(k) for k in camera_keys},
        )
        print("Image preprocessing moved into DataLoader workers "
              "(augment + Qwen image_processor per-sample, parallel + overlapped).")

    # task_index → description. Primary source is ref_meta.tasks, which
    # LeRobotDatasetMetadata ALREADY loaded from meta/tasks.parquet during
    # construction — if that file were missing we would never have got this far.
    # It is a DataFrame INDEXED BY THE TASK STRING with a `task_index` column.
    #
    # This used to re-read the parquet by hand off `first_root`. When that read
    # produced nothing the failure was silent: the map stayed empty, no batch
    # ever got a `task_description`, and the model fell through to batch["task"]
    # — which holds the task INDEX, not the text. The result was a policy
    # training with NO instruction at all, with nothing in the logs saying so.
    task_idx_to_description: dict[int, str] = {}
    try:
        tasks_df = getattr(ref_meta, "tasks", None)
        if tasks_df is not None and "task_index" in getattr(tasks_df, "columns", []):
            # index = task text, column = index. (A "task" column, if a future
            # schema adds one, wins over the DataFrame index.)
            if "task" in tasks_df.columns:
                task_idx_to_description = {
                    int(r["task_index"]): str(r["task"]) for _, r in tasks_df.iterrows()}
            else:
                task_idx_to_description = {
                    int(r["task_index"]): str(k) for k, r in tasks_df.iterrows()}
        else:
            # Fall back to the on-disk parquet only if the metadata object did
            # not expose what we expect (schema drift across lerobot versions).
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
        # Hard stop. Continuing trains a vision-only policy that looks healthy
        # in every metric until eval, and no contrastive weight or prompt
        # rewrite can compensate for language never reaching the VLM.
        raise RuntimeError(
            f"Could not build a task_index → description map for "
            f"'{dataset_ids[0]}'. ref_meta.tasks="
            f"{type(getattr(ref_meta, 'tasks', None)).__name__}, columns="
            f"{list(getattr(getattr(ref_meta, 'tasks', None), 'columns', []))}. "
            f"Without it no instruction reaches the VLM and the run is "
            f"vision-only. Inspect ref_meta.tasks before rerunning."
        )
    if len(dataset_ids) > 1:
        print("  NOTE: task_index is dataset-LOCAL; with multiple datasets it "
              "only disambiguates if each batch also carries the task string.")

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
    # only because the MODEL moved. Capped at --val_max_batches so a pass stays a
    # rounding error against --val_every training steps.
    val_loader = None
    if val_indices:
        cap = val_max_batches * batch_size
        if len(val_indices) > cap:
            val_indices = np.random.default_rng(val_seed + 1).choice(
                val_indices, size=cap, replace=False).tolist()
            val_indices.sort()
        # augment=None: the val loss must not include augmentation noise, or it
        # measures the augmentation as much as the model.
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
    done = False
    lang_check_done = False  # one-shot: fires on first batch even when resuming

    def _prep_batch(batch, augment: bool):
        """Device move + instruction resolution + preprocessor.

        Shared by training and validation so the two cannot drift apart -- a val
        batch built even slightly differently from a train batch makes the gap
        between the two losses report the difference in PREPARATION, which is
        indistinguishable from the overfitting it is meant to detect. `augment`
        is the one deliberate difference.
        """
        for key in list(batch.keys()):
            # Keep the worker-computed VLM pixels on CPU; _encode_images moves
            # them to GPU just-in-time (transient) so they aren't resident
            # through the backward pass and don't inflate the memory peak.
            if isinstance(batch[key], torch.Tensor) and not key.startswith(_VLM_PIX_PREFIX):
                batch[key] = batch[key].to(device, non_blocking=True)

        # ── Task description handling ────────────────────────────
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
                # An index outside the map yields "" — silently vision-only
                # for that sample, so surface it rather than let it ride.
                n_unmapped = sum(1 for d in batch["task_description"] if not d)
                if n_unmapped and not lang_check_done:
                    print(f"⚠️  {n_unmapped}/{len(idx_src)} task indices are not "
                          f"in the description map (known: "
                          f"{sorted(task_idx_to_description)[:10]}). Those "
                          f"samples train with an EMPTY instruction.")

        if augment:
            # Image augmentation: in-loop only when the workers aren't doing it.
            if not preprocess_in_workers:
                present_cams = [c for c in camera_keys if c in batch]
                batch = apply_image_augmentations(batch, present_cams, image_transforms)
            if "observation.state" in batch:
                batch = apply_joint_augmentations(batch, "observation.state")

        # Hold out the worker-computed VLM pixels so the LeRobot preprocessor
        # (normalizer / device / add-batch-dim steps) never touches them.
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
        policy.eval()   # kills dropout, vision-KV dropout and the contrastive
                        # branch -- all gated on self.training
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

            # One-time language-reaches-model sanity check (step 0). The model's
            # _encode_language reads task_description, falling back to task; if
            # both are absent/empty here, it silently runs vision-only and no
            # contrastive weight can fix that. This logs AFTER the preprocessor,
            # which historically stripped task_description (the 2026-05 goal-9%
            # bug). Verify language is present BEFORE blaming contrastive_loss_weight.
            if not lang_check_done:
                lang_check_done = True
                descs = batch.get("task_description")
                if descs is None:
                    descs = batch.get("task")
                if descs is None:
                    print("⚠️  LANG CHECK: neither 'task_description' nor 'task' in "
                          "batch after preprocessor — model will train VISION-ONLY.")
                elif not (isinstance(descs, (list, tuple))
                          and all(isinstance(d, str) for d in descs)):
                    # A (B,) tensor of task INDICES lands here. It used to slip
                    # through: iterating a 1-D tensor yields 0-dim tensors whose
                    # truth test is legal, so `sum(1 for d in descs if d)` counted
                    # happily and the check reported success on integers.
                    print(f"\n--- LANG CHECK (step 0) ---")
                    print(f"⚠️  'task'/'task_description' is {type(descs).__name__}"
                          + (f" of shape {tuple(descs.shape)}" if hasattr(descs, 'shape') else "")
                          + f", NOT strings: {list(descs[:2])}")
                    print("    This is the task INDEX, not the text — tasks.parquet "
                          "did not load, so NO instruction reaches the VLM.")
                    print("    Check the 'Loaded N task descriptions' line above; if "
                          "it is missing, meta/tasks.parquet was not found.")
                    print("--- end LANG CHECK ---\n")
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
                        print("⚠️  ALL descriptions empty — language is NOT reaching "
                              "the model. Fix data flow before tuning contrastive loss.")
                    print("--- end LANG CHECK ---\n")

            # Arm the attention-mass diagnostic on the same cadence as
            # gradient analysis. The model self-disarms after one capture.
            if step % progress_update_freq == 0:
                policy.model._capture_attention_stats = True

            # Vision-dropout curriculum: anneal RobotCNN token dropout from a high
            # start (no/weak CNN shortcut → forces Qwen-vision grounding first)
            # linearly down to vision_dropout_prob (the floor) over
            # vision_dropout_anneal_steps. <0 start disables (constant dropout).
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
            # validation forward overwrites every _last_* diagnostic on the model
            # (attention stats, loss components), so running it earlier would
            # make the gradient report describe the HELD-OUT batch while
            # labelling it as the training step.
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
        description="Train WiltechsVLA on one or more homogeneous LeRobot datasets.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints")
    parser.add_argument("--dataset_id", type=str, nargs="+", default=["ISdept/piper_arm"],
                        help="One or more LeRobot dataset ids. Multiple are concatenated and "
                             "must share a homogeneous schema (same robot/cameras/dims/fps); "
                             "their normalization stats are aggregated.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from a checkpoint")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (Qwen3-VL-4B backbone is memory-heavy; 8-24).")
    parser.add_argument("--training_steps", type=int, default=300000, help="Total training steps")
    parser.add_argument("--reset_lang_params", action="store_true",
                        help="Reset language conditioning params after loading checkpoint")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Recompute DiT layer activations in backward to save GPU memory "
                             "(trades ~extra forward compute; frozen VLM is unaffected). "
                             "Recommended when using the contrastive loss, which runs a 2nd DiT forward.")
    parser.add_argument("--num_dit_layers", type=int, default=16,
                        help="DiT decoder depth = number of VLM layers whose KV the DiT "
                             "cross-attends to, one per DiT layer. 36 = every layer, the same "
                             "LAYER COUNT as WiltechsMoE (4 experts x 9 layers, all of which run "
                             "every forward) but as one sequential stack -- to match its "
                             "PARAMETERS too you also need --dit_hidden_size 1280 (its 92% run's "
                             "width; 36L at 640 is only 1/3 the params). Biggest memory lever; "
                             "pair with --dit_hidden_size to control per-layer width. Must be <= 36.")
    parser.add_argument("--vlm_capture_mode", type=str, default="last",
                        choices=["last", "spread"],
                        help="Which VLM layers the DiT reads when --vlm_capture_layers is empty. "
                             "'last' (default) takes the deepest N (16 -> layers 20..35); 'spread' "
                             "spaces them over the full depth. A real trade: the VLM runs all 36 "
                             "either way, so 'last' discards the shallow layers' KV for free, but "
                             "gives every DiT layer a fully fused representation. Measured on the "
                             "MoE variant, cross-attn language share is ~44% at VLM layer 8 vs "
                             "~91% at layer 35, and under text_first the language K/V never sees "
                             "the image -- so 'last' reads the least visually grounded band.")
    parser.add_argument("--vlm_capture_layers", type=int, nargs="+", default=[],
                        help="Explicit VLM layer indices for the DiT to read, overriding "
                             "--vlm_capture_mode. Must have exactly --num_dit_layers entries.")
    parser.add_argument("--num_register_tokens", type=int, default=32,
                        help="Learned register tokens placed between the state and the actions: "
                             "[state(1), register(R), action(H)]. Unlike --num_latent_tokens these "
                             "hold no observation at init and are rewritten at EVERY DiT layer "
                             "(self-attention, plus cross-attention, which has no causal mask and "
                             "covers all DiT positions). 0 disables.")
    parser.add_argument("--num_latent_tokens", type=int, default=0,
                        help="Learned-query Q-Former 'thought' tokens prepended to the DiT "
                             "sequence, distilled ONCE from the DEEPEST captured VLM layer. "
                             "Superseded by --num_register_tokens; 0 (default) disables.")
    parser.add_argument("--use_robot_cnn", action="store_true",
                        help="Re-enable the parallel ResNet-18 visual encoder (OFF by default). "
                             "This is the single largest measured lever in the MoE variant of this "
                             "architecture: removing it took libero_spatial task 0 from 92%% to "
                             "58%%, and the residual failures were grasp precision, not object "
                             "selection. With it off the frozen VLM's ~32 px/token grid is the "
                             "only visual input in the model.")
    parser.add_argument("--vision_input_size", type=int, default=0,
                        help="Square side length (px) fed to the Qwen image processor. 0 = the "
                             "processor's smart-resize default. Qwen3-VL merges 32x32 native px "
                             "into one token, so 256->8x8=64 tok/cam and 512->16x16=256 tok/cam. "
                             "L_vlm is also every DiT layer's cross-attn K/V length, so the cost "
                             "multiplies by --num_dit_layers. Check the '[wiltechs_vla] vision "
                             "grid' startup line to confirm it took effect.")
    parser.add_argument("--vision_hires_cameras", type=str, nargs="+", default=[],
                        help="Cameras that get --vision_input_size; empty = all of them. Naming "
                             "just the third-person view roughly halves the added cost, since the "
                             "relations that need the resolution are not resolvable at the wrist "
                             "camera's scale anyway.")
    parser.add_argument("--text_last", action="store_true",
                        help="Legacy [images, instruction] VLM layout. The VLM is causal, so this "
                             "leaves every vision K/V the DiT reads LANGUAGE-BLIND and the model "
                             "tends to use the instruction as a coarse location prior rather than "
                             "an object selector. Default is text-first.")
    parser.add_argument("--dit_hidden_size", type=int, default=0,
                        help="DiT decoder width. 0 (default) = match the VLM hidden size (2560). "
                             "Set a smaller multiple of the VLM head_dim (e.g. 1280) to shrink the "
                             "DiT self-attn/FFN/adaLN (~quadratic param savings); cross-attention is "
                             "bridged back up to the frozen VLM KV. Lower = less capacity.")
    parser.add_argument("--use_8bit_adam", action="store_true",
                        help="Use bitsandbytes 8-bit Adam (int8 optimizer state) instead of fp32 Adam, "
                             "cutting optimizer memory ~4x. Requires `pip install bitsandbytes` + CUDA.")
    parser.add_argument("--val_episodes", type=float, default=0.0,
                        help="Held-out EPISODES for validation: <1 a fraction, >=1 a count, "
                             "0 (default) disables. The split is on EPISODES, never frames -- "
                             "consecutive frames are 0.1s apart and near copies, so a frame "
                             "split leaves a sample's own neighbours on the other side and the "
                             "val loss tracks the train loss forever, detecting nothing. "
                             "STRATIFIED by dataset#task, so every task is covered; the value "
                             "must resolve to at least as many episodes as there are tasks or "
                             "some get none (the startup line says so if it happens).")
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
                        help="Filter to episodes with index <= this value "
                             "(piper_arm holdout convention; omit for full dataset).")
    parser.add_argument("--lock_joint_index", type=int, default=None,
                        help="Action dim with weight 0 (e.g. piper_arm joint 4 = index 3 is "
                             "mechanically locked). Omit / pass -1 to weight all dims.")
    parser.add_argument("--contrastive_loss_weight", type=float, default=0.1,
                        help="Weight for the language-permute contrastive loss (default: 0.1).")
    parser.add_argument("--contrastive_margin", type=float, default=0.05,
                        help="Hinge margin on MSE between v_t and v_wrong (default: 0.05).")
    parser.add_argument("--contrastive_hard_negatives", action="store_true",
                        help="Pair each sample with its hardest in-batch negative (most word "
                             "overlap, different instruction) instead of a random one, so the "
                             "contrastive hinge pressures fine-grained object grounding (the "
                             "confusable minimal pairs that fail at eval) rather than trivially-"
                             "different tasks. Expect the reported contrastive value to spike "
                             "when first enabled, then decline. Off = legacy random pairing.")
    parser.add_argument("--vision_kv_dropout_prob", type=float, default=0.0,
                        help="Training-time dropout on the VLM VISION positions of the DiT "
                             "cross-attn memory (language is never dropped). Weakens the "
                             "~25:1 vision:language shortcut to force language reliance. "
                             "Try 0.25-0.4; 0 disables (default).")
    parser.add_argument("--vision_dropout_prob", type=float, default=0.3,
                        help="RobotCNN token dropout (regularizer). ALSO the FLOOR/end value of "
                             "the curriculum when --vision_dropout_start is set. Default 0.3.")
    parser.add_argument("--vision_dropout_start", type=float, default=-1.0,
                        help="CURRICULUM: starting (high) RobotCNN dropout at step 0, annealed "
                             "linearly down to --vision_dropout_prob over "
                             "--vision_dropout_anneal_steps. <0 disables the schedule (constant "
                             "--vision_dropout_prob). E.g. 0.9 forces the model to ground on Qwen "
                             "vision first (no CNN shortcut), then reintroduces the CNN for spatial. "
                             "Keep the floor (--vision_dropout_prob) ≥0.5 to avoid re-shortcutting.")
    parser.add_argument("--vision_dropout_anneal_steps", type=int, default=0,
                        help="Steps to anneal RobotCNN dropout from --vision_dropout_start to "
                             "--vision_dropout_prob. 0 disables the schedule. E.g. 20000.")
    parser.add_argument("--no_chat_template", dest="use_chat_template",
                        action="store_false", default=True,
                        help="Disable the Qwen ChatML wrapping (<|im_start|>user + "
                             "<|vision_start|>[cam]<|vision_end|> per camera + task + "
                             "<|im_end|> + assistant header) and feed the raw [vision|task] "
                             "concat instead. The template is ON by default: the raw concat "
                             "is a token sequence the instruct-tuned VLM never saw, and every "
                             "KV the DiT reads is computed from it. Required to reproduce a "
                             "checkpoint trained before this became the default -- the "
                             "template moves the KV geometry its ca_q was fit to.")
    parser.add_argument("--chat_directive", type=str, default="",
                        help="Optional short directive prepended to the task inside the user "
                             "turn (only with --use_chat_template), e.g. 'Identify the objects "
                             "mentioned in the instruction and where they are, then perform:'.")
    parser.add_argument("--use_descriptive_objects", action="store_true",
                        help="Rewrite ambiguous LIBERO object/region names into visually-"
                             "groundable descriptions (e.g. 'alphabet soup' -> 'blue can of "
                             "alphabet soup') via task_rewrites.py before the VLM sees them. "
                             "Persisted in the saved config so eval inherits it automatically. "
                             "Off = legacy phrasing.")
    parser.add_argument("--robot_encoder_tokens", type=int, default=16,
                        help="Robot CNN tokens per camera. Must be a perfect square "
                             "(grid side = sqrt). Default: 16 (4x4).")
    parser.add_argument("--robot_cnn_cameras", type=str, nargs="+", default=None,
                        help="Explicit camera key(s) the trainable RobotCNN ingests "
                             "(must be among the detected cameras). Default: all of them "
                             "(legacy). Use this OR --robot_cnn_wrist_only to specialize "
                             "the CNN to the wrist view so it complements the frozen VLM.")
    parser.add_argument("--robot_cnn_wrist_only", action="store_true",
                        help="Restrict the RobotCNN to the auto-detected WRIST/gripper "
                             "camera (matches image2/wrist/gripper/eye_in_hand/hand), "
                             "leaving scene/color/spatial grounding to the frozen VLM. "
                             "Design A: stops the trainable CNN from out-competing the "
                             "VLM on the agentview where color-binding lives (fixes T0 "
                             "structurally). Ignored if --robot_cnn_cameras is given.")
    parser.add_argument("--noise_temporal_correlation", type=float, default=0.0,
                        help="AR(1) coefficient correlating the flow-matching source noise "
                             "along the action horizon (0=white; ~0.9=temporally smooth).")
    parser.add_argument("--preprocess_in_workers", action="store_true",
                        help="Run image augmentation + the Qwen image_processor inside the "
                             "DataLoader workers (parallel, overlapped with GPU) instead of on "
                             "the critical path in _encode_images. Speeds up training when the "
                             "per-step image preprocessing is a bottleneck. Inference is "
                             "unaffected (the model preprocesses raw frames itself).")
    args = parser.parse_args()

    _v = args.robot_encoder_tokens
    if int(_v ** 0.5) ** 2 != _v:
        parser.error(f"--robot_encoder_tokens must be a perfect square, got {_v}")
    # Argparse can't express None for an int, so use -1 sentinel.
    if args.lock_joint_index is not None and args.lock_joint_index < 0:
        args.lock_joint_index = None

    train(**vars(args))
