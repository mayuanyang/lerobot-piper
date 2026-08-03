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
import argparse
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
        mw = getattr(policy.model, "_last_router_max_w", None)
        ent = getattr(policy.model, "_last_router_entropy", None)
        if mw is not None and ent is not None:
            E = int(usage_cpu.numel())
            import math
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
    gradient_checkpointing: bool = False,
    num_experts: int = 4,
    expert_num_layers: int = 4,
    dit_hidden_size: int = 1280,
    use_8bit_adam: bool = False,
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
    robot_encoder_tokens: int = 16,
    noise_temporal_correlation: float = 0.0,
    vision_dropout_prob: float = 0.3,
    vision_dropout_start: float = -1.0,
    vision_dropout_anneal_steps: int = 0,
    robot_cnn_cameras: Optional[list] = None,
    robot_cnn_wrist_only: bool = False,
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
    checkpoint_freq = 1000
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

    # ── Resolve the RobotCNN camera list ────────────────────────────────
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
    obs = 2
    horizon = 64
    n_action_steps = 64

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
        num_cameras=len(camera_keys),
        cameras_for_vision_state_concat=camera_keys,
        robot_cnn_cameras=robot_cnn_camera_keys,
        action_dim_weights=action_dim_weights,
        pos_decay_lambda=0.0,
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

        preprocessor, postprocessor = make_pre_post_processors(
            policy.config, dataset_stats=combined_stats,
        )

        base_lr = cfg.optimizer_lr
        resume_warmup = saved_cfg_json.get("scheduler_warmup_steps", cfg.scheduler_warmup_steps)
        print(f"Scheduler base (peak) LR: {base_lr:.2e}  (decay rebuilt by "
              f"fast-forwarding to step {step})")

        trainable_params = [p for p in policy.model.parameters() if p.requires_grad]
        optimizer = make_optimizer(trainable_params, base_lr, cfg.optimizer_weight_decay, use_8bit_adam)
        opt_state_path = local_ckpt / "optimizer_state.pth"
        if opt_state_path.exists():
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
        changes = np.where(np.diff(ep_ids) != 0)[0] + 1
        starts = np.concatenate([[0], changes])
        ends = np.concatenate([changes, [len(ep_ids)]])
        kept = 0
        for s, e in zip(starts, ends):
            if max_episode_index is not None and int(ep_ids[s]) > max_episode_index:
                continue
            ep_from.append(offset + int(s))
            ep_to.append(offset + int(e))
            kept += 1
        suffix = f" (<= ep {max_episode_index})" if max_episode_index is not None else ""
        print(f"  {did}: {len(ds)} frames, {kept} episodes{suffix}")
        sub_datasets.append(ds)
        offset += len(ds)

    dataset = ConcatDataset(sub_datasets)
    print(f"Combined dataset: {len(dataset)} frames, {len(ep_from)} episodes "
          f"across {len(sub_datasets)} dataset(s)")

    if preprocess_in_workers:
        dataset = VLMImagePreprocDataset(
            dataset, policy.model.processor.image_processor, camera_keys,
            augment=image_transforms,
            cam_target_sizes={k: policy.model.cam_target_size(k) for k in camera_keys},
        )
        print("Image preprocessing moved into DataLoader workers "
              "(augment + Qwen image_processor per-sample, parallel + overlapped).")

    task_idx_to_description: dict[int, str] = {}
    try:
        tasks_parquet_path = first_root / "meta" / "tasks.parquet"
        if tasks_parquet_path.exists():
            tasks_df = pd.read_parquet(tasks_parquet_path)
            if "task_index" in tasks_df.columns:
                if "task" in tasks_df.columns:
                    task_idx_to_description = {
                        int(row["task_index"]): str(row["task"])
                        for _, row in tasks_df.iterrows()
                    }
                else:
                    task_idx_to_description = {
                        int(row["task_index"]): str(idx)
                        for idx, row in tasks_df.iterrows()
                    }
            print(f"Loaded {len(task_idx_to_description)} task descriptions from tasks.parquet")
        else:
            print("tasks.parquet not found; task_description will not be added to batches.")
    except Exception as e:
        print(f"Warning: could not load tasks.parquet: {e}")

    sampler = EpisodeAwareSampler(
        dataset_from_indices=ep_from,
        dataset_to_indices=ep_to,
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

    # ── Training loop ────────────────────────────────────────────────────
    print(f"\nStarting training loop ({training_steps} steps, batch_size={batch_size})...")
    print(f"MoE config: {num_experts} experts x {expert_num_layers} layers each "
          f"(dit_hidden={dit_hidden_size})")
    done = False
    lang_check_done = False
    prog_bar = tqdm(total=training_steps, desc="Training Progress", initial=step)

    while not done:
        epoch += 1
        for batch in dataloader:
            for key in list(batch.keys()):
                if isinstance(batch[key], torch.Tensor) and not key.startswith(_VLM_PIX_PREFIX):
                    batch[key] = batch[key].to(device, non_blocking=True)

            if "task" in batch and isinstance(batch["task"], (list, tuple)):
                batch["task_description"] = batch["task"]
            elif task_idx_to_description and "task_index" in batch:
                task_indices = batch["task_index"]
                if isinstance(task_indices, torch.Tensor) and task_indices.dim() > 1:
                    task_indices = task_indices[:, 0]
                batch["task_description"] = [
                    task_idx_to_description.get(int(ti), "") for ti in task_indices
                ]

            if not preprocess_in_workers:
                present_cams = [c for c in camera_keys if c in batch]
                batch = apply_image_augmentations(batch, present_cams, image_transforms)

            if "observation.state" in batch:
                batch = apply_joint_augmentations(batch, "observation.state")

            vlm_pix = {k: batch.pop(k) for k in list(batch) if k.startswith(_VLM_PIX_PREFIX)}
            batch = preprocessor(batch)
            batch.update(vlm_pix)

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

            autocast_ctx = (
                torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                if device.type == "cuda"
                else torch.autocast(device_type="cpu", enabled=False)
            )
            with autocast_ctx:
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
    parser.add_argument("--robot_cnn_cameras", type=str, nargs="+", default=None,
                        help="Explicit camera key(s) the trainable RobotCNN ingests.")
    parser.add_argument("--robot_cnn_wrist_only", action="store_true",
                        help="Restrict the RobotCNN to the auto-detected WRIST/gripper camera.")
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
    parser.add_argument("--image_aug_scale", type=float, default=0.0,
                        help="Random image scale jitter (+-this). Default 0, same reasoning as "
                             "--image_aug_translate. 0.05 restores the previous behaviour.")
    args = parser.parse_args()

    _v = args.robot_encoder_tokens
    if int(_v ** 0.5) ** 2 != _v:
        parser.error(f"--robot_encoder_tokens must be a perfect square, got {_v}")
    if args.lock_joint_index is not None and args.lock_joint_index < 0:
        args.lock_joint_index = None

    # Validate MoE layer count
    total_capture = args.num_experts * args.expert_num_layers
    if total_capture > 36:
        parser.error(f"num_experts ({args.num_experts}) x expert_num_layers ({args.expert_num_layers}) "
                     f"= {total_capture} > 36 (Qwen3-VL-4B layer count). Reduce one or both.")
    if args.router_top_k > args.num_experts:
        parser.error(f"--router_top_k ({args.router_top_k}) cannot exceed --num_experts ({args.num_experts}).")

    train(**vars(args))