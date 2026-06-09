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
# Gradient analysis
# ---------------------------------------------------------------------------
def _log_gradient_analysis(policy, step: int) -> None:
    print(f"\n--- Gradient Analysis at Step {step} ---")

    def _grad_stats(prefix: str):
        total, count = 0.0, 0
        for name, param in policy.model.named_parameters():
            if param.requires_grad and prefix in name and param.grad is not None:
                total += param.grad.abs().mean().item() * param.numel()
                count += param.numel()
        return (total / count, count) if count > 0 else (None, 0)

    for label, prefix in [
        ("Vision",         "visual"),
        ("State Enc",      "state_encoder"),
        ("Robot CNN",      "robot_visual_encoder"),
        ("Expert Layers",  "expert_layers"),
        ("DiT Layers",     "dit_layers"),
        ("Action In/Out",  "action_"),
        ("Final Norm",     "final_norm"),
        ("Latent Gen",     "latent_generator"),
        ("Lang Adaptor",   "lang_adaptor"),
    ]:
        grad, n = _grad_stats(prefix)
        if grad is not None:
            print(f"  {label:14s} - Avg Abs Grad: {grad:.6f} ({n} params)")
        else:
            print(f"  {label:14s} - no grad")

    if hasattr(policy.model, "latent_generator"):
        gen = policy.model.latent_generator
        w_norm_sq = 0.0
        g_norm_sq = 0.0
        for p in gen.parameters():
            w_norm_sq += p.detach().norm().item() ** 2
            if p.grad is not None:
                g_norm_sq += p.grad.norm().item() ** 2
        out_layer = gen[-1]
        out_w_norm = out_layer.weight.detach().norm().item()
        print(f"  Latent gen     - weight_norm: {w_norm_sq ** 0.5:.4e}   "
              f"grad_norm: {g_norm_sq ** 0.5:.4e}   out_layer_w: {out_w_norm:.4e}")

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
    dit_hidden_size: int = 0,
    use_8bit_adam: bool = False,
    max_episode_index: Optional[int] = None,
    lock_joint_index: Optional[int] = None,
    contrastive_loss_weight: float = 0.1,
    contrastive_margin: float = 0.05,
    robot_encoder_tokens: int = 16,
    noise_temporal_correlation: float = 0.0,
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
        dit_hidden_size=dit_hidden_size,
        num_cameras=len(camera_keys),
        cameras_for_vision_state_concat=camera_keys,
        action_dim_weights=action_dim_weights,
        pos_decay_lambda=0.0,
        num_latent_tokens=8,
        vlm_attends_to_expert=True,
        contrastive_loss_weight=contrastive_loss_weight,
        contrastive_margin=contrastive_margin,
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

        ckpt_state = load_safetensors(model_file, device=str(device))
        policy.train()
        policy.to(device)
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
        print(f"Loaded {len(filtered)}/{len(cur_state)} model keys")

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
                optimizer.load_state_dict(torch.load(opt_state_path, map_location=device))
                for pg in optimizer.param_groups:
                    pg["lr"] = base_lr
                    pg["initial_lr"] = base_lr
                print(f"Optimizer state loaded. Scheduler base LR set to peak {base_lr:.2e}")
            except ValueError as e:
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

    # task_index → description (from the first dataset's tasks.parquet). Batches
    # carry the per-frame "task" string directly (preferred); for multi-dataset
    # task_index is dataset-local, so we rely on batch["task"].
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
    done = False
    prog_bar = tqdm(total=training_steps, desc="Training Progress", initial=step)

    while not done:
        epoch += 1
        for batch in dataloader:
            for key in list(batch.keys()):
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)

            # Task description handling
            if "task" in batch and isinstance(batch["task"], (list, tuple)):
                batch["task_description"] = batch["task"]
            elif task_idx_to_description and "task_index" in batch:
                task_indices = batch["task_index"]
                if isinstance(task_indices, torch.Tensor) and task_indices.dim() > 1:
                    task_indices = task_indices[:, 0]
                batch["task_description"] = [
                    task_idx_to_description.get(int(ti), "") for ti in task_indices
                ]

            present_cams = [c for c in camera_keys if c in batch]
            batch = apply_image_augmentations(batch, present_cams, image_transforms)

            if "observation.state" in batch:
                batch = apply_joint_augmentations(batch, "observation.state")

            batch = preprocessor(batch)

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
                        help="DiT decoder depth = number of trailing VLM layers whose KV the DiT "
                             "cross-attends to. Each layer is ~180M params, so this is the BIGGEST "
                             "memory lever: 16 (default) ~2.9B trainable params; drop to 6-8 to fit "
                             "a 22-24GB GPU. Lower = less capacity. Must be <= the VLM's layer count (36).")
    parser.add_argument("--dit_hidden_size", type=int, default=0,
                        help="DiT decoder width. 0 (default) = match the VLM hidden size (2560). "
                             "Set a smaller multiple of the VLM head_dim (e.g. 1280) to shrink the "
                             "DiT self-attn/FFN/adaLN (~quadratic param savings); cross-attention is "
                             "bridged back up to the frozen VLM KV. Lower = less capacity.")
    parser.add_argument("--use_8bit_adam", action="store_true",
                        help="Use bitsandbytes 8-bit Adam (int8 optimizer state) instead of fp32 Adam, "
                             "cutting optimizer memory ~4x. Requires `pip install bitsandbytes` + CUDA.")
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
    parser.add_argument("--robot_encoder_tokens", type=int, default=16,
                        help="Robot CNN tokens per camera. Must be a perfect square "
                             "(grid side = sqrt). Default: 16 (4x4).")
    parser.add_argument("--noise_temporal_correlation", type=float, default=0.0,
                        help="AR(1) coefficient correlating the flow-matching source noise "
                             "along the action horizon (0=white; ~0.9=temporally smooth).")
    args = parser.parse_args()

    _v = args.robot_encoder_tokens
    if int(_v ** 0.5) ** 2 != _v:
        parser.error(f"--robot_encoder_tokens must be a perfect square, got {_v}")
    # Argparse can't express None for an int, so use -1 sentinel.
    if args.lock_joint_index is not None and args.lock_joint_index < 0:
        args.lock_joint_index = None

    train(**vars(args))
