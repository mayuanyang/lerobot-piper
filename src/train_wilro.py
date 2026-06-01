from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
import huggingface_hub
from safetensors.torch import load_file as load_safetensors
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import dataset_to_policy_features
import numpy as np
from torch.utils.data import Subset

# Wilro-specific components
from models.wilro.wilro_config import WilroConfig
from models.wilro.wilro_policy import WilroPolicy
from models.wilro.processor_wilro import make_pre_post_processors

from torchvision.transforms import v2
from transformers import get_cosine_schedule_with_warmup


# Detect the best available device
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
# Augmentation helpers (same recipe as train_transformer.py)
# ---------------------------------------------------------------------------
def get_augmentations():
    """Image augmentation transform shared across all cameras of a sample."""
    return v2.Compose([
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    ])


def apply_joint_augmentations(batch):
    """Add small Gaussian noise to observation.state (50% probability)."""
    if torch.rand(1).item() > 0.5:
        if "observation.state" in batch:
            noise = torch.randn_like(batch["observation.state"]) * 0.01
            batch["observation.state"] = batch["observation.state"] + noise
    return batch


def apply_image_augmentations(batch, camera_keys, transform):
    """Apply the same random color jitter to all cameras within each sample.

    For each sample in the batch, all camera images are stacked into a single
    tensor and passed through the transform in one call. torchvision v2 samples
    random parameters once per forward() call and applies them identically to
    every image in the tensor — so front/gripper/right cameras always receive
    the same brightness/contrast/saturation/hue shift, keeping cross-camera
    color consistency.

    Handles both (C, H, W) and (T, C, H, W) camera tensors.
    """
    present_keys = [k for k in camera_keys if k in batch and isinstance(batch[k], torch.Tensor)]
    if not present_keys:
        return batch

    B = batch[present_keys[0]].shape[0]
    for b in range(B):
        sample_img = batch[present_keys[0]][b]
        has_time_dim = sample_img.dim() == 4
        if has_time_dim:
            T = sample_img.shape[0]
            stacked = torch.cat([batch[k][b] for k in present_keys], dim=0)
            stacked_aug = transform(stacked)
            for i, k in enumerate(present_keys):
                batch[k][b] = stacked_aug[i * T:(i + 1) * T]
        else:
            stacked = torch.stack([batch[k][b] for k in present_keys], dim=0)
            stacked_aug = transform(stacked)
            for i, k in enumerate(present_keys):
                batch[k][b] = stacked_aug[i]
    return batch


# ---------------------------------------------------------------------------
# Gradient analysis tailored to wilro components
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
        ("Vision (frozen)",  "vision_model"),
        ("Connector (frzn)", "connector"),
        ("Text (frozen)",    "text_model"),
        ("State Enc",        "state_encoder"),
        ("Robot CNN",        "robot_visual_encoder"),
        ("DiT layers",       "dit_layers"),
        ("Action In/Out",    "action_"),
        ("Sink token",       "sink_token"),
        ("Final Norm",       "final_norm"),
        ("Time MLP",         "time_embedder"),
        ("Latent Gen",       "latent_generator"),
    ]:
        grad, n = _grad_stats(prefix)
        if grad is not None:
            print(f"  {label:18s} - Avg Abs Grad: {grad:.6f} ({n} params)")

    stats = getattr(policy.model, "_last_attention_stats", None)
    if stats:
        # Match DiT sequence order: [SINK, latent, state, prefix, robot, action]
        order = ["sink", "latent", "state", "prefix", "robot", "action"]
        ordered = [(k, stats[k]) for k in order if k in stats]
        cells = "  ".join(f"{k}={v*100:5.1f}%" for k, v in ordered)
        print(f"  Action→ self-attn : {cells}    (last DiT layer)")

    x_stats = getattr(policy.model, "_last_cross_attention_stats", None)
    if x_stats:
        order = ["vision", "language"]
        ordered = [(k, x_stats[k]) for k in order if k in x_stats]
        cells = "  ".join(f"{k}={v*100:5.1f}%" for k, v in ordered)
        print(f"  Action→ x-attn    : {cells}    (cross-attn to VLM KV)")

    comps = getattr(policy.model, "_last_loss_components", None)
    cw = getattr(policy.model.config, "contrastive_loss_weight", 0.0)
    if comps is not None and cw > 0.0:
        margin = getattr(policy.model.config, "contrastive_margin", 0.05)
        main_v = comps.get("main", float("nan"))
        contr_v = comps.get("contrastive", float("nan"))
        pct = (contr_v / margin * 100.0) if margin > 0 else float("nan")
        print(f"  Contrastive       - main: {main_v:.4f}   contrastive: {contr_v:.4f} "
              f"({pct:.0f}% of margin {margin:.3f})   weight: {cw}")

    print("--- End Gradient Analysis ---\n")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train(output_dir, dataset_id="ISdept/piper_arm", resume_from_checkpoint=None,
          gradient_checkpointing=False, max_episode_index=None, batch_size=64,
          contrastive_loss_weight=0.1, contrastive_margin=0.05,
          lock_joint_index: int | None = 3):
    """Train the Wilro (SmolVLM2 KV-cache → DiT) flow matching model."""
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    training_steps = 200000
    progress_update_freq = 200
    checkpoint_freq = 1000
    image_transforms = get_augmentations()

    # Load dataset metadata
    dataset_metadata = LeRobotDatasetMetadata(dataset_id, force_cache_sync=True, revision="main")
    features = dataset_to_policy_features(dataset_metadata.features)
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

    # Training parameters — match train_transformer.py for like-for-like comparison
    obs = 2
    horizon = 64
    n_action_steps = 64

    # Build action_dim_weights — uniform by default. piper_arm's joint 4
    # (index 3) is always 0, so for that dataset pass --lock_joint_index 3
    # (the default) to zero out its loss contribution. For LIBERO / other
    # full-DOF robots, pass --lock_joint_index "" (None) to weight all dims.
    action_dim_weights = [1.0] * action_dim
    if lock_joint_index is not None and 0 <= lock_joint_index < action_dim:
        action_dim_weights[lock_joint_index] = 0.0
        print(f"Locking action dim {lock_joint_index} (weight=0); "
              f"action_dim_weights={action_dim_weights}")
    else:
        print(f"All {action_dim} action dims weighted equally; "
              f"action_dim_weights={action_dim_weights}")

    # Build wilro config
    cfg = WilroConfig(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=obs,
        horizon=horizon,
        n_action_steps=n_action_steps,
        state_dim=state_dim,
        action_dim=action_dim,
        num_vlm_layers=16,  # DiT depth = trailing N VLM layers consumed
        num_cameras=len(camera_keys),
        cameras_for_vision_state_concat=camera_keys,
        action_dim_weights=action_dim_weights,
        # n_action_steps == horizon → no exponential decay needed.
        pos_decay_lambda=0.0,
        contrastive_loss_weight=contrastive_loss_weight,
        contrastive_margin=contrastive_margin,
    )

    # Model + checkpoint loading
    if resume_from_checkpoint is not None:
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        policy = WilroPolicy(cfg)

        ckpt_path = Path(resume_from_checkpoint)
        if ckpt_path.exists():
            local_ckpt_path = ckpt_path
            print(f"Using local checkpoint: {local_ckpt_path}")
        else:
            print(f"Local path not found, downloading from HuggingFace Hub: {resume_from_checkpoint}")
            local_ckpt_path = Path(huggingface_hub.snapshot_download(resume_from_checkpoint))

        model_file = local_ckpt_path / "model.safetensors"
        if not model_file.exists():
            candidates = list(local_ckpt_path.glob("*.safetensors"))
            if not candidates:
                raise FileNotFoundError(f"No .safetensors file found in {local_ckpt_path}")
            model_file = candidates[0]

        import json
        step, epoch = 0, 0
        saved_cfg_json = {}
        for config_name in ("config.json", "pretrained_config.json"):
            config_file = local_ckpt_path / config_name
            if config_file.exists():
                with open(config_file) as f:
                    saved_cfg_json = json.load(f)
                step = saved_cfg_json.get("training_step", 0)
                epoch = saved_cfg_json.get("training_epoch", 0)
                saved_total = saved_cfg_json.get("training_steps_total", 0)
                if saved_total > 0:
                    training_steps = saved_total
                print(f"Read config from {config_file.name}: step={step}, epoch={epoch}, training_steps_total={training_steps}")
                break
        if step == 0 and local_ckpt_path.name.startswith("checkpoint-"):
            step = int(local_ckpt_path.name.split("-")[1])
        print(f"Resuming from step {step}, epoch {epoch}")

        print(f"Loading weights from: {model_file}")
        ckpt_state = load_safetensors(model_file, device=str(device))

        policy.train()
        policy.to(device)
        cur_state = policy.state_dict()
        filtered = {k: v for k, v in ckpt_state.items() if k in cur_state and cur_state[k].shape == v.shape}

        skipped_ckpt = [k for k in ckpt_state if k not in filtered]
        missing_from_ckpt = [k for k in cur_state if k not in ckpt_state]
        if skipped_ckpt:
            print(f"Skipped {len(skipped_ckpt)} checkpoint keys (shape mismatch / removed): {skipped_ckpt[:10]}")
        if missing_from_ckpt:
            print(f"Missing {len(missing_from_ckpt)} keys not in checkpoint (will use init values): {missing_from_ckpt[:10]}")
        policy.load_state_dict(filtered, strict=False)
        print(f"Loaded {len(filtered)}/{len(cur_state)} model keys from checkpoint ({len(ckpt_state)} keys in file)")

        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            dataset_stats=dataset_metadata.stats,
        )

        resume_lr = saved_cfg_json.get("optimizer_lr", cfg.optimizer_lr)
        resume_warmup = saved_cfg_json.get("scheduler_warmup_steps", cfg.scheduler_warmup_steps)
        print(f"Initializing optimizer with learning rate: {resume_lr}")

        trainable_params = [p for p in policy.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=resume_lr, weight_decay=cfg.optimizer_weight_decay)
        print(f"Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        optimizer_state_path = local_ckpt_path / "optimizer_state.pth"
        if optimizer_state_path.exists():
            try:
                optimizer.load_state_dict(torch.load(optimizer_state_path, map_location=device))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = resume_lr
                    param_group['initial_lr'] = resume_lr
                print(f"Optimizer state loaded. LR reset to {resume_lr}")
            except ValueError as e:
                print(f"Skipping optimizer state — architecture mismatch ({e})")

        warmup_steps = resume_warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=training_steps,
        )
        for _ in range(step):
            scheduler.step()
        print(f"Scheduler fast-forwarded to step {step}, LR = {optimizer.param_groups[0]['lr']:.2e}")
    else:
        policy = WilroPolicy(cfg)
        policy.train()
        policy.to(device)

        preprocessor, postprocessor = make_pre_post_processors(
            cfg,
            dataset_stats=dataset_metadata.stats,
        )
        step = 0
        epoch = 0

        trainable_params = [p for p in policy.parameters() if p.requires_grad]
        n_frozen = sum(p.numel() for p in policy.parameters() if not p.requires_grad)
        print(f"Total trainable parameters: {sum(p.numel() for p in trainable_params):,}  "
              f"(frozen: {n_frozen:,})")

        fresh_lr = cfg.optimizer_lr
        fresh_warmup = cfg.scheduler_warmup_steps
        optimizer = torch.optim.Adam(trainable_params, lr=fresh_lr, weight_decay=cfg.optimizer_weight_decay)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=fresh_warmup,
            num_training_steps=training_steps,
        )

    # Optional DiT gradient checkpointing (frozen VLM is unaffected — it runs in no_grad).
    if gradient_checkpointing and hasattr(policy.model, "gradient_checkpointing_enable"):
        policy.model.gradient_checkpointing_enable()

    if isinstance(preprocessor, torch.nn.Module):
        preprocessor.to(device)

    # Dataset setup — read fps from metadata instead of hardcoding. piper_arm
    # is 30 fps but libero / community datasets are commonly 10 fps; using a
    # mismatched frame_time makes every requested delta_timestamp fall outside
    # tolerance_s and the constructor raises.
    fps = int(getattr(dataset_metadata, "fps", 30) or 30)
    frame_time = 1 / fps
    print(f"Dataset fps: {fps} (frame_time={frame_time:.4f}s)")

    # Observation window: last `obs` frames ending at t=0
    obs_temporal_window = [-i * frame_time for i in range(obs)][::-1]
    # Action window: `horizon` steps starting at t=0
    action_temporal_window = [i * frame_time for i in range(horizon)]

    delta_timestamps = {
        "observation.state": obs_temporal_window,
        "action": action_temporal_window,
        # Cameras only need the current frame — the model always uses imgs[:, -1].
        **{key: [0.0] for key in camera_keys},
    }

    # `tolerance_s` must accommodate the dataset's frame interval — too tight
    # and every delta lookup raises. Half a frame is a safe upper bound.
    tolerance_s = max(0.005, frame_time / 2)
    dataset = LeRobotDataset(
        dataset_id, delta_timestamps=delta_timestamps,
        force_cache_sync=True, revision="main", tolerance_s=tolerance_s,
    )

    # Build task_index → description mapping from tasks.parquet
    task_idx_to_description: dict[int, str] = {}
    try:
        tasks_parquet_path = dataset.root / "meta" / "tasks.parquet"
        if tasks_parquet_path.exists():
            tasks_df = pd.read_parquet(tasks_parquet_path)
            if "task_index" in tasks_df.columns:
                task_idx_to_description = {
                    int(row["task_index"]): str(idx)
                    for idx, row in tasks_df.iterrows()
                }
            print(f"Loaded {len(task_idx_to_description)} task descriptions from tasks.parquet:")
            for idx, desc in task_idx_to_description.items():
                print(f"  [{idx}] {desc}")
        else:
            print("tasks.parquet not found; task_description will not be added to batches.")
    except Exception as e:
        print(f"Warning: could not load tasks.parquet: {e}")

    if dataset_metadata.stats and "observation.state" in dataset_metadata.stats:
        s = dataset_metadata.stats["observation.state"]
        print(f"\nNorm stats observation.state:")
        print(f"  mean={s.get('mean', 'N/A')}")
        print(f"  std ={s.get('std',  'N/A')}")
    else:
        print("WARNING: observation.state not found in dataset_metadata.stats — will not be normalized!")
    if dataset_metadata.stats and "action" in dataset_metadata.stats:
        s = dataset_metadata.stats["action"]
        print(f"Norm stats action:")
        print(f"  mean={s.get('mean', 'N/A')}")
        print(f"  std ={s.get('std',  'N/A')}")

    episode_ids = np.array(dataset.hf_dataset["episode_index"])
    if max_episode_index is not None:
        valid_indices = np.where(episode_ids <= max_episode_index)[0].tolist()
        if len(valid_indices) == 0:
            raise ValueError(
                f"max_episode_index={max_episode_index} excluded every sample "
                f"(dataset episode range: {episode_ids.min()}..{episode_ids.max()})."
            )
        dataset = Subset(dataset, valid_indices)
        print(f"Dataset subset: {len(dataset)} samples (episodes <= {max_episode_index})")
    else:
        valid_indices = list(range(len(episode_ids)))
        print(f"Dataset: {len(dataset)} samples (no episode filter)")

    episode_ids_subset = episode_ids[np.array(valid_indices)]
    ep_changes = np.where(np.diff(episode_ids_subset) != 0)[0] + 1
    ep_from = np.concatenate([[0], ep_changes]).tolist()
    ep_to = np.concatenate([ep_changes, [len(valid_indices)]]).tolist()
    sampler = EpisodeAwareSampler(
        dataset_from_indices=ep_from,
        dataset_to_indices=ep_to,
        drop_n_first_frames=0,
        drop_n_last_frames=0,
        shuffle=True,
    )
    print(f"EpisodeAwareSampler: {len(sampler)} frames")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=8,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Training loop
    print("Starting training loop...")
    done = False
    prog_bar = tqdm(total=training_steps, desc="Training Progress", initial=step)
    while not done:
        epoch += 1
        for batch in dataloader:
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)

            # Enrich batch with task description strings
            if "task" in batch and isinstance(batch["task"], (list, tuple)):
                batch["task_description"] = batch["task"]
            elif task_idx_to_description and "task_index" in batch:
                task_indices = batch["task_index"]
                if isinstance(task_indices, torch.Tensor) and task_indices.dim() > 1:
                    task_indices = task_indices[:, 0]
                batch["task_description"] = [task_idx_to_description.get(int(ti), "") for ti in task_indices]

            batch = apply_image_augmentations(batch, camera_keys, image_transforms)
            batch = apply_joint_augmentations(batch)

            if step == 0:
                raw_st = batch["observation.state"].float()
                print(f"\nRaw (pre-norm) observation.state: min={raw_st.min():.4f}  max={raw_st.max():.4f}  std={raw_st.std():.4f}")

            batch = preprocessor(batch)

            if step == 0:
                pad_key = next((k for k in ("action_is_pad", "actions_id_pad") if k in batch), None)
                if pad_key is None:
                    print("WARNING: no action pad key found in batch — padded episode steps will pollute loss!")
                    print(f"  Available keys: {[k for k in batch.keys() if 'pad' in k.lower() or 'action' in k.lower()]}")
                else:
                    pad_frac = batch[pad_key].float().mean().item()
                    print(f"Action pad key='{pad_key}', pad fraction in first batch: {pad_frac:.2%}")

            # Forward & Backward
            # Arm the attention-mass diagnostic on the same cadence as
            # gradient analysis. The model self-disarms after one capture.
            if step % progress_update_freq == 0:
                policy.model._capture_attention_stats = True

            autocast_ctx = (
                torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                if device.type == "cuda"
                else torch.autocast(device_type="cpu", enabled=False)
            )
            with autocast_ctx:
                loss, _ = policy.forward(batch)

            if loss.item() > 100 and step < 2000:
                act = batch["action"].float()
                st = batch["observation.state"].float()
                print(f"\n[DIAG step={step}] loss={loss.item():.1f}")
                print(f"  action  : min={act.min():.2f}  max={act.max():.2f}  std={act.std():.3f}")
                print(f"  state   : min={st.min():.2f}  max={st.max():.2f}  std={st.std():.3f}")
                pad_key = next((k for k in ("action_is_pad", "actions_id_pad") if k in batch), None)
                if pad_key is not None:
                    print(f"  pad frac: {batch[pad_key].float().mean().item():.2%}")

            loss.backward()

            if step % progress_update_freq == 0:
                _log_gradient_analysis(policy, step)

            trainable_params = [p for p in policy.parameters() if p.requires_grad]
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if step % progress_update_freq == 0:
                lr = optimizer.param_groups[0]['lr']
                prog_bar.set_description(f"Epoch {epoch}, Step {step}")
                prog_bar.set_postfix({
                    "loss": f"{loss.item():.3f}",
                    "lr": f"{lr:.2e}",
                    "grad_norm": f"{grad_norm:.2f}"
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
                prog_bar.set_description(f"Epoch {epoch}, Step {step}")

            if step >= training_steps:
                done = True
                prog_bar.close()
                break
    prog_bar.close()

    # Final save
    policy.config.training_step = step
    policy.config.training_epoch = epoch
    policy.config.optimizer_lr = optimizer.param_groups[0]["lr"]
    policy.config.current_lr = optimizer.param_groups[0]["lr"]
    policy.config.training_steps_total = training_steps
    policy.save_pretrained(output_directory)
    torch.save(optimizer.state_dict(), output_directory / "optimizer_state.pth")
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_id", type=str, default="ISdept/piper_arm")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Recompute DiT activations in backward to save memory.")
    parser.add_argument("--max_episode_index", type=int, default=None,
                        help="Filter to episodes with index <= this value "
                             "(piper_arm holdout convention; omit for full dataset).")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="DataLoader batch size (default: 64).")
    parser.add_argument("--contrastive_loss_weight", type=float, default=0.1,
                        help="Weight for the language-permute contrastive loss "
                             "(default: 0.1). Bump to ~0.5 for LIBERO / datasets "
                             "with diverse task descriptions.")
    parser.add_argument("--contrastive_margin", type=float, default=0.05,
                        help="Hinge margin on MSE between v_t and v_wrong "
                             "(default: 0.05). Bump to ~0.2 to force the model "
                             "to differentiate velocities by language.")
    parser.add_argument("--lock_joint_index", type=int, default=3,
                        help="Action dim with weight 0 (piper_arm joint 4 = "
                             "index 3 is mechanically locked). Pass -1 to "
                             "disable for LIBERO / other full-DOF robots.")
    args = parser.parse_args()
    # Argparse can't express None for an int, so use -1 sentinel.
    if args.lock_joint_index is not None and args.lock_joint_index < 0:
        args.lock_joint_index = None
    train(**vars(args))
