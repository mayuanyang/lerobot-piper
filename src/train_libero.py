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

from models.transformer_flow_matching.transformer_flow_matching_config import TransformerFlowMatchingConfig
from models.transformer_flow_matching.transformer_flow_matching_policy import TransformerFlowMatchingPolicy
from models.transformer_flow_matching.processor_transformer_flow_matching import make_pre_post_processors

from torchvision.transforms import v2
from transformers import get_cosine_schedule_with_warmup


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


def get_augmentations():
    return v2.Compose([
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    ])


def apply_joint_augmentations(batch):
    if torch.rand(1).item() > 0.5:
        if "observation.state" in batch:
            noise = torch.randn_like(batch["observation.state"]) * 0.01
            batch["observation.state"] = batch["observation.state"] + noise
    return batch


def apply_image_augmentations(batch, camera_keys, transform):
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
                batch[k][b] = stacked_aug[i * T:(i + 1) * T]
        else:
            stacked = torch.stack([batch[k][b] for k in present_keys], dim=0)
            stacked_aug = transform(stacked)
            for i, k in enumerate(present_keys):
                batch[k][b] = stacked_aug[i]

    return batch


def train(output_dir, dataset_id="physical-intelligence/libero", resume_from_checkpoint=None):
    """Train the TransformerFlowMatching model on the Libero dataset."""
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

    obs = 2
    horizon = 32
    n_action_steps = 32

    cfg = TransformerFlowMatchingConfig(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=obs,
        horizon=horizon,
        n_action_steps=n_action_steps,
        state_dim=state_dim,
        action_dim=action_dim,
        d_model=512,
        nhead=8,
        num_decoder_layers=16,
        dim_feedforward=2048,
        num_vlm_layers=16,
        num_cameras=len(camera_keys),
        cameras_for_vision_state_concat=camera_keys,
        # All action dims equal — libero has no locked joints
        action_dim_weights=[1.0] * action_dim,
        pos_decay_lambda=0.0,
    )

    if resume_from_checkpoint is not None:
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        policy = TransformerFlowMatchingPolicy(cfg)

        ckpt_path = Path(resume_from_checkpoint)
        if ckpt_path.exists():
            local_ckpt_path = ckpt_path
        else:
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
        optimizer = torch.optim.Adam(trainable_params, lr=resume_lr, weight_decay=1e-6)
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
        policy = TransformerFlowMatchingPolicy(cfg)
        policy.train()
        policy.to(device)

        preprocessor, postprocessor = make_pre_post_processors(
            cfg,
            dataset_stats=dataset_metadata.stats,
        )
        step = 0
        epoch = 0

        trainable_params = [p for p in policy.parameters() if p.requires_grad]
        print(f"Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        fresh_lr = cfg.optimizer_lr
        fresh_warmup = cfg.scheduler_warmup_steps
        optimizer = torch.optim.Adam(trainable_params, lr=fresh_lr, weight_decay=cfg.optimizer_weight_decay)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=fresh_warmup,
            num_training_steps=training_steps,
        )

    if isinstance(preprocessor, torch.nn.Module):
        preprocessor.to(device)

    # Libero is typically 30 fps
    fps = dataset_metadata.fps if hasattr(dataset_metadata, "fps") else 30
    frame_time = 1 / fps

    obs_temporal_window = [-i * frame_time for i in range(obs)][::-1]
    action_temporal_window = [i * frame_time for i in range(horizon)]

    delta_timestamps = {
        "observation.state": obs_temporal_window,
        "action": action_temporal_window,
        **{key: [0.0] for key in camera_keys},
    }

    dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps, force_cache_sync=True, revision="main", tolerance_s=0.1)
    print(f"Dataset loaded: {len(dataset)} total frames")

    # Build task_index → description mapping.
    # Libero stores per-frame task strings in the `task` column (populated by LeRobot
    # from tasks.parquet). We also fall back to reading tasks.parquet directly.
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
            print(f"Loaded {len(task_idx_to_description)} task descriptions from tasks.parquet")
        else:
            print("tasks.parquet not found; task_description will come from batch['task'] if present.")
    except Exception as e:
        print(f"Warning: could not load tasks.parquet: {e}")

    if dataset_metadata.stats and "observation.state" in dataset_metadata.stats:
        s = dataset_metadata.stats["observation.state"]
        print(f"\nNorm stats observation.state:")
        print(f"  mean={s.get('mean', 'N/A')}")
        print(f"  std ={s.get('std',  'N/A')}")
    if dataset_metadata.stats and "action" in dataset_metadata.stats:
        s = dataset_metadata.stats["action"]
        print(f"Norm stats action:")
        print(f"  mean={s.get('mean', 'N/A')}")
        print(f"  std ={s.get('std',  'N/A')}")

    episode_ids = np.array(dataset.hf_dataset["episode_index"])
    ep_changes = np.where(np.diff(episode_ids) != 0)[0] + 1
    ep_from = np.concatenate([[0], ep_changes]).tolist()
    ep_to = np.concatenate([ep_changes, [len(episode_ids)]]).tolist()
    sampler = EpisodeAwareSampler(
        dataset_from_indices=ep_from,
        dataset_to_indices=ep_to,
        drop_n_first_frames=0,
        drop_n_last_frames=0,
        shuffle=True,
    )
    print(f"EpisodeAwareSampler: {len(sampler)} frames across {len(ep_from)} episodes")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=8,
        batch_size=64,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    print("Starting training loop...")
    done = False
    prog_bar = tqdm(total=training_steps, desc="Training Progress", initial=step)
    while not done:
        epoch += 1
        for batch in dataloader:
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)

            # Task descriptions: prefer the `task` column that LeRobot populates
            # directly on each frame for image-based datasets (like libero).
            # Fall back to tasks.parquet index lookup.
            if "task" in batch and isinstance(batch["task"], (list, tuple)):
                batch["task_description"] = batch["task"]
            elif task_idx_to_description and "task_index" in batch:
                task_indices = batch["task_index"]
                if task_indices.dim() > 1:
                    task_indices = task_indices[:, 0]
                batch["task_description"] = [
                    task_idx_to_description.get(int(ti.item()), "") for ti in task_indices
                ]

            batch = apply_image_augmentations(batch, camera_keys, image_transforms)
            batch = apply_joint_augmentations(batch)

            if step == 0:
                raw_st = batch["observation.state"].float()
                print(f"\nRaw (pre-norm) observation.state: min={raw_st.min():.4f}  max={raw_st.max():.4f}  std={raw_st.std():.4f}")

            batch = preprocessor(batch)

            if step == 0:
                pad_key = next((k for k in ("action_is_pad", "actions_id_pad") if k in batch), None)
                if pad_key is None:
                    print("WARNING: no action pad key found in batch")
                else:
                    print(f"Action pad key='{pad_key}', pad fraction: {batch[pad_key].float().mean().item():.2%}")

            autocast_ctx = (
                torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                if device.type == "cuda"
                else torch.autocast(device_type="cpu", enabled=False)
            )
            with autocast_ctx:
                loss, _ = policy.forward(batch)

            loss.backward()

            if step % progress_update_freq == 0:
                print(f"\n--- Gradient Analysis at Step {step} ---")

                def _grad_stats(prefix):
                    total, count = 0.0, 0
                    for name, param in policy.model.named_parameters():
                        if param.requires_grad and prefix in name and param.grad is not None:
                            total += param.grad.abs().mean().item() * param.numel()
                            count += param.numel()
                    return total / count if count > 0 else None, count

                vision_grad, vision_n       = _grad_stats('vision_model')
                connector_grad, conn_n      = _grad_stats('connector')
                action_grad, action_n       = _grad_stats('action_expert')

                print(f"Vision         - Avg Abs Grad: {vision_grad:.6f} ({vision_n} params)" if vision_grad is not None else "Vision         - no grad")
                print(f"Connector      - Avg Abs Grad: {connector_grad:.6f} ({conn_n} params)" if connector_grad is not None else "Connector      - no grad")
                print(f"Actions Expert - Avg Abs Grad: {action_grad:.6f} ({action_n} params)" if action_grad is not None else "Actions Expert - no grad")
                print("--- End Gradient Analysis ---\n")

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
                prog_bar.set_description(f"Epoch {epoch}, Step {step}")

            if step >= training_steps:
                done = True
                prog_bar.close()
                break
    prog_bar.close()

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
    parser.add_argument("--dataset_id", type=str, default="physical-intelligence/libero")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()
    train(**vars(args))
