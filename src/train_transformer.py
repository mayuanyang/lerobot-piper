from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
import huggingface_hub
from safetensors.torch import load_file as load_safetensors
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.factory import make_pre_post_processors
import numpy as np
from torch.utils.data import Subset

# Import transformer-specific components
from models.transformer_flow_matching.transformer_flow_matching_config import TransformerFlowMatchingConfig
from models.transformer_flow_matching.transformer_flow_matching_policy import TransformerFlowMatchingPolicy
from models.transformer_flow_matching.processor_transformer_flow_matching import make_pre_post_processors

# Import visualization utilities
from spatial_softmax_visualizer import SpatialSoftmaxVisualizer

# Import torchvision for augmentation
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


# Data Augmentation Setup
def get_augmentations():
    """Return an image augmentation transform for training.

    ColorJitter is intentionally NOT applied per-camera inside the dataset
    (which would give each camera independent random params, creating physically
    inconsistent lighting across views).  Instead this transform is applied in
    the training loop via apply_image_augmentations(), which stacks all cameras
    for a given sample and runs the transform once so every camera in that sample
    receives the same random brightness/contrast/saturation/hue shift.
    """
    return v2.Compose([
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    ])


# Helper to apply joint data augmentation
def apply_joint_augmentations(batch):
    """Apply noise to joint data (observation.state and action)"""
    # Apply augmentation with 50% probability
    if torch.rand(1).item() > 0.5:
        # Add small Gaussian noise to joint states
        if "observation.state" in batch:
            noise_scale = 0.01  # 1% of the typical joint range
            noise = torch.randn_like(batch["observation.state"]) * noise_scale
            batch["observation.state"] = batch["observation.state"] + noise
            
        # Add small Gaussian noise to actions
        # if "action" in batch:
        #     noise_scale = 0.01  # 1% of the typical action range
        #     noise = torch.randn_like(batch["action"]) * noise_scale
        #     batch["action"] = batch["action"] + noise
            
    return batch


# Helper to randomly drop one camera
def apply_camera_dropout(batch, camera_keys=["observation.images.front", "observation.images.gripper", "observation.images.right"], dropout_prob=0.2):
    """Randomly drop one camera from the batch during training."""
    # Only apply during training and with specified probability
    if torch.rand(1).item() > dropout_prob:
        return batch
    
    # If we only have one camera, don't drop it
    if len(camera_keys) <= 1:
        return batch
    
    # Randomly select one camera to drop
    camera_to_drop = torch.randint(0, len(camera_keys), (1,)).item()
    dropped_camera_key = camera_keys[camera_to_drop]
    
    # Remove the selected camera from the batch
    if dropped_camera_key in batch:
        batch[dropped_camera_key] = torch.zeros_like(batch[dropped_camera_key])
    
    return batch


def apply_image_augmentations(batch, camera_keys, transform):
    """Apply the same random color jitter to all cameras within each sample.

    For each sample in the batch, all camera images are stacked into a single
    tensor and passed through the transform in one call.  torchvision v2 samples
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
        has_time_dim = sample_img.dim() == 4  # (T, C, H, W)

        if has_time_dim:
            T = sample_img.shape[0]
            # Cat along time/batch dim: (N_cams * T, C, H, W)
            stacked = torch.cat([batch[k][b] for k in present_keys], dim=0)
            stacked_aug = transform(stacked)
            for i, k in enumerate(present_keys):
                batch[k][b] = stacked_aug[i * T:(i + 1) * T]
        else:
            # Stack cameras: (N_cams, C, H, W)
            stacked = torch.stack([batch[k][b] for k in present_keys], dim=0)
            stacked_aug = transform(stacked)
            for i, k in enumerate(present_keys):
                batch[k][b] = stacked_aug[i]

    return batch


def apply_state_dropout(batch, state_key="observation.state", dropout_prob=0.05):
    state = batch[state_key]
    mask = (torch.rand(state.size(0), 1, 1, device=state.device) > dropout_prob).float()
    batch[state_key] = state * mask
    return batch



def train(output_dir, dataset_id="ISdept/piper_arm", push_to_hub=False, resume_from_checkpoint=None, visualize_every_n_batches=1000):
    """Train the TransformerFlowMatching model."""
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    training_steps = 200000
    progress_update_freq = 200  # Single frequency for all progress updates
    checkpoint_freq = 1000
    image_transforms = get_augmentations()
    


    # Load dataset metadata
    dataset_metadata = LeRobotDatasetMetadata(dataset_id, force_cache_sync=True, revision="main")
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    # Safety check
    if len(output_features) == 0:
        raise ValueError("No output features (actions) found! Check your dataset schema.")

    print('input_features:', input_features)
    print('output_features:', output_features)

    # Auto-detect camera keys and dims from dataset features
    camera_keys = sorted([key for key, ft in input_features.items() if ft.type is FeatureType.VISUAL])
    state_dim = input_features["observation.state"].shape[-1] if "observation.state" in input_features else 7
    action_dim = next(iter(output_features.values())).shape[-1]
    print(f"Detected cameras ({len(camera_keys)}): {camera_keys}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Training parameters
    obs = 4
    horizon = 16
    n_action_steps = 8

    # Create transformer configuration
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
        num_decoder_layers=8,
        dim_feedforward=2048,
        num_vlm_layers=16,
        num_cameras=len(camera_keys),
        cameras_for_vision_state_concat=camera_keys,
    )
    
    
    # Model loading logic
    if resume_from_checkpoint is not None:
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        # Initialize fresh policy with current config, then load compatible weights.
        # strict=False alone won't handle size mismatches (PyTorch raises even then),
        # so we pre-filter to only load keys whose shapes match the current model.
        policy = TransformerFlowMatchingPolicy(cfg)

        ckpt_path = Path(resume_from_checkpoint)
        if ckpt_path.exists():
            local_ckpt_path = ckpt_path
        else:
            # HuggingFace Hub — download full snapshot to local cache
            local_ckpt_path = Path(huggingface_hub.snapshot_download(resume_from_checkpoint))

        # Prefer model.safetensors explicitly — checkpoint dirs also contain
        # preprocessor/postprocessor safetensors files that would match *.safetensors
        # but contain no model weights, causing silent zero-match loading.
        model_file = local_ckpt_path / "model.safetensors"
        if not model_file.exists():
            candidates = list(local_ckpt_path.glob("*.safetensors"))
            if not candidates:
                raise FileNotFoundError(f"No .safetensors file found in {local_ckpt_path}")
            model_file = candidates[0]

        # Read saved config FIRST — must know if VLM/LoRA was enabled before loading
        # weights, because LoRA wrapping changes the state dict key names.
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
                # Restore total training steps so the cosine schedule reproduces
                # the exact same LR curve — a mismatch here causes a 7× LR jump.
                saved_total = saved_cfg_json.get("training_steps_total", 0)
                if saved_total > 0:
                    training_steps = saved_total
                print(f"Read config from {config_file.name}: step={step}, epoch={epoch}, training_steps_total={training_steps}")
                break
        # Fall back to parsing the directory name for older checkpoints
        if step == 0 and local_ckpt_path.name.startswith("checkpoint-"):
            step = int(local_ckpt_path.name.split("-")[1])
        print(f"Resuming from step {step}, epoch {epoch}")

        # Apply LoRA before loading weights — peft wrapping renames keys
        # (adds base_model.model. prefix), so structure must match checkpoint.
        policy.model.enable_lora()

        # Load checkpoint state dict
        print(f"Loading weights from: {model_file}")
        ckpt_state = load_safetensors(model_file, device=str(device))

        policy.train()
        policy.to(device)
        cur_state = policy.state_dict()
        filtered = {k: v for k, v in ckpt_state.items() if k in cur_state and cur_state[k].shape == v.shape}

        # Keys in checkpoint not loaded (shape mismatch or key removed from model)
        skipped_ckpt = [k for k in ckpt_state if k not in filtered]
        # Keys in current model not in checkpoint (new params — will stay at init values)
        missing_from_ckpt = [k for k in cur_state if k not in ckpt_state]

        if skipped_ckpt:
            print(f"Skipped {len(skipped_ckpt)} checkpoint keys (shape mismatch / removed): {skipped_ckpt[:10]}")
        if missing_from_ckpt:
            print(f"Missing {len(missing_from_ckpt)} keys not in checkpoint (will use init values): {missing_from_ckpt[:10]}")
        policy.load_state_dict(filtered, strict=False)
        print(f"Loaded {len(filtered)}/{len(cur_state)} model keys from checkpoint ({len(ckpt_state)} keys in file)")

        # Use the policy's configuration for creating processors
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            dataset_stats=dataset_metadata.stats,
            add_grid_overlay=True,
            grid_overlay_cameras=["front", "right"]
        )

        # Optimizer — read lr/warmup from saved config, fall back to cfg
        resume_lr = saved_cfg_json.get("optimizer_lr", cfg.optimizer_lr)
        resume_warmup = saved_cfg_json.get("scheduler_warmup_steps", cfg.scheduler_warmup_steps)
        print(f"Initializing optimizer with learning rate: {resume_lr}")

        trainable_params = [p for p in policy.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=resume_lr, weight_decay=1e-6)
        print(f"Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        # Print learning rate groups
        print(f"Optimizer parameter groups: {len(optimizer.param_groups)}")
        for i, group in enumerate(optimizer.param_groups):
            group_param_count = sum(p.numel() for p in group['params'])
            print(f"  Group {i}: lr={group['lr']}, params={group_param_count}")

        # Load optimizer state (keep Adam momentum/variance but reset LR to new config value).
        # Skip if the checkpoint has no LoRA weights — the param count will differ since
        # LoRA A/B matrices are new params not present in the old optimizer state.
        ckpt_has_lora = any("lora_" in k for k in ckpt_state)
        optimizer_state_path = local_ckpt_path / "optimizer_state.pth"
        if optimizer_state_path.exists() and ckpt_has_lora:
            optimizer.load_state_dict(torch.load(optimizer_state_path, map_location=device))
            for param_group in optimizer.param_groups:
                param_group['lr'] = resume_lr
                param_group['initial_lr'] = resume_lr
            print(f"Optimizer state loaded. LR reset to {resume_lr}")
        elif optimizer_state_path.exists() and not ckpt_has_lora:
            print("Checkpoint has no LoRA weights — skipping optimizer state (LoRA params are new, moments would be mismatched)")

        # Create cosine scheduler and fast-forward to match saved step.
        # This ensures LR is correct on resume — not restarting from warmup.
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
        # Initialize fresh policy
        policy = TransformerFlowMatchingPolicy(cfg)
        policy.model.enable_lora()
        policy.train()
        policy.to(device)

        preprocessor, postprocessor = make_pre_post_processors(
            cfg,
            dataset_stats=dataset_metadata.stats,
            add_grid_overlay=True,
            grid_overlay_cameras=["front", "right"]
        )
        step = 0
        epoch = 0

        trainable_params = [p for p in policy.parameters() if p.requires_grad]
        print(f"Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        fresh_lr = cfg.optimizer_lr
        fresh_warmup = cfg.scheduler_warmup_steps
        optimizer = torch.optim.Adam(trainable_params, lr=fresh_lr, weight_decay=cfg.optimizer_weight_decay)

        warmup_steps = fresh_warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=training_steps
        )

    # Ensure preprocessors are on the correct device
    if isinstance(preprocessor, torch.nn.Module):
        preprocessor.to(device)

    # Dataset setup
    fps = 10
    frame_time = 1 / fps
    
    # Observation window: last `obs` frames ending at t=0 (current frame).
    # e.g. obs=2 → [-0.1, 0.0]
    obs_temporal_window = [ -i * frame_time for i in range(obs) ][::-1]

    # Action window: `horizon` steps starting at t=0.
    # The first action (t=0) corresponds to the current observation — standard LeRobot convention.
    # e.g. horizon=16 → [0.0, 0.1, ..., 1.5]
    action_temporal_window = [i * frame_time for i in range(horizon)]
    
    delta_timestamps = {
        "observation.state": obs_temporal_window,
        "action": action_temporal_window,
        # Cameras only need the current frame — the model always uses imgs[:, -1].
        # Loading the full obs_temporal_window per camera wastes memory with no benefit.
        **{key: [0.0] for key in camera_keys},
    }

    # Load dataset
    try:
        dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps, force_cache_sync=True, revision="main", tolerance_s=0.005)
    except Exception as e:
        print(f"Error loading remote dataset: {e}")
        local_dataset_path = "./src/output" 
        print(f"Trying local dataset at {local_dataset_path}...")
        dataset = LeRobotDataset(local_dataset_path, delta_timestamps=delta_timestamps, force_cache_sync=True, tolerance_s=0.005)  # Tighter timestamp alignment

    # Build task_index → description mapping from tasks.parquet
    task_idx_to_description: dict[int, str] = {}
    try:
        tasks_parquet_path = dataset.root / "meta" / "tasks.parquet"
        if tasks_parquet_path.exists():
            tasks_df = pd.read_parquet(tasks_parquet_path)
            # tasks.parquet uses task strings as the index and task_index as a column
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

    # Print normalization stats for state and action to detect near-zero std
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

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=8,
        batch_size=128,  # Reduced batch size for better gradient flow
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Create visualizer
    #visualizer = SpatialSoftmaxVisualizer(Path(output_dir) / "spatial_softmax_visualizations")

    episode_ids = np.array(dataset.hf_dataset["episode_index"])
    valid_indices = np.where(episode_ids <= 400)[0]  # first 40 episodes

    dataset = Subset(dataset, valid_indices)
    print('The partial dataset length', len(dataset))

    # Training loop
    print("Starting training loop...")
    done = False
    prog_bar = tqdm(total=training_steps, desc="Training Progress", initial=step)
    while not done:
        epoch += 1
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)

            # Enrich batch with task description strings (looked up from tasks.parquet mapping)
            if task_idx_to_description and "task_index" in batch:
                task_indices = batch["task_index"]  # (B,) or (B, T)
                # Use the first element's task_index for each sample
                if task_indices.dim() > 1:
                    task_indices = task_indices[:, 0]
                batch["task_description"] = [
                    task_idx_to_description.get(int(ti.item()), "") for ti in task_indices
                ]

            # Apply image augmentation (same random params across all cameras per sample)
            batch = apply_image_augmentations(batch, camera_keys, image_transforms)

            # Apply joint data augmentation
            batch = apply_joint_augmentations(batch)

            # Apply state dropout
            # NOTE: Disabled - was running before normalization, causing spurious gradients in state encoder
            # batch = apply_state_dropout(batch)

            # Apply camera dropout
            #batch = apply_camera_dropout(batch)

            # Step-0 diagnostic: print raw state before normalization
            if step == 0:
                raw_st = batch["observation.state"].float()
                print(f"\nRaw (pre-norm) observation.state: min={raw_st.min():.4f}  max={raw_st.max():.4f}  std={raw_st.std():.4f}")

            # Preprocess (Normalize)
            batch = preprocessor(batch)

            # One-time diagnostic: confirm action padding mask is present
            if step == 0:
                pad_key = next((k for k in ("action_is_pad", "actions_id_pad") if k in batch), None)
                if pad_key is None:
                    print("WARNING: no action pad key found in batch — padded episode steps will pollute loss!")
                    print(f"  Available keys: {[k for k in batch.keys() if 'pad' in k.lower() or 'action' in k.lower()]}")
                else:
                    pad_frac = batch[pad_key].float().mean().item()
                    print(f"Action pad key='{pad_key}', pad fraction in first batch: {pad_frac:.2%}")

            # Forward & Backward
            loss, _ = policy.forward(batch)

            # Diagnostic: print value ranges when loss is unexpectedly large.
            # This helps distinguish between data outliers, normalization bugs,
            # and model instability (e.g. exploding context / state tokens).
            if loss.item() > 100 and step < 2000:
                act = batch["action"].float()
                st  = batch["observation.state"].float()
                print(f"\n[DIAG step={step}] loss={loss.item():.1f}")
                print(f"  action  : min={act.min():.2f}  max={act.max():.2f}  std={act.std():.3f}")
                print(f"  state   : min={st.min():.2f}  max={st.max():.2f}  std={st.std():.3f}")
                pad_key = next((k for k in ("action_is_pad", "actions_id_pad") if k in batch), None)
                if pad_key is not None:
                    print(f"  pad frac: {batch[pad_key].float().mean().item():.2%}")

            loss.backward()

            # Print gradient information for all components (for debugging)
            if step % progress_update_freq == 0:  # Print every 10 progress update intervals
                print(f"\n--- Gradient Analysis at Step {step} ---")
                
                
                # State encoder gradients
                total_state_grad = 0.0
                total_state_params = 0
                for name, param in policy.model.named_parameters():
                    if param.requires_grad and 'state_encoder' in name and param.grad is not None:
                        grad_mean = param.grad.abs().mean().item()
                        param_count = param.numel()
                        total_state_grad += grad_mean * param_count
                        total_state_params += param_count
                
                # Box encoder gradients (individual components)
                total_box_grad = 0.0
                total_box_params = 0
                box_component_names = ['box_encoder']
                for name, param in policy.model.named_parameters():
                    if param.requires_grad and any(comp_name in name for comp_name in box_component_names) and param.grad is not None:
                        grad_mean = param.grad.abs().mean().item()
                        param_count = param.numel()
                        total_box_grad += grad_mean * param_count
                        total_box_params += param_count
                        
                # Vision encoder gradients
                total_vision_grad = 0.0
                total_vision_params = 0
                for name, param in policy.model.named_parameters():
                    if param.requires_grad and 'vision_encoder' in name and param.grad is not None:
                        grad_mean = param.grad.abs().mean().item()
                        param_count = param.numel()
                        total_vision_grad += grad_mean * param_count
                        total_vision_params += param_count
                        
                # actions expert gradients
                total_action_expert_grad = 0.0
                total_action_expert_params = 0
                for name, param in policy.model.named_parameters():
                    if param.requires_grad and 'actions_expert' in name and param.grad is not None:
                        grad_mean = param.grad.abs().mean().item()
                        param_count = param.numel()
                        total_action_expert_grad += grad_mean * param_count
                        total_action_expert_params += param_count
                
                print(f"State Encoder - Avg Abs Grad: {total_state_grad / total_state_params:.6f} (Total Params: {total_state_params})")
                if total_box_params > 0:
                    print(f"Box Encoder - Avg Abs Grad: {total_box_grad / total_box_params:.6f} (Total Params: {total_box_params})")
                else:
                    print(f"Box Encoder - Avg Abs Grad: N/A (Total Params: {total_box_params})")
                if total_vision_params > 0:
                    print(f"Vision Encoder - Avg Abs Grad: {total_vision_grad / total_vision_params:.6f} (Total Params: {total_vision_params})")
                else:
                    print(f"Vision Encoder - Avg Abs Grad: N/A (Total Params: {total_vision_params})")
                print(f"Actions Expert - Avg Abs Grad: {total_action_expert_grad / total_action_expert_params:.6f} (Total Params: {total_action_expert_params})")
                
                print("--- End Gradient Analysis ---\n")
            
            # Clip gradients and calculate norm (only for trainable parameters)
            trainable_params = [p for p in policy.parameters() if p.requires_grad]
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Update learning rate scheduler (cosine scheduler steps every iteration)
            scheduler.step()

            if step % progress_update_freq == 0:
                # Get learning rate from optimizer
                lr = optimizer.param_groups[0]['lr']
                prog_bar.set_description(f"Epoch {epoch}, Step {step}")
                prog_bar.set_postfix({
                    "loss": f"{loss.item():.3f}", 
                    "lr": f"{lr:.2e}",
                    "grad_norm": f"{grad_norm:.2f}"
                })
            
            
            # Save checkpoint
            if step > 0 and step % checkpoint_freq == 0:
                checkpoint_dir = output_directory / f"checkpoint-{step}"
                checkpoint_dir.mkdir(exist_ok=True)
                policy.config.training_step = step
                policy.config.training_epoch = epoch
                policy.config.current_lr = optimizer.param_groups[0]["lr"]
                policy.config.training_steps_total = training_steps
                policy.save_pretrained(checkpoint_dir)
                torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer_state.pth")
                preprocessor.save_pretrained(checkpoint_dir)
                postprocessor.save_pretrained(checkpoint_dir)
                print(f"\nCheckpoint saved at step {step}")
                
            step += 1
            # Update progress bar less frequently to reduce Colab verbosity
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
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()
    
    train(**vars(args))