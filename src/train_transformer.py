from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
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

    training_steps = 100000
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
    has_box = "observation.box" in dataset_metadata.features
    state_dim = input_features["observation.state"].shape[-1] if "observation.state" in input_features else 7
    action_dim = next(iter(output_features.values())).shape[-1]
    print(f"Detected cameras ({len(camera_keys)}): {camera_keys}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    if has_box:
        print("Dataset has observation.box — bounding box encoding enabled")

    # Training parameters
    obs = 1
    horizon = 50
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
        num_decoder_layers=10,
        dim_feedforward=2048,
        diffusion_step_embed_dim=512,
        num_cameras=len(camera_keys),
        cameras_for_vision_state_concat=camera_keys,
    )
    
    
    # Model loading logic
    if resume_from_checkpoint is not None:
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        # Load policy with its original configuration
        policy = TransformerFlowMatchingPolicy.from_pretrained(resume_from_checkpoint, strict=False)
        policy.train()
        policy.to(device)
        
        # Use the policy's configuration for creating processors
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config, 
            dataset_stats=dataset_metadata.stats, 
            add_grid_overlay=True,
            grid_overlay_cameras=["front", "right"]  # Front and right cameras (original names)
        )
            
        # Define trainable parameters
        trainable_params = [p for p in policy.model.parameters() if p.requires_grad]
        
        # Print optimizer information
        resume_lr = policy.config.optimizer_lr if hasattr(policy.config, 'optimizer_lr') else 3e-4
        resume_warmup = policy.config.scheduler_warmup_steps if hasattr(policy.config, 'scheduler_warmup_steps') else 1500
        print(f"Initializing optimizer with learning rate: {resume_lr}")
        total_trainable_params = sum(p.numel() for p in trainable_params)
        print(f"Number of trainable parameters: {total_trainable_params}")
        
        optimizer = torch.optim.Adam(trainable_params, lr=resume_lr, weight_decay=1e-6)
        
        # Print learning rate groups
        print(f"Optimizer parameter groups: {len(optimizer.param_groups)}")
        for i, group in enumerate(optimizer.param_groups):
            group_param_count = sum(p.numel() for p in group['params'])
            print(f"  Group {i}: lr={group['lr']}, params={group_param_count}")

        # Load optimizer state (keep Adam momentum/variance but reset LR to new config value)
        if not resume_from_checkpoint.startswith("http") and not resume_from_checkpoint.startswith("huggingface.co"):
            checkpoint_path = Path(resume_from_checkpoint)
            optimizer_state_path = checkpoint_path / "optimizer_state.pth"
            if optimizer_state_path.exists():
                optimizer.load_state_dict(torch.load(optimizer_state_path, map_location=device))
                # Override the stale LR from the checkpoint with the new config value
                for param_group in optimizer.param_groups:
                    param_group['lr'] = resume_lr
                    param_group['initial_lr'] = resume_lr
                print(f"Optimizer state loaded. LR reset to {resume_lr}")

        # Fresh cosine scheduler (always recreated so warmup restarts correctly)
        warmup_steps = resume_warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=training_steps
        )

        # Read step/epoch counters from config (written at save time); fall back to
        # parsing the directory name for checkpoints saved before this change.
        step = getattr(policy.config, "training_step", 0)
        if step == 0 and not resume_from_checkpoint.startswith("http") and not resume_from_checkpoint.startswith("huggingface.co"):
            checkpoint_path = Path(resume_from_checkpoint)
            if checkpoint_path.name.startswith("checkpoint-"):
                step = int(checkpoint_path.name.split("-")[1])
        epoch = getattr(policy.config, "training_epoch", 0)
        print(f"Resuming from step {step}, epoch {epoch}")
    else:
        # Initialize fresh policy
        policy = TransformerFlowMatchingPolicy(cfg)
        policy.train()
        policy.to(device)
        
        # Print total trainable parameters
        total_params = sum(p.numel() for p in policy.model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params:,}")
        # Ensure all submodules are on the correct device
        if hasattr(policy, 'transformer') and hasattr(policy.transformer, 'feature_projection'):
            if policy.transformer.feature_projection is not None:
                policy.transformer.feature_projection = policy.transformer.feature_projection.to(device)
        preprocessor, postprocessor = make_pre_post_processors(
            cfg, 
            dataset_stats=dataset_metadata.stats, 
            add_grid_overlay=True,
            grid_overlay_cameras=["front", "right"]  # Front and right cameras (original names)
        )
        step = 0
        epoch = 0

        trainable_params = [p for p in policy.parameters() if p.requires_grad]
        fresh_lr = cfg.optimizer_lr
        fresh_warmup = cfg.scheduler_warmup_steps
        optimizer = torch.optim.Adam(trainable_params, lr=fresh_lr, weight_decay=cfg.optimizer_weight_decay)
        
        # Print learning rate groups
        print(f"Optimizer parameter groups: {len(optimizer.param_groups)}")
        for i, group in enumerate(optimizer.param_groups):
            group_param_count = sum(p.numel() for p in group['params'])
            print(f"  Group {i}: lr={group['lr']}, params={group_param_count}")
        
        # Cosine scheduler with warmup
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
    
    # Create observation temporal window
    obs_temporal_window = [ -i * frame_time for i in range(obs) ][::-1]
    
    # Shift action timestamps by 1 position to prevent overlap with observations
    action_temporal_window = [i * frame_time for i in range(horizon)]
    
    delta_timestamps = {
        "observation.state": obs_temporal_window,
        "action": action_temporal_window,
        **{key: obs_temporal_window for key in camera_keys},
        **({"observation.box": obs_temporal_window} if has_box else {}),
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

            # Preprocess (Normalize)
            batch = preprocessor(batch)

            # Forward & Backward
            loss, _ = policy.forward(batch)
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
                policy.save_pretrained(checkpoint_dir)
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
    policy.save_pretrained(output_directory)
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