from pathlib import Path
import torch
from tqdm import tqdm
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.factory import make_pre_post_processors

# Import transformer-specific components
from models.transformer_diffusion.transformer_diffusion_config import TransformerDiffusionConfig
from models.transformer_diffusion.transformer_diffusion_policy import TransformerDiffusionPolicy
from models.transformer_diffusion.processor_transformer_diffusion import make_pre_post_processors

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
    """Create data augmentations for training."""
    # Return RGBD-friendly augmentations
    return v2.Compose([
        # Gentle color jittering for RGB channels
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        # Mild geometric transforms to preserve physical consistency
        v2.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.98, 1.02)),
        # Randomly apply Gaussian noise with 30% probability
        v2.RandomApply([v2.GaussianNoise()], p=0.3),
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
        if "action" in batch:
            noise_scale = 0.01  # 1% of the typical action range
            noise = torch.randn_like(batch["action"]) * noise_scale
            batch["action"] = batch["action"] + noise
            
    return batch


# Helper to randomly drop one camera
def apply_camera_dropout(batch, camera_keys=["observation.images.gripper", "observation.images.depth"], dropout_prob=0.2):
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
        del batch[dropped_camera_key]
        # Uncomment the next line for debugging
        # print(f"Dropped camera: {dropped_camera_key}")
    
    return batch


def train(output_dir, dataset_id="ISdept/piper_arm", push_to_hub=False, resume_from_checkpoint=None, visualize_every_n_batches=1000):
    """Train the TransformerDiffusion model."""
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    training_steps = 100000 
    progress_update_freq = 200  # Single frequency for all progress updates
    checkpoint_freq = 1000
    visualization_freq = visualize_every_n_batches  # Save visualizations every N steps
    
    # Counter for batch visualization
    batch_counter = 0
    
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
    
    # Training parameters
    obs = 2
    horizon = 16
    n_action_steps = 8
    
    # Create transformer configuration
    cfg = TransformerDiffusionConfig(
        input_features=input_features, 
        output_features=output_features, 
        n_obs_steps=obs, 
        horizon=horizon, 
        n_action_steps=n_action_steps, 
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        state_dim=7,  # Adjust based on your robot's state dimension
        action_dim=7,  # Adjust based on your robot's action dimension
        d_model=512,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=512,
        diffusion_step_embed_dim=256,
        down_dims=(512, 1024, 2048),
        kernel_size=5,
        n_groups=8,
        use_film_scale_modulation=True
    )
    
    
    # Model loading logic
    if resume_from_checkpoint is not None:
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        # Load policy with its original configuration
        policy = TransformerDiffusionPolicy.from_pretrained(resume_from_checkpoint)
        policy.train()
        policy.to(device)
        
        # Use the policy's configuration for creating processors
        preprocessor, postprocessor = make_pre_post_processors(policy.config, dataset_stats=dataset_metadata.stats)
            
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
        # Cosine scheduler with warmup
        warmup_steps = 1000
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=training_steps
        )
        
        # Load optimizer state
        if not resume_from_checkpoint.startswith("http") and not resume_from_checkpoint.startswith("huggingface.co"):
            checkpoint_path = Path(resume_from_checkpoint)
            optimizer_state_path = checkpoint_path / "optimizer_state.pth"
            if optimizer_state_path.exists():
                optimizer.load_state_dict(torch.load(optimizer_state_path, map_location=device))
                print(f"Optimizer state loaded.")
            
            # Extract step
            if checkpoint_path.name.startswith("checkpoint-"):
                step = int(checkpoint_path.name.split("-")[1])
            else:
                step = 0
        else:
            step = 0
    else:
        # Initialize fresh policy
        policy = TransformerDiffusionPolicy(cfg)
        policy.train()
        policy.to(device)
        
        # Print total trainable parameters
        total_params = sum(p.numel() for p in policy.model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params:,}")
        # Ensure all submodules are on the correct device
        if hasattr(policy, 'transformer') and hasattr(policy.transformer, 'feature_projection'):
            if policy.transformer.feature_projection is not None:
                policy.transformer.feature_projection = policy.transformer.feature_projection.to(device)
        preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)
        step = 0
        
        # Implement differential learning rates for better gradient flow
        vision_params = []
        other_params = []
        
        for name, param in policy.named_parameters():
            if 'image_encoders' in name:
                vision_params.append(param)
            else:
                other_params.append(param)
        
        # Higher learning rate for vision encoders to improve gradient flow
        # Standard learning rate for other components
        optimizer = torch.optim.Adam([
            {'params': vision_params, 'lr': 1e-5},   # Higher LR for vision encoders
            {'params': other_params, 'lr': 1e-4}     # Standard LR for other components
        ], weight_decay=1e-4)
        
        # Store parameter groups for dynamic adjustment
        vision_param_group = optimizer.param_groups[0]
        other_param_group = optimizer.param_groups[1]
        
        # Cosine scheduler with warmup
        warmup_steps = 100
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
    
    delta_timestamps = {
        "observation.images.gripper": obs_temporal_window,  
        "observation.images.depth": obs_temporal_window,
        "observation.state": obs_temporal_window,
        "action": [i * frame_time for i in range(horizon)]
    }

    # Load dataset
    try:
        dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps, image_transforms=image_transforms, force_cache_sync=True, revision="main", tolerance_s=0.005)  # Tighter timestamp alignment
    except Exception as e:
        print(f"Error loading remote dataset: {e}")
        local_dataset_path = "./src/output" 
        print(f"Trying local dataset at {local_dataset_path}...")
        dataset = LeRobotDataset(local_dataset_path, delta_timestamps=delta_timestamps, force_cache_sync=True, tolerance_s=0.005)  # Tighter timestamp alignment

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=20,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Create visualizer
    visualizer = SpatialSoftmaxVisualizer(Path(output_dir) / "spatial_softmax_visualizations")

    # Training loop
    print("Starting training loop...")
    done = False
    epoch = 0
    prog_bar = tqdm(total=training_steps, desc="Training Progress", initial=step)
    while not done:
        epoch += 1
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)

            # Apply joint data augmentation
            #batch = apply_joint_augmentations(batch)
            
            # Apply camera dropout
            #batch = apply_camera_dropout(batch)

            # Preprocess (Normalize)
            batch = preprocessor(batch)

            # Forward & Backward
            loss, _ = policy.forward(batch)
            loss.backward()
            
            # Print gradient information for vision encoders, state encoder, and UNet (for debugging)
            if step % progress_update_freq == 0:  # Print every 10 progress update intervals
                print(f"\n--- Gradient Analysis at Step {step} ---")
                
                # Vision encoder gradients
                total_vision_grad = 0.0
                total_vision_params = 0
                print("\n--- Vision Encoder Gradients ---")
                for name, param in policy.model.named_parameters():
                    if 'image_encoders' in name and param.grad is not None:
                        grad_mean = param.grad.abs().mean().item()
                        param_count = param.numel()
                        total_vision_grad += grad_mean * param_count
                        total_vision_params += param_count
                
                if total_vision_params > 0:
                    avg_vision_grad = total_vision_grad / total_vision_params
                    print(f"Average vision encoder gradient: {avg_vision_grad:.6f}")
                
                # State encoder gradients
                total_state_grad = 0.0
                total_state_params = 0
                print("\n--- State Encoder Gradients ---")
                for name, param in policy.model.named_parameters():
                    if 'state_encoder' in name and param.grad is not None:
                        grad_mean = param.grad.abs().mean().item()
                        param_count = param.numel()
                        total_state_grad += grad_mean * param_count
                        total_state_params += param_count
                
                if total_state_params > 0:
                    avg_state_grad = total_state_grad / total_state_params
                    print(f"Average state encoder gradient: {avg_state_grad:.6f}")
                
                # UNet gradients
                total_unet_grad = 0.0
                total_unet_params = 0
                print("\n--- UNet Gradients ---")
                for name, param in policy.model.named_parameters():
                    if 'denoiser' in name and param.grad is not None:
                        grad_mean = param.grad.abs().mean().item()
                        param_count = param.numel()
                        total_unet_grad += grad_mean * param_count
                        total_unet_params += param_count
                
                if total_unet_params > 0:
                    avg_unet_grad = total_unet_grad / total_unet_params
                    print(f"Average UNet gradient: {avg_unet_grad:.6f}")
                
                print("--- End Gradient Analysis ---\n")
            
            # Calculate gradient norm for monitoring and clip gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
            
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
            
            # Save visualizations
            if step % visualization_freq == 0:
                # Get spatial softmax outputs for visualization
                with torch.no_grad():
                    policy.model.eval()
                    obs_context, spatial_outputs = policy.model.get_condition(batch)
                    policy.model.train()
                    
                    # Try to get episode index and frame index from batch if available
                    episode_index = None
                    if "episode_index" in batch:
                        episode_index = batch["episode_index"][0].item()  # Get first item in batch
                    
                    frame_index = None
                    if "frame_index" in batch:
                        frame_index = batch["frame_index"][0].item()  # Get first item in batch
                    
                    # Update visualizer with spatial outputs (multiple timesteps from the same window)
                    for cam_key, spatial_data in spatial_outputs.items():
                        if spatial_data is not None:
                            img_tensor, spatial_coords = spatial_data
                            if img_tensor is not None and spatial_coords is not None:
                                # Visualize all timesteps from the first item in the batch
                                # img_tensor shape: (B, T, C, H, W)
                                # spatial_coords shape: (B, T, num_points*2)
                                batch_idx = 0
                                # Save each timestep with proper zero-padded numbering for correct sorting
                                for t in range(img_tensor.shape[1]):  # Iterate through timesteps
                                    # Use zero-padded timestep numbering for proper sorting
                                    padded_t = f"{t:03d}"  # e.g., 000, 001, 002, ...
                                    visualizer.update(f"{cam_key}_t{padded_t}", img_tensor[batch_idx, t], spatial_coords[batch_idx, t])
                    
                    # Save visualizations with episode and frame index if available
                    visualizer.save_visualizations(step, episode=episode_index, frame=frame_index)
                    visualizer.reset_trajectories()
            
            # Save checkpoint
            if step > 0 and step % checkpoint_freq == 0:
                checkpoint_dir = output_directory / f"checkpoint-{step}"
                checkpoint_dir.mkdir(exist_ok=True)
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
