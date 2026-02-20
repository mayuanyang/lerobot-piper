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
        # Resize images to 224x224
        v2.Resize((224, 224)),
        # Gentle color jittering for RGB channels
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        # Mild geometric transforms to preserve physical consistency
        v2.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        # Randomly apply Gaussian noise with 30% probability
        #v2.RandomApply([v2.GaussianNoise()], p=0.3),
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


def apply_state_dropout(batch, state_key="observation.state", dropout_prob=0.3):
    """Randomly drop observation.state values by setting them to zero with 30% probability."""
    # Only apply during training and with specified probability
    if torch.rand(1).item() > dropout_prob:
        return batch
    
    # Apply dropout to observation.state
    if state_key in batch:
        batch[state_key] = torch.zeros_like(batch[state_key])
    
    return batch


def train(output_dir, dataset_id="ISdept/piper_arm", push_to_hub=False, resume_from_checkpoint=None, visualize_every_n_batches=1000):
    """Train the SimpleTransformerDiffusion model."""
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
    
    # Create transformer configuration - simplified version with smaller model
    cfg = TransformerDiffusionConfig(
        input_features=input_features, 
        output_features=output_features, 
        n_obs_steps=obs, 
        horizon=horizon, 
        n_action_steps=n_action_steps, 
        vision_backbone="vit_b_16",
        state_dim=7,  # 7 joints (not removing the 4th joint)
        action_dim=7,  # 7 joints (not removing the 4th joint)
        d_model=512,  # Smaller model for better gradient flow
        nhead=8,
        num_encoder_layers=4,  # Fewer layers
        num_decoder_layers=4,  # Configurable denoising transformer layers
        dim_feedforward=512,  # Smaller feedforward dimension
        diffusion_step_embed_dim=256,
        kernel_size=3,
        n_groups=8,
        num_cameras=3,  # Set number of cameras based on input features
        vision_freeze_layers=0  # UNFREEZE ALL LAYERS for better gradient flow
    )
    
    
    # Model loading logic
    if resume_from_checkpoint is not None:
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        # Load policy with its original configuration
        policy = TransformerDiffusionPolicy.from_pretrained(resume_from_checkpoint)
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
        print(f"Initializing optimizer with learning rate: 1e-4")
        total_trainable_params = sum(p.numel() for p in trainable_params)
        print(f"Number of trainable parameters: {total_trainable_params}")
        
        # Check if we have different learning rates for vision vs other parameters
        vision_params = []
        other_params = []
        
        for name, param in policy.named_parameters():
            if 'image_encoders' in name:
                vision_params.append(param)
            else:
                other_params.append(param)
        
        vision_param_count = sum(p.numel() for p in vision_params)
        other_param_count = sum(p.numel() for p in other_params)
        print(f"Vision parameters: {vision_param_count}, Other parameters: {other_param_count}")
        
        optimizer = torch.optim.Adam(trainable_params, lr=1e-4)
        
        # Print learning rate groups
        print(f"Optimizer parameter groups: {len(optimizer.param_groups)}")
        for i, group in enumerate(optimizer.param_groups):
            group_param_count = sum(p.numel() for p in group['params'])
            print(f"  Group {i}: lr={group['lr']}, params={group_param_count}")
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
        preprocessor, postprocessor = make_pre_post_processors(
            cfg, 
            dataset_stats=dataset_metadata.stats, 
            add_grid_overlay=True,
            grid_overlay_cameras=["front", "right"]  # Front and right cameras (original names)
        )
        step = 0
        
        # Implement differential learning rates for better gradient flow
        vision_params = []
        other_params = []
        
        for name, param in policy.named_parameters():
            if 'image_encoders' in name:
                vision_params.append(param)
            else:
                other_params.append(param)
        
        # Count actual number of parameters, not just parameter tensors
        vision_param_count = sum(p.numel() for p in vision_params)
        other_param_count = sum(p.numel() for p in other_params)
        print(f"Vision parameters: {vision_param_count}, Other parameters: {other_param_count}")
        
        trainable_params = [p for p in policy.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=1e-4)
        
        # Print learning rate groups
        print(f"Optimizer parameter groups: {len(optimizer.param_groups)}")
        for i, group in enumerate(optimizer.param_groups):
            group_param_count = sum(p.numel() for p in group['params'])
            print(f"  Group {i}: lr={group['lr']}, params={group_param_count}")
        
        # Cosine scheduler with warmup
        warmup_steps = 500
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
        "observation.images.front": obs_temporal_window,
        "observation.images.gripper": obs_temporal_window,  
        "observation.images.right": obs_temporal_window,
        "observation.state": obs_temporal_window,
        "action": action_temporal_window
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
        batch_size=16,  # Reduced batch size for better gradient flow
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
            batch = apply_joint_augmentations(batch)
            
            # Apply state dropout
            batch = apply_state_dropout(batch)
            
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
                
                # Print detailed gradients for VisionEncoder components (aggregated by component type)
                print("\n=== VisionEncoder Gradients (Per Camera) ===")
                
                # Check gradients for each camera encoder separately
                for camera_name, encoder in policy.model.image_encoders.items():
                    print(f"\n--- Camera: {camera_name} ---")
                    
                    # Check gradients for each component in this encoder
                    for component_name, component in encoder.named_children():
                        component_grad_norm = 0.0
                        component_param_count = 0
                        component_params_with_grad = 0
                        component_zero_grad_count = 0
                        
                        for param_name, param in component.named_parameters():
                            param_count = param.numel()
                            component_param_count += param_count
                            
                            if param.requires_grad:
                                if param.grad is not None:
                                    grad_norm = param.grad.abs().sum().item()
                                    component_grad_norm += grad_norm
                                    component_params_with_grad += param_count
                                    
                                    if grad_norm == 0.0:
                                        component_zero_grad_count += param_count
                                    elif grad_norm < 1e-6:
                                        print(f"  WARNING: Very small gradient in {component_name}.{param_name}: {grad_norm:.2e}")
                                else:
                                    print(f"  WARNING: No gradient for {component_name}.{param_name}")
                        
                        # Print component summary for this camera
                        if component_param_count > 0:
                            if component_params_with_grad > 0:
                                avg_grad_norm = component_grad_norm / component_params_with_grad
                                zero_grad_percent = (component_zero_grad_count / component_params_with_grad) * 100 if component_params_with_grad > 0 else 0
                                print(f"  {component_name}: avg_grad_norm={avg_grad_norm:.2e}, params={component_param_count}, "
                                      f"with_grad={component_params_with_grad}, zero_grad={zero_grad_percent:.1f}%")
                            else:
                                print(f"  {component_name}: NO GRADIENTS, params={component_param_count}")
                
                # Check for frozen backbone layers per camera
                print("\n=== Vision Backbone Freeze Status (Per Camera) ===")
                for camera_name, encoder in policy.model.image_encoders.items():
                    frozen_layers = 0
                    total_layers = 0
                    for name, param in encoder.backbone.named_parameters():
                        total_layers += 1
                        if not param.requires_grad:
                            frozen_layers += 1
                    print(f"  Camera {camera_name}: {frozen_layers}/{total_layers} layers frozen ({frozen_layers/total_layers*100:.1f}%)")
                
                # Aggregate gradients by component type across all cameras (existing code)
                print("\n=== VisionEncoder Gradients (Aggregated by Component) ===")
                component_gradients = {}
                
                for camera_name, encoder in policy.model.image_encoders.items():
                    for component_name, component in encoder.named_children():
                        if component_name not in component_gradients:
                            component_gradients[component_name] = {
                                'total_grad_norm': 0.0,
                                'total_param_count': 0,
                                'params_with_grad': 0,
                                'params_without_grad': 0,
                                'zero_grad_count': 0
                            }
                        
                        for param_name, param in component.named_parameters():
                            param_count = param.numel()
                            component_gradients[component_name]['total_param_count'] += param_count
                            
                            if param.requires_grad:
                                if param.grad is not None:
                                    grad_norm = param.grad.norm().item()
                                    component_gradients[component_name]['total_grad_norm'] += grad_norm
                                    component_gradients[component_name]['params_with_grad'] += param_count
                                    
                                    if grad_norm == 0.0:
                                        component_gradients[component_name]['zero_grad_count'] += param_count
                                else:
                                    component_gradients[component_name]['params_without_grad'] += param_count
                # Print aggregated gradients with more details
                for component_name, grad_info in component_gradients.items():
                    if grad_info['total_param_count'] > 0:
                        if grad_info['params_with_grad'] > 0:
                            avg_grad_norm = grad_info['total_grad_norm'] / grad_info['params_with_grad']
                            zero_grad_percent = (grad_info['zero_grad_count'] / grad_info['params_with_grad']) * 100 if grad_info['params_with_grad'] > 0 else 0
                            print(f"  {component_name}: avg_grad_norm={avg_grad_norm:.2e}, params={grad_info['total_param_count']}, "
                                  f"with_grad={grad_info['params_with_grad']}, zero_grad={zero_grad_percent:.1f}%")
                        else:
                            print(f"  {component_name}: NO GRADIENTS, params={grad_info['total_param_count']}")
                
                # Collect gradients for trainable components only
                total_vision_grad = 0.0
                total_vision_params = 0
                for name, param in policy.model.named_parameters():
                    if param.requires_grad and 'cross_camera_transformer' in name and param.grad is not None:
                        grad_mean = param.grad.abs().mean().item()
                        param_count = param.numel()
                        total_vision_grad += grad_mean * param_count
                        total_vision_params += param_count
                
                total_img_grad = 0.0
                total_img_params = 0
                for name, param in policy.model.named_parameters():
                    if param.requires_grad and 'image_encoders' in name and param.grad is not None:
                        grad_mean = param.grad.abs().mean().item()
                        param_count = param.numel()
                        total_img_grad += grad_mean * param_count
                        total_img_params += param_count
                
                # State encoder gradients
                total_state_grad = 0.0
                total_state_params = 0
                for name, param in policy.model.named_parameters():
                    if param.requires_grad and 'state_encoder' in name and param.grad is not None:
                        grad_mean = param.grad.abs().mean().item()
                        param_count = param.numel()
                        total_state_grad += grad_mean * param_count
                        total_state_params += param_count
                
                # Transformer encoder gradients
                total_encoder_grad = 0.0
                total_encoder_params = 0
                for name, param in policy.model.named_parameters():
                    if param.requires_grad and 'fusion_encoder' in name and param.grad is not None:
                        grad_mean = param.grad.abs().mean().item()
                        param_count = param.numel()
                        total_encoder_grad += grad_mean * param_count
                        total_encoder_params += param_count
                
                # Transformer denoiser gradients
                total_denoiser_grad = 0.0
                total_denoiser_params = 0
                for name, param in policy.model.named_parameters():
                    if param.requires_grad and 'denoising_transformer' in name and param.grad is not None:
                        grad_mean = param.grad.abs().mean().item()
                        param_count = param.numel()
                        total_denoiser_grad += grad_mean * param_count
                        total_denoiser_params += param_count
                
                # Print all gradients in one line
                grad_info = []
                
                if total_vision_params > 0:
                    avg_vision_grad = total_vision_grad / total_vision_params
                    grad_info.append(f"cross_camera_transformer: {avg_vision_grad:.6f}")
                    
                if total_img_params > 0:
                    avg_img_grad = total_img_grad / total_img_params
                    grad_info.append(f"image_encoders: {avg_img_grad:.6f}")
                
                if total_state_params > 0:
                    avg_state_grad = total_state_grad / total_state_params
                    grad_info.append(f"state_enc: {avg_state_grad:.6f}")
                
                if total_encoder_params > 0:
                    avg_encoder_grad = total_encoder_grad / total_encoder_params
                    grad_info.append(f"fusion_encoder: {avg_encoder_grad:.6f}")
                
                if total_denoiser_params > 0:
                    avg_denoiser_grad = total_denoiser_grad / total_denoiser_params
                    grad_info.append(f"denoising_transformer: {avg_denoiser_grad:.6f}")
                
                print(f"\nOverall Gradients -> {' | '.join(grad_info)}")
                print("--- End Gradient Analysis ---\n")
            
            # Calculate gradient norm for monitoring and clip gradients (only for trainable parameters)
            trainable_params = [p for p in policy.parameters() if p.requires_grad]
            
            
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
                    # Enable heatmap saving for debugging
                    heatmap_dir = Path(output_dir) / "heatmaps"
                    heatmap_dir.mkdir(exist_ok=True)
                    
                    # Enable heatmap saving in vision encoders
                    for encoder in policy.model.image_encoders.values():
                        encoder.save_heatmaps = True
                        encoder.heatmap_save_dir = heatmap_dir
                    
                    obs_context, spatial_outputs = policy.model.get_condition(batch, generate_heatmaps=True)
                    policy.model.train()
                    
                    # Disable heatmap saving after visualization
                    for encoder in policy.model.image_encoders.values():
                        encoder.save_heatmaps = False
                    
                    # Try to get episode index and frame index from batch if available
                    episode_index = None
                    if "episode_index" in batch:
                        episode_index = batch["episode_index"][0].item()  # Get first item in batch
                    
                    frame_index = None
                    if "frame_index" in batch:
                        frame_index = batch["frame_index"][0].item()  # Get first item in batch
                    
                    # Update visualizer with spatial outputs for all three cameras
                    for cam_key, spatial_coords in spatial_outputs.items():
                        if spatial_coords is not None and not cam_key.endswith('_heatmap'):
                            # Get the corresponding image tensor from the batch
                            # Convert cam_key back to batch key format
                            batch_key = cam_key.replace('_', '.')
                            if batch_key in batch:
                                img_tensor = batch[batch_key]
                                # Visualize all timesteps from the first item in the batch
                                # img_tensor shape: (B, T, C, H, W)
                                # spatial_coords shape: (B, T, K, 2)
                                batch_idx = 0
                                # Save each timestep with proper zero-padded numbering for correct sorting
                                for t in range(min(img_tensor.shape[1], spatial_coords.shape[1])):  # Iterate through timesteps
                                    # Use zero-padded timestep numbering for proper sorting
                                    padded_t = f"{t:03d}"  # e.g., 000, 001, 002, ...
                                    # Flatten coordinates to match visualizer expectations: (K, 2) -> (K*2,)
                                    flattened_coords = spatial_coords[batch_idx, t].flatten()
                                    visualizer.update(f"{cam_key}_t{padded_t}", img_tensor[batch_idx, t], flattened_coords)
                    
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
