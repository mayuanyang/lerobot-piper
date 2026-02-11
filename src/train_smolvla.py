from pathlib import Path
import gc

import torch
from tqdm import tqdm
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.datasets.utils import dataset_to_policy_features
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms import v2
from torch.utils.data import Subset
import random
import torchvision
from models.transformer_diffusion.grid_overlay_processor import GridOverlayProcessorStep

# Detect the best available device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# Data Augmentation Setup
def get_rgb_augmentations(img_size=(400, 640)):
    return v2.Compose([
        # 1. SPATIAL: The most important for generalization
        # Scales and shifts the image slightly so the model learns "relative" positions
        v2.RandomResizedCrop(size=img_size, scale=(0.9, 1.0), ratio=(0.95, 1.05), antialias=True),
        
        # 2. PHOTOMETRIC: Your existing color logic
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        
        # 3. SENSOR NOISE: Simulates real-world camera artifacts
        v2.RandomGrayscale(p=0.05),
        v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),
        
        # 4. SHARPNESS: Helps with fine edges of small objects
        v2.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
    ])


# Custom transform wrapper to handle different image types
class CameraSpecificTransforms:
    def __init__(self, apply_augmentations=True):
        self.apply_augmentations = apply_augmentations
        if apply_augmentations:
            self.rgb_transforms = get_rgb_augmentations()
        
    def __call__(self, data):
        # If not applying augmentations, return data unchanged
        if not self.apply_augmentations:
            return data
            
        # Handle both batch dictionaries and individual tensors
        if isinstance(data, dict):
            # This is a batch dictionary
            camera_keys = [k for k in data.keys() if k.startswith("observation.images.")]
            for key in camera_keys:
                data[key] = self.rgb_transforms(data[key])
            return data
        else:
            # This is a single tensor (individual image)
            return self.rgb_transforms(data)


# Helper to apply joint data augmentation
def apply_joint_augmentations(batch, std_tensor):
    key = "observation.state"
    if key not in batch:
        return batch

    q = batch[key]
    
    # 1. Create noise relative to the specific joint's STD
    # This ensures "quiet" joints don't get overwhelmed
    rel_noise = torch.randn_like(q) * (std_tensor * 0.05) # 5% of each joint's typical range
    
    # 2. Mask out joints that are disabled (where std is 0)
    active_mask = (std_tensor > 0).float()
    
    # 3. Apply the 30% probability mask
    prob_mask = (torch.rand_like(q) < 0.3).float()
    
    # Apply combined mask
    batch[key] = q + (rel_noise * prob_mask * active_mask)
    return batch


def random_drop_camera_views(batch, drop_prob=0.3):
    """Randomly drop one camera from the batch during training."""
    # Only apply during training and with specified probability
    if torch.rand(1).item() > drop_prob:
        return batch

    # Get camera keys
    camera_keys = [k for k in batch if k.startswith("observation.images.")]
    
    # If we only have one camera, don't drop it
    if len(camera_keys) <= 1:
        return batch

    # Randomly select one camera to drop
    camera_to_drop = torch.randint(0, len(camera_keys), (1,)).item()
    dropped_camera_key = camera_keys[camera_to_drop]

    # Remove the selected camera from the batch
    if dropped_camera_key in batch:
        del batch[dropped_camera_key]

    return batch


def create_feature_mapping(batch_keys):
    """Create a mapping from dataset camera names to policy camera names."""
    # Fixed camera name mapping
    camera_mapping = {
        'observation.images.front': 'observation.images.camera1',
        'observation.images.gripper': 'observation.images.camera2',
        'observation.images.right': 'observation.images.camera3'
    }
    
    # Create feature mapping based on available keys in the batch
    feature_mapping = {}
    for dataset_key, policy_key in camera_mapping.items():
        if dataset_key in batch_keys:
            feature_mapping[dataset_key] = policy_key
            
    return feature_mapping


def remap_batch_features(batch):
    """Remap batch features to match policy expectations."""
    feature_mapping = create_feature_mapping(batch.keys())
    
    # Create a new batch with remapped keys
    remapped_batch = {}
    for key, value in batch.items():
        new_key = feature_mapping.get(key, key)
        remapped_batch[new_key] = value
        
    return remapped_batch


def save_camera_frames(batch, step, output_dir, prefix="before_grid"):
    """Save frames for each camera to visualize grid overlay effect."""
    # Create directory for saved frames
    frames_dir = Path(output_dir) / "saved_frames"
    frames_dir.mkdir(exist_ok=True)
    
    saved_count = 0
    # Save frames only for specific camera keys
    target_keys = ['observation.images.camera1', 'observation.images.camera2', 'observation.images.camera3']
    
    for key in target_keys:
        if key in batch and isinstance(batch[key], torch.Tensor):
            value = batch[key]
            # Handle 5D tensor format [B, T, C, H, W]
            if len(value.shape) == 5:  # Batched format [batch_size, time_steps, channels, height, width]
                # Extract first frame from batch and first time step: value[0, 0, :, :, :]
                first_frame = value[0, 0]  # Shape: [channels, height, width]
            else:
                # Unsupported tensor shape, skip this key
                continue
            
            # Handle different image formats (should be 3D: CHW)
            if len(first_frame.shape) == 3:  # Multi-channel image (CHW format)
                channels, height, width = first_frame.shape
                
                # Convert to PIL Image (need to permute from CHW to HWC and scale to 0-255)
                if channels == 3:  # RGB image
                    # Ensure the tensor is in the correct range [0, 1]
                    img_tensor = torch.clamp(first_frame, 0, 1)
                    
                    # Convert to PIL Image directly (ToPILImage handles the conversion)
                    img = torchvision.transforms.ToPILImage()(img_tensor)
                    
                    # Save image
                    camera_name = key.replace("observation.images.", "")
                    filename = frames_dir / f"step_{step:06d}_{prefix}_{camera_name}.png"
                    img.save(filename)
                    saved_count += 1
                elif channels == 1:  # Grayscale image
                    # Squeeze the channel dimension and convert to PIL Image
                    img_tensor = first_frame.squeeze(0)  # Shape: [height, width]
                    img_tensor = torch.clamp(img_tensor, 0, 1)  # Ensure values are in [0, 1]
                    
                    # Convert to PIL Image
                    img = torchvision.transforms.ToPILImage()(img_tensor)
                    
                    # Save image
                    camera_name = key.replace("observation.images.", "")
                    filename = frames_dir / f"step_{step:06d}_{prefix}_{camera_name}.png"
                    img.save(filename)
                    saved_count += 1
            elif len(first_frame.shape) == 2:  # Grayscale image (HW format)
                height, width = first_frame.shape
                
                # Convert to PIL Image
                img_tensor = torch.clamp(first_frame, 0, 1)  # Ensure values are in [0, 1]
                
                # Convert to PIL Image
                img = torchvision.transforms.ToPILImage()(img_tensor)
                
                # Save image
                camera_name = key.replace("observation.images.", "")
                filename = frames_dir / f"step_{step:06d}_{prefix}_{camera_name}.png"
                img.save(filename)
                saved_count += 1
    
    # Only print a message if frames were saved
    if saved_count > 0:
        print(f"Saved {saved_count} camera frames for step {step} ({prefix})")


def validate_model(policy, val_dataloader, preprocessor):
    policy.eval()
    val_loss = 0.0
    val_batches = 0
    
    print(f"Starting validation with {len(val_dataloader)} batches...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            # Check if batch is empty or invalid
            if not batch:
                print(f"Warning: Empty batch at index {batch_idx}")
                continue
                
            # Remap features to match policy expectations
            batch = remap_batch_features(batch)
                            
            # State and action is already normalized
            batch = preprocessor(batch)
            
            # Move all tensor values in batch to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

            # Forward pass
            try:
                loss, _ = policy.forward(batch)
                # Check if loss is valid
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss value at batch {batch_idx}: {loss.item()}")
                    continue
                val_loss += loss.item()
                val_batches += 1
                print(f"Batch {batch_idx}: loss = {loss.item():.4f}")
            except Exception as e:
                print(f"Error during forward pass at batch {batch_idx}: {e}")
                continue
            
            # Limit validation to a reasonable number of batches to save time
            if batch_idx >= min(10, len(val_dataloader) - 1):  # Validate on at least 10 batches or all available batches
                break
    
    policy.train()
    avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
    print(f"Validation completed: {val_batches} batches, total loss: {val_loss:.4f}, average loss: {avg_val_loss:.4f}")
    return avg_val_loss


def train(output_dir, dataset_id="ISdept/piper_arm", model_id="ISdept/smolvla-piper", push_to_hub=False, resume_from_checkpoint=True):
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    training_steps = 100000 
    log_freq = 200
    checkpoint_freq = 1000 
    frame_save_freq = 5000  # Save frames every 100 steps

    dataset_metadata = LeRobotDatasetMetadata(dataset_id, force_cache_sync=True, revision="main")
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    # Safety check
    if len(output_features) == 0:
        raise ValueError("No output features (actions) found! Check your dataset schema.")

    print('input_features:', input_features)
    print('output_features:', output_features)

    n_obs_steps = 2
    chunk_size = 16
    n_action_steps = 16
    
    if dataset_metadata.stats is None:
        raise ValueError("Dataset stats are required to initialize the policy.")

    # Initialize Transforms
    train_image_transforms = CameraSpecificTransforms(apply_augmentations=True)
    val_image_transforms = CameraSpecificTransforms(apply_augmentations=False)
    training_step = 0

    # Model Loading Logic
    if resume_from_checkpoint:
        policy = SmolVLAPolicy.from_pretrained(model_id)
        policy.train()
        policy.to(device)

        optimizer = torch.optim.Adam(policy.parameters(), lr=policy.config.optimizer_lr)
        policy.config.chunk_size = chunk_size
        policy.config.n_action_steps = n_action_steps
        policy.config.n_obs_steps = n_obs_steps
        
        preprocessor, postprocessor = make_pre_post_processors(policy.config, dataset_stats=dataset_metadata.stats)
        
    else:
        # Initialize a new model from configuration
        policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        
        if "observation.state" in policy.config.input_features:
            policy.config.input_features["observation.state"].shape = [7]  # 7 joints
        
        if "action" in policy.config.output_features:
            policy.config.output_features["action"].shape = [7]  # 7 joints
            
        # Update image shapes - All RGB images get 3 channels
        if "observation.images.camera1" in policy.config.input_features:
            policy.config.input_features["observation.images.camera1"].shape = [3, 400, 640]
        if "observation.images.camera2" in policy.config.input_features:
            policy.config.input_features["observation.images.camera2"].shape = [3, 400, 640]
        if "observation.images.camera3" in policy.config.input_features:
            policy.config.input_features["observation.images.camera3"].shape = [3, 400, 640]
        
        # Update normalization mapping
        if hasattr(policy.config, 'normalization_mapping'):
            policy.config.normalization_mapping = {
                "VISUAL": "IDENTITY",
                "STATE": "MEAN_STD",
                "ACTION": "MEAN_STD"
            }
        policy.config.chunk_size = chunk_size
        policy.config.n_action_steps = n_action_steps
        policy.config.n_obs_steps = n_obs_steps
        policy.config.load_vlm_weights = True
        
        policy.train()
        policy.to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=policy.config.optimizer_lr)
        
        # Create preprocessor and postprocessor
        preprocessor, postprocessor = make_pre_post_processors(policy.config, dataset_stats=dataset_metadata.stats)
    
    def cosine_decay_with_warmup(scheduler_step):
        """
        Cosine decay with warmup learning rate scheduler
        """
        warmup_steps = policy.config.scheduler_warmup_steps
        decay_steps = policy.config.scheduler_decay_steps
        initial_lr = policy.config.optimizer_lr
        final_lr = policy.config.scheduler_decay_lr
        
        if scheduler_step < warmup_steps:
            # Warmup phase: linearly increase learning rate
            return scheduler_step / warmup_steps
        elif scheduler_step < warmup_steps + decay_steps:
            # Decay phase: cosine decay from initial_lr to final_lr
            progress = (scheduler_step - warmup_steps) / decay_steps
            cosine_factor = (1 + torch.cos(torch.tensor(progress * 3.14159))) / 2
            return (final_lr + (initial_lr - final_lr) * cosine_factor) / initial_lr
        else:
            # After decay: constant final learning rate
            return final_lr / initial_lr
    
    scheduler = LambdaLR(optimizer, cosine_decay_with_warmup)

    # Ensure preprocessors are on the correct device
    if isinstance(preprocessor, torch.nn.Module):
        preprocessor.to(device)

    # Dataset Setup
    fps = 10
    frame_time = 1 / fps
    
    obs_temporal_window = [ -i * frame_time for i in range(policy.config.n_obs_steps) ][::-1]

    # Dynamically build delta_timestamps based on available features in the dataset
    delta_timestamps = {}
    
    # Add camera features that exist in the dataset
    for feature_name in dataset_metadata.features.keys():
        if feature_name.startswith("observation.images."):
            delta_timestamps[feature_name] = obs_temporal_window
    
    # Add state and action features
    delta_timestamps["observation.state"] = obs_temporal_window
    delta_timestamps["action"] = [i * frame_time for i in range(policy.config.n_action_steps)]

    # Load full dataset to determine episode indices
    # Note: We'll create separate datasets for training and validation with different transforms
    # First, load a dataset without transforms to get episode indices
    temp_dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps, force_cache_sync=True, revision="main", tolerance_s=0.01)
    
    # Get unique episode indices
    episode_indices = list(set([int(idx.item()) if hasattr(idx, 'item') else int(idx) for idx in temp_dataset.hf_dataset["episode_index"]]))
    episode_indices.sort()
    
    # Split episodes: 95% for training, 5% for validation
    num_episodes = len(episode_indices)
    num_train_episodes = int(0.95 * num_episodes)
    
    # Shuffle episodes for random split
    shuffled_indices = episode_indices.copy()
    random.shuffle(shuffled_indices)
    
    train_episode_indices = shuffled_indices[:num_train_episodes]
    val_episode_indices = shuffled_indices[num_train_episodes:]
    
    print(f"Total episodes: {len(episode_indices)}")
    print(f"Training episodes: {len(train_episode_indices)} ({len(train_episode_indices)/len(episode_indices)*100:.1f}%)")
    print(f"Validation episodes: {len(val_episode_indices)} ({len(val_episode_indices)/len(episode_indices)*100:.1f}%)")
    
    # Create separate datasets for training and validation with appropriate transforms
    train_dataset = LeRobotDataset(
        dataset_id, 
        delta_timestamps=delta_timestamps, 
        image_transforms=train_image_transforms, 
        force_cache_sync=True, 
        revision="main", 
        tolerance_s=0.01
    )
    
    val_dataset = LeRobotDataset(
        dataset_id, 
        delta_timestamps=delta_timestamps, 
        image_transforms=val_image_transforms, 
        force_cache_sync=True, 
        revision="main", 
        tolerance_s=0.01
    )
    
    # Create boolean masks for training and validation data based on episode indices
    train_dataset_episode_indices = [int(idx.item()) if hasattr(idx, 'item') else int(idx) for idx in train_dataset.hf_dataset["episode_index"]]
    val_dataset_episode_indices = [int(idx.item()) if hasattr(idx, 'item') else int(idx) for idx in val_dataset.hf_dataset["episode_index"]]
    
    train_mask = [idx in train_episode_indices for idx in train_dataset_episode_indices]
    val_mask = [idx in val_episode_indices for idx in val_dataset_episode_indices]
    
    # Create subsets for training and validation
    train_dataset = Subset(train_dataset, [i for i, mask in enumerate(train_mask) if mask])
    val_dataset = Subset(val_dataset, [i for i, mask in enumerate(val_mask) if mask])
    
    # Check dataset sizes
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    batch_size = 36
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=4,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=4,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )
    
    # Check if we have any validation batches
    val_batches = len(val_dataloader)
    print(f"Validation batches: {val_batches}")
    if val_batches == 0:
        print("WARNING: No validation batches! Check batch_size and dataset size.")


    # Extract std_tensor for joint augmentations
    # Get the standard deviation for observation.state from dataset stats
    state_stats = dataset_metadata.stats["observation.state"]
    std_tensor = torch.tensor(state_stats["std"], dtype=torch.float32, device=device)
    
            # Add grid overlay processor step to the preprocessor pipeline
    if hasattr(preprocessor, 'steps'):
        # Insert grid overlay step after the rename step but before normalization
        
        grid_step = GridOverlayProcessorStep(grid_cell_size=40, camera_names=["camera1", "camera3"])
        # Find the position to insert the grid step (after rename, before normalization)
        insert_pos = 1  # Default position (after Rename)
        for i, proc_step in enumerate(preprocessor.steps):
            if hasattr(proc_step, '__class__') and 'Normalizer' in proc_step.__class__.__name__:
                insert_pos = i
                break
        preprocessor.steps.insert(insert_pos, grid_step)
    
    # Training Loop
    print("Starting training loop...")
    done = False
    epoch = 0
    # Calculate total steps per epoch
    total_steps_per_epoch = len(train_dataloader)
    # Initialize progress bar outside the epoch loop
    prog_bar = tqdm(total=training_steps, desc="Training Progress")

    while not done and training_step < training_steps:
        epoch += 1
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Update progress bar with current training step and total steps per epoch
            prog_bar.set_description(f"Training Progress (Step: {training_step}/{total_steps_per_epoch})")
            prog_bar.update(1)
            # Remap features to match policy expectations
            batch = remap_batch_features(batch)
            
            # Save frames before grid overlay for visualization
            if training_step % frame_save_freq == 0:
                save_camera_frames(batch, training_step, output_directory, prefix="before_grid")
            
            # Ensure observation.state has 7 dimensions
            if "observation.state" in batch:
                state = batch["observation.state"]
                if state.shape[-1] != 7:
                    print(f"Warning: observation.state has unexpected shape {state.shape}")
            
            batch = preprocessor(batch)
            
            # Save frames after grid overlay for visualization
            if training_step % frame_save_freq == 0:
                save_camera_frames(batch, training_step, output_directory, prefix="after_grid")
            
            batch = apply_joint_augmentations(batch, std_tensor)
            batch = random_drop_camera_views(batch, drop_prob=0.3)

            # Move all tensor values in batch to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

            # Forward & Backward
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate
            optimizer.zero_grad()

            # Update progress bar every step (not just log_freq)
            lr = optimizer.param_groups[0]['lr']
            
            
            if training_step % log_freq == 0:
                # Optional: Log to file or tensorboard here
                prog_bar.set_postfix({
                    "epoch": epoch,
                    "step": training_step,
                    "loss": f"{loss.item():.3f}",
                    "lr": f"{lr:.2e}"
                })
                
            
            # Run validation every 500 steps
            if training_step > 0 and training_step % 500 == 0:
                val_loss = validate_model(policy, val_dataloader, preprocessor)
                # Consider adding val_loss to postfix if you want to track it
            
            # Save Checkpoint
            if training_step > 0 and training_step % checkpoint_freq == 0:
                checkpoint_dir = output_directory / f"checkpoint-{training_step}"
                checkpoint_dir.mkdir(exist_ok=True)
                policy.save_pretrained(checkpoint_dir)
                preprocessor.save_pretrained(checkpoint_dir)
                postprocessor.save_pretrained(checkpoint_dir)
                print(f"Checkpoint saved at step {training_step}")
            
            training_step += 1
            
            # Check if we've reached training_steps
            if training_step >= training_steps:
                done = True
                break

    # Close progress bar when done
    prog_bar.close()
                
        

    # Final Save
    policy.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)

    if push_to_hub:
        repo = 'ISdept/piper_arm_diffusion'
        policy.push_to_hub(repo)
        preprocessor.push_to_hub(repo)
        postprocessor.push_to_hub(repo)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_id", type=str, default="ISdept/piper_arm")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()
    
    train(**vars(args))
