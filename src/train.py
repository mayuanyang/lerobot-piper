from pathlib import Path

import torch
from tqdm import tqdm
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from src.models.long_task_diffusion.long_task_diffusion_config import LongTaskDiffusionConfig
from src.models.long_task_diffusion.long_task_diffusion_policy import LongTaskDiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

# 🟢 ADDED: Import torchvision for augmentation
from torchvision.transforms import v2
from torch.optim.lr_scheduler import ReduceLROnPlateau


# Detect the best available device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# 🟢 ADDED: Data Augmentation Setup
def get_rgb_augmentations():
    return v2.Compose([
        # 1. Color/Lighting is safe and helpful for RGB images
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        v2.RandomGrayscale(p=0.1),
        # 2. Gaussian Blur helps with motion blur during fast moves
        v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        # 3. DO NOT use Affine/Crop unless you transform the actions!
    ])

def get_depth_augmentations():
    return v2.Compose([
        # Only apply transforms that make sense for depth images
        # Gaussian Blur can help with noise in depth sensors
        v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
    ])

# Custom transform wrapper to handle different image types
class CameraSpecificTransforms:
    def __init__(self):
        self.rgb_transforms = get_rgb_augmentations()
        self.depth_transforms = get_depth_augmentations()
        
    def __call__(self, batch):
        # Apply transforms based on camera type
        # Use batch.keys() to safely check for keys
        batch_keys = batch.keys() if hasattr(batch, 'keys') else []
        
        if "observation.images.gripper" in batch_keys:
            batch["observation.images.gripper"] = self.rgb_transforms(batch["observation.images.gripper"])
            
        if "observation.images.depth" in batch_keys:
            batch["observation.images.depth"] = self.depth_transforms(batch["observation.images.depth"])
            
        return batch


# 🟢 ADDED: Helper to apply joint data augmentation
def apply_joint_augmentations(batch):
    """Apply random cropping to joint data (observation.state and action)"""
    # Randomly decide whether to apply augmentation (50% chance)
    if torch.rand(1).item() > 0.5:
        for key in ["observation.state"]:
            if key in batch and isinstance(batch[key], torch.Tensor):
                value = batch[key]
                # Generate a smaller random crop percentage (reduced to 0.05% max)
                max_crop_percentage = 0.0005  # 0.05%
                # Generate random crop percentage between [-max_crop_percentage, max_crop_percentage]
                crop_percentage = (torch.rand(1).item() - 0.5) * 2 * max_crop_percentage
                # Add independent Gaussian noise scaled by crop_percentage to each joint value
                # Using crop_percentage directly (with its sign) rather than abs(crop_percentage)
                noise = value * crop_percentage
                batch[key] = value + noise
    return batch


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
                
            # Move all tensor values in batch to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

            # Preprocess (Normalize)
            batch = preprocessor(batch)

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
            # But make sure we have at least a few batches for meaningful average
            if batch_idx >= min(10, len(val_dataloader) - 1):  # Validate on at least 10 batches or all available batches
                break
    
    policy.train()
    avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
    print(f"Validation completed: {val_batches} batches, total loss: {val_loss:.4f}, average loss: {avg_val_loss:.4f}")
    return avg_val_loss


def train(output_dir, dataset_id="ISdept/piper_arm", push_to_hub=False, resume_from_checkpoint=None):
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    training_steps = 100000 
    log_freq = 10 # Reduced frequency to reduce console spam
    checkpoint_freq = 1000
    
    # Initialize Transforms - using camera-specific transforms
    image_transforms = CameraSpecificTransforms()

    dataset_metadata = LeRobotDatasetMetadata(dataset_id, force_cache_sync=True, revision="main")
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    # 🟢 ADDED: Safety check
    if len(output_features) == 0:
        raise ValueError("No output features (actions) found! Check your dataset schema.")

    print('input_features:', input_features)
    print('output_features:', output_features)
    
    obs = 4
    horizon = 8
    n_action_steps = 4
    

    cfg = LongTaskDiffusionConfig(
        vision_backbone="resnet50",
        pretrained_backbone_weights="ResNet50_Weights.IMAGENET1K_V2",
        use_group_norm=False,
        input_features=input_features, 
        output_features=output_features, 
        n_obs_steps=obs, 
        horizon=horizon, 
        n_action_steps=n_action_steps, 
        crop_shape=(320, 320),
        crop_is_random=True,
        use_separate_rgb_encoder_per_camera=True
    )
    
    # Update image shapes - RGB images get 3 channels, depth images get 1 channel
    if "observation.images.gripper" in cfg.input_features:
        # This is the gripper camera (RGB)
        cfg.input_features["observation.images.gripper"].shape = [3, 400, 640]
    if "observation.images.depth" in cfg.input_features:
        # This is the depth camera (1 channel)
        cfg.input_features["observation.images.depth"].shape = [1, 400, 640]
    
    if dataset_metadata.stats is None:
        raise ValueError("Dataset stats are required to initialize the policy.")

    
    # --- MODEL LOADING LOGIC ---
    if resume_from_checkpoint is not None:
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        # Try to load LongTaskDiffusionPolicy first, fallback to DiffusionPolicy
        try:
            policy = LongTaskDiffusionPolicy.from_pretrained(resume_from_checkpoint)
        except:
            policy = DiffusionPolicy.from_pretrained(resume_from_checkpoint)
        policy.train()
        policy.to(device)
        
        try:
            from lerobot.policies.factory import load_pre_post_processors
            preprocessor, postprocessor = load_pre_post_processors(resume_from_checkpoint)
        except Exception as e:
            print(f"Could not load preprocessors: {e}. Creating new ones.")
            preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)
            
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
        # Initialize learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=1000)
        
        # Load Optimizer State
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
        # Initialize Fresh Policy
        policy = LongTaskDiffusionPolicy(cfg)
        policy.train()
        policy.to(device)
        preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)
        step = 0
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
        # Initialize learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1000)

    # Ensure preprocessors are on the correct device
    # (Some LeRobot versions keep them as modules)
    if isinstance(preprocessor, torch.nn.Module):
        preprocessor.to(device)

    # --- DATASET SETUP ---
    fps = 10
    frame_time = 1 / fps
    
    # Create observation temporal window for n_obs_steps frames
    obs_temporal_window = [ -i * frame_time for i in range(obs) ][::-1] # Reverse to get [-0.9, ... 0.0]
    
    delta_timestamps = {
        "observation.images.gripper": obs_temporal_window,  
        "observation.images.depth": obs_temporal_window,
        "observation.state": obs_temporal_window,
        "action": [i * frame_time for i in range(horizon)]
    }

    try:
        full_dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps, image_transforms=image_transforms, force_cache_sync=True, revision="main", tolerance_s=0.01)
    except Exception as e:
        print(f"Error loading remote dataset: {e}")
        local_dataset_path = "./src/output" 
        print(f"Trying local dataset at {local_dataset_path}...")
        full_dataset = LeRobotDataset(local_dataset_path, delta_timestamps=delta_timestamps, force_cache_sync=True, tolerance_s=0.01)
    
    # Get unique episode indices (convert tensors to Python values)
    episode_indices = list(set([int(idx.item()) if hasattr(idx, 'item') else int(idx) for idx in full_dataset.hf_dataset["episode_index"]]))
    episode_indices.sort()
    
    # Split episodes based on episode_index: use last 20% for validation, first 80% for training
    num_validation_episodes = max(1, len(episode_indices) // 5)  # 20% for validation
    train_episode_indices = episode_indices[:-num_validation_episodes]
    val_episode_indices = episode_indices[-num_validation_episodes:]
    
    print(f"Total episodes: {len(episode_indices)}")
    print(f"Training episodes: {len(train_episode_indices)}")
    print(f"Validation episodes: {len(val_episode_indices)}")
    
    # Create boolean masks for training and validation data
    dataset_episode_indices = [int(idx.item()) if hasattr(idx, 'item') else int(idx) for idx in full_dataset.hf_dataset["episode_index"]]
    train_mask = [idx in train_episode_indices for idx in dataset_episode_indices]
    val_mask = [idx in val_episode_indices for idx in dataset_episode_indices]
    
    # Create subsets for training and validation
    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, [i for i, mask in enumerate(train_mask) if mask])
    val_dataset = Subset(full_dataset, [i for i, mask in enumerate(val_mask) if mask])
    
    # Check dataset sizes
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    batch_size = 6
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

    # --- TRAINING LOOP ---
    print("Starting training loop...")
    done = False
    epoch = 0
    while not done:
        epoch += 1
        prog_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch}, Step {step}")
        for batch_idx, batch in prog_bar:
            
            # 1. Move to Device FIRST (Efficient)
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)

            # 3. Apply Joint Data Augmentation
            batch = apply_joint_augmentations(batch)

            # 4. Preprocess (Normalize)
            batch = preprocessor(batch)

            # 5. Forward & Backward
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # Update learning rate scheduler
            scheduler.step(loss)

            if step % log_freq == 0:
                # Get learning rate from optimizer
                lr = optimizer.param_groups[0]['lr']
                prog_bar.set_postfix({"step": step, "loss": f"{loss.item():.3f}", "lr": f"{lr:.2e}"})
            
            # Run validation every 500 steps
            if step > 0 and step % 500 == 0:
                print(f"\nRunning validation at step {step}...")
                val_loss = validate_model(policy, val_dataloader, preprocessor)
                print(f"Validation loss at step {step}: {val_loss:.4f}")
            
            # 5. Save Checkpoint
            if step > 0 and step % checkpoint_freq == 0:
                checkpoint_dir = output_directory / f"checkpoint-{step}"
                checkpoint_dir.mkdir(exist_ok=True)
                policy.save_pretrained(checkpoint_dir)
                preprocessor.save_pretrained(checkpoint_dir)
                postprocessor.save_pretrained(checkpoint_dir)
                #torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer_state.pth")
                print(f"\nCheckpoint saved at step {step}")
                
            step += 1
            if step >= training_steps:
                done = True
                break

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
