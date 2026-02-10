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
from lerobot.processor import ProcessorStepRegistry

# Import GridOverlayProcessorStep directly from the module
from models.transformer_diffusion.grid_overlay_processor import GridOverlayProcessorStep


# 🟢 ADDED: Import torchvision for augmentation
from torchvision.transforms import v2
from torch.utils.data import Subset

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

# Custom transform wrapper to handle different image types
class CameraSpecificTransforms:
    def __init__(self):
        self.rgb_transforms = get_rgb_augmentations()
        
    def __call__(self, batch):
        # Apply transforms based on camera type
        # Use batch.keys() to safely check for keys
        batch_keys = batch.keys() if hasattr(batch, 'keys') else []
        
        # Apply RGB transforms to all camera images
        camera_keys = [k for k in batch_keys if k.startswith("observation.images.")]
        for key in camera_keys:
            batch[key] = self.rgb_transforms(batch[key])
            
        return batch

# 🟢 ADDED: Helper to apply joint data augmentation
def apply_joint_augmentations(batch):
    key = "observation.state"
    if key not in batch:
        return batch

    q = batch[key]

    # Per-timestep noise
    noise = torch.randn_like(q) * 0.01  # ~0.6°
    mask = (torch.rand_like(q) < 0.3).float()

    batch[key] = q + noise * mask
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
        # Uncomment the next line for debugging
        # print(f"Dropped camera: {dropped_camera_key}")

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
                
            # Remap image feature names to match what the policy expects
            feature_mapping = {
                "observation.images.front": "observation.images.camera1",
                "observation.images.gripper": "observation.images.camera2",
                "observation.images.right": "observation.images.camera3"
            }
            
            # Create a new batch with remapped keys
            remapped_batch = {}
            for key, value in batch.items():
                # Remap image feature keys
                new_key = feature_mapping.get(key, key)
                remapped_batch[new_key] = value
            
            batch = remapped_batch
                            
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
            # But make sure we have at least a few batches for meaningful average
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
    log_freq = 10 # Reduced frequency to reduce console spam
    checkpoint_freq = 1000 

    dataset_metadata = LeRobotDatasetMetadata(dataset_id, force_cache_sync=True, revision="main")
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    
    # 🟢 ADDED: Safety check
    if len(output_features) == 0:
        raise ValueError("No output features (actions) found! Check your dataset schema.")

    print('input_features:', input_features)
    print('output_features:', output_features)

    
    n_obs_steps = 2
    chunk_size = 24
    n_action_steps = 24
    
    
    
    if dataset_metadata.stats is None:
        raise ValueError("Dataset stats are required to initialize the policy.")

    # Initialize Transforms - using camera-specific transforms
    image_transforms = CameraSpecificTransforms()

    # --- MODEL LOADING LOGIC ---
    if resume_from_checkpoint:
        
        policy = SmolVLAPolicy.from_pretrained(model_id)
        policy.train()
        policy.to(device)

        optimizer = torch.optim.Adam(policy.parameters(), lr=policy.config.optimizer_lr)
        policy.config.chunk_size = chunk_size
        policy.config.n_action_steps = n_action_steps
        policy.config.n_obs_steps = n_obs_steps
        
        
        preprocessor, postprocessor = make_pre_post_processors(policy.config, dataset_stats=dataset_metadata.stats)
        
        step = 0
    else:
        
        # Initialize a new model from configuration
        policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        
        if "observation.state" in policy.config.input_features:
            policy.config.input_features["observation.state"].shape = [7]  # 7 joints (not removing the 4th joint)
        
        if "action" in policy.config.output_features:
            policy.config.output_features["action"].shape = [7]  # 7 joints (not removing the 4th joint)
            
        # Update image shapes - All RGB images get 3 channels
        # Based on the feature mapping: front -> camera1, gripper -> camera2, right -> camera3
        if "observation.images.camera1" in policy.config.input_features:
            # This is the front camera (mapped from observation.images.front)
            policy.config.input_features["observation.images.camera1"].shape = [3, 400, 640]
        if "observation.images.camera2" in policy.config.input_features:
            # This is the gripper camera (mapped from observation.images.gripper)
            policy.config.input_features["observation.images.camera2"].shape = [3, 400, 640]
        if "observation.images.camera3" in policy.config.input_features:
            # This is the right camera (mapped from observation.images.right)
            policy.config.input_features["observation.images.camera3"].shape = [3, 400, 640]
        
        # Update normalization mapping to match command line arguments
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
        
        # Add grid overlay processor step to the preprocessor pipeline
        if hasattr(preprocessor, 'steps'):
            # Insert grid overlay step after the rename step but before normalization
            grid_step = GridOverlayProcessorStep(grid_cell_size=48, camera_names=["camera1", "camera3"])
            # Find the position to insert the grid step (after rename, before normalization)
            insert_pos = 1  # Default position (after Rename)
            for i, step in enumerate(preprocessor.steps):
                if hasattr(step, '__class__') and 'Normalizer' in step.__class__.__name__:
                    insert_pos = i
                    break
            preprocessor.steps.insert(insert_pos, grid_step)
                
        # Start from step 0
        step = 0
    
    def cosine_decay_with_warmup(step):
        """
        Cosine decay with warmup learning rate scheduler
        """
        warmup_steps = policy.config.scheduler_warmup_steps
        decay_steps = policy.config.scheduler_decay_steps
        initial_lr = policy.config.optimizer_lr
        final_lr = policy.config.scheduler_decay_lr
        
        if step < warmup_steps:
            # Warmup phase: linearly increase learning rate
            return step / warmup_steps
        elif step < warmup_steps + decay_steps:
            # Decay phase: cosine decay from initial_lr to final_lr
            progress = (step - warmup_steps) / decay_steps
            cosine_factor = (1 + torch.cos(torch.tensor(progress * 3.14159))) / 2
            return (final_lr + (initial_lr - final_lr) * cosine_factor) / initial_lr
        else:
            # After decay: constant final learning rate
            return final_lr / initial_lr
    
    scheduler = LambdaLR(optimizer, cosine_decay_with_warmup)

    # Ensure preprocessors are on the correct device
    # (Some LeRobot versions keep them as modules)
    if isinstance(preprocessor, torch.nn.Module):
        preprocessor.to(device)

    # --- DATASET SETUP ---
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
    full_dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps, image_transforms=image_transforms, force_cache_sync=True, revision="main", tolerance_s=0.01)
    
    # Get unique episode indices (convert tensors to Python values)
    episode_indices = list(set([int(idx.item()) if hasattr(idx, 'item') else int(idx) for idx in full_dataset.hf_dataset["episode_index"]]))
    episode_indices.sort()
    
    # Split episodes based on episode_index: >= 200 for validation, < 200 for training
    num_of_training_episodes = 25
    train_episode_indices = [idx for idx in episode_indices if idx < num_of_training_episodes]
    val_episode_indices = [idx for idx in episode_indices if idx >= num_of_training_episodes]
    
    print(f"Total episodes: {len(episode_indices)}")
    print(f"Training episodes: {len(train_episode_indices)}")
    print(f"Validation episodes: {len(val_episode_indices)}")
    
    # Check if we have any validation episodes
    if len(val_episode_indices) == 0:
        print("WARNING: No validation episodes found!")
        # Use a small portion of training episodes for validation if no validation episodes
        val_episode_indices = train_episode_indices[-10:]  # Last 10 training episodes
        train_episode_indices = train_episode_indices[:-10]  # Remaining training episodes
        print(f"Adjusted - Training episodes: {len(train_episode_indices)}, Validation episodes: {len(val_episode_indices)}")
    
    # Create boolean masks for training and validation data
    dataset_episode_indices = [int(idx.item()) if hasattr(idx, 'item') else int(idx) for idx in full_dataset.hf_dataset["episode_index"]]
    train_mask = [idx in train_episode_indices for idx in dataset_episode_indices]
    val_mask = [idx in val_episode_indices for idx in dataset_episode_indices]
    
    # Create subsets for training and validation
    
    train_dataset = Subset(full_dataset, [i for i, mask in enumerate(train_mask) if mask])
    val_dataset = Subset(full_dataset, [i for i, mask in enumerate(val_mask) if mask])
    
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

    # --- TRAINING LOOP ---
    print("Starting training loop...")
    done = False
    epoch = 0
    while not done:
        epoch += 1
        prog_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch}, Step {step}")
        for batch_idx, batch in prog_bar:
            
            # Remap image feature names to match what the policy expects
            feature_mapping = {
                "observation.images.front": "observation.images.camera1",
                "observation.images.gripper": "observation.images.camera2",
                "observation.images.right": "observation.images.camera3"
            }
            
            # Create a new batch with remapped keys
            remapped_batch = {}
            for key, value in batch.items():
                # Remap image feature keys
                new_key = feature_mapping.get(key, key)
                remapped_batch[new_key] = value
            
            batch = remapped_batch
            
            # Ensure observation.state has 7 dimensions (not removing 4th joint)
            if "observation.state" in batch:
                state = batch["observation.state"]
                if state.shape[-1] == 7:
                    # This is what we expect
                    pass
                else:
                    # Unexpected dimensionality
                    print(f"Warning: observation.state has unexpected shape {state.shape}")
            
            batch = preprocessor(batch)
                        
            batch = apply_joint_augmentations(batch)
            
            batch = random_drop_camera_views(batch, drop_prob=0.3)

            # 5. Move all tensor values in batch to device
            # Note: Some items may be strings or other non-tensor types that cannot be moved to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

            # 6. Forward & Backward
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate
            optimizer.zero_grad()

            if step % log_freq == 0:
                # Get learning rate from optimizer
                lr = optimizer.param_groups[0]['lr']
                prog_bar.set_postfix({"step": step, "loss": f"{loss.item():.3f}", "lr": f"{lr:.2e}"})
                # Explicitly delete variables to free up memory
                del loss
            
            # Run validation every 100 steps
            if step > 0 and step % 500 == 0:
                print(f"\nRunning validation at step {step}...")
                val_loss = validate_model(policy, val_dataloader, preprocessor)
                print(f"Validation loss at step {step}: {val_loss:.4f}")
            
            # 7. Save Checkpoint
            if step > 0 and step % checkpoint_freq == 0:
                checkpoint_dir = output_directory / f"checkpoint-{step}"
                checkpoint_dir.mkdir(exist_ok=True)
                policy.save_pretrained(checkpoint_dir)
                preprocessor.save_pretrained(checkpoint_dir)
                postprocessor.save_pretrained(checkpoint_dir)
                #torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer_state.pth")
                print(f"\nCheckpoint saved at step {step}")
                
                # Force garbage collection after checkpoint save
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            step += 1
            if step >= training_steps:
                done = True
                break
                
        # Force garbage collection after each epoch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
