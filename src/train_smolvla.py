from pathlib import Path
import gc

import torch
from tqdm import tqdm
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, SmolVLAConfig
from lerobot.datasets.utils import dataset_to_policy_features
import json


# 🟢 ADDED: Import torchvision for augmentation
from torchvision.transforms import v2
from lerobot.datasets.transforms import ImageTransforms, ImageTransformsConfig, ImageTransformConfig



# Detect the best available device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# 🟢 ADDED: Data Augmentation Setup
def get_augmentations():
    # Basic augmentation for Diffusion Policy often includes:
    # 1. Color Jitter (lighting invariance)
    # 2. Small translations/rotations (position invariance)
    return v2.Compose([
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        # Using RandomAffine instead of Crop because we don't know your exact image dimensions 
        # and don't want to accidentally crop out the gripper.
        v2.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    ])


# 🟢 ADDED: Helper to apply joint data augmentation
def apply_joint_augmentations(batch):
    """Apply margin-based augmentation to joint data (observation.state) with per-joint variations"""
    # Randomly decide whether to apply augmentation (50% chance)
    if torch.rand(1).item() > 0.5:
        key = "observation.state"
        if key in batch and isinstance(batch[key], torch.Tensor):
            value = batch[key]
            noise = torch.randn_like(value) * 0.003  # ~0.17°C standard deviation
            batch[key] = value + noise
    return batch


def random_drop_camera_views(batch, drop_prob=0.1):
    """Randomly drop camera views in the batch with a given probability."""
    camera_keys = [key for key in batch.keys() if key.startswith("observation.images.")]
    for key in camera_keys:
        if torch.rand(1).item() < drop_prob:
            batch[key] = torch.zeros_like(batch[key])  # Replace with zeros
            break # Drop only one camera view per batch
    return batch

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

    policy = SmolVLAPolicy.from_pretrained(model_id)
    cfg = policy.config

    cfg.n_obs_steps = 2
    cfg.chunk_size = 50
    cfg.n_action_steps = 50
    
    # Update the configuration to use 7-dimensional state and action
    # and 400x640 images
    if "observation.state" in cfg.input_features:
        cfg.input_features["observation.state"].shape = [7]
    if "action" in cfg.output_features:
        cfg.output_features["action"].shape = [7]
        
    # Update image shapes to 400x640
    for img_key in ["observation.images.camera1", "observation.images.camera2", "observation.images.camera3"]:
        if img_key in cfg.input_features:
            cfg.input_features[img_key].shape = [3, 400, 640]
    
    if dataset_metadata.stats is None:
        raise ValueError("Dataset stats are required to initialize the policy.")

    # Initialize Transforms
    image_transforms = get_augmentations()

    # --- MODEL LOADING LOGIC ---
    if resume_from_checkpoint:
        
        policy = SmolVLAPolicy.from_pretrained(model_id)
        policy.train()
        policy.to(device)

        optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.optimizer_lr)
        policy.config.chunk_size = cfg.chunk_size
        policy.config.n_action_steps = cfg.n_action_steps
        policy.config.n_obs_steps = cfg.n_obs_steps
        
        preprocessor, postprocessor = make_pre_post_processors(policy.config, dataset_stats=dataset_metadata.stats)
        
        step = 0
    else:
        print("Starting fresh training from scratch", cfg)
        # Initialize a new model from configuration
        policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        policy.config.chunk_size = cfg.chunk_size
        policy.config.n_action_steps = cfg.n_action_steps
        policy.config.n_obs_steps = cfg.n_obs_steps
        
        policy.train()
        policy.to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.optimizer_lr)
        # Create new preprocessor and postprocessor
        preprocessor, postprocessor = make_pre_post_processors(policy.config, dataset_stats=dataset_metadata.stats)
                
        # Start from step 0
        step = 0

    # Create learning rate scheduler with warmup and cosine decay
    from torch.optim.lr_scheduler import LambdaLR
    
    def cosine_decay_with_warmup(step):
        """
        Cosine decay with warmup learning rate scheduler
        """
        warmup_steps = cfg.scheduler_warmup_steps
        decay_steps = cfg.scheduler_decay_steps
        initial_lr = cfg.optimizer_lr
        final_lr = cfg.scheduler_decay_lr
        
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
    
    obs_temporal_window = [ -i * frame_time for i in range(cfg.n_obs_steps) ][::-1]

    delta_timestamps = {
        "observation.images.gripper": obs_temporal_window,  
        "observation.images.right": obs_temporal_window,
        "observation.images.front": obs_temporal_window,
        "observation.state": obs_temporal_window,
        "action": [i * frame_time for i in range(cfg.n_action_steps)] 
    }

    dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps, image_transforms=image_transforms, force_cache_sync=True, revision="main", tolerance_s=0.01)
    

    batch_size = 36
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # --- TRAINING LOOP ---
    print("Starting training loop...")
    done = False
    epoch = 0
    while not done:
        epoch += 1
        prog_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}, Step {step}")
        for batch_idx, batch in prog_bar:
            batch["task"] = ["Pick and place the cube into the container"] * batch_size

            # 🟢 ADDED: Add frame_index to let the model know the position in the sequence
            # The frame_index field from the dataset represents the local frame index within an episode
            # if "frame_index" in batch:
            #     batch["observation.sequence_index"] = batch["frame_index"].unsqueeze(-1)  # Add dimension to match expected shape

            # Remap image feature names to match what the policy expects
            feature_mapping = {
                "observation.images.front": "observation.images.camera1",
                "observation.images.right": "observation.images.camera2", 
                "observation.images.gripper": "observation.images.camera3"
            }
            
            # Create a new batch with remapped keys
            remapped_batch = {}
            for key, value in batch.items():
                # Remap image feature keys
                new_key = feature_mapping.get(key, key)
                remapped_batch[new_key] = value
            
            batch = remapped_batch
            
            # Ensure observation.state has 7 dimensions as expected by our dataset
            # The pretrained model might expect 6 dimensions, but our dataset has 7
            if "observation.state" in batch:
                state = batch["observation.state"]
                if state.shape[-1] == 6:
                    # If we somehow get 6-dimensional state, we need to handle it
                    # But according to the user, the dataset provides 7-dimensional state
                    print(f"Warning: observation.state has shape {state.shape}, expected 7 dimensions")
                elif state.shape[-1] == 7:
                    # This is what we expect - 7 dimensional state from the dataset
                    pass
                else:
                    # Unexpected dimensionality
                    print(f"Warning: observation.state has unexpected shape {state.shape}")
            
            batch = preprocessor(batch)
            
            batch = apply_joint_augmentations(batch)
            
            batch = random_drop_camera_views(batch, drop_prob=0.2)

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
