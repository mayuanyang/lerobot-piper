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


def train(output_dir, dataset_id="ISdept/piper_arm", push_to_hub=False, resume_from_checkpoint=None):
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    training_steps = 100000 
    log_freq = 10 # Reduced frequency to reduce console spam
    checkpoint_freq = 1000
    
    image_transforms = get_augmentations()

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
    horizon = 16
    n_action_steps = 8
    

    cfg = LongTaskDiffusionConfig(
        input_features=input_features, 
        output_features=output_features, 
        n_obs_steps=obs, 
        horizon=horizon, 
        n_action_steps=n_action_steps, 
        crop_shape=(320, 320),
        crop_is_random=True,
        use_separate_rgb_encoder_per_camera=True
    )
    
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
            
        optimizer = torch.optim.Adam(policy.parameters(), lr=2e-5)
        # Initialize learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
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
        "observation.images.front": obs_temporal_window,
        "observation.images.right": obs_temporal_window,
        "observation.state": obs_temporal_window,
        "action": [i * frame_time for i in range(horizon)]
    }

    try:
        dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps, image_transforms=image_transforms, force_cache_sync=True, revision="main", tolerance_s=0.01)
    except Exception as e:
        print(f"Error loading remote dataset: {e}")
        local_dataset_path = "./src/output" 
        print(f"Trying local dataset at {local_dataset_path}...")
        dataset = LeRobotDataset(local_dataset_path, delta_timestamps=delta_timestamps, force_cache_sync=True, tolerance_s=0.01)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=24,
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
