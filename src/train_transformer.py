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
from models.transformer_diffusion.processor_transformer_diffusion import make_transformer_diffusion_pre_post_processors

# Import torchvision for augmentation
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


# Data Augmentation Setup
def get_augmentations():
    """Create data augmentations for training."""
    return v2.Compose([
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        v2.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    ])


# Helper to apply joint data augmentation
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
                noise = value * crop_percentage
                batch[key] = value + noise
    return batch


def train(output_dir, dataset_id="ISdept/piper_arm", push_to_hub=False, resume_from_checkpoint=None):
    """Train the TransformerDiffusion model."""
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    training_steps = 100000 
    log_freq = 10
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
    
    # Training parameters
    obs = 8
    horizon = 16
    n_action_steps = 8
    
    # Create transformer configuration
    cfg = TransformerDiffusionConfig(
        input_features=input_features, 
        output_features=output_features, 
        n_obs_steps=obs, 
        horizon=horizon, 
        n_action_steps=n_action_steps, 
        crop_shape=(84, 84),
        crop_is_random=True,
        use_separate_rgb_encoder_per_camera=False,
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        state_dim=7,  # Adjust based on your robot's state dimension
        action_dim=7,  # Adjust based on your robot's action dimension
        d_model=128,
        nhead=4,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=512,
        dropout=0.1
    )
    
    if dataset_metadata.stats is None:
        raise ValueError("Dataset stats are required to initialize the policy.")

    # Model loading logic
    if resume_from_checkpoint is not None:
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        policy = TransformerDiffusionPolicy.from_pretrained(resume_from_checkpoint)
        policy.train()
        policy.to(device)
        
        try:
            from lerobot.policies.factory import load_pre_post_processors
            preprocessor, postprocessor = load_pre_post_processors(resume_from_checkpoint)
        except Exception as e:
            print(f"Could not load preprocessors: {e}. Creating new ones.")
            preprocessor, postprocessor = make_transformer_diffusion_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)
            
        optimizer = torch.optim.Adam(policy.parameters(), lr=2e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
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
        # Ensure all submodules are on the correct device
        if hasattr(policy, 'transformer') and hasattr(policy.transformer, 'feature_projection'):
            if policy.transformer.feature_projection is not None:
                policy.transformer.feature_projection = policy.transformer.feature_projection.to(device)
        preprocessor, postprocessor = make_transformer_diffusion_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)
        step = 0
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1000)

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
        "observation.images.front": obs_temporal_window,
        "observation.images.right": obs_temporal_window,
        "observation.state": obs_temporal_window,
        "action": [i * frame_time for i in range(horizon)]
    }

    # Load dataset
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
        batch_size=8,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Training loop
    print("Starting training loop...")
    done = False
    epoch = 0
    while not done:
        epoch += 1
        prog_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}, Step {step}")
        for batch_idx, batch in prog_bar:
            
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)

            # Apply joint data augmentation
            batch = apply_joint_augmentations(batch)

            # Preprocess (Normalize)
            batch = preprocessor(batch)

            # Forward & Backward
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
            
            # Save checkpoint
            if step > 0 and step % checkpoint_freq == 0:
                checkpoint_dir = output_directory / f"checkpoint-{step}"
                checkpoint_dir.mkdir(exist_ok=True)
                policy.save_pretrained(checkpoint_dir)
                preprocessor.save_pretrained(checkpoint_dir)
                postprocessor.save_pretrained(checkpoint_dir)
                print(f"\nCheckpoint saved at step {step}")
                
            step += 1
            if step >= training_steps:
                done = True
                break

    # Final save
    policy.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)

    if push_to_hub:
        repo = 'ISdept/piper_arm_transformer'
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
