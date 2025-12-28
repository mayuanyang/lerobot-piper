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
from .models.smooth_diffusion.smooth_diffusion import SigLipDiffusionPolicy
from .models.sig_lip_diffusion.sig_lip_diffusion_config import SigLipDiffusionConfig
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
    # Create ImageTransformsConfig with desired augmentations
    cfg = ImageTransformsConfig(
        enable=True,
        max_num_transforms=2,  # Reduced from 3 to decrease memory usage
        random_order=True,
        tfs={
            "brightness": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"brightness": (0.9, 1.1)},  # Reduced range to decrease memory usage
            ),
            "contrast": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"contrast": (0.9, 1.1)},  # Reduced range to decrease memory usage
            ),
        }
    )
    
    # Create ImageTransforms object from config
    return ImageTransforms(cfg)


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


def train(output_dir, dataset_id="ISdept/piper_arm", model_id="ISdept/smolvla-piper", push_to_hub=False, resume_from_checkpoint=True):
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    training_steps = 100000 
    log_freq = 10 # Reduced frequency to reduce console spam
    checkpoint_freq = 1000 

    dataset_metadata = LeRobotDatasetMetadata(dataset_id, force_cache_sync=True, revision="main")
    if dataset_metadata.stats is None:
        raise ValueError("Dataset stats are required to initialize the policy.")
    
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    
    # 🟢 ADDED: Safety check
    if len(output_features) == 0:
        raise ValueError("No output features (actions) found! Check your dataset schema.")

    print('input_features:', input_features)
    print('output_features:', output_features)

    policy = SigLipDiffusionPolicy()
    cfg = SigLipDiffusionConfig()
    
    # Initialize Transforms
    image_transforms = get_augmentations()

    # --- MODEL LOADING LOGIC ---
    if resume_from_checkpoint:
        
        policy = SmolVLAPolicy.from_pretrained(model_id)
        policy.train()
        policy.to(device)

        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
        
        preprocessor, postprocessor = make_pre_post_processors(policy.config, dataset_stats=dataset_metadata.stats)
        
        step = 0
    else:
        print("Starting fresh training from scratch", cfg)
        # Initialize a new model from configuration
        policy.train()
        policy.to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
        # Create new preprocessor and postprocessor
        preprocessor, postprocessor = make_pre_post_processors(policy.config, dataset_stats=dataset_metadata.stats)
                
        # Start from step 0
        step = 0

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

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=1,
        batch_size=1,
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
            
            # The tokenizer processor expects a "task" key in the batch
            if "task_description" in batch:
                # Map task_description to task key for tokenizer processor
                batch["task"] = batch["task_description"]

            batch = preprocessor(batch)
            
            batch = apply_joint_augmentations(batch)

            # 5. Move all tensor values in batch to device
            # Note: Some items may be strings or other non-tensor types that cannot be moved to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

            # 6. Forward & Backward
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
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
