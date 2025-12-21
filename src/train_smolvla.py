from pathlib import Path
import gc

import torch
from tqdm import tqdm
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from models.smooth_diffusion.custom_diffusion_config import CustomDiffusionConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, SmolVLAConfig

# 🟢 ADDED: Import torchvision for augmentation
from torchvision.transforms import v2
from lerobot.datasets.transforms import ImageTransforms, ImageTransformsConfig, ImageTransformConfig

# Import JointSmoothDiffusion instead of DiffusionPolicy
# Ensure this path is reachable from your running directory
try:
    from models.smooth_diffusion.joint_smooth_diffusion import JointSmoothDiffusion
except ImportError:
    # Fallback for checking script logic without the custom model
    print("WARNING: Custom JointSmoothDiffusion not found. Using standard DiffusionPolicy for syntax check.")
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy as JointSmoothDiffusion

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
            # "affine": ImageTransformConfig(
            #     weight=1.0,
            #     type="RandomAffine",
            #     kwargs={"degrees": (-3.0, 3.0), "translate": (0.03, 0.03)},  # Reduced range to decrease memory usage
            # ),
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
            # Generate per-joint random margin percentages (reduced to 0.01% max per joint)
            max_margin_percentage = 0.05  # 0.01% (reduced from 0.05%)
            # Generate random margin percentage for each joint independently
            # Shape will be [1, 1, num_joints] to broadcast correctly with [batch, time, joints]
            joint_margins = (torch.rand(1, 1, value.shape[-1]).to(value.device) - 0.5) * 2 * max_margin_percentage
            # Apply margin-based noise (multiplicative to simulate proportional variations)
            noise = value * joint_margins
            batch[key] = value + noise
    return batch


def train(output_dir, dataset_id="ISdept/piper_arm", push_to_hub=False, resume_from_checkpoint='ISdept/smolvla-piper'):
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

    cfg = SmolVLAConfig()
    
    if dataset_metadata.stats is None:
        raise ValueError("Dataset stats are required to initialize the policy.")

    # Initialize Transforms
    image_transforms = get_augmentations()

    # --- MODEL LOADING LOGIC ---
    if resume_from_checkpoint is not None:
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        policy = SmolVLAPolicy.from_pretrained(resume_from_checkpoint)
        policy.train()
        policy.to(device)
        
        try:
            from lerobot.policies.factory import load_pre_post_processors
            preprocessor, postprocessor = load_pre_post_processors(resume_from_checkpoint)
        except Exception as e:
            print(f"Could not load preprocessors: {e}. Creating new ones.")
            preprocessor, postprocessor = make_pre_post_processors(policy.config, dataset_stats=dataset_metadata.stats)
            
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
        
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
        raise NotImplemented

    # Ensure preprocessors are on the correct device
    # (Some LeRobot versions keep them as modules)
    if isinstance(preprocessor, torch.nn.Module):
        preprocessor.to(device)

    # --- DATASET SETUP ---
    fps = 10
    frame_time = 1 / fps
    
    # Align with piper_smolvla.config chunk_size of 50
    # Using 25 observation frames to balance with action frames
    obs_temporal_window = [ -i * frame_time for i in range(25) ][::-1] # 25 frames
    
    delta_timestamps = {
        "observation.images.gripper": obs_temporal_window,  
        "observation.images.right": obs_temporal_window,
        "observation.images.front": obs_temporal_window,
        "observation.state": obs_temporal_window,
        "action": [i * frame_time for i in range(50)]  # Match chunk_size of 50
    }

    try:
        dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps, image_transforms=image_transforms, force_cache_sync=True, revision="main", tolerance_s=0.01)
    except Exception as e:
        print(f"Error loading remote dataset: {e}")
        local_dataset_path = "./src/output" 
        print(f"Trying local dataset at {local_dataset_path}...")
        dataset = LeRobotDataset(local_dataset_path, delta_timestamps=delta_timestamps, image_transforms=image_transforms, force_cache_sync=True, tolerance_s=0.01)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=8,
        batch_size=12,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # --- TRAINING LOOP ---
    print("Starting training loop...")
    done = False
    while not done:
        prog_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Training Step {step}")
        for batch_idx, batch in prog_bar:
            

            # 2. Apply Joint Data Augmentation
            batch = apply_joint_augmentations(batch)

            # 4. Ensure task information is properly formatted for tokenizer
            # The tokenizer processor expects a "task" key in the batch
            if "task_description" in batch:
                # Map task_description to task key for tokenizer processor
                batch["task"] = batch["task_description"]

            # 4. Preprocess (Normalize)
            batch = preprocessor(batch)

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
                prog_bar.set_postfix({"step": step, "loss": f"{loss.item():.3f}"})
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
