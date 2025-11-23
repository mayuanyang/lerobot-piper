from pathlib import Path

import torch
from tqdm import tqdm
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
# Import JointSmoothDiffusion instead of DiffusionPolicy
from models.smooth_diffusion.joint_smooth_diffusion import JointSmoothDiffusion

# Detect the best available device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        # Return previous 1 frame, itself, and next 13 frames
        return [i / fps for i in range(-1, 14)]  # -1, 0, 1, 2, ..., 13

    return [i / fps for i in delta_indices]


def train(output_dir, dataset_id="ISdept/piper_arm", push_to_hub=False, resume_from_checkpoint=None):
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

        # Number of offline training steps (we'll only do offline training for this example.)
    # Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
    training_steps = 30000  # Increased to demonstrate checkpoint saving
    log_freq = 1
    checkpoint_freq = 1000  # Save checkpoint every 1000 steps

    # When starting from scratch (i.e. not from a pretrained policy), we need to specify 2 things before
    # creating the policy:
    #   - input/output shapes: to properly size the policy
    #   - dataset stats: for normalization and denormalization of input/outputs
    dataset_metadata = LeRobotDatasetMetadata(dataset_id, force_cache_sync=True, revision="main")
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    print('input_features:', input_features)
    print('output_features:', output_features)

    # Policies are initialized with a configuration class, in this case `DiffusionConfig`. For this example,
    # we'll just use the defaults and so no arguments other than input/output features need to be passed.
    # NOTE: We need to update n_obs_steps to match our obs_temporal_window length (4 steps)
    # Also explicitly set horizon to match our action sequence length (16 steps)
    # Fixed: Set use_group_norm=False when using pretrained weights to avoid BatchNorm replacement error
    cfg = DiffusionConfig(input_features=input_features, output_features=output_features, n_obs_steps=10, horizon=16, pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1", use_group_norm=False, use_separate_rgb_encoder_per_camera=True)
    
    if dataset_metadata.stats is None:
        raise ValueError("Dataset stats are required to initialize the policy.")

    # Check if we're resuming from a checkpoint
    if resume_from_checkpoint is not None:
        checkpoint_path = Path(resume_from_checkpoint)
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint path {resume_from_checkpoint} does not exist")
        
        # Extract step number from checkpoint directory name
        checkpoint_name = checkpoint_path.name
        if checkpoint_name.startswith("checkpoint-"):
            step = int(checkpoint_name.split("-")[1])
        else:
            raise ValueError(f"Invalid checkpoint directory name: {checkpoint_name}. Expected format: checkpoint-<step>")
        
        print(f"Resuming training from checkpoint at step {step}")
        
        # Load policy, preprocessor, and postprocessor from checkpoint
        policy = JointSmoothDiffusion.from_pretrained(checkpoint_path)
        policy.train()
        policy.to(device)
        
        # Try to load preprocessors from checkpoint, fallback to creating new ones
        try:
            from lerobot.policies.factory import load_pre_post_processors
            preprocessor, postprocessor = load_pre_post_processors(checkpoint_path)
        except Exception as e:
            print(f"Could not load preprocessors from checkpoint: {e}. Creating new preprocessors.")
            preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)
            
        # Create optimizer and try to load its state
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
        optimizer_state_path = checkpoint_path / "optimizer_state.pth"
        if optimizer_state_path.exists():
            try:
                optimizer.load_state_dict(torch.load(optimizer_state_path))
                print(f"Optimizer state loaded from checkpoint")
            except Exception as e:
                print(f"Could not load optimizer state from checkpoint: {e}")
        else:
            print("No optimizer state found in checkpoint")
    else:
        # We can now instantiate our policy with this config and the dataset stats.
        # Use JointSmoothDiffusion instead of DiffusionPolicy
        policy = JointSmoothDiffusion(cfg, velocity_loss_weight=1.0, acceleration_loss_weight=0.5, jerk_loss_weight=0.1)  # Add all loss weight parameters
        policy.train()
        policy.to(device)
        preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)
        step = 0
        # Create optimizer
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

 
    fps = 10
    frame_time = 1 / fps  # 0.1 seconds
    obs_temporal_window = [
        -9 * frame_time,
        -8 * frame_time,
        -7 * frame_time,
        -6 * frame_time,
        -5 * frame_time,
        -4 * frame_time,
        -3 * frame_time,
        -2 * frame_time,
        -1 * frame_time,
        0.0              
    ]

    delta_timestamps = {
        # 🟢 NEW: EXPLICITLY list all camera keys with the same temporal sequence
        "observation.images.gripper": obs_temporal_window,  
        "observation.images.rgb": obs_temporal_window,
        "observation.images.depth": obs_temporal_window,
        
        
        # NOTE: observation.states is usually low-dimensional proprioception
        # and should be named "observation.state" (singular) in most LeRobot datasets.
        "observation.state": obs_temporal_window,  # Assuming 'observation.state' is correct feature name
        
        # Action stream remains the same, as it's independent of the cameras
        "action": [
            0.0 * frame_time, 
            1 * frame_time, 
            2 * frame_time, 
            3 * frame_time, 
            4 * frame_time, 
            5 * frame_time, 
            6 * frame_time, 
            7 * frame_time, 
            8 * frame_time, 
            9 * frame_time, 
            10 * frame_time, 
            11 * frame_time, 
            12 * frame_time, 
            13 * frame_time, 
            14 * frame_time, 
            15 * frame_time, 
        ]
    }

    # We can then instantiate the dataset with these delta_timestamps configuration.
    # NOTE: Using local dataset instead of remote to avoid schema mismatch
    try:
        dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps, force_cache_sync=True, revision="main", tolerance_s=0.01)
    except Exception as e:
        print(f"Error loading dataset with ID {dataset_id}: {e}")
        print("Trying to load local dataset...")
        # Try to load from local path
        local_dataset_path = "./src/output"  # Adjust this path as needed
        dataset = LeRobotDataset(local_dataset_path, delta_timestamps=delta_timestamps, force_cache_sync=True, tolerance_s=0.01)

    # Then we create our dataloader for offline training.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=16,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Run training loop.
    step = 0
    done = False
    while not done:
        prog_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Training Step {step}")
        for batch_idx, batch in prog_bar:
            batch = preprocessor(batch)
            # Move batch tensors to the correct device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                prog_bar.set_postfix({"step": step, "loss": f"{loss.item():.3f}"})
            
            # Save checkpoint every checkpoint_freq steps
            if step > 0 and step % checkpoint_freq == 0:
                checkpoint_dir = output_directory / f"checkpoint-{step}"
                checkpoint_dir.mkdir(exist_ok=True)
                policy.save_pretrained(checkpoint_dir)
                preprocessor.save_pretrained(checkpoint_dir)
                postprocessor.save_pretrained(checkpoint_dir)
                # Save optimizer state
                torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer_state.pth")
                print(f"Checkpoint saved at step {step}")
                
            step += 1
            if step >= training_steps:
                done = True
                break

    policy.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)

    if push_to_hub:
        repo = 'ISdept/piper_arm_diffusion'
        policy.push_to_hub(repo)
        preprocessor.push_to_hub(repo)
        postprocessor.push_to_hub(repo)
        postprocessor.push_to_hub(repo)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train the diffusion policy")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints and final model")
    parser.add_argument("--dataset_id", type=str, default="ISdept/piper_arm", help="Dataset ID to use for training")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the model to Hugging Face Hub")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint directory to resume training from")
    
    args = parser.parse_args()
    
    train(
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        push_to_hub=args.push_to_hub,
        resume_from_checkpoint=args.resume_from_checkpoint
    )
