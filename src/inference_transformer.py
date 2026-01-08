#!/usr/bin/env python

"""
Inference script specifically for the LongTaskTransformer model.
"""

import torch
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType
from lerobot.policies.factory import load_pre_post_processors

# Import transformer-specific components
from src.models.long_task_transformer.long_task_transformer_policy import LongTaskTransformerPolicy


# Detect the best available device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


def run_inference(model_path, dataset_id="ISdept/piper_arm"):
    """Run inference using the trained LongTaskTransformer model."""
    print(f"Loading model from: {model_path}")
    
    # Load the trained policy
    policy = LongTaskTransformerPolicy.from_pretrained(model_path)
    policy.eval()
    policy.to(device)
    
    # Load preprocessors
    preprocessor, postprocessor = load_pre_post_processors(model_path)
    preprocessor.to(device)
    
    # Load dataset metadata for context
    dataset_metadata = LeRobotDatasetMetadata(dataset_id, force_cache_sync=True, revision="main")
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    print('Input features:', input_features)
    print('Output features:', output_features)
    
    # For demonstration, we'll create a mock observation
    # In practice, you would get this from your robot's sensors
    batch_size = 1
    obs_steps = policy.config.n_obs_steps
    state_dim = policy.config.state_dim
    
    # Create mock observation state (replace with real sensor data)
    mock_observation = {
        "observation.state": torch.randn(batch_size, obs_steps, state_dim).to(device)
    }
    
    print(f"Created mock observation with shape: {mock_observation['observation.state'].shape}")
    
    # Preprocess the observation
    processed_observation = preprocessor(mock_observation)
    
    # Run inference
    with torch.no_grad():
        predicted_actions = policy.select_action(processed_observation)
    
    print(f"Predicted actions shape: {predicted_actions.shape}")
    print(f"Action steps: {policy.config.n_action_steps}")
    print(f"Action dimension: {policy.config.action_dim}")
    
    # Post-process actions (denormalize)
    # Note: This would typically be done with the postprocessor, but for simplicity
    # we're just showing the raw predicted actions
    print("First few predicted actions:")
    print(predicted_actions[0, :3, :])  # Show first 3 actions of first batch
    
    return predicted_actions


def run_inference_on_dataset(model_path, dataset_id="ISdept/piper_arm"):
    """Run inference on a sample from the dataset."""
    print(f"Loading model from: {model_path}")
    
    # Load the trained policy
    policy = LongTaskTransformerPolicy.from_pretrained(model_path)
    policy.eval()
    policy.to(device)
    
    # Load preprocessors
    preprocessor, postprocessor = load_pre_post_processors(model_path)
    preprocessor.to(device)
    
    # Setup dataset
    obs = policy.config.n_obs_steps
    horizon = policy.config.horizon
    
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
        dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps, force_cache_sync=True, revision="main", tolerance_s=0.01)
    except Exception as e:
        print(f"Error loading remote dataset: {e}")
        local_dataset_path = "./src/output" 
        print(f"Trying local dataset at {local_dataset_path}...")
        dataset = LeRobotDataset(local_dataset_path, delta_timestamps=delta_timestamps, force_cache_sync=True, tolerance_s=0.01)
    
    # Get a sample from the dataset
    sample = dataset[0]  # First sample
    
    # Convert to batch format
    batch = {}
    for key in sample:
        batch[key] = sample[key].unsqueeze(0).to(device)  # Add batch dimension
    
    print(f"Loaded sample with keys: {list(batch.keys())}")
    
    # Preprocess the observation
    processed_batch = preprocessor(batch)
    
    # Run inference
    with torch.no_grad():
        predicted_actions = policy.select_action(processed_batch)
    
    print(f"Predicted actions shape: {predicted_actions.shape}")
    print(f"Ground truth actions shape: {batch['action'].shape}")
    
    # Compare with ground truth (for evaluation purposes)
    ground_truth = batch['action'][:, :policy.config.n_action_steps, :]
    mse = torch.mean((predicted_actions - ground_truth) ** 2)
    print(f"MSE between predicted and ground truth actions: {mse.item():.6f}")
    
    return predicted_actions, ground_truth


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--dataset_id", type=str, default="ISdept/piper_arm")
    parser.add_argument("--use_dataset", action="store_true", help="Use a dataset sample for inference")
    args = parser.parse_args()
    
    if args.use_dataset:
        predicted_actions, ground_truth = run_inference_on_dataset(args.model_path, args.dataset_id)
    else:
        predicted_actions = run_inference(args.model_path, args.dataset_id)
