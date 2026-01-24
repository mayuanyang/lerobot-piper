#!/usr/bin/env python3
"""
Script to visualize SpatialSoftmax outputs during inference.
This script demonstrates how to use the visualization functionality.
"""

import torch
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import init_hydra_config
from spatial_softmax_visualizer import SpatialSoftmaxVisualizer


def visualize_spatial_softmax_during_inference(
    dataset_repo_id: str = "ISdept/piper-pick-place-depth",
    policy_repo_id: str = "ISdept/transformer-diffusion",
    output_dir: str = "spatial_softmax_inference_visualizations",
    num_episodes: int = 3,
    num_steps_per_episode: int = 10
):
    """
    Run inference on a trained policy and visualize SpatialSoftmax outputs.
    
    Args:
        dataset_repo_id: ID of the dataset repository
        policy_repo_id: ID of the policy repository
        output_dir: Directory to save visualizations
        num_episodes: Number of episodes to visualize
        num_steps_per_episode: Number of steps per episode to visualize
    """
    
    # Create visualizer
    visualizer = SpatialSoftmaxVisualizer(output_dir)
    
    # Load dataset (for preprocessing)
    dataset = LeRobotDataset(dataset_repo_id)
    
    # Load policy
    policy = make_policy(hydra_cfg=init_hydra_config(f"lerobot/configs/policy/{policy_repo_id.split('/')[-1]}.yaml"))
    policy.eval()
    
    # Get episode indices
    episode_indices = dataset.episode_data_index["from"]
    num_episodes = min(num_episodes, len(episode_indices))
    
    print(f"Visualizing SpatialSoftmax outputs for {num_episodes} episodes...")
    
    for episode_idx in range(num_episodes):
        print(f"\nProcessing episode {episode_idx + 1}/{num_episodes}")
        
        # Reset visualizer trajectories for each episode
        visualizer.reset_trajectories()
        
        # Get episode bounds
        start_idx = episode_indices[episode_idx].item()
        if episode_idx + 1 < len(episode_indices):
            end_idx = episode_indices[episode_idx + 1].item()
        else:
            end_idx = len(dataset)
            
        # Limit steps per episode
        end_idx = min(end_idx, start_idx + num_steps_per_episode)
        
        for step_idx in range(start_idx, end_idx):
            # Load data
            item = dataset[step_idx]
            
            # Convert to batch format
            batch = {key: value.unsqueeze(0) for key, value in item.items()}
            
            # Preprocess batch
            batch = dataset.prepare_for_training(batch)
            
            # Run inference
            with torch.no_grad():
                # Get policy predictions and spatial outputs
                predicted_actions, spatial_outputs = policy.model(batch)
                
                # Update visualizer with spatial outputs
                for cam_key, (img_tensor, spatial_coords) in spatial_outputs.items():
                    if img_tensor is not None and spatial_coords is not None:
                        # Take the first timestep for visualization
                        visualizer.update(cam_key, img_tensor[0], spatial_coords[0])
                
                # Save visualizations every few steps
                if (step_idx - start_idx) % 3 == 0:  # Save every 3rd step
                    global_step = episode_idx * num_steps_per_episode + (step_idx - start_idx)
                    visualizer.save_visualizations(global_step)
        
        # Save final visualizations for the episode
        global_step = (episode_idx + 1) * num_steps_per_episode
        visualizer.save_visualizations(global_step)
        print(f"Saved visualizations for episode {episode_idx + 1}")
    
    print(f"\nVisualization complete! Results saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize SpatialSoftmax outputs during inference")
    parser.add_argument("--dataset-repo-id", type=str, default="ISdept/piper-pick-place-depth",
                        help="Dataset repository ID")
    parser.add_argument("--policy-repo-id", type=str, default="ISdept/transformer-diffusion",
                        help="Policy repository ID")
    parser.add_argument("--output-dir", type=str, default="spatial_softmax_inference_visualizations",
                        help="Output directory for visualizations")
    parser.add_argument("--num-episodes", type=int, default=3,
                        help="Number of episodes to visualize")
    parser.add_argument("--num-steps-per-episode", type=int, default=10,
                        help="Number of steps per episode to visualize")
    
    args = parser.parse_args()
    
    visualize_spatial_softmax_during_inference(
        dataset_repo_id=args.dataset_repo_id,
        policy_repo_id=args.policy_repo_id,
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        num_steps_per_episode=args.num_steps_per_episode
    )
