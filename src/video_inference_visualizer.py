"""
Visualization tool for video inference results.
This script visualizes the action arrays from video inference results,
where the action array represents joints 1-6 and the gripper.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys

def load_inference_results(results_path):
    """
    Load inference results from JSON file.
    
    Args:
        results_path (str): Path to the JSON file containing inference results
        
    Returns:
        list: List of inference results
    """
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} inference results")
        return results
    except Exception as e:
        print(f"Error loading results: {e}")
        return []

def extract_actions(results):
    """
    Extract action arrays from inference results.
    
    Args:
        results (list): List of inference results
        
    Returns:
        list: List of action arrays
    """
    actions = []
    frame_indices = []
    
    for result in results:
        if result.get("success", False) and "result" in result and "action" in result["result"]:
            action = np.array(result["result"]["action"])
            actions.append(action)
            frame_indices.append(result.get("frame_index", len(actions)-1))
    
    print(f"Extracted {len(actions)} action arrays")
    return actions, frame_indices

def plot_single_action(action, title="Action Visualization"):
    """
    Plot a single action array.
    
    Args:
        action (np.array): Action array with 7 values (joints 1-6 + gripper)
        title (str): Title for the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Joint names
    joint_names = [f"Joint {i+1}" for i in range(6)] + ["Gripper"]
    
    # Create bar chart
    bars = ax.bar(range(len(action)), action, color=['blue']*6 + ['red'])
    
    # Customize plot
    ax.set_xlabel('Joint/Gripper')
    ax.set_ylabel('Action Value')
    ax.set_title(title)
    ax.set_xticks(range(len(action)))
    ax.set_xticklabels(joint_names, rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, action)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def plot_action_sequence(actions, frame_indices=None, title="Action Sequence Visualization"):
    """
    Plot action sequence over time.
    
    Args:
        actions (list): List of action arrays
        frame_indices (list): List of frame indices
        title (str): Title for the plot
    """
    if not actions:
        print("No actions to plot")
        return
    
    # Convert to numpy array for easier manipulation
    actions_array = np.array(actions)
    
    # Create time axis
    if frame_indices is None:
        time_axis = range(len(actions))
    else:
        time_axis = frame_indices
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Joint names
    joint_names = [f"Joint {i+1}" for i in range(6)] + ["Gripper"]
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']
    
    # Plot each joint's action over time
    for i in range(actions_array.shape[1]):
        ax.plot(time_axis, actions_array[:, i], 
                label=joint_names[i], color=colors[i], marker='o', markersize=3)
    
    # Customize plot
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Action Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_action_heatmap(actions, frame_indices=None, title="Action Heatmap"):
    """
    Create a heatmap visualization of actions over time.
    
    Args:
        actions (list): List of action arrays
        frame_indices (list): List of frame indices
        title (str): Title for the plot
    """
    if not actions:
        print("No actions to plot")
        return
    
    # Convert to numpy array
    actions_array = np.array(actions)
    
    # Create time axis
    if frame_indices is None:
        time_axis = range(len(actions))
    else:
        time_axis = frame_indices
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create heatmap
    im = ax.imshow(actions_array.T, aspect='auto', cmap='RdYlBu_r')
    
    # Customize plot
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Joint/Gripper')
    
    # Set y-axis labels
    joint_names = [f"Joint {i+1}" for i in range(6)] + ["Gripper"]
    ax.set_yticks(range(len(joint_names)))
    ax.set_yticklabels(joint_names)
    
    # Set x-axis labels (show every 10th frame to avoid clutter)
    if len(time_axis) > 20:
        step = max(1, len(time_axis) // 20)
        ax.set_xticks(range(0, len(time_axis), step))
        ax.set_xticklabels(time_axis[::step])
    else:
        ax.set_xticks(range(len(time_axis)))
        ax.set_xticklabels(time_axis)
    
    ax.set_title(title)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Action Value')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the visualization tool."""
    parser = argparse.ArgumentParser(description="Visualize video inference results")
    parser.add_argument("results_file", nargs="?", default="src/output/video_inference_results.json",
                        help="Path to the JSON file containing inference results")
    parser.add_argument("--plot-type", choices=["single", "sequence", "heatmap", "all"], 
                        default="all", help="Type of plot to generate")
    parser.add_argument("--frame-index", type=int, default=0,
                        help="Frame index for single plot (default: 0)")
    
    args = parser.parse_args()
    
    # Load results
    results = load_inference_results(args.results_file)
    if not results:
        print("No results to visualize")
        return
    
    # Extract actions
    actions, frame_indices = extract_actions(results)
    if not actions:
        print("No valid actions found in results")
        return
    
    # Generate plots based on plot type
    if args.plot_type == "single" or args.plot_type == "all":
        if 0 <= args.frame_index < len(actions):
            action = actions[args.frame_index]
            title = f"Action Visualization - Frame {frame_indices[args.frame_index] if frame_indices else args.frame_index}"
            plot_single_action(action, title)
        else:
            print(f"Frame index {args.frame_index} out of range [0, {len(actions)-1}]")
    
    if args.plot_type == "sequence" or args.plot_type == "all":
        title = "Action Sequence Over Time"
        plot_action_sequence(actions, frame_indices, title)
    
    if args.plot_type == "heatmap" or args.plot_type == "all":
        title = "Action Heatmap"
        plot_action_heatmap(actions, frame_indices, title)

if __name__ == "__main__":
    main()
