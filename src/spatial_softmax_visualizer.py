import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt


class SpatialSoftmaxVisualizer:
    """
    Visualization tool for monitoring SpatialSoftmax outputs.
    Logs (x, y) coordinates over time and plots them on images.
    """
    
    def __init__(self, output_dir: str = "spatial_softmax_visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Store trajectories for each camera
        self.trajectories: Dict[str, List[Tuple[float, float]]] = {}
        # Store images for each camera
        self.images: Dict[str, np.ndarray] = {}
        
    def update(self, camera_name: str, image: torch.Tensor, spatial_coords: torch.Tensor):
        """
        Update visualization data for a camera.
        
        Args:
            camera_name: Name of the camera (e.g., "gripper", "depth")
            image: Original image tensor (C, H, W) or (H, W, C)
            spatial_coords: Spatial softmax coordinates (B, C*2) where C is num_channels
                           Format: [x1, y1, x2, y2, ..., xn, yn]
        """
        # Convert image tensor to numpy if needed
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:  # (B, C, H, W)
                image_np = image[0].detach().cpu().numpy()
            elif image.dim() == 3:  # (C, H, W) or (H, W, C)
                image_np = image.detach().cpu().numpy()
            else:  # (H, W)
                image_np = image.detach().cpu().numpy()
            
            # Handle different channel orders
            if image_np.shape[0] in [1, 3]:  # (C, H, W)
                image_np = np.transpose(image_np, (1, 2, 0))
            
            # Convert to uint8 if needed
            if image_np.dtype != np.uint8:
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                else:
                    image_np = image_np.astype(np.uint8)
        else:
            image_np = image
            
        # Store image
        self.images[camera_name] = image_np.copy()
        
        # Process spatial coordinates
        if isinstance(spatial_coords, torch.Tensor):
            coords_np = spatial_coords.detach().cpu().numpy()
        else:
            coords_np = spatial_coords
            
        # Handle batch dimension
        if coords_np.ndim == 2:  # (B, C*2)
            coords_np = coords_np[0]  # Take first batch item
            
        # Reshape to (num_points, 2) format
        num_points = len(coords_np) // 2
        coords_reshaped = coords_np.reshape(num_points, 2)
        
        # Convert from [-1, 1] to pixel coordinates
        height, width = image_np.shape[:2]
        pixel_coords = []
        for x_norm, y_norm in coords_reshaped:
            x_pixel = int((x_norm + 1) * width / 2)
            y_pixel = int((y_norm + 1) * height / 2)
            pixel_coords.append((x_pixel, y_pixel))
            
        # Store trajectory
        if camera_name not in self.trajectories:
            self.trajectories[camera_name] = []
        self.trajectories[camera_name].extend(pixel_coords)
        
    def save_visualizations(self, step: int = 0):
        """
        Save visualizations for all cameras.
        
        Args:
            step: Training/inference step number for naming
        """
        for camera_name, image in self.images.items():
            # Create a copy of the image for drawing
            vis_image = image.copy()
            
            # Draw trajectory
            if camera_name in self.trajectories and len(self.trajectories[camera_name]) > 1:
                points = np.array(self.trajectories[camera_name], dtype=np.int32)
                # Draw the path
                for i in range(1, len(points)):
                    cv2.line(vis_image, 
                            tuple(points[i-1]), 
                            tuple(points[i]), 
                            (0, 255, 0),  # Green line
                            2)
                    
                # Draw points
                for i, (x, y) in enumerate(points):
                    color = (0, 0, 255) if i == len(points) - 1 else (255, 0, 0)  # Red for latest, blue for others
                    cv2.circle(vis_image, (x, y), 5, color, -1)
                    
            # Save image
            camera_dir = self.output_dir / camera_name
            camera_dir.mkdir(exist_ok=True)
            output_path = camera_dir / f"step_{step:06d}.png"
            cv2.imwrite(str(output_path), vis_image)
            
            # Also save a plot of the trajectory
            self._save_trajectory_plot(camera_name, step)
            
    def _save_trajectory_plot(self, camera_name: str, step: int):
        """Save a matplotlib plot of the trajectory."""
        if camera_name not in self.trajectories or len(self.trajectories[camera_name]) == 0:
            return
            
        points = np.array(self.trajectories[camera_name])
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        plt.figure(figsize=(10, 8))
        plt.plot(x_coords, y_coords, 'b-', linewidth=1, alpha=0.7, label='Trajectory')
        plt.scatter(x_coords, y_coords, c=range(len(x_coords)), cmap='viridis', s=30)
        plt.colorbar(label='Time')
        plt.scatter(x_coords[-1], y_coords[-1], c='red', s=100, marker='*', label='Latest')
        
        plt.xlabel('X Pixel Coordinate')
        plt.ylabel('Y Pixel Coordinate')
        plt.title(f'{camera_name.capitalize()} Camera SpatialSoftmax Trajectory - Step {step}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        camera_dir = self.output_dir / camera_name
        plot_path = camera_dir / f"trajectory_step_{step:06d}.png"
        plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
        plt.close()
        
    def reset_trajectories(self):
        """Reset trajectories for all cameras."""
        self.trajectories.clear()


def create_spatial_softmax_visualizer(output_dir: str = "spatial_softmax_visualizations") -> SpatialSoftmaxVisualizer:
    """Factory function to create a visualizer."""
    return SpatialSoftmaxVisualizer(output_dir)
