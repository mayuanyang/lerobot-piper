import torch
from lerobot.processor import ProcessorStep, ProcessorStepRegistry
from lerobot.configs.types import PipelineFeatureType, PolicyFeature

def draw_grid_overlay(image_tensor, grid_cell_size=48):
    """
    Draw a grid overlay on the image tensor.
    
    Args:
        image_tensor: Tensor of shape (C, H, W) or (T, C, H, W) or (B, T, C, H, W)
        grid_cell_size: Size of each grid cell in pixels (default 48)
    
    Returns:
        Image tensor with grid overlay
    """
    # Handle different tensor shapes
    original_shape = image_tensor.shape
    if len(original_shape) == 5:  # (B, T, C, H, W)
        batch_size, time_steps, channels, height, width = original_shape
        # Reshape to process all images at once
        image_flat = image_tensor.view(-1, channels, height, width)
    elif len(original_shape) == 4:  # (T, C, H, W)
        time_steps, channels, height, width = original_shape
        image_flat = image_tensor
    elif len(original_shape) == 3:  # (C, H, W)
        channels, height, width = original_shape
        image_flat = image_tensor.unsqueeze(0)  # Add batch dimension
    else:
        return image_tensor  # Unsupported shape
    
    # Draw grid on each image in the batch
    batch_size_flat = image_flat.shape[0]
    for i in range(batch_size_flat):
        # Get image dimensions
        _, h, w = image_flat[i].shape
        
        # Draw vertical lines
        for x in range(0, w, grid_cell_size):
            # Draw a thin line (2 pixels wide) in red color (channel 0)
            start_x = max(0, x - 1)
            end_x = min(w, x + 1)
            image_flat[i, 0, :, start_x:end_x] = 1.0  # Red channel
            image_flat[i, 1, :, start_x:end_x] = 0.0  # Green channel
            image_flat[i, 2, :, start_x:end_x] = 0.0  # Blue channel
        
        # Draw horizontal lines
        for y in range(0, h, grid_cell_size):
            # Draw a thin line (2 pixels wide) in red color (channel 0)
            start_y = max(0, y - 1)
            end_y = min(h, y + 1)
            image_flat[i, 0, start_y:end_y, :] = 1.0  # Red channel
            image_flat[i, 1, start_y:end_y, :] = 0.0  # Green channel
            image_flat[i, 2, start_y:end_y, :] = 0.0  # Blue channel
    
    # Reshape back to original shape
    if len(original_shape) == 5:
        image_tensor = image_flat.view(batch_size, time_steps, channels, height, width)
    elif len(original_shape) == 4:
        image_tensor = image_flat
    elif len(original_shape) == 3:
        image_tensor = image_flat.squeeze(0)
    
    return image_tensor

@ProcessorStepRegistry.register("grid_overlay_processor")
class GridOverlayProcessorStep(ProcessorStep):
    def __init__(self, grid_cell_size=48, camera_names=None):
        self.grid_cell_size = grid_cell_size
        # Default to front and right cameras if not specified
        self.camera_names = camera_names or ["camera1", "camera3"]
    
    def __call__(self, transition):
        # Apply grid overlay only to specified cameras
        observation = transition.get("observation", {})
        if observation:
            camera_keys = [k for k in observation.keys() if k.startswith("images.")]
            for key in camera_keys:
                # Check if any of the specified camera names are in the key
                if any(cam_name in key for cam_name in self.camera_names) and isinstance(observation[key], torch.Tensor):
                    observation[key] = draw_grid_overlay(observation[key], self.grid_cell_size)
        return transition
    
    def transform_features(self, features):
        # This step doesn't change the features, so return them as-is
        return features
