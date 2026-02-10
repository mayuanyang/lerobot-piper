import torch
from lerobot.processor import ProcessorStep, ProcessorStepRegistry
from lerobot.configs.types import PipelineFeatureType, PolicyFeature

def draw_grid_overlay(image_tensor, grid_cell_size=48, alpha=0.2):
    # Ensure we have at least 3 dimensions (C, H, W)
    # If input is (H, W), make it (1, H, W)
    if image_tensor.ndim == 2:
        image_tensor = image_tensor.unsqueeze(0)
    
    # Get shape of the last 3 dims
    c, h, w = image_tensor.shape[-3:]
    
    # Create a grid mask on the same device as the input
    mask = torch.zeros((h, w), device=image_tensor.device, dtype=image_tensor.dtype)
    
    # Vectorized line drawing
    # Use [::grid_cell_size] to mark every Nth pixel
    mask[::grid_cell_size, :] = 1.0  
    mask[:, ::grid_cell_size] = 1.0  
    
    # Define grid color: Cyan (0, 1, 1) if RGB, or 1.0 if Grayscale
    if c == 3:
        grid_color = torch.tensor([0.0, 1.0, 1.0], device=image_tensor.device, dtype=image_tensor.dtype).view(3, 1, 1)
    else:
        # For depth maps or grayscale, just use a white/max-value line
        grid_color = torch.tensor([1.0], device=image_tensor.device, dtype=image_tensor.dtype).view(1, 1, 1)
    
    # Apply Alpha Blending
    blend_factor = mask * alpha
    
    # We use ellipsis (...) to handle any number of leading (Batch/Time) dimensions
    image_tensor[..., :, :, :] = (grid_color * blend_factor) + (image_tensor[..., :, :, :] * (1.0 - blend_factor))
    
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
            camera_keys = [k for k in observation.keys() if k.startswith("observation.images.")]
            for key in camera_keys:
                # Check if any of the specified camera names are in the key
                if any(cam_name in key for cam_name in self.camera_names) and isinstance(observation[key], torch.Tensor):
                    observation[key] = draw_grid_overlay(observation[key], self.grid_cell_size)
        return transition
    
    def transform_features(self, features):
        # This step doesn't change the features, so return them as-is
        return features
