from typing import Any, Dict
import torch
import torchvision.transforms.functional as F
from lerobot.processor import ProcessorStep


class ImageResizeProcessorStep(ProcessorStep):
    """Processor step to resize images to the expected input size of the vision backbone."""
    
    def __init__(self, image_size: tuple[int, int], camera_keys: list[str]):
        """
        Args:
            image_size: Target image size (height, width)
            camera_keys: List of camera keys to resize (e.g., ["observation.images.front", "observation.images.gripper"])
        """
        super().__init__()
        self.image_size = image_size
        self.camera_keys = camera_keys
        
    def __call__(self, transition: Dict[str, Any]) -> Dict[str, Any]:
        """Resize images in the transition to match the expected input size."""
        # Apply resizing to observation images
        observation = transition.get("observation", {})
        for key in self.camera_keys:
            # Convert policy key format to observation key format
            obs_key = key
            if key in observation:
                image_tensor = observation[key]
                if isinstance(image_tensor, torch.Tensor) and image_tensor.dim() >= 3:
                    # Handle different tensor formats:
                    # - 3D: (C, H, W) - single image
                    # - 4D: (B, C, H, W) or (T, C, H, W) - batch or time series
                    # - 5D: (B, T, C, H, W) - batch with time series
                    original_shape = image_tensor.shape
                    H, W = image_tensor.shape[-2:]
                    
                    # Check if resizing is needed
                    target_H, target_W = self.image_size
                    if H != target_H or W != target_W:
                        # Handle 3D tensor (C, H, W) specially by adding batch dimension
                        if image_tensor.dim() == 3:
                            # Add batch dimension: (C, H, W) -> (1, C, H, W)
                            image_tensor = image_tensor.unsqueeze(0)
                            # Resize the image
                            resized_image = F.resize(image_tensor, (target_H, target_W))
                            # Remove batch dimension: (1, C, H, W) -> (C, H, W)
                            resized_image = resized_image.squeeze(0)
                        else:
                            # For 4D+ tensors, resize directly
                            resized_image = F.resize(image_tensor, (target_H, target_W))
                            
                        observation[key] = resized_image
                        
        return transition
        
    def transform_features(self, features):
        """This step doesn't change the features, so return them as-is."""
        return features
