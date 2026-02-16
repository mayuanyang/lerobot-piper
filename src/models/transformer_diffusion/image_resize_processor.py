from typing import Any, Dict
import torch
import torchvision.transforms.functional as F
from lerobot.processor import ProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register("image_resize_processor")
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
            if key in observation:
                image_tensor = observation[key]
                if isinstance(image_tensor, torch.Tensor) and image_tensor.dim() >= 3:
                    # For tensors with 3+ dimensions, the last 2 are spatial dimensions
                    H, W = image_tensor.shape[-2:]
                    
                    # Check if resizing is needed
                    target_H, target_W = self.image_size
                    if H != target_H or W != target_W:
                        # For 3D tensors (C, H, W), we need to add a batch dimension
                        # For 4D+ tensors (..., C, H, W), F.resize should work directly
                        if image_tensor.dim() == 3:
                            # 3D: (C, H, W) - Add batch dimension for resize operation
                            # Add batch dimension: (C, H, W) -> (1, C, H, W)
                            expanded_tensor = image_tensor.unsqueeze(0)
                            # Resize the image
                            resized_image = F.resize(expanded_tensor, (target_H, target_W))
                            # Remove batch dimension: (1, C, H, W) -> (C, H, W)
                            resized_image = resized_image.squeeze(0)
                        else:
                            # 4D+: (..., C, H, W) - Resize directly
                            resized_image = F.resize(image_tensor, (target_H, target_W))
                        observation[key] = resized_image
                        
        return transition
        
    def transform_features(self, features):
        """This step doesn't change the features, so return them as-is."""
        return features
