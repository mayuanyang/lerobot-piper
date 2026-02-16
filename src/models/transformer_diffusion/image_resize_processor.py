from typing import Any, Dict
import torch
import torchvision.transforms.functional as F
from lerobot.processor import ProcessorStep


class ImageResizeProcessorStep(ProcessorStep[Dict[str, Any], Dict[str, Any]]):
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
        
    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Resize images in the batch to match the expected input size."""
        for key in self.camera_keys:
            if key in batch:
                image_tensor = batch[key]
                if isinstance(image_tensor, torch.Tensor) and image_tensor.dim() >= 4:
                    # Image tensor should be (B, T, C, H, W) or (B, C, H, W)
                    B, *dims = image_tensor.shape
                    H, W = image_tensor.shape[-2:]
                    
                    # Check if resizing is needed
                    target_H, target_W = self.image_size
                    if H != target_H or W != target_W:
                        # Resize the image
                        resized_image = F.resize(image_tensor, (target_H, target_W))
                        batch[key] = resized_image
                        
        return batch
