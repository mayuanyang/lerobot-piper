import torch
import torch.nn as nn
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from .long_task_diffusion_model import LongTaskDiffusionModel
from .long_task_diffusion_config import LongTaskDiffusionConfig

class LongTaskDiffusionPolicy(DiffusionPolicy):
    """
    A diffusion policy for long task execution with a custom loss function.
    Inherits from DiffusionPolicy and uses a custom diffusion model with overridden compute_loss.
    """
    
    def __init__(self, config: LongTaskDiffusionConfig, *args, **kwargs):
        # Initialize the base DiffusionPolicy
        super().__init__(config, *args, **kwargs)
        
        # Override the diffusion model with our custom model
        self.diffusion = LongTaskDiffusionModel(config)
        
        # Store custom loss weight
        self.custom_loss_weight = config.custom_loss_weight
        
    def forward(self, batch):
        """
        Run the batch through the model and compute the loss for training or validation.
        
        This method uses the base DiffusionPolicy forward method which will call the 
        custom compute_loss method from our LongTaskDiffusionModel.
        
        Args:
            batch: Dictionary containing the input batch data
            
        Returns:
            tuple: (total_loss, None) where total_loss is computed by the base policy
        """
        # The base DiffusionPolicy.forward will handle the loss computation
        # by calling self.diffusion.compute_loss which is our custom implementation
        return super().forward(batch)
