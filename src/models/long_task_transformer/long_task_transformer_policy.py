import torch
import torch.nn as nn
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.policies.pretrained import PreTrainedPolicy

from .long_task_transformer_model import LongTaskTransformerModel
from .long_task_transformer_config import LongTaskTransformerConfig


class LongTaskTransformerPolicy(PreTrainedPolicy):
    """
    A transformer-based policy for long task execution.
    Uses ResNet for image encoding, tokenizes state observations,
    and employs a transformer decoder to generate actions autoregressively.
    """
    
    config_class = LongTaskTransformerConfig
    name = "long_task_transformer"
    
    def __init__(self, config: LongTaskTransformerConfig):
        super().__init__(config)
        self.config = config
        
        # Initialize the transformer model
        self.transformer = LongTaskTransformerModel(config)
        
    def get_optim_params(self) -> dict:
        """Return the policy parameters for optimization."""
        return self.parameters()
    
    def reset(self):
        """Reset the policy. Called when environment is reset."""
        # For this policy, there's no internal state to reset
        pass
        
    def forward(self, batch):
        """
        Run the batch through the model and compute the loss for training or validation.
        
        Args:
            batch: Dictionary containing the input batch data
            
        Returns:
            tuple: (total_loss, None) where total_loss is computed by the transformer model
        """
        # Ensure batch is in the correct format
        if len(self.config.image_features) > 0:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
        
        # Compute the loss using the transformer model
        loss = self.transformer.compute_loss(batch)
        
        # no output_dict so returning None
        return loss, None
    
    @torch.no_grad()
    def predict_action_chunk(self, batch, **kwargs):
        """
        Predict a chunk of actions given environment observations.
        
        Args:
            batch: Dictionary containing the input batch data
            
        Returns:
            actions: Tensor of predicted actions
        """
        # Generate actions using the transformer model
        predicted_actions = self.transformer.forward(batch)
        return predicted_actions
    
    @torch.no_grad()
    def select_action(self, batch, **kwargs):
        """
        Select actions for inference.
        
        Args:
            batch: Dictionary containing the input batch data
            
        Returns:
            actions: Tensor of predicted actions
        """
        # Ensure batch is in the correct format
        if len(self.config.image_features) > 0:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            
        # Generate actions using the transformer model
        predicted_actions = self.transformer.forward(batch)
        
        # Return the first action step for immediate execution
        return predicted_actions[:, :self.config.n_action_steps, :]
