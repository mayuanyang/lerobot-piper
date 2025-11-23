import torch
import torch.nn as nn
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

class SmoothDiffusion(DiffusionPolicy):
    """
    A diffusion policy that adds a velocity loss to encourage smooth action sequences.
    Inherits from DiffusionPolicy and overrides the forward method.
    """
    
    def __init__(self, *args, velocity_loss_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.velocity_loss_weight = velocity_loss_weight
        
    def forward(self, batch):
        """
        Run the batch through the model and compute the loss for training or validation.
        
        This method extends the base DiffusionPolicy forward method by adding a velocity loss
        that encourages smooth action sequences.
        
        Args:
            batch: Dictionary containing the input batch data
            
        Returns:
            tuple: (total_loss, None) where total_loss is the sum of the diffusion loss and 
                   the weighted velocity loss
        """
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        
        # Compute the main diffusion loss
        loss = self.diffusion.compute_loss(batch)
        
        """
        The Regularization GoalIn Imitation Learning (IL), the goal is for the policy to match the distribution of the expert's demonstrations.
        When you add the velocity loss to the ground-truth actions 
        You are effectively telling the model:"Learn the noise distribution required to reconstruct this ground-truth action sequence via L(diff). 
        And, by the way, the ground-truth action sequence itself is smooth (via $L_{\text{vel}}(\mathbf{a}_{\text{GT}})$).
        This regularizer works by:Penalizing Jagged Data: If a demonstration happens to be slightly jagged, $L_{\text{vel}}(\mathbf{a}_{\text{GT}})$ will be high, 
        increasing the total loss for that sample.
        Biasing the Policy: The model learns that high-loss samples (jagged ground-truth) are "bad" and tries to predict noise that 
        leads to a distribution closer to low-loss samples (smooth ground-truth).The end result is that the policy's learned action distribution 
        is biased toward the smoother parts of the expert data, forcing the final generated actions during inference to be smoother than they would be without the regularizer.
        """
        ground_truth_actions = batch[ACTION]
                
        # Calculate velocity loss to encourage smooth action sequences
        velocity_loss = self._calculate_velocity_loss(ground_truth_actions)
        
        # Combine losses with weighting
        total_loss = loss + self.velocity_loss_weight * velocity_loss
        
        # no output_dict so returning None
        return total_loss, None
      
    
    def _calculate_velocity_loss(self, actions):
        """
        Calculate the velocity loss that penalizes large changes in actions between adjacent time steps.
        
        Formula: L_vel = sum_{t=1}^{H-1} || a_{t+1} - a_t ||^2
        
        Args:
            actions: Predicted action sequence with shape (batch_size, sequence_length, action_dim)
            
        Returns:
            velocity_loss: Scalar tensor with the velocity loss
        """
        if actions.dim() != 3:
            # If actions don't have the expected shape, return zero loss
            # Make sure the tensor is on the same device as the actions
            return torch.tensor(0.0, device=actions.device, dtype=actions.dtype)
            
        # Calculate the difference between consecutive actions
        # actions shape: (batch_size, sequence_length, action_dim)
        # diff shape: (batch_size, sequence_length-1, action_dim)
        diff = actions[:, 1:, :] - actions[:, :-1, :]
        
        # Calculate squared L2 norm for each difference
        # squared_diff shape: (batch_size, sequence_length-1)
        squared_diff = torch.sum(diff ** 2, dim=-1)
        
        # Sum over time steps and batch dimension
        velocity_loss = torch.sum(squared_diff)
        
        return velocity_loss
