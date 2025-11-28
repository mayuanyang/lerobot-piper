import torch
import torch.nn as nn
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from .custom_diffusion_config import CustomDiffusionConfig

class JointSmoothDiffusion(DiffusionPolicy):
    """
    A diffusion policy that adds per-joint regularization losses to encourage smooth action sequences.
    Inherits from DiffusionPolicy and overrides the forward method.
    """
    
    def __init__(self, config: CustomDiffusionConfig, *args, velocity_loss_weight=1.0, acceleration_loss_weight=0.5, 
                 jerk_loss_weight=0.1, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.velocity_loss_weight = velocity_loss_weight
        self.acceleration_loss_weight = acceleration_loss_weight
        self.jerk_loss_weight = jerk_loss_weight
        
    def forward(self, batch):
        """
        Run the batch through the model and compute the loss for training or validation.
        
        This method extends the base DiffusionPolicy forward method by adding per-joint
        regularization losses that encourage smooth action sequences.
        
        Args:
            batch: Dictionary containing the input batch data
            
        Returns:
            tuple: (total_loss, None) where total_loss is the sum of the diffusion loss and 
                   the weighted regularization losses
        """
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        
        # Compute the main diffusion loss
        loss = self.diffusion.compute_loss(batch)
        
        ground_truth_actions = batch[ACTION]
                
        # Calculate per-joint regularization losses for smoother action sequences
        if ground_truth_actions.dim() != 3 or ground_truth_actions.shape[1] < 3:
            # Need at least 3 time steps for acceleration calculation
            return torch.tensor(0.0, device=ground_truth_actions.device, dtype=ground_truth_actions.dtype)
        
        # Calculate velocities (first differences) for each joint
        velocities = ground_truth_actions[:, 1:, :] - ground_truth_actions[:, :-1, :]
        
        # Calculate accelerations (second differences) for each joint
        accelerations = velocities[:, 1:, :] - velocities[:, :-1, :]
        
        # Calculate squared L2 norm for each acceleration per joint
        squared_acc = accelerations ** 2
        
        # Sum over time steps, joints, and batch dimension
        acceleration_loss = torch.mean(squared_acc)
        
        joint2_loss = self._calculate_per_joint_jerk_loss(ground_truth_actions, joint_index=1)
        joint5_loss = self._calculate_per_joint_jerk_loss(ground_truth_actions, joint_index=4)
        
        # Combine losses with weighting
        total_loss = loss + \
        self.acceleration_loss_weight * acceleration_loss + \
        self.jerk_loss_weight * joint2_loss + \
        self.jerk_loss_weight * joint5_loss
        
        # no output_dict so returning None
        return total_loss, None
      
    
    def _calculate_per_joint_acceleration_loss(self, actions):
        """
        Calculate the acceleration loss per joint that penalizes large changes in velocity.
        
        Formula: L_acc = sum_{joint=1}^{D} sum_{t=2}^{H-1} || (a_{t+1,joint} - a_{t,joint}) - (a_{t,joint} - a_{t-1,joint}) ||^2
        
        Args:
            actions: Predicted action sequence with shape (batch_size, sequence_length, action_dim)
            
        Returns:
            acceleration_loss: Scalar tensor with the acceleration loss
        """
        if actions.dim() != 3 or actions.shape[1] < 3:
            # Need at least 3 time steps for acceleration calculation
            return torch.tensor(0.0, device=actions.device, dtype=actions.dtype)
        
        # Calculate velocities (first differences) for each joint
        velocities = actions[:, 1:, :] - actions[:, :-1, :]
        
        # Calculate accelerations (second differences) for each joint
        accelerations = velocities[:, 1:, :] - velocities[:, :-1, :]
        
        # Calculate squared L2 norm for each acceleration per joint
        squared_acc = accelerations ** 2
        
        # Sum over time steps, joints, and batch dimension
        acceleration_loss = torch.mean(squared_acc)
        
        return acceleration_loss
    
    def _calculate_per_joint_jerk_loss(self, actions, joint_index):
        """
        Calculate the jerk loss per joint that penalizes large changes in acceleration.
        
        Formula: L_jerk = sum_{joint=1}^{D} sum_{t=3}^{H-1} || jerk_{t,joint} ||^2
        where jerk_{t,joint} = a_{t+1,joint} - 3*a_{t,joint} + 3*a_{t-1,joint} - a_{t-2,joint}
        
        Args:
            actions: Predicted action sequence with shape (batch_size, sequence_length, action_dim)
            
        Returns:
            jerk_loss: Scalar tensor with the jerk loss
        """
        if actions.dim() != 3 or actions.shape[1] < 4:
            # Need at least 4 time steps for jerk calculation
            return torch.tensor(0.0, device=actions.device, dtype=actions.dtype)
        
        # Calculate velocities (first differences) for the specified joint
        velocities = actions[:, 1:, joint_index] - actions[:, :-1, joint_index]
        
        # Calculate accelerations (second differences) for the specified joint
        accelerations = velocities[:, 1:] - velocities[:, :-1]
        
        # Calculate squared L2 norm for each acceleration per joint
        squared_acc = accelerations ** 2
        
        # Sum over time steps, joints, and batch dimension
        acceleration_loss = torch.mean(squared_acc)
        
        return acceleration_loss
