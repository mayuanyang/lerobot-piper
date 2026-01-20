import torch
from lerobot.policies.pretrained import PreTrainedPolicy
from .transformer_diffusion_model import DiffusionTransformer
from .transformer_diffusion_config import TransformerDiffusionConfig


class TransformerDiffusionPolicy(PreTrainedPolicy):
    """
    Refactored policy for the Piper 7-DOF robot.
    Integrates Spatial Softmax vision and token-based state fusion.
    """
    
    config_class = TransformerDiffusionConfig
    name = "transformer_diffusion"
    
    def __init__(self, config: TransformerDiffusionConfig):
        super().__init__(config)
        self.config = config
        
        # Initialize the refactored sophisticated model
        self.model = DiffusionTransformer(config)
        
    def get_optim_params(self) -> dict:
        """Return the policy parameters for optimization."""
        return self.model.parameters()
    
    def reset(self):
        """Reset the policy. Use if you implement temporal consistency (like RNNs)."""
        pass
        
    def forward(self, batch: dict) -> tuple:
        """
        Training forward pass.
        
        Returns:
            tuple: (loss, None)
        """
        # LeRobot batches often include metadata; we pass the dict directly 
        # to the model which extracts the necessary keys.
        loss = self.model.compute_loss(batch)
        
        return loss, None
    
    @torch.no_grad()
    def predict_action_chunk(self, batch: dict) -> torch.Tensor:
        """
        Predict a sequence (chunk) of actions. 
        Useful for high-frequency control loops.
        """
        self.model.eval()
        predicted_actions = self.model(batch)
        return predicted_actions
    
    @torch.no_grad()
    def select_action(self, batch: dict) -> torch.Tensor:
        """
        Selection logic for real-time inference on the Piper arm.
        """
        self.model.eval()
        
        # Generate the full horizon of actions (e.g., 16 steps)
        predicted_actions = self.model(batch)
        
        # We take 'n_action_steps' (usually 1, 4, or 8) 
        # This reduces jitter by following the predicted trajectory
        # instead of recalculating every single millisecond.
        action_to_execute = predicted_actions[:, :self.config.n_action_steps, :]
        
        return action_to_execute
