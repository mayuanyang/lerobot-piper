from typing import Any, Dict
import torch
from lerobot.processor import BaseProcessorStep


class RemoveFourthJointProcessorStep(BaseProcessorStep):
    """Processor step to remove the 4th joint (index 3) from observation.state and action tensors."""
    
    def __init__(self):
        super().__init__()
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove the 4th joint (index 3) from observation.state and action tensors."""
        # Process observation.state if it exists
        if "observation.state" in data:
            state = data["observation.state"]
            if isinstance(state, torch.Tensor) and state.dim() >= 1:
                # Remove the 4th joint (index 3) from the last dimension
                if state.shape[-1] == 7:
                    # Concatenate the first 3 joints and the last 3 joints, removing index 3
                    data["observation.state"] = torch.cat([
                        state[..., :3],  # First 3 joints
                        state[..., 4:]   # Last 3 joints (skipping index 3)
                    ], dim=-1)
                elif state.shape[-1] == 6:
                    # Already processed, do nothing
                    pass
                else:
                    # Unexpected dimensionality, log warning
                    print(f"Warning: observation.state has unexpected shape {state.shape}")
        
        # Process action if it exists
        if "action" in data:
            action = data["action"]
            if isinstance(action, torch.Tensor) and action.dim() >= 1:
                # Remove the 4th joint (index 3) from the last dimension
                if action.shape[-1] == 7:
                    # Concatenate the first 3 joints and the last 3 joints, removing index 3
                    data["action"] = torch.cat([
                        action[..., :3],  # First 3 joints
                        action[..., 4:]   # Last 3 joints (skipping index 3)
                    ], dim=-1)
                elif action.shape[-1] == 6:
                    # Already processed, do nothing
                    pass
                else:
                    # Unexpected dimensionality, log warning
                    print(f"Warning: action has unexpected shape {action.shape}")
        
        return data
