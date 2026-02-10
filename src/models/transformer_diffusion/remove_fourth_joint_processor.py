from typing import Any, Dict
import torch
from lerobot.processor import ProcessorStep, ProcessorStepRegistry
from lerobot.configs.types import PolicyFeature


@ProcessorStepRegistry.register("remove_fourth_joint_processor")
class RemoveFourthJointProcessorStep(ProcessorStep):
    """Processor step to remove the 4th joint (index 3) from observation.state and action tensors."""
    
    def __init__(self, **kwargs):
        super().__init__()
    
    def _remove_fourth_joint_from_stats(self, stats_dict):
        """Remove the 4th joint (index 3) from statistics arrays."""
        if stats_dict is None:
            return stats_dict
            
        # Create a copy of the stats dictionary
        new_stats = {}
        for key, value in stats_dict.items():
            if isinstance(value, (list, tuple)) and len(value) == 7:
                # Remove the 4th element (index 3) from 7-element arrays
                new_stats[key] = list(value[:3]) + list(value[4:])
            elif isinstance(value, torch.Tensor) and value.dim() == 1 and value.shape[0] == 7:
                # Remove the 4th element (index 3) from 7-element tensors
                new_stats[key] = torch.cat([value[:3], value[4:]], dim=0)
            else:
                # Keep other values as-is
                new_stats[key] = value
        return new_stats
    
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
    
    def transform_features(self, features):
        """Update feature shapes to reflect the removal of the 4th joint."""
        # Create a copy of features to avoid modifying the original
        transformed_features = features.copy()
        
        # Update observation.state shape if it exists
        if "observation.state" in transformed_features:
            feature = transformed_features["observation.state"]
            if hasattr(feature, 'shape') and len(feature.shape) > 0:
                # Reduce the last dimension by 1 (from 7 to 6)
                if feature.shape[-1] == 7:
                    new_shape = list(feature.shape)
                    new_shape[-1] = 6
                    # Update the shape attribute
                    feature.shape = tuple(new_shape)
        
        # Update action shape if it exists
        if "action" in transformed_features:
            feature = transformed_features["action"]
            if hasattr(feature, 'shape') and len(feature.shape) > 0:
                # Reduce the last dimension by 1 (from 7 to 6)
                if feature.shape[-1] == 7:
                    new_shape = list(feature.shape)
                    new_shape[-1] = 6
                    # Update the shape attribute
                    feature.shape = tuple(new_shape)
        
        return transformed_features
