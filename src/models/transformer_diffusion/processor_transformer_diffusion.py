from typing import Any

import torch

from .transformer_diffusion_config import TransformerDiffusionConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)

# Import from local module
from .grid_overlay_processor import GridOverlayProcessorStep
from .remove_fourth_joint_processor import RemoveFourthJointProcessorStep
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


def make_pre_post_processors(
    config: TransformerDiffusionConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    add_grid_overlay: bool = False,
    grid_overlay_cameras: list[str] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    # Create a copy of dataset_stats and modify it to remove the 4th joint
    modified_stats = None
    if dataset_stats is not None:
        modified_stats = {}
        for key, stats_dict in dataset_stats.items():
            if key in ["observation.state", "action"] and stats_dict is not None:
                # Remove the 4th joint (index 3) from stats
                modified_stats[key] = {}
                for stat_key, stat_value in stats_dict.items():
                    # Handle different types of stats values
                    if isinstance(stat_value, (list, tuple)) and len(stat_value) == 7:
                        # Remove the 4th element (index 3) from 7-element arrays
                        modified_stats[key][stat_key] = list(stat_value[:3]) + list(stat_value[4:])
                    elif isinstance(stat_value, torch.Tensor) and stat_value.dim() == 1 and stat_value.shape[0] == 7:
                        # Remove the 4th element (index 3) from 7-element tensors
                        modified_stats[key][stat_key] = torch.cat([stat_value[:3], stat_value[4:]], dim=0)
                    elif hasattr(stat_value, 'shape') and len(stat_value.shape) >= 1 and stat_value.shape[-1] == 7:
                        # Handle multi-dimensional arrays/tensors where the last dimension is 7
                        if len(stat_value.shape) == 1:
                            # 1D case
                            if isinstance(stat_value, torch.Tensor):
                                modified_stats[key][stat_key] = torch.cat([stat_value[:3], stat_value[4:]], dim=0)
                            else:  # numpy array or similar
                                import numpy as np
                                modified_stats[key][stat_key] = np.concatenate([stat_value[:3], stat_value[4:]], axis=0)
                        else:
                            # Multi-dimensional case - remove 4th element from last dimension
                            if isinstance(stat_value, torch.Tensor):
                                modified_stats[key][stat_key] = torch.cat([stat_value[..., :3], stat_value[..., 4:]], dim=-1)
                            else:  # numpy array or similar
                                import numpy as np
                                modified_stats[key][stat_key] = np.concatenate([stat_value[..., :3], stat_value[..., 4:]], axis=-1)
                    elif isinstance(stat_value, (list, tuple)):
                        # Handle list/tuple with different length
                        modified_stats[key][stat_key] = stat_value
                    elif isinstance(stat_value, torch.Tensor):
                        # Handle tensor with different shape
                        modified_stats[key][stat_key] = stat_value
                    else:
                        # Keep other values as-is
                        modified_stats[key][stat_key] = stat_value
            else:
                # Keep other features as-is
                modified_stats[key] = stats_dict
    
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        RemoveFourthJointProcessorStep(),  # Remove 4th joint before normalization
    ]
    
    # Add grid overlay processor step if requested
    if add_grid_overlay:
        input_steps.append(GridOverlayProcessorStep(grid_cell_size=40, camera_names=grid_overlay_cameras))
    
    input_steps.extend([
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=modified_stats,  # Use modified stats
        ),
    ])
    
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=modified_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
