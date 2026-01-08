#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig


@PreTrainedConfig.register_subclass("long_task_transformer")
@dataclass
class LongTaskTransformerConfig(PreTrainedConfig):
    """Long Task Transformer Configuration for long-horizon tasks with transformer-based architecture."""
    
    # Input dimensions
    n_obs_steps: int = 24
    horizon: int = 24
    n_action_steps: int = 12
    
    # Image processing
    image_features: dict = field(default_factory=dict)
    crop_shape: tuple = (320, 320)
    crop_is_random: bool = True
    use_separate_rgb_encoder_per_camera: bool = True
    
    # State processing
    state_dim: int = 7  # Default for 7-DOF arm
    
    # Action dimensions
    action_dim: int = 7  # Default for 7-DOF arm
    
    # Transformer architecture
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    activation: str = "relu"
    
    # ResNet encoder
    resnet_model: str = "resnet18"
    pretrained_resnet: bool = True
    
    # Tokenization
    state_token_dim: int = 64
    
    def validate_features(self) -> None:
        if len(self.image_features) == 0 and not hasattr(self, 'state_dim'):
            raise ValueError("You must provide at least one image or the state dimension.")

        if self.crop_shape is not None and len(self.image_features) > 0:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the images shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for "
                        f"`{key}`."
                    )
