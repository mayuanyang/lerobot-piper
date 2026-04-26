from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("transformer_flow_matching")
@dataclass
class TransformerFlowMatchingConfig(PreTrainedConfig):
    """Long Task Transformer Configuration for long-horizon tasks with transformer-based architecture."""
    
    # Inputs / output structure.
    n_obs_steps: int = 2  # 2 obs steps gives the model velocity information (position delta)
    horizon: int = 50
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Image processing
    freeze_vision_backbone: bool = True
    vision_input_size: int = 384  # 384/16=24 patches/side → 576 patches → 144 tokens after connector (scale_factor=4)

    # VLM backbone configuration
    num_cameras: int = 3
    # Number of VLM text layers to use (SmolVLM2-500M has 32 total).
    # 16 layers balances representation quality vs speed/memory (same as SmolVLA default).
    num_vlm_layers: int = 16

    # Zero-shot object detection (YOLOWorld inference)
    # Set these to the names of the objects your robot needs to pick and place
    detection_classes: list = field(default_factory=lambda: ["cube", "container"])
    detection_conf: float = 0.1  # Confidence threshold (lower = more recalls, higher = fewer false positives)
    
    # Selective camera processing - list of camera keys to use for vision-state token concatenation
    # Empty list means use all cameras
    cameras_for_vision_state_concat: list[str] = field(default_factory=lambda: [
        'observation.images.front',
        'observation.images.gripper',
        'observation.images.right'
    ])
        
    
    # State processing
    state_dim: int = 7  # Default for 7-DOF arm

    # Action dimensions
    action_dim: int = 7  # Default for 7-DOF arm
    
    # Transformer architecture
    d_model: int = 512       # 512-dim model; each attention head is 64-dim (standard)
    nhead: int = 8
    num_decoder_layers: int = 8   # deeper decoder for richer action generation
    dim_feedforward: int = 2048   # standard 4×d_model feedforward width
    
    # UNet denoiser parameters
    diffusion_step_embed_dim: int = 128
    
        
    # Flow matching sampling parameters
    num_inference_steps: int = 20

    # Position-decay loss weight: loss *= exp(-lambda * position).
    # Higher values concentrate gradient on early steps (those actually executed).
    # 0.0 = uniform weighting (original behaviour). 0.1 is a reasonable starting point.
    pos_decay_lambda: float = 0.1
        
    # Training presets
    optimizer_lr: float = 1e-4        # Conservative for frozen VLM + fresh action expert (SmolVLA uses 1e-4)
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_warmup_steps: int = 1500  # Scaled 3× for batch_size=96

    use_peft: bool = False

    # Persisted across checkpoints so resume training starts at the correct step/epoch.
    training_step: int = 0
    training_epoch: int = 0
    
    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")


        # Check that all input images have the same shape.
        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                    )

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=90000,  # Span full training budget (100k - 1.5k warmup ≈ 98.5k)
            peak_lr=self.optimizer_lr,
            decay_lr=self.optimizer_lr * 0.01  # 3e-6 floor for final convergence
        )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
