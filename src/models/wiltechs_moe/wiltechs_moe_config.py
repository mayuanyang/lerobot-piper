"""
Config for the WiltechsMoE — Mixture-of-Experts encoder-decoder flow matching policy.

Architecture: Frozen Qwen3-VL-4B (all 36 layers) + N independent expert decoders,
each cross-attending to a DIFFERENT contiguous block of VLM KV caches, with a learned
router that dynamically mixes expert outputs based on state / task / timestep.

All VLM parameters remain frozen; only the experts, router, embeddings, and action
head are trainable.
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("wiltechs_moe")
@dataclass
class WiltechsMoEConfig(PreTrainedConfig):
    """Configuration for the WiltechsMoE mixture-of-experts flow matching policy."""

    # -------- I/O structure --------
    n_obs_steps: int = 2
    horizon: int = 64
    n_action_steps: int = 32

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # -------- Image processing --------
    vision_input_size: int = 448

    # -------- VLM backbone --------
    num_cameras: int = 3
    num_vlm_layers: int = 36
    vlm_capture_layers: list[int] = field(default_factory=list)

    cameras_for_vision_state_concat: list[str] = field(default_factory=lambda: [
        'observation.images.front',
        'observation.images.gripper',
        'observation.images.right',
    ])

    # -------- State / action dims --------
    state_dim: int = 7
    action_dim: int = 7

    # -------- Expert architecture --------
    num_experts: int = 4
    expert_num_layers: int = 4
    d_model: int = 2560
    dit_hidden_size: int = 1280
    dropout: float = 0.1

    # -------- Flow matching sampling --------
    num_inference_steps: int = 5
    noise_temporal_correlation: float = 0.0

    action_dim_weights: list = field(default_factory=list)
    pos_decay_lambda: float = 0.1
    future_steps_weight: float = 0.3

    # -------- Training presets --------
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_warmup_steps: int = 1500

    # -------- Robot visual encoder --------
    robot_encoder_tokens: int = 16
    robot_encoder_input_size: int = 224
    use_robot_cnn: bool = True
    robot_cnn_cameras: list[str] = field(default_factory=list)

    # -------- Latent "thought" tokens --------
    num_latent_tokens: int = 4
    num_latent_qformer_layers: int = 2

    # -------- Vision token dropout --------
    vision_dropout_prob: float = 0.3
    vision_kv_dropout_prob: float = 0.0

    # -------- Chat-template input format --------
    use_chat_template: bool = False
    chat_directive: str = ""
    use_descriptive_objects: bool = False

    # -------- Auxiliary contrastive loss --------
    contrastive_loss_weight: float = 0.1
    contrastive_margin: float = 0.05
    contrastive_hard_negatives: bool = False

    # -------- MoE router --------
    router_temperature: float = 1.0
    router_balance_weight: float = 0.01
    router_top_k: int = 0

    # -------- Resume bookkeeping --------
    training_step: int = 0
    training_epoch: int = 0
    current_lr: float = 0.0
    training_steps_total: int = 0

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("Provide at least one image feature or env state.")
        if len(self.image_features) > 0:
            first_key, first_ft = next(iter(self.image_features.items()))
            for key, ft in self.image_features.items():
                if ft.shape != first_ft.shape:
                    raise ValueError(
                        f"`{key}` shape {ft.shape} does not match `{first_key}` {first_ft.shape}"
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
            num_decay_steps=90000,
            peak_lr=self.optimizer_lr,
            decay_lr=self.optimizer_lr * 0.01,
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