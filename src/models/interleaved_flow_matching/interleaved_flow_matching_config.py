"""
Config for the SmolVLA-style interleaved flow matching policy.

Key architectural difference vs `transformer_flow_matching`:
  - Encoder-decoder model: VLM runs to completion frozen, action expert
    cross-attends to its outputs. VLM is not influenced by the action context.
  - **This model (interleaved)**: at every VLM layer, action / latent tokens
    join the VLM in a single self-attention pass. VLM tokens *and* expert
    tokens share the same QKV pool — VLM still uses its frozen QKV/FFN, but
    a parallel trainable "expert layer" handles the action-side QKV/FFN.
    This lets VLM tokens attend to expert tokens (and vice versa) every
    layer, so the VLM's activations become task-aware even though its
    weights are frozen.

Cost of this design:
  - Expert dim must match VLM hidden (960) → larger expert (~200M trainable
    vs ~83M in the encoder-decoder variant).
  - Existing `ISdept/fm64-libero` checkpoints will not load — params /
    structure differ.
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("interleaved_flow_matching")
@dataclass
class InterleavedFlowMatchingConfig(PreTrainedConfig):
    """Configuration for the interleaved (SmolVLA-style) flow matching policy."""

    # -------- I/O structure --------
    n_obs_steps: int = 1
    horizon: int = 4
    n_action_steps: int = 4

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # -------- Image processing --------
    vision_input_size: int = 384

    # -------- VLM backbone --------
    num_cameras: int = 3
    # Number of SmolVLM2 text layers used. The expert mirrors this 1:1
    # (one trainable expert layer per VLM layer). 16 is the SmolVLA default.
    num_vlm_layers: int = 16

    # Selective camera list for vision token construction.
    cameras_for_vision_state_concat: list[str] = field(default_factory=lambda: [
        'observation.images.front',
        'observation.images.gripper',
        'observation.images.right',
    ])

    # -------- State / action dims --------
    state_dim: int = 7
    action_dim: int = 7

    # -------- Expert architecture --------
    # `d_model` is forced to match the VLM hidden dim at construction time
    # (joint attention requires both sides to share embedding dim). The field
    # is here only for surfacing it in saved configs.
    d_model: int = 960
    # Dropout used inside expert layers (self-attn output and FFN output).
    dropout: float = 0.1
    # Allow VLM queries to attend to expert keys.
    # True  → true SmolVLA-style interleaving (VLM perception becomes action-aware).
    # False → expert reads VLM but not vice versa (closer to encoder-decoder).
    # Flip to False if you observe target-leakage symptoms (training loss
    # collapsing while benchmark stagnates).
    vlm_attends_to_expert: bool = True

    # -------- Flow matching sampling --------
    num_inference_steps: int = 10
    noise_temporal_correlation: float = 0.0

    # Per-dimension and positional loss weights.
    action_dim_weights: list = field(default_factory=list)
    pos_decay_lambda: float = 0.1
    future_steps_weight: float = 0.3

    # -------- Training presets --------
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_warmup_steps: int = 1500

    # -------- Robot visual encoder (parallel ResNet-18) --------
    robot_encoder_tokens: int = 16
    robot_encoder_input_size: int = 224
    # Enable / disable the parallel ResNet visual encoder entirely.
    # True (default): create RobotVisualEncoder and route its tokens through
    #                 the expert side of joint attention. Layout becomes
    #                 [robot, latent, action] on the expert side (~ +48 tokens).
    # False:          no ResNet, no robot tokens. For ablation experiments to
    #                 measure whether the CNN actually contributes anything
    #                 beyond what SmolVLM2 already provides.
    use_robot_cnn: bool = True

    # -------- Latent "thought" tokens --------
    # Prepended to the expert sequence. 0 disables.
    num_latent_tokens: int = 8

    # -------- Vision token dropout (language-forcing regularizer) --------
    # Per-token Bernoulli dropout on vision tokens during training. Each
    # vision token (= one SigLIP patch from the connector) is independently
    # zeroed with this probability. Approximates random spatial erasing /
    # cutout on the image. The point isn't to make vision unavailable
    # outright — it's to make the *exact* visual pattern unreliable across
    # samples, so language (which is stable per task) becomes the only
    # reliable disambiguating signal and the model is pressured to use it.
    # 0.0 disables; 0.2-0.4 is a reasonable range.
    vision_dropout_prob: float = 0.3

    # -------- LoRA (vision; text stays frozen — single-task) --------
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])
    vision_lora_num_layers: int = 0  # default off (joint attention already heavy)

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
