"""
Config for the WILRO (VLM KV-cache → DiT cross-attention) flow matching policy.

Same encoder-decoder MoT pattern as `wiltechs_vla`, but built on the smaller
SmolVLM2-500M backbone instead of Qwen3-VL-4B (≈8× fewer VLM parameters).

  - Encoder: frozen SmolVLM2 runs ONCE per observation. K/V from the trailing
    `num_vlm_layers` text layers are cached and exposed to the DiT.
  - Decoder: `num_vlm_layers` trainable DiT layers. Each layer = causal self-attn
    over [SINK, state, prefix?, robot, latent, action] + cross-attn to one matched
    VLM KV pair + SwiGLU FFN, all modulated by adaLN-Zero from the flow-matching t.

The DiT shares the VLM's attention shape (hidden_size / num_heads / num_kv_heads
/ head_dim / intermediate_size) so cross-attention GQA aligns automatically.
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("wilro")
@dataclass
class WilroConfig(PreTrainedConfig):
    """Configuration for the WILRO (KV-cache → DiT) flow matching policy."""

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
    # DiT depth = number of trailing VLM layers whose KV cache the DiT
    # cross-attends to. The VLM itself always runs ALL of its layers — this
    # field controls only how many of its KV pairs are consumed.
    # (Field name kept for backwards-compat with saved configs.)
    num_vlm_layers: int = 16

    # Which VLM layers' KV the DiT sources from:
    #   "last"    — the trailing `num_vlm_layers` layers (VLM[V-D..V-1]).
    #               All highly next-token-specialised; uses the most refined
    #               semantics but no multi-scale signal.
    #   "stride2" — evenly spaced every other layer, end-anchored so the final
    #               (most refined) layer is always included: VLM[1,3,..,V-1].
    #               Gives the DiT multi-scale features — shallow DiT layers read
    #               shallow VLM layers (local/token-level), deep DiT layers read
    #               deep VLM layers (abstract/task-level).
    #   "custom"  — use exactly the layer indices in `kv_capture_layers`. The DiT
    #               depth becomes len(kv_capture_layers) (overrides num_vlm_layers
    #               as the depth source). Indices are sorted ascending; DiT layer
    #               j reads the j-th smallest index.
    # NOTE: switching this is NOT resume-compatible — each DiT layer's cross-attn
    # is trained against a specific VLM layer's statistics.
    kv_capture_strategy: str = "last"

    # Explicit VLM layer indices for kv_capture_strategy="custom" (0-based, each
    # in [0, total_VLM_layers)). Ignored for "last"/"stride2". Example for a
    # 32-layer VLM: [3, 7, 11, 15, 19, 23, 27, 31].
    kv_capture_layers: list = field(default_factory=list)

    # Selective camera list for vision token construction.
    cameras_for_vision_state_concat: list[str] = field(default_factory=lambda: [
        'observation.images.front',
        'observation.images.gripper',
        'observation.images.right',
    ])

    # -------- State / action dims --------
    state_dim: int = 7
    action_dim: int = 7

    # -------- DiT architecture --------
    # `d_model` is forced to match the VLM hidden dim at construction time
    # (cross-attention requires both sides to share embedding dim). The DiT
    # also inherits num_heads / num_kv_heads / head_dim / intermediate_size
    # from the VLM's text config, so GQA alignment is automatic.
    # SmolVLM2-500M hidden size is 960.
    d_model: int = 960

    # Dropout used inside DiT layers (self-attn, cross-attn, FFN output).
    dropout: float = 0.1

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
    robot_encoder_tokens: int = 49
    robot_encoder_input_size: int = 224
    use_robot_cnn: bool = True
    # Give one camera a denser token grid than the rest. The gripper / wrist
    # view drives close-range placement precision, so a finer grid there buys
    # spatial detail where it matters. Shares the same ResNet backbone (no extra
    # params — only the pooling grid differs). Must be a perfect square. Set
    # equal to robot_encoder_tokens to disable the per-camera difference.
    gripper_camera: str = "observation.images.gripper"
    gripper_encoder_tokens: int = 100

    # -------- Latent "thought" tokens --------
    num_latent_tokens: int = 8

    # -------- Vision token dropout (regularizer) --------
    vision_dropout_prob: float = 0.15

    # -------- Auxiliary contrastive loss (language forcing) --------
    contrastive_loss_weight: float = 0.1
    contrastive_margin: float = 0.05
    contrastive_hard_negatives: bool = False

    # -------- Action prefix for async execution (paper Sec 2.2.2) --------
    # Max number of clean action prefix steps to condition on. During training,
    # Δt_c is sampled from {0, 1, ..., max_action_prefix_steps}. When > 0,
    # earlier actions are prepended to the noisy action sequence in DiT.
    # 0 disables (synchronous execution mode).
    max_action_prefix_steps: int = 0

    # Λ-shape attention mask: noisy action tokens of later timesteps cannot
    # attend to the conditioned action prefix, forcing them to rely on visual
    # and language signals. (paper Fig 4)
    lambda_mask_window: int = 3

    # -------- LoRA (vision; text stays frozen — single-task) --------
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])
    vision_lora_num_layers: int = 0

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