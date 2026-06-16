"""
Config for the WiltechsVLA encoder-decoder flow matching policy.

Backbone: Qwen/Qwen3-VL-4B-Instruct (bf16; non-FP8 to avoid the finegrained-fp8 kernel dependency)
Architecture: Mixture-of-Transformers (MoT) — encoder-decoder with KV cache.

  - **Encoder (frozen VLM)**: all 36 Qwen3-VL text layers run ONCE per
    inference on [vision tokens, language tokens]. K, V tensors from the
    last `num_vlm_layers` (re-purposed below as DiT depth) layers are
    cached and exposed to the DiT.

  - **Decoder (trainable DiT)**: `num_vlm_layers` independent DiT layers,
    each with self-attention (causal) + cross-attention to one matched
    VLM KV pair + SwiGLU FFN, all modulated by adaLN-Zero from the
    flow-matching time t. DiT runs `num_inference_steps` times per
    inference, but the VLM cache is computed only once.

  - **DiT input sequence**: [SINK, state, robot_cnn_tokens, latent_tokens,
    action_tokens]. The VLM never sees state/action tokens, preserving
    its pretrained vision-language capabilities exactly.

Compared to the previous interleaved version:
  - VLM runs 1× instead of `num_inference_steps`× → ~5-10× faster inference.
  - All 36 VLM layers are used; the DiT just reads from the trailing N.
  - VLM activations are not perturbed by an untrained expert.

Checkpoints from the previous (interleaved) WiltechsVLA are NOT compatible.
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("wiltechs_vla")
@dataclass
class WiltechsVLAConfig(PreTrainedConfig):
    """Configuration for the WiltechsVLA interleaved flow matching policy."""

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
    # Qwen3-VL uses dynamic resolution; this is a nominal target.
    vision_input_size: int = 448

    # -------- VLM backbone --------
    num_cameras: int = 3
    # DiT depth = number of trailing VLM layers whose KV cache the DiT
    # cross-attends to. The VLM itself always runs ALL 36 layers — this
    # field controls only how many of its KV pairs are consumed.
    # (Field name kept for backwards-compat with saved configs.)
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
    # Qwen3-VL-4B-Instruct-FP8 text hidden size is 2560.
    d_model: int = 2560
    # DiT decoder width. 0 → match the VLM hidden size (d_model). Set a smaller
    # multiple of the VLM head_dim (e.g. 1280) to shrink the DiT residual stream /
    # self-attn / FFN / adaLN (~quadratic param savings) while cross-attention is
    # bridged back up to the frozen VLM KV geometry. Big GPU-memory lever.
    dit_hidden_size: int = 0
    # Dropout used inside DiT layers (self-attn output, cross-attn output, FFN output).
    dropout: float = 0.1
    # Kept for backwards-compat — has no effect in the encoder-decoder model
    # (VLM never attends to DiT tokens; cross-attention is strictly DiT → VLM).
    vlm_attends_to_expert: bool = True

    # -------- Flow matching sampling --------
    # Xiaomi-Robotics-0 uses 5 steps; reducing from 10 halves inference time
    # with negligible quality loss in well-trained flow matching policies.
    num_inference_steps: int = 5
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
    use_robot_cnn: bool = True
    # Cameras the trainable RobotCNN ingests. EMPTY = use every camera in
    # `cameras_for_vision_state_concat` (legacy behavior: the CNN re-encodes the
    # same scene views as the frozen VLM, so it competes with — instead of
    # complements — the VLM). Set this to the WRIST/gripper view(s) only to
    # specialize the CNN to close-range manipulation detail the frozen VLM is
    # worst at, and leave scene/color/spatial grounding to the VLM where it
    # demonstrably lives (libero wrist key: 'observation.images.image2').
    robot_cnn_cameras: list[str] = field(default_factory=list)

    # -------- Latent "thought" tokens --------
    # Prepended to the expert sequence. 0 disables.
    num_latent_tokens: int = 4
    # Number of Q-Former cross-attention blocks that distill the VLM KV cache
    # into the latent tokens (learned queries → cross-attn to VLM vision+lang).
    num_latent_qformer_layers: int = 2

    # -------- Vision token dropout (regularizer) --------
    # Applied to the robot-CNN tokens only (see _compute_robot_tokens).
    vision_dropout_prob: float = 0.3
    # Training-time dropout on the VLM vision positions of the KV cross-attn
    # memory (masks vision slots in vlm_kv_pad_mask; the VLM forward itself is
    # untouched). Language slots are never dropped, so this directly weakens
    # the visual shortcut and forces the DiT/QFormer to lean on language.
    # 0 disables (default, checkpoint-compatible).
    vision_kv_dropout_prob: float = 0.0

    # -------- Chat-template input format (Qwen ChatML) --------
    # Wrap the VLM input as a proper instruct-style turn:
    #   <|im_start|>user\n
    #   (<|vision_start|> [cam tokens] <|vision_end|>) x num_cameras
    #   {chat_directive }{task}<|im_end|>\n<|im_start|>assistant\n
    # instead of the raw [vision | task] concatenation. In-distribution for
    # the instruct-tuned VLM; the trailing assistant header adds "answer
    # preparation" registers the DiT can cross-attend to. Off by default
    # (exact legacy behavior, checkpoint-compatible).
    use_chat_template: bool = False
    # Optional short directive prepended to the task inside the user turn,
    # e.g. "Identify the objects mentioned in the instruction and where they
    # are, then perform:". Empty disables. Only used with use_chat_template.
    chat_directive: str = ""

    # Rewrite ambiguous LIBERO object/region names into visually-groundable
    # descriptions (e.g. "alphabet soup" -> "blue can of alphabet soup") via
    # the single-source-of-truth map in task_rewrites.py. Applied to every
    # task string the model consumes, so training/RL/eval stay consistent.
    # Off by default (legacy phrasing); enable for the descriptive-grounding
    # experiment and use the SAME setting at eval.
    use_descriptive_objects: bool = False

    # -------- Auxiliary contrastive loss (language forcing) --------
    contrastive_loss_weight: float = 0.1

    # Minimum mean-squared L2 distance between correct-lang and wrong-lang
    # velocity predictions.
    contrastive_margin: float = 0.05

    # Pair each sample with its HARDEST in-batch negative (most word overlap,
    # different instruction) instead of a random one. Random pairs are almost
    # always grossly-different tasks the model already separates, so the hinge
    # is satisfied without ever forcing fine-grained object grounding (e.g.
    # "alphabet soup" vs "tomato sauce" in the same basket template). Hard
    # negatives focus the gradient on the confusable minimal pairs that fail at
    # eval. Off by default (legacy random pairing). Expect the reported
    # contrastive value to JUMP UP when first enabled — it is now measuring the
    # hard cases — then decline as training installs the discrimination.
    contrastive_hard_negatives: bool = False

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
