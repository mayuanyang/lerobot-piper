"""
Config for the WiltechsVLA encoder-decoder flow matching policy.

Backbone: Qwen/Qwen3-VL-4B-Instruct (bf16; non-FP8 to avoid the finegrained-fp8 kernel dependency)
Architecture: Mixture-of-Transformers (MoT) — encoder-decoder with KV cache.

  - **Encoder (frozen VLM)**: all 36 Qwen3-VL text layers run ONCE per
    inference on [language tokens, vision tokens] (see `text_first`). The
    K, V tensors of `num_vlm_layers` of them are cached and exposed to the
    DiT — by default all 36.

  - **Decoder (trainable DiT)**: `num_vlm_layers` independent DiT layers,
    each with self-attention (causal) + cross-attention to one matched
    VLM KV pair + SwiGLU FFN, all modulated by adaLN-Zero from the
    flow-matching time t. DiT runs `num_inference_steps` times per
    inference, but the VLM cache is computed only once.

  - **DiT input sequence**: [SINK, state, robot_cnn_tokens, latent_tokens,
    action_tokens]. The VLM never sees state/action tokens, preserving
    its pretrained vision-language capabilities exactly.

Relationship to WiltechsMoE: same frozen backbone and the same total DiT
layer budget (MoE's 4 experts x 9 layers all run on every forward), but one
sequential stack instead of four parallel ones convex-combined in action
space, and no router. See wiltechs_moe/FINDINGS.md.

Checkpoints from before 2026-08-04 are NOT compatible: the DiT depth default
moved 16 -> 36 and the FFN width formula changed to match MoE.
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
    # Side length (px) fed to the Qwen image processor. 0 = leave the processor
    # on its own smart-resize defaults.
    #
    # This used to read 448 and was NEVER passed anywhere -- _encode_images
    # called preprocess_camera_to_pixels without target_size -- so every run
    # before 2026-08-04 was on the processor default regardless of this value.
    # It is now plumbed through; 0 reproduces the old behaviour exactly.
    #
    # Qwen3-VL uses patch=16 with spatial_merge=2, so one merged vision token
    # covers a 32x32 block of whatever the processor emits:
    #
    #   input 256 -> 8x8 grid  =  64 tok/cam
    #   input 512 -> 16x16 grid = 256 tok/cam
    #
    # With 256x256 source frames, 512 is not empty upsampling: the detail is in
    # the source and the 32px-per-token quantisation is what discards it. See
    # wiltechs_moe/FINDINGS.md -- at 64 tokens the colour probe failed its own
    # consistency control on libero_spatial.
    #
    # COST: L_vlm is also the K/V length of every DiT layer's cross-attention,
    # so it multiplies through num_dit_layers, not just the VLM.
    vision_input_size: int = 0
    # Cameras that get vision_input_size. Empty = all of them. Restricting this
    # to the third-person view is usually the right trade: the spatial relations
    # that need the resolution are not resolvable at the wrist camera's scale
    # anyway, and it roughly halves the added cost.
    vision_hires_cameras: list[str] = field(default_factory=list)

    # -------- VLM backbone --------
    num_cameras: int = 3
    # DiT depth. The VLM always runs ALL 36 layers; this controls how many of
    # its KV pairs the DiT consumes and therefore how deep the DiT is.
    # (Field name kept for backwards-compat with saved configs.)
    #
    # 36 = one DiT layer per VLM layer, i.e. every layer's KV is used. This is
    # the same LAYER COUNT as WiltechsMoE (4 experts x 9 layers, all of which
    # run every forward), but composed sequentially instead of averaged in
    # action space.
    #
    # These are NOT the same function class. An earlier note here claimed a
    # fixed 4-way average is a special case of a 36-layer stack; it is not --
    # f4.f3.f2.f1 != (f1+f2+f3+f4)/4, and embedding four independent branches
    # in one stack needs ~4x the residual width or they interfere. Measured:
    # matched width/batch/data/schedule, the sequential stack scored 33% (4/12)
    # at 12k against the MoE's 92% (46/50) at 18k.
    #
    # Layer count alone does NOT make the two comparable: parameters scale with
    # dit_hidden_size, and the MoE's 92% checkpoint runs 1280, i.e. 1.23B expert
    # params against 401M for 36L at 640. Match the width too.
    #
    # Below 36 the captured layers are spread evenly over the full depth
    # (np.linspace), NOT taken from the tail: the old behaviour read only layers
    # 20..35 and discarded the KV of the first 20, which cost nothing to keep.
    num_vlm_layers: int = 36
    # Explicit VLM layer indices to capture, overriding the even spread. Must
    # have exactly num_vlm_layers entries. Empty = automatic.
    vlm_capture_layers: list[int] = field(default_factory=list)

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

    # -------- Language placement in the VLM sequence --------
    # The VLM is CAUSAL. With the legacy layout ([images..., instruction]) a
    # vision token's K/V never attends to the instruction, so every vision KV
    # the DiT cross-attends to is language-BLIND: referring-expression
    # disambiguation ("the black bowl BETWEEN the plate and the ramekin")
    # survives only in the trailing text positions, which then compete in a
    # softmax against several hundred vision positions. The model degenerates to
    # using language as a coarse location prior (reaching for the midpoint)
    # instead of as an object selector.
    #
    # text_first=True moves the instruction BEFORE the images, so every patch's
    # K/V at every layer is conditioned on the instruction.
    #
    # NOTE: this changes the contrastive branch's cost. With text_first the
    # language is baked into the vision KV, so swapping only the language KV
    # slice is self-inconsistent; the frozen VLM's language model is re-run with
    # permuted instructions instead (ViT output is reused, so the extra cost is
    # the 36 LM layers only, under no_grad).
    text_first: bool = True

    # -------- Latent "thought" tokens --------
    # Prepended to the DiT sequence, produced by a learned-query Q-Former
    # cross-attending to the DEEPEST captured VLM layer. 0 disables.
    num_latent_tokens: int = 8
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
