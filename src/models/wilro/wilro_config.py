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

    # Flow-matching TIME sampling. "uniform" (default) spends equal capacity at
    # every noise level; "lognormal" (SD3-style logit-normal) biases toward LOW t
    # — t≈0 is x_t≈actions, where the FINE action detail that sets placement
    # precision is denoised. A negative mean shifts mass toward 0.
    time_sampling: str = "uniform"          # "uniform" | "lognormal"
    time_lognormal_mean: float = -0.5       # <0 => bias toward low t (fine detail)
    time_lognormal_std: float = 1.0

    # Per-dimension and positional loss weights.
    action_dim_weights: list = field(default_factory=list)
    pos_decay_lambda: float = 0.1
    future_steps_weight: float = 0.3

    # Phase weighting: up-weight the flow-matching loss on the precision-critical
    # frames around a gripper open<->close transition (grasp / release). Uniform
    # MSE dilutes those few frames among many easy transport frames; concentrating
    # capacity there sharpens the placement the policy keeps fumbling. Weight 1.0
    # = OFF (no behavior change). Assumes the gripper is one action channel
    # (LIBERO OSC: last dim). Folded into the loss denominator so it reweights
    # rather than rescales — LR is unaffected.
    gripper_phase_weight: float = 1.0       # >1 up-weights; 1.0 disables
    gripper_action_index: int = -1          # gripper channel in the action vector
    gripper_transition_window: int = 2      # frames each side of a transition
    gripper_transition_thresh: float = 0.5  # min |Δgripper| to count as a transition

    # -------- Training presets --------
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_warmup_steps: int = 1500

    # -------- Robot visual cross-attention (spatial grounding) --------
    # Robot CA uses intermediate hidden states from the VLM's own SigLIP ViT
    # encoder (with LoRA adapters for robot-domain adaptation). No separate
    # ResNet model — features are extracted during the VLM forward pass, with
    # natural language-vision alignment from SigLIP's contrastive pretraining.
    #
    # When True, each DiT layer has an ADDITIONAL cross-attention sublayer
    # where action queries attend directly to high-resolution SigLIP features
    # (~729 tokens @ 384x384). This provides fine-grained spatial grounding
    # for precise object localization in spatial reasoning tasks.
    #
    # Architecture:
    #   - DiT layer: self-attn + VLM cross-attn + Robot cross-attn + FFN
    #   - adaLN-Zero: 12 modulation vectors (4 sublayers × 3) vs 9 (3 × 3)
    #   - Additional params: robot_ca_q/k/v/o_proj per DiT layer
    #   - Robot features from SigLIP ViT intermediate layer (layer_offset)
    use_robot_ca: bool = True
    # Which intermediate layer of SigLIP ViT to use for Robot CA features.
    # -1 = last layer (most semantic), -3 = third-to-last (more spatial detail).
    # SigLIP ViT has ~27 layers in SmolVLM2-500M. -3 gives the best trade-off
    # between spatial resolution and semantic richness.
    robot_vlm_layer_offset: int = -3

    # Where Robot CA's K/V actually come from.
    #
    #   "vlm_intermediate" -- SigLIP ViT layer `robot_vlm_layer_offset`. Base
    #       frozen; only the LoRA adapters and the connector train (~0.39M in
    #       the robot-visual path). This is what ships and what every 2026-08/09
    #       eval measured.
    #   "resnet"           -- a separate, fully trainable ResNet-18 truncated
    #       after layer3. 3.03M at out_dim 960, MEASURED -- not the 11.7M of a
    #       stock ResNet-18, because layer4 is 72% of that and is excluded. The
    #       "~11M" in the notes and in interleaved's ARCHITECTURE.md is the
    #       stock figure and overstates this by ~4x, which matters: a rank-64
    #       vision LoRA over 27 layers is 5.31M, i.e. ALREADY LARGER than this.
    #       So if this source helps, raw trainable count is not the reason --
    #       what differs is the KIND of pathway: full-resolution pixels (256px
    #       native, stride 16 -> a 16x16 map), ImageNet init, and no frozen
    #       semantic tower underneath. This was the source until 2026-07-06 (2446dbe
    #       swapped it, 18fa4de deleted the encoder) and is still what
    #       wiltechs_moe uses; removing it there cost 34 points of spatial
    #       success (92 -> 58). wilro's own 82.5 on 2026-06-21 predates the swap
    #       and the replacement was never A/B'd.
    #
    # This REPLACES the source, it does not add a second pathway alongside it.
    # A parallel second visual encoder has been measured getting gated off by
    # the optimizer on the sibling (wiltechs_x wrist encoder: 1e-3 -> 6.2e-4,
    # confirmed twice), so "add and let the model choose" is not a neutral
    # design -- it reliably chooses the pathway that is already trained.
    robot_ca_source: str = "vlm_intermediate"

    # ResNet source only. `robot_encoder_tokens` is the pooled grid per camera.
    #
    # The default here is deliberately NOT moe's 16. At input_size 224 a 14x14
    # map pooled to 4x4 makes each token cover 64 native px of a 256px LIBERO
    # frame -- HALF the granularity of the frozen VLM's 32 px merged patches,
    # for a module whose stated purpose is the precision the ViT cannot reach.
    # input_size 256 is the native frame (no resample) giving a 16x16 map:
    #   64 tok  -> 32 px/token  (parity with the VLM)  <- default
    #  144 tok  -> 21.3 px/token
    #  256 tok  -> 16 px/token  (ceiling; every feature cell kept)
    # Cost is per DiT layer -- Robot CA runs in all `num_vlm_layers` of them --
    # and scales with num_cameras, so 3 x 256 is 768 extra K/V per layer.
    robot_encoder_tokens: int = 64
    robot_encoder_input_size: int = 256
    # "avg" = adaptive average pooling (what moe runs). "attn" = AttentionPool2d,
    # learned queries seeded to the position grid. attn cannot honour a per-call
    # token override, which is why avg stays the default.
    robot_encoder_pool: str = "avg"
    # Which cameras get the ResNet. Empty = all of
    # `cameras_for_vision_state_concat`.
    robot_cnn_cameras: list[str] = field(default_factory=list)

    # -------- Temporal input (Stage B / Stage C) --------
    # wilro has no temporal input of any kind: `_encode_images` takes imgs[:, -1]
    # and `_suffix_pass` slices state_tok[:, -1:], so `--n_obs_steps` has never
    # changed anything the model sees. These two flags are what turn that off.
    #
    # B: keep every state frame instead of slicing to the last one. The leak
    #    control is already run (notes/wiltechs_x_ablations.md): the momentum
    #    shortcut sits 33x above the model's own residual, so this channel is
    #    not a shortcut, and a four-condition dose-response (frozen < noise <
    #    shuffled < real, Cochran-Armitage z=4.77, p=1.8e-06) says it carries
    #    real information. The counter-risk is the same file's task 5, where
    #    three independent corruptions of the window each cut time-to-success
    #    195 -> ~110 steps: the window can sustain a dithering loop. wilro's
    #    policy_chunks pins the 700 cap on exactly T1/T4/T5/T8/T9, which is that
    #    signature -- so read this per task, not in aggregate.
    use_state_history: bool = False

    # C: a second, older camera frame through the SAME ResNet backbone, with the
    #    FEATURE MAPS differenced and pooled to this many extra tokens. 0 off.
    #    The VLM still sees one frame -- it is 40.8% of step time and semantics
    #    do not change in 100ms; what changes is motion, which is the ResNet's
    #    job. Requires robot_ca_source="resnet".
    robot_cnn_motion_tokens: int = 0
    # How many frames back the second frame is drawn from. At 10Hz demos and
    # n_action_steps=2 the policy re-plans every 200ms, so 1 frame = 100ms is
    # the natural pairing.
    robot_cnn_motion_stride: int = 1

    # -------- Latent "thought" tokens --------
    # Task-conditional latent tokens generated from pooled language.
    # 0 disables (no latent tokens in DiT sequence).
    num_latent_tokens: int = 0

    # -------- Vision token dropout (regularizer) --------
    vision_dropout_prob: float = 0.15

    # -------- Auxiliary contrastive loss (language forcing) --------
    contrastive_loss_weight: float = 0.1
    contrastive_margin: float = 0.05
    contrastive_hard_negatives: bool = False

    # -------- Instruction surface-form augmentation --------
    # Draw a different phrasing of the same instruction per sample per step, so
    # the surface string stops being a usable key. Measured on the sibling
    # (wiltechs-x-114k, libero_spatial T7): 60% on its own instruction, 0% on a
    # PARAPHRASE of that same instruction -- it had memorised the ~40 strings
    # and was retrieving, not reading. The table lives in src/libero_paraphrase.py.
    #
    # The draw happens in `_encode_language` only, so the contrastive hinge
    # still sees the CANONICAL strings from the batch and keeps deciding "same
    # instruction or not" by exact equality. Paraphrasing before that check
    # would make two phrasings of one task read as two tasks, and the hinge
    # would penalise the model for agreeing with itself in other words --
    # fighting precisely what this is for.
    paraphrase_augment: bool = False
    # Cap on variants per instruction (0 = all). The table has 5-7 each.
    paraphrase_limit: int = 8
    # JSON table overriding the built-in one, for instructions it does not
    # cover. `python -m libero_paraphrase --dataset_id ... --out f.json`, then
    # hand-edit. Templates are NOT consulted at training time.
    paraphrase_file: str = ""
    # The trainer preflight refuses to start when any instruction has fewer
    # variants than this. Partial augmentation is worse than none: the unvaried
    # tasks keep surface form as a key, and the run answers nothing.
    paraphrase_min_variants: int = 5

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

    # -------- LoRA (SigLIP ViT vision + text_model) --------
    # LoRA adapters on the last N layers of SigLIP ViT enable the vision
    # encoder to adapt to robot-domain features (gripper aperture, object
    # distance, contact state) while preserving SigLIP's contrastive
    # language-vision alignment.
    # LoRA adapters on text_model layers enable language adaptation for
    # robot-specific instructions and spatial grounding.
    # Base weights stay frozen; only LoRA params are trainable.
    lora_rank: int = 16
    # float, not int: LoRALinear uses it only as the ratio alpha/rank, and
    # the trainer's default of 2 x rank is written out as e.g. 128.0. Declared
    # as int, draccus refused to load any checkpoint carrying such a value --
    #   DecodingError: `lora_alpha`: Couldn't parse '128.0' into an int
    # so a run trained fine and then could not be evaluated. Widening also
    # reads the older checkpoints, whose value is a plain 32.
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])
    vision_lora_num_layers: int = 8  # Last 8 layers of SigLIP ViT get LoRA
    text_lora_num_layers: int = 0    # DISABLED — encoder-decoder requires detached VLM outputs

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