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


# StarVLA's co-training CoT prompt (starvla_cotrain_libero.yaml). Defined once
# here so the dataclass default and the train script's CLI default cannot drift
# apart -- two copies of a prompt string is exactly the kind of divergence that
# produces two runs that look identically configured and are not.
#
# The bounding boxes are never produced: this VLM is frozen and never decodes.
# The prompt earns its place by CONDITIONING the vision K/V -- under text_first
# it precedes the images, so every vision position is computed with it in scope.
STARVLA_COT_TEMPLATE = (
    "Your task is {instruction}. To identify the key objects for your task. "
    "Locate their bounding boxes in [x1,y1,x2,y2] format."
)


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
    # Side length (px) fed to the Qwen image processor. 0 = leave the processor
    # on its own smart-resize defaults (the historical behaviour -- note the
    # old `vision_input_size = 448` was NEVER passed anywhere, so 0 and 448 are
    # the same run).
    #
    # Qwen3-VL uses patch=16 with spatial_merge=2, so one merged vision token
    # covers a 32x32 block of whatever the processor emits:
    #
    #   input 256 -> 8x8 grid  =  64 tok/cam   (32x32 native px per token)
    #   input 512 -> 16x16 grid = 256 tok/cam  (16x16 native px per token)
    #   input 1024 -> 32x32 grid = 1024 tok/cam ( 8x8  native px per token)
    #
    # With 256x256 source frames, 512 is NOT empty upsampling: the detail is
    # already in the source, the 32px-per-token quantisation is what discards
    # it. libero_spatial has to separate the distractor bowl from the ramekin,
    # 0.127 m apart in the sim (measured, --list_bodies task 0). Earlier notes
    # here and in task_rewrites.py converted that to "~17 native px" / "~19
    # native px" -- both were eyeballed, neither was ever measured, and they
    # disagree. To get the real figure, read the row/col that
    # `kv_grounding_probe.py --annotate` prints per object and divide by 32
    # (8x8 grid) or 16 (16x16 grid) to see whether the two land in one token.
    #
    # The claim that 8x8 is too coarse does NOT rest on that number: at 64
    # tokens the qwen_color_probe failed its own consistency control, naming
    # one bowl as both nearest to and farthest from the plate. That is direct
    # evidence, and it is why the between rewrite requires 512.
    #
    # COST: L_vlm is also the K/V length of every expert's cross-attention, so
    # it multiplies through num_experts x expert_num_layers, not just the VLM.
    vision_input_size: int = 0
    # Cameras that get vision_input_size. Empty = all of them. Restricting this
    # to the third-person view is usually the right trade: the spatial relations
    # that need the resolution are not resolvable at the wrist camera's scale
    # anyway, and it roughly halves the added cost.
    vision_hires_cameras: list[str] = field(default_factory=list)

    # -------- VLM backbone --------
    # Empty = the stock hub id. A local directory loads a LoRA-merged encoder
    # produced by lora_finetune_qwen.py --merge_and_save. Stored in the
    # checkpoint, so an eval reloads the same encoder the policy was trained
    # against -- a mismatch here is silent and would move every KV cache.
    vlm_model_id: str = ""
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
    expert_num_layers: int = 9
    d_model: int = 2560
    dit_hidden_size: int = 640
    dropout: float = 0.1

    # -------- Flow matching sampling --------
    num_inference_steps: int = 5
    noise_temporal_correlation: float = 0.0

    action_dim_weights: list = field(default_factory=list)
    pos_decay_lambda: float = 0.1
    future_steps_weight: float = 0.3
    # Horizon index where future_steps_weight starts. 0 = fall back to
    # n_action_steps, which is what the loss used to read directly.
    #
    # That coupling made a knob silently inert: n_action_steps serves two
    # unrelated purposes -- the loss boundary here, and how many actions the
    # policy executes per replan at inference -- and train_wiltechs_moe.py set
    # it to 64 to match `horizon`, so `pos_w[n_exec:]` was an EMPTY slice and
    # future_steps_weight never applied to anything. With pos_decay_lambda at
    # 0.0 that left the whole 64-step horizon exactly uniform.
    #
    # Why it matters: the LIBERO datasets are 10 Hz, so 64 steps is 6.4 s
    # predicted from a single observation, against episodes capped at 150 steps.
    # The far end of that chunk is dominated by aleatoric uncertainty -- no
    # amount of visual grounding predicts where the gripper is three seconds out
    # -- yet at uniform weighting it takes most of the gradient, while the ~4
    # steps an eval actually executes take 6.25% of it.
    loss_exec_steps: int = 0

    # -------- Training presets --------
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_warmup_steps: int = 1500

    # -------- Robot visual encoder --------
    # The CNN is a ResNet-18 truncated after layer3: total stride 16, so the
    # feature map is input_size/16 on a side. `robot_encoder_tokens` then
    # adaptive-avg-pools that map down to sqrt(tokens) per side.
    #
    # The historical defaults threw away almost all of it. At input_size=224 the
    # feature map is 14x14 = 196 cells, pooled to 4x4 = 16 tokens: 92% of the
    # spatial positions are averaged away, and each surviving token covers
    # 56x56 of the 224 input = 64x64 native px of a 256px LIBERO frame.
    #
    # The frozen VLM's merged vision tokens cover 32x32 native px. So the CNN --
    # whose entire stated purpose is "spatial, pixel-level precision" that the
    # ViT's patch size is too coarse to reach -- was running at HALF the frozen
    # backbone's spatial granularity. Removing it still cost 34 points of
    # success (92% -> 58% on libero_spatial task 0, same checkpoint lineage,
    # same n_action_steps=4), so what it contributes is real; it just was never
    # contributing resolution.
    #
    # input_size 256 matches the native LIBERO frame exactly (no resample) and
    # gives a 16x16 feature map:
    #   8x8  =  64 tok -> 32 px/token  (parity with the VLM)
    #   10x10 = 100 tok -> 25.6 px/token
    #   12x12 = 144 tok -> 21.3 px/token
    #   16x16 = 256 tok -> 16 px/token (ceiling: every feature cell kept)
    robot_encoder_tokens: int = 16
    robot_encoder_input_size: int = 224
    use_robot_cnn: bool = True
    robot_cnn_cameras: list[str] = field(default_factory=list)
    # Per-camera token override. Cameras listed in `robot_cnn_fine_cameras` emit
    # `robot_cnn_fine_tokens` instead of `robot_encoder_tokens`. 0 disables.
    #
    # This exists so "give the wrist more resolution" does not have to be
    # bundled with "drop the front camera's CNN pathway". Those are two changes,
    # and the 34-point ablation says the pathway as a whole is load-bearing --
    # running both at once would leave a regression unattributable. The encoder
    # already supports it: RobotVisualEncoder.forward takes an `out_tokens`
    # override and only the pooling grid depends on it (proj/norm are
    # per-token), so one shared backbone serves both grids at no parameter cost.
    robot_cnn_fine_cameras: list[str] = field(default_factory=list)
    robot_cnn_fine_tokens: int = 0

    # -------- Thought tokens (spatial reasoning bottleneck) --------
    # Learned-query Q-Former cross-attends to the DEEPEST captured VLM layer's
    # KV cache (most semantic: fuses vision + language instruction) to produce
    # K "thought" tokens.  These tokens distill spatial reasoning — "where is
    # the target object", "relative position of gripper to target", "goal
    # placement coordinates" — into a compact representation prepended to the
    # expert input sequence so every action token can attend to the thought.
    #
    # Unlike the old num_latent_tokens (which is kept at 0 for backward compat),
    # thought tokens target the DEEPEST layer (max semantic reasoning) rather
    # than a single arbitrary layer, and serve as an explicit reasoning
    # bottleneck that complements (not replaces) per-expert cross-attention.
    num_thought_tokens: int = 8
    thought_qformer_layers: int = 2
    # Which VLM layer to read KV from for thought generation.
    # -1  → deepest captured layer (max semantic fusion of vision+language)
    # >=0 → specific layer index
    thought_vlm_layer_idx: int = -1
    # Weight for auxiliary "thought consistency" loss: encourages thought
    # tokens to be similar across denoising timesteps (they are noise-
    # independent by design, so this is a regularizer). 0 disables.
    thought_consistency_weight: float = 0.0

    # -------- Legacy latent tokens (kept for backward compat, unused) --------
    num_latent_tokens: int = 0
    num_latent_qformer_layers: int = 2

    # -------- Vision token dropout --------
    vision_dropout_prob: float = 0.3
    vision_kv_dropout_prob: float = 0.0

    # -------- Chat-template input format --------
    use_chat_template: bool = False
    chat_directive: str = ""
    use_descriptive_objects: bool = False
    # Prompt template wrapping each raw instruction, with `{instruction}` marking
    # where it goes. Unlike chat_directive (prefix only) this allows the
    # instruction to sit MID-prompt, which is what STARVLA_COT_TEMPLATE above
    # needs -- see it for the text itself.
    #
    # Nothing is generated here -- the VLM is frozen and never decodes, so a
    # request for bounding boxes cannot be answered. What it does is condition
    # the vision K/V: under text_first the whole prompt precedes the images, so
    # every vision position is computed with it in context. Measured with
    # kv_grounding_probe.py on libero_spatial task 0 (50 layouts, paired, same
    # images, only the text differs), the StarVLA prompt moved TARGET R^2 from
    # 0.180 to 0.289 at matched alpha=1e3 and err/motion from 0.905 to 0.835.
    #
    # It does NOT improve SELECTION: the target-vs-distractor err/motion gap went
    # 0.020 -> 0.000, i.e. both bowls got equally easier to localise. Localisation
    # was never the missing piece; do not expect this to fix referring expressions.
    #
    # Empty = use chat_directive (or the bare instruction). A template WITHOUT
    # `{instruction}` is rejected by the train script: it would drop the
    # instruction entirely and train every sample on one constant prompt.
    #
    # Defaults ON: the measurement above is a paired A/B on identical images, and
    # three independent axes agree (R^2 at every alpha, err/motion, and how badly
    # the shuffled-label control overfits at low alpha -- -0.239 here vs -0.433
    # for the bare instruction, i.e. a better-conditioned feature space). Pass
    # `--instruction_template ""` for the bare instruction.
    #
    # Hand-edited variants tested WORSE: one rewrite kept only 43% of the gain at
    # alpha=1e1 and fell BELOW the bare instruction by 1e4. Do not tune this by
    # hand without re-running kv_grounding_probe.py.
    instruction_template: str = STARVLA_COT_TEMPLATE
    # Token budget for the instruction. The CoT template above adds ~30 tokens;
    # combined with use_descriptive_objects (~105) it overflows the old
    # hardcoded 128 and the truncation takes the TAIL -- which is the template's
    # own trailing clause, so the experiment silently becomes a no-op.
    lang_max_len: int = 128

    # -------- Language placement in the VLM sequence --------
    # The VLM is CAUSAL. With the legacy layout ([images..., instruction]) a
    # vision token's K/V never attends to the instruction, so every vision KV
    # the experts cross-attend to is language-BLIND: referring-expression
    # disambiguation ("the black bowl BETWEEN the plate and the ramekin")
    # survives only in the ~50 trailing text positions, which then compete in
    # a softmax against ~590 vision positions.  The model degenerates to using
    # language as a coarse location prior (reaching for the midpoint) instead
    # of as an object selector.
    #
    # text_first=True moves the instruction BEFORE the images, so every patch's
    # K/V at every layer is conditioned on the instruction and the experts
    # cross-attend to a language-grounded feature map.
    #
    # NOTE: this changes the contrastive branch's cost. With text_first the
    # language is baked into the vision KV, so swapping only the language KV
    # slice is self-inconsistent; the frozen VLM's language model is re-run
    # with permuted instructions instead (ViT output is reused, so the extra
    # cost is the 36 LM layers only, under no_grad).
    text_first: bool = True

    # -------- Auxiliary contrastive loss --------
    contrastive_loss_weight: float = 0.1
    contrastive_margin: float = 0.05
    contrastive_hard_negatives: bool = False

    # -------- MoE router --------
    router_temperature: float = 1.0
    router_balance_weight: float = 0.1
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