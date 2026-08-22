"""WiltechsX configuration.

Mixture-of-Transformers VLA designed as an RL post-training SUBSTRATE rather
than as a standalone SFT policy. See ARCHITECTURE.md for the reasoning behind
every default here; the short version is that on standard LIBERO the SFT
architecture spread is <1.5 points while RL post-training is worth 8-12, so
rollout throughput and a non-zero per-task floor matter more than SFT average.

Differences from WiltechsVLA that a reader should not have to dig for:
  - the VLM is TRAINABLE (LoRA), not frozen
  - one joint attention per layer instead of decoder cross-attention to a
    captured KV cache; `vlm_capture_mode` and friends do not exist here
  - the contrastive hinge is replaced by knowledge insulation (stop-grad +
    a discrete FAST token head on the VLM side)
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("wiltechs_x")
@dataclass
class WiltechsXConfig(PreTrainedConfig):
    """Configuration for the WiltechsX joint-attention flow matching policy."""

    # =====================================================================
    # I/O structure
    # =====================================================================
    n_obs_steps: int = 1
    # 16/8 is the OpenVLA-OFT setting, not the 64/32 WiltechsVLA runs. A long
    # horizon costs suffix length in EVERY layer of the joint attention, and
    # the executed prefix is what the reward actually sees.
    horizon: int = 16
    # Executed in full before replanning (OFT). Fewer VLM forwards per episode
    # is the whole point: at 10 Hz and ~280 steps/episode this is 35 prefix
    # computations instead of 280.
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    state_dim: int = 8
    action_dim: int = 7

    # =====================================================================
    # VLM backbone (TRAINABLE via LoRA)
    # =====================================================================
    # Backbone size sets rollout throughput, which sets the stage-B RL budget.
    # Prefer the smallest variant that clears the stage-A gate. VERIFY which
    # Qwen3-VL sizes actually exist on HF before changing this -- 4B is the one
    # this repo has plumbing for.
    vlm_model_id: str = "Qwen/Qwen3-VL-4B-Instruct"

    # Full-freeze is available only as an ablation baseline. It is not the
    # recommended setting and it is not what any top-10 LIBERO method does; it
    # is the configuration that produced this repo's vision collapse.
    freeze_vlm: bool = False

    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    # LoRA on the ViT as well. Off by default: the vision tower is the part
    # most likely to lose general features, and the wrist encoder (below) is
    # the designated trainable visual path.
    lora_on_vision_tower: bool = False

    # Prefix tokens attend BOTH directions; only the action suffix is causal.
    # Unlike the wiltechs_vla flag of the same intent, this one is trained
    # under the mask (LoRA adapts), which is the setting pi0/PaliGemma/X-VLA
    # actually validate. See ARCHITECTURE.md 3.2.
    bidirectional_prefix: bool = True

    num_cameras: int = 2
    cameras_for_vlm: list[str] = field(default_factory=list)
    # 0 = leave the Qwen processor on its smart-resize default.
    vision_input_size: int = 0

    # -------- Instruction text --------
    # The instruction is padded to EXACTLY this length (padding="max_length"),
    # not to the batch maximum. A few dead tokens buy a constant prefix length,
    # so the M-RoPE phase of every later segment stops drifting between
    # batches -- wiltechs_vla records that drift as an unquantified noise
    # source it never removed.
    lang_max_len: int = 48
    # `{instruction}` marks where the raw task string goes. Empty = bare
    # instruction. Nothing is generated from it; the effect is on the prefix
    # representation the expert reads.
    instruction_template: str = ""
    # Rewrite ambiguous LIBERO object names into groundable descriptions via
    # wiltechs_vla/task_rewrites.py. Use the SAME setting at eval.
    use_descriptive_objects: bool = False

    # =====================================================================
    # Knowledge insulation (replaces the contrastive hinge)
    # =====================================================================
    # Stop-grad on the action-expert -> VLM path. Flow-matching gradients
    # flowing into the VLM degrade language grounding; the expert reads the
    # VLM, it does not rewrite it.
    knowledge_insulation: bool = True
    # Discrete FAST action-token head on the VLM side, cross-entropy trained.
    # This is how the VLM still learns the task -- through a token objective it
    # was pretrained for rather than a regression objective it was not.
    #
    # FIRST ABLATION TO RUN. The KI result comes from large cross-embodiment
    # corpora; on LIBERO's 50 demos/task LoRA's rank constraint may already
    # provide the insulation, making this head dead weight.
    fast_token_head: bool = True
    fast_token_loss_weight: float = 0.5
    # NOTE the field names say "fast" for continuity with the design doc, but
    # the implementation is uniform per-dimension BINNING predicted in
    # parallel (RT-2/OpenVLA style), not FAST's DCT+BPE with autoregressive
    # decoding. Knowledge insulation needs *a* discrete token objective on the
    # VLM side; FAST's contribution is sequence-length efficiency for
    # autoregressive decoding, which nothing here does.
    #
    # `fast_tokenizer_id` and `fast_max_tokens` used to live here and were
    # read by nothing. Removed rather than left as dead knobs -- a setting
    # that silently does nothing is worse than no setting.

    # =====================================================================
    # Action expert (Mixture-of-Transformers suffix)
    # =====================================================================
    # One expert block per VLM layer, running in the SAME attention op. There
    # is no cross-attention module and therefore no "which layers does the
    # decoder read" question -- the one `spread` already answered badly.
    expert_hidden_size: int = 1024
    expert_intermediate_size: int = 0     # 0 = match expert_hidden_size
    # Rank of the adaLN-Zero modulation factorisation. A plain Linear(d, 6d)
    # is 32% of the whole expert -- 226M parameters over 36 layers at d=1024,
    # spent producing six vectors per layer, and it OOM'd a 22 GiB card at the
    # optimizer step. 7*d*r instead: 465K per layer at r=64.
    # 0 restores the full-rank form.
    ada_rank: int = 64
    # 0 = one expert block per VLM layer (the pi0 form). A smaller number
    # attaches experts only to the deepest N layers.
    expert_num_layers: int = 0
    num_register_tokens: int = 8

    # =====================================================================
    # Precision path: self-supervised wrist encoder
    # =====================================================================
    # The 34-point RobotCNN result says the gap is high-frequency detail near
    # the gripper. Two changes vs that CNN: self-supervised features (much
    # better dense correspondence than contrastive ones -- OpenVLA fuses
    # SigLIP+DINOv2 for this reason), and placement in the SHARED prefix
    # instead of a privileged side channel. The observed "reliance migrated to
    # the RobotCNN" is a consequence of side-channel placement.
    #
    # NOTE: dinov3 HF ids were NOT verified when this was written. Confirm
    # before switching off the dinov2 default.
    use_wrist_encoder: bool = True
    wrist_encoder_id: str = "facebook/dinov2-small"
    wrist_cameras: list[str] = field(default_factory=list)
    wrist_input_size: int = 256
    # Tokens kept per wrist camera after pooling. Must be a perfect square.
    #
    # Granularity is wrist_input_size / sqrt(wrist_tokens) px per token, and it
    # MUST come out below the Qwen grid's 32 or this path buys nothing -- it is
    # here for high-frequency detail the VLM tokens cannot resolve, not for a
    # second copy of the same information. 256/sqrt(64) = 32.0 is exactly the
    # VLM grid, i.e. the setting this defaulted to on the first run was a
    # no-op on resolution. 256 tokens -> 16 px/token.
    #
    # This is the same trap wiltechs_vla documented for RobotCNN (224/16 = 56
    # px/token, "coarser than the VLM's 32, which defeats the purpose").
    # COST: these tokens sit in the prefix, so they lengthen the K/V every
    # expert layer attends to.
    wrist_tokens: int = 256
    freeze_wrist_encoder: bool = False

    # =====================================================================
    # Long horizon: motion vectors + progress
    # =====================================================================
    # Hindsight as low-dimensional motion vectors rather than stacked frames
    # (HiF-VLA). Frame stacking invites causal confusion, which is exactly the
    # failure LIBERO-Long punishes.
    #
    # RISK: motion vectors can leak the demonstrator's action and reintroduce
    # causal confusion through the back door. Control: a motion-vector-ONLY
    # model must not score above chance.
    use_motion_vectors: bool = True
    motion_history_len: int = 8
    motion_vector_tokens: int = 8

    # Auxiliary regression on normalized time-to-completion. Gives the policy
    # an explicit phase signal and makes stage-B credit assignment tractable --
    # a binary terminal reward on a 10-stage task is the hardest credit
    # assignment problem in the pipeline.
    progress_head: bool = True
    progress_loss_weight: float = 0.1

    # =====================================================================
    # Flow matching / decoding
    # =====================================================================
    # "flow"     standard conditional flow matching, Euler at inference
    # "meanflow" average-velocity field, 1-NFE inference (ElasticFlow)
    # "shortcut" self-consistency across step sizes, 1-4 NFE
    #
    # Few-step decoding is here for the RL ROLLOUT BUDGET, not for a headline
    # Hz number. At 5 Euler steps stage B costs 5x per env step.
    flow_objective: str = "shortcut"
    num_inference_steps: int = 4
    # Fraction of each batch that trains the shortcut self-consistency term
    # instead of the plain d=0 flow objective. 0 disables (= plain flow with a
    # step-size input the model ignores).
    shortcut_consistency_frac: float = 0.25
    # Sampling temperature on the initial noise. Kept as a knob because a flow
    # policy annealed to near-determinism cannot explore and RL dies silently.
    sample_noise_scale: float = 1.0
    # AR(1) correlation across the horizon when drawing noise. 0 = iid.
    noise_temporal_correlation: float = 0.0
    # Draw the initial noise ONCE per episode and reuse it for every replan,
    # instead of drawing fresh noise every n_action_steps.
    #
    # The Euler integration is deterministic given x_1, so the noise is not a
    # perturbation on top of a mean action -- it INDEXES which sample of
    # p(action | obs) comes out. p(action | obs) is genuinely multimodal in
    # manipulation (approach the bowl from the left or from the right; grasp
    # now or close the distance first), so a fresh draw every 0.8 s can hand
    # back a different branch each replan. The arm then commits 0.8 s to one
    # plan, re-decides, and commits to another: the observed stumbling, and
    # the reason the siblings look decisive is partly that their 32-step
    # chunks re-decide 4x less often (wiltechs_vla/moe: horizon 64, n_action_
    # steps 32; here 16/8).
    #
    # With the noise held fixed the map obs -> action stays continuous, so the
    # policy is still fully reactive to the new observation; only the branch
    # selection stops flip-flopping.
    #
    # The trade is real and untested here: a draw that lands on a bad branch
    # is now bad for the whole episode rather than for 0.8 s, so this can
    # LOWER the mean while it raises within-episode consistency. Judge it on
    # the success-rate distribution, not only the mean. Off by default.
    #
    # Not the same as annealing the noise to zero: x_1 = 0 is far outside the
    # distribution the field was fit on (training sees ||x_1|| ~ sqrt(H*A)).
    # Nor the same as averaging several samples -- the mean of two valid
    # branches (left, right) is a path into the object.
    fixed_episode_noise: bool = False

    # =====================================================================
    # Losses
    # =====================================================================
    action_loss_weight: float = 1.0
    # Steps the loss treats as "executed"; the tail beyond it is down-weighted
    # by future_steps_weight. 0 = the full horizon (no down-weighting).
    # Deliberately NOT tied to n_action_steps: that couples the LOSS to an
    # INFERENCE knob, which is the bug wiltechs_vla had to back out.
    loss_exec_steps: int = 0
    future_steps_weight: float = 1.0
    # Ported from wiltechs_vla: without class balancing the gripper dimension
    # sits in the majority-class optimum and never learns to close.
    gripper_bce_weight: float = 0.05
    gripper_action_dim: int = -1
    gripper_class_balance: bool = True
    gripper_bce_temp: float = 0.25
    # NaN means "not calibrated" and DISABLES the gripper term. It is the
    # normalized action value separating open from closed, so it depends on
    # the dataset statistics -- the train script must compute it.
    gripper_threshold_norm: float = float("nan")

    # -------- Contrastive language hinge --------
    # This was "deliberately absent" on the premise that knowledge insulation
    # replaces it (3.3). MEASURED FALSE on 2026-08-17 at checkpoints
    # 7000-10000: the discrete CE does depend on the instruction, but only by
    # 4.2% of the strongest control, and that share is FLAT over 3000 steps
    # while reliance on vision and state both grow. A rollout ablation
    # confirmed it behaviourally -- feeding libero_spatial task 0 the wrong
    # instruction moved the success rate 25% -> 20% (Fisher p = 1.0) where
    # following the instruction would have driven it to ~0.
    #
    # The cause is that nothing in the objective penalises producing the same
    # action under a different instruction. The discrete CE rewards predicting
    # the action, and in one fixed scene vision plus proprioception already
    # explain 96% of it. Only a hinge prices instruction-invariance directly.
    # 0.0 keeps the term off, which is the pre-2026-08-17 behaviour.
    contrastive_loss_weight: float = 0.0
    # How far apart the two velocity predictions must be, in the same units as
    # the flow loss. wiltechs_vla ran 0.05 and saw the hinge saturate around
    # 15k steps, so treat this as a floor to raise rather than a ceiling.
    contrastive_margin: float = 0.05
    # Fraction of the batch that gets the extra suffix pass. The prefix is NOT
    # recomputed (see compute_loss), so this is priced like the shortcut term.
    contrastive_frac: float = 0.5
    # Token-Jaccard above which two instructions count as the same "suite", and
    # so as valid hard negatives for one another. The negative is drawn
    # UNIFORMLY from the bucket, not argmax-similar: argmax collapses to one
    # fixed partner per task (40 ordered pairs over LIBERO) and invites
    # overfitting to that specific contrast, while the bucket keeps 9 and every
    # one of them hard.
    #
    # Why the bucket is the right grain rather than a cheap approximation: the
    # hinge keeps sample i's IMAGE and swaps in j's instruction. Draw j from
    # another suite and its nouns are absent from the scene, so the model
    # separates the two predictions by noticing the referent is not there --
    # object-presence, no relation parsing. Within a suite every referent IS
    # present and only the relation disambiguates, which is the gradient we
    # are paying for. Measured on LIBERO's 40 instructions: within-suite
    # Jaccard 0.67-0.79, across-suite 0.21-0.38, so 0.5 separates them with
    # room on both sides. Set to 0.0 to restore uniform-random negatives.
    contrastive_suite_jaccard: float = 0.5

    # Train on several phrasings of each instruction, drawn per sample per step.
    #
    # Measured on wiltechs-x-114k, libero_spatial task 7, identical harness:
    # its own instruction scored 60%; ANOTHER TASK's instruction verbatim
    # scored 0% with the arm going cleanly to that task's object; and a
    # PARAPHRASE of its own instruction -- "that is on the stove", "put it onto
    # the plate" -- also scored 0%. A model reading the sentence is untouched
    # by that rewording. This one has memorised the ~40 strings in the dataset
    # and maps each to a behaviour: hand it a table entry and it retrieves,
    # hand it anything else and it does not.
    #
    # The language probe does not catch this. It substitutes OTHER TASKS'
    # instructions, all table entries, so its d(lang) measures only whether the
    # 40 known strings are told apart.
    #
    # Sampling per step is the point; a single fixed rewrite (see
    # use_descriptive_objects) is just a second table to memorise. The original
    # string is always among the variants because evaluation uses it.
    paraphrase_augment: bool = False
    paraphrase_limit: int = 8
    # Instructions the templates decline to restructure -- libero_goal's "turn
    # on the stove", libero_10's "put both X and Y in the basket" -- need a
    # hand-written table, or they train unaugmented while everything else
    # varies. Partial augmentation is worse than none: the model keeps surface
    # form as a usable key for exactly the tasks that were left alone, and the
    # run cannot say whether augmentation works. A supplied entry overrides the
    # templates. `python -m models.wiltechs_x.paraphrase --dataset_id ... --out`
    # writes a starting table and names the entries that need editing.
    paraphrase_file: str = ""
    # Trainer preflight: refuse to start if any instruction has fewer variants
    # than this. Failing at startup beats discovering it in the eval.
    paraphrase_min_variants: int = 5

    # =====================================================================
    # Optimizer / schedule
    # =====================================================================
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 60000

    # =====================================================================
    # Resume bookkeeping
    # =====================================================================
    training_step: int = 0
    training_epoch: int = 0
    current_lr: float = 0.0
    training_steps_total: int = 0

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("Provide at least one image feature or env state.")
        # Guarded: with only an env state, image_features is empty and next()
        # would raise StopIteration instead of anything readable.
        if len(self.image_features) > 0:
            first_key, first_ft = next(iter(self.image_features.items()))
            for key, ft in self.image_features.items():
                if ft.shape != first_ft.shape:
                    raise ValueError(
                        f"`{key}` shape {ft.shape} does not match "
                        f"`{first_key}` {first_ft.shape}")

    def __post_init__(self):
        # PreTrainedConfig.__post_init__ does device selection and the AMP
        # fallback -- skipping it leaves self.device unvalidated.
        super().__post_init__()
        if self.flow_objective not in ("flow", "meanflow", "shortcut"):
            raise ValueError(
                f"flow_objective must be flow|meanflow|shortcut, got {self.flow_objective!r}"
            )
        root = int(round(self.wrist_tokens ** 0.5))
        if root * root != self.wrist_tokens:
            raise ValueError(
                f"wrist_tokens must be a perfect square, got {self.wrist_tokens}"
            )
        if self.n_action_steps > self.horizon:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) exceeds horizon ({self.horizon})"
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
            num_decay_steps=self.scheduler_decay_steps,
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
