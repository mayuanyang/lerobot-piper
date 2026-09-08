#!/usr/bin/env bash
# =============================================================================
# wilro_moe — precision run
#
# Baseline: ISdept/wilro-sft-wilromoe-30k, 87.5% libero_spatial (20 ep/task).
# Seven of ten tasks are >=95. The mean is held down by exactly three:
#   T1 "next to the ramekin"  ~50    T8 "next to the plate"  70    T5 "on the ramekin"  75
# Video review (2026-09-08): NO selection errors. The failures are grasp and
# place precision.
#
# FRESH RUN, not a resume. The 30k is fully annealed (LR ~0), so any resume must
# re-heat the cosine, which is a known confound (the 2026-09-01 row: 30k->40k
# re-heated and bought +2.1, not significant). A fresh run also gives a curve
# that pairs point-for-point against the already-evaluated 17k / 21k / 30k.
#
# THIS IS A BUNDLE OF THREE CHANGES. It cannot attribute. That is deliberate:
# budget on this config is doubly disconfirmed (30k->40k +2.1 n.s.; 21k->30k
# exactly 0.0), so the next run has to change something, and these are the three
# with the most evidence pointing at precision. Ablate after, if it moves.
# =============================================================================
set -euo pipefail

OUT=${OUT:-./outputs/wilro_moe_precision}
DATA=${DATA:-lerobot/libero}
BATCH=${BATCH:-48}

python src/train_wilro_moe.py \
  --output_dir "$OUT" \
  --dataset_id "$DATA" \
  --training_steps 30000 \
  --batch_size "$BATCH" \
  --lr 1e-4 \
  --warmup_steps 1500 \
  --n_obs_steps 1 \
  --gradient_checkpointing \
  \
  `# ---- CHANGE 1: the wrist camera stops being pooled away -----------------` \
  `# ResNet-18 cut at layer3 is /16, so 224px -> a 14x14 = 196 native map.` \
  `# resnet_tokens 100 pools that to 10x10. On image2 (the wrist view, which` \
  `# carries contact geometry) that is exactly the resolution a grasp needs.` \
  `# 196 is a 1:1 read. Costs NO parameters -- same backbone, different pool.` \
  --vision_token_source resnet \
  --resnet_input_size 224 \
  --resnet_tokens 100 \
  --resnet_fine_cameras observation.images.image2 \
  --resnet_fine_tokens 196 \
  \
  `# ---- CHANGE 2: weight the grasp/release moments -------------------------` \
  `# Up-weights frames within +-2 of a gripper open<->close transition. This is` \
  `# what the feature was written for and it has been off (1.0) in every run.` \
  `# Safe: w_pos goes into the DENOMINATOR too, so it reweights rather than` \
  `# rescales and the effective LR is unchanged.` \
  --gripper_phase_weight 3.0 \
  \
  `# ---- CHANGE 3: dim 3 is no longer randomised ----------------------------` \
  `# NOT A FLAG -- do not pass --lock_joint_index. It now defaults to None,` \
  `# which derives the lock from the data (std <= 0.1% of the widest). LIBERO's` \
  `# dim 3 has std 0.0392 and sits 39x above that line, so it trains. It used to` \
  `# default to 3 for piper_arm's dead joint 4, which made action_out_proj row 3` \
  `# permanently zero -> v_t[...,3] == 0 -> the emitted roll was the initial` \
  `# noise draw, a fresh N(0.0005, 0.039) every chunk, during every grasp.` \
  \
  `# ---- everything below MATCHES the 30k run exactly -----------------------` \
  --num_experts 4 \
  --expert_num_layers 8 \
  --dit_hidden_size 960 \
  --router_temperature 1.0 \
  --router_top_k 0 \
  --router_balance_weight 0.1 \
  --lora_rank 16 \
  --vision_lora_num_layers 0 \
  --contrastive_loss_weight 0.1 \
  --contrastive_margin 0.05 \
  --paraphrase_augment \
  --paraphrase_limit 8 \
  --time_sampling uniform \
  --val_every 500 \
  --val_episodes 40

# =============================================================================
# WATCH THE FIRST 200 STEPS FOR THESE FOUR LINES
#
# 1. "Action dim std: [0]0.3355 [1]0.3784 [2]0.4447 [3]0.0392 ..."
#    with NO star on [3], and "All 7 action dims weighted equally".
#    A star on [3] means the old piper_arm default came back.
#
# 2. "fine grid: ['observation.images.image2'] -> 196 tok (16.0 px/token)"
#    "vision tokens in the DiT sequence: 296 (was 200 ...) -> sequence length 362"
#    If you do not see these, change 1 never took.
#
# 3. "Router usage : E0=..  E1=..  E2=..  E3=..   CV^2=.."
#    CV^2 -> 3.0 with one expert at 100% is router collapse. It should sit low.
#
# 4. "expert ambiguity=0.00xxx = N% of flow"
#    N < 1%  -> the four experts are redundant; the parameters would do more as
#              depth (--num_experts 1 --expert_num_layers 32 is the SAME params
#              and the SAME FLOPs, and runs today). Worth knowing either way.
#
# IF IT OOMs -- in this order, and never reorder:
#   1. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  (fragmentation, not usage)
#   2. --resnet_tokens 64      (third-person view to 8x8; its own docstring says
#                               it "only supplies coarse approach context")
#   3. BATCH=36 ... LAST RESORT. A batch change breaks the step-to-samples
#      mapping, and the whole point of this run is a curve that pairs against
#      the existing 17k / 21k / 30k evals.
#
# EVAL (identical config to every row in the tracker -- keeps the pairing):
#   python src/eval_wiltechs_x.py \
#     --checkpoint "$OUT/checkpoint-30000" \
#     --suites libero_spatial --episodes_per_task 20 \
#     --seed 10000 --fixed_init_states --control_freq 10 \
#     --n_action_steps 2 --num_inference_steps 10
#
# Read T1 / T5 / T8 first. Those three are the run. The other seven are already
# at >=95 and can only lose points -- watch them for negative transfer, but a
# +-10 swing on any single task at 20 ep/task is noise (discordance runs 13-30%).
# =============================================================================
