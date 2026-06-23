#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Data-parallel GRPO RL across 8 GPUs (torchrun).
#
#   conda activate wilro
#   bash run_rl_8gpu.sh
#
# Each rank rolls out its own init-state slice on its own GPU; rank 0 gathers the
# records, runs the GRPO update, and broadcasts weights. world_size>1 multiplies
# the per-iter batch and the env-subprocess count — read the NOTES below before
# scaling NPROC or ENV_WORKERS.
# ---------------------------------------------------------------------------
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-wilro}"
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT/src"

# Save a full, timestamped console log under the repo (logs/ at the repo root).
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/rl_$(date +%Y%m%d_%H%M%S).log"
echo "[run] logging console output to $LOG_FILE"

# Headless rendering. Default EGL (GPU); override to osmesa (CPU) for the whole
# run with: MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa bash run_rl_8gpu.sh
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
# MuJoCo is CPU-bound per env; with many env subprocesses, cap threads per proc
# so they don't oversubscribe cores and thrash.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
# Unbuffered stdout/stderr: without this the `| tee` pipe block-buffers Python's
# output, so nothing shows on screen / in the log until ~8 KB accumulates.
export PYTHONUNBUFFERED=1

NPROC="${NPROC:-8}"          # = number of GPUs to use
GROUPS_PER_ITER="${GROUPS_PER_ITER:-2}"   # PER RANK -> global batch = NPROC * this
ENV_WORKERS="${ENV_WORKERS:-3}"           # env processes per task PER RANK
RL_ITERS="${RL_ITERS:-300}"               # number of GRPO iterations to run
SAVE_FREQ="${SAVE_FREQ:-20}"              # checkpoint every N iters
OUTPUT_DIR="${OUTPUT_DIR:-outputs/rl/wilro_spatial_8gpu}"
# Spread the EGL init burst across workers (raise for more ranks if you hit the
# "framebuffer not complete" race; 8 ranks may want 6-10).
export RL_EGL_STAGGER="${RL_EGL_STAGGER:-3}"

torchrun --standalone --nproc_per_node="$NPROC" train_wilro_rl.py \
    --policy_path ISdept/Wilro-ed-137k-l16 \
    --env_task libero_spatial \
    --task_ids 0 1 2 3 4 5 6 7 8 9 \
    --control_freq 10 \
    --lr 1e-6 \
    --exploration_std 0.2 \
    --target_kl 0.02 \
    --clip_low 0.2 --clip_high 0.28 \
    --update_epochs 1 \
    --group_size 6 \
    --update_minibatch 6 \
    --groups_per_iter "$GROUPS_PER_ITER" \
    --env_workers "$ENV_WORKERS" \
    --n_action_steps 4 \
    --max_episode_steps 300 \
    --gradient_checkpointing \
    --rl_iterations "$RL_ITERS" \
    --output_dir "$OUTPUT_DIR" \
    --save_freq "$SAVE_FREQ" \
    2>&1 | tee "$LOG_FILE"

# ---------------------------------------------------------------------------
# NOTES — why these numbers (tune via the env vars above, e.g.
#   GROUPS_PER_ITER=3 ENV_WORKERS=2 NPROC=4 bash run_rl_8gpu.sh):
#
# * GLOBAL batch = NPROC * GROUPS_PER_ITER = 8 * 2 = 16 groups/iter (the single-GPU
#   run used 10). DON'T set GROUPS_PER_ITER=10 here — that's 80 groups/iter, and
#   the single-learner update (rank 0 alone, unchanged) would then process 80
#   groups and dominate wall-clock. With this design update_s grows ~linearly with
#   the global batch, so keep GROUPS_PER_ITER small at high NPROC.
#
# * Expected speedup ~4-5x (not 8x): rollout parallelizes across GPUs but the
#   update stays on rank 0, so at 8 ranks update_s ≈ rollout_s. That's the known
#   ceiling of this design; scaling the update too needs the DDP-update follow-up.
#
# * HOST RAM ≈ NPROC * (#tasks) * ENV_WORKERS * ~0.8 GB of import footprint.
#   8 * 10 * 3 * 0.8 ≈ 190 GB. If RAM is tight: lower ENV_WORKERS to 2 (~125 GB)
#   or drop NPROC to 4 (~95 GB, still ~3x). VRAM is unaffected by these (set by
#   group_size). Watch `free -g` and `nvidia-smi` on the first couple of iters.
#
# * Sanity on iter 0: log should show `world_size=8`, `control_freq=10 Hz
#   (matches ...)`, `rollout_drift=0`, and all 8 GPUs busy in `nvidia-smi`.
# ---------------------------------------------------------------------------
