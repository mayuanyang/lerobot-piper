#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Disconnect-safe single-process GRPO RL launcher (tmux + EGL render spread).
#
#   bash run_rl_tmux.sh                 # start in a detached tmux session
#   tmux attach -t rl                   # watch it
#   (Ctrl-b then d to detach; the run keeps going if SSH drops)
#
# Renders each env worker on a different GPU (RL_RENDER_GPUS) so the EGL
# offscreen framebuffers don't exhaust one GPU's memory ("0x8cdd"). Compute
# (the policy) stays on GPU 0; render spreads across the GPUs you list.
# Tune any value via env vars, e.g.:
#   RL_RENDER_GPUS="2,3,4,5" ENV_WORKERS=2 RL_ITERS=200 bash run_rl_tmux.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SESSION="${SESSION:-rl}"
ENV_NAME="${ENV_NAME:-wilro}"
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

# --- knobs ----------------------------------------------------------------
RL_RENDER_GPUS="${RL_RENDER_GPUS:-1,2,3,4,5,6,7}"  # GPUs to spread render over
ENV_WORKERS="${ENV_WORKERS:-3}"      # env processes per task (more = more EGL ctx)
RL_ITERS="${RL_ITERS:-100}"
GROUPS_PER_ITER="${GROUPS_PER_ITER:-10}"
GROUP_SIZE="${GROUP_SIZE:-6}"
TARGET_KL="${TARGET_KL:-0.1}"
MAX_EP_STEPS="${MAX_EP_STEPS:-150}"
SAVE_FREQ="${SAVE_FREQ:-10}"
N_ACTION_STEPS="${N_ACTION_STEPS:-4}"
CONTROL_FREQ="${CONTROL_FREQ:-10}"
POLICY_PATH="${POLICY_PATH:-ISdept/Wilro-ed-137k-l16}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/rl/wilro}"
# --------------------------------------------------------------------------

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists. Attach with:  tmux attach -t $SESSION"
    echo "Or kill it first:  tmux kill-session -t $SESSION"
    exit 1
fi

mkdir -p "$REPO_ROOT/logs"
LOG_FILE="$REPO_ROOT/logs/rl_$(date +%Y%m%d_%H%M%S).log"

# Built as one string so tmux runs it in a fresh login shell (conda available).
read -r -d '' CMD <<EOF || true
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}
cd "${REPO_ROOT}/src"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export RL_RENDER_GPUS="${RL_RENDER_GPUS}"
# The policy is already cached locally; the box's network to HF is flaky and the
# revalidation round-trip kept crashing startup. Use the cache (set 0 to fetch).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
echo "[run] logging to ${LOG_FILE}"
echo "[run] render GPUs = ${RL_RENDER_GPUS} | env_workers = ${ENV_WORKERS}"
python train_wilro_rl.py \
    --policy_path ${POLICY_PATH} \
    --env_task libero_spatial \
    --task_ids 0 1 2 3 4 5 6 7 8 9 \
    --control_freq ${CONTROL_FREQ} \
    --lr 1e-6 \
    --exploration_std 0.2 \
    --target_kl ${TARGET_KL} \
    --clip_low 0.2 --clip_high 0.28 \
    --update_epochs 1 \
    --group_size ${GROUP_SIZE} \
    --update_minibatch 6 \
    --groups_per_iter ${GROUPS_PER_ITER} \
    --env_workers ${ENV_WORKERS} \
    --n_action_steps ${N_ACTION_STEPS} \
    --max_episode_steps ${MAX_EP_STEPS} \
    --gradient_checkpointing \
    --rl_iterations ${RL_ITERS} \
    --output_dir ${OUTPUT_DIR} \
    --save_freq ${SAVE_FREQ} \
    2>&1 | tee "${LOG_FILE}"
echo "[run] training process exited (rc=\${PIPESTATUS[0]}). Session stays open; press enter."
exec bash
EOF

tmux new-session -d -s "$SESSION" "bash -lc '$CMD'"

echo "=========================================================================="
echo "Started RL in tmux session '$SESSION'."
echo "  Watch:   tmux attach -t $SESSION       (detach: Ctrl-b then d)"
echo "  Log:     tail -f $LOG_FILE"
echo "  Stop:    tmux kill-session -t $SESSION"
echo "  GPUs:    nvidia-smi   (render 'G' procs should be on $RL_RENDER_GPUS)"
echo "=========================================================================="
