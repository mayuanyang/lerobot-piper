#!/usr/bin/env bash
# Re-search one suite minimum. ONE per invocation, because these are meant to
# run in separate Colab sessions on separate GPUs:
#
#     bash research_min_tasks.sh object     # T4 "pick up the ketchup"   18.7 h
#     bash research_min_tasks.sh spatial    # T8 "bowl next to plate"     8.1 h
#
# In parallel that is 18.7 h of wall clock instead of 26.8 h sequential. The
# two write to different --out directories, which is required rather than tidy:
# ticket_bundle.save_ticket is read-modify-write with no lock, so two searches
# sharing a directory silently erase each other's tickets. Merge at the end.
#
# THE CHECKPOINT IS THE SAME FOR BOTH. All four suites were evaluated on
# snapshot b61cf9a9 of wilro-wilromoe-8x4-22k-obs2; only the bundles are split.
#
# What differs between the two searches is the episode cap:
#
#   object T4 runs at 280, matching eval. Its successes average 115.5 steps and
#   its failures run to eval's 280 cap, so a 150-step search cap scored some of
#   its real successes as failures and ranked candidates by speed rather than by
#   whether they solve the task. No other object task is close: the rest average
#   66-103 steps.
#
#   spatial T8 stays at 150. Its successes average 68.1, so the cap is 2.2x the
#   mean and is not touching them; 280 would cost 87% more wall clock for
#   nothing.
#
# Both take a fresh --seed, because candidates are seeded per (seed, suite,
# task) and the old seed redraws the old tickets, and --certify_layouts 15 so
# the use-it-or-not call is made on layouts that took no part in choosing the
# winner. Keep that separation and the reported number stays honest:
#
#     20-34  search          35-49  decide          0-19  report, once
set -euo pipefail

WHICH=${1:-}
REPO=/content/lerobot-piper
PY=/content/wilro/bin/python
CKPT=ISdept/wilro-wilromoe-8x4-22k-obs2
DRIVE=/content/drive/MyDrive/wilro_moe

case "$WHICH" in
  object)  SUITE=libero_object;  TASK=4; CAP=280; EST="18.7 h" ;;
  spatial) SUITE=libero_spatial; TASK=8; CAP=150; EST="8.1 h"  ;;
  *) echo "usage: $0 object|spatial" >&2; exit 2 ;;
esac

OUT="$DRIVE/${WHICH}_tickets_retry"

cd "$REPO" && git pull
cd "$REPO/src"

echo "### $SUITE task $TASK   cap $CAP   estimate $EST   -> $OUT"
MPLBACKEND=Agg "$PY" search_golden_ticket.py \
  --checkpoint "$CKPT" \
  --suites "$SUITE" --task_ids "$TASK" \
  --tickets 256 --envs_per_tier 5 --tiers 3 \
  --certify_layouts 15 \
  --seed 20000 \
  --max_episode_steps "$CAP" \
  --out "$OUT"

echo
"$PY" ticket_bundle.py report "$OUT"

cat <<'MSG'

READ THE FIRST HOUR, NOT THE LAST. Tier 1's first layout prints

    layout 1 pass rate NN% vs Gaussian MM%; the floor needs K/5,
    so about NNN candidates per survivor
    -> --tickets 256 expects N.NN survivors.  Killing this now ... costs less

Under 0.5 expected survivors, kill it: the task needs an order of magnitude
more candidates and the remaining hours only confirm that. object T5 spent 3.7
hours confirming exactly that.

Killed for any other reason, rerun the same line -- progress resumes per layout,
and the baseline resumes per layout too.
MSG
