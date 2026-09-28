#!/usr/bin/env bash
# Re-search every libero_object task that is not at 20/20.
#
#     T2 95   ticket 15/15, one failure          T3 90   ticket rejected, runs Gaussian
#     T4 80   ticket 12/15, suite minimum        T5 90   ticket rejected, runs Gaussian
#     T9 90   ticket 15/15, two failures
#
# T6 is left alone: it is already 20/20. On Gaussian that is one draw rather
# than a guarantee, so a ticket there would buy determinism and nothing else --
# worth doing, but after the five that are actually losing episodes.
#
#     bash research_object.sh            all five, in the order below, ~72 h
#     bash research_object.sh 4 9        just those task ids
#
# EVERY TASK RUNS AT --max_episode_steps 280, which is eval's own cap, rather
# than the 150 used so far. T4 is why: its successes average 115.5 steps, so a
# 150-step cap was scoring real successes as failures and ranking candidates by
# speed instead of by whether they solve the task. T3 and T5 average 84 and 86,
# close enough that the same bias is plausible. Matching eval removes the whole
# question at 1.9x the wall clock, and a search whose success criterion differs
# from the one being reported is not measuring the thing.
#
# Ordered by expected survivors at --tickets 256, most likely first, so a
# session that dies partway has spent its hours on the tasks most likely to
# produce something:
#
#     T4  5.3 expected    T2  2.6    T9  2.6    T3  1.3    T5  0.6
#
# T5 at 0.6 is marginal by construction -- a 90% baseline means a 5/5 floor and
# its tickets passed only 30% of single layouts last time. Watch its first-hour
# line and kill it rather than spend thirteen hours confirming the estimate.
set -euo pipefail

REPO=/content/lerobot-piper
PY=/content/wilro/bin/python
CKPT=ISdept/wilro-wilromoe-8x4-22k-obs2
OUT=/content/drive/MyDrive/wilro_moe/object_tickets_retry

TASKS=${*:-4 2 9 3 5}

cd "$REPO" && git pull
cd "$REPO/src"

echo "### libero_object tasks: $TASKS   cap 280   -> $OUT"
MPLBACKEND=Agg "$PY" search_golden_ticket.py \
  --checkpoint "$CKPT" \
  --suites libero_object --task_ids $TASKS \
  --tickets 256 --envs_per_tier 5 --tiers 3 \
  --certify_layouts 15 \
  --seed 20000 \
  --max_episode_steps 280 \
  --out "$OUT"

echo
"$PY" ticket_bundle.py report "$OUT"

cat <<'MSG'

READ THE FIRST HOUR OF EACH TASK. Tier 1's first layout prints

    layout 1 pass rate NN% vs Gaussian MM%; the floor needs K/5,
    so about NNN candidates per survivor
    -> --tickets 256 expects N.NN survivors.  Killing this now ... costs less

Under 0.5, kill that task and move on -- it needs an order of magnitude more
candidates, and the remaining twelve hours only confirm the estimate. Restart
with the remaining ids:  bash research_object.sh 2 9 3

A task killed mid-search resumes where it stopped if you rerun it; progress and
the baseline both checkpoint per layout.

Merge refuses duplicate keys, and every one of these will collide with
object_tickets. Compare the two lines it prints -- it shows each side's search
rate and verdict -- and delete the loser before merging.
MSG
