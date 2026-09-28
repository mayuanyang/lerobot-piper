#!/usr/bin/env bash
# Re-search the two suite minima that are not goal's problem:
#
#   object T4  "pick up the ketchup"   SR 80%, HAS a ticket (search 12/15)
#   spatial T8 "black bowl next to the plate"  SR 85%, ticket was REJECTED
#
# They need different caps, which is the whole reason this is two invocations.
#
# object T4 runs at --max_episode_steps 280, matching eval. Its successes
# average 115.5 steps and its failures run to eval's 280 cap, so a 150-step
# search cap was scoring some of its real successes as failures -- the ranking
# became "who grabs the ketchup FAST" instead of "who grabs it". No other
# object task is near that: the rest average 66-103 steps.
#
# spatial T8 stays at 150. Its successes average 68.1 steps, so 150 is 2.2x the
# mean and the cap is not touching them. Raising it would cost 87% more wall
# clock for nothing.
#
# Both use a fresh --seed (candidates are seeded per (seed, suite, task), so the
# old seed redraws the old tickets) and --certify_layouts 15, which decides on
# layouts 35-49. Keep the discipline that makes the reported number honest:
#
#     20-34  search          35-49  decide          0-19  report, once
#
# Fresh --out directories, which is also why neither needs --overwrite: a
# directory with no bundle has nothing to skip, and progress resumes normally.
set -euo pipefail

REPO=/content/lerobot-piper
PY=/content/wilro/bin/python
CKPT=ISdept/wilro-wilromoe-8x4-22k-obs2
DRIVE=/content/drive/MyDrive/wilro_moe

cd "$REPO" && git pull
cd "$REPO/src"

echo "############ object T4 (cap 280, matches eval) ############"
MPLBACKEND=Agg "$PY" search_golden_ticket.py \
  --checkpoint "$CKPT" \
  --suites libero_object --task_ids 4 \
  --tickets 256 --envs_per_tier 5 --tiers 3 \
  --certify_layouts 15 \
  --seed 20000 \
  --max_episode_steps 280 \
  --out "$DRIVE/object_tickets_retry"

echo "############ spatial T8 (cap 150, successes avg 68 steps) ############"
MPLBACKEND=Agg "$PY" search_golden_ticket.py \
  --checkpoint "$CKPT" \
  --suites libero_spatial --task_ids 8 \
  --tickets 256 --envs_per_tier 5 --tiers 3 \
  --certify_layouts 15 \
  --seed 20000 \
  --max_episode_steps 150 \
  --out "$DRIVE/spatial_tickets_retry"

echo
echo "=== what came out ==="
"$PY" ticket_bundle.py report "$DRIVE/object_tickets_retry"  || true
"$PY" ticket_bundle.py report "$DRIVE/spatial_tickets_retry" || true

cat <<'MSG'

READ THE FIRST HOUR, NOT THE LAST. Tier 1's first layout prints

    layout 1 pass rate NN% vs Gaussian MM%; the floor needs K/5,
    so about NNN candidates per survivor
    -> --tickets 256 expects N.NN survivors.  Killing this now ... costs less

If it says 256 expects under 0.5 survivors, kill it: the task needs an order of
magnitude more candidates and the remaining eleven hours only confirm that.
object T5 spent 3.7 hours doing exactly that.

Then merge only the ones whose certification PASSED. object T4 already has a
ticket in object_tickets, so merge will refuse the duplicate and print both
sides' numbers -- delete the loser from its bundle first.
MSG
