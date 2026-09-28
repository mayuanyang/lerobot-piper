#!/usr/bin/env bash
# Re-search goal T5 / T6 / T9, the three tickets that cleared the search's own
# check and then scored 70 / 75 / 55 on the reported layouts.
#
# Two things differ from the run that produced them, and BOTH are needed:
#
#   --seed 20000      Candidates are seeded per (seed, suite, task), so the
#                     same seed redraws the same tickets and reaches the same
#                     answer. --overwrite alone would burn 12 hours proving it.
#
#   --certify_layouts 15
#                     The winner and Gaussian both run on layouts 35-49, which
#                     take no part in the selection. A ticket that does not win
#                     there is banked as weak and eval uses Gaussian. This is
#                     the gate the three of them needed: the search rate is a
#                     MAXIMUM over candidates, so it is optimistic by
#                     construction and cannot be compared to an unselected
#                     baseline.
#
# --tickets 256 rather than 1024: more candidates means more selection pressure
# on the same 15 search layouts, which is the direction that produced the
# problem. Certification is the fix; the ticket count is not. 256 raises the
# chance a genuinely general ticket exists in the pool without turning the
# search into a layout-overfitting machine.
#
# Writes to a SEPARATE directory. save_ticket is read-modify-write with no
# lock, and the current bundle holds six goal tickets that are working -- do
# not risk them. Merge after you have checked the report.
set -euo pipefail

REPO=/content/lerobot-piper
PY=/content/wilro/bin/python
CKPT=ISdept/wilro-wilromoe-8x4-22k-obs2
OUT=/content/drive/MyDrive/wilro_moe/goal_tickets_retry

cd "$REPO" && git pull
cd "$REPO/src"

MPLBACKEND=Agg "$PY" search_golden_ticket.py \
  --checkpoint "$CKPT" \
  --suites libero_goal \
  --task_ids 5 6 9 \
  --tickets 256 \
  --envs_per_tier 5 --tiers 3 \
  --certify_layouts 15 \
  --seed 20000 \
  --max_episode_steps 150 \
  --out "$OUT"

echo
echo "=== what came out ==="
"$PY" ticket_bundle.py report "$OUT"
cat <<'MSG'

Next, only for the tasks whose certification PASSED:

  python ticket_bundle.py report /content/drive/MyDrive/wilro_moe/goal_tickets
  # delete libero_goal.5/6/9 from the OLD bundle if the retry beat them,
  # then merge -- merge refuses a duplicate key rather than picking one.

  python ticket_bundle.py merge \
    /content/drive/MyDrive/wilro_moe/all_tickets \
    /content/drive/MyDrive/wilro_moe/goal_tickets \
    /content/drive/MyDrive/wilro_moe/goal_tickets_retry \
    /content/drive/MyDrive/wilro_moe/object_tickets \
    /content/drive/MyDrive/wilro_moe/spatial_tickets

A ticket that fails certification is inert -- eval reads beats_baseline and
runs Gaussian -- so leaving it in the bundle costs nothing.
MSG
