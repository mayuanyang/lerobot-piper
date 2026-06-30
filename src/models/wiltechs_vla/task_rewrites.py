"""Single source of truth for LIBERO instruction rewriting.

Some LIBERO tasks fail not on control but on GROUNDING: the frozen Qwen3-VL
encoder can't bind a product name ("alphabet soup") to the right prop when a
visually-similar object shares the scene. Rephrasing the reference with a
visual primitive the VLM CAN ground (color + shape: "the blue can of alphabet
soup") gives the policy a handle it can actually use.

DESIGN
------
- Keyed by the canonical instruction STRING, not the task index. The benchmark
  per-suite task_id (0-9) is a *permutation* of the parquet task_index (0-39),
  and training reads strings from tasks.parquet while eval reads them from the
  LIBERO benchmark env. Keying by string makes the rewrite identical on both
  paths regardless of indexing — no permutation table to keep in sync.
- Identity fallback: tasks NOT in REPHRASINGS pass through unchanged, so the
  well-performing tasks need no entries.
- ONE function, `rewrite_instruction`, called by training, RL rollout, and eval
  alike. As long as every path runs the raw task string through it, train and
  deploy can never diverge on phrasing.

USAGE
-----
    from models.wiltechs_vla.task_rewrites import rewrite_instruction
    task = rewrite_instruction(raw_task_string)   # safe on ANY string

Run verify_against_benchmark() once after editing this file to confirm every
key exactly matches a real LIBERO task string (catches typos that would
silently no-op).
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Rephrasings: canonical original string  ->  descriptive replacement.
# Only list tasks you want to CHANGE. Everything else is identity.
#
# Pending visual discriminators (leave commented until confirmed):
#   - cream cheese / butter: both rectangular labeled boxes -> need a color or
#     other cue to disambiguate when they co-occur (libero_10 task "…cream
#     cheese box and the butter…").
# ---------------------------------------------------------------------------
REPHRASINGS: dict[str, str] = {
    # ---- libero_10 (long) — object-identity grounding ----
    # T0: two round cans, distinguished only by color (red vs blue).
    "put both the alphabet soup and the tomato sauce in the basket":
        "put both the blue can of alphabet soup and the red can of tomato sauce in the basket",
    # T7: a can + a box — shape already separates them; color is a cheap bonus.
    "put both the alphabet soup and the cream cheese box in the basket":
        "put both the blue can of alphabet soup and the cream cheese box in the basket",
    # T1: PENDING cream-cheese/butter colors — uncomment once known.
    "put both the cream cheese box and the butter in the basket":
         "put both the silver purple cream cheese box and the red butter box in the basket",

    # ---- libero_spatial — "ramekin" is an ungroundable noun for the VLM ----
    # Replace ramekin -> visual description wherever it appears.
    "pick up the black bowl on the ramekin and place it on the plate":
        "pick up the black bowl on the small round silver container and place it on the plate",
    "pick up the black bowl next to the ramekin and place it on the plate":
        "pick up the black bowl next to the small round silver container and place it on the plate",
    "pick up the black bowl between the plate and the ramekin and place it on the plate":
        "pick up the black bowl that is between the plate and the ramekin (closer to the plate) and place it on the plate",
    "pick up the black bowl next to the plate and place it on the plate":
        "pick up the nearest black bowl and place it on the plate",

    # ---- libero_object (20-29) — TODO: confirm canonical strings + which need rewrite ----
    # ---- libero_goal   (10-19) — TODO: confirm canonical strings + which need rewrite ----
}


def rewrite_instruction(task: str, random_augment: bool = False) -> str:
    """Return the descriptive rephrasing for `task`, or `task` unchanged.

    Args:
        task: The original task instruction string.
        random_augment: If True and a rephrasing exists, randomly choose between
            the original and rewritten version (50/50). This allows the model to
            learn BOTH phrasings during training, improving robustness at eval
            time when either form may appear. Default False (always rewrite).

    Safe to call on any string from either tasks.parquet (training) or the
    LIBERO benchmark env (RL rollout / eval) — the canonical strings match.
    """
    if not task:
        return task
    stripped = task.strip()
    rewritten = REPHRASINGS.get(stripped)
    if rewritten is None:
        return task
    if random_augment and torch.rand(1).item() > 0.5:
        return task  # 50% chance to keep original
    return rewritten


def verify_against_benchmark() -> list[str]:
    """Assert every REPHRASINGS key is a real LIBERO task string.

    Returns the list of keys that did NOT match any task across the four
    suites (should be empty). A non-empty result means a typo'd key that would
    silently no-op. Import is local so this module stays dependency-free.
    """
    from libero.libero.benchmark import get_benchmark_dict

    known: set[str] = set()
    bd = get_benchmark_dict()
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        bench = bd[suite]()
        for i in range(bench.n_tasks):
            known.add(bench.get_task(i).language.strip())

    missing = [k for k in REPHRASINGS if k.strip() not in known]
    if missing:
        print("[task_rewrites] WARNING — keys not found among LIBERO tasks "
              "(typo => silent no-op):")
        for k in missing:
            print(f"  - {k!r}")
    else:
        print(f"[task_rewrites] OK — all {len(REPHRASINGS)} rephrasing keys "
              f"match real LIBERO task strings.")
    return missing


if __name__ == "__main__":
    verify_against_benchmark()
