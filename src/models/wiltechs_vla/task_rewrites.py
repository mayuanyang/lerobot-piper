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

CHAIN-OF-THOUGHT (CoT) REWRITES
-------------------------------
For tasks where the model mis-grounds the target object (e.g. grasping the
midpoint between two reference objects instead of the actual target), a
flat descriptive rephrase is insufficient — the model needs an explicit
reasoning trace that:

  1. Names the TARGET object and its visual signature (color/shape/material).
  2. Locates the target RELATIVE to the anchors (spatial relation + which side).
  3. Resolves common confusions (e.g. "not the midpoint", "not the anchor").
  4. States the ACTION to perform on the resolved target.

Because the VLM (Qwen3-VL) is frozen and does NOT autoregressively generate
text at inference time, the CoT cannot be "generated" by the model itself.
Instead, the CoT is PRE-WRITTEN in this table and injected as the language
input the VLM encodes. The VLM's KV cache then carries the structured
grounding trace, which the trainable DiT decoder cross-attends to at every
denoising step. Training on these CoT strings teaches the DiT to *use* the
trace; at eval/RL-rollout time the identical string flows through
`rewrite_instruction`, so the model sees the same reasoning it was trained on
— no runtime generation required.

This is analogous to "pre-filled assistant reasoning" in VLA literature: the
reasoning is provided as context, not produced online.

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
# Rephrasings: canonical original string  ->  replacement (may include CoT).
#
# Two value styles are supported (both are plain strings):
#
#   (a) Flat descriptive rephrase  —  original nouns swapped for visual
#       primitives, no reasoning trace. Backwards compatible.
#
#   (b) Chain-of-thought rephrase  —  a multi-clause string that first
#       identifies the target, then locates it, then states the action. Use
#       the helper `cot(...)` (defined below) to keep the format consistent.
#
# The model does NOT differentiate between (a) and (b): both are just the
# language string the VLM encodes. The CoT form is strictly richer and tends
# to help on tasks where flat rephrasing still leaves spatial ambiguity.
#
# Pending visual discriminators (leave commented until confirmed):
#   - cream cheese / butter: both rectangular labeled boxes -> need a color or
#     other cue to disambiguate when they co-occur (libero_10 task "…cream
#     cheese box and the butter…").
# ---------------------------------------------------------------------------

def cot(
    target: str,
    location: str,
    action: str,
    *,
    not_the: str = "",
    visual: str = "",
) -> str:
    """Format a chain-of-thought grounding trace as a single instruction string.

    The output is a compact, declarative trace the frozen VLM can encode and
    the trainable DiT can cross-attend to. It is NOT free-form generation —
    it is a pre-written reasoning template filled per task.

    Args:
        target:   The object to grasp, with visual discriminators
                  (e.g. "the black bowl" or "the blue can of alphabet soup").
        location: How to FIND the target. Use the spatial relation only to pick
                  WHICH object among look-alikes; never to describe where in the
                  frame to reach. Phrasings that parameterise a position on the
                  anchor axis ("in the gap between A and B", "closer to the A
                  side", "the midpoint of") are read literally by the policy and
                  make it interpolate between the anchors instead of localising
                  the object — see the "between" entry below. Prefer
                  "among the <look-alikes>, the target is the one that <relation>"
                  plus "aim at the center of the <object> itself".
        action:   The action to perform, phrased as an imperative
                  (e.g. "grasp the black bowl and place it on the plate").
        not_the:  Optional anti-grounding cue for the most likely confusion, and
                  only when the confusion is another OBJECT ("not the container
                  itself"). Do not use it against a POSITION: negation is a weak
                  signal for a frozen encoder, so "not the midpoint between A and
                  B" mostly just injects "midpoint between A and B".
        visual:   Optional extra visual signature for the target
                  (e.g. "a round dark ceramic object").

    Returns:
        A single string. Example:
        "Target: the black bowl — a round dark ceramic object. Location: among
        the black bowls in the scene, the target is the one whose position falls
        between the plate and the ramekin. Aim at the center of the bowl itself.
        Action: grasp the black bowl and place it on the plate."
    """
    parts = [f"Target: {target}"]
    if visual:
        parts.append(f" — {visual}")
    parts.append(f". Location: {location}")
    if not_the:
        parts.append(f"; {not_the}")
    parts.append(f". Action: {action}.")
    return "".join(parts)


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

    # ---- libero_spatial ----
    # The failing task grounds the SPATIAL RELATION (between/next-to/on) to the
    # GEOMETRIC MIDPOINT of the anchors rather than to the object that satisfies
    # the relation. A flat rephrase ("between the plate and the ramekin, closer
    # to the plate") still leaves the midpoint as a strong attractor. The CoT
    # form names the target first, so the DiT attends to object identity, then
    # relation.
    #
    # ONLY REWRITE WHAT FAILS. In every one of the 10 tasks the BDDL target is
    # akita_black_bowl_1 and the distractor akita_black_bowl_2, so the referring
    # expression's whole job is to separate two bowls. Measured on init state 0
    # (src/kv_grounding_probe.py --list_bodies), distance ratio between the two
    # bowls under each task's OWN named anchor:
    #
    #   t0 between the plate and the ramekin  ramekin  1.20x   <-- the outlier
    #   t1 next to the ramekin                ramekin  3.61x
    #   t2 from table center                  centre   4.02x
    #   t3 on the cookie box                  cookies    inf   (stacked)
    #   t4 in the top drawer                  cabinet 13.28x   (in drawer vs on top)
    #   t5 on the ramekin                     ramekin 20.82x   (stacked)
    #   t6 next to the cookie box             cookies  3.33x
    #   t7 on the stove                       stove    2.91x   (support surface)
    #   t8 next to the plate                  plate    2.19x
    #   t9 on the wooden cabinet              cabinet 17.83x   (stacked)
    #
    # Nine of ten relations are >=2x, most of them stacking or support-surface
    # relations that are visually unambiguous. Only "between" is weak, because it
    # names a LINE rather than an anchor, and the target's position along that
    # line changes every episode. That -- not vocabulary -- is why this one task
    # fails while libero_spatial as a whole sits at ~87%.
    #
    # Two corollaries, both learned by getting them wrong first:
    #
    #  * Do NOT unify the tasks onto the plate anchor. It is the right anchor for
    #    t0 (2.30x vs the ramekin's 1.20x) and for t8, and useless elsewhere --
    #    for t1 the plate separates the bowls 1.01x, a coin flip.
    #  * Do NOT rewrite the nine working tasks to purge "black"/"ramekin". The
    #    qwen_color_probe evidence that Qwen cannot see a "black bowl" comes from
    #    its GENERATIVE path; the policy only ever reads cross-attention KV, and
    #    ~87% success is direct evidence that path grounds these nouns fine.
    #    Rewriting working tasks risks the minimal-pair collateral for no gain.
    "pick up the black bowl on the ramekin and place it on the plate":
        cot(
            target="the black bowl",
            location="on top of the small round silver container (ramekin)",
            action="grasp the black bowl and place it on the plate",
            visual="a round dark bowl, distinct from the flat plate and the small silver ramekin",
            not_the="not the container itself",
        ),
    "pick up the black bowl next to the ramekin and place it on the plate":
        cot(
            target="the black bowl",
            location="next to the small round silver container (ramekin)",
            action="grasp the black bowl and place it on the plate",
            visual="a round dark bowl",
            not_the="not the container itself",
        ),
    # "between" is a SELECTOR, not a position. The scene holds TWO identical
    # black bowls: akita_black_bowl_1 is the target, akita_black_bowl_2 the
    # distractor (also the target of the "next to the ramekin" task).
    #
    # Ground truth from the sim, one initial state (xy metres, --list_bodies):
    #   ramekin (-0.211, 0.201)   plate (0.053, 0.209)
    #   bowl_1  (-0.058, 0.210)   t=0.58 along ramekin->plate,   4mm off the line
    #   bowl_2  (-0.175, 0.323)   t=0.15,                      121mm off the line
    #
    # So the target really is ON the plate->ramekin segment -- an earlier note
    # here claimed the three objects were not collinear, which came from
    # eyeballing a screenshot and was wrong. That makes the relation
    # geometrically true but still useless as a coordinate: the target sits at
    # t=0.58 in this layout and elsewhere in others, so "between" pins the LINE
    # and nothing along it.
    #
    # Which is exactly what the policy learned. The old wording described a 1-D
    # parametric point on that axis ("in the gap between ..., closer to the
    # plate side") and it was executed literally: after the text-first fix the
    # arm stopped aiming at the midpoint and started landing on bare table on
    # the line -- right line, wrong point along it, because picking the point
    # requires seeing the bowl.
    #
    # The wording below is not guesswork -- qwen_color_probe.py was run on the
    # real env frame at 64 / 256 / 1024 vision tokens with three question sets,
    # and only one combination survived (see --probe spatial{,_visual,_plate}):
    #
    #   probe           vocabulary   anchor    @256 tok (= --vision_input_size 512)
    #   spatial         LIBERO       ramekin   FAIL "there are 0 black bowls"
    #   spatial_visual  perceptual   ramekin   FAIL relation answered backwards
    #   spatial_plate   perceptual   plate     PASS consistent + matches truth
    #
    # Two independent defects, both of which had to go:
    #
    # 1. VOCABULARY. Qwen does not see a "black bowl". At 256 tokens it replies
    #    "there are 0 black bowls ... two metallic, possibly aluminum, bowls
    #    that are silver in color", and it calls the ramekin a "small silver
    #    tray". Higher resolution makes this WORSE, since it resolves the
    #    speckled contents and reads them as metal. A referring expression whose
    #    nouns the encoder rejects cannot resolve at any resolution. So: no
    #    colour words, no "ramekin" -- the containers are separated by depth,
    #    size, emptiness and contents, which is how Qwen itself described them.
    #
    # 2. ANCHOR. The ramekin shares a vision token with the distractor bowl
    #    (~19 native px apart on a 256px frame, one token spans 32px), so
    #    "distance to the ramekin" asks the policy to measure from a landmark it
    #    cannot separate from one of the things being measured. The plate is
    #    large, uniquely coloured and far from both bowls, and every
    #    plate-anchored judgement in the probe was correct from 256 tokens up.
    #    The swap is safe, and the sim numbers say it is the better anchor by a
    #    wide margin. Both selectors pick the target, but not equally well:
    #      "nearest the plate"    bowl_1 0.111 m vs bowl_2 0.255 m -> 2.30x
    #      "farthest from ramekin" bowl_1 0.153 m vs bowl_2 0.127 m -> 1.20x
    #    A 1.20x margin flips on a 20% distance-estimation error, which is what
    #    the probe did when anchored on the ramekin. 2.30x has room to be wrong.
    #
    # Still no coordinate the policy can interpolate -- "whichever of those two"
    # is a binary choice over discrete candidates, not a position. "Aim at the
    # center of the target bowl itself" pins the last step to vision. The
    # not_the clause stays gone: negation is weak for a frozen encoder, and the
    # old "not the midpoint between the plate and the ramekin" merely injected
    # "midpoint between the plate and the ramekin".
    #
    # Do NOT encode the observed offset direction: placement is randomised per
    # episode, so a hard-coded "to the right" overfits one layout.
    #
    # REQUIRES --vision_input_size 512 (16x16 grid). At the default 8x8 the
    # probe FAILED its own consistency control, naming the same bowl as both
    # nearest and farthest from the plate. This wording buys nothing at 64
    # tokens; the two changes ship together or not at all.
    "pick up the black bowl between the plate and the ramekin and place it on the plate":
        cot(
            target="the deep speckled bowl",
            location="two deep bowls hold speckled granular contents; the target is "
                     "whichever of those two is nearer to the round white plate with the "
                     "red rim, and the other one is not the target. Aim at the center of "
                     "the target bowl itself",
            action="grasp that bowl and place it on the round white plate with the red rim",
            visual="a bowl with a dark rim and speckled granular contents, clearly deeper "
                   "than the small shallow empty cup",
        ),
    "pick up the black bowl next to the plate and place it on the plate":
        cot(
            target="the nearest black bowl",
            location="next to the plate",
            action="grasp the nearest black bowl and place it on the plate",
            visual="a round dark bowl",
            not_the="not the plate itself",
        ),

    # ---- libero_object (20-29) — TODO: confirm canonical strings + which need rewrite ----
    # ---- libero_goal   (10-19) — TODO: confirm canonical strings + which need rewrite ----
}


def rewrite_instruction(task: str, random_augment: bool = False) -> str:
    """Return the (possibly CoT-enriched) rephrasing for `task`, or `task` unchanged.

    Args:
        task: The original task instruction string.
        random_augment: If True and a rephrasing exists, randomly choose between
            the original and rewritten version (50/50). This allows the model to
            learn BOTH phrasings during training, improving robustness at eval
            time when either form may appear. Default False (always rewrite).

    Safe to call on any string from either tasks.parquet (training) or the
    LIBERO benchmark env (RL rollout / eval) — the canonical strings match.

    NOTE ON INFERENCE-TIME CoT: because the VLM is frozen and does not
    generate text, the chain-of-thought is NOT produced at inference time —
    it is pre-written in this table and injected as the language input. The
    model is trained on these CoT strings, so at eval it simply receives the
    same CoT-enriched instruction via this function. No VLM autoregression,
    no separate "reasoning model" call, no extra latency at inference.
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