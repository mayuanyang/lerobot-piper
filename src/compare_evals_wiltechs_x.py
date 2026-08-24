#!/usr/bin/env python
"""Compare two eval JSONs as a PAIRED sample.

Why this exists. Two checkpoints evaluated at 20 episodes gave 65% and 45% on
one task, which reads as "2000 more steps made it worse". It is not: the
unpaired standard error of that difference is ~15pp, so a 20-point gap is
z = 1.3, p = 0.19 -- noise. Earlier in this project the SAME checkpoint, task
and config scored 80% and 90% on two runs.

Episode i always starts from the same canonical init state (fixed_init_states),
so the two runs are paired and McNemar applies to the discordant pairs only.
That is a much sharper test than comparing the two rates, and it needs no extra
episodes -- only the per-episode vector that `eval_wiltechs_x` now records.

    python compare_evals_wiltechs_x.py before.json after.json

Older JSONs have no `episode_success`; those tasks fall back to the unpaired
two-proportion z-test and are marked UNPAIRED.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _tasks(payload: dict) -> dict:
    """-> {(suite, task_id): entry} over every suite in the file."""
    out = {}
    for suite, d in payload.get("suites", {}).items():
        for tid, entry in d.get("per_task", {}).items():
            out[(suite, str(tid))] = entry
    return out


def _mcnemar(a: list[int], b: list[int]):
    """-> (b01, b10, two-sided exact p) for paired binary outcomes.

    b01 = was a failure in A and a success in B; b10 = the reverse. Only the
    discordant pairs carry information, which is exactly why pairing is worth
    it: episodes both runs get right (or both get wrong) add noise to a rate
    comparison and nothing to a paired one.
    """
    b01 = sum(1 for x, y in zip(a, b) if not x and y)
    b10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    # Exact binomial test at p=0.5 over the discordant pairs. n is small enough
    # (<= episodes) that the normal approximation is not worth its error.
    k = min(b01, b10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return b01, b10, min(1.0, 2 * tail)


# The commit that stopped task ORDER from reaching the policy noise. Anything
# written before it used a noise stream that depended on which tasks ran first,
# which is how the same checkpoint scored task 0 at 45% and 85%. Such a file can
# still be read as a level estimate; it cannot be one arm of an A/B.
SEED_FIX = "bc86296"


def _predates(commit: str) -> bool | None:
    """True if `commit` is an ancestor of SEED_FIX (i.e. older). None if unknown."""
    import subprocess
    from pathlib import Path as _P
    try:
        r = subprocess.run(
            ["git", "-C", str(_P(__file__).resolve().parent), "merge-base",
             "--is-ancestor", commit, SEED_FIX],
            capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return None


def _warn_build(ca, cb):
    for name, c in (("before", ca), ("after", cb)):
        if not c:
            print(f"  [WARN] the {name} file records no eval_commit -- it "
                  f"predates {SEED_FIX}, so its policy noise depended on task "
                  f"ORDER. Not usable as an A/B arm; re-run it.")
        elif _predates(c) and c != SEED_FIX:
            print(f"  [WARN] the {name} file was written by {c}, which predates "
                  f"the {SEED_FIX} seeding fix: task ORDER reached the policy "
                  f"noise there. Re-run it before trusting any delta below.")
    if ca and cb and ca != cb:
        print(f"  [note] different builds ({ca} vs {cb}); fine if neither "
              f"predates {SEED_FIX} and nothing in the eval path changed")


def _unpaired_z(n1, k1, n2, k2):
    p1, p2 = k1 / n1, k2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    if se == 0:
        return 0.0, 1.0
    z = (p2 - p1) / se
    p = math.erfc(abs(z) / math.sqrt(2))
    return z, p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()

    A = json.loads(Path(a.before).read_text())
    B = json.loads(Path(a.after).read_text())

    for key in ("n_action_steps", "num_inference_steps", "control_freq",
                "episodes_per_task", "fixed_init_states", "seed",
                "ablate_lang", "instruction_override"):
        va, vb = A.get(key, "<absent>"), B.get(key, "<absent>")
        if va != vb:
            print(f"  [WARN] {key}: {va!r} vs {vb!r} -- the runs are not "
                  f"comparable on this axis")
    if A.get("seed", "<absent>") == "<absent>" or B.get("seed", "<absent>") == "<absent>":
        print("  [WARN] a file predates seed recording; if it also predates "
              "92ec163 the policy noise was unseeded there")
    _warn_build(A.get("eval_commit"), B.get("eval_commit"))

    ta, tb = _tasks(A), _tasks(B)
    shared = sorted(set(ta) & set(tb))
    if not shared:
        raise SystemExit("no task appears in both files")

    print(f"\n{'task':<34} {'before':>7} {'after':>7} {'delta':>7}  test")
    print("-" * 78)
    d_sum, n_pair, n_sig = 0.0, 0, 0
    for k in shared:
        ea, eb = ta[k], tb[k]
        sa, sb = ea["success_rate"], eb["success_rate"]
        d_sum += sb - sa
        va, vb = ea.get("episode_success"), eb.get("episode_success")
        name = f"{k[0].replace('libero_', '')} {k[1]}: {ea['task'][:20]}"
        if va and vb and len(va) == len(vb):
            b01, b10, p = _mcnemar(va, vb)
            n_pair += 1
            mark = " *" if p < a.alpha else ""
            n_sig += p < a.alpha
            test = f"McNemar {b10}->fail {b01}->pass  p={p:.3f}{mark}"
        else:
            z, p = _unpaired_z(ea["n_episodes"], ea["n_success"],
                               eb["n_episodes"], eb["n_success"])
            test = f"UNPAIRED z={z:+.2f} p={p:.3f}"
        print(f"{name:<34} {sa:6.1f}% {sb:6.1f}% {sb - sa:+6.1f}   {test}")

    n = len(shared)
    mean_d = d_sum / n
    # The suite mean is the estimator to judge a checkpoint on: averaging over
    # tasks divides the per-task binomial noise by sqrt(n_tasks), which is why
    # one task at 20 episodes cannot rank two checkpoints and ten can.
    var = sum((tb[k]["success_rate"] - ta[k]["success_rate"] - mean_d) ** 2
              for k in shared) / (n - 1) if n > 1 else float("nan")
    se = math.sqrt(var / n) if n > 1 else float("nan")
    print("-" * 78)
    print(f"{'MEAN over ' + str(n) + ' task(s)':<34} "
          f"{sum(ta[k]['success_rate'] for k in shared) / n:6.1f}% "
          f"{sum(tb[k]['success_rate'] for k in shared) / n:6.1f}% "
          f"{mean_d:+6.1f}", end="")
    if n > 1:
        t = mean_d / se if se else 0.0
        print(f"   paired t={t:+.2f} over tasks (SE {se:.1f}pp)")
    else:
        print("   -- one task cannot establish a trend; see the header")
    print(f"\n{n_pair}/{n} task(s) compared paired; {n_sig} at p < {a.alpha}")
    if n == 1:
        print("A single task at 20 episodes has ~11pp of binomial noise, so a\n"
              "20-point gap is not a result. Add tasks before adding episodes:\n"
              "the mean over 10 tasks has ~3pp, the same 200 episodes spent\n"
              "on one task still tells you nothing about the other nine.")


if __name__ == "__main__":
    main()
