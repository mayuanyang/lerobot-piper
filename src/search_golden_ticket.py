"""Search for a golden ticket: one constant x_1 that beats sampling from N(0,I).

Patil et al. 2026, "You've Got a Golden Ticket". A frozen diffusion/flow policy
is improved by replacing the per-step Gaussian draw with a single well-chosen
noise vector, found by Monte-Carlo rollout search. No weights change, no new
network is trained.

WHY THIS FITS THIS PROJECT'S OWN MEASUREMENTS. The 2026-09-23 best-of-N result
showed chunk-level selection buys nothing here (p=1.0000 against a random-pick
control): four draws at one state are interchangeable. But a `policy_seed`
change -- which is a change to the WHOLE episode's noise -- flips outcomes, and
this file prices the per-chunk re-draw at 25 points. A constant ticket acts at
the episode level, which is the level that was measured to matter.

SEARCH AND EVAL LAYOUTS ARE DISJOINT, BY CONSTRUCTION. LIBERO ships 50
canonical initial states and eval_task maps episode index -> layout id
directly, so a standard 20-episode eval uses ids 0-19 and nothing else. This
searches at --init_state_offset 20 by default, leaving 0-19 untouched: the
ticket is then REPORTED on exactly the layouts every number in the tracker was
measured on, without having been fitted to them.

    python src/search_golden_ticket.py \
        --checkpoint ISdept/wilro-wilromoe-8x4-22k-obs2 \
        --suites libero_goal --task_ids 9 \
        --tickets 128 --out ./tickets

Then, to report it:

    python src/eval_wiltechs_x.py --checkpoint <same> --suites libero_goal \
        --task_ids 9 --episodes 20 --noise_ticket ./tickets/<file>.npy
"""

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

def binom_ub(k, n, conf=0.90):
    """One-sided Clopper-Pearson upper bound on the true rate given k of n.

    Used to prune candidates that CANNOT plausibly reach the baseline, which
    is the only pruning rule that carries no risk of dropping a good ticket.
    A flat threshold does carry that risk and is also mis-calibrated: 80% sits
    12 points ABOVE goal's 68% search baseline and 6 points BELOW object's
    86.5%, so the same number is too strict on one suite and keeps
    worse-than-Gaussian tickets on the other. At 5 episodes a flat 80% drops a
    truly-75% ticket 37% of the time and a truly-89% ticket 10% of the time --
    and goal T0's winning ticket sits at 0.81-0.89 posterior.
    """
    from math import comb
    lo, hi = 0.0, 1.0
    for _ in range(60):
        m = (lo + hi) / 2
        cdf = sum(comb(n, i) * m ** i * (1 - m) ** (n - i) for i in range(k + 1))
        lo, hi = (m, hi) if cdf > 1 - conf else (lo, m)
    return (lo + hi) / 2


def binom_sf(k, n, p):
    """P(X >= k) for X ~ Bin(n, p), exact, no scipy."""
    from math import comb
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def dispersion(wins, runs):
    """Do the tickets actually DIFFER, or is the spread just binomial noise?

    This is the question tier 1 exists to answer, and eyeballing the spread
    cannot answer it: with 5 layouts per ticket, 64 IDENTICAL tickets at p=0.7
    still produce observed rates with sd 0.205, i.e. a typical range of
    29%-100%. Any histogram of that looks convincingly "spread out".

    The test is for overdispersion. Under H0 every ticket has the same true
    rate p, so w_i ~ Bin(M, p) and

        chi2 = sum_i (w_i - M p)^2 / (M p (1-p))   ~   chi2(N-1)

    dispersion = chi2/df is 1.0 when the tickets are interchangeable and grows
    with real between-ticket variance. z = (chi2-df)/sqrt(2 df) is the
    normal-approximation score, which is accurate at these df.

    Also returns sd_between, the between-ticket sd left after subtracting the
    binomial part. It is the effect size in success-rate units, and it can sit
    BELOW sd_binomial_only while dispersion is large -- that is not a
    contradiction. When most tickets score zero and a few score high, chi2
    responds to the tail and the variance decomposition does not.

    Fed the CUMULATIVE totals, including tickets already eliminated. The
    halving selects on score, so that looks like it should inflate the
    statistic; simulated under H0 (64 tickets all at p=0.03, three tiers with
    halving) it does not -- cumulative dispersion runs 0.64-0.69, if anything
    conservative -- and cumulative has more episodes behind it than one tier
    does.
    """
    import math
    w = np.asarray(wins, float); r = np.asarray(runs, float)
    keep = r > 0
    w, r = w[keep], r[keep]
    N = len(w)
    if N < 2 or r.sum() == 0:
        return None
    pbar = w.sum() / r.sum()
    if not (0 < pbar < 1):
        return {"n": N, "p_bar": pbar, "note": "every ticket identical "
                "(all 0 or all 1); no variance to test"}
    chi2 = float((((w - r * pbar) ** 2) / (r * pbar * (1 - pbar))).sum())
    df = N - 1
    var_obs = float(np.var(w / r, ddof=1))
    var_bin = float(pbar * (1 - pbar) * np.mean(1.0 / r))
    return {"n": N, "p_bar": round(pbar, 4), "chi2": round(chi2, 1), "df": df,
            "dispersion": round(chi2 / df, 3),
            "z": round((chi2 - df) / math.sqrt(2 * df), 2),
            "sd_between": round(max(0.0, var_obs - var_bin) ** 0.5, 4),
            "sd_binomial_only": round(var_bin ** 0.5, 4)}


import eval_wiltechs_x as ev
import ticket_bundle as tb


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--suites", nargs="+", default=["libero_goal"])
    p.add_argument("--task_ids", nargs="+", type=int, default=None)
    p.add_argument("--out", default="./tickets")
    p.add_argument("--tickets", type=int, default=128,
                   help="Candidates per task. The paper used 1081-1416 per "
                        "LIBERO task; the returns are steep at the start "
                        "because most random tickets are bad, so a few hundred "
                        "already surfaces something.")
    p.add_argument("--envs_per_tier", type=int, default=5,
                   help="Layouts a candidate is scored on in its first tier. "
                        "Survivors accumulate another --envs_per_tier at each "
                        "subsequent tier.")
    p.add_argument("--tiers", type=int, default=4,
                   help="Sequential halving: every candidate is scored on tier "
                        "1, the bottom half is dropped, survivors get a fresh "
                        "disjoint tier, and so on. Cost is about 2 x tickets x "
                        "envs_per_tier instead of tickets x (tiers x "
                        "envs_per_tier), and the deepest survivors are still "
                        "scored at full fidelity.")
    p.add_argument("--init_state_offset", type=int, default=20,
                   help="First canonical layout used for SEARCH. 20 keeps the "
                        "reportable 0-19 out of the search entirely. Lower it "
                        "only if you intend to overfit on purpose.")
    p.add_argument("--num_envs", type=int, default=10)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--max_episode_steps", type=int, default=0,
                   help="Cap search rollouts shorter than eval's to buy "
                        "candidates: failures run to the cap and dominate the "
                        "wall clock, while successes average ~85 steps. 0 uses "
                        "the env's own cap.")
    p.add_argument("--dataset_id", default=None)
    p.add_argument("--num_inference_steps", type=int, default=None)
    p.add_argument("--n_action_steps", type=int, default=2,
                   help="Steps of each chunk executed before replanning. The "
                        "CHECKPOINT SAYS 64 AND EVERY EVAL IN THIS PROJECT "
                        "PASSES 2 -- leaving it at the checkpoint's value runs "
                        "a policy that replans twice per episode instead of "
                        "150, which is the same mismatch that made the RFT "
                        "collector return 0/200. A ticket is only valid for "
                        "the inference config it was searched under, so this "
                        "must match the eval command.")
    p.add_argument("--vision_input_size", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--control_freq", type=int, default=10,
                   help="MUST be 10: the LIBERO datasets are 10 Hz and the "
                        "stock env is 20, so a search at 20 optimises a ticket "
                        "for a policy that is not the one being reported.")
    p.add_argument("--render_gpu", type=int, default=0)
    p.add_argument("--stock_init", action="store_true",
                   help="Use lerobot's unpatched reset order, i.e. the sampler "
                        "distribution. Matches --stock_init in eval and is not "
                        "for anything reportable.")
    p.add_argument("--baseline_layouts", type=int, default=0,
                   help="Layouts the Gaussian reference is measured on. 0 "
                        "means --envs_per_tier, which is fine at 5 and wrong "
                        "at 2: splitting tier 1 finer to save time would drop "
                        "the baseline to two layouts while the winner is "
                        "judged on ten, and the beats_baseline comparison "
                        "would then be across different layout sets. Set it to "
                        "5 whenever --envs_per_tier is below 5.")
    p.add_argument("--keep_frac", type=float, default=0.5,
                   help="Fraction of survivors carried to the next tier. 0.5 "
                        "is plain sequential halving. 0.25 cuts the later "
                        "tiers by about 40%% and on goal T0's real tier-1 "
                        "distribution the top quarter still contains every "
                        "4/5 and 5/5 -- but this is a BUDGET knob, not a "
                        "correctness one: a smaller fraction can drop a good "
                        "ticket that had a bad five episodes.")
    p.add_argument("--prune_confidence", type=float, default=0.0,
                   help="Also drop candidates whose one-sided upper confidence "
                        "bound at this level is BELOW the Gaussian baseline -- "
                        "tickets that statistically cannot be as good as not "
                        "using one. Unlike a flat threshold this is calibrated "
                        "to the task's own baseline and to how many episodes "
                        "each candidate has actually run, so it carries no "
                        "risk of dropping a ticket that might be good. 0.90 is "
                        "a sensible value; on goal T0's tier 1 it takes the "
                        "survivors from 32 to 24. 0 disables it.")
    p.add_argument("--prune_floor", type=float, default=0.0,
                   help="Also drop candidates whose upper bound cannot reach "
                        "this ABSOLUTE rate -- for when the aim is a "
                        "high final number rather than merely beating "
                        "Gaussian. Still an upper bound, not the point "
                        "estimate: at five episodes a truly-85%% ticket scores "
                        "3/5 about 14%% of the time, and a rule that dropped "
                        "everything under 4/5 would discard it. The bound "
                        "keeps 3/5 (upper bound 0.888) and drops 2/5 (0.753). "
                        "Stacks with --prune_confidence, which uses the task's "
                        "own baseline instead of a fixed number; the stricter "
                        "of the two wins.")
    p.add_argument("--abort_below_ratio", type=float, default=0.0,
                   help="After tier 1, give up on a task whose "
                        "sd_between / sd_binomial_only is below this and move "
                        "to the next one. THE RATIO IS THE EFFECT SIZE; the "
                        "dispersion z is not -- z is driven by how many "
                        "episodes were run and does not separate the cases "
                        "(goal T0 z=9.85 succeeded, object T4 z=10.25 failed), "
                        "while the ratio does: 1.33 and 2.07 on the two tasks "
                        "that produced a working ticket against 1.07 on the "
                        "one that did not. THAT IS THREE DATA POINTS, so this "
                        "is off by default; 1.15 is the value those three "
                        "suggest. Worth enabling when a task costs 12 h, as "
                        "libero_10 does -- it turns a failed search into a "
                        "2 h answer instead of a 12 h one.")
    p.add_argument("--allow_zero_baseline", action="store_true",
                   help="Search on even when the Gaussian reference scores 0. "
                        "Without it the run aborts, because a zero baseline "
                        "almost always means the env is misconfigured rather "
                        "than the task being hard -- and finding that out "
                        "after 8 hours instead of 30 minutes is the expensive "
                        "version of the mistake.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-search tasks already in the bundle. Off by "
                        "default: a Colab session dies at 24 h and losing "
                        "finished tasks to a restart is the expensive mistake "
                        "this file is arranged around.")
    p.add_argument("--verbose", action="store_true",
                   help="Let eval_task print its per-call banner. Off by "
                        "default: the search makes hundreds of calls and the "
                        "banners bury the tier summaries, which are the only "
                        "lines worth watching.")
    a = p.parse_args()

    if a.init_state_offset + max(a.tiers * a.envs_per_tier,
                                 a.baseline_layouts or 0) > 50:
        print(f"ERROR: tiers x envs_per_tier = "
              f"{a.tiers * a.envs_per_tier} layouts starting at "
              f"{a.init_state_offset} runs past the canonical 50 and would "
              f"wrap into the reportable 0-19. Reduce --tiers, "
              f"--envs_per_tier, or --init_state_offset.", file=sys.stderr)
        return 1

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    from checkpoint_utils import resolve_checkpoint
    # One call, the same one eval_wiltechs_x.main() makes. Copying the patches
    # individually is how this script shipped two separate bugs.
    ev.setup_libero_env(a.control_freq, a.render_gpu, a.stock_init)
    ckpt = resolve_checkpoint(a.checkpoint, for_resume=False)
    policy = ev.load_policy(ckpt, device, a.num_inference_steps,
                            n_action_steps=a.n_action_steps,
                            vision_input_size=a.vision_input_size)
    pre, post = ev.load_processors(ckpt, device, a.dataset_id)
    cams = list(policy.config.cameras_for_vision_state_concat) \
        if hasattr(policy.config, "cameras_for_vision_state_concat") else []
    if not hasattr(getattr(policy, "model", None), "sample_actions"):
        print(f"ERROR: {type(policy).__name__} has no .model.sample_actions, so "
              f"there is nowhere to inject a ticket. Ticket search is "
              f"implemented for the wilro_moe family.", file=sys.stderr)
        return 1
    H = int(policy.config.horizon)
    D = int(policy.config.action_dim)
    # A ticket is only valid for the inference config it was searched under,
    # and a silent mismatch looks exactly like "the method does not work".
    infcfg = ev.inference_config(policy, a.control_freq, a.max_episode_steps,
                                 a.stock_init)
    ev.report_inference_config(
        infcfg, "these must match the eval command you will report with; "
                "eval refuses a ticket whose " + "/".join(ev.MUST_MATCH) +
                " differ")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    eps, _a = 0, a.tickets
    for _t in range(a.tiers):
        eps += _a * a.envs_per_tier
        if _t < a.tiers - 1:
            _a = max(1, _a // 2)
    batches = -(-eps // a.num_envs) + a.envs_per_tier     # + the Gaussian tier
    hours = batches * 12 * ((a.max_episode_steps or 300) / 300) / 60
    print(f"ticket shape ({H}, {D}) = {H * D} dims\n"
          f"{a.tickets} candidates, {a.tiers} tiers x {a.envs_per_tier} layouts "
          f"from id {a.init_state_offset}\n"
          f"{eps} search episodes = ~{batches} batches of {a.num_envs}\n"
          f"BATCHES ARE THE COST, NOT EPISODES: a batch runs until its slowest "
          f"env finishes and most reach the cap. At this project's measured "
          f"12 min/batch at cap 300, that is ~{hours:.1f} h for this task.",
          flush=True)

    rng = np.random.default_rng(a.seed)
    results = {}
    from lerobot.envs.libero import LiberoEnv, _get_suite
    for suite_name in a.suites:
        suite = _get_suite(suite_name)
        n_tasks = getattr(suite, "n_tasks", None) or len(suite.tasks)
        ids = a.task_ids if a.task_ids is not None else list(range(n_tasks))
        for tid in ids:
            t0 = time.time()
            if not a.overwrite:
                try:
                    done, _ = tb.load_bundle(out)
                    if tb.key(suite_name, tid) in done:
                        print(f"\n=== {suite_name} task {tid}: already in the "
                              f"bundle, skipping (--overwrite to redo) ===",
                              flush=True)
                        continue
                except FileNotFoundError:
                    pass
            print(f"\n=== {suite_name} task {tid} ===", flush=True)
            # Built ONCE per task and handed to every eval_task call.
            # Construction takes seconds per env and the search makes hundreds
            # of calls, so building them per call would dominate the run.
            n_par = a.num_envs
            print(f"  building {n_par} envs...", end="", flush=True)
            _tb = time.time()
            envs = [LiberoEnv(task_suite=suite, task_id=tid,
                              task_suite_name=suite_name,
                              obs_type="pixels_agent_pos",
                              init_states=True, episode_index=0)
                    for _ in range(n_par)]
            print(f" {time.time() - _tb:.0f}s", flush=True)
            # Candidates are fixed up front so every tier scores the SAME
            # tickets, and the baseline (all-Gaussian) is not among them: it is
            # measured separately, at the same layouts, as ticket id -1.
            # Within-task progress, rewritten after EVERY tier. A task is
            # hours; losing it at tier 3 to a 24 h cutoff is what this guards
            # against. `cands` is saved too, so a resumed run scores the SAME
            # candidates -- otherwise the accumulated wins/runs would describe
            # tickets that no longer exist.
            prog = out / f"_progress_{suite_name}_t{tid}.npz"
            if prog.exists() and not a.overwrite:
                # Progress from a differently-configured run is WORSE than no
                # progress: it carries that run's counts and its base_done
                # flag, so the Gaussian reference is never re-measured and the
                # zero-baseline guard fires on stale numbers without a single
                # new rollout. That is exactly what happened on the first run
                # after the patch_lerobot_libero and n_action_steps fixes --
                # the env was finally right and the abort still read 0/50 from
                # the broken run's file.
                _z = np.load(prog, allow_pickle=True)
                if "infcfg" not in _z.files:
                    _bad = {"(unstamped)": ("pre-dates the config stamp", "")}
                else:
                    _was = json.loads(str(_z["infcfg"]))
                    _bad = {k: (_was.get(k), infcfg[k]) for k in ev.MUST_MATCH
                            if _was.get(k) != infcfg[k]}
                if _bad:
                    print(f"  DISCARDING {prog.name}: written under "
                          + ", ".join(f"{k}={w}{f' (now {n_})' if n_ != '' else ''}"
                                      for k, (w, n_) in _bad.items())
                          + " -- starting this task fresh.", flush=True)
                    prog.unlink()
            if prog.exists() and not a.overwrite:
                z = np.load(prog)
                cands, wins, runs = z["cands"], z["wins"], z["runs"]
                alive, first_tier = [int(x) for x in z["alive"]], int(z["next_tier"])
                base_w0, base_r0 = float(z["base_w"]), float(z["base_r"])
                # Kept in the progress file because a run killed BETWEEN the
                # last tier's save and the bundle write resumes with an empty
                # tier loop: the ticket is recovered correctly but nothing
                # would re-read the task string.
                desc0 = str(z["desc"]) if "desc" in z.files else None
                # Layouts already finished INSIDE next_tier, and whether the
                # Gaussian reference has run. Tier 1 is 3.7 of a task's 6.1
                # hours, so checkpointing only between tiers leaves 3.7 hours
                # exposed to a Colab cutoff; per layout it is about 40 min.
                start_k0 = int(z["done_k"]) if "done_k" in z.files else 0
                base_done0 = bool(z["base_done"]) if "base_done" in z.files else False
                print(f"  resuming from {prog.name}: tier {first_tier + 1}, "
                      f"{len(alive)} candidates still alive", flush=True)
            else:
                cands = rng.standard_normal((a.tickets, H, D)).astype(np.float32)
                alive, first_tier = list(range(a.tickets)), 0
                wins = np.zeros(a.tickets); runs = np.zeros(a.tickets)
                base_w0 = base_r0 = 0.0
                desc0 = None
                start_k0, base_done0 = 0, False

            base_done = [base_done0]

            def _save(tier, done_k, alive_now):
                np.savez(prog, cands=cands, wins=wins, runs=runs,
                         alive=np.array(alive_now, dtype=np.int64),
                         next_tier=tier, done_k=done_k,
                         base_w=base_w[0], base_r=base_r[0],
                         base_done=base_done[0], desc=np.array(desc or ""),
                         infcfg=np.array(json.dumps(infcfg)))

            def score(idx_list, tier, start_k=0):
                """One batch = n_par CANDIDATES on ONE layout.

                Wall clock is set by the number of BATCHES, not episodes: a
                batch runs until its slowest env finishes, and at a 70%
                success rate 97% of ten-env batches reach the cap. Scoring one
                candidate per batch therefore burns a whole batch on five
                episodes. A per-env ticket puts a different candidate in every
                env against the same layout -- which is also the fairest
                comparison available: identical problem, identical seed, only
                the ticket differs.
                """
                nonlocal desc
                for k in range(start_k, a.envs_per_tier):
                    layout = a.init_state_offset + tier * a.envs_per_tier + k
                    for g0 in range(0, len(idx_list), n_par):
                        grp = idx_list[g0:g0 + n_par]
                        tk = torch.from_numpy(
                            np.stack([cands[i] for i in grp])).to(device)
                        policy.model._noise_ticket = tk
                        sink = (contextlib.nullcontext() if a.verbose
                                else contextlib.redirect_stdout(io.StringIO()))
                        with sink:
                            _, _, _, _, desc, ep_ok = ev.eval_task(
                                policy, pre, post, suite, suite_name, tid,
                                len(grp), len(grp), device,
                                a.max_episode_steps, a.seed, cams,
                                envs=envs[:len(grp)],
                                init_state_offset=layout, init_state_stride=0)
                        for j, i in enumerate(grp):
                            wins[i] += ep_ok[j]; runs[i] += 1
                    # `alive` is not touched until score() returns, so saving
                    # it here records the tier's INPUT list -- which is what a
                    # mid-tier resume has to continue from.
                    _save(tier, k + 1, idx_list)
                return desc

            def score_baseline(tier):
                """The Gaussian reference, from the first search layout on."""
                if base_done[0]:
                    return
                policy.model._noise_ticket = None
                n_lay = a.baseline_layouts or a.envs_per_tier
                for k in range(n_lay):
                    layout = a.init_state_offset + tier * a.envs_per_tier + k
                    sink = (contextlib.nullcontext() if a.verbose
                            else contextlib.redirect_stdout(io.StringIO()))
                    with sink:
                        n_ok, n_ep, _, _, _, _ = ev.eval_task(
                            policy, pre, post, suite, suite_name, tid,
                            n_par, n_par, device, a.max_episode_steps,
                            a.seed, cams, envs=envs,
                            init_state_offset=layout, init_state_stride=0)
                    base_w[0] += n_ok; base_r[0] += n_ep
                base_done[0] = True
                _save(tier, a.envs_per_tier, alive)

            base_w, base_r = [base_w0], [base_r0]
            desc, abandoned = desc0, False
            for tier in range(first_tier, a.tiers):
                print(f"  tier {tier + 1}/{a.tiers}: {len(alive)} candidates "
                      f"on {a.envs_per_tier} layouts "
                      f"(ids {a.init_state_offset + tier * a.envs_per_tier}"
                      f"..{a.init_state_offset + (tier + 1) * a.envs_per_tier - 1})",
                      flush=True)
                if tier == 0:
                    # BEFORE the candidates, not after. This is the only cheap
                    # check that the env is set up the way the reported evals
                    # set it up, and it has to happen before hours are spent.
                    score_baseline(tier)
                    if base_w[0] == 0 and not a.allow_zero_baseline:
                        for _e in envs:
                            try:
                                _e.close()
                            except Exception:
                                pass
                        raise SystemExit(
                            f"\nGaussian baseline scored 0/{base_r[0]:.0f} on "
                            f"layouts {a.init_state_offset}-"
                            f"{a.init_state_offset + a.envs_per_tier - 1} of "
                            f"{suite_name} task {tid}.\n"
                            f"This policy is not at 0% on this task, so the "
                            f"env is almost certainly not the one the evals "
                            f"use. Check that patch_lerobot_libero and "
                            f"--control_freq {a.control_freq} match the eval "
                            f"command, and that --max_episode_steps "
                            f"{a.max_episode_steps} is not cutting successes "
                            f"off.\nSearching for a ticket on a task the "
                            f"policy cannot do at all learns nothing. "
                            f"--allow_zero_baseline to override.")
                desc = score(alive, tier, start_k0 if tier == first_tier else 0)
                rate = np.where(runs > 0, wins / np.maximum(runs, 1), -1.0)
                alive = sorted(alive, key=lambda i: -rate[i])
                if tier < a.tiers - 1:
                    n_before = len(alive)
                    alive = alive[:max(1, int(len(alive) * a.keep_frac))]
                    floor = 0.0
                    if a.prune_confidence > 0 and base_r[0] > 0:
                        floor = base_w[0] / base_r[0]
                    floor = max(floor, a.prune_floor)
                    if floor > 0:
                        conf = a.prune_confidence or 0.90
                        br = floor
                        kept = [i for i in alive
                                if binom_ub(int(wins[i]), int(runs[i]),
                                            conf) >= floor]
                        if kept:                    # never prune to nothing
                            dropped = len(alive) - len(kept)
                            alive = kept
                            if dropped:
                                print(f"    pruned {dropped} more whose "
                                      f"{a.prune_confidence:.0%} upper bound is "
                                      f"below the {br:.0%} baseline -- they "
                                      f"cannot be worth using", flush=True)
                    print(f"    {n_before} -> {len(alive)} carried forward",
                          flush=True)
                top = alive[0]
                print(f"    best so far: ticket {top} "
                      f"{wins[top]:.0f}/{runs[top]:.0f} = "
                      f"{100 * rate[top]:.0f}%   "
                      f"(baseline Gaussian {base_w[0]:.0f}/{base_r[0]:.0f})",
                      flush=True)
                _save(tier + 1, 0, alive)
                # The full distribution, not just the winner. After tier 1 the
                # SPREAD across candidates is what says whether this policy is
                # steerable at all, and the winner of a 5-episode tier is
                # mostly luck: 128 identical tickets at p=0.7 throw ~21 perfect
                # scores by chance.
                disp = dispersion(wins, runs)
                if disp and "dispersion" in disp:
                    verdict = ("TICKETS DIFFER -- worth continuing"
                               if disp["z"] >= 3 else
                               "suggestive, needs more layouts per ticket"
                               if disp["z"] >= 1.5 else
                               "NO real spread: the observed range is what "
                               "binomial noise alone produces. This policy is "
                               "not steerable by the initial noise")
                    # NORMALISED TO 5 LAYOUTS PER TIER. sd_binomial is
                    # sqrt(pq/M), so the raw ratio is proportional to sqrt(M)
                    # and a threshold calibrated at M=5 is 1.58x too strict at
                    # M=2. goal T0 -- the only task whose ticket has held up
                    # out of sample -- reads 1.33 at M=5 and would read 0.84
                    # at M=2, i.e. a gate at 1.15 would have killed it.
                    raw = (disp["sd_between"] / disp["sd_binomial_only"]
                           if disp["sd_binomial_only"] > 0 else float("inf"))
                    ratio = raw * (5.0 / max(a.envs_per_tier, 1)) ** 0.5
                    print(f"    dispersion {disp['dispersion']:.2f} over "
                          f"{disp['n']} candidates (1.00 = interchangeable)  "
                          f"z={disp['z']:+.1f}  "
                          f"sd_between {disp['sd_between']:.3f} vs "
                          f"binomial {disp['sd_binomial_only']:.3f}  "
                          f"ratio {ratio:.2f}"
                          + (f" (raw {raw:.2f} at M={a.envs_per_tier}, "
                             f"normalised to M=5)" if a.envs_per_tier != 5
                             else "") + "\n"
                          f"    -> {verdict}", flush=True)
                    _br = base_w[0] / max(base_r[0], 1)
                    if (tier == 0 and a.abort_below_ratio > 0 and _br >= 0.95):
                        print(f"    gate SKIPPED: the Gaussian baseline is "
                              f"{_br:.0%}, so there is no room for "
                              f"between-ticket variance to show and the ratio "
                              f"cannot discriminate. Searching on -- but a task "
                              f"this close to ceiling has no headroom to win "
                              f"either, and should probably not be in "
                              f"--task_ids.", flush=True)
                    elif (tier == 0 and a.abort_below_ratio > 0
                            and ratio < a.abort_below_ratio):
                        print(f"    ABANDONING this task: effect size "
                              f"{ratio:.2f} < --abort_below_ratio "
                              f"{a.abort_below_ratio:g}. Almost all of the "
                              f"spread between candidates is binomial noise, "
                              f"which is what a failed search looks like at "
                              f"tier 1 (object T4 read 1.07 and cost 12 h to "
                              f"confirm). No ticket is banked.", flush=True)
                        abandoned = True
                        break
                (out / f"scores_{suite_name}_t{tid}.json").write_text(json.dumps(
                    {"tier": tier + 1, "task": desc,
                     "baseline": f"{base_w[0]:.0f}/{base_r[0]:.0f}",
                     "dispersion_test": disp,
                     "candidates": {str(i): [int(wins[i]), int(runs[i])]
                                    for i in range(a.tickets) if runs[i] > 0}},
                    indent=1))

            if abandoned:
                # No ticket, and the progress file is kept: the candidates are
                # still on disk if a later run wants to resume with a lower
                # threshold or more tiers.
                results[f"{suite_name}_t{tid}"] = {
                    "task": desc, "abandoned_at_tier": 1,
                    "reason": f"effect size below --abort_below_ratio "
                              f"{a.abort_below_ratio:g}",
                    "baseline_search": f"{base_w[0]:.0f}/{base_r[0]:.0f}",
                    "minutes": round((time.time() - t0) / 60, 1)}
                (out / "search_summary.json").write_text(json.dumps(results, indent=1))
                for e_ in (envs or []):
                    try:
                        e_.close()
                    except Exception:
                        pass
                policy.model._noise_ticket = None
                continue

            best = alive[0]
            f = out / f"{suite_name}_t{tid}_ticket.npy"
            np.save(f, cands[best])
            # DOES THE WINNER ACTUALLY BEAT GAUSSIAN? Nothing checked this
            # before, and `best = alive[0]` banks a ticket unconditionally --
            # so a task where the search simply fails still produces one, and
            # using it is WORSE than not: a ticket that only matches the
            # baseline still removes the per-chunk re-draw, which this
            # benchmark prices at 25 points. object T4 ketchup is the case
            # that surfaced it: 32 survivors averaged 26.9% against a 72%
            # baseline and the best was 8/10, which a baseline-quality ticket
            # produces 44% of the time.
            base_rate = base_w[0] / max(base_r[0], 1)
            win_rate = wins[best] / max(runs[best], 1)
            p_raw = binom_sf(int(wins[best]), int(runs[best]), base_rate)
            # THREE-VALUED, because "not significant" and "not better" are
            # different and only one of them is a reason to discard the ticket.
            # 15 episodes cannot detect an improvement over a 90% baseline, so
            # a two-valued test would mark object T9's 15/15 weak -- while its
            # REPORTABLE baseline is 70% and the headroom is real. The search
            # baseline and the eval headroom are measured on different layouts
            # and routinely disagree by 20 points.
            if win_rate <= base_rate:
                beats, verdict = False, ("NOT BETTER than Gaussian -- eval will "
                                         "fall back to Gaussian for this task")
            elif p_raw < 0.05:
                beats, verdict = True, "BEATS the baseline"
            else:
                beats, verdict = None, ("better but not significantly; 15 "
                                        "episodes cannot resolve this against "
                                        "a high baseline. Banked, and eval "
                                        "will use it and say so -- the 20-episode "
                                        "held-out run is the real test")
            print(f"  winner {win_rate:.0%} vs Gaussian {base_rate:.0%}   "
                  f"one-sided p={p_raw:.4f} (uncorrected; {a.tickets} candidates "
                  f"searched)  ->  {verdict}", flush=True)
            meta = {
                "task": desc, "ticket_index": int(best),
                "beats_baseline": beats,
                "baseline_rate": round(base_rate, 4),
                "p_vs_baseline": round(p_raw, 5),
                "search_success": f"{wins[best]:.0f}/{runs[best]:.0f}",
                "search_rate": float(wins[best] / max(runs[best], 1)),
                "baseline_search": f"{base_w[0]:.0f}/{base_r[0]:.0f}",
                "tickets": a.tickets, "tiers": a.tiers,
                "envs_per_tier": a.envs_per_tier,
                "init_state_offset": a.init_state_offset,
                **infcfg,
                "checkpoint": str(a.checkpoint), "horizon": H, "action_dim": D,
            }
            # Banked the moment the task finishes, before the next one starts.
            bf = tb.save_ticket(out, suite_name, tid, cands[best], meta)
            # KEPT, not deleted. The bundle stores one ticket per task, but
            # the runner-up VECTORS exist only here -- scores_*.json records
            # every candidate's wins/runs and none of the noise. Discarding
            # them forecloses top-k sampling, which the paper (D.6.2) shows
            # performs as well as a single ticket while restoring
            # stochasticity. That matters more here than in the paper: one
            # fixed ticket makes this policy fully deterministic, and the
            # per-chunk re-draw it removes is worth 25 points by this
            # project's own measurement. 64 x 64 x 7 float32 is 115 KB.
            prog.replace(prog.with_name(prog.name.replace('_progress_', '_done_')))
            results[f"{suite_name}_t{tid}"] = {
                "file": str(f), "bundle": str(bf), "task": desc,
                "ticket_index": int(best),
                "search_success": f"{wins[best]:.0f}/{runs[best]:.0f}",
                "search_rate": float(wins[best] / max(runs[best], 1)),
                "baseline_search_rate": float(base_w[0] / max(base_r[0], 1)),
                "baseline_search": f"{base_w[0]:.0f}/{base_r[0]:.0f}",
                "tickets": a.tickets, "tiers": a.tiers,
                "envs_per_tier": a.envs_per_tier,
                "init_state_offset": a.init_state_offset,
                "minutes": round((time.time() - t0) / 60, 1),
            }
            (out / "search_summary.json").write_text(json.dumps(results, indent=1))
            print(f"  -> {f}  search {wins[best]:.0f}/{runs[best]:.0f} vs "
                  f"Gaussian {base_w[0]:.0f}/{base_r[0]:.0f}  "
                  f"({(time.time() - t0) / 60:.0f} min)", flush=True)
            policy.model._noise_ticket = None
            for e_ in (envs or []):
                try:
                    e_.close()
                except Exception:
                    pass

    print(f"\nbundle: {out / tb.BUNDLE}   (+ {tb.META})")
    print("Upload both next to the checkpoint so eval can find them by task id:")
    print(f"  huggingface-cli upload <repo> {out / tb.BUNDLE} {tb.BUNDLE}")
    print(f"  huggingface-cli upload <repo> {out / tb.META} {tb.META}")
    print("The search rate is NOT the result -- it is the number the ticket was "
          "selected on, and selecting on it is what makes it optimistic. Report "
          "the ticket with eval_wiltechs_x.py at --init_state_offset 0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
