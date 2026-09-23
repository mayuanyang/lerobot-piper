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

import eval_wiltechs_x as ev


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
    p.add_argument("--n_action_steps", type=int, default=None)
    p.add_argument("--vision_input_size", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--control_freq", type=int, default=10,
                   help="MUST be 10: the LIBERO datasets are 10 Hz and the "
                        "stock env is 20, so a search at 20 optimises a ticket "
                        "for a policy that is not the one being reported.")
    p.add_argument("--render_gpu", type=int, default=0)
    p.add_argument("--verbose", action="store_true",
                   help="Let eval_task print its per-call banner. Off by "
                        "default: the search makes hundreds of calls and the "
                        "banners bury the tier summaries, which are the only "
                        "lines worth watching.")
    a = p.parse_args()

    if a.init_state_offset + a.tiers * a.envs_per_tier > 50:
        print(f"ERROR: tiers x envs_per_tier = "
              f"{a.tiers * a.envs_per_tier} layouts starting at "
              f"{a.init_state_offset} runs past the canonical 50 and would "
              f"wrap into the reportable 0-19. Reduce --tiers, "
              f"--envs_per_tier, or --init_state_offset.", file=sys.stderr)
        return 1

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    from checkpoint_utils import resolve_checkpoint
    # Before any env is built, and with the same value the evals use: the
    # dataset is 10 Hz and the stock LiberoEnv is 20.
    ev.patch_control_freq(a.control_freq, a.render_gpu)
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
            cands = rng.standard_normal((a.tickets, H, D)).astype(np.float32)
            alive = list(range(a.tickets))
            wins = np.zeros(a.tickets); runs = np.zeros(a.tickets)

            def score(idx_list, tier):
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
                desc = None
                for k in range(a.envs_per_tier):
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
                return desc

            def score_baseline(tier):
                """The Gaussian reference, on the SAME layouts as this tier."""
                policy.model._noise_ticket = None
                for k in range(a.envs_per_tier):
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

            base_w, base_r = [0.0], [0.0]
            for tier in range(a.tiers):
                print(f"  tier {tier + 1}/{a.tiers}: {len(alive)} candidates "
                      f"on {a.envs_per_tier} layouts "
                      f"(ids {a.init_state_offset + tier * a.envs_per_tier}"
                      f"..{a.init_state_offset + (tier + 1) * a.envs_per_tier - 1})",
                      flush=True)
                desc = score(alive, tier)
                if tier == 0:
                    score_baseline(tier)
                rate = np.where(runs > 0, wins / np.maximum(runs, 1), -1.0)
                alive = sorted(alive, key=lambda i: -rate[i])
                if tier < a.tiers - 1:
                    alive = alive[:max(1, len(alive) // 2)]
                top = alive[0]
                print(f"    best so far: ticket {top} "
                      f"{wins[top]:.0f}/{runs[top]:.0f} = "
                      f"{100 * rate[top]:.0f}%   "
                      f"(baseline Gaussian {base_w[0]:.0f}/{base_r[0]:.0f})",
                      flush=True)

            best = alive[0]
            f = out / f"{suite_name}_t{tid}_ticket.npy"
            np.save(f, cands[best])
            results[f"{suite_name}_t{tid}"] = {
                "file": str(f), "task": desc, "ticket_index": int(best),
                "search_success": f"{wins[best]:.0f}/{runs[best]:.0f}",
                "search_rate": float(wins[best] / max(runs[best], 1)),
                "baseline_search_rate": float(base_w[0] / max(base_r[0], 1)),
                "baseline_search": f"{base_w[0]:.0f}/{base_r[0]:.0f}",
                "tickets": a.tickets, "tiers": a.tiers,
                "envs_per_tier": a.envs_per_tier,
                "init_state_offset": a.init_state_offset,
                "minutes": round((time.time() - t0) / 60, 1),
            }
            print(f"  -> {f}  search {wins[best]:.0f}/{runs[best]:.0f} vs "
                  f"Gaussian {base_w[0]:.0f}/{base_r[0]:.0f}  "
                  f"({(time.time() - t0) / 60:.0f} min)", flush=True)
            policy.model._noise_ticket = None
            for e_ in (envs or []):
                try:
                    e_.close()
                except Exception:
                    pass

    (out / "search_summary.json").write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out / 'search_summary.json'}")
    print("The search rate is NOT the result -- it is the number the ticket was "
          "selected on, and selecting on it is what makes it optimistic. Report "
          "the ticket with eval_wiltechs_x.py at --init_state_offset 0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
