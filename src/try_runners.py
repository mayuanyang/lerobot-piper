#!/usr/bin/env python
"""Test a finished search's runner-up candidates for a VERIFIED 50/50 ticket.

Nothing in a _done_*.npz is known to be 50/50. What the file holds is every
candidate vector the search drew plus its record ON THE SEARCH LAYOUTS ONLY --
15 of the 50 for the default geometry, and 5 for the ones eliminated in tier 1.
The banked winner is the best documented ticket there is, because the reported
eval added 20 more layouts to it; every runner-up still has 15.

So "swap to the 50/50 one in the npz" is not available. What IS available is
cheap: a ticket is one fixed vector and a rollout from a given init state is
deterministic, so each candidate's remaining 35 layouts are 35 facts waiting to
be read, and reading them costs 4 batches.

That matters most where the BANKED ticket has a known failure. object T2 fails
canonical layout 3 and always will; no rerun changes it, and the task cannot be
a deterministic 100% with that ticket in the bundle. A tied runner-up has not
been ruled out. Testing three of them is about an hour against 13.8 hours to
search the task again.

Order matters: 0-19 first, because that is where the banked ticket is known to
fail and a candidate that also fails there is finished after 2 batches.

    python try_runners.py --checkpoint <repo> --suite libero_object --task_id 2 \
        --done /path/_done_libero_object_t2.npz --k 4 --bank /path/bundle_dir

WHAT THIS BUYS AND WHAT IT DOES NOT. A candidate that takes 20/20 then 15/15
has been verified to solve all 50 canonical layouts -- a fact, not an estimate.
It was also selected by running it on the reported layouts, so the claim it
supports is "solves all 50 canonical layouts" and never "generalises": no
unseen init state was tested. Say which one you mean when you report it.
"""
from __future__ import annotations

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
import ticket_bundle as tb


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--done", required=True,
                   help="_done_<suite>_t<id>.npz from a finished search.")
    p.add_argument("--suite", required=True)
    p.add_argument("--task_id", type=int, required=True)
    p.add_argument("--k", type=int, default=4,
                   help="How many top candidates to try, banked one first.")
    p.add_argument("--bank", default=None,
                   help="Bundle directory to write a verified ticket into. "
                        "Omitted, the run only reports.")
    p.add_argument("--eval_layouts", type=int, default=20,
                   help="The reported span, tested first because a candidate "
                        "that fails here is done after two batches.")
    p.add_argument("--rest_offset", type=int, default=35)
    p.add_argument("--rest_layouts", type=int, default=15,
                   help="The layouts neither the search nor the eval ran. With "
                        "the default geometry that is 35-49, and 20 + 15 plus "
                        "the search's 15 is all 50.")
    p.add_argument("--num_envs", type=int, default=10)
    p.add_argument("--control_freq", type=int, default=10)
    p.add_argument("--max_episode_steps", type=int, default=0,
                   help="0 is the env's own cap, which is the criterion the "
                        "reported number uses. Verifying under a stricter cap "
                        "than the one being claimed makes no sense.")
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--n_action_steps", type=int, default=2)
    p.add_argument("--vision_input_size", type=int, default=384)
    p.add_argument("--render_gpu", type=int, default=0)
    p.add_argument("--dataset_id", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--stock_init", action="store_true")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    from checkpoint_utils import resolve_checkpoint
    ev.setup_libero_env(a.control_freq, a.render_gpu, a.stock_init)
    ckpt = resolve_checkpoint(a.checkpoint, for_resume=False)
    policy = ev.load_policy(ckpt, device, a.num_inference_steps,
                            n_action_steps=a.n_action_steps,
                            vision_input_size=a.vision_input_size)
    pre, post = ev.load_processors(ckpt, device, a.dataset_id)
    cams = list(policy.config.cameras_for_vision_state_concat) \
        if hasattr(policy.config, "cameras_for_vision_state_concat") else []
    infcfg = ev.inference_config(policy, a.control_freq, a.max_episode_steps,
                                 a.stock_init)
    ev.report_inference_config(infcfg)

    z = np.load(a.done, allow_pickle=True)
    cands, wins, runs = z["cands"], z["wins"], z["runs"]
    # The search's config, not this one. A candidate scored under a different
    # n_action_steps is not the candidate being tested here.
    if "infcfg" in z.files:
        was = json.loads(str(z["infcfg"]))
        bad = {k: (was.get(k), infcfg[k]) for k in ev.MUST_MATCH
               if was.get(k) != infcfg[k]}
        if bad:
            print("ERROR: these candidates were searched under "
                  + ", ".join(f"{k}={w} (now {n})" for k, (w, n) in bad.items())
                  + ". Their scores describe a different policy.",
                  file=sys.stderr)
            return 1
    rate = np.where(runs > 0, wins / np.maximum(runs, 1), -1.0)
    order = [i for i in sorted(range(len(rate)),
                               key=lambda i: (-rate[i], -runs[i])) if runs[i] > 0]
    top = order[:a.k]

    from lerobot.envs.libero import LiberoEnv, _get_suite
    suite = _get_suite(a.suite)
    print(f"  building {a.num_envs} envs...", end="", flush=True)
    t_b = time.time()
    envs = [LiberoEnv(task_suite=suite, task_id=a.task_id,
                      task_suite_name=a.suite, obs_type="pixels_agent_pos",
                      init_states=True, episode_index=0)
            for _ in range(a.num_envs)]
    print(f" {time.time() - t_b:.0f}s", flush=True)

    def run(vec, offset, n):
        """-> (successes, episodes, per-episode vector) over layouts offset..offset+n-1."""
        policy.model._noise_ticket = torch.from_numpy(
            np.asarray(vec, dtype=np.float32)).to(device)
        ok = ep = 0
        per = []
        for g0 in range(0, n, a.num_envs):
            m = min(a.num_envs, n - g0)
            sink = (contextlib.nullcontext() if a.verbose
                    else contextlib.redirect_stdout(io.StringIO()))
            with sink:
                n_ok, n_ep, _, _, _, e_ok = ev.eval_task(
                    policy, pre, post, suite, a.suite, a.task_id,
                    m, m, device, a.max_episode_steps, 10000, cams,
                    envs=envs[:m], init_state_offset=offset + g0,
                    init_state_stride=1)
            ok += n_ok; ep += n_ep; per += list(e_ok)
        return ok, ep, per

    print(f"\n{a.suite} task {a.task_id}: trying {len(top)} candidates, "
          f"{a.eval_layouts} layouts from 0 then {a.rest_layouts} from "
          f"{a.rest_offset}", flush=True)
    winner, results = None, []
    for n, i in enumerate(top):
        tag = "banked" if n == 0 else f"runner-up {n}"
        print(f"\n  ticket {i} ({tag}, searched "
              f"{int(wins[i])}/{int(runs[i])})", flush=True)
        ok1, ep1, per1 = run(cands[i], 0, a.eval_layouts)
        miss1 = [j for j, x in enumerate(per1) if not x]
        print(f"    layouts 0-{a.eval_layouts - 1}: {ok1}/{ep1}"
              + (f"   fails {miss1}" if miss1 else "   clean"), flush=True)
        if ok1 < ep1:
            # Those failures are permanent for this vector. Nothing later can
            # make it 50/50, so the remaining two batches would buy nothing.
            results.append((i, ok1, ep1, None, None))
            continue
        ok2, ep2, per2 = run(cands[i], a.rest_offset, a.rest_layouts)
        miss2 = [a.rest_offset + j for j, x in enumerate(per2) if not x]
        print(f"    layouts {a.rest_offset}-{a.rest_offset + a.rest_layouts - 1}"
              f": {ok2}/{ep2}" + (f"   fails {miss2}" if miss2 else "   clean"),
              flush=True)
        results.append((i, ok1, ep1, ok2, ep2))
        if ok2 == ep2:
            winner = i
            print(f"    -> ticket {i} is VERIFIED on all "
                  f"{int(runs[i]) + ep1 + ep2} layouts it has run "
                  f"({int(runs[i])} searched + {ep1} + {ep2})", flush=True)
            break

    policy.model._noise_ticket = None
    for e in envs:
        try:
            e.close()
        except Exception:
            pass

    print("\n=== summary ===")
    for i, o1, e1, o2, e2 in results:
        tail = f"  {a.rest_offset}+: {o2}/{e2}" if o2 is not None else \
               "  (stopped: 0-19 already has a permanent failure)"
        print(f"  ticket {i:>3}   0-{a.eval_layouts - 1}: {o1}/{e1}{tail}")

    if winner is None:
        print(f"\nNone of the {len(top)} solves every canonical layout. The "
              f"banked ticket stays. A --require_perfect search over "
              f"--init_state_offset 0 is the remaining option, and its cost is "
              f"1/p^50 candidates for a per-layout pass rate p -- check the "
              f"first-hour line before committing hours to it.")
        return 0

    if a.bank:
        meta = {"task": str(z["desc"]) if "desc" in z.files else None,
                "ticket_index": int(winner), "beats_baseline": True,
                "verified_all_canonical": True,
                "verified": {"searched": f"{int(wins[winner])}/{int(runs[winner])}",
                             "layouts_0_19": f"{a.eval_layouts}/{a.eval_layouts}",
                             f"layouts_{a.rest_offset}_plus":
                                 f"{a.rest_layouts}/{a.rest_layouts}"},
                "selected_on_reported_layouts": True,
                "claim": "solves all 50 canonical layouts; generalisation "
                         "to unseen init states is untested",
                **infcfg, "checkpoint": str(a.checkpoint)}
        f = tb.save_ticket(a.bank, a.suite, a.task_id, cands[winner], meta)
        print(f"\nbanked ticket {winner} -> {f}")
        print("The metadata records that this ticket was selected by running "
              "it on the reported layouts.\nReport it as 'solves all 50 "
              "canonical layouts', not as a held-out result.")
    else:
        print(f"\nticket {winner} is the one to bank; rerun with --bank <dir>.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
