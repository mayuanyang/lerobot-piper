"""Per-init-state geometry for a libero_spatial task, joined against failures.

The "between" task succeeds 46/50. The open question is what the four failing
layouts have in common. This dumps, for every initial state, the quantities the
referring expression actually depends on:

  t     where the target projects onto the anchor0 -> anchor1 segment
        (0 = at anchor0, 1 = at anchor1, outside [0,1] = not between them)
  perp  perpendicular distance from that segment -- how far the target sits
        OFF the line the word "between" describes
  ratio the selector's margin: distance(distractor, plate) / distance(target,
        plate). Below 1.0 means the plate-anchored selector picks the WRONG bowl
        in that layout, i.e. the instruction is not merely hard but incorrect.

Then it splits the table by the episodes that failed and prints both groups'
means, so "the failures are the layouts where the target is off the line" is
tested rather than eyeballed.

Positions come from the canonical reset order (reset -> set_init_state), which
src/libero_init_state_check.py showed is exactly deterministic, so these are the
true body positions rather than qpos anchors with a per-object frame offset.

Usage:

    python src/libero_layout_geometry.py --task_id 0 \\
        --target akita_black_bowl_1 --distractor akita_black_bowl_2 \\
        --anchors glazed_rim_porcelain_ramekin_1 plate_1 \\
        --selector_anchor plate_1 \\
        --failed_episodes 13 14 26 27

--failed_episodes takes lerobot-eval's episode numbers and converts them with
the same rule the eval used (init_state = (start_seed + episode) % n_states),
so paste them straight from the metrics dict. Use --failed_states to pass
initial-state ids directly instead.
"""

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task_id", type=int, required=True)
    ap.add_argument("--target", default="akita_black_bowl_1")
    ap.add_argument("--distractor", default="akita_black_bowl_2")
    ap.add_argument("--anchors", nargs=2,
                    default=["glazed_rim_porcelain_ramekin_1", "plate_1"],
                    help="the two objects the word 'between' names, in order")
    ap.add_argument("--selector_anchor", default="plate_1",
                    help="the object the rewrite's selector measures from")
    ap.add_argument("--n_states", type=int, default=50)
    ap.add_argument("--failed_episodes", type=int, nargs="*", default=[],
                    help="episode indices from the eval metrics dict")
    ap.add_argument("--failed_states", type=int, nargs="*", default=[],
                    help="initial-state ids, if you already converted them")
    ap.add_argument("--start_seed", type=int, default=1000,
                    help="lerobot-eval's cfg.seed; only used to map episodes "
                         "to states. Must match the eval run.")
    args = ap.parse_args()

    import numpy as np
    from lerobot.envs.libero import LiberoEnv, _get_suite

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from libero_env_fixed import patch_lerobot_libero
    from rl_staged_reward import get_sim_env

    patch_lerobot_libero(enable=True, init_from_seed=False)

    env = LiberoEnv(task_suite=_get_suite(args.suite), task_id=args.task_id,
                    task_suite_name=args.suite, obs_type="pixels_agent_pos",
                    init_states=True, episode_index=0)
    print(f"task {args.task_id}: {getattr(env, 'task_description', '')!r}")
    sim_env = get_sim_env(env)
    if sim_env is None:
        raise RuntimeError("get_sim_env() could not reach the robosuite env.")

    states = getattr(env, "_init_states", None)
    n_init = 0 if states is None else len(states)
    n = min(args.n_states, n_init)
    print(f"init states: {n_init}, reading {n}")

    def positions():
        bid = getattr(sim_env, "obj_body_id", None) or {}
        return {k: np.array(sim_env.sim.data.body_xpos[v]) for k, v in bid.items()}

    known = set(positions())
    for name in (args.target, args.distractor, *args.anchors, args.selector_anchor):
        if name not in known:
            ap.error(f"object {name!r} not in this task. Known: {sorted(known)}")

    failed = set(args.failed_states)
    if args.failed_episodes:
        failed |= {(args.start_seed + e) % n_init for e in args.failed_episodes}
    if failed:
        print(f"failing initial states: {sorted(failed)}")

    rows = []
    for i in range(n):
        env._init_state_id = i
        env.reset()
        p = positions()
        a0, a1 = p[args.anchors[0]][:2], p[args.anchors[1]][:2]
        tgt, dis = p[args.target][:2], p[args.distractor][:2]
        seg = a1 - a0
        seg_len = float(np.linalg.norm(seg))
        t = float(np.dot(tgt - a0, seg) / max(seg_len ** 2, 1e-9))
        perp = float(np.linalg.norm((tgt - a0) - t * seg))
        sel = p[args.selector_anchor][:2]
        d_t = float(np.linalg.norm(tgt - sel))
        d_d = float(np.linalg.norm(dis - sel))
        rows.append({"i": i, "t": t, "perp": perp, "d_t": d_t, "d_d": d_d,
                     "ratio": d_d / max(d_t, 1e-9), "fail": i in failed})
    env.close()

    print(f"\n{'state':>5} {'t':>7} {'perp(m)':>9} {'d_tgt':>7} {'d_dis':>7} "
          f"{'ratio':>7}  {'result':>7}")
    for r in rows:
        print(f"{r['i']:>5} {r['t']:>7.3f} {r['perp']:>9.4f} {r['d_t']:>7.3f} "
              f"{r['d_d']:>7.3f} {r['ratio']:>7.2f}  "
              f"{'FAIL' if r['fail'] else 'ok':>7}")

    def summarise(sel, label):
        if not sel:
            return None
        f = lambda k: np.array([r[k] for r in sel])
        print(f"  {label:<10} n={len(sel):<3} "
              f"t={f('t').mean():.3f}+-{f('t').std():.3f}  "
              f"perp={f('perp').mean():.4f}+-{f('perp').std():.4f}  "
              f"ratio={f('ratio').mean():.2f}+-{f('ratio').std():.2f}")
        return f

    if failed:
        print("\ngroup means:")
        bad = summarise([r for r in rows if r["fail"]], "FAILED")
        good = summarise([r for r in rows if not r["fail"]], "succeeded")
        if bad is not None and good is not None:
            print("\nreading:")
            dp = bad("perp").mean() - good("perp").mean()
            dr = bad("ratio").mean() - good("ratio").mean()
            if abs(dp) > 2 * good("perp").std():
                print(f"  perp differs by {dp:+.4f} m, beyond the successes' own"
                      f" spread -- the failures ARE the off-the-line layouts.")
            else:
                print(f"  perp differs by only {dp:+.4f} m, inside the successes'"
                      f" spread ({good('perp').std():.4f}) -- distance from the"
                      f" line does NOT separate the failures.")
            if bad("ratio").min() < 1.0:
                print(f"  at least one failure has ratio < 1.0 (min "
                      f"{bad('ratio').min():.2f}): the selector points at the"
                      f" WRONG bowl there, so the instruction is incorrect for"
                      f" that layout, not just hard.")
            elif abs(dr) > 2 * good("ratio").std():
                print(f"  selector margin differs by {dr:+.2f} -- the failures"
                      f" are the layouts where the selector is weakest.")
            else:
                print(f"  selector margin differs by only {dr:+.2f}; geometry"
                      f" does not explain these four. Look at the videos for a"
                      f" control-side cause (grasp slip, timeout).")
    else:
        t = np.array([r["t"] for r in rows]); perp = np.array([r["perp"] for r in rows])
        ratio = np.array([r["ratio"] for r in rows])
        print(f"\nover {n} states: t={t.mean():.3f}+-{t.std():.3f}  "
              f"perp={perp.mean():.4f}+-{perp.std():.4f}  "
              f"ratio={ratio.mean():.2f}+-{ratio.std():.2f}  "
              f"(ratio<1 in {int((ratio < 1).sum())} states)")


if __name__ == "__main__":
    main()
