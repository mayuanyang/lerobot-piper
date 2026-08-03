"""Does env._init_state_id actually select the layout?

lerobot's LiberoEnv.reset() does:

    set_init_state(_init_states[_init_state_id])   # write MuJoCo state
    _env.reset()                                   # robosuite _reset_internal()

LIBERO's own eval loop does the opposite order (reset first, then set state).
robosuite's reset re-samples the placement initializer, so the ordering above
may be discarding the init state entirely -- in which case every layout is a
fresh random draw and `_init_state_id` is decorative. Everything that sweeps
init states (train_rft.py, train_wiltechs_vla_rl.py, kv_grounding_probe.py)
depends on this being false.

Three collection modes, same env:

  vary   _init_state_id = i, then reset()          <- what lerobot does today
  fixed  _init_state_id = 0, then reset() x N      <- isolates non-init randomness
  after  reset(), then set_init_state(states[i])   <- canonical LIBERO order

Read it like this:

  spread(vary) ~= spread(fixed)          -> the id does nothing. Layout comes
                                            from the placement sampler.
  spread(vary) >> spread(fixed)          -> the id works; layouts are the 50
                                            canonical ones and eval sees them.
  match(vary[i], after[i]) far apart     -> today's path is not honouring the
                                            requested state (same conclusion,
                                            measured directly per-state).

Usage:

    python src/libero_init_state_check.py --suite libero_spatial --task_id 3
    python src/libero_init_state_check.py --suite libero_spatial --task_id 3 --sheet

--sheet writes a contact sheet per mode so you can eyeball them against your
eval screenshot.
"""

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task_id", type=int, required=True)
    ap.add_argument("--n", type=int, default=12, help="layouts per mode")
    ap.add_argument("--camera", default="agentview")
    ap.add_argument("--sheet", action="store_true",
                    help="save a contact sheet png per mode")
    args = ap.parse_args()

    import numpy as np
    from PIL import Image
    from lerobot.envs.libero import LiberoEnv, _get_suite, get_libero_dummy_action

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rl_staged_reward import get_sim_env

    suite = _get_suite(args.suite)
    env = LiberoEnv(task_suite=suite, task_id=args.task_id, task_suite_name=args.suite,
                    obs_type="pixels_agent_pos", init_states=True, episode_index=0)
    print(f"task {args.task_id}: {getattr(env, 'task_description', '')!r}")

    sim_env = get_sim_env(env)
    if sim_env is None:
        raise RuntimeError("get_sim_env() could not find the robosuite problem env.")

    states = getattr(env, "_init_states", None)
    n_init = 0 if states is None else len(states)
    print(f"init states on file: {n_init}")
    if not n_init:
        raise RuntimeError("no _init_states -- nothing to test.")
    n = min(args.n, n_init)

    def positions():
        bid = getattr(sim_env, "obj_body_id", None) or {}
        return {k: np.array(sim_env.sim.data.body_xpos[v]) for k, v in bid.items()}

    def frame(obs):
        f = np.asarray(obs["pixels"][args.camera])
        if f.dtype != np.uint8:
            f = (f.clip(0, 1) * 255).astype(np.uint8) if f.max() <= 1.0 else f.astype(np.uint8)
        return Image.fromarray(f).convert("RGB")

    def settle():
        """Same post-reset settling LiberoEnv.reset() applies, so 'after' is
        compared on equal footing rather than mid-fall. Returns the last raw
        obs, which is what _format_raw_obs consumes."""
        raw = None
        for _ in range(env.num_steps_wait):
            raw, _, _, _ = env._env.step(get_libero_dummy_action())
        return raw

    results = {}   # mode -> (list[dict xyz], list[Image])

    # vary: today's path -----------------------------------------------------
    pos, imgs = [], []
    for i in range(n):
        env._init_state_id = i
        obs, _ = env.reset()
        pos.append(positions())
        imgs.append(frame(obs))
    results["vary"] = (pos, imgs)

    # fixed: same id every time ---------------------------------------------
    pos, imgs = [], []
    for _ in range(n):
        env._init_state_id = 0
        obs, _ = env.reset()
        pos.append(positions())
        imgs.append(frame(obs))
    results["fixed"] = (pos, imgs)

    # after: canonical LIBERO order -----------------------------------------
    pos, imgs = [], []
    for i in range(n):
        env._env.reset()
        env._env.set_init_state(states[i])
        raw = settle()
        pos.append(positions())
        imgs.append(frame(env._format_raw_obs(raw)) if raw is not None else None)
    results["after"] = (pos, imgs)

    # report -----------------------------------------------------------------
    names = sorted(results["vary"][0][0])
    print(f"\nper-object xyz spread over {n} layouts (metres, max over x/y/z)")
    print(f"  {'object':<34} {'vary':>10} {'fixed':>10} {'after':>10}")
    for name in names:
        row = []
        for mode in ("vary", "fixed", "after"):
            arr = np.stack([p[name] for p in results[mode][0]])
            row.append(arr.std(0).max())
        print(f"  {name:<34} {row[0]:>10.4f} {row[1]:>10.4f} {row[2]:>10.4f}")

    def agg(mode):
        return max(np.stack([p[name] for p in results[mode][0]]).std(0).max()
                   for name in names)

    v, f_, a = agg("vary"), agg("fixed"), agg("after")
    print(f"\n  overall  vary={v:.4f}  fixed={f_:.4f}  after={a:.4f}")

    # per-state agreement: does today's path land where the state says?
    d = [max(np.linalg.norm(results["vary"][0][i][k] - results["after"][0][i][k])
             for k in names) for i in range(n)]
    print(f"  vary[i] vs after[i] displacement: mean={np.mean(d):.4f} "
          f"max={np.max(d):.4f} m")

    print("\nverdict:")
    if v <= f_ * 1.5:
        print("  _init_state_id does NOT select the layout -- spread with varying ids")
        print("  is no larger than with a fixed id. Layouts come from the placement")
        print("  sampler, so every reset is a fresh draw and no probe run can")
        print("  reproduce a specific eval scene.")
        if a > f_ * 1.5:
            print("  Reversing the order (reset -> set_init_state) DOES select it;")
            print("  that is the fix.")
        else:
            print("  Reversing the order does not help either -- the 50 states may")
            print("  genuinely be near-identical for this task. Check 'after' spread")
            print("  against the object sizes before concluding.")
    else:
        print("  _init_state_id works: varying it moves objects well beyond the")
        print("  per-reset noise floor. Then the eval scene IS among the 50 and the")
        print("  mismatch is elsewhere (task_id? suite? camera?).")

    if args.sheet:
        import math
        for mode, (_, imgs) in results.items():
            imgs = [im for im in imgs if im is not None]
            if not imgs:
                continue
            cols = min(6, len(imgs))
            rows = math.ceil(len(imgs) / cols)
            w, h = imgs[0].size
            sheet = Image.new("RGB", (cols * w, rows * h), "black")
            for j, im in enumerate(imgs):
                sheet.paste(im, ((j % cols) * w, (j // cols) * h))
            path = f"initstates_{args.suite}_t{args.task_id}_{mode}.png"
            sheet.save(path)
            print(f"  saved {path}  ({len(imgs)} layouts, row-major id order)")

    env.close()


if __name__ == "__main__":
    main()
