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
    ap.add_argument("--camera", default="image",
                    help="key inside obs['pixels']. LiberoEnv remaps the raw "
                         "robosuite names, so it is 'image' (agentview) / "
                         "'image2' (wrist), NOT 'agentview'.")
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
        pix = obs["pixels"]
        if args.camera not in pix:
            raise KeyError(
                f"--camera {args.camera!r} not in obs['pixels']. "
                f"Available: {sorted(pix)}")
        f = np.asarray(pix[args.camera])
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

    # Ground truth, no sim resets involved --------------------------------
    # _init_states[i] is a flattened MuJoCo state: [time] + qpos + qvel. Each
    # prop is a free body, so its xyz sits at its first joint's qpos address.
    # Reading it answers "how different ARE the 50 canonical layouts?" without
    # touching reset ordering at all -- which is the question the achieved
    # spreads cannot separate from sampler noise.
    def requested_positions(state):
        m = sim_env.sim.model
        out = {}
        for name, bid in (getattr(sim_env, "obj_body_id", None) or {}).items():
            if m.body_jntnum[bid] < 1:
                continue
            adr = m.jnt_qposadr[m.body_jntadr[bid]]
            out[name] = np.asarray(state[1 + adr: 1 + adr + 3], dtype=float)
        return out

    req = [requested_positions(states[i]) for i in range(n_init)]
    req_names = sorted(req[0]) if req and req[0] else []
    if req_names:
        print(f"\nrequested positions read straight out of the {n_init} init "
              f"states (no reset involved):")
        for name in req_names:
            arr = np.stack([r[name] for r in req])
            rng = arr.max(0) - arr.min(0)
            print(f"  {name:<34} std={np.round(arr.std(0), 4)}  "
                  f"range={np.round(rng, 4)}")
    else:
        print("\n!! could not map bodies to qpos addresses; skipping the "
              "requested-position readout.")

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

    # after_fixed: canonical order, same state every time. Control for "after"
    # -- if this is ~0 the canonical order is deterministic and honours the
    # state, so a small spread in "after" means the 50 states really are close.
    # If it is as large as "after", the canonical order is not honouring it
    # either and nothing here selects a layout.
    pos, imgs = [], []
    for _ in range(n):
        env._env.reset()
        env._env.set_init_state(states[0])
        raw = settle()
        pos.append(positions())
        imgs.append(frame(env._format_raw_obs(raw)) if raw is not None else None)
    results["after_fixed"] = (pos, imgs)

    # report -----------------------------------------------------------------
    names = sorted(results["vary"][0][0])
    modes = ("vary", "fixed", "after", "after_fixed")
    print(f"\nper-object xyz spread over {n} layouts (metres, max over x/y/z)")
    print("  " + f"{'object':<34}" + "".join(f"{m:>12}" for m in ("requested",) + modes))
    have_req = bool(req) and bool(req[0])
    for name in names:
        req_s = (np.stack([r[name] for r in req]).std(0).max()
                 if have_req and name in req[0] else float("nan"))
        row = [np.stack([p[name] for p in results[m][0]]).std(0).max() for m in modes]
        print(f"  {name:<34}{req_s:>12.4f}" + "".join(f"{x:>12.4f}" for x in row))

    def agg(mode):
        return max(np.stack([p[name] for p in results[mode][0]]).std(0).max()
                   for name in names)

    v, f_, a, af = (agg("vary"), agg("fixed"), agg("after"), agg("after_fixed"))
    print(f"\n  overall  vary={v:.4f}  fixed={f_:.4f}  after={a:.4f}  "
          f"after_fixed={af:.4f}")

    # Pixel-level spread. "the image changes when I change the id" is true under
    # BOTH hypotheses -- a resampled layout looks just as different as a
    # selected one. What separates them is whether holding the id fixed changes
    # the image by the same amount, so print all three on the same scale.
    def pix_spread(mode):
        ims = [im for im in results[mode][1] if im is not None]
        if len(ims) < 2:
            return float("nan")
        arr = np.stack([np.asarray(im, dtype=np.float32) for im in ims])
        return float(np.abs(arr - arr.mean(0)).mean())

    print("  mean |pixel - mode mean|  " + "  ".join(
        f"{m}={pix_spread(m):.2f}" for m in modes) + "  (0-255)")

    # Tracking error: how far each mode lands from what state i actually asked
    # for. This is the direct measurement -- spread comparisons can be confounded
    # by two random draws happening to have similar variance.
    if have_req:
        # A free joint's qpos[0:3] is the joint anchor, body_xpos is the body
        # origin; they differ by a constant per-object translation. Raw
        # |achieved - requested| therefore measures that offset, not tracking
        # error. after_fixed is exactly deterministic and is the canonical
        # order applied to state 0, so it calibrates the offset directly.
        tracked = [k for k in names if k in req[0]]
        off = {k: results["after_fixed"][0][0][k] - req[0][k] for k in tracked}
        print("\n  frame offset (qpos anchor -> body origin), calibrated on "
              "after_fixed:")
        for k in tracked:
            print(f"    {k:<34} {np.round(off[k], 4)}")

        def track_err(mode, use_state_i=True):
            errs = []
            for i in range(n):
                r = req[i if use_state_i else 0]
                errs.append(max(np.linalg.norm(results[mode][0][i][k] - r[k] - off[k])
                                for k in tracked))
            return float(np.mean(errs)), float(np.max(errs))

        print("\n  offset-corrected distance from the layout state i asks for (m):")
        for mode, same in (("vary", True), ("after", True), ("after_fixed", False)):
            mu, mx = track_err(mode, same)
            print(f"    {mode:<12} mean={mu:.4f}  max={mx:.4f}")
        print("    (after_fixed is 0 by construction -- it defined the offset)")

    print("\nverdict:")
    if pix_spread("fixed") > 1.0 and pix_spread("vary") <= pix_spread("fixed") * 1.5:
        print("  NOTE: holding the id fixed changes the image about as much as")
        print("  varying it. Visible frame-to-frame variation is therefore NOT")
        print("  evidence that the id is selecting anything.")
    if have_req:
        req_spread = max(np.stack([r[k] for r in req]).std(0).max() for k in req[0])
        print(f"  the {n_init} canonical states span {req_spread:.4f} m "
              f"(max per-object std) -- objects jitter, they never swap places.")
        v_err = track_err("vary")[0]
        a_err = track_err("after")[0]
        print(f"  offset-corrected tracking error: vary={v_err:.4f} m, "
              f"after={a_err:.4f} m")
        if a_err < req_spread and v_err > 3 * max(a_err, 1e-6):
            print("  => reset() then set_init_state() reproduces the requested layout;")
            print("     set_init_state() then reset() does not. lerobot's LiberoEnv")
            print("     uses the latter, so the layouts it serves are placement-sampler")
            print("     draws, NOT the canonical states the demos were recorded on.")
            print("     Everything running through LiberoEnv is affected: eval, RL")
            print("     rollouts, and this repo's probes.")
        elif v_err <= a_err:
            print("  => today's ordering tracks the requested layout as well as the")
            print("     canonical one. The ordering is not your problem.")
        if v > req_spread * 2:
            print(f"  Note the served spread ({v:.4f} m) is much LARGER than the")
            print("  canonical one. Eval is drawing layouts the training demos never")
            print("  contained -- a train/eval shift on exactly the geometry this")
            print("  task depends on.")

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
