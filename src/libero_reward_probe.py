"""One-shot probe of the LIBERO env internals needed for a STAGED reward.

Run ONCE on Colab (where `libero` is installed) and paste the output back.
It does not train anything — it resets one libero_10 task, walks the wrapper
chain down to the robosuite sim, and prints the exact attribute / method names
a staged-reward tracker needs:

  - the target object names + their mujoco body ids (for position / lift)
  - the BDDL goal predicates (per-object placement = the "placed" stage)
  - whether `_check_grasp` / `_check_success` exist and their call shape
  - the end-effector site (for the "reached" stage)

Usage on Colab:
    python src/libero_reward_probe.py --suite libero_10 --task_id 0

The reward function (src/rl_staged_reward.py, written after this confirms the
API) keys off whatever names this prints.
"""
from __future__ import annotations

import argparse


def _hr(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def walk_to_sim(top) -> list:
    """Follow .env / ._env / .unwrapped down to the object holding `.sim`,
    printing each layer's type. Returns the chain (outermost -> innermost)."""
    chain = [top]
    seen = {id(top)}
    cur = top
    for _ in range(12):
        nxt = None
        for attr in ("env", "_env", "unwrapped", "sim_env"):
            cand = getattr(cur, attr, None)
            if cand is not None and id(cand) not in seen:
                nxt = cand
                break
        if nxt is None:
            break
        chain.append(nxt)
        seen.add(id(nxt))
        cur = nxt
        if getattr(cur, "sim", None) is not None and hasattr(cur, "_check_success"):
            break
    for i, obj in enumerate(chain):
        has_sim = getattr(obj, "sim", None) is not None
        print(f"  [{i}] {type(obj).__module__}.{type(obj).__name__}"
              f"{'   <-- has .sim + _check_success' if has_sim and hasattr(obj, '_check_success') else ''}")
    return chain


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--task_id", type=int, default=0)
    args = ap.parse_args()

    from lerobot.envs.libero import LiberoEnv, _get_suite

    _hr(f"Building {args.suite} task {args.task_id}")
    suite = _get_suite(args.suite)
    env = LiberoEnv(
        task_suite=suite,
        task_id=args.task_id,
        task_suite_name=args.suite,
        obs_type="pixels_agent_pos",
        init_states=True,
        episode_index=0,
    )
    obs, _ = env.reset(seed=0)
    print("task_description:", getattr(env, "task_description", "?"))
    print("obs pixels cameras:", list(obs["pixels"].keys()))

    _hr("Wrapper chain (LiberoEnv -> ... -> robosuite sim)")
    chain = walk_to_sim(env)
    sim_env = next((o for o in reversed(chain)
                    if getattr(o, "sim", None) is not None), None)
    if sim_env is None:
        print("!! Could not find an object with `.sim` — inspect chain above and "
              "adjust walk_to_sim attr list.")
        return
    print(f"\nsim_env = {type(sim_env).__name__}")

    _hr("Goal predicates (BDDL) — these define the 'placed' stage per object")
    for attr in ("parsed_problem", "problem_info", "goal_state", "obj_of_interest"):
        val = getattr(sim_env, attr, None)
        if val is not None:
            print(f"sim_env.{attr} = {val!r}"[:1500])
    pp = getattr(sim_env, "parsed_problem", None)
    if isinstance(pp, dict):
        print("\nparsed_problem keys:", list(pp.keys()))
        print("goal_state:", pp.get("goal_state"))
        print("objects:", pp.get("objects"))
        print("fixtures:", pp.get("fixtures"))
        print("regions:", list(pp.get("regions", {}).keys())
              if isinstance(pp.get("regions"), dict) else pp.get("regions"))

    _hr("Object registry (names -> body ids / states) for position & lift stage")
    for attr in ("objects_dict", "object_states_dict", "object_sites_dict",
                 "obj_body_id", "objects", "object_site_ids"):
        val = getattr(sim_env, attr, None)
        if val is not None:
            keys = list(val.keys()) if isinstance(val, dict) else val
            print(f"sim_env.{attr}: {keys}"[:600])

    _hr("Success / grasp methods")
    print("has _check_success:", hasattr(sim_env, "_check_success"))
    try:
        print("  _check_success() =", sim_env._check_success())
    except Exception as e:
        print("  _check_success() raised:", repr(e))
    print("has check_success (env-level):", hasattr(env, "check_success"))
    print("has _check_grasp:", hasattr(sim_env, "_check_grasp"))
    if hasattr(sim_env, "_check_grasp"):
        import inspect
        try:
            print("  _check_grasp signature:", inspect.signature(sim_env._check_grasp))
        except (TypeError, ValueError):
            print("  (could not introspect _check_grasp signature)")
    # robosuite grippers / eef site
    for attr in ("robots", "_eef_xpos", "gripper", "robot_model"):
        val = getattr(sim_env, attr, None)
        if val is not None:
            print(f"sim_env.{attr}: {type(val).__name__} -> {val if not hasattr(val, '__len__') or isinstance(val,str) else f'len {len(val)}'}")
    try:
        print("robots[0].robot_model.eef_name:", sim_env.robots[0].robot_model.eef_name)
        print("robots[0].gripper geom groups:",
              list(sim_env.robots[0].gripper.important_geoms.keys()))
    except Exception as e:
        print("  (eef/gripper introspection:", repr(e), ")")

    _hr("Per-predicate evaluation — can we score goal conjuncts individually?")
    # The 'placed' stage needs each goal conjunct evaluated separately. Probe the
    # evaluator LIBERO uses internally.
    for m in ("_eval_predicate", "eval_predicate", "_eval_predicate_fn"):
        print(f"has {m}:", hasattr(sim_env, m))
    if isinstance(pp, dict) and pp.get("goal_state") and hasattr(sim_env, "_eval_predicate"):
        for g in pp["goal_state"]:
            try:
                print(f"  _eval_predicate({g}) =", sim_env._eval_predicate(g))
            except Exception as e:
                print(f"  _eval_predicate({g}) raised:", repr(e))

    _hr("Object world positions (z = lift detection)")
    sim = sim_env.sim
    try:
        names = list(getattr(sim_env, "objects_dict", {}) or
                     getattr(sim_env, "object_states_dict", {}))
        for nm in names:
            bid = None
            for fn in ("body_name2id",):
                try:
                    bid = sim.model.body_name2id(nm + "_main")
                    break
                except Exception:
                    try:
                        bid = sim.model.body_name2id(nm)
                        break
                    except Exception:
                        pass
            if bid is not None:
                print(f"  {nm}: body_id={bid} xpos={sim.data.body_xpos[bid]}")
            else:
                print(f"  {nm}: (body id lookup failed — note the real body name)")
    except Exception as e:
        print("  position probe raised:", repr(e))

    print("\nDONE. Paste this whole output back so the reward fn keys off the real names.")
    env.close()


if __name__ == "__main__":
    main()
