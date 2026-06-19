"""
Validate the SUBPROCESS vec-env (TaskEnvGroup) added to train_wilro_rl.py.

Background: threaded env stepping was proven to silently corrupt frames — the
EGL render context is current-per-thread, so worker-thread render() returned
garbage (reset frames matched, every threaded step diffed by 255 vs sequential).
The fix runs each env in its own PROCESS (own EGL context). This script checks
the fix is correct: a single-process SEQUENTIAL reference vs the subprocess
TaskEnvGroup, same seed / init state / action sequence. MuJoCo is deterministic,
so correct subprocess rendering ⇒ frames are BITWISE identical.

Run on the GPU box:
    python src/verify_parallel_env_render.py \
        --env_task libero_goal --task_id 3 --group_size 8 --n_steps 16
"""

from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse

import numpy as np

from train_wilro_rl import TaskEnvGroup


def frames(obs):
    """cam -> uint8 array, copied so later steps can't alias it."""
    return {c: np.array(v) for c, v in obs["pixels"].items()}


def max_pixel_diff(obs_a, obs_b) -> int:
    fa, fb = frames(obs_a), frames(obs_b)
    return max(
        int(np.abs(fa[c].astype(np.int32) - fb[c].astype(np.int32)).max())
        for c in fa
    )


def main():
    p = argparse.ArgumentParser(description="Verify subprocess vec-env matches sequential")
    p.add_argument("--env_task", type=str, default="libero_goal")
    p.add_argument("--task_id", type=int, default=3)
    p.add_argument("--group_size", type=int, default=8)
    p.add_argument("--n_steps", type=int, default=16)
    p.add_argument("--init_state_id", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from lerobot.envs.libero import LiberoEnv, _get_suite
    suite = _get_suite(args.env_task)
    G = args.group_size

    # ── Subprocess group under test ──
    print(f"[verify] spawning {G} env subprocesses for task {args.task_id} ...")
    group = TaskEnvGroup(suite, args.env_task, args.task_id, G)
    obs_par = group.reset_group(args.init_state_id, args.seed)

    # ── Single-process sequential reference (same seed/init/actions) ──
    print(f"[verify] building {G} in-process reference envs ...")
    ref_envs = [
        LiberoEnv(task_suite=suite, task_id=args.task_id, task_suite_name=args.env_task,
                  obs_type="pixels_agent_pos", init_states=True, episode_index=0)
        for _ in range(G)
    ]
    isid = args.init_state_id % group.n_init_states
    obs_ref = []
    for i, env in enumerate(ref_envs):
        env._init_state_id = isid
        o, _ = env.reset(seed=args.seed + i)
        obs_ref.append(o)

    action_dim = ref_envs[0].action_space.shape[0]
    rng = np.random.default_rng(args.seed)
    # (n_steps, G, n_exec, action_dim) so a single env_actions[i, k] indexing
    # matches group.step's signature exactly.
    actions = rng.uniform(-0.3, 0.3, size=(args.n_steps, G, 1, action_dim)).astype(np.float32)

    init_max = max(max_pixel_diff(obs_ref[i], obs_par[i]) for i in range(G))
    print(f"[verify] reset-frame max |diff|: {init_max} "
          f"({'OK, identical' if init_max == 0 else 'WARNING: nondeterministic reset'})")

    alive = set(range(G))
    worst = 0
    mismatch_steps = []

    for s in range(args.n_steps):
        # Subprocess group: parallel step of all alive envs.
        par_out = group.step(alive, actions[s], 0)  # k=0 in the size-1 chunk axis

        # Reference: sequential, clip + step exactly as the worker does.
        ref_out = {}
        for i in list(alive):
            env = ref_envs[i]
            a = np.clip(actions[s, i, 0], env.action_space.low, env.action_space.high).astype(np.float32)
            o, _r, term, _tr, info = env.step(a)
            ref_out[i] = (o, term)

        for i in list(alive):
            obs_p, term_p, _succ, _rew = par_out[i]
            obs_r, term_r = ref_out[i]
            if term_p != term_r:
                print(f"[verify] step {s} env {i}: termination mismatch par={term_p} ref={term_r}")
                if s not in mismatch_steps:
                    mismatch_steps.append(s)
            if term_p or term_r:
                alive.discard(i)  # auto-reset → frames diverge legitimately
                continue
            d = max_pixel_diff(obs_r, obs_p)
            if d > worst:
                worst = d
            if d != 0 and s not in mismatch_steps:
                mismatch_steps.append(s)
        print(f"[verify] step {s:3d}: max |ref - subproc| pixel diff so far = {worst}  (alive {len(alive)})")

    group.close()
    for env in ref_envs:
        env.close()

    print("\n" + "=" * 60)
    if worst == 0 and not mismatch_steps:
        print("PASS: subprocess vec-env is BITWISE identical to sequential.")
        print("→ Parallel env stepping is correct. Safe to run RL.")
    else:
        print(f"FAIL: max pixel diff = {worst}, mismatch at step(s) {mismatch_steps}.")
        print("→ Subprocess rendering/stepping diverges from the reference — investigate.")
    print("=" * 60)


if __name__ == "__main__":
    main()
