"""Staged (dense) reward for GRPO RL on LIBERO.

Replaces the binary success reward with a per-rollout scalar that gives partial
credit for HOW FAR a rollout got, so:
  - single-object grasp fumbles ("grasped but dropped" = 0.5 vs "never touched"
    = 0.0) create within-group variance, rescuing all-fail GRPO groups that the
    binary reward leaves degenerate (std 0 -> dropped, zero gradient);
  - compound "put both X and Y" tasks score per-object (0.5 for one placed),
    fixing the p^2 binary starvation (libero_10 T0/T7/T8).

Design (confirmed against the live env via src/libero_reward_probe.py):
  reward = mean over the task's GOAL CONJUNCTS of the per-conjunct MAX stage
           reached during the episode (transient grasps/lifts bank credit).

Per-conjunct stage ladder (each banked as a running max over the episode):
  0.00  untouched
  0.25  reached   (eef within REACH_EPS of the object center)
  0.50  grasped   (robosuite _check_grasp)
  0.75  lifted    (grasped AND object z > rest_z + LIFT_DZ)
  1.00  placed    (_eval_predicate(conjunct) — LIBERO's own goal test)

Conjuncts that don't name a graspable object (e.g. ['open', drawer]) get only
the 0/1 placed test — no intermediate stages — so this generalizes across all
libero_10 tasks. The reward is computed ONCE per rollout (a single scalar fed to
GRPO's trajectory advantage), so there is no per-timestep shaping to game.

Confirmed env API (Libero_*_Tabletop_Manipulation, robosuite sim):
  sim_env.parsed_problem['goal_state'] -> [['in','alphabet_soup_1','basket_1_contain_region'], ...]
  sim_env._eval_predicate(conjunct) -> bool   (per-predicate, not just the AND)
  sim_env.obj_body_id[name] -> body id; sim.data.body_xpos[bid] -> xyz
  sim_env._check_grasp(gripper, object_geoms);  gripper = robots[0].gripper
  sim_env.objects_dict[name].contact_geoms;  sim_env._eef_xpos -> xyz
"""
from __future__ import annotations

import numpy as np

# Placement-type predicates whose object (conj[1]) is a graspable prop we can
# stage with reach/grasp/lift. Other predicates fall back to the 0/1 placed test.
_PLACEMENT_PREDICATES = {"in", "on"}


def get_sim_env(libero_env):
    """Walk LiberoEnv -> OffScreenRenderEnv -> robosuite problem env (the object
    that has both .sim and _check_success). Returns None if not found."""
    cur = libero_env
    seen = {id(cur)}
    for _ in range(12):
        if getattr(cur, "sim", None) is not None and hasattr(cur, "_check_success"):
            return cur
        nxt = None
        for attr in ("env", "_env", "unwrapped"):
            cand = getattr(cur, attr, None)
            if cand is not None and id(cand) not in seen:
                nxt = cand
                break
        if nxt is None:
            break
        seen.add(id(nxt))
        cur = nxt
    if getattr(cur, "sim", None) is not None and hasattr(cur, "_check_success"):
        return cur
    return None


class StagedRewardTracker:
    """One per env. Call update() after each env.step on a NON-terminal state to
    bank stage progress; read reward() at episode end (use 1.0 on success-
    termination, since LiberoEnv auto-resets and the final state is gone)."""

    REACH_EPS = 0.07   # m, eef-to-object-center
    LIFT_DZ = 0.03     # m above the object's rest height

    def __init__(self, libero_env):
        self.sim_env = get_sim_env(libero_env)
        self.ok = self.sim_env is not None
        self.rest_z: dict[str, float] = {}
        self.goals: list = []
        if not self.ok:
            return
        pp = getattr(self.sim_env, "parsed_problem", {}) or {}
        self.goals = list(pp.get("goal_state", []))
        self.obj_body_id = getattr(self.sim_env, "obj_body_id", {}) or {}
        self.objects_dict = getattr(self.sim_env, "objects_dict", {}) or {}
        try:
            self.gripper = self.sim_env.robots[0].gripper
        except Exception:
            self.gripper = None
        self.max_stage = [0.0] * len(self.goals)
        # Record rest heights NOW (env just reset → objects at init pose) so a
        # first-chunk grasp can't poison the lift baseline.
        for conj in self.goals:
            obj = conj[1] if len(conj) >= 2 else None
            if conj and conj[0] in _PLACEMENT_PREDICATES and obj in self.obj_body_id:
                try:
                    self.rest_z[obj] = float(self._obj_pos(obj)[2])
                except Exception:
                    pass

    def _obj_pos(self, name) -> np.ndarray:
        bid = self.obj_body_id[name]
        return np.asarray(self.sim_env.sim.data.body_xpos[bid], dtype=np.float64)

    def _grasped(self, name) -> bool:
        if self.gripper is None:
            return False
        geoms = getattr(self.objects_dict.get(name), "contact_geoms", None)
        if geoms is None:
            return False
        try:
            return bool(self.sim_env._check_grasp(self.gripper, geoms))
        except Exception:
            return False

    def update(self) -> None:
        if not self.ok:
            return
        for ci, conj in enumerate(self.goals):
            if self.max_stage[ci] >= 1.0:
                continue
            # placed? — LIBERO's own predicate test; the ONLY signal for
            # non-graspable conjuncts (e.g. ['open', drawer]).
            try:
                if self.sim_env._eval_predicate(conj):
                    self.max_stage[ci] = 1.0
                    continue
            except Exception:
                pass
            pred = conj[0] if conj else None
            obj = conj[1] if len(conj) >= 2 else None
            if pred not in _PLACEMENT_PREDICATES or obj not in self.obj_body_id:
                continue
            pos = self._obj_pos(obj)
            self.rest_z.setdefault(obj, float(pos[2]))
            stage = 0.0
            try:
                eef = np.asarray(self.sim_env._eef_xpos, dtype=np.float64)
                if np.linalg.norm(eef - pos) < self.REACH_EPS:
                    stage = 0.25
            except Exception:
                pass
            if self._grasped(obj):
                stage = max(stage, 0.5)
                if pos[2] > self.rest_z[obj] + self.LIFT_DZ:
                    stage = 0.75
            if stage > self.max_stage[ci]:
                self.max_stage[ci] = stage

    def reward(self) -> float:
        if not self.ok or not self.max_stage:
            return 0.0
        return float(np.mean(self.max_stage))


# ---------------------------------------------------------------------------
# Self-test: run ONCE on Colab before a real RL run to confirm every stage
# signal resolves against the live env (the grasp-geom path especially).
#     python src/rl_staged_reward.py --suite libero_10 --task_id 0
# ---------------------------------------------------------------------------
def _selftest() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--task_id", type=int, default=0)
    ap.add_argument("--steps", type=int, default=40)
    args = ap.parse_args()

    from lerobot.envs.libero import LiberoEnv, _get_suite

    suite = _get_suite(args.suite)
    env = LiberoEnv(task_suite=suite, task_id=args.task_id, task_suite_name=args.suite,
                    obs_type="pixels_agent_pos", init_states=True, episode_index=0)
    env.reset(seed=0)
    print("task:", getattr(env, "task_description", "?"))

    t = StagedRewardTracker(env)
    print("tracker.ok:", t.ok)
    print("goal conjuncts:", t.goals)
    print("rest_z recorded for:", t.rest_z)
    print("gripper resolved:", t.gripper is not None)
    print("eef readable:", end=" ")
    try:
        print(np.asarray(t.sim_env._eef_xpos))
    except Exception as e:
        print("NO —", repr(e))

    print("\nPer-conjunct stage-signal wiring:")
    for conj in t.goals:
        obj = conj[1] if len(conj) >= 2 else None
        graspable = (conj and conj[0] in _PLACEMENT_PREDICATES and obj in t.obj_body_id)
        geoms = getattr(t.objects_dict.get(obj), "contact_geoms", None) if graspable else None
        ev_ok = True
        try:
            t.sim_env._eval_predicate(conj)
        except Exception:
            ev_ok = False
        print(f"  {conj}")
        print(f"     placed(_eval_predicate) ok: {ev_ok}")
        print(f"     graspable object: {graspable}  body_id: "
              f"{t.obj_body_id.get(obj) if graspable else '—'}")
        print(f"     contact_geoms resolved: {geoms is not None}"
              f"{'  (' + str(geoms)[:120] + ')' if geoms is not None else '  <-- GRASP STAGE WILL NOT FIRE'}")

    # Step random actions; confirm update() runs and stages can advance.
    import numpy as _np
    for _ in range(args.steps):
        a = env.action_space.sample().astype(_np.float32)
        env.step(a)
        t.update()
    print("\nafter", args.steps, "random steps — max_stage per conjunct:", t.max_stage,
          " reward:", round(t.reward(), 3))
    print("(random actions rarely grasp; the point is update() ran cleanly and "
          "the wiring above is all True.)")
    env.close()


if __name__ == "__main__":
    _selftest()
