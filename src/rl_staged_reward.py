"""Staged (dense) reward for GRPO RL on LIBERO.

Replaces the binary success reward with a per-rollout scalar that gives partial
credit for HOW FAR a rollout got, so:
  - single-object grasp fumbles ("grasped but dropped" = 0.5 vs "never touched"
    = 0.0) create within-group variance, rescuing all-fail GRPO groups that the
    binary reward leaves degenerate (std 0 -> dropped, zero gradient);
  - compound "put both X and Y" tasks score per-object (0.5 for one placed),
    fixing the p^2 binary starvation (libero_10 T0/T7/T8).

Design (confirmed against the live env via src/libero_reward_probe.py):
  per-conjunct value = 0.5 * MAX stage reached  +  0.5 * FINAL stage held
  reward = mean over the task's GOAL CONJUNCTS of that per-conjunct value.
The 0.5/0.5 max+final split penalizes regressions (grasp-then-drop scores below
grasp-and-hold) without punishing recover-and-succeed — true success is hard-set
to 1.0 by the caller, and placed conjuncts lock to 1.0 here.

Per-conjunct stage ladder — BACK-LOADED so completion dominates the gradient,
and CONTINUOUS within the reach/lift/place bands so groups rarely go zero-variance
(degenerate -> dropped -> no gradient) at an intermediate stage:
  0.00 .. 0.10  reaching  (ramps as eef closes from REACH_FAR to REACH_EPS)
  0.25          grasped   (robosuite _check_grasp)
  0.25 .. 0.40  lifting   (ramps with object height from LIFT_DZ to LIFT_FULL)
  0.40 .. 0.85  placing   (ramps as the HELD object DESCENDS onto the goal region;
                           3D dist so hovering over the target does NOT max it)
  1.00          placed    (_eval_predicate(conjunct) — LIBERO's own goal test)

Conjuncts that don't name a graspable object (e.g. ['open', drawer]) get only
the 0/1 placed test — no intermediate stages — so this generalizes across all
libero_10 tasks. Stage is aggregated to ONE scalar per rollout (banked max+final),
so there is no per-timestep dense reward for GRPO to game.

Place-approach ramp ([0.40, 0.85)): once an object is grasped AND lifted, give
continuous credit for the held object's 3D distance to the goal region's owning
body, ramped PLACE_FAR -> PLACE_NEAR. This fills the gap the old discrete
lift->place jump left, which plateaued compound "put both X and Y" tasks at "one
object placed": nothing guided the held SECOND object onto the target, so RL had
no gradient past 0.40. **3D (not xy)** is deliberate: an xy-only ramp maxes the
moment the object is over the target AT ANY HEIGHT, i.e. it rewards HOVERING and
gives no gradient for the precise descent/placement — the exact "gets near but
won't place" failure. With 3D, credit only climbs as the object comes DOWN onto
the region (the lift band 0.40 covers transport via max(), so carrying high isn't
penalized). The region center is proxied by the owning body's xyz — basket_1 /
plate_1 resolve directly via obj_body_id; fixtures (stove/cabinet/microwave)
resolve by stripping the region suffix and looking the body up. The proxy has a
center offset (a tall fixture's body origin sits below its place surface), so the
ramp may saturate a bit BELOW 0.85 at true placement — fine, the 1.0 top stays
gated on _eval_predicate and the descent gradient is what matters. If the target
can't be resolved the ramp is SKIPPED (old discrete behavior, no regression).
CALIBRATION: run an RL pass with STAGED_PLACE_DEBUG=1 to log the real obj->region
3D/xy distance at each successful placement, then set PLACE_NEAR/PLACE_FAR off
those numbers. Run the self-test (bottom of file) on the compound tasks first to
confirm resolution fires.

Confirmed env API (Libero_*_Tabletop_Manipulation, robosuite sim):
  sim_env.parsed_problem['goal_state'] -> [['in','alphabet_soup_1','basket_1_contain_region'], ...]
  sim_env._eval_predicate(conjunct) -> bool   (per-predicate, not just the AND)
  sim_env.obj_body_id[name] -> body id; sim.data.body_xpos[bid] -> xyz
  sim_env._check_grasp(gripper, object_geoms);  gripper = robots[0].gripper
  sim_env.objects_dict[name].contact_geoms;  sim_env._eef_xpos -> xyz
"""
from __future__ import annotations

import os
import re

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

    REACH_EPS = 0.07    # m, eef-to-object-center: full reach credit at/under this
    REACH_FAR = 0.30    # m: zero reach credit at/over this (linear ramp between)
    LIFT_DZ = 0.03      # m above rest height: lift credit starts here
    LIFT_FULL = 0.15    # m above rest height: full lift credit at/over this
    PLACE_FAR = 0.30    # m, HELD object-to-region 3D dist: place credit starts (= W_LIFT)
    PLACE_NEAR = 0.05   # m: near-full place credit at/under this (calibrate via
    #                     STAGED_PLACE_DEBUG=1 — placement dist is proxy-offset)

    # Back-loaded band tops — completion (1.0) dominates the gradient.
    W_REACH = 0.10
    W_GRASP = 0.25
    W_LIFT = 0.40
    W_PLACE = 0.85      # place-ramp top; 0.85->1.0 commit gap keeps a real reward
    #                     for actually releasing (vs hovering over the target).
    #                     The 1.0 stays gated on _eval_predicate.

    def __init__(self, libero_env):
        self.sim_env = get_sim_env(libero_env)
        self.ok = self.sim_env is not None
        self.rest_z: dict[str, float] = {}
        self.goals: list = []
        self._target_bid_cache: dict[str, object] = {}  # region name -> body id | None
        self._place_debug = bool(os.environ.get("STAGED_PLACE_DEBUG"))
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
        self.final_stage = [0.0] * len(self.goals)
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

    def _resolve_target_bid(self, region_name):
        """Body id of the goal region's owning object/fixture, or None.
        Region names look like 'basket_1_contain_region' / 'flat_stove_1_cook_region';
        the owning body is the leading '<name>_<idx>'. Plain object targets
        ('plate_1') resolve directly via obj_body_id. None => caller skips the
        place-ramp and keeps the discrete lift->place behavior (no regression)."""
        if not region_name:
            return None
        if region_name in self._target_bid_cache:
            return self._target_bid_cache[region_name]
        cands = [region_name]
        m = re.match(r"^(.*?_\d+)", region_name)  # 'basket_1_contain_region' -> 'basket_1'
        if m and m.group(1) != region_name:
            cands.append(m.group(1))
        bid = None
        for c in cands:                                   # registered objects first
            if c in self.obj_body_id:
                bid = self.obj_body_id[c]
                break
        if bid is None:                                   # then fixtures by body name
            model = getattr(getattr(self.sim_env, "sim", None), "model", None)
            for c in cands:
                for nm in (c, c + "_main"):
                    try:
                        bid = model.body_name2id(nm)
                        break
                    except Exception:
                        bid = None
                if bid is not None:
                    break
        self._target_bid_cache[region_name] = bid
        return bid

    def _target_pos(self, conj):
        """xyz (np.float64, shape (3,)) of the goal region for this conjunct, or None."""
        if len(conj) < 3:
            return None
        bid = self._resolve_target_bid(conj[2])
        if bid is None:
            return None
        try:
            return np.asarray(self.sim_env.sim.data.body_xpos[bid], dtype=np.float64)
        except Exception:
            return None

    def _current_stage(self, conj) -> float:
        """Continuous stage in [0, 1] for ONE conjunct at the current sim state."""
        # placed? — LIBERO's own predicate test; the ONLY signal for
        # non-graspable conjuncts (e.g. ['open', drawer]).
        try:
            if self.sim_env._eval_predicate(conj):
                return 1.0
        except Exception:
            pass
        pred = conj[0] if conj else None
        obj = conj[1] if len(conj) >= 2 else None
        if pred not in _PLACEMENT_PREDICATES or obj not in self.obj_body_id:
            return 0.0
        pos = self._obj_pos(obj)
        self.rest_z.setdefault(obj, float(pos[2]))
        # reaching: continuous ramp 0 -> W_REACH as the eef closes in.
        stage = 0.0
        try:
            eef = np.asarray(self.sim_env._eef_xpos, dtype=np.float64)
            dist = float(np.linalg.norm(eef - pos))
            frac = (self.REACH_FAR - dist) / (self.REACH_FAR - self.REACH_EPS)
            stage = self.W_REACH * min(1.0, max(0.0, frac))
        except Exception:
            pass
        # grasped, then lifting: continuous ramp W_GRASP -> W_LIFT with height.
        if self._grasped(obj):
            stage = self.W_GRASP
            dz = pos[2] - self.rest_z[obj]
            if dz > self.LIFT_DZ:
                frac = (dz - self.LIFT_DZ) / (self.LIFT_FULL - self.LIFT_DZ)
                stage = self.W_GRASP + (self.W_LIFT - self.W_GRASP) * min(1.0, max(0.0, frac))
                # placing: once lifted, ramp W_LIFT -> W_PLACE as the held object
                # DESCENDS onto the goal region (3D dist, the [0.40, 0.85) band the
                # discrete lift->place jump was missing). 3D not xy so hovering over
                # the target doesn't max it — the descent is what earns credit.
                # Skipped if the region body can't be resolved → discrete fallback.
                # 1.0 stays on _eval_predicate.
                tgt = self._target_pos(conj)
                if tgt is not None:
                    d = float(np.linalg.norm(pos - tgt))
                    frac_p = (self.PLACE_FAR - d) / (self.PLACE_FAR - self.PLACE_NEAR)
                    place_stage = self.W_LIFT + (self.W_PLACE - self.W_LIFT) * min(1.0, max(0.0, frac_p))
                    stage = max(stage, place_stage)
        return stage

    def update(self) -> None:
        if not self.ok:
            return
        for ci, conj in enumerate(self.goals):
            if self.max_stage[ci] >= 1.0:
                self.final_stage[ci] = 1.0  # placement is sticky
                continue
            cur = self._current_stage(conj)
            self.final_stage[ci] = cur
            if cur >= 1.0:
                if self.max_stage[ci] < 1.0 and self._place_debug:
                    self._report_placement(conj)   # log real placement dist (once)
                self.max_stage[ci] = 1.0
            elif cur > self.max_stage[ci]:
                self.max_stage[ci] = cur

    def _report_placement(self, conj) -> None:
        """STAGED_PLACE_DEBUG: print the obj->region 3D/xy distance at the moment a
        conjunct first becomes placed, to calibrate PLACE_NEAR/PLACE_FAR off real
        successes (the body-center proxy has an offset from the true place pose)."""
        obj = conj[1] if len(conj) >= 2 else None
        tgt = self._target_pos(conj)
        if obj not in self.obj_body_id or tgt is None:
            return
        try:
            d = self._obj_pos(obj) - tgt
            print(f"[staged] PLACED {conj}: obj->region 3D={float(np.linalg.norm(d)):.3f} "
                  f"xy={float(np.linalg.norm(d[:2])):.3f} m "
                  f"(band {self.PLACE_NEAR}->{self.PLACE_FAR})", flush=True)
        except Exception:
            pass

    def reward(self) -> float:
        if not self.ok or not self.max_stage:
            return 0.0
        # 0.5*max + 0.5*final per conjunct: credits how far it got AND that it
        # held there (drops lose credit); placed conjuncts are locked to 1.0.
        vals = [
            1.0 if m >= 1.0 else 0.5 * m + 0.5 * f
            for m, f in zip(self.max_stage, self.final_stage)
        ]
        return float(np.mean(vals))


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
        region = conj[2] if len(conj) >= 3 else None
        tgt_bid = t._resolve_target_bid(region) if (graspable and region) else None
        print(f"     goal region: {region}  -> resolved body_id: {tgt_bid}"
              f"{'  <-- PLACE RAMP WILL NOT FIRE (discrete lift->place fallback)' if (graspable and tgt_bid is None) else ''}")
        if graspable and tgt_bid is not None:
            try:
                d = t._obj_pos(obj) - t._target_pos(conj)
                # @rest the object is FAR; the PLACED distance (logged separately
                # via STAGED_PLACE_DEBUG=1 in an RL run) is what calibrates the
                # band. The 3D number is what the ramp uses; xy shown for context.
                print(f"     obj->region dist @rest: 3D={float(np.linalg.norm(d)):.3f} "
                      f"xy={float(np.linalg.norm(d[:2])):.3f} m "
                      f"(place band {t.PLACE_FAR}->{t.PLACE_NEAR} m, 3D)")
            except Exception as e:
                print(f"     obj->region dist @rest: ERROR {e!r}")

    # Step random actions; confirm update() runs and stages can advance.
    import numpy as _np
    for _ in range(args.steps):
        a = env.action_space.sample().astype(_np.float32)
        env.step(a)
        t.update()
    print("\nafter", args.steps, "random steps — max_stage:", [round(s, 3) for s in t.max_stage],
          " final_stage:", [round(s, 3) for s in t.final_stage],
          " reward:", round(t.reward(), 3))
    print("(random actions rarely grasp; the point is update() ran cleanly and "
          "the wiring above is all True.)")
    env.close()


if __name__ == "__main__":
    _selftest()
