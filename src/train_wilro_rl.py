"""
GRPO online RL for WilR (SmolVLM2-based flow-matching VLA) on LIBERO.

Adapted from train_wiltechs_vla_rl.py with WilR-specific adjustments:
  - SmolVLM2-500M backbone (smaller VLM, faster inference)
  - Different flow matching interface (sample_actions vs flow_actions_from_noise)
  - Latent generator (MLP-based) instead of QFormer
  - No chat template (simple [vision | language] concatenation)

Usage:
    python src/train_wilro_rl.py \
        --policy_path outputs/train/wilro_checkpoint_10k \
        --env_task libero_goal --task_ids 3 8 9 \
        --output_dir outputs/rl/wilro_goal \
        --group_size 8 --n_action_steps 8 --exploration_std 0.2

Data-parallel rollout across N GPUs (each rank rolls out its own slice of the
groups; rank 0 runs the GRPO update and broadcasts weights). The global batch is
N × groups_per_iter — pass the SAME flags, just launch with torchrun:

    torchrun --standalone --nproc_per_node=4 src/train_wilro_rl.py \
        --policy_path ... --env_task libero_spatial \
        --output_dir outputs/rl/wilro_spatial \
        --group_size 8 --groups_per_iter 6 --n_action_steps 8

(For the same gradient batch as a single GPU but ~N× faster, divide
groups_per_iter by N. Needs ~N× the env-subprocess RAM.)
"""

from __future__ import annotations

import os
# MuJoCo/LIBERO offscreen rendering: force GPU (EGL) before robosuite/mujoco is imported.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import json
import math
import random
import time
import multiprocessing as mp
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Pure GRPO math (identical to wiltechs_vla_rl.py)
# ---------------------------------------------------------------------------

def gaussian_logp_per_step(actions: torch.Tensor, mu: torch.Tensor, sigma: float) -> torch.Tensor:
    """Log N(actions; mu, sigma^2 I) summed over action dims, kept per timestep.

    actions/mu: (B, T, D) -> returns (B, T).
    """
    var = sigma * sigma
    logp = -0.5 * ((actions - mu) ** 2) / var - math.log(sigma) - 0.5 * math.log(2.0 * math.pi)
    return logp.sum(dim=-1)


def grpo_group_advantages(rewards: list[float], eps: float = 1e-4) -> Optional[list[float]]:
    """Group-relative advantages; None if the group is degenerate (all same outcome)."""
    r = np.asarray(rewards, dtype=np.float64)
    if r.std() < 1e-8:
        return None
    adv = (r - r.mean()) / (r.std() + eps)
    return adv.tolist()


def grpo_clip_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    advantages: torch.Tensor,
    clip_low: float = 0.2,
    clip_high: float = 0.28,
    weights: Optional[torch.Tensor] = None,
    dual_clip: Optional[float] = 3.0,
) -> tuple[torch.Tensor, dict]:
    """PPO-clip objective with decoupled clip range (DAPO clip-higher), no KL."""
    log_ratio = (logp_new - logp_old).clamp(-20.0, 20.0)
    ratio = torch.exp(log_ratio)
    adv = advantages.unsqueeze(-1)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high) * adv
    per_elem = torch.minimum(unclipped, clipped)
    if dual_clip is not None:
        per_elem = torch.where(adv < 0, torch.maximum(per_elem, dual_clip * adv), per_elem)
    if weights is None:
        loss = -per_elem.mean()
    else:
        w = weights.unsqueeze(-1)
        loss = -(per_elem * w).sum() / (w.sum() * per_elem.shape[-1])
    with torch.no_grad():
        kl_elems = (ratio - 1.0) - log_ratio
        approx_kl = float(kl_elems.mean())
        approx_kl_median = float(kl_elems.median())
    stats = {
        "ratio_mean": float(ratio.detach().mean()),
        "ratio_max": float(ratio.detach().max()),
        "clip_frac": float(((ratio < 1.0 - clip_low) | (ratio > 1.0 + clip_high)).float().mean()),
        "approx_kl": approx_kl,
        "approx_kl_median": approx_kl_median,
    }
    return loss, stats


# ---------------------------------------------------------------------------
# Rollout storage
# ---------------------------------------------------------------------------

@dataclass
class ChunkRecord:
    """Everything needed to recompute log pi(a|s, x1) under the current policy."""
    pixels: dict[str, np.ndarray]      # cam -> (H, W, 3) uint8 (raw env obs)
    agent_pos: np.ndarray              # (state_dim,) float
    task: str
    x1: np.ndarray                     # (horizon, action_dim) f32 — flow noise latent
    action: np.ndarray                 # (n_exec, action_dim) f32 — executed, NORMALIZED space
    logp_old: np.ndarray               # (n_exec,) f32
    advantage: float = 0.0             # filled in after the group finishes
    weight: float = 1.0                # 1/episode_chunks — per-trajectory normalization


@dataclass
class GroupResult:
    task_id: int
    init_state_id: int
    successes: list[bool] = field(default_factory=list)
    records_per_env: list[list[ChunkRecord]] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Batched observation -> model batch
# ---------------------------------------------------------------------------

def obs_list_to_model_batch(obs_list: list[dict], tasks: list[str], preprocessor):
    from lerobot.envs.utils import preprocess_observation

    stacked = {
        "pixels": {
            cam: np.stack([o["pixels"][cam] for o in obs_list])
            for cam in obs_list[0]["pixels"]
        },
        "agent_pos": np.stack([o["agent_pos"] for o in obs_list]),
    }
    observation = preprocess_observation(stacked)
    observation["task"] = list(tasks)
    return preprocessor(observation)


# ---------------------------------------------------------------------------
# Environment pool — one LiberoEnv PER SUBPROCESS (each its own EGL context).
#
# MuJoCo offscreen rendering (MUJOCO_GL=egl) is NOT thread-safe: the GL context
# is current-per-thread, so stepping envs in ThreadPoolExecutor workers silently
# returns garbage frames (verified: reset frames match, every threaded step
# diffs by 255 vs sequential). Process isolation gives each env its own context
# and true (GIL-free) parallel physics+render. The StagedRewardTracker reads
# deep sim state (body_xpos, _eef_xpos, _check_grasp, _eval_predicate), so it
# MUST live inside the worker where the env object is — the main process only
# ever sees obs/terminated/success/reward over the pipe.
# ---------------------------------------------------------------------------

def _patch_libero_control_freq(control_freq: int):
    """Build LiberoEnv's OffScreenRenderEnv at `control_freq` Hz instead of
    robosuite's 20 Hz default. The LIBERO demos (and eval) run at 10 Hz, so each
    delta-EEF action is sized for a 1/10 s control interval; rolling out at 20 Hz
    holds each action half as long → the same action moves a different distance,
    taking the policy off-distribution from both the data and eval. Matching the
    rollout freq to the dataset is required for RL gains to transfer to eval.

    Applied inside each spawned worker (the class is patched per process)."""
    import os as _os
    from lerobot.envs.libero import LiberoEnv as _LiberoEnv
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero import get_libero_path

    def _make_envs_task(self, task_suite, task_id: int = 0):
        task = task_suite.get_task(task_id)
        self.task = task.name
        self.task_description = task.language
        bddl = _os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        # robosuite picks the EGL render GPU from render_gpu_device_id (it ignores
        # MUJOCO_EGL_DEVICE_ID). RL_RENDER_GPU is set per rank in init_distributed
        # so each rank renders on its own GPU (EGL idx 0..7 are renderable here);
        # defaults to 0 for single-GPU / non-torchrun runs.
        _render_gpu = int(_os.environ.get("RL_RENDER_GPU", "0"))
        env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=self.observation_height,
            camera_widths=self.observation_width,
            control_freq=control_freq,   # match dataset/eval fps (default LIBERO=20)
            render_gpu_device_id=_render_gpu,
        )
        env.reset()
        return env

    _LiberoEnv._make_envs_task = _make_envs_task


def _env_worker(conn, suite_name: str, task_id: int, max_episode_steps: int,
                env_ids: list[int], control_freq: int = 10, render_gpu: int = -1):
    """Host len(env_ids) LiberoEnvs in ONE process, stepped sequentially within
    the worker (shards run in parallel across workers). One EGL context per
    process is fine for multiple envs as long as they render on this one thread.
    Protocol over `conn` — payloads are keyed by absolute env id so the main
    process can reassemble across shards:
      send ("ready", meta) on startup (meta from this worker's first env); loop:
        ("reset", init_state_id, {eid: seed})  -> {eid: obs}
        ("make_tracker",)                      -> {eid: tracker.ok}
        ("step", {eid: action_np})             -> {eid: (obs, term, succ, reward|None)}
        ("reward", [eid, ...])                 -> {eid: tracker.reward()}
        ("close",)                             -> exit
    The worker auto-banks staged progress: tracker.update() on each non-terminal
    step, and reward = 1.0 if success else tracker.reward() on termination."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    # Render this worker on its assigned GPU so EGL contexts (and their GPU memory)
    # spread across devices instead of exhausting one. _make_envs_task reads
    # RL_RENDER_GPU; set it here, per worker, before any env/EGL is created.
    if render_gpu is not None and render_gpu >= 0:
        os.environ["RL_RENDER_GPU"] = str(render_gpu)
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(render_gpu)
    import time as _time
    import random as _random
    import numpy as _np
    from lerobot.envs.libero import LiberoEnv, _get_suite

    if control_freq and control_freq > 0:
        _patch_libero_control_freq(control_freq)

    suite = _get_suite(suite_name)

    # Concurrent EGL context / framebuffer creation across the many worker
    # processes races on some drivers ("Offscreen framebuffer is not complete,
    # 0x8cdd"), and the error is a mujoco.FatalError that poisons the process —
    # so an in-process retry can't recover. Instead SERIALIZE creation with a
    # cross-process file lock: only one EGL context is built at a time, BOX-WIDE.
    # The lock MUST be global (one path for every rank+worker): a per-GPU lock
    # lets ranks pinned to different GPUs (RL_PIN_EGL=1) build framebuffers
    # concurrently, which is exactly the race that triggers 0x8cdd. Init is
    # one-time at startup, so a global lock only slows env creation, not rollout.
    import fcntl
    _egl_lock_path = os.environ.get("RL_EGL_LOCK", "/tmp/wilro_egl_init.lock")

    def _make_env(eid):
        with open(_egl_lock_path, "w") as _lf:
            fcntl.flock(_lf, fcntl.LOCK_EX)   # released when the block exits
            return LiberoEnv(
                task_suite=suite, task_id=task_id, task_suite_name=suite_name,
                obs_type="pixels_agent_pos", init_states=True, episode_index=0,
            )

    envs = {eid: _make_env(eid) for eid in env_ids}
    trackers: dict[int, Any] = {eid: None for eid in env_ids}
    bounds = {eid: (e.action_space.low, e.action_space.high) for eid, e in envs.items()}

    first = envs[env_ids[0]]
    obs, _ = first.reset(seed=0)  # populates cams/metadata for the handshake
    conn.send(("ready", {
        "cameras": sorted(obs["pixels"].keys()),
        "max_episode_steps": first._max_episode_steps,
        "n_init_states": len(first._init_states),
        "task_description": first.task_description,
    }))

    try:
        while True:
            cmd = conn.recv()
            op = cmd[0]
            if op == "reset":
                _, init_state_id, seeds = cmd
                out = {}
                for eid in env_ids:
                    env = envs[eid]
                    env._init_state_id = init_state_id
                    o, _ = env.reset(seed=seeds[eid])
                    trackers[eid] = None  # rebuilt via make_tracker AFTER reset
                    out[eid] = o
                conn.send(out)
            elif op == "make_tracker":
                from rl_staged_reward import StagedRewardTracker
                out = {}
                for eid in env_ids:
                    trackers[eid] = StagedRewardTracker(envs[eid])
                    out[eid] = trackers[eid].ok
                conn.send(out)
            elif op == "step":
                out = {}
                for eid, action in cmd[1].items():
                    lo, hi = bounds[eid]
                    a = _np.clip(action, lo, hi).astype(_np.float32)
                    o, _r, terminated, _trunc, info = envs[eid].step(a)
                    tr = trackers[eid]
                    if terminated:
                        succ = bool(info.get("is_success", False))
                        # Env auto-resets on termination → final state is gone;
                        # full success ⇒ all conjuncts placed ⇒ 1.0, else banked.
                        rew = (1.0 if succ else tr.reward()) if tr is not None else None
                        out[eid] = (o, True, succ, rew)
                    else:
                        if tr is not None:
                            tr.update()  # bank stage on the post-action state
                        out[eid] = (o, False, False, None)
                conn.send(out)
            elif op == "reward":
                conn.send({eid: (trackers[eid].reward() if trackers[eid] is not None else 0.0)
                           for eid in cmd[1]})
            elif op == "close":
                break
    finally:
        for e in envs.values():
            try:
                e.close()
            except Exception:
                pass
        conn.close()


class TaskEnvGroup:
    """group_size LiberoEnvs of ONE task, sharded across `n_workers` processes.
    Envs in the same worker step sequentially; workers run in parallel. Fewer
    workers => less per-process import RAM (torch/lerobot/mujoco are re-imported
    per spawned process) but slower rollout (~group_size/n_workers serial steps).
    Default n_workers == group_size (one env per process, max parallelism)."""

    _ctx = mp.get_context("spawn")  # fresh interpreter per worker: no inherited
    #                                 CUDA/EGL state from the model-holding parent

    # Render GPUs to spread EGL contexts across (one offscreen FBO per env eats
    # GPU memory; piling them all on one device hits "framebuffer not complete,
    # 0x8cdd" once that GPU is full). Set RL_RENDER_GPUS="1,2,3,4,5,6,7" to round-
    # robin workers across those devices. Falls back to the single RL_RENDER_GPU
    # (default 0). Class-level counter so the round-robin spans ALL tasks' workers.
    _render_idx = 0

    @classmethod
    def _render_gpu_list(cls):
        raw = os.environ.get("RL_RENDER_GPUS", "").strip()
        if raw:
            return [int(x) for x in raw.replace(",", " ").split()]
        return [int(os.environ.get("RL_RENDER_GPU", "0"))]

    def __init__(self, suite, suite_name: str, task_id: int, group_size: int,
                 expected_cams: Optional[list[str]] = None,
                 max_episode_steps: int = 0, n_workers: Optional[int] = None,
                 control_freq: int = 10):
        self.task_id = task_id
        self.group_size = group_size
        n_workers = group_size if not n_workers else max(1, min(n_workers, group_size))
        self.n_workers = n_workers
        # Contiguous env-id shards, one per worker process.
        self._shards = [[int(e) for e in s]
                        for s in np.array_split(np.arange(group_size), n_workers)]
        self._worker_of = {eid: w for w, shard in enumerate(self._shards) for eid in shard}

        render_gpus = self._render_gpu_list()
        self._conns = []
        self._procs = []
        for shard in self._shards:
            parent, child = self._ctx.Pipe()
            rg = render_gpus[TaskEnvGroup._render_idx % len(render_gpus)]
            TaskEnvGroup._render_idx += 1
            p = self._ctx.Process(
                target=_env_worker,
                args=(child, suite_name, task_id, max_episode_steps, shard,
                      control_freq, rg),
                daemon=True,
            )
            p.start()
            child.close()  # only the worker holds the child end
            self._conns.append(parent)
            self._procs.append(p)

        metas = [c.recv() for c in self._conns]
        for tag, _m in metas:
            assert tag == "ready", f"unexpected worker handshake: {tag}"
        meta = metas[0][1]

        if expected_cams:
            got = set(meta["cameras"])
            missing = [c for c in expected_cams if c not in got]
            if missing:
                raise RuntimeError(
                    f"LIBERO env obs cameras {sorted(got)} are missing the policy's "
                    f"expected camera(s) {missing} (policy cameras: {expected_cams})."
                )
            print(f"[rl] camera check OK — env provides {sorted(got)}, "
                  f"policy expects {expected_cams}")
        print(f"[rl] task {task_id}: {group_size} envs across {n_workers} worker "
              f"process(es) (~{int(np.ceil(group_size / n_workers))} env/proc)")
        self.task_description = meta["task_description"]
        suite_default = meta["max_episode_steps"]
        self.max_steps = min(suite_default, max_episode_steps) if max_episode_steps else suite_default
        if self.max_steps < suite_default:
            print(f"[rl] task {task_id}: capping episode length {suite_default} -> {self.max_steps}")
        self.n_init_states = meta["n_init_states"]

    def reset_group(self, init_state_id: int, base_seed: int):
        isid = init_state_id % self.n_init_states
        for w, shard in enumerate(self._shards):
            self._conns[w].send(("reset", isid, {e: base_seed + e for e in shard}))
        obs_by_id: dict[int, Any] = {}
        for c in self._conns:
            obs_by_id.update(c.recv())
        return [obs_by_id[i] for i in range(self.group_size)]

    def make_trackers(self):
        """Build a StagedRewardTracker in every worker env (AFTER reset)."""
        for c in self._conns:
            c.send(("make_tracker",))
        ok: dict[int, bool] = {}
        for c in self._conns:
            ok.update(c.recv())
        return [ok[i] for i in range(self.group_size)]

    def _group_by_worker(self, indices):
        per_worker: dict[int, list[int]] = {}
        for i in indices:
            per_worker.setdefault(self._worker_of[i], []).append(i)
        return per_worker

    def step_send(self, indices, env_actions, k: int):
        """Fire ("step", ...) to every worker handling `indices` WITHOUT waiting.
        Returns the per-worker id map to hand to step_recv. Splitting send/recv
        lets several groups' steps run concurrently (see rollout_groups_concurrent):
        send for all groups first, then recv, so the workers step in parallel."""
        per_worker = self._group_by_worker(indices)
        for w, ids in per_worker.items():
            self._conns[w].send(("step", {i: env_actions[i, k] for i in ids}))
        return per_worker

    def step_recv(self, per_worker) -> dict:
        out: dict[int, Any] = {}
        for w in per_worker:
            out.update(self._conns[w].recv())
        return out

    def step(self, indices, env_actions, k: int) -> dict:
        """Step envs `indices` with env_actions[i, k] (blocking send+recv). Workers
        handling >1 active env step them sequentially; all workers run in parallel.
        Returns {i: (obs, terminated, is_success, reward|None)}."""
        return self.step_recv(self.step_send(indices, env_actions, k))

    def get_rewards(self, indices) -> dict:
        """Banked staged reward for timed-out (still-active) envs."""
        per_worker = self._group_by_worker(indices)
        for w, ids in per_worker.items():
            self._conns[w].send(("reward", ids))
        out: dict[int, float] = {}
        for w in per_worker:
            out.update(self._conns[w].recv())
        return out

    def close(self):
        for c in self._conns:
            try:
                c.send(("close",))
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


# ---------------------------------------------------------------------------
# Rollout: one GRPO group
# ---------------------------------------------------------------------------

@torch.no_grad()
def rollout_group(
    policy, preprocessor, postprocessor, group: TaskEnvGroup,
    init_state_id: int, n_exec: int, sigma: float, device, base_seed: int,
    staged_reward: bool = False,
) -> GroupResult:
    G = group.group_size
    result = GroupResult(task_id=group.task_id, init_state_id=init_state_id,
                         successes=[False] * G, records_per_env=[[] for _ in range(G)],
                         rewards=[0.0] * G)

    policy.model.eval()
    obs_list = group.reset_group(init_state_id, base_seed)
    # Staged-reward trackers live in the workers; built AFTER reset (objects at
    # init pose). The worker auto-updates them on each step and returns reward.
    if staged_reward:
        group.make_trackers()
    active = list(range(G))
    steps_done = 0
    horizon = policy.config.horizon
    action_dim = policy.config.action_dim

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False)
    )

    all_idx = list(range(G))
    while active and steps_done < group.max_steps:
        # Always forward the FULL group (fixed batch dim = G) so cuBLAS picks the
        # same GEMM kernel as grpo_update's minibatches (which is why
        # update_minibatch MUST == group_size). With matching M, per-row bf16
        # output is reproducible at update time → rollout_drift ~0 and the
        # importance ratio stays sane. Terminated envs are forwarded (cheap waste)
        # but never stored/executed. Indices below are ABSOLUTE env ids (i).
        cur_obs = [obs_list[i] for i in all_idx]
        batch = obs_list_to_model_batch(
            cur_obs, [group.task_description] * G, preprocessor,
        )

        # Sample flow noise (stored as latent for recomputation)
        x1 = torch.randn(G, horizon, action_dim, device=device)

        # Run flow matching denoising from x1 (model method, differentiable).
        # Must run under the SAME autocast policy as grpo_update so rollout-time
        # and update-time logp agree numerically (controls rollout_drift).
        with autocast_ctx:
            mu_full = policy.model.flow_actions_from_noise(batch, x1)
        mu = mu_full[:, :n_exec].float()

        # Add exploration noise
        actions = mu + sigma * torch.randn_like(mu)
        logp_old = gaussian_logp_per_step(actions, mu, sigma)  # (G, n_exec)

        # Post-process actions (denormalize)
        env_actions = postprocessor(actions.reshape(G * n_exec, action_dim))
        env_actions = env_actions.reshape(G, n_exec, action_dim).numpy()

        # Store records — ACTIVE envs only, indexed by absolute env id i.
        for i in active:
            result.records_per_env[i].append(ChunkRecord(
                pixels={c: np.ascontiguousarray(v) for c, v in obs_list[i]["pixels"].items()},
                agent_pos=np.asarray(obs_list[i]["agent_pos"], dtype=np.float32),
                task=group.task_description,
                x1=x1[i].cpu().numpy().astype(np.float32),
                action=actions[i].cpu().numpy().astype(np.float32),
                logp_old=logp_old[i].cpu().numpy().astype(np.float32),
            ))

        # Execute chunk step-by-step (active envs only) — each env steps in its
        # own process, so the G sends fan out and run truly in parallel. Worker
        # banks staged progress and returns reward on termination.
        still_active = set(active)
        for k in range(n_exec):
            if not still_active:
                break
            for i, (obs, terminated, succ, rew) in group.step(still_active, env_actions, k).items():
                obs_list[i] = obs
                if terminated:
                    result.successes[i] = succ
                    if staged_reward:
                        result.rewards[i] = rew
                    still_active.discard(i)
        active = list(still_active)
        steps_done += n_exec

    # Timed-out envs (never terminated): state is intact, banked stage is current.
    if staged_reward and active:
        for i, rew in group.get_rewards(active).items():
            result.rewards[i] = rew

    return result


@torch.no_grad()
def rollout_groups_concurrent(
    policy, preprocessor, postprocessor,
    specs: "list[tuple[TaskEnvGroup, int]]",
    n_exec: int, sigma: float, device, base_seeds: "list[int]",
    staged_flags: "list[bool]",
) -> "list[GroupResult]":
    """Roll out several GRPO groups AT ONCE, overlapping their env stepping.

    Each group's policy forward still runs at its own batch dim = group_size, so
    the determinism invariant (update_minibatch == group_size -> rollout_drift ~0,
    see rollout_group) is untouched. The ONLY thing shared across groups is the
    env step: we fan out every group's worker `step_send` first, then collect with
    `step_recv`, so the slow CPU physics+render of all groups runs in parallel
    instead of group-by-group. GPU-0 memory is unchanged (one forward at a time).
    `specs` MUST hold DISTINCT TaskEnvGroups (the same group's workers can't serve
    two rollouts at once). Returns one GroupResult per spec, in input order.
    """
    policy.model.eval()
    horizon = policy.config.horizon
    action_dim = policy.config.action_dim
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False)
    )

    states = []
    for (group, isid), seed, staged in zip(specs, base_seeds, staged_flags):
        G = group.group_size
        obs_list = group.reset_group(isid, seed)
        if staged:
            group.make_trackers()
        states.append({
            "group": group, "G": G, "staged": staged, "obs_list": obs_list,
            "active": list(range(G)), "steps_done": 0,
            "result": GroupResult(
                task_id=group.task_id, init_state_id=isid,
                successes=[False] * G, records_per_env=[[] for _ in range(G)],
                rewards=[0.0] * G),
        })

    def _running(s):
        return s["active"] and s["steps_done"] < s["group"].max_steps

    while any(_running(s) for s in states):
        # ── Phase 1: per-group forward (batch=G) + store records (GPU, one at a
        #    time — cheap, keeps GPU-0 memory flat and the GEMM shape == G). ──
        for s in states:
            if not _running(s):
                s["_chunk"] = None
                continue
            group, G, obs_list, active = s["group"], s["G"], s["obs_list"], s["active"]
            cur_obs = [obs_list[i] for i in range(G)]
            batch = obs_list_to_model_batch(
                cur_obs, [group.task_description] * G, preprocessor)
            x1 = torch.randn(G, horizon, action_dim, device=device)
            with autocast_ctx:
                mu_full = policy.model.flow_actions_from_noise(batch, x1)
            mu = mu_full[:, :n_exec].float()
            actions = mu + sigma * torch.randn_like(mu)
            logp_old = gaussian_logp_per_step(actions, mu, sigma)
            env_actions = postprocessor(
                actions.reshape(G * n_exec, action_dim)
            ).reshape(G, n_exec, action_dim).numpy()
            for i in active:
                s["result"].records_per_env[i].append(ChunkRecord(
                    pixels={c: np.ascontiguousarray(v)
                            for c, v in obs_list[i]["pixels"].items()},
                    agent_pos=np.asarray(obs_list[i]["agent_pos"], dtype=np.float32),
                    task=group.task_description,
                    x1=x1[i].cpu().numpy().astype(np.float32),
                    action=actions[i].cpu().numpy().astype(np.float32),
                    logp_old=logp_old[i].cpu().numpy().astype(np.float32),
                ))
            s["_chunk"] = {"env_actions": env_actions, "still_active": set(active)}

        # ── Phase 2: n_exec sub-steps, env stepping OVERLAPPED across groups:
        #    send every running group's step, THEN recv them all. ──
        for k in range(n_exec):
            pending = []
            for s in states:
                ch = s.get("_chunk")
                if not ch or not ch["still_active"]:
                    continue
                pw = s["group"].step_send(ch["still_active"], ch["env_actions"], k)
                pending.append((s, pw))
            if not pending:
                break
            for s, pw in pending:
                ch = s["_chunk"]
                for i, (obs, terminated, succ, rew) in s["group"].step_recv(pw).items():
                    s["obs_list"][i] = obs
                    if terminated:
                        s["result"].successes[i] = succ
                        if s["staged"]:
                            s["result"].rewards[i] = rew
                        ch["still_active"].discard(i)

        # ── Phase 3: advance each group's active set / step counter. ──
        for s in states:
            ch = s.get("_chunk")
            if not ch:
                continue
            s["active"] = list(ch["still_active"])
            s["steps_done"] += n_exec

    # Timed-out (still-active) envs: banked staged reward.
    for s in states:
        if s["staged"] and s["active"]:
            for i, rew in s["group"].get_rewards(s["active"]).items():
                s["result"].rewards[i] = rew

    return [s["result"] for s in states]


# ---------------------------------------------------------------------------
# GRPO update
# ---------------------------------------------------------------------------

def grpo_update(
    policy, preprocessor, optimizer, records: list[ChunkRecord], args, device,
) -> dict:
    policy.model.train()
    # Keep RobotCNN in eval mode (BatchNorm running stats mismatch)
    if getattr(policy.model, "robot_visual_encoder", None) is not None:
        policy.model.robot_visual_encoder.eval()
    horizon = policy.config.horizon

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False)
    )
    trainable = [p for p in policy.model.parameters() if p.requires_grad]

    # FIXED minibatch partition. Drop the final partial minibatch: a tail of
    # size < update_minibatch has a different GEMM shape than the rollout/pre-pass
    # forwards (M=group_size), so those few chunks drift and can explode the
    # importance ratio. order is a fresh permutation each iter, so the dropped
    # (<=update_minibatch-1) records are random — no systematic bias.
    order = np.random.permutation(len(records))
    n_full = (len(records) // args.update_minibatch) * args.update_minibatch
    minibatches = [
        [records[i] for i in order[s:s + args.update_minibatch]]
        for s in range(0, n_full, args.update_minibatch)
    ]

    def mb_tensors(mb):
        obs_list = [{"pixels": r.pixels, "agent_pos": r.agent_pos} for r in mb]
        batch = obs_list_to_model_batch(obs_list, [r.task for r in mb], preprocessor)
        x1 = torch.from_numpy(np.stack([r.x1 for r in mb])).to(device)
        actions = torch.from_numpy(np.stack([r.action for r in mb])).to(device)
        adv = torch.tensor([r.advantage for r in mb], device=device, dtype=torch.float32)
        w = torch.tensor([r.weight for r in mb], device=device, dtype=torch.float32)
        return batch, x1, actions, adv, w

    def forward_logp(batch, x1, actions):
        # Same autocast policy as rollout_group (see rollout_drift note there).
        with autocast_ctx:
            mu_full = policy.model.flow_actions_from_noise(batch, x1)
        mu = mu_full[:, :args.n_action_steps].float()
        return gaussian_logp_per_step(actions, mu, args.exploration_std)

    # Pre-pass: logp_old for ALL minibatches BEFORE any optimizer step
    logp_old_list: list[torch.Tensor] = []
    agg = {"loss": 0.0, "ratio_mean": 0.0, "ratio_max": 0.0, "clip_frac": 0.0,
           "rollout_drift": 0.0, "n_minibatches": 0}
    with torch.no_grad():
        for mb in minibatches:
            batch, x1, actions, _adv, _w = mb_tensors(mb)
            lp = forward_logp(batch, x1, actions)
            logp_old_list.append(lp)
            lp_roll = torch.from_numpy(np.stack([r.logp_old for r in mb])).to(device)
            agg["rollout_drift"] += float((lp - lp_roll).abs().mean())

    # Clipped updates
    agg["approx_kl"] = 0.0
    agg["stopped_early"] = 0
    agg["nonfinite_skipped"] = 0
    done = False
    for _epoch in range(args.update_epochs):
        for mi, mb in enumerate(minibatches):
            batch, x1, actions, adv, w = mb_tensors(mb)
            logp_new = forward_logp(batch, x1, actions)

            loss, stats = grpo_clip_loss(
                logp_new, logp_old_list[mi], adv,
                clip_low=args.clip_low, clip_high=args.clip_high,
                weights=w, dual_clip=args.dual_clip,
            )

            if stats["approx_kl_median"] > args.target_kl:
                agg["stopped_early"] = 1
                print(f"[rl]   early stop at minibatch {agg['n_minibatches']}: "
                      f"median kl {stats['approx_kl_median']:.4f} > {args.target_kl}")
                done = True
                break

            optimizer.zero_grad()
            if not torch.isfinite(loss):
                agg["nonfinite_skipped"] += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()

            agg["loss"] += float(loss.detach())
            agg["ratio_mean"] += stats["ratio_mean"]
            agg["ratio_max"] = max(agg["ratio_max"], stats["ratio_max"])
            agg["clip_frac"] += stats["clip_frac"]
            agg["approx_kl"] += stats["approx_kl"]
            agg["approx_kl_median"] = agg.get("approx_kl_median", 0.0) + stats["approx_kl_median"]
            agg["n_minibatches"] += 1
        if done:
            break

    n = max(1, agg["n_minibatches"])
    for k in ("loss", "ratio_mean", "clip_frac", "approx_kl", "approx_kl_median"):
        agg[k] = agg.get(k, 0.0) / n
    agg["rollout_drift"] /= max(1, len(minibatches))
    return agg


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def load_policy_and_processors(args, device):
    """Load WilR SFT checkpoint and zero all dropout."""
    from models.wilro.wilro_policy import WilroPolicy
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors

    # Pin the checkpoint's device to THIS process's GPU BEFORE loading. lerobot
    # maps the safetensors weights to config.device, which is cuda:0 by default —
    # so under torchrun every rank would otherwise load onto GPU 0 and OOM it.
    cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    cfg.device = str(device)
    policy = WilroPolicy.from_pretrained(args.policy_path, config=cfg)
    policy.config.pretrained_path = args.policy_path

    # Zero all dropout for deterministic rollout/update matching
    policy.config.vision_dropout_prob = 0.0
    n_zeroed = 0
    for m in policy.model.modules():
        if isinstance(m, torch.nn.Dropout) and m.p > 0:
            m.p = 0.0
            n_zeroed += 1
    print(f"[rl] zeroed {n_zeroed} Dropout modules + vision dropout config")

    if args.gradient_checkpointing and hasattr(policy.model, "gradient_checkpointing_enable"):
        policy.model.gradient_checkpointing_enable()

    policy.to(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, preprocessor, postprocessor


def make_optimizer(params, lr, use_8bit: bool):
    if use_8bit:
        try:
            import bitsandbytes as bnb
            print("[rl] using 8-bit Adam (bitsandbytes)")
            return bnb.optim.Adam8bit(params, lr=lr)
        except ImportError:
            print("[rl] bitsandbytes not installed; falling back to fp32 Adam")
    return torch.optim.Adam(params, lr=lr)


def save_checkpoint(policy, preprocessor, postprocessor, out_dir: Path, tag: str):
    ckpt = out_dir / f"rl-checkpoint-{tag}"
    ckpt.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(ckpt)
    preprocessor.save_pretrained(ckpt)
    postprocessor.save_pretrained(ckpt)
    print(f"[rl] checkpoint saved: {ckpt}")


# ---------------------------------------------------------------------------
# Distributed (data-parallel ROLLOUT, single learner)
# ---------------------------------------------------------------------------
#
# Rollout is 85-90% of wall-clock and embarrassingly parallel across groups
# (each group is a self-contained (task, init_state) unit for GRPO advantage),
# so we fan ROLLOUT out across GPUs and keep ONE learner:
#   * each rank loads its own policy replica + env subprocesses, rolls out its
#     own slice of the groups (init states strided by rank -> disjoint starts),
#   * all ranks gather their records onto rank 0 (gloo; records are CPU/numpy),
#   * rank 0 ALONE runs grpo_update (UNCHANGED -> every determinism invariant
#     preserved: full-group-G forward, update_minibatch==group_size, the
#     per-minibatch median-KL early stop), then broadcasts the new weights.
# The update stays single-GPU (it is only ~12% at world_size=1); its cost grows
# ~linearly with world_size because rank 0 sees world_size× the records, so this
# design is best for world_size<=~4. Beyond that, scale the update too with a
# DDP grad-average + collective early-stop (the natural follow-up).

def init_distributed():
    """Bring up data-parallel rollout when launched via torchrun.

    torchrun sets RANK / WORLD_SIZE / LOCAL_RANK. Returns
    (rank, world_size, local_rank, obj_group); obj_group is a gloo group used to
    gather the CPU/numpy rollout records (the default NCCL group handles GPU
    weight broadcast). No torchrun -> (0, 1, 0, None) = original single-GPU path.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return 0, 1, 0, None
    import torch.distributed as dist
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # Pin this rank's MuJoCo EGL rendering to its own GPU so the env subprocesses
    # spread render load across GPUs instead of piling every rank's offscreen
    # framebuffers onto one device (which exhausts it -> "framebuffer not complete,
    # 0x8cdd"). IMPORTANT: robosuite uses its OWN EGLGLContext selected by the
    # `render_gpu_device_id` kwarg and IGNORES MUJOCO_EGL_DEVICE_ID — so the real
    # knob is RL_RENDER_GPU, read in the worker and passed to OffScreenRenderEnv.
    # (EGL device indices 0..7 are the renderable GPUs on this box; the probe in
    # probe_egl.py confirmed 8..15 are phantom/non-renderable.) Set RL_PIN_EGL=0 to
    # concentrate all render on EGL device 0 instead of spreading.
    if os.environ.get("RL_PIN_EGL", "1") == "1":
        # Render GPU = (local_rank + RL_RENDER_OFFSET) % RL_RENDER_NGPU. EGL FBO
        # creation appears to fail when a live NCCL/CUDA context sits on the SAME
        # GPU (single-GPU runs with no NCCL render fine; multi-GPU with NCCL on the
        # render GPU hits 0x8cdd). Set RL_RENDER_OFFSET to push render onto the
        # GPUs NOT used for compute, e.g. NPROC=4 + RL_RENDER_OFFSET=4 -> compute
        # on 0-3, render on 4-7. EGL idx 0-7 are all renderable here (probe_egl.py).
        _ngpu = int(os.environ.get("RL_RENDER_NGPU", "8"))
        _off = int(os.environ.get("RL_RENDER_OFFSET", "0"))
        _render_gpu = (local_rank + _off) % _ngpu
        os.environ["RL_RENDER_GPU"] = str(_render_gpu)
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(_render_gpu)
        os.environ.setdefault("EGL_DEVICE_ID", str(_render_gpu))
    else:
        os.environ["RL_RENDER_GPU"] = "0"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    obj_group = dist.new_group(backend="gloo")
    print(f"[rl] distributed: rank {rank}/{world_size} local_rank {local_rank}")
    return rank, world_size, local_rank, obj_group


def gather_rollouts(local, rank, world_size, obj_group):
    """Gather each rank's rollout payload onto rank 0. Returns the per-rank list
    on rank 0, else None. world_size==1 short-circuits to [local]."""
    if world_size == 1:
        return [local]
    import torch.distributed as dist
    out = [None] * world_size if rank == 0 else None
    dist.gather_object(local, out, dst=0, group=obj_group)
    return out


def broadcast_params(params, world_size):
    """In-place broadcast of rank-0's updated weights to every rank (NCCL)."""
    if world_size == 1:
        return
    import torch.distributed as dist
    with torch.no_grad():
        for p in params:
            dist.broadcast(p.data, src=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GRPO RL for WilR on LIBERO")
    parser.add_argument("--policy_path", type=str, required=True,
                        help="SFT checkpoint (local dir or HF id)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--env_task", type=str, default="libero_goal",
                        help="LIBERO suite: libero_spatial/object/goal/libero_10")
    parser.add_argument("--task_ids", type=int, nargs="*", default=None,
                        help="Task ids to train on (default: all)")
    parser.add_argument("--group_size", type=int, default=8,
                        help="G rollouts per (task, init_state)")
    parser.add_argument("--groups_per_iter", type=int, default=2,
                        help="Groups collected per update iteration")
    parser.add_argument("--n_action_steps", type=int, default=8,
                        help="Executed chunk length between replans")
    parser.add_argument("--rollout_concurrency", type=int, default=0,
                        help="Max DISTINCT groups rolled out at once (env stepping "
                             "overlapped across them). 0 = as many as fit in one "
                             "groups_per_iter batch (full concurrency). Lower it if "
                             "the box is CPU-starved and env workers oversubscribe.")
    parser.add_argument("--staged_reward", action="store_true",
                        help="Use a dense staged reward (reach/grasp/lift/place per goal object, "
                             "mean over conjuncts) instead of binary success. Gives all-fail GRPO "
                             "groups within-group variance so grasp-fumble and compound 'put both' "
                             "tasks produce gradient. True task success is still logged separately.")
    parser.add_argument("--binary_task_ids", type=int, nargs="*", default=None,
                        help="Task ids that use BINARY (0/1 success) reward even when "
                             "--staged_reward is on. Use for tasks where the dense reward is being "
                             "hacked (staged reward flat while true success drops). These tasks skip "
                             "the StagedRewardTracker entirely. Default: none (all staged).")
    parser.add_argument("--task_sample_weights", type=str, nargs="*", default=None,
                        help="Per-task sampling weights as 'tid:weight' (e.g. 0:1 8:2 9:1) for a "
                             "weighted round-robin — a task with weight w is sampled w× as often, "
                             "shifting gradient share toward it. DEFAULT: weight 1 for every task "
                             "in --task_ids (equal sampling); any task you omit also stays at 1.")
    parser.add_argument("--max_episode_steps", type=int, default=0,
                        help="Cap rollout episode length (0 = suite default)")
    parser.add_argument("--control_freq", type=int, default=10,
                        help="LIBERO env control frequency (Hz). MUST match the dataset "
                             "and eval (LIBERO demos are 10 Hz). Stock lerobot uses "
                             "robosuite's 20 Hz default, which makes each delta-EEF action "
                             "move half as far → policy runs off-distribution. 0 = leave "
                             "the env at its default (20 Hz). Keep at 10 to match eval.")
    parser.add_argument("--exploration_std", type=float, default=0.2,
                        help="Gaussian exploration std in NORMALIZED action units. "
                             "logp sensitivity to bf16 mu-drift scales 1/std, so values "
                             "below ~0.15 blow up the importance ratio (ratio_max -> e^20, "
                             "clip_frac ~0.5). Keep >= 0.15.")
    parser.add_argument("--rl_iterations", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--clip_low", type=float, default=0.2)
    parser.add_argument("--clip_high", type=float, default=0.28, help="DAPO clip-higher")
    parser.add_argument("--update_epochs", type=int, default=1)
    parser.add_argument("--update_minibatch", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--target_kl", type=float, default=0.05,
                        help="Early-stop threshold on median approx-KL")
    parser.add_argument("--dual_clip", type=float, default=3.0,
                        help="Dual-clip PPO bound on negative-advantage branch")
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--save_freq", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env_workers", type=int, default=None,
                        help="Worker PROCESSES per task group (default: group_size, i.e. one "
                             "env per process = max parallelism). Lower it to cut per-process "
                             "import RAM (torch/lerobot/mujoco are re-imported per spawned proc) "
                             "when RAM-limited; envs in a worker step sequentially, so rollout "
                             "slows ~group_size/env_workers. Total env count is always group_size.")
    args = parser.parse_args()

    # rollout forwards a fixed batch of group_size envs; grpo_update recomputes
    # logp in minibatches of update_minibatch. They MUST match or the bf16 GEMM
    # kernel differs between the two forwards → mu drifts → importance ratio
    # explodes (ratio_max -> e^20, clip_frac ~0.5). See rollout_group.
    assert args.update_minibatch == args.group_size, (
        f"update_minibatch ({args.update_minibatch}) must == group_size "
        f"({args.group_size}) so rollout/update GEMM shapes match (rollout_drift)."
    )

    rank, world_size, local_rank, obj_group = init_distributed()
    is_main = rank == 0

    # Per-rank seed offset so each replica draws different exploration noise /
    # flow noise (their init-state slices already differ).
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    device = (torch.device(f"cuda:{local_rank}") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"[rl] rank {rank}/{world_size} device: {device}")
    out_dir = Path(args.output_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "rl_log.jsonl"

    policy, preprocessor, postprocessor = load_policy_and_processors(args, device)
    trainable = [p for p in policy.model.parameters() if p.requires_grad]
    print(f"[rl] trainable params: {sum(p.numel() for p in trainable):,}")
    optimizer = make_optimizer(trainable, args.lr, args.use_8bit_adam)

    from lerobot.envs.libero import _get_suite
    suite = _get_suite(args.env_task)
    task_ids = args.task_ids if args.task_ids else list(range(len(suite.tasks)))
    print(f"[rl] suite={args.env_task} task_ids={task_ids}")
    print(f"[rl] env control_freq={args.control_freq} Hz "
          f"({'matches 10 Hz dataset/eval' if args.control_freq == 10 else 'WARNING: not 10 Hz — check it matches your dataset+eval'})")
    for tid in task_ids:
        print(f"  task {tid}: {suite.get_task(tid).language}")

    # Per-task reward mode: tasks in binary_task_ids use 0/1 success even under
    # --staged_reward (and skip the StagedRewardTracker in rollout). When
    # --staged_reward is off, EVERY task is binary regardless of binary_task_ids.
    binary_task_ids = set(args.binary_task_ids or [])
    staged_ids = [tid for tid in task_ids
                  if args.staged_reward and tid not in binary_task_ids]
    binary_ids = [tid for tid in task_ids if tid not in staged_ids]
    print(f"[rl] reward mode (--staged_reward={'on' if args.staged_reward else 'off'}):")
    print(f"[rl]   STAGED (dense): {staged_ids if staged_ids else 'none'}")
    print(f"[rl]   BINARY (0/1)  : {binary_ids if binary_ids else 'none'}")
    for tid in task_ids:
        mode = "staged" if tid in staged_ids else "binary"
        print(f"[rl]   task {tid:>2} [{mode}]: {suite.get_task(tid).language}")

    # Weighted round-robin schedule: a task with weight w appears w× per cycle,
    # so it is sampled w× as often (more kept groups -> larger gradient share).
    task_weights = {tid: 1 for tid in task_ids}
    for item in (args.task_sample_weights or []):
        tid_str, w_str = item.split(":")
        tid_w = int(tid_str)
        if tid_w not in task_weights:
            raise ValueError(f"--task_sample_weights tid {tid_w} not in task_ids {task_ids}")
        task_weights[tid_w] = max(1, int(w_str))
    task_schedule = [tid for tid in task_ids for _ in range(task_weights[tid])]
    print(f"[rl] task sampling weights: {task_weights} -> schedule {task_schedule}"
          f"{'  (default: equal)' if all(w == 1 for w in task_weights.values()) else ''}")

    # Camera verification
    expected_cams = [
        k.split(".")[-1] for k in policy.config.cameras_for_vision_state_concat
    ]

    # Env pool
    env_pool: dict[int, TaskEnvGroup] = {}

    def get_group(tid: int) -> TaskEnvGroup:
        if tid not in env_pool:
            t0 = time.time()
            env_pool[tid] = TaskEnvGroup(
                suite, args.env_task, tid, args.group_size,
                expected_cams=expected_cams,
                max_episode_steps=args.max_episode_steps,
                n_workers=args.env_workers,
                control_freq=args.control_freq,
            )
            print(f"[rl] built {args.group_size} envs for task {tid} in {time.time()-t0:.1f}s")
        return env_pool[tid]

    sr_track: dict[int, deque] = {tid: deque(maxlen=50) for tid in task_ids}
    # Phase-shift the task cursor by rank so the ranks cover DIFFERENT tasks each
    # iter (more per-update task diversity under data-parallel rollout) rather
    # than all picking the same task. world_size==1 => rank 0 => starts at 0.
    task_cycle = rank
    # Init states strided by rank: rank r walks {r, r+world_size, ...} mod 50, so
    # no two ranks ever roll the same (task, init_state). world_size==1 => {0,1,..}
    # identical to the original single-GPU schedule.
    init_state_cycle: dict[int, int] = {tid: rank for tid in task_ids}

    # Environment pool restart interval to prevent resource exhaustion.
    # MuJoCo EGL contexts leak memory over time; restarting clears them.
    ENV_RESTART_INTERVAL = 5

    for it in range(args.rl_iterations):
        t_iter = time.time()

        # Periodic env pool restart to prevent GPU memory / subprocess exhaustion
        if it > 0 and it % ENV_RESTART_INTERVAL == 0 and is_main:
            print(f"[rl] restarting env pool at iter {it} (preventing resource exhaustion)")
            for tid, group in env_pool.items():
                group.close()
            env_pool.clear()
            import gc
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[rl] env pool cleared, continuing...")

        # ── Collect groups (this rank's slice) ──
        records: list[ChunkRecord] = []
        group_summaries = []
        kept, attempts = 0, 0
        cap = args.rollout_concurrency or args.groups_per_iter
        while kept < args.groups_per_iter and attempts < args.groups_per_iter * 6:
            # Build a batch of DISTINCT groups to roll out concurrently (the same
            # TaskEnvGroup can't serve two rollouts at once — its workers are shared).
            specs, batch_meta, seeds = [], [], []
            batch_ids: set[int] = set()
            limit = min(cap, args.groups_per_iter - kept)
            while len(specs) < limit and attempts < args.groups_per_iter * 6:
                tid = task_schedule[task_cycle % len(task_schedule)]
                group = get_group(tid)
                if id(group) in batch_ids:
                    break  # would duplicate a group in this batch -> next batch
                attempts += 1
                task_cycle += 1
                this_staged = args.staged_reward and tid not in binary_task_ids
                isid = init_state_cycle[tid] % group.n_init_states
                init_state_cycle[tid] += world_size  # stride so ranks stay disjoint
                specs.append((group, isid))
                batch_meta.append((tid, isid, this_staged))
                seeds.append(args.seed + 100_000_000 * rank + 100_000 * it
                             + 1000 * attempts)
                batch_ids.add(id(group))
            if not specs:
                break

            results = rollout_groups_concurrent(
                policy, preprocessor, postprocessor, specs,
                args.n_action_steps, args.exploration_std, device,
                base_seeds=seeds, staged_flags=[m[2] for m in batch_meta],
            )

            for (tid, isid, this_staged), res in zip(batch_meta, results):
                n_succ = sum(res.successes)
                summary = {"task_id": tid, "init_state": isid,
                           "successes": int(n_succ), "of": len(res.successes)}
                if this_staged:
                    summary["reward_mean"] = round(float(np.mean(res.rewards)), 3)
                group_summaries.append(summary)

                # Staged (dense, in [0,1]) for staged tasks; binary 0/1 for tasks in
                # binary_task_ids (and whenever --staged_reward is off).
                reward_vals = (res.rewards if this_staged
                               else [1.0 if s else 0.0 for s in res.successes])
                adv = grpo_group_advantages(reward_vals)
                if adv is None:
                    continue  # all-success or all-fail: skip
                kept += 1
                for env_i, env_records in enumerate(res.records_per_env):
                    for r in env_records:
                        r.advantage = adv[env_i]
                        r.weight = 1.0 / max(1, len(env_records))
                    records.extend(env_records)

        rollout_s = time.time() - t_iter

        # ── Gather every rank's rollouts onto rank 0 (collective: ALL ranks). ──
        gathered = gather_rollouts(
            {"records": records, "summaries": group_summaries, "kept": kept},
            rank, world_size, obj_group,
        )

        # ── Update on rank 0 only (grpo_update unchanged) ──
        stats = {"loss": float("nan")}
        update_s = 0.0
        if is_main:
            all_records = [r for part in gathered for r in part["records"]]
            all_summaries = [s for part in gathered for s in part["summaries"]]
            total_kept = sum(part["kept"] for part in gathered)
            # SR over EVERY attempted group (degenerate groups still count toward
            # true success rate). Order is irrelevant for the deque mean.
            for s in all_summaries:
                sr_track[s["task_id"]].extend(
                    [1.0] * s["successes"] + [0.0] * (s["of"] - s["successes"]))

            t_upd = time.time()
            if all_records:
                stats = grpo_update(policy, preprocessor, optimizer, all_records, args, device)
            update_s = time.time() - t_upd

            sr_str = "  ".join(
                f"T{tid}={np.mean(sr_track[tid]) * 100:.0f}%" if sr_track[tid] else f"T{tid}=–"
                for tid in task_ids
            )
            line = {
                "iter": it,
                "groups": all_summaries,
                "kept_groups": total_kept,
                "n_records": len(all_records),
                "world_size": world_size,
                "elapsed_s": round(time.time() - t_iter, 1),
                "rollout_s": round(rollout_s, 1),
                "update_s": round(update_s, 1),
                **{k: v for k, v in stats.items() if k != "n_minibatches"},
                "sr": {tid: (float(np.mean(sr_track[tid])) if sr_track[tid] else None) for tid in task_ids},
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(line) + "\n")
            print(f"[rl] iter {it}: kept {total_kept} groups (×{world_size} ranks), "
                  f"{len(all_records)} chunks, "
                  f"loss={stats.get('loss', float('nan')):.4f}, "
                  f"ratio={stats.get('ratio_mean', float('nan')):.3f}, "
                  f"clip={stats.get('clip_frac', float('nan')):.2f}, "
                  f"kl={stats.get('approx_kl', float('nan')):.3f}/"
                  f"med={stats.get('approx_kl_median', float('nan')):.4f}, "
                  f"drift={stats.get('rollout_drift', float('nan')):.3f}, "
                  f"{line['elapsed_s']}s (rollout {line['rollout_s']}s / update {line['update_s']}s) "
                  f"| rolling SR: {sr_str}")

            if (it + 1) % args.save_freq == 0:
                save_checkpoint(policy, preprocessor, postprocessor, out_dir, tag=str(it + 1))

        # ── Sync rank-0's updated weights back to every actor (collective). ──
        broadcast_params(trainable, world_size)

    if is_main:
        save_checkpoint(policy, preprocessor, postprocessor, out_dir, tag="final")
    for g in env_pool.values():
        g.close()
    if world_size > 1:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()