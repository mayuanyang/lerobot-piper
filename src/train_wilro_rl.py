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

def _env_worker(conn, suite_name: str, task_id: int, max_episode_steps: int,
                env_ids: list[int]):
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
    import numpy as _np
    from lerobot.envs.libero import LiberoEnv, _get_suite

    suite = _get_suite(suite_name)
    envs = {
        eid: LiberoEnv(
            task_suite=suite, task_id=task_id, task_suite_name=suite_name,
            obs_type="pixels_agent_pos", init_states=True, episode_index=0,
        )
        for eid in env_ids
    }
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

    def __init__(self, suite, suite_name: str, task_id: int, group_size: int,
                 expected_cams: Optional[list[str]] = None,
                 max_episode_steps: int = 0, n_workers: Optional[int] = None):
        self.task_id = task_id
        self.group_size = group_size
        n_workers = group_size if not n_workers else max(1, min(n_workers, group_size))
        self.n_workers = n_workers
        # Contiguous env-id shards, one per worker process.
        self._shards = [[int(e) for e in s]
                        for s in np.array_split(np.arange(group_size), n_workers)]
        self._worker_of = {eid: w for w, shard in enumerate(self._shards) for eid in shard}

        self._conns = []
        self._procs = []
        for shard in self._shards:
            parent, child = self._ctx.Pipe()
            p = self._ctx.Process(
                target=_env_worker,
                args=(child, suite_name, task_id, max_episode_steps, shard),
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

    def step(self, indices, env_actions, k: int) -> dict:
        """Step envs `indices` with env_actions[i, k]. Workers handling >1 active
        env step them sequentially; all workers run in parallel. Returns
        {i: (obs, terminated, is_success, reward|None)}."""
        per_worker = self._group_by_worker(indices)
        for w, ids in per_worker.items():
            self._conns[w].send(("step", {i: env_actions[i, k] for i in ids}))
        out: dict[int, Any] = {}
        for w in per_worker:
            out.update(self._conns[w].recv())
        return out

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
    from lerobot.policies.factory import make_pre_post_processors

    policy = WilroPolicy.from_pretrained(args.policy_path)
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[rl] device: {device}")
    out_dir = Path(args.output_dir)
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
    for tid in task_ids:
        print(f"  task {tid}: {suite.get_task(tid).language}")

    # Per-task reward mode: tasks in binary_task_ids use 0/1 success even under
    # --staged_reward (and skip the StagedRewardTracker in rollout).
    binary_task_ids = set(args.binary_task_ids or [])
    if binary_task_ids:
        print(f"[rl] binary-reward tasks: {sorted(binary_task_ids)} "
              f"(staged for the rest: {sorted(set(task_ids) - binary_task_ids)})")

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
            )
            print(f"[rl] built {args.group_size} envs for task {tid} in {time.time()-t0:.1f}s")
        return env_pool[tid]

    sr_track: dict[int, deque] = {tid: deque(maxlen=50) for tid in task_ids}
    task_cycle = 0
    init_state_cycle: dict[int, int] = {tid: 0 for tid in task_ids}

    for it in range(args.rl_iterations):
        t_iter = time.time()

        # ── Collect groups ──
        records: list[ChunkRecord] = []
        group_summaries = []
        kept, attempts = 0, 0
        while kept < args.groups_per_iter and attempts < args.groups_per_iter * 6:
            attempts += 1
            tid = task_schedule[task_cycle % len(task_schedule)]
            task_cycle += 1
            this_staged = args.staged_reward and tid not in binary_task_ids
            group = get_group(tid)
            isid = init_state_cycle[tid] % group.n_init_states
            init_state_cycle[tid] += 1

            res = rollout_group(
                policy, preprocessor, postprocessor, group, isid,
                args.n_action_steps, args.exploration_std, device,
                base_seed=args.seed + 100_000 * it + 1000 * attempts,
                staged_reward=this_staged,
            )
            sr_track[tid].extend([float(s) for s in res.successes])
            n_succ = sum(res.successes)
            summary = {"task_id": tid, "init_state": isid,
                       "successes": n_succ, "of": len(res.successes)}
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

        # ── Update ──
        t_upd = time.time()
        stats = {"loss": float("nan")}
        if records:
            stats = grpo_update(policy, preprocessor, optimizer, records, args, device)
        update_s = time.time() - t_upd

        sr_str = "  ".join(
            f"T{tid}={np.mean(sr_track[tid]) * 100:.0f}%" if sr_track[tid] else f"T{tid}=–"
            for tid in task_ids
        )
        line = {
            "iter": it,
            "groups": group_summaries,
            "kept_groups": kept,
            "n_records": len(records),
            "elapsed_s": round(time.time() - t_iter, 1),
            "rollout_s": round(rollout_s, 1),
            "update_s": round(update_s, 1),
            **{k: v for k, v in stats.items() if k != "n_minibatches"},
            "sr": {tid: (float(np.mean(sr_track[tid])) if sr_track[tid] else None) for tid in task_ids},
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(line) + "\n")
        print(f"[rl] iter {it}: kept {kept}/{attempts} groups, {len(records)} chunks, "
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

    save_checkpoint(policy, preprocessor, postprocessor, out_dir, tag="final")
    for g in env_pool.values():
        g.close()


if __name__ == "__main__":
    main()