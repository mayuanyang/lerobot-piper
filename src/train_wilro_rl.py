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
        --group_size 8 --n_action_steps 8 --exploration_std 0.1
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
# Environment pool
# ---------------------------------------------------------------------------

class TaskEnvGroup:
    """G LiberoEnv instances of ONE task."""

    def __init__(self, suite, suite_name: str, task_id: int, group_size: int,
                 expected_cams: Optional[list[str]] = None,
                 max_episode_steps: int = 0):
        from lerobot.envs.libero import LiberoEnv

        self.task_id = task_id
        self.envs = [
            LiberoEnv(
                task_suite=suite,
                task_id=task_id,
                task_suite_name=suite_name,
                obs_type="pixels_agent_pos",
                init_states=True,
                episode_index=0,
            )
            for _ in range(group_size)
        ]
        if expected_cams:
            obs, _ = self.envs[0].reset(seed=0)
            got = set(obs["pixels"].keys())
            missing = [c for c in expected_cams if c not in got]
            if missing:
                raise RuntimeError(
                    f"LIBERO env obs cameras {sorted(got)} are missing the policy's "
                    f"expected camera(s) {missing} (policy cameras: {expected_cams})."
                )
            print(f"[rl] camera check OK — env provides {sorted(got)}, "
                  f"policy expects {expected_cams}")
        self.task_description = self.envs[0].task_description
        suite_default = self.envs[0]._max_episode_steps
        self.max_steps = min(suite_default, max_episode_steps) if max_episode_steps else suite_default
        if self.max_steps < suite_default:
            print(f"[rl] task {task_id}: capping episode length {suite_default} -> {self.max_steps}")
        self.n_init_states = len(self.envs[0]._init_states)

    def reset_group(self, init_state_id: int, base_seed: int):
        obs_list = []
        for i, env in enumerate(self.envs):
            env._init_state_id = init_state_id % self.n_init_states
            obs, _ = env.reset(seed=base_seed + i)
            obs_list.append(obs)
        return obs_list

    def close(self):
        for env in self.envs:
            env.close()


# ---------------------------------------------------------------------------
# WilR-specific: flow matching denoising from stored noise
# ---------------------------------------------------------------------------

@torch.no_grad()
def flow_actions_from_noise_wilro(
    model, batch: dict, x1: torch.Tensor, n_exec: int,
) -> torch.Tensor:
    """Run WilR's flow matching denoising starting from stored noise x1.

    This mirrors WilR's sample_actions() but:
      1. Uses the provided x1 instead of sampling new noise
      2. Returns the FULL horizon (not just n_exec) so the caller can slice
      3. Runs under no_grad (used during rollout and pre-pass)

    Returns: mu_full (B, horizon, action_dim) — the deterministic denoised output
    """
    B = batch["observation.state"].shape[0]
    device = batch["observation.state"].device
    horizon = model.config.horizon
    action_dim = model.config.action_dim

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False)
    )

    with autocast_ctx:
        # Stage A: VLM encoder (run once)
        kv_cache, vlm_kv_pad_mask, L_vis, L_lang = model._run_vlm_and_cache_kv(batch)
        robot_tokens = model._compute_robot_tokens(batch)
        latents = model._generate_latents(batch, B, device, torch.bfloat16)

        # Stage B: DiT denoising from x1
        N = int(getattr(model.config, "num_inference_steps", 10))
        x_t = x1.clone()  # Start from the provided noise
        dt = -1.0 / N
        t = torch.ones(B, device=device, dtype=torch.float32)

        for _ in range(N):
            v_t = model._run_dit(
                batch, x_t.to(torch.bfloat16), t, kv_cache, vlm_kv_pad_mask,
                robot_tokens, latents, action_prefix=None,
                L_vis=L_vis, L_lang=L_lang,
            ).float()
            x_t = x_t + dt * v_t
            t = t + dt

    return x_t  # (B, horizon, action_dim) — full denoised output


# ---------------------------------------------------------------------------
# Rollout: one GRPO group
# ---------------------------------------------------------------------------

@torch.no_grad()
def rollout_group(
    policy, preprocessor, postprocessor, group: TaskEnvGroup,
    init_state_id: int, n_exec: int, sigma: float, device, base_seed: int,
    staged_reward: bool = False,
) -> GroupResult:
    G = len(group.envs)
    result = GroupResult(task_id=group.task_id, init_state_id=init_state_id,
                         successes=[False] * G, records_per_env=[[] for _ in range(G)],
                         rewards=[0.0] * G)

    policy.model.eval()
    obs_list = group.reset_group(init_state_id, base_seed)
    # Staged-reward trackers must be built AFTER reset (objects at init pose)
    trackers = None
    if staged_reward:
        from rl_staged_reward import StagedRewardTracker
        trackers = [StagedRewardTracker(group.envs[i]) for i in range(G)]
    active = list(range(G))
    steps_done = 0
    horizon = policy.config.horizon
    action_dim = policy.config.action_dim

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False)
    )

    while active and steps_done < group.max_steps:
        cur_obs = [obs_list[i] for i in active]
        batch = obs_list_to_model_batch(
            cur_obs, [group.task_description] * len(active), preprocessor,
        )

        # Sample flow noise (stored as latent for recomputation)
        x1 = torch.randn(len(active), horizon, action_dim, device=device)

        # Run flow matching denoising from x1
        mu_full = flow_actions_from_noise_wilro(policy.model, batch, x1, n_exec)
        mu = mu_full[:, :n_exec].float()

        # Add exploration noise
        actions = mu + sigma * torch.randn_like(mu)
        logp_old = gaussian_logp_per_step(actions, mu, sigma)  # (B, n_exec)

        # Post-process actions (denormalize)
        env_actions = postprocessor(actions.reshape(len(active) * n_exec, action_dim))
        env_actions = env_actions.reshape(len(active), n_exec, action_dim).numpy()

        # Store records
        for j, i in enumerate(list(active)):
            result.records_per_env[i].append(ChunkRecord(
                pixels={c: np.ascontiguousarray(v) for c, v in cur_obs[j]["pixels"].items()},
                agent_pos=np.asarray(cur_obs[j]["agent_pos"], dtype=np.float32),
                task=group.task_description,
                x1=x1[j].cpu().numpy().astype(np.float32),
                action=actions[j].cpu().numpy().astype(np.float32),
                logp_old=logp_old[j].cpu().numpy().astype(np.float32),
            ))

        # Execute chunk step-by-step
        still_active = []
        for j, i in enumerate(list(active)):
            env = group.envs[i]
            terminated = False
            for k in range(n_exec):
                a = np.clip(env_actions[j, k], env.action_space.low, env.action_space.high)
                obs, _r, terminated, _trunc, info = env.step(a.astype(np.float32))
                obs_list[i] = obs
                if terminated:
                    succ = bool(info.get("is_success", False))
                    result.successes[i] = succ
                    if trackers is not None:
                        # Env auto-reset on termination → can't read final state;
                        # full success ⇒ all conjuncts placed ⇒ 1.0, else use the
                        # stage banked before this terminating step.
                        result.rewards[i] = 1.0 if succ else trackers[i].reward()
                    break
                if trackers is not None:
                    trackers[i].update()  # bank stage on the post-action state
            if not terminated:
                still_active.append(i)
        active = still_active
        steps_done += n_exec

    # Timed-out envs (never terminated): state is intact, banked stage is current.
    if trackers is not None:
        for i in active:
            result.rewards[i] = trackers[i].reward()

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

    # FIXED minibatch partition
    order = np.random.permutation(len(records))
    minibatches = [
        [records[i] for i in order[s:s + args.update_minibatch]]
        for s in range(0, len(records), args.update_minibatch)
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
        mu_full = flow_actions_from_noise_wilro(policy.model, batch, x1, args.n_action_steps)
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
    parser.add_argument("--max_episode_steps", type=int, default=0,
                        help="Cap rollout episode length (0 = suite default)")
    parser.add_argument("--exploration_std", type=float, default=0.1,
                        help="Gaussian exploration std in NORMALIZED action units")
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
    args = parser.parse_args()

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
            tid = task_ids[task_cycle % len(task_ids)]
            task_cycle += 1
            group = get_group(tid)
            isid = init_state_cycle[tid] % group.n_init_states
            init_state_cycle[tid] += 1

            res = rollout_group(
                policy, preprocessor, postprocessor, group, isid,
                args.n_action_steps, args.exploration_std, device,
                base_seed=args.seed + 100_000 * it + 1000 * attempts,
                staged_reward=args.staged_reward,
            )
            sr_track[tid].extend([float(s) for s in res.successes])
            n_succ = sum(res.successes)
            summary = {"task_id": tid, "init_state": isid,
                       "successes": n_succ, "of": len(res.successes)}
            if args.staged_reward:
                summary["reward_mean"] = round(float(np.mean(res.rewards)), 3)
            group_summaries.append(summary)

            # Staged reward (dense, in [0,1]) gives partial credit so all-fail
            # groups stop being degenerate; binary success is the fallback.
            reward_vals = (res.rewards if args.staged_reward
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