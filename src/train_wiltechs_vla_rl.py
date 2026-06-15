"""
GRPO online RL for WiltechsVLA on LIBERO — SimpleVLA-RL recipe adapted to a
flow-matching policy (arXiv:2509.09674).

SimpleVLA-RL runs GRPO on OpenVLA-OFT's discrete action-token log-probs. A
flow-matching policy has no token likelihoods, so this script uses the
noise-conditioned Gaussian formulation (DPPO/ReinFlow-style):

    x1 ~ N(0, I)                       (initial flow noise, STORED as a latent)
    mu = flow_ODE(s, x1)               (deterministic 5-step integration)
    a  = mu[:n_exec] + sigma * eps     (executed chunk; sigma = exploration_std)
    log pi(a | s, x1) = sum log N(a; mu, sigma^2)   -> exact, differentiable

which makes PPO ratios computable and lets the SimpleVLA-RL recipe carry over:
  - group sampling: G rollouts per (task, init_state), reward = success (0/1)
  - group-relative advantage  A = (R - mean) / (std + eps)
  - dynamic sampling: drop groups where all rollouts succeed or all fail
  - clip-higher (eps_low=0.2, eps_high=0.28), NO KL term (DAPO-style)
  - lr 5e-6, grad clip 1.0

Differences from the paper, by necessity or pragmatism:
  - continuous Gaussian head instead of token sampling temperature; exploration
    is controlled by --exploration_std (normalized action units, MEAN_STD space)
  - single-GPU synchronous rollouts (no veRL); LIBERO envs stepped in-process
  - all model dropout is zeroed at load so rollout mu and update mu match
    (exploration comes from sigma, not dropout)

Usage (Colab, resume from the 34k SFT checkpoint):
    python src/train_wiltechs_vla_rl.py \
        --policy_path ISdept/wiltechs-vla-34k \
        --env_task libero_goal --task_ids 3 8 9 \
        --output_dir outputs/rl/wiltechs_goal \
        --group_size 8 --n_action_steps 8 --exploration_std 0.1

Start RL from a checkpoint with NON-ZERO success on the chosen tasks: the paper
shows outcome-reward RL gets zero gradient from 0%-success tasks (its failure
mode), and dynamic sampling will simply skip them.
"""

from __future__ import annotations

import os
# MuJoCo/LIBERO offscreen rendering: force GPU (EGL) before robosuite/mujoco is
# imported. The GL backend is locked in at first import, so this MUST run above
# the lerobot/LIBERO imports below. setdefault lets a launch-time override win.
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
# Pure GRPO math (no heavy deps — unit-testable without lerobot/libero)
# ---------------------------------------------------------------------------

def gaussian_logp_per_step(actions: torch.Tensor, mu: torch.Tensor, sigma: float) -> torch.Tensor:
    """Log N(actions; mu, sigma^2 I) summed over action dims, kept per timestep.

    actions/mu: (B, T, D) -> returns (B, T). Summing over D (not T) gives
    per-timestep ratios, which clip more gracefully than one ratio over the
    whole 56-dim chunk.
    """
    var = sigma * sigma
    logp = -0.5 * ((actions - mu) ** 2) / var - math.log(sigma) - 0.5 * math.log(2.0 * math.pi)
    return logp.sum(dim=-1)


def grpo_group_advantages(rewards: list[float], eps: float = 1e-4) -> Optional[list[float]]:
    """Group-relative advantages; None if the group is degenerate (all same
    outcome) and should be dropped — the paper's Dynamic Sampling (eq. 10)."""
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
    """PPO-clip objective with decoupled clip range (DAPO clip-higher), no KL.

    logp_new/logp_old: (B, T) per-timestep log-probs; advantages: (B,) per
    trajectory, broadcast over the chunk's timesteps. `weights` (B,) optionally
    weights each chunk (used for per-trajectory length normalization, paper
    eq. 11 — otherwise long failed episodes dominate the batch).
    """
    # Clamp the log-ratio before exp: with sigma=0.1 the Gaussian logp is so
    # sensitive that outlier samples can overflow exp() to inf, and for adv<0
    # the unclipped branch is UNBOUNDED below -> inf loss -> NaN params
    # (observed). Clamping keeps everything finite; the gradient information
    # past |log r|=20 is meaningless anyway.
    log_ratio = (logp_new - logp_old).clamp(-20.0, 20.0)
    ratio = torch.exp(log_ratio)
    adv = advantages.unsqueeze(-1)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high) * adv
    per_elem = torch.minimum(unclipped, clipped)
    if dual_clip is not None:
        # Dual-clip PPO (Ye et al. 2020): the standard clip never bounds the
        # pessimistic branch for negative advantages; cap it at dual_clip*adv.
        per_elem = torch.where(adv < 0, torch.maximum(per_elem, dual_clip * adv), per_elem)
    if weights is None:
        loss = -per_elem.mean()
    else:
        w = weights.unsqueeze(-1)
        loss = -(per_elem * w).sum() / (w.sum() * per_elem.shape[-1])
    with torch.no_grad():
        # k3 estimator of KL(old || new): non-negative, low variance.
        kl_elems = (ratio - 1.0) - log_ratio
        # Mean is dominated by single outliers (one sample at the -20 clamp
        # contributes ~19 — observed tripping early-stop at bulk kl 0.003).
        # Those outliers are already gradient-dead via clipping; the MEDIAN
        # measures bulk policy movement and is what the early stop should see.
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
    # Staged (dense) reward per env, in [0,1]; filled only when staged_reward is
    # on. Defaults to 0.0; success-terminations are set to 1.0 (env auto-resets,
    # so the final sim state is gone — but full success ⇒ all conjuncts placed).
    rewards: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Batched observation -> model batch (mirrors lerobot_eval's rollout exactly)
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
# Environment pool — raw LiberoEnv instances, lockstep stepping
# ---------------------------------------------------------------------------

class TaskEnvGroup:
    """G LiberoEnv instances of ONE task. init_state_id is set per reset so a
    GRPO group shares the same initial state (the paper repeats each input G
    times). Stepped serially; policy inference is batched across members."""

    def __init__(self, suite, suite_name: str, task_id: int, group_size: int,
                 expected_cams: Optional[list[str]] = None,
                 max_episode_steps: int = 0):
        from lerobot.envs.libero import LiberoEnv

        self.task_id = task_id
        # Use the installed lerobot's DEFAULT camera naming — verified correct
        # on the Colab build (2-cam eval print). Forcing a camera_name_mapping
        # here broke a correct setup once (2026-06-12); instead we VERIFY the
        # obs keys against the policy's camera list below and hard-error on
        # mismatch, since _encode_images silently drops missing cameras.
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
                    f"expected camera(s) {missing} (policy cameras: {expected_cams}). "
                    f"The model would SILENTLY run without them. Pass a matching "
                    f"camera_name_mapping to LiberoEnv in TaskEnvGroup for this "
                    f"lerobot version."
                )
            print(f"[rl] camera check OK — env provides {sorted(got)}, "
                  f"policy expects {expected_cams}")
        self.task_description = self.envs[0].task_description
        suite_default = self.envs[0]._max_episode_steps
        self.max_steps = min(suite_default, max_episode_steps) if max_episode_steps else suite_default
        if self.max_steps < suite_default:
            print(f"[rl] task {task_id}: capping episode length {suite_default} -> {self.max_steps} "
                  f"(rollouts past {self.max_steps} steps count as failure)")
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
# Rollout: one GRPO group (G episodes, same task + init state)
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
    # Staged-reward trackers must be built AFTER reset (objects at init pose) so
    # the lift baseline is the rest height.
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

        x1 = torch.randn(len(active), horizon, action_dim, device=device)
        with autocast_ctx:
            mu_full = policy.model.flow_actions_from_noise(batch, x1)
        mu = mu_full[:, :n_exec].float()
        actions = mu + sigma * torch.randn_like(mu)
        logp_old = gaussian_logp_per_step(actions, mu, sigma)        # (B, n_exec)

        env_actions = postprocessor(actions.reshape(len(active) * n_exec, action_dim))
        env_actions = env_actions.reshape(len(active), n_exec, action_dim).numpy()

        for j, i in enumerate(list(active)):
            result.records_per_env[i].append(ChunkRecord(
                pixels={c: np.ascontiguousarray(v) for c, v in cur_obs[j]["pixels"].items()},
                agent_pos=np.asarray(cur_obs[j]["agent_pos"], dtype=np.float32),
                task=group.task_description,
                x1=x1[j].cpu().numpy().astype(np.float32),
                action=actions[j].cpu().numpy().astype(np.float32),
                logp_old=logp_old[j].cpu().numpy().astype(np.float32),
            ))

        # Execute the chunk step-by-step; an env that terminates mid-chunk
        # stops (LiberoEnv auto-resets on termination, so never step it again).
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
                    trackers[i].update()   # bank stage on the post-action state
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
# GRPO update over collected records
# ---------------------------------------------------------------------------

def grpo_update(
    policy, preprocessor, optimizer, records: list[ChunkRecord], args, device,
) -> dict:
    policy.model.train()   # enables DiT gradient checkpointing; dropout is zeroed at load
    # The ResNet robot encoder contains BatchNorm: in train mode it switches to
    # BATCH statistics (and updates running stats), but rollout ran in eval mode
    # with RUNNING stats — the mu mismatch collapses importance ratios
    # (observed: iter-0 ratio 0.38, clip_frac 0.97). Keep the CNN in eval; the
    # model-level train() flag stays on for the DiT checkpointing gate.
    if getattr(policy.model, "robot_visual_encoder", None) is not None:
        policy.model.robot_visual_encoder.eval()
    horizon = policy.config.horizon

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False)
    )
    trainable = [p for p in policy.model.parameters() if p.requires_grad]

    # FIXED minibatch partition, reused by the pre-pass and all epochs:
    # identical batch composition => identical kernel shapes => exact logp
    # baselines (bf16 reduction-order noise between DIFFERENT compositions
    # previously zeroed 81% of gradients via spurious clipping).
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
        with autocast_ctx:
            mu_full = policy.model.flow_actions_from_noise(batch, x1)
        mu = mu_full[:, :args.n_action_steps].float()
        return gaussian_logp_per_step(actions, mu, args.exploration_std)

    # ── Pre-pass: logp_old for ALL minibatches BEFORE any optimizer step ──
    # This is PPO's trust region: ratios during the update measure ONLY genuine
    # policy movement within this iteration, and clipping caps it. (Computing
    # logp_old lazily at first use removed the anchor — ~110 sequential steps
    # on one rollout batch moved mu by ~0.2 normalized units, drift=201.)
    logp_old_list: list[torch.Tensor] = []
    agg = {"loss": 0.0, "ratio_mean": 0.0, "ratio_max": 0.0, "clip_frac": 0.0,
           "rollout_drift": 0.0, "n_minibatches": 0}
    with torch.no_grad():
        for mb in minibatches:
            batch, x1, actions, _adv, _w = mb_tensors(mb)
            lp = forward_logp(batch, x1, actions)
            logp_old_list.append(lp)
            # Diagnostic only: recompute-vs-rollout logp gap (bf16/batch-shape
            # noise, since theta is unchanged here). Expect ~0.1-0.5.
            lp_roll = torch.from_numpy(np.stack([r.logp_old for r in mb])).to(device)
            agg["rollout_drift"] += float((lp - lp_roll).abs().mean())

    # ── Clipped updates ──
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

            # Trust-region early stop on the MEDIAN per-sample KL: once the
            # BULK of the policy has moved target_kl from the pre-pass anchor,
            # further steps on this rollout batch are off-policy extrapolation.
            # (Mean k3 is outlier-dominated and fired at bulk kl ~0.003.)
            if stats["approx_kl_median"] > args.target_kl:
                agg["stopped_early"] = 1
                print(f"[rl]   early stop at minibatch {agg['n_minibatches']}: "
                      f"median kl {stats['approx_kl_median']:.4f} > {args.target_kl} "
                      f"(mean kl {stats['approx_kl']:.3f})")
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
    """Load the SFT checkpoint the same way lerobot_eval does, then strip every
    stochastic training behavior so rollout mu == update mu (exploration must
    come from sigma alone, or the importance ratios are garbage)."""
    from models.wiltechs_vla.wiltechs_vla_policy import WiltechsVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = WiltechsVLAPolicy.from_pretrained(args.policy_path)
    policy.config.pretrained_path = args.policy_path

    policy.config.vision_dropout_prob = 0.0
    policy.config.vision_kv_dropout_prob = 0.0
    n_zeroed = 0
    for m in policy.model.modules():
        if isinstance(m, torch.nn.Dropout) and m.p > 0:
            m.p = 0.0
            n_zeroed += 1
    print(f"[rl] zeroed {n_zeroed} Dropout modules + vision/vision-KV dropout configs")

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
    parser = argparse.ArgumentParser(description="GRPO RL for WiltechsVLA on LIBERO (SimpleVLA-RL recipe)")
    parser.add_argument("--policy_path", type=str, required=True,
                        help="SFT checkpoint (local dir or HF id, e.g. ISdept/wiltechs-vla-34k). "
                             "Must have non-zero success on the chosen tasks.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--env_task", type=str, default="libero_goal",
                        help="LIBERO suite: libero_spatial/object/goal/libero_10")
    parser.add_argument("--task_ids", type=int, nargs="*", default=None,
                        help="Bench task ids to train on (default: all 10). Focus on weak-but-not-zero "
                             "tasks, e.g. goal: --task_ids 3 8 9")
    parser.add_argument("--group_size", type=int, default=8, help="G rollouts per (task, init_state)")
    parser.add_argument("--groups_per_iter", type=int, default=2,
                        help="Groups collected per update iteration (round-robin over task_ids)")
    parser.add_argument("--n_action_steps", type=int, default=8,
                        help="Executed chunk length between replans (paper uses 8 on LIBERO)")
    parser.add_argument("--staged_reward", action="store_true",
                        help="Use a dense staged reward (reach/grasp/lift/place per goal object, "
                             "mean over conjuncts) instead of binary success. Gives all-fail GRPO "
                             "groups within-group variance so grasp-fumble and compound 'put both' "
                             "tasks produce gradient. True task success is still logged separately.")
    parser.add_argument("--max_episode_steps", type=int, default=0,
                        help="Cap rollout episode length (0 = use the LIBERO suite default, "
                             "e.g. 520 for libero_10). Shortening speeds rollout but truncates "
                             "long tasks to a failure, so only set below the suite default "
                             "if those tasks usually finish sooner.")
    parser.add_argument("--exploration_std", type=float, default=0.1,
                        help="Gaussian exploration std in NORMALIZED action units (MEAN_STD space)")
    parser.add_argument("--rl_iterations", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--clip_low", type=float, default=0.2)
    parser.add_argument("--clip_high", type=float, default=0.28, help="DAPO clip-higher")
    parser.add_argument("--update_epochs", type=int, default=1)
    parser.add_argument("--update_minibatch", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--target_kl", type=float, default=0.05,
                        help="Early-stop the update phase once the MEDIAN per-timestep "
                             "approx-KL to the pre-pass policy exceeds this (bulk trust "
                             "region; robust to gradient-dead outlier samples).")
    parser.add_argument("--dual_clip", type=float, default=3.0,
                        help="Dual-clip PPO bound on the negative-advantage branch "
                             "(Ye et al. 2020). Prevents unbounded loss as ratios grow.")
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--save_freq", type=int, default=20, help="Iterations between checkpoints")
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

    # Cameras the policy was trained on, e.g. "observation.images.wrist_image"
    # -> env pixel key "wrist_image". Verified against actual env obs on first
    # env build (a mismatch silently drops cameras inside _encode_images).
    expected_cams = [
        k.split(".")[-1] for k in policy.config.cameras_for_vision_state_concat
    ]

    # Env pool: one TaskEnvGroup per task, built lazily, kept alive (MuJoCo env
    # construction is expensive). With many tasks this holds many sim instances;
    # restrict --task_ids if RAM becomes a problem.
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

        # ── Collect groups (dynamic sampling: keep only mixed-outcome groups) ──
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
            summary = {"task_id": tid, "init_state": isid, "successes": n_succ,
                       "of": len(res.successes)}
            if args.staged_reward:
                summary["reward_mean"] = round(float(np.mean(res.rewards)), 3)
            group_summaries.append(summary)

            # Staged reward (dense, in [0,1]) gives partial credit so all-fail
            # groups stop being degenerate; binary success is the fallback.
            reward_vals = (res.rewards if args.staged_reward
                           else [1.0 if s else 0.0 for s in res.successes])
            adv = grpo_group_advantages(reward_vals)
            if adv is None:
                continue  # all-success or all-fail: zero advantage, skip (paper eq. 10)
            kept += 1
            for env_i, env_records in enumerate(res.records_per_env):
                for r in env_records:
                    r.advantage = adv[env_i]
                    # Per-trajectory normalization (paper eq. 11): each episode
                    # contributes equally, so long failed episodes don't dominate.
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
