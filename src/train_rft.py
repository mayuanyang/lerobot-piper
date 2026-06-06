"""
Reward-Filtered fine-Tuning (RFT) for flow-matching VLA policies on LIBERO.

A stable, BC-based self-improvement loop ("RL via supervised learning",
a.k.a. rejection-sampling / reward-weighted fine-tuning):

  1. Roll out the current policy in the LIBERO sim (the same lerobot env your
     `lerobot-eval` uses).
  2. Keep only SUCCESSFUL episodes (LIBERO's sparse success reward).
  3. Window each success into (obs, horizon-action-chunk) samples using the
     EXACT normalized actions the policy emitted — so we reinforce what worked.
  4. Fine-tune with the policy's own flow-matching `compute_loss`.
  5. Repeat, keeping a persistent success buffer across iterations.

Why this and not PPO: a flow-matching policy generates actions by integrating an
ODE, so it has no cheap action log-prob — vanilla policy-gradient RL is a
research-grade build (DPPO / flow-GRPO). RFT optimizes the TRUE objective (task
success) instead of imitation MSE, which is exactly what BC plateaus on, while
reusing your entire training pipeline. Start here; graduate to DPPO if it flattens.

Run (mirrors your lerobot-eval invocation, plus --rft.* flags):

    python src/train_rft.py \
        --policy.path=ISdept/Wilro-ed-138k-l16 \
        --policy.n_action_steps=2 \
        --env.type=libero --env.task=libero_object \
        --eval.batch_size=8 \
        --rft.iterations=50 \
        --rft.rollouts_per_task=16 \
        --rft.updates_per_iter=400 \
        --rft.train_batch_size=16 \
        --rft.lr=1e-5 \
        --rft.output_dir=outputs/rft/wilro_object

Notes / things to VERIFY on the first run (printed at startup):
  - Success is read from info["final_info"]["is_success"] (same as lerobot rollout).
  - policy.select_action() returns the NORMALIZED action (postprocessor un-normalizes
    after) — that is precisely the BC target compute_loss wants, so no re-normalization.
  - Obs are stored raw (post preprocess_observation) and re-run through the SAME
    preprocessor at train time — identical normalization to eval. Images are cached
    as uint8 to keep buffer memory sane.
  - The frozen SmolVLM stays frozen (only requires_grad params are optimized).

Known limitation (intentional, to keep the scaffold runnable):
  - Demo anchoring is NOT wired yet. Collapse is mitigated by (a) a persistent
    success buffer across iterations, (b) low LR, (c) limited updates/iter.
    See _train_on_buffer() for the hook to mix in demo batches / add a KL anchor.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device


# ---------------------------------------------------------------------------
# Config: EvalPipelineConfig (env / policy / eval) + an `rft` block.
# Reuses lerobot's parser so --policy.path / --env.* behave exactly as in eval.
# ---------------------------------------------------------------------------
@dataclass
class RFTParams:
    iterations: int = 50              # collect→train cycles
    rollouts_per_task: int = 16       # episodes to roll out per task per iteration
    updates_per_iter: int = 400       # optimizer steps per iteration
    train_batch_size: int = 16
    lr: float = 1e-5
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    buffer_size: int = 20000          # max (obs, action-chunk) samples retained
    min_buffer_to_train: int = 256    # don't train until the buffer has this many
    save_freq: int = 5                # save a checkpoint every N iterations
    output_dir: str = "outputs/rft/run"


@dataclass
class RFTConfig(EvalPipelineConfig):
    rft: RFTParams = field(default_factory=RFTParams)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _slice_obs(obs: dict, i: int) -> dict:
    """Extract env i's slice of a batched lerobot obs dict, on CPU.

    Images are stored as uint8 (×255) to cut buffer memory 4×; everything else
    is kept as float. `task` is a per-env list of strings.
    """
    out = {}
    for k, v in obs.items():
        if k == "task":
            out[k] = v[i] if isinstance(v, (list, tuple)) else v
        elif isinstance(v, torch.Tensor):
            t = v[i].detach().to("cpu")
            if "image" in k:
                t = (t.clamp(0, 1) * 255).round().to(torch.uint8)
            out[k] = t
        else:
            out[k] = v
    return out


def _episode_to_samples(obs_hist: list[dict], act_hist: list[torch.Tensor], horizon: int):
    """Window one successful episode into (obs_t, action_chunk, is_pad) samples.

    Single-step obs (matches eval — the model treats state.dim()==2 as 1 obs step);
    the action chunk is the next `horizon` executed normalized actions, tail-padded.
    """
    T = len(act_hist)
    samples = []
    for t in range(T):
        chunk = act_hist[t : t + horizon]
        n_real = len(chunk)
        action = torch.stack(chunk, dim=0)                      # (n_real, action_dim)
        if n_real < horizon:
            pad = torch.zeros(horizon - n_real, action.shape[-1], dtype=action.dtype)
            action = torch.cat([action, pad], dim=0)            # (horizon, action_dim)
        is_pad = torch.zeros(horizon, dtype=torch.bool)
        is_pad[n_real:] = True
        samples.append((obs_hist[t], action, is_pad))
    return samples


@torch.no_grad()
def _rft_rollout(env, policy, preprocessor, postprocessor, device, action_dim):
    """One batched rollout. Returns a flat list of (obs_t, action_chunk, is_pad)
    samples harvested ONLY from successful episodes. Non-success episode buffers
    are dropped as soon as they terminate, to bound peak memory."""
    policy.eval()
    policy.reset()
    obs, _ = env.reset()
    B = env.num_envs
    obs_hist = [[] for _ in range(B)]
    act_hist = [[] for _ in range(B)]
    done = np.zeros(B, dtype=bool)
    max_steps = int(env.call("_max_episode_steps")[0])

    samples: list = []
    n_success = 0
    horizon = policy.config.horizon
    step = 0
    while not np.all(done) and step < max_steps:
        obs_lr = preprocess_observation(obs)            # → lerobot keys, float images
        obs_lr = add_envs_task(env, obs_lr)             # inject per-env "task" string
        proc = preprocessor(obs_lr)                     # model-space input (normalized)
        with torch.inference_mode():
            norm_action = policy.select_action(proc)    # (B, action_dim) — NORMALIZED
        env_action = postprocessor(norm_action.clone())  # → env (un-normalized) space

        # Record each still-active env's (raw obs, normalized action) BEFORE step.
        for i in range(B):
            if not done[i]:
                obs_hist[i].append(_slice_obs(obs_lr, i))
                act_hist[i].append(norm_action[i].detach().to("cpu").float())

        obs, _, terminated, truncated, info = env.step(env_action.to("cpu").numpy())

        successes = (
            info["final_info"]["is_success"]
            if "final_info" in info else np.zeros(B, dtype=bool)
        )
        newly_done = (terminated | truncated) & (~done)
        for i in range(B):
            if newly_done[i]:
                if bool(successes[i]):
                    samples.extend(_episode_to_samples(obs_hist[i], act_hist[i], horizon))
                    n_success += 1
                # free this episode's buffers either way
                obs_hist[i], act_hist[i] = [], []
        done = terminated | truncated | done
        step += 1

    return samples, n_success, B


def _collate(batch_samples, preprocessor, device, action_dim):
    """Stack samples → batch, re-run the eval preprocessor on the obs (normalize +
    rename, exactly as at inference), then attach the already-normalized action."""
    obs_keys = [k for k in batch_samples[0][0].keys() if k != "task"]
    batch: dict = {}
    for k in obs_keys:
        vals = []
        for s in batch_samples:
            t = s[0][k]
            if "image" in k:
                t = t.float() / 255.0                   # uint8 → float [0,1]
            vals.append(t)
        batch[k] = torch.stack(vals, dim=0).to(device)
    batch["task"] = [s[0].get("task", "") for s in batch_samples]

    # Normalize obs through the SAME pipeline as eval (action absent → untouched).
    batch = preprocessor(batch)

    actions = torch.stack([s[1] for s in batch_samples], dim=0).to(device)   # (b, H, A)
    is_pad = torch.stack([s[2] for s in batch_samples], dim=0).to(device)    # (b, H)
    batch["action"] = actions                            # already normalized
    batch["action_is_pad"] = is_pad
    batch["action_dim_pad"] = torch.zeros(actions.shape[0], action_dim,
                                          dtype=torch.bool, device=device)
    return batch


def _train_on_buffer(policy, optimizer, buffer, cfg: RFTParams, preprocessor,
                     device, action_dim):
    policy.train()
    trainable = [p for p in policy.parameters() if p.requires_grad]
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if device.type == "cuda" else nullcontext()
    )
    running = 0.0
    for _ in range(cfg.updates_per_iter):
        picks = random.sample(buffer, min(cfg.train_batch_size, len(buffer)))
        # ---- collapse-prevention hooks (extend here) -------------------------
        #   * demo anchoring: append a few demo samples to `picks`
        #   * KL anchor: add a penalty to keep v_theta close to a frozen reference
        # ----------------------------------------------------------------------
        batch = _collate(picks, preprocessor, device, action_dim)
        with autocast_ctx:
            loss, _ = policy.forward(batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
        optimizer.step()
        running += loss.item()
    return running / max(1, cfg.updates_per_iter)


def _save(policy, preprocessor, postprocessor, out_dir: Path, step: int):
    ckpt = out_dir / f"checkpoint-{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    policy.config.training_step = step
    policy.save_pretrained(ckpt)
    preprocessor.save_pretrained(ckpt)
    postprocessor.save_pretrained(ckpt)
    print(f"  [save] {ckpt}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@parser.wrap()
def main(cfg: RFTConfig):
    device = get_safe_torch_device(cfg.policy.device, log=True)
    if cfg.seed is not None:
        set_seed(cfg.seed)
    out_dir = Path(cfg.rft.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Env / policy / processors — identical construction to lerobot-eval.
    envs = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={"rename_observations_processor": {"rename_map": cfg.rename_map}},
    )
    policy.to(device)
    if isinstance(preprocessor, nn.Module):
        preprocessor.to(device)

    action_dim = policy.config.action_dim
    trainable = [p for p in policy.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_frozen = sum(p.numel() for p in policy.parameters() if not p.requires_grad)
    optimizer = torch.optim.Adam(trainable, lr=cfg.rft.lr, weight_decay=cfg.rft.weight_decay)

    # Flatten {suite: {task_id: vec_env}} into a list of (label, env).
    task_envs = [
        (f"{suite}/{tid}", e)
        for suite, by_task in envs.items()
        for tid, e in by_task.items()
    ]

    print("\n" + "=" * 64)
    print("Reward-Filtered Fine-Tuning")
    print("=" * 64)
    print(f"  policy        : {cfg.policy.type}  (path={cfg.policy.pretrained_path})")
    print(f"  trainable     : {n_train:,}   frozen: {n_frozen:,}")
    print(f"  env/task      : {cfg.env.type}/{cfg.env.task}  ({len(task_envs)} task env(s))")
    print(f"  n_action_steps: {policy.config.n_action_steps}  horizon: {policy.config.horizon}")
    print(f"  lr={cfg.rft.lr:.1e}  updates/iter={cfg.rft.updates_per_iter}  "
          f"train_bs={cfg.rft.train_batch_size}  buffer<={cfg.rft.buffer_size}")
    print("=" * 64 + "\n")

    buffer: deque = deque(maxlen=cfg.rft.buffer_size)
    global_step = 0

    for it in range(1, cfg.rft.iterations + 1):
        # ---- 1) Collect successful trajectories --------------------------------
        ep_total, ep_success, new_samples = 0, 0, 0
        for label, env in task_envs:
            n_batches = -(-cfg.rft.rollouts_per_task // env.num_envs)  # ceil
            for _ in range(n_batches):
                samples, n_succ, B = _rft_rollout(
                    env, policy, preprocessor, postprocessor, device, action_dim
                )
                buffer.extend(samples)
                ep_total += B
                ep_success += n_succ
                new_samples += len(samples)
        succ_rate = 100.0 * ep_success / max(1, ep_total)
        print(f"[iter {it:3d}] rollout: {ep_success}/{ep_total} success "
              f"({succ_rate:4.1f}%)  +{new_samples} samples  buffer={len(buffer)}")

        # ---- 2) Reward-filtered BC on the success buffer -----------------------
        if len(buffer) < cfg.rft.min_buffer_to_train:
            print(f"           buffer < {cfg.rft.min_buffer_to_train}; skipping update "
                  f"(policy too weak — start RFT from a stronger checkpoint).")
            continue
        avg_loss = _train_on_buffer(
            policy, optimizer, buffer, cfg.rft, preprocessor, device, action_dim
        )
        global_step += cfg.rft.updates_per_iter
        print(f"           train : {cfg.rft.updates_per_iter} updates  "
              f"avg_loss={avg_loss:.4f}  (global_step={global_step})")

        if it % cfg.rft.save_freq == 0 or it == cfg.rft.iterations:
            _save(policy, preprocessor, postprocessor, out_dir, global_step)

    print("\nRFT complete.")


if __name__ == "__main__":
    main()
