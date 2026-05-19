#!/usr/bin/env python3
"""
LIBERO Simulation Benchmark for TransformerFlowMatchingPolicy.

Usage:
    python src/benchmark_libero.py \
        --checkpoint ./outputs/libero_run/checkpoint-50000 \
        --suites libero_spatial libero_object \
        --num_rollouts 20 \
        --max_steps 300

For headless Linux (no display):
    export MUJOCO_GL=egl
    Xvfb :99 -screen 0 1400x900x24 &
    export DISPLAY=:99
    python src/benchmark_libero.py --checkpoint ... --headless

Requirements:
    pip install libero robosuite==1.4.1 bddl mujoco
    pip install pyvirtualdisplay   # only needed for --headless
"""

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Robosuite compatibility patches (required for robosuite 1.4.x + LIBERO)
# ---------------------------------------------------------------------------

def _apply_robosuite_patches():
    import json as _json
    import robosuite.controllers as controllers
    import robosuite.robots as robots
    from robosuite.robots.robot import Robot

    def _load_controller_config(custom_fpath=None, default_controller=None):
        if custom_fpath is not None:
            with open(custom_fpath) as f:
                return _json.load(f)
        return {"type": default_controller} if default_controller else {}

    if not hasattr(controllers, "load_controller_config"):
        setattr(controllers, "load_controller_config", _load_controller_config)

    if not hasattr(robots, "SingleArm"):
        setattr(robots, "SingleArm", Robot)


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def img_to_tensor(img: np.ndarray) -> torch.Tensor:
    """(H, W, C) uint8 → (C, H, W) float32 in [0, 1].

    Applies a 180° flip (`[::-1, ::-1]`) to match the `lerobot/libero` dataset
    convention. MuJoCo's offscreen renderer returns frames with Y-axis inverted
    (OpenGL origin), and the dataset's conversion script flipped them for human
    readability — without this flip, the policy sees every frame upside-down.
    """
    img = img[::-1, ::-1]
    return torch.from_numpy(img.copy()).float().div_(255.0).permute(2, 0, 1)


def build_state(obs: dict) -> np.ndarray:
    """
    Map LIBERO env obs → lerobot/libero observation.state (8-dim).

      [0:6]  robot0_joint_pos[:6]   — arm joints (rad)
      [6:8]  robot0_gripper_qpos    — gripper fingers (m)

    Two coordinate-frame corrections relative to robosuite 1.4.1:

    - joint_pos[3] is sign-flipped: the lerobot/libero dataset was built with
      an older robosuite that defines joint 4's axis in the opposite direction.
      Dataset values are always positive (≈+2.97 rad at rest); robosuite 1.4.1
      reports ≈ −π. Negating restores alignment.

    - joint_pos[5] has a 3π/4 origin offset: dataset values are centered around
      0 (mean ≈ −0.13, range [−1.84, +1.39]), whereas robosuite 1.4.1 reports
      ≈ +2.23 at rest. Subtracting 3π/4 shifts the physical wrist-flex range
      [0.52, 3.75] back into the dataset's recorded frame.
    """
    joint_pos = np.array(obs["robot0_joint_pos"], dtype=np.float64)
    gripper   = np.array(obs["robot0_gripper_qpos"], dtype=np.float64)
    joint_pos[3] = -joint_pos[3]
    joint_pos[5] = joint_pos[5] - 3 * np.pi / 4
    return np.concatenate([joint_pos[:6], gripper[:2]]).astype(np.float32)


def obs_to_frame(obs: dict, cam_key_map: dict[str, str]) -> dict:
    frame = {"state": build_state(obs)}
    for libero_key, policy_key in cam_key_map.items():
        frame[policy_key] = img_to_tensor(obs[libero_key])
    return frame


def _sanity_check_state_distribution(
    benchmark_dict: dict,
    suite_name: str,
    image_size: int,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    control_freq: int = 10,
    z_threshold: float = 5.0,
) -> None:
    """Reset one LIBERO env and verify build_state lands within the training distribution.

    Raises RuntimeError if any state dim has |z-score| > z_threshold. A large
    z-score on a single dim is the signature of a silent coordinate-frame flip
    between the dataset's robosuite version and the live one (e.g. the joint_pos[3]
    flip already handled in build_state). Catching this here prevents a wasted
    multi-hour run that ends in 0% success.
    """
    from libero.libero.envs import OffScreenRenderEnv

    task_suite = benchmark_dict[suite_name]()
    env = OffScreenRenderEnv(**{
        "bddl_file_name": task_suite.get_task_bddl_file_path(0),
        "camera_heights": image_size,
        "camera_widths":  image_size,
        "control_freq":   control_freq,
    })
    obs = env.reset()
    env.close()

    state = build_state(obs)
    state_t = torch.tensor(state, dtype=torch.float32, device=state_mean.device)
    z = (state_t - state_mean) / state_std

    mean_cpu = state_mean.cpu().numpy()
    std_cpu  = state_std.cpu().numpy()
    z_cpu    = z.cpu().numpy()

    print(f"\nState-distribution sanity check (suite={suite_name}, task 0, reset frame):")
    print("  dim |   raw       |   mean      |   std     |  z-score")
    print("  ----+-------------+-------------+-----------+---------")
    for i in range(len(state)):
        flag = "  <-- OUT OF RANGE" if abs(z_cpu[i]) > z_threshold else ""
        print(f"  {i:3d} | {state[i]:+11.4f} | {mean_cpu[i]:+11.4f} | {std_cpu[i]:9.4f} | {z_cpu[i]:+7.2f}{flag}")

    max_abs_z = float(abs(z_cpu).max())
    if max_abs_z > z_threshold:
        bad_dims = [i for i, zi in enumerate(z_cpu) if abs(zi) > z_threshold]
        raise RuntimeError(
            f"State distribution sanity check FAILED: max |z| = {max_abs_z:.2f} > {z_threshold:.1f} "
            f"on dim(s) {bad_dims}. This usually means a coordinate-frame mismatch between the "
            f"training dataset and the live robosuite env (e.g. a sign-flipped joint axis like the "
            f"joint_pos[3] flip in build_state). Fix build_state or investigate before continuing — "
            f"the policy will not behave correctly otherwise. Pass --skip_sanity_check to bypass."
        )
    print(f"\nSanity check passed (max |z| = {max_abs_z:.2f} ≤ {z_threshold:.1f}).")


def build_batch(
    obs_buffer: deque,
    task_description: str,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    cam_key_map: dict[str, str],
    device: torch.device,
) -> dict:
    raw_state = torch.stack(
        [torch.tensor(o["state"], dtype=torch.float32) for o in obs_buffer]
    ).unsqueeze(0).to(device)           # (1, n_obs_steps, state_dim)

    norm_state = (raw_state - state_mean) / state_std

    batch: dict = {
        "observation.state": norm_state,
        "task_description": [task_description],
    }
    for policy_key in cam_key_map.values():
        batch[policy_key] = obs_buffer[-1][policy_key].unsqueeze(0).to(device)
    return batch


# ---------------------------------------------------------------------------
# Episode rollout
# ---------------------------------------------------------------------------

def run_episode(
    env,
    task_description: str,
    seed: int,
    policy,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    cam_key_map: dict[str, str],
    n_obs_steps: int,
    n_action_steps: int,
    max_steps: int,
    device: torch.device,
    step_logger=None,
) -> bool:
    """Run one rollout. Returns True on task success.

    If `step_logger` is provided, it is called once per environment step *before*
    `env.step(action)` with the signature:

        step_logger(step_idx, raw_obs, state, action_norm, action_unnorm)

    where state/action_* are 1-D numpy arrays. Used by `--actions_log` and
    `--inspect_only` to record what the policy actually produced.
    """
    obs = env.reset()

    # Set seed for reproducible initial conditions within the episode
    try:
        env.env.seed(seed)
    except AttributeError:
        pass

    first_frame  = obs_to_frame(obs, cam_key_map)
    obs_buffer   = deque([first_frame] * n_obs_steps, maxlen=n_obs_steps)
    action_queue: deque = deque()

    for step_idx in range(max_steps):
        if not action_queue:
            batch = build_batch(
                obs_buffer, task_description,
                state_mean, state_std, cam_key_map, device,
            )
            with torch.no_grad():
                actions_norm = policy.predict_action_chunk(batch)  # (1, horizon, action_dim)

            a_norm_np   = actions_norm[0, :n_action_steps].cpu().numpy()
            a_unnorm_np = (actions_norm * action_std + action_mean)[0, :n_action_steps].cpu().numpy()
            for a_n, a_u in zip(a_norm_np, a_unnorm_np):
                action_queue.append((a_n, a_u))

        a_norm, a_unnorm = action_queue.popleft()
        if step_logger is not None:
            step_logger(step_idx, obs, obs_buffer[-1]["state"], a_norm, a_unnorm)
        obs, _reward, done, _info = env.step(a_unnorm)
        obs_buffer.append(obs_to_frame(obs, cam_key_map))

        if done:
            return True

    return False


# ---------------------------------------------------------------------------
# Deep-inspect: one rollout, full per-step dump
# ---------------------------------------------------------------------------

def _run_inspect_episode(
    benchmark_dict: dict,
    suite_name: str,
    task_id: int,
    seed: int,
    image_size: int,
    policy,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    cam_key_map: dict[str, str],
    n_obs_steps: int,
    n_action_steps: int,
    max_steps: int,
    device: torch.device,
    output_path: Path,
    control_freq: int = 10,
) -> None:
    """Run a single rollout and dump everything we can record per step.

    Captures both camera frames (uint8 HWC), the 8-dim observation state, and
    the normalized + unnormalized action vectors. Frames are saved with the
    same 180° flip `img_to_tensor` applies before feeding the policy — i.e. in
    the lerobot/libero dataset's right-side-up convention. This means the saved
    arrays match *what the policy saw*, not raw MuJoCo OpenGL output.
    """
    from libero.libero.envs import OffScreenRenderEnv

    if suite_name not in benchmark_dict:
        raise ValueError(f"Unknown suite '{suite_name}'.")

    task_suite = benchmark_dict[suite_name]()
    task      = task_suite.get_task(task_id)
    bddl_file = task_suite.get_task_bddl_file_path(task_id)
    task_desc = task.language

    env = OffScreenRenderEnv(**{
        "bddl_file_name": bddl_file,
        "camera_heights": image_size,
        "camera_widths":  image_size,
        "control_freq":   control_freq,
    })

    records: list = []

    def step_logger(step_idx, raw_obs, state, a_norm, a_unnorm):
        # Match the 180° flip img_to_tensor applies before feeding the policy,
        # so the saved frames are in the same orientation the policy reasons over.
        agent = np.asarray(raw_obs["agentview_image"],          dtype=np.uint8)[::-1, ::-1].copy()
        wrist = np.asarray(raw_obs["robot0_eye_in_hand_image"], dtype=np.uint8)[::-1, ::-1].copy()
        records.append({
            "step":          step_idx,
            "state":         np.asarray(state,    dtype=np.float32),
            "action_norm":   np.asarray(a_norm,   dtype=np.float32),
            "action_unnorm": np.asarray(a_unnorm, dtype=np.float32),
            "agentview":     agent,
            "wrist":         wrist,
        })

    print(f"\nInspect-only: suite={suite_name} task={task_id} ({task.name}) seed={seed}")
    success = run_episode(
        env=env,
        task_description=task_desc,
        seed=seed,
        policy=policy,
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
        cam_key_map=cam_key_map,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_steps=max_steps,
        device=device,
        step_logger=step_logger,
    )
    env.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        suite=np.array(suite_name),
        task_name=np.array(task.name),
        task_description=np.array(task_desc),
        success=np.array(success),
        seed=np.array(seed),
        steps=np.array([r["step"] for r in records], dtype=np.int32),
        states=np.stack([r["state"] for r in records]),
        actions_norm=np.stack([r["action_norm"] for r in records]),
        actions_unnorm=np.stack([r["action_unnorm"] for r in records]),
        agentview=np.stack([r["agentview"] for r in records]),
        wrist=np.stack([r["wrist"] for r in records]),
    )
    print(
        f"Inspect episode → {output_path}  "
        f"(success={success}, steps={len(records)})"
    )


# ---------------------------------------------------------------------------
# Dataset-replay: step recorded teleop actions through the live env (no policy)
# ---------------------------------------------------------------------------

def _meta_row(meta_obj, index: int) -> dict:
    """Fetch row `index` from a lerobot meta container as a plain dict.

    lerobot 0.4.x stores `meta.episodes` and `meta.tasks` as pandas DataFrames
    (indexed by episode_index / task_index). Earlier versions used dicts or
    lists of dicts. This helper papers over the three shapes.
    """
    # pandas DataFrame
    if hasattr(meta_obj, "loc") and hasattr(meta_obj, "iloc"):
        try:
            row = meta_obj.loc[index]
        except (KeyError, TypeError):
            row = meta_obj.iloc[index]
        return row.to_dict() if hasattr(row, "to_dict") else dict(row)
    # dict (int- or str-keyed)
    if isinstance(meta_obj, dict):
        entry = meta_obj.get(index, meta_obj.get(str(index)))
        if entry is None:
            raise KeyError(index)
        return dict(entry) if not isinstance(entry, dict) else entry
    # list / sequence
    return dict(meta_obj[index])


def _episode_frame_range(ds, episode_index: int) -> tuple[int, int]:
    """Return [from, to) global frame indices for an episode.

    Tries `ds.meta.episodes` (lerobot 0.4.x), falling back to a scan of the
    underlying HF dataset's `episode_index` column. The older
    `ds.episode_data_index` attribute is gone in 0.4.x.
    """
    try:
        episodes_meta = ds.meta.episodes
        n_eps = len(episodes_meta)
        if not 0 <= episode_index < n_eps:
            raise IndexError(f"episode_index={episode_index} not in [0, {n_eps})")
        from_idx = sum(int(_meta_row(episodes_meta, i)["length"]) for i in range(episode_index))
        to_idx   = from_idx + int(_meta_row(episodes_meta, episode_index)["length"])
        return from_idx, to_idx
    except (AttributeError, KeyError, IndexError, TypeError) as e:
        print(f"  meta.episodes lookup failed ({e}); scanning hf_dataset…")
        hf_ds = getattr(ds, "hf_dataset", None) or getattr(ds, "dataset", None)
        if hf_ds is None:
            raise RuntimeError(
                "Could not find episode boundaries: no `meta.episodes` and no "
                "`hf_dataset`/`dataset` attribute on the LeRobotDataset."
            )
        ep_col = np.asarray(hf_ds["episode_index"])
        mask = ep_col == episode_index
        if not mask.any():
            raise RuntimeError(
                f"Episode {episode_index} not found in dataset "
                f"(available: 0..{int(ep_col.max())})."
            )
        idxs = np.where(mask)[0]
        return int(idxs[0]), int(idxs[-1]) + 1


# ---------------------------------------------------------------------------
# Dataset-replay implementation
# ---------------------------------------------------------------------------

def _replay_dataset_episode(
    benchmark_dict: dict,
    dataset_id: str,
    episode_index: int,
    image_size: int,
    output_path: Path,
    control_freq: int = 10,
    action_repeat: int = 1,
) -> None:
    """Replay one recorded episode through the live env, bypassing the policy.

    Isolates env / action-space / coordinate-frame mismatches from policy bugs:
    if a known-good teleop trajectory fails to reproduce the recorded states
    here, the issue is the env, not the model. The dump includes both the
    dataset's `recorded_states` and the live env's `live_states` at each step
    so divergence is plottable.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from libero.libero.envs import OffScreenRenderEnv

    print(f"\nLoading dataset {dataset_id} for replay …")
    ds = LeRobotDataset(dataset_id, revision="main")

    from_idx, to_idx = _episode_frame_range(ds, episode_index)
    n_frames = to_idx - from_idx
    print(f"Episode {episode_index}: frames [{from_idx}, {to_idx}), length={n_frames}")

    actions:    list = []
    rec_states: list = []
    task_idx  = None
    task_desc = None
    for i in range(from_idx, to_idx):
        s = ds[i]
        a = s["action"]; st = s["observation.state"]
        actions.append(a.numpy()    if hasattr(a,  "numpy") else np.asarray(a))
        rec_states.append(st.numpy() if hasattr(st, "numpy") else np.asarray(st))
        if task_idx is None and "task_index" in s:
            task_idx = int(s["task_index"])
        if task_desc is None:
            # lerobot 0.4.x samples typically carry the description string under "task"
            for key in ("task", "language_instruction", "task_description", "language"):
                if key in s and isinstance(s[key], str) and s[key]:
                    task_desc = s[key]
                    break
    actions    = np.stack(actions).astype(np.float32)
    rec_states = np.stack(rec_states).astype(np.float32)

    # Fall back to scanning meta.tasks if the sample didn't include the description
    if task_desc is None and task_idx is not None:
        try:
            task_entry = _meta_row(ds.meta.tasks, task_idx)
        except Exception:
            task_entry = {}
        for key in ("task", "description", "language_instruction", "language", "instruction"):
            v = task_entry.get(key)
            if isinstance(v, str) and v:
                task_desc = v
                break
        if task_desc is None:
            for v in task_entry.values():
                if isinstance(v, str) and v:
                    task_desc = v
                    break

    if not isinstance(task_desc, str) or not task_desc:
        raise RuntimeError(
            f"Could not resolve task description for episode {episode_index} "
            f"(task_index={task_idx}). Sample keys seen: "
            f"{list(ds[from_idx].keys()) if from_idx < to_idx else 'none'}"
        )
    print(f"Task description: {task_desc!r}")

    matched = None
    skipped: list = []
    for suite_name, suite_cls in benchmark_dict.items():
        try:
            suite = suite_cls()
        except Exception as e:
            # e.g. `libero_100` is a meta-suite that fails to instantiate
            skipped.append((suite_name, type(e).__name__))
            continue
        for tid in range(suite.get_num_tasks()):
            task = suite.get_task(tid)
            if task.language.strip() == task_desc.strip():
                matched = (suite_name, tid, suite, task)
                break
        if matched:
            break

    if skipped:
        print(f"  (skipped un-instantiable suites: {skipped})")

    if matched is None:
        raise RuntimeError(
            f"Could not find suite/task matching description: {task_desc!r}. "
            f"Available suites: {list(benchmark_dict.keys())}"
        )

    suite_name, tid, task_suite, task = matched
    print(f"Matched: suite={suite_name} task_id={tid} ({task.name})")

    env = OffScreenRenderEnv(**{
        "bddl_file_name": task_suite.get_task_bddl_file_path(tid),
        "camera_heights": image_size,
        "camera_widths":  image_size,
        "control_freq":   control_freq,
    })
    obs = env.reset()
    print(f"Replay: control_freq={control_freq} Hz, action_repeat={action_repeat}")

    live_states:      list = []
    agentview_frames: list = []
    wrist_frames:     list = []
    success_at = None

    for step_idx in range(len(actions)):
        live_states.append(build_state(obs))
        agentview_frames.append(np.asarray(obs["agentview_image"],          dtype=np.uint8)[::-1, ::-1].copy())
        wrist_frames.append(    np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8)[::-1, ::-1].copy())

        done = False
        for _ in range(action_repeat):
            obs, _r, done, _info = env.step(actions[step_idx])
            if done:
                break
        if done:
            success_at = step_idx + 1
            break

    env.close()

    live_states_np  = np.stack(live_states)
    agentview_np    = np.stack(agentview_frames)
    wrist_np        = np.stack(wrist_frames)
    actions_used    = actions[:len(live_states_np)]
    rec_states_used = rec_states[:len(live_states_np)]

    success = success_at is not None
    print(f"Replay finished: success={success}, env steps={len(live_states_np)}")
    if success:
        print(f"  succeeded at step {success_at}/{len(actions)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        suite=np.array(suite_name),
        task_name=np.array(task.name),
        task_description=np.array(task_desc),
        episode_index=np.array(episode_index),
        success=np.array(success),
        actions=actions_used,
        recorded_states=rec_states_used,
        live_states=live_states_np,
        agentview=agentview_np,
        wrist=wrist_np,
    )
    print(f"Replay → {output_path}")


# ---------------------------------------------------------------------------
# Suite evaluation
# ---------------------------------------------------------------------------

def evaluate_suite(
    suite_name: str,
    benchmark_dict: dict,
    image_size: int,
    num_rollouts: int,
    max_steps: int,
    policy,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    cam_key_map: dict[str, str],
    n_obs_steps: int,
    n_action_steps: int,
    device: torch.device,
    actions_log: list | None = None,
    control_freq: int = 10,
) -> dict:
    from libero.libero.envs import OffScreenRenderEnv

    print(f"\n{'='*60}")
    print(f"Suite: {suite_name}")
    print(f"{'='*60}")

    if suite_name not in benchmark_dict:
        raise ValueError(
            f"Unknown suite '{suite_name}'. "
            f"Available: {sorted(benchmark_dict.keys())}"
        )

    task_suite = benchmark_dict[suite_name]()
    num_tasks  = task_suite.get_num_tasks()
    results    = {}

    for task_id in range(num_tasks):
        task      = task_suite.get_task(task_id)
        bddl_file = task_suite.get_task_bddl_file_path(task_id)
        task_desc = task.language

        env = OffScreenRenderEnv(**{
            "bddl_file_name": bddl_file,
            "camera_heights": image_size,
            "camera_widths":  image_size,
            "control_freq":   control_freq,
        })

        successes = 0
        for rollout_idx in range(num_rollouts):
            step_logger = None
            if actions_log is not None:
                def step_logger(
                    step_idx, _raw_obs, state, a_norm, a_unnorm,
                    _suite=suite_name, _task=task_id, _roll=rollout_idx,
                ):
                    actions_log.append({
                        "suite":         _suite,
                        "task_id":       _task,
                        "rollout":       _roll,
                        "step":          step_idx,
                        "state":         np.asarray(state,    dtype=np.float32),
                        "action_norm":   np.asarray(a_norm,   dtype=np.float32),
                        "action_unnorm": np.asarray(a_unnorm, dtype=np.float32),
                    })

            success    = run_episode(
                env=env,
                task_description=task_desc,
                seed=rollout_idx,
                policy=policy,
                state_mean=state_mean,
                state_std=state_std,
                action_mean=action_mean,
                action_std=action_std,
                cam_key_map=cam_key_map,
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_steps=max_steps,
                device=device,
                step_logger=step_logger,
            )
            successes += int(success)
            running_rate = successes / (rollout_idx + 1) * 100
            print(
                f"  Task {task_id:2d}/{num_tasks-1}  "
                f"rollout {rollout_idx+1:3d}/{num_rollouts}  "
                f"{'✓' if success else '✗'}  "
                f"running={running_rate:.1f}%",
                end="\r",
            )

        env.close()

        rate = successes / num_rollouts * 100
        results[task.name] = {"successes": successes, "total": num_rollouts, "rate": rate}
        print(f"\n  Task {task_id:2d} [{task.name}]  → {rate:.1f}%  ({successes}/{num_rollouts})")

    suite_avg = float(np.mean([v["rate"] for v in results.values()]))
    print(f"\n  {suite_name} average: {suite_avg:.1f}%")
    return {"tasks": results, "suite_avg": suite_avg}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LIBERO simulation benchmark")
    parser.add_argument("--checkpoint", required=True,
                        help="Local checkpoint dir or HuggingFace repo ID")
    parser.add_argument("--suites", nargs="+",
                        default=["libero_spatial", "libero_object", "libero_goal", "libero_long"],
                        help="LIBERO suites to evaluate")
    parser.add_argument("--num_rollouts",   type=int, default=50,
                        help="Rollouts per task (paper standard = 50)")
    parser.add_argument("--max_steps",      type=int, default=300,
                        help="Max env steps per rollout")
    parser.add_argument("--image_size",     type=int, default=256,
                        help="Camera resolution (height = width)")
    parser.add_argument("--n_action_steps", type=int, default=8,
                        help="Actions to execute before re-planning (action chunking)")
    parser.add_argument("--dataset_id",    default="lerobot/libero",
                        help="LeRobot dataset ID for normalization stats")
    parser.add_argument("--output_json",   default=None,
                        help="Optional path to save results as JSON")
    parser.add_argument("--headless",      action="store_true",
                        help="Start pyvirtualdisplay (requires Xvfb)")
    parser.add_argument("--skip_sanity_check", action="store_true",
                        help="Skip the state-distribution sanity check at startup")
    parser.add_argument("--actions_log", default=None,
                        help="If set, every executed action across the run is appended to this "
                             ".npz file (keys: suites, task_ids, rollouts, steps, states, "
                             "actions_norm, actions_unnorm). Cheap; safe for full benchmarks.")
    parser.add_argument("--inspect_only", action="store_true",
                        help="Skip the full benchmark; run ONE rollout on the first suite/task "
                             "and dump per-step state, action (norm + unnorm), and both camera "
                             "frames to --inspect_output. Useful for diagnosing a single failure.")
    parser.add_argument("--inspect_output", default="inspect_episode.npz",
                        help="Output .npz path used by --inspect_only.")
    parser.add_argument("--inspect_task_id", type=int, default=0,
                        help="Task index within the first suite to use for --inspect_only.")
    parser.add_argument("--inspect_seed", type=int, default=0,
                        help="Rollout seed for --inspect_only.")
    parser.add_argument("--replay_episode", type=int, default=None,
                        help="Replay this episode's recorded teleop actions through the live "
                             "env (bypassing the policy entirely). Use to isolate env/coord-"
                             "frame/action-space mismatches: if the recorded actions don't "
                             "reproduce the recorded states, the env is wrong — not the policy.")
    parser.add_argument("--replay_output", default="replay_episode.npz",
                        help="Output .npz path used by --replay_episode.")
    parser.add_argument("--replay_action_repeat", type=int, default=1,
                        help="Apply each recorded action this many env steps in --replay_episode. "
                             "Useful as a fallback if --control_freq isn't honored: e.g. setting "
                             "this to 2 mimics running a 10 Hz dataset against a 20 Hz env.")
    parser.add_argument("--control_freq", type=int, default=10,
                        help="Control frequency (Hz) for robosuite envs. Defaults to 10 to match "
                             "the lerobot/libero dataset's recording rate. Robosuite's default of "
                             "20 makes the env apply each action for only half the intended sim "
                             "time, which silently breaks both replay and policy rollouts.")
    args = parser.parse_args()

    # ── Virtual display ───────────────────────────────────────────────────────
    if args.headless:
        import os
        from pyvirtualdisplay import Display
        display = Display(visible=0, size=(1400, 900))
        display.start()
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("DISPLAY", ":99")
        print("Virtual display started.")

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Robosuite patches ─────────────────────────────────────────────────────
    _apply_robosuite_patches()

    # ── Replay mode: skip the policy entirely and run recorded actions ───────
    if args.replay_episode is not None:
        from libero.libero import benchmark as libero_benchmark
        _replay_dataset_episode(
            benchmark_dict=libero_benchmark.get_benchmark_dict(),
            dataset_id=args.dataset_id,
            episode_index=args.replay_episode,
            image_size=args.image_size,
            output_path=Path(args.replay_output),
            control_freq=args.control_freq,
            action_repeat=args.replay_action_repeat,
        )
        return

    # ── Load policy ───────────────────────────────────────────────────────────
    import huggingface_hub
    from models.transformer_flow_matching.transformer_flow_matching_policy import (
        TransformerFlowMatchingPolicy,
    )

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Downloading checkpoint: {args.checkpoint}")
        ckpt_path = Path(huggingface_hub.snapshot_download(args.checkpoint))

    print(f"Loading policy from {ckpt_path} …")
    policy = TransformerFlowMatchingPolicy.from_pretrained(ckpt_path)
    policy.eval()
    policy.to(device)
    print("Policy loaded.")

    # ── Normalization stats ───────────────────────────────────────────────────
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    try:
        meta  = LeRobotDatasetMetadata(args.dataset_id, force_cache_sync=True, revision="main")
        stats = meta.stats
        print(f"Normalization stats loaded from {args.dataset_id}.")
    except Exception as e:
        print(f"Could not fetch stats from hub ({e}). Trying checkpoint dir …")
        stats_file = ckpt_path / "dataset_stats.json"
        if not stats_file.exists():
            raise RuntimeError(f"No stats found in checkpoint dir: {stats_file}")
        with open(stats_file) as f:
            stats = json.load(f)
        print("Stats loaded from checkpoint.")

    def _stat(key: str, field: str) -> torch.Tensor:
        v = stats[key][field]
        t = torch.tensor(v, dtype=torch.float32) if not isinstance(v, torch.Tensor) else v.float()
        return t.to(device)

    state_mean  = _stat("observation.state", "mean")
    state_std   = _stat("observation.state", "std")
    action_mean = _stat("action", "mean")
    action_std  = _stat("action", "std")

    # ── Camera key mapping ────────────────────────────────────────────────────
    # lerobot/libero renders two cameras whose env names must map to the policy
    # keys the model was trained on (cameras_for_vision_state_concat).
    LIBERO_CAM_NAMES = ["agentview_image", "robot0_eye_in_hand_image"]
    policy_cam_keys  = policy.config.cameras_for_vision_state_concat
    if len(policy_cam_keys) > len(LIBERO_CAM_NAMES):
        print(
            f"WARNING: policy expects {len(policy_cam_keys)} cameras but "
            f"LIBERO provides {len(LIBERO_CAM_NAMES)}. "
            f"Extra policy cameras will be missing from the batch."
        )
    cam_key_map = {lk: pk for lk, pk in zip(LIBERO_CAM_NAMES, policy_cam_keys)}
    print(f"Camera mapping: {cam_key_map}")

    n_obs_steps = policy.config.n_obs_steps
    print(
        f"n_obs_steps={n_obs_steps}, "
        f"n_action_steps={args.n_action_steps}, "
        f"rollouts={args.num_rollouts}, "
        f"max_steps={args.max_steps}"
    )

    # ── LIBERO benchmark ──────────────────────────────────────────────────────
    from libero.libero import benchmark as libero_benchmark
    benchmark_dict = libero_benchmark.get_benchmark_dict()

    # ── Sanity check: state lands inside the training distribution ────────────
    if not args.skip_sanity_check:
        _sanity_check_state_distribution(
            benchmark_dict=benchmark_dict,
            suite_name=args.suites[0],
            image_size=args.image_size,
            state_mean=state_mean,
            state_std=state_std,
            control_freq=args.control_freq,
        )

    # ── Inspect-only mode: one rollout, full per-step dump, then exit ─────────
    if args.inspect_only:
        _run_inspect_episode(
            benchmark_dict=benchmark_dict,
            suite_name=args.suites[0],
            task_id=args.inspect_task_id,
            seed=args.inspect_seed,
            image_size=args.image_size,
            policy=policy,
            state_mean=state_mean,
            state_std=state_std,
            action_mean=action_mean,
            action_std=action_std,
            cam_key_map=cam_key_map,
            n_obs_steps=n_obs_steps,
            n_action_steps=args.n_action_steps,
            max_steps=args.max_steps,
            device=device,
            output_path=Path(args.inspect_output),
            control_freq=args.control_freq,
        )
        return

    actions_log: list | None = [] if args.actions_log else None

    all_results: dict = {}
    for suite in args.suites:
        try:
            all_results[suite] = evaluate_suite(
                suite_name=suite,
                benchmark_dict=benchmark_dict,
                image_size=args.image_size,
                num_rollouts=args.num_rollouts,
                max_steps=args.max_steps,
                policy=policy,
                state_mean=state_mean,
                state_std=state_std,
                action_mean=action_mean,
                action_std=action_std,
                cam_key_map=cam_key_map,
                n_obs_steps=n_obs_steps,
                n_action_steps=args.n_action_steps,
                device=device,
                actions_log=actions_log,
                control_freq=args.control_freq,
            )
        except Exception as e:
            print(f"\nSkipped {suite}: {e}")
            import traceback
            traceback.print_exc()

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n\n" + "="*60)
    print("RESULTS")
    print("="*60)

    rows = []
    for suite, data in all_results.items():
        for task_name, td in data["tasks"].items():
            rows.append({
                "Suite": suite,
                "Task": task_name,
                "Success": f"{td['successes']}/{td['total']}",
                "Rate (%)": f"{td['rate']:.1f}",
            })

    if rows:
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))

    print("\nSuite Averages:")
    avgs = []
    for suite, data in all_results.items():
        avg = data["suite_avg"]
        avgs.append(avg)
        print(f"  {suite:20s}  {avg:5.1f}%")
    if avgs:
        print(f"  {'Overall':20s}  {np.mean(avgs):5.1f}%")

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved → {args.output_json}")

    if args.actions_log and actions_log is not None and len(actions_log) > 0:
        out = Path(args.actions_log)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            suites=np.array([r["suite"]   for r in actions_log]),
            task_ids=np.array([r["task_id"] for r in actions_log], dtype=np.int32),
            rollouts=np.array([r["rollout"] for r in actions_log], dtype=np.int32),
            steps=np.array([r["step"]    for r in actions_log], dtype=np.int32),
            states=np.stack([r["state"]         for r in actions_log]),
            actions_norm=np.stack([r["action_norm"]   for r in actions_log]),
            actions_unnorm=np.stack([r["action_unnorm"] for r in actions_log]),
        )
        print(f"Actions log saved → {args.actions_log}  ({len(actions_log)} steps)")


if __name__ == "__main__":
    # Ensure src/ is on sys.path when the script is run from the project root
    _src_dir = str(Path(__file__).parent)
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)
    main()
