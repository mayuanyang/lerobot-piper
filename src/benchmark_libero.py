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
    """(H, W, C) uint8 → (C, H, W) float32 in [0, 1]."""
    return torch.from_numpy(img.copy()).float().div_(255.0).permute(2, 0, 1)


def build_state(obs: dict) -> np.ndarray:
    """
    Map LIBERO env obs → lerobot/libero observation.state (8-dim).

      [0:6]  robot0_joint_pos[:6]   — arm joints (rad)
      [6:8]  robot0_gripper_qpos    — gripper fingers (m)

    joint_pos[3] is sign-flipped: the lerobot/libero dataset was built with an
    older robosuite that defines joint 4's axis in the opposite direction to
    robosuite 1.4.1. The raw dataset values are always positive (≈3.67 rad at
    rest), whereas robosuite 1.4.1 reports ≈ −π. Negating restores alignment.
    """
    joint_pos = np.array(obs["robot0_joint_pos"], dtype=np.float64)
    gripper   = np.array(obs["robot0_gripper_qpos"], dtype=np.float64)
    joint_pos[3] = -joint_pos[3]
    return np.concatenate([joint_pos[:6], gripper[:2]]).astype(np.float32)


def obs_to_frame(obs: dict, cam_key_map: dict[str, str]) -> dict:
    frame = {"state": build_state(obs)}
    for libero_key, policy_key in cam_key_map.items():
        frame[policy_key] = img_to_tensor(obs[libero_key])
    return frame


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
) -> bool:
    """Run one rollout. Returns True on task success."""
    obs = env.reset()

    # Set seed for reproducible initial conditions within the episode
    try:
        env.env.seed(seed)
    except AttributeError:
        pass

    first_frame  = obs_to_frame(obs, cam_key_map)
    obs_buffer   = deque([first_frame] * n_obs_steps, maxlen=n_obs_steps)
    action_queue: deque = deque()

    for _ in range(max_steps):
        if not action_queue:
            batch = build_batch(
                obs_buffer, task_description,
                state_mean, state_std, cam_key_map, device,
            )
            with torch.no_grad():
                actions_norm = policy.predict_action_chunk(batch)  # (1, horizon, action_dim)

            actions = (actions_norm * action_std + action_mean)[0, :n_action_steps].cpu().numpy()
            action_queue.extend(actions)

        action = action_queue.popleft()
        obs, _reward, done, _info = env.step(action)
        obs_buffer.append(obs_to_frame(obs, cam_key_map))

        if done:
            return True

    return False


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
        })

        successes = 0
        for rollout_idx in range(num_rollouts):
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


if __name__ == "__main__":
    # Ensure src/ is on sys.path when the script is run from the project root
    _src_dir = str(Path(__file__).parent)
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)
    main()
