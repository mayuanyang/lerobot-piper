"""LIBERO evaluation for WiltechsX checkpoints.

    python src/eval_wiltechs_x.py \
        --checkpoint outputs/wx_a/checkpoint-5000 \
        --suites libero_spatial libero_object libero_goal libero_10 \
        --episodes 50 --num_envs 10

READ THE `min` COLUMN, NOT THE AVERAGE. ARCHITECTURE.md §1: stage-B RL recovers
a task sitting at 10% and can do nothing with one sitting at 0, because a binary
reward has no gradient where every rollout fails. 93% average with a per-task
floor of 15% is a better stage-A checkpoint than 95% with two zeros. The gate
this prints is `avg >= 93 AND min > 5`.

Three things this harness pins that a naive eval gets wrong, each of which has
already cost this repo a set of non-comparable numbers:

  * `control_freq=10`. The LIBERO demos are 10 Hz and stock robosuite is 20, so
    a delta-EEF action sized for 1/10 s is held for 1/20 s instead and moves
    half as far. Numbers taken at 20 Hz do not transfer (rollout 0.86 against
    eval 77.5% in this repo's own RL runs).
  * The canonical 50 initial states. lerobot's LiberoEnv.reset() writes the init
    state and THEN lets robosuite re-sample the placement initializer over it,
    serving layouts 3-10x more spread out than the ones the demos were recorded
    on. `libero_env_fixed.patch_lerobot_libero()` restores LIBERO's own order.
    `--stock_init` runs without the fix, for an A/B on one checkpoint.
  * The proprioceptive HISTORY. Training feeds `observation.state` as
    (B, motion_history_len, D) via delta_timestamps, and MotionVectorEncoder
    takes first differences over it. An eval that passes the single current
    frame leaves the model with an all-zero motion signal -- it will still run,
    score lower, and say nothing about why. See `StateHistory` below.

Few-step inference is a claim, not a given: `--num_inference_steps` defaults to
the config's 4. Re-running at 16 measures whether the shortcut consistency term
actually made 4 NFE valid. A large gap means it did not, and the flow objective
-- not the policy -- is what to fix.
"""
from __future__ import annotations

import os

# Must precede any robosuite/mujoco import.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# Same reason, one layer up: LIBERO's env_wrapper imports matplotlib.cm, and
# matplotlib resolves MPLBACKEND at import time. A notebook exports
# MPLBACKEND=module://matplotlib_inline.backend_inline, which is only importable
# inside the notebook's OWN interpreter -- run this script from a Colab cell
# against any other environment and matplotlib raises before LIBERO loads. This
# is a headless eval that draws nothing, so agg is the right backend; a
# deliberate non-inline choice is left alone.
_mpl = os.environ.get("MPLBACKEND", "")
if not _mpl or "inline" in _mpl:
    os.environ["MPLBACKEND"] = "agg"

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from libero_env_fixed import patch_lerobot_libero
from models.wiltechs_x.wiltechs_x_config import WiltechsXConfig  # noqa: F401  (registers "wiltechs_x")
from models.wiltechs_x.wiltechs_x_policy import WiltechsXPolicy


def _git_commit() -> str | None:
    """Which build produced this JSON. Cheap, and the alternative is guessing
    from which keys happen to be present in the file."""
    import subprocess
    try:
        r = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent),
                            "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except Exception:
        return None


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Proprioceptive history
# ---------------------------------------------------------------------------
class StateHistory:
    """Rolling (T, D) window of `observation.state`, one per env.

    Training builds this with delta_timestamps
    (`[-i*ft for i in range(n_obs_steps)]`), so the model has always seen T
    frames. At reset there is only one, and LeRobot's own left-padding
    convention repeats the earliest frame -- MotionVectorEncoder does exactly
    that internally, so seeding the deque full of the reset state reproduces it
    without relying on the encoder's fallback.
    """

    def __init__(self, n_envs: int, history_len: int):
        self.t = max(1, int(history_len))
        self.buf = [deque(maxlen=self.t) for _ in range(n_envs)]

    def reset(self, i: int, state: np.ndarray):
        self.buf[i].clear()
        for _ in range(self.t):
            self.buf[i].append(np.asarray(state, dtype=np.float32))

    def push(self, i: int, state: np.ndarray):
        self.buf[i].append(np.asarray(state, dtype=np.float32))

    def stack(self) -> np.ndarray:
        """-> (n_envs, T, D)."""
        return np.stack([np.stack(list(b)) for b in self.buf])


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_policy(ckpt: Path, device: str, num_inference_steps: int | None,
                n_action_steps: int | None = None,
                fixed_episode_noise: bool = False):
    from lerobot.configs.policies import PreTrainedConfig

    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.device = str(device)
    if num_inference_steps:
        cfg.num_inference_steps = int(num_inference_steps)
    if n_action_steps:
        n = int(n_action_steps)
        if n > int(cfg.horizon):
            raise SystemExit(
                f"--n_action_steps {n} exceeds the trained horizon "
                f"{cfg.horizon}: the chunk has no steps past that to execute.")
        cfg.n_action_steps = n
    if fixed_episode_noise:
        cfg.fixed_episode_noise = True
    policy = WiltechsXPolicy.from_pretrained(ckpt, config=cfg)
    policy.to(device)
    policy.eval()
    for m in policy.model.modules():                      # deterministic rollout
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    return policy


def load_processors(ckpt: Path, device: str, dataset_id: str | None):
    """Prefer the pipelines saved next to the weights.

    They carry the dataset statistics the policy was TRAINED against, which is
    the only correct choice: rebuilding from a dataset that has since gained
    episodes would unnormalize with different numbers than the model learned.
    `--dataset_id` exists only for checkpoints written before the trainer saved
    them, and says so loudly.
    """
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        policy_action_to_transition,
        transition_to_policy_action,
    )
    from lerobot.utils.constants import (
        POLICY_POSTPROCESSOR_DEFAULT_NAME,
        POLICY_PREPROCESSOR_DEFAULT_NAME,
    )

    pre_json = ckpt / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
    if pre_json.exists():
        pre = PolicyProcessorPipeline.from_pretrained(
            ckpt, config_filename=pre_json.name,
            overrides={"device_processor": {"device": str(device)}})
        post = PolicyProcessorPipeline.from_pretrained(
            ckpt, config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
            overrides={"device_processor": {"device": "cpu"}})
        print(f"[eval] processors loaded from {ckpt.name} (training statistics)")
        return pre, post

    if not dataset_id:
        raise SystemExit(
            f"{pre_json} is missing and no --dataset_id was given. Without the "
            f"normalization statistics the policy's inputs and outputs are on "
            f"the wrong scale and every number this script prints is noise.")

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from models.wiltechs_x.processor_wiltechs_x import make_pre_post_processors

    print(f"[eval] WARNING: no saved processors in {ckpt}; rebuilding from "
          f"{dataset_id}. Valid only if that dataset is byte-identical to the "
          f"one trained on.")
    cfg = WiltechsXConfig.from_pretrained(ckpt)
    cfg.device = str(device)
    stats = LeRobotDatasetMetadata(dataset_id, revision="main").stats
    return make_pre_post_processors(cfg, dataset_stats=stats)


def patch_control_freq(control_freq: int, render_gpu: int):
    """Build LiberoEnv's OffScreenRenderEnv at `control_freq` Hz.

    Same patch as `train_wilro_rl._patch_libero_control_freq`, inlined rather
    than imported: that module is a 1200-line trainer that sets up multiprocess
    EGL at import, and an eval should not drag that in for six lines.
    """
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from lerobot.envs.libero import LiberoEnv

    def _make_envs_task(self, task_suite, task_id: int = 0):
        task = task_suite.get_task(task_id)
        self.task = task.name
        self.task_description = task.language
        bddl = os.path.join(get_libero_path("bddl_files"),
                            task.problem_folder, task.bddl_file)
        # robosuite reads render_gpu_device_id, NOT MUJOCO_EGL_DEVICE_ID.
        env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=self.observation_height,
            camera_widths=self.observation_width,
            control_freq=control_freq,
            render_gpu_device_id=max(render_gpu, 0),
        )
        env.reset()
        return env

    LiberoEnv._make_envs_task = _make_envs_task
    print(f"[eval] env control_freq={control_freq} Hz "
          f"({'matches the 10 Hz dataset' if control_freq == 10 else 'NOT 10 Hz — numbers will not transfer'})")


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------
def build_batch(obs_list, tasks, hist: StateHistory, preprocessor, device):
    from lerobot.envs.utils import preprocess_observation

    stacked = {
        "pixels": {cam: np.stack([o["pixels"][cam] for o in obs_list])
                   for cam in obs_list[0]["pixels"]},
        "agent_pos": np.stack([o["agent_pos"] for o in obs_list]),
    }
    batch = preprocess_observation(stacked)
    # Override the single current frame with the full window. preprocess_observation
    # only ever produces (B, D); the model wants (B, T, D) and slices [:, -1] for
    # the state token, exactly as in training.
    batch["observation.state"] = torch.from_numpy(hist.stack()).float()
    batch["task"] = list(tasks)
    return preprocessor(batch)


@torch.no_grad()
def eval_task(policy, preprocessor, postprocessor, suite, suite_name: str,
              task_id: int, episodes: int, num_envs: int, device: str,
              max_episode_steps: int, seed: int, expected_cams: list[str],
              video_cb=None, videos_per_task: int = 0, heartbeat: int = 50,
              instruction: str | None = None):
    """-> (n_success, n_episodes, mean_success_steps, n_chunks, task_description,
    per_episode_success).

    The per-episode vector is what makes two checkpoints COMPARABLE. Episode i
    starts from the same canonical init state in every run (fixed_init_states),
    so two evals are a PAIRED sample and McNemar applies. Comparing only the
    two rates throws that away and leaves ~15pp of unpaired noise at n=20 --
    enough to invent a 20-point "regression" between adjacent checkpoints.
    """
    from lerobot.envs.libero import LiberoEnv

    # Building an OffScreenRenderEnv takes seconds and there are num_envs of
    # them PER TASK (LiberoEnv binds its bddl file at construction, so they
    # cannot be reused across tasks). Say so: this is minutes of silence
    # before a single rollout step happens.
    t_build = time.time()
    print(f"  task {task_id:2d}: building {num_envs} envs...", end="", flush=True)
    envs = [LiberoEnv(task_suite=suite, task_id=task_id,
                      task_suite_name=suite_name, obs_type="pixels_agent_pos",
                      init_states=True, episode_index=0)
            for _ in range(num_envs)]
    print(f" {time.time() - t_build:.0f}s", flush=True)
    try:
        probe, _ = envs[0].reset(seed=seed)
        got = sorted(probe["pixels"].keys())
        want = [c.split(".")[-1] for c in expected_cams]
        missing = [c for c in want if c not in got]
        if missing:
            raise SystemExit(
                f"LIBERO provides cameras {got}; the policy expects {want} "
                f"(from cameras_for_vlm={expected_cams}). _encode_images drops "
                f"missing cameras SILENTLY, so this would score a "
                f"differently-conditioned model. Pass a camera_name_mapping to "
                f"LiberoEnv for this lerobot version.")

        task_desc = envs[0].task_description
        n_states = len(envs[0]._init_states)
        horizon_cap = min(envs[0]._max_episode_steps, max_episode_steps) \
            if max_episode_steps else envs[0]._max_episode_steps
        hist = StateHistory(num_envs, policy.config.n_obs_steps)

        autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if device == "cuda"
                    else torch.autocast(device_type="cpu", enabled=False))

        n_batches = (episodes + num_envs - 1) // num_envs
        # Never truncate the instruction. LIBERO's tasks share a prefix and
        # differ at the END ("...between the plate and the ramekin"), so
        # clipping the tail hides the only part that distinguishes them -- and
        # makes a display limit look like the tokenizer dropping text.
        # The success criterion always comes from the env, i.e. from the REAL
        # task. Only what the policy is told changes.
        told = instruction if instruction is not None else task_desc
        print(f"    {episodes} episodes in {n_batches} batch(es) of {num_envs}, "
              f"cap {horizon_cap} steps\n    task: {task_desc!r}", flush=True)
        if instruction is not None:
            print(f"    ABLATED, policy is told: {told!r}", flush=True)

        successes, steps_to_success, n_chunks = [], [], 0
        for start in range(0, episodes, num_envs):
            t_batch = time.time()
            n_live = min(num_envs, episodes - start)
            policy.reset()
            obs_list, frames = [], [[] for _ in range(num_envs)]
            for i in range(num_envs):
                # Episode index -> init state. Explicit, not seed-derived: the
                # canonical set is 50 layouts and coverage should be exact and
                # reproducible, not a hash of the seed.
                envs[i]._init_state_id = (start + i) % n_states
                o, _ = envs[i].reset(seed=seed + start + i)
                obs_list.append(o)
                hist.reset(i, o["agent_pos"])

            done = [i >= n_live for i in range(num_envs)]   # pad slots start done
            succ = [False] * num_envs
            steps = [0] * num_envs
            t = 0
            while not all(done) and t < horizon_cap:
                # The batch dim stays at num_envs even as envs finish: the action
                # queue inside select_action is keyed on batch size, and resizing
                # it mid-chunk would drop the actions the live envs still owe.
                batch = build_batch(obs_list, [told] * num_envs, hist,
                                    preprocessor, device)
                # An empty queue means this call will run the prefix. Counting
                # it here rather than after the call keeps it correct at
                # n_action_steps=1, where the queue is empty again on return.
                drew_chunk = not policy._action_queue
                with autocast:
                    action = policy.select_action(batch)
                n_chunks += int(drew_chunk)
                env_action = postprocessor(action.float().cpu()).numpy()

                for i in range(num_envs):
                    if done[i]:
                        continue
                    lo, hi = envs[i].action_space.low, envs[i].action_space.high
                    a = np.clip(env_action[i], lo, hi).astype(np.float32)
                    o, _r, terminated, truncated, info = envs[i].step(a)
                    obs_list[i] = o
                    hist.push(i, o["agent_pos"])
                    steps[i] += 1
                    # Only the first few env slots buffer frames. A full 256x256
                    # episode is ~100 MB; recording all of them would cost more
                    # RAM than the policy does, to write videos we then discard.
                    if video_cb is not None and i < videos_per_task:
                        frames[i].append(o["pixels"][got[0]])
                    if terminated or truncated:
                        # LiberoEnv auto-resets on termination, so the env must
                        # not be stepped again or it silently starts a new
                        # episode and pollutes the next chunk's observation.
                        done[i] = True
                        succ[i] = bool(info.get("is_success", False))
                t += 1

                # Heartbeat. Without it a batch that runs to the cap is many
                # minutes of total silence, and there is no way to tell a slow
                # rollout from a hung one. `live` falling to 0 early means the
                # episodes are terminating; live staying at num_envs to the cap
                # means everything is timing out, i.e. failing.
                if heartbeat and t % heartbeat == 0:
                    live = sum(1 for i in range(n_live) if not done[i])
                    hit = sum(succ[:n_live])
                    el = time.time() - t_batch
                    print(f"      t={t:4d}/{horizon_cap}  live={live}/{n_live}  "
                          f"success={hit}  {el:5.0f}s  "
                          f"({el / max(t, 1) * horizon_cap:.0f}s if it runs to cap)",
                          flush=True)

            hit = sum(succ[:n_live])
            print(f"    batch {start // num_envs + 1}/{n_batches}: "
                  f"{hit}/{n_live} success in {t} steps, "
                  f"{time.time() - t_batch:.0f}s", flush=True)

            for i in range(n_live):
                successes.append(succ[i])
                if succ[i]:
                    steps_to_success.append(steps[i])
                elif video_cb is not None:
                    video_cb(task_id, start + i, frames[i])

        mean_steps = float(np.mean(steps_to_success)) if steps_to_success else float("nan")
        return (sum(successes), len(successes), mean_steps, n_chunks, task_desc,
                [int(s) for s in successes])
    finally:
        for e in envs:
            try:
                e.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
def make_video_writer(video_dir: Path | None, max_per_task: int):
    if video_dir is None:
        return None
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("[eval] --video_dir set but imageio is not installed; skipping video.")
        return None
    video_dir.mkdir(parents=True, exist_ok=True)
    written: dict[int, int] = {}

    def cb(task_id, episode, frames):
        if not frames or written.get(task_id, 0) >= max_per_task:
            return
        written[task_id] = written.get(task_id, 0) + 1
        path = video_dir / f"task{task_id:02d}_ep{episode:03d}_FAIL.mp4"
        imageio.mimwrite(path, [np.asarray(f) for f in frames], fps=10)

    return cb


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--suites", nargs="+",
                   default=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    p.add_argument("--task_ids", nargs="+", type=int, default=None,
                   help="Default: every task in each suite.")
    p.add_argument("--episodes", type=int, default=50,
                   help="Per task. 50 is the canonical LIBERO count and matches "
                        "the number of init states, so each is visited once.")
    p.add_argument("--num_envs", type=int, default=10,
                   help="Envs stepped in lockstep in THIS process, batched "
                        "through one policy forward. MuJoCo's EGL context is "
                        "per-thread, so these must not be threaded; sequential "
                        "stepping in one process is correct and the policy "
                        "forward (the expensive part) still batches.")
    p.add_argument("--control_freq", type=int, default=10,
                   help="MUST be 10 to match the dataset. Changing it invalidates "
                        "comparison with every other number in this repo.")
    p.add_argument("--max_episode_steps", type=int, default=0,
                   help="0 = the suite default.")
    p.add_argument("--num_inference_steps", type=int, default=0,
                   help="0 = the checkpoint's config (4). Re-run at 16 to test "
                        "whether the shortcut term made few-step inference valid.")
    p.add_argument("--n_action_steps", type=int, default=0,
                   help="0 = the checkpoint's config (8). Steps of each chunk "
                        "executed open-loop before replanning. At 10 Hz, 8 is "
                        "0.8 s and ~35 replans per episode; wiltechs_vla and "
                        "wiltechs_moe run 32 of a 64 horizon, so they "
                        "re-decide 4x less often. Each replan redraws the "
                        "noise, i.e. resamples WHICH plan to follow, so a high "
                        "replan rate is a candidate cause of the stumbling "
                        "approach. Cannot exceed the trained horizon.")
    p.add_argument("--fixed_episode_noise", action="store_true",
                   help="Draw x_1 once per episode and reuse it for every "
                        "replan. The integration is deterministic given the "
                        "noise, so this keeps the policy on ONE branch of a "
                        "multimodal action distribution while staying fully "
                        "reactive to the observation. A bad branch now costs "
                        "the whole episode instead of 0.8 s, so read the "
                        "success distribution, not only the mean.")
    p.add_argument("--stock_init", action="store_true",
                   help="Disable the init-state ordering fix. For an A/B against "
                        "the canonical 50 layouts; not for reportable numbers.")
    p.add_argument("--dataset_id", default=None,
                   help="Only for checkpoints saved without their processors.")
    p.add_argument("--device", default=None)
    p.add_argument("--render_gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--video_dir", default=None,
                   help="Write up to --videos_per_task FAILED episodes per task. "
                        "This repo's grasp-vs-selection diagnoses came from "
                        "watching these, not from the success rate.")
    p.add_argument("--videos_per_task", type=int, default=2)
    p.add_argument("--ablate_lang", action="store_true",
                   help="Tell the policy ANOTHER task's instruction while "
                        "scoring against the real one. The bridge between "
                        "'the CE depends on language a little' and 'behaviour "
                        "depends on language': if the success rate does not "
                        "move, the instruction is not driving the policy and "
                        "no amount of further training changes that. Run it "
                        "against an identical non-ablated run -- same "
                        "checkpoint, seed, episodes and cap.")
    p.add_argument("--instruction_override", default=None,
                   help="Tell the policy this exact instruction instead of the "
                        "task's own, while still scoring against the real task. "
                        "For a SPECIFIC confusion, where --ablate_lang's fixed "
                        "half-suite offset does not put the two tasks in "
                        "question against each other. Also takes a rephrasing, "
                        "to ask whether a more distinctive wording is followed.")
    p.add_argument("--instruction_from_task", type=int, default=None,
                   help="Same, but pulls the instruction off another task in "
                        "this suite by id -- no chance of a typo silently "
                        "testing a different sentence. Use with --task_ids to "
                        "swap a pair: --task_ids 7 --instruction_from_task 9.")
    p.add_argument("--heartbeat", type=int, default=50,
                   help="Env steps between progress lines inside a rollout. A "
                        "batch that runs to the episode cap is minutes of "
                        "silence otherwise, with no way to tell slow from hung. "
                        "0 = off.")
    p.add_argument("--out", default=None, help="JSON results path.")
    a = p.parse_args()

    device = a.device or pick_device()
    from train_wiltechs_x import resolve_checkpoint

    # --seed used to reach only env.reset(). The POLICY is stochastic -- flow
    # matching draws x_1 fresh for every chunk, which at n_action_steps=2 is
    # 140 draws per episode -- so two runs of the identical command walked
    # different trajectories through identical layouts. That also made the
    # --ablate_lang instruction to use "the same checkpoint, seed, episodes and
    # cap" impossible to honour.
    #
    # Seeding does NOT shrink the error on a single estimate: at n=20 the
    # binomial SE near p=0.85 is ~8 points, which is what an 80% and a 90% run
    # of the same command actually differ by. What it buys is a PAIRED A/B --
    # same layouts and same noise, only the setting under test moving.
    #
    # Pairing is exact only when the two arms draw the same number of samples.
    # Comparing n_action_steps settings changes that count, so the noise
    # streams diverge after the first chunk; --fixed_episode_noise draws once
    # per episode and pairs across those too.
    torch.manual_seed(a.seed)
    np.random.seed(a.seed % (2 ** 32))

    n_lang = sum(x is not None and x is not False
                 for x in (a.ablate_lang or None, a.instruction_override,
                           a.instruction_from_task))
    if n_lang > 1:
        raise SystemExit(
            "--ablate_lang, --instruction_override and --instruction_from_task "
            "all replace what the policy is told. Pick one; combining them "
            "would report a number nobody could attribute.")

    ckpt = resolve_checkpoint(a.checkpoint, for_resume=False)

    patch_lerobot_libero(enable=not a.stock_init)
    patch_control_freq(a.control_freq, a.render_gpu)

    policy = load_policy(ckpt, device, a.num_inference_steps,
                         a.n_action_steps, a.fixed_episode_noise)
    pre, post = load_processors(ckpt, device, a.dataset_id)
    cams = list(policy.config.cameras_for_vlm)
    print(f"[eval] {ckpt}  device={device}  cameras={cams}\n"
          f"[eval] horizon={policy.config.horizon} "
          f"n_action_steps={policy.config.n_action_steps} "
          f"NFE={policy.config.num_inference_steps} "
          f"state_history={policy.config.n_obs_steps} "
          f"noise={'fixed/episode' if a.fixed_episode_noise else 'per-chunk'}")

    video_cb = make_video_writer(Path(a.video_dir) if a.video_dir else None,
                                 a.videos_per_task)

    from lerobot.envs.libero import _get_suite

    results, t0 = {}, time.time()
    for suite_name in a.suites:
        suite = _get_suite(suite_name)
        n_tasks = getattr(suite, "n_tasks", None) or len(suite.tasks)
        task_ids = a.task_ids if a.task_ids is not None else list(range(n_tasks))
        # Read the instructions off the suite rather than off an env: LiberoEnv
        # binds one task at construction, so collecting them the other way
        # would mean building (and rendering) every task just to read a string.
        wrong = {}
        if a.ablate_lang:
            if n_tasks < 2:
                raise SystemExit(
                    f"--ablate_lang needs a suite with >1 task; {suite_name} "
                    f"has {n_tasks}, so the 'wrong' instruction would be the "
                    f"right one and the run would report a false null.")
            all_desc = {t: suite.get_task(t).language for t in range(n_tasks)}
            # A fixed half-suite offset: deterministic, and it never lands on
            # the task itself. Every libero_spatial task shares one tabletop,
            # so the wrong instruction is still valid FOR THAT SCENE -- it asks
            # for a different object, which is exactly the confusion to test.
            # A random or out-of-scene string would test novelty instead.
            wrong = {t: all_desc[(t + max(n_tasks // 2, 1)) % n_tasks]
                     for t in task_ids}
        elif a.instruction_override or a.instruction_from_task is not None:
            # --ablate_lang's half-suite offset asks "does ANY wrong instruction
            # change behaviour". That is the wrong question for a specific
            # confusion: libero_spatial task 7 ("on the stove") is scored at 60%
            # with its failures reaching for the cabinet, and task 9 ("on the
            # wooden cabinet") at 50% -- a pair the offset never puts against
            # each other. Naming the instruction directly is what tests whether
            # the policy can tell those two apart.
            if a.instruction_from_task is not None:
                if not 0 <= a.instruction_from_task < n_tasks:
                    raise SystemExit(
                        f"--instruction_from_task {a.instruction_from_task} is "
                        f"outside {suite_name}'s 0..{n_tasks - 1}")
                text = suite.get_task(a.instruction_from_task).language
            else:
                text = a.instruction_override
            for t in task_ids:
                real = suite.get_task(t).language
                if text.strip() == real.strip():
                    # Silently scoring a task against its own instruction would
                    # look like "language has no effect" when nothing was
                    # actually swapped.
                    raise SystemExit(
                        f"task {t}'s own instruction is {real!r}, which is what "
                        f"the override supplies. That is a null test, not a "
                        f"result -- pick a different task or string.")
            wrong = {t: text for t in task_ids}

        tag = ("  [LANGUAGE ABLATED]" if a.ablate_lang
               else "  [INSTRUCTION OVERRIDDEN]" if wrong else "")
        print(f"\n=== {suite_name}: {len(task_ids)} tasks x {a.episodes} episodes"
              f"{tag} ===")
        per_task = {}
        for k, tid in enumerate(task_ids):
            t_task = time.time()
            n_ok, n_ep, mean_steps, n_chunks, desc, ep_ok = eval_task(
                policy, pre, post, suite, suite_name, tid, a.episodes,
                a.num_envs, device, a.max_episode_steps, a.seed, cams, video_cb,
                a.videos_per_task, a.heartbeat, wrong.get(tid))
            sr = 100.0 * n_ok / max(n_ep, 1)
            per_task[tid] = {"success_rate": sr, "n_success": n_ok,
                             "n_episodes": n_ep, "mean_success_steps": mean_steps,
                             "policy_chunks": n_chunks, "task": desc,
                             # Ordered by episode index == canonical init state,
                             # so two runs of this line are paired. See eval_task.
                             "episode_success": ep_ok}
            done_n, total_n = k + 1, len(task_ids)
            eta = (time.time() - t0) / done_n * (total_n - done_n) / 60
            print(f"  task {tid:2d}  SR {sr:5.1f}%  ({n_ok}/{n_ep})  "
                  f"steps~{mean_steps:.0f}  [{done_n}/{total_n}, "
                  f"{(time.time() - t_task) / 60:.1f} min, ETA {eta:.0f} min]  "
                  f"{desc}", flush=True)
        rates = [v["success_rate"] for v in per_task.values()]
        results[suite_name] = {
            "per_task": per_task,
            "avg": float(np.mean(rates)),
            "min": float(np.min(rates)),
            "n_zero_tasks": int(sum(1 for r in rates if r == 0.0)),
        }
        s = results[suite_name]
        print(f"  {suite_name}: avg {s['avg']:.1f}%  MIN {s['min']:.1f}%  "
              f"tasks at zero: {s['n_zero_tasks']}")

    all_rates = [v["success_rate"] for s in results.values()
                 for v in s["per_task"].values()]
    avg, mn = float(np.mean(all_rates)), float(np.min(all_rates))
    zeros = [(s, t) for s, d in results.items()
             for t, v in d["per_task"].items() if v["success_rate"] == 0.0]

    print(f"\n{'=' * 62}")
    print(f"OVERALL  avg {avg:.1f}%   per-task MIN {mn:.1f}%   "
          f"tasks at zero: {len(zeros)}")
    gate = avg >= 93.0 and mn > 5.0
    print(f"stage-A gate (avg >= 93 AND min > 5): {'PASS' if gate else 'FAIL'}")
    if zeros:
        print("Tasks at 0% — stage-B RL cannot recover these; a binary reward "
              "has no gradient where every rollout fails:")
        for s, t in zeros:
            print(f"  {s} task {t}: {results[s]['per_task'][t]['task']}")
    print(f"{(time.time() - t0) / 60:.1f} min")

    if a.ablate_lang:
        print("\nThis was a LANGUAGE ABLATION -- the policy was given another "
              "task's instruction.\nCompare it against a non-ablated run with "
              "the same checkpoint, seed, episodes\nand cap. An unchanged "
              "success rate means the instruction is not driving the\npolicy, "
              "which no amount of further training changes.")

    # Everything a later run must match to be comparable. `seed` and the eval
    # commit were missing, and both bit: the seed only started reaching the
    # policy in 92ec163, so a JSON written before it recorded a draw that
    # cannot be reproduced -- and nothing in the file said which side it was
    # on. A baseline you cannot re-run is not a baseline.
    payload = {"checkpoint": str(ckpt), "control_freq": a.control_freq,
               "fixed_init_states": not a.stock_init,
               "seed": a.seed,
               "max_episode_steps": a.max_episode_steps,
               "eval_commit": _git_commit(),
               "num_inference_steps": policy.config.num_inference_steps,
               "n_action_steps": policy.config.n_action_steps,
               "fixed_episode_noise": bool(a.fixed_episode_noise),
               "episodes_per_task": a.episodes, "ablate_lang": a.ablate_lang,
               "instruction_override": a.instruction_override,
               "instruction_from_task": a.instruction_from_task,
               "overall_avg": avg, "overall_min": mn, "gate_pass": gate,
               "suites": results}
    # A separate filename: an ablation result overwriting the real one is a
    # mistake you only notice much later.
    default_name = ("eval_libero_ablated.json" if a.ablate_lang
                    else "eval_libero_override.json"
                    if (a.instruction_override or a.instruction_from_task is not None)
                    else "eval_libero.json")
    out = Path(a.out) if a.out else ckpt / default_name
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
