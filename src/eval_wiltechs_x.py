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

# One harness, several policies -- deliberately NOT a second script.
#
# Everything that makes two numbers comparable lives in this file: the
# canonical-50 init states (patch_lerobot_libero), control_freq=10, per-task
# policy seeding, the paired per-episode vectors. A copy for another model
# would be a second place for those to drift, and this repo has already paid
# for exactly that -- the sibling's 92% was produced by a script that is no
# longer in the tree, so it cannot be re-run against the current fixes at all.
#
# Adding a model here is two lines and it inherits every fix.
POLICIES = {
    "wiltechs_x":   ("models.wiltechs_x.wiltechs_x_policy", "WiltechsXPolicy"),
    "wiltechs_moe": ("models.wiltechs_moe.wiltechs_moe_policy", "WiltechsMoEPolicy"),
    "wiltechs_vla": ("models.wiltechs_vla.wiltechs_vla_policy", "WiltechsVLAPolicy"),
    "wilro":        ("models.wilro.wilro_policy", "WilroPolicy"),
}


def _register_configs():
    """Importing a config module is what registers its `type` string, and
    PreTrainedConfig.from_pretrained resolves the checkpoint by that string.
    Failures are per-model and non-fatal: a missing sibling must not stop an
    eval of the one that is present."""
    for mod in ("models.wiltechs_moe.wiltechs_moe_config",
                "models.wiltechs_vla.wiltechs_vla_config",
                "models.wilro.wilro_config"):
        try:
            __import__(mod)
        except Exception:
            pass


_register_configs()


def _policy_class(kind: str):
    import importlib
    if kind not in POLICIES:
        raise SystemExit(
            f"checkpoint declares policy type {kind!r}, which this harness does "
            f"not know.\n    Known: {', '.join(sorted(POLICIES))}\n"
            f"    Add it to POLICIES in {Path(__file__).name} -- two lines, and "
            f"it inherits every eval fix.")
    mod, cls = POLICIES[kind]
    return getattr(importlib.import_module(mod), cls)


def _policy_cameras(cfg) -> list[str]:
    """Camera keys the checkpoint expects, from the checkpoint itself.

    `cameras_for_vlm` is WiltechsX's name for it. Every lerobot policy config
    also carries input_features, so fall back to the VISUAL ones there rather
    than hard-coding a per-model attribute name.
    """
    cams = list(getattr(cfg, "cameras_for_vlm", None) or [])
    if cams:
        return cams
    feats = getattr(cfg, "input_features", None) or {}
    cams = [k for k, v in feats.items()
            if getattr(getattr(v, "type", None), "name", "") == "VISUAL"]
    if cams:
        return sorted(cams)
    raise SystemExit(
        "cannot tell which cameras this checkpoint expects: its config has "
        "neither cameras_for_vlm nor VISUAL input_features.")


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

    MODES = ("real", "frozen", "shuffled", "noise")

    def __init__(self, n_envs: int, history_len: int, mode: str = "real",
                 seed: int = 0):
        self.t = max(1, int(history_len))
        self.buf = [deque(maxlen=self.t) for _ in range(n_envs)]
        if mode not in self.MODES:
            raise SystemExit(f"--history_mode must be one of {self.MODES}")
        self.mode = mode
        self.rng = np.random.default_rng(seed)

    def reset(self, i: int, state: np.ndarray):
        self.buf[i].clear()
        for _ in range(self.t):
            self.buf[i].append(np.asarray(state, dtype=np.float32))

    def push(self, i: int, state: np.ndarray):
        self.buf[i].append(np.asarray(state, dtype=np.float32))

    def stack(self) -> np.ndarray:
        """-> (n_envs, T, D), after whatever ablation `mode` asks for.

        The one place the window is assembled, so the one place to intervene.
        See ARCHITECTURE.md 8.2: the model can form `s_t - s_{t-1}` from this
        window, and under a position controller that difference IS the
        previously executed action. Demos are smooth, so extrapolating it
        explains the near horizon without reading the image -- and at
        `n_action_steps=2` the near horizon is the ONLY part ever executed.

          real      untouched.
          frozen    newest frame repeated. Velocity is identically zero. This
                    is NOT out of distribution: `reset` builds exactly this,
                    so every episode's first inference call already sees it.
          shuffled  the OLDER T-1 frames permuted; ordering dies, every
                    marginal survives.
          noise     the older T-1 frames replaced by the newest plus Gaussian
                    noise at the real window's own per-dim std. Motion
                    MAGNITUDE is preserved, direction is gone, and -- unlike
                    `frozen` -- the window makes no coherent claim.

        EVERY mode leaves frame -1 untouched, because the state token is
        `st[:, -1]` (wiltechs_x_model, _suffix_pass). A permutation over all T
        moves an older frame into that slot and displaces the CURRENT
        proprioceptive reading by up to T-1 frames, which is a second
        intervention on top of the intended one. Results taken before this was
        fixed under-state nothing -- they destroyed ordering AND the state
        token -- but they cannot be read against `frozen`, which never had the
        defect.

        Why three: `frozen` alone is ambiguous. It does not merely remove the
        signal, it asserts a self-consistent falsehood ("this arm has been
        still for T frames") that a phase detector can lock onto, so a
        collapse under it can mean either "the signal was needed" or "the lie
        selected the wrong mode". `noise` carries variance without either a
        true velocity or that assertion, and separates the two.
        """
        out = np.stack([np.stack(list(b)) for b in self.buf])
        if self.mode == "frozen":
            out[:] = out[:, -1:, :]
        elif self.mode == "shuffled" and self.t > 1:
            for i in range(len(out)):
                out[i, :-1] = out[i][self.rng.permutation(self.t - 1)]
        elif self.mode == "noise" and self.t > 1:
            sd = out.std(axis=1, keepdims=True)
            jitter = self.rng.normal(0.0, 1.0, out.shape) * sd
            out[:, :-1] = (out[:, -1:, :] + jitter[:, :-1]).astype(out.dtype)
        return out


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_policy(ckpt: Path, device: str, num_inference_steps: int | None,
                n_action_steps: int | None = None,
                fixed_episode_noise: bool = False,
                sample_noise_scale: float | None = None):
    from lerobot.configs.policies import PreTrainedConfig

    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.device = str(device)
    # Guarded: these are flow-policy knobs and not every sibling has them.
    # Setting an attribute a config does not declare would silently do nothing
    # on a dataclass with slots, or silently invent a field on one without.
    def _set(name, value, flag):
        if not hasattr(cfg, name):
            raise SystemExit(
                f"{flag} was passed, but {type(cfg).__name__} has no "
                f"'{name}'. That knob does not exist for this policy.")
        setattr(cfg, name, value)

    if num_inference_steps:
        _set("num_inference_steps", int(num_inference_steps),
             "--num_inference_steps")
    if n_action_steps:
        n = int(n_action_steps)
        if n > int(cfg.horizon):
            raise SystemExit(
                f"--n_action_steps {n} exceeds the trained horizon "
                f"{cfg.horizon}: the chunk has no steps past that to execute.")
        cfg.n_action_steps = n
    if fixed_episode_noise:
        _set("fixed_episode_noise", True, "--fixed_episode_noise")
    if sample_noise_scale is not None:
        # Temperature on x_1. NOT the same experiment as
        # --fixed_episode_noise: that commits to one RANDOM draw, this moves
        # every draw toward the centre of the policy's distribution. Fixing
        # the noise cost 25 points here, which says the per-chunk lottery is
        # rescuing episodes -- but a lottery only helps when the distribution
        # is too broad, and shrinking it is the other way to answer that.
        _set("sample_noise_scale", float(sample_noise_scale),
             "--sample_noise_scale")
    kind = getattr(cfg, "type", None) or getattr(cfg, "name", "")
    print(f"policy type: {kind}")
    policy = _policy_class(kind).from_pretrained(ckpt, config=cfg)
    policy.to(device)
    policy.eval()
    for m in policy.model.modules():                      # deterministic rollout
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    return policy


def report_missing_weights(policy, ckpt: Path, allow: bool):
    """Account for the tensors the checkpoint did not supply -- by requires_grad.

    lerobot logs these as one `WARNING:root:Missing key(s)` line and carries on,
    which can mean an eval silently scores a DIFFERENT model than the one that
    was trained. But "missing" alone is not the signal, and the first version of
    this function got that wrong: it flagged all 714 frozen Qwen3-VL tensors of
    a WiltechsMoE checkpoint and would have refused to run.

    Those are missing BY DESIGN. train_wiltechs_moe strips `model.vlm_model.*`
    at save time, and says why: "the encoder is always loaded by
    from_pretrained(model_id) before this point, so the checkpoint's copy is
    redundant either way". Their values in state_dict() are the correct
    pretrained weights, not an initialisation.

    So the discriminator is requires_grad, not presence:

      frozen and missing     expected -- the value came from the pretrained
                             source at construction. Summarised, not listed.
      TRAINABLE and missing  a learned weight sitting at its init. THAT is the
                             alarm, and whether it matters depends on what the
                             init is: zero contributes nothing, anything else
                             changes the model.

    The case that prompted the whole check is the benign end of that:
    `model.robot_pos_gate` is trainable, arrived in 4c06db6 (2026-08-08) after
    the checkpoint that scored 92% (2026-08-04), and is
    nn.Parameter(torch.zeros(1)) multiplying an additive term -- so at 0 it is
    exactly the model that was trained. A norm weight initialised to 1.0 would
    have looked identical in the log and would not have been.
    """
    try:
        from safetensors import safe_open
    except ImportError:
        return
    files = sorted(ckpt.glob("*.safetensors"))
    if not files:
        return
    have = set()
    for f in files:
        with safe_open(f, framework="pt") as fh:
            have |= set(fh.keys())
    sd = policy.state_dict()
    if len(have & set(sd)) < 0.2 * len(sd):
        print(f"[weights] key naming does not line up with the checkpoint "
              f"({len(have & set(sd))}/{len(sd)} matched); skipping the audit.")
        return
    grads = {k: p.requires_grad for k, p in policy.named_parameters()}
    missing = sorted(k for k in sd if k not in have)
    if not missing:
        return
    # Buffers are not parameters; grads.get(...) is False for them, which puts
    # them on the expected side. That is right -- they are constants or caches.
    frozen = [k for k in missing if not grads.get(k, False)]
    live = [(k, float(sd[k].detach().abs().max()))
            for k in missing if grads.get(k, False)]

    if frozen:
        n = sum(sd[k].numel() for k in frozen)
        print(f"[weights] {len(frozen)} FROZEN tensor(s) / {n/1e6:.0f}M params not "
              f"in the checkpoint -- expected: a frozen backbone is loaded at "
              f"construction, so the checkpoint does not carry a second copy.")
    if not live:
        print("[weights] every TRAINABLE tensor was supplied. Good.")
        return
    print(f"[weights] {len(live)} TRAINABLE tensor(s) not in the checkpoint "
          f"-- each keeps its INITIALISATION:")
    for k, m in live:
        print(f"    {k:<50s} {sd[k].numel():>10,} el   |max| {m:.3e}"
              + ("   inert (exactly 0)" if m == 0.0 else "   *** NONZERO ***"))
    if all(m == 0.0 for _, m in live):
        print("  All zero, so they contribute nothing: this is numerically the "
              "model the checkpoint was trained as.")
        return
    msg = ("Some are NONZERO, so this is not the model that was trained and the "
           "number below would not be comparable to anything.")
    if not allow:
        raise SystemExit(f"  {msg}\n  Pass --allow_missing_weights to score it "
                         f"anyway, knowing that.")
    print(f"  {msg}  --allow_missing_weights was passed; continuing.")


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
def state_scale(preprocessor):
    """Per-dimension std of observation.state, for printing sigma in real units.

    Best effort: the pipeline's shape is lerobot's, not ours. Returns None
    rather than guessing, and the banner then reports normalized units only.
    """
    for step in getattr(preprocessor, "steps", []) or []:
        stats = getattr(step, "stats", None)
        if isinstance(stats, dict) and "observation.state" in stats:
            s = stats["observation.state"]
            for key in ("std", "max"):
                v = s.get(key) if isinstance(s, dict) else getattr(s, key, None)
                if v is not None:
                    return np.asarray(v, dtype=float), key
    return None


def blur_images(batch: dict, factor: int, cams=None) -> int:
    """Destroy detail finer than `factor` pixels, keeping every shape identical.

    Downsample then upsample back, so the ViT still sees the same resolution
    and produces the same token count -- only the information below `factor`
    pixels is gone. That is what makes this a clean test of whether the policy
    USES fine visual detail: an image at a genuinely lower resolution would
    also change the token grid, and then a drop could not be attributed.

    Flat success rate under factor 2 means adding vision capacity (a finer
    DINO path, a larger vision_input_size, LoRA on the tower) is buying
    resolution the policy already declines to use.
    """
    n = 0
    for k in list(batch):
        if not k.startswith("observation.image"):
            continue
        if cams and not any(c in k for c in cams):
            continue
        v = batch[k]
        if not torch.is_tensor(v) or v.dim() < 3:
            continue
        flat = v.reshape(-1, *v.shape[-3:]) if v.dim() > 4 else v
        h, w = flat.shape[-2:]
        small = torch.nn.functional.interpolate(
            flat, size=(max(1, h // factor), max(1, w // factor)),
            mode="area")
        back = torch.nn.functional.interpolate(
            small, size=(h, w), mode="bilinear", align_corners=False)
        batch[k] = back.reshape(v.shape)
        n += 1
    return n


def build_batch(obs_list, tasks, hist: StateHistory, preprocessor, device,
                state_noise: float = 0.0, state_noise_dims=None,
                blur: int = 0, blur_cams=None):
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
    # Before the preprocessor, on the raw [0, 1] frames: that is where "detail
    # finer than N pixels" is a statement about the camera rather than about
    # whatever affine the normalizer applies.
    if blur > 1:
        blur_images(batch, blur, blur_cams)
    batch = preprocessor(batch)

    if state_noise > 0.0:
        # AFTER the preprocessor, so sigma is in the same normalized units the
        # sibling trainers use (apply_joint_augmentations: randn * 0.02). That
        # makes this measurement directly answer "what sigma should training
        # use", instead of needing a unit conversion to be trusted.
        #
        # ONE offset for the whole (B, T, D) window, not per frame: the motion
        # encoder reads differences, so independent per-frame noise would inject
        # a velocity spike that the real failure mode -- being a few millimetres
        # off -- does not produce. A constant offset leaves every difference
        # unchanged and moves only the position.
        s = batch["observation.state"]
        off = torch.randn(s.shape[0], 1, s.shape[-1], device=s.device,
                          dtype=s.dtype) * state_noise
        if state_noise_dims is not None:
            keep = torch.zeros(s.shape[-1], device=s.device, dtype=s.dtype)
            keep[list(state_noise_dims)] = 1.0
            off = off * keep
        batch["observation.state"] = s + off.expand_as(s)
    return batch


@torch.no_grad()
def eval_task(policy, preprocessor, postprocessor, suite, suite_name: str,
              task_id: int, episodes: int, num_envs: int, device: str,
              max_episode_steps: int, seed: int, expected_cams: list[str],
              policy_seed: int | None = None,
              video_cb=None, videos_per_task: int = 0, heartbeat: int = 50,
              instruction: str | None = None, state_noise: float = 0.0,
              state_noise_dims=None, blur: int = 0, blur_cams=None,
              history_mode: str = "real"):
    """-> (n_success, n_episodes, mean_success_steps, n_chunks, task_description,
    per_episode_success).

    The per-episode vector is what makes two checkpoints COMPARABLE. Episode i
    starts from the same canonical init state in every run (fixed_init_states),
    so two evals are a PAIRED sample and McNemar applies. Comparing only the
    two rates throws that away and leaves ~15pp of unpaired noise at n=20 --
    enough to invent a 20-point "regression" between adjacent checkpoints.
    """
    from lerobot.envs.libero import LiberoEnv

    # Re-seed PER TASK, not once per process. The policy draws its flow noise
    # from the global torch RNG, which advances with every chunk, so a run of
    # `--task_ids 4 0 5` reached task 0 with thousands of draws already spent
    # and gave it a different noise stream than a run of `--task_ids 0 ...`.
    # That is not a subtle effect: it produced task 0 at 45% in one ordering
    # and 85% in another, on the SAME checkpoint -- a 40-point artefact that
    # reads as a result.
    #
    # Keyed on task_id so ordering, and which tasks are in the run at all,
    # cannot reach the noise. Two runs are then paired on BOTH the layout and
    # the noise: episode i sees the same x_1 sequence in both, and diverges
    # only where the policy itself does.
    #
    # `policy_seed` is separate from `seed` on purpose: `seed` picks the
    # LAYOUTS (env.reset below) and policy_seed picks the flow noise. Holding
    # the first and moving the second is the null control for every A/B run
    # through this script -- how many episodes change outcome when NOTHING
    # about the policy or its inputs changed, only the sampled x_1. Without
    # that number, "state noise flipped 31 of 60 episodes" cannot be told
    # apart from "this policy flips 31 of 60 episodes on its own".
    ps = seed if policy_seed is None else policy_seed
    torch.manual_seed(ps + task_id)
    np.random.seed((ps + task_id) % (2 ** 32))

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
        # Seeded off the task, like the policy noise, so `shuffled` is a
        # reproducible intervention rather than a fresh coin every run.
        hist = StateHistory(num_envs, policy.config.n_obs_steps,
                            mode=history_mode, seed=ps + task_id)

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
                                    preprocessor, device,
                                    state_noise, state_noise_dims,
                                    blur, blur_cams)
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
    p.add_argument("--sample_noise_scale", type=float, default=None,
                   help="Temperature on the initial flow noise x_1 (trained "
                        "value 1.0). Below 1 pulls every sample toward the "
                        "centre of the action distribution; 0 makes the policy "
                        "deterministic. Distinct from --fixed_episode_noise, "
                        "which commits to one RANDOM draw rather than moving "
                        "toward the middle.")
    p.add_argument("--policy_seed", type=int, default=None,
                   help="Seed for the POLICY's flow noise, separate from "
                        "--seed which picks the layouts. Defaults to --seed. "
                        "Re-running with only this changed is the null control: "
                        "same checkpoint, same layouts, same inputs, different "
                        "x_1. However many episodes flip is the floor that any "
                        "--state_noise or --image_blur delta has to clear.")
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
    p.add_argument("--allow_missing_weights", action="store_true",
                   help="Score a checkpoint that does not supply every tensor "
                        "the current code declares. Refused by default when any "
                        "of them initialises NONZERO, because then the loaded "
                        "model is not the one that was trained and the number "
                        "compares to nothing. Zero-init tensors contribute "
                        "nothing and are allowed without this.")
    p.add_argument("--history_mode", default="real",
                   choices=("real", "frozen", "shuffled", "noise"),
                   help="Ablate the observation.state WINDOW the motion-vector "
                        "encoder reads. 'shuffled' permutes the T frames per "
                        "call: every marginal is preserved and only the "
                        "temporal order dies, so a drop cannot be blamed on "
                        "unfamiliar values -- it is the control for "
                        "ARCHITECTURE.md 8.2 (does this policy ride the "
                        "demonstrator's momentum instead of the image?). "
                        "'frozen' repeats the newest frame, which is what "
                        "every episode's first step already looks like. "
                        "Read 'shuffled'; 'frozen' alone is ambiguous.")
    p.add_argument("--state_noise", type=float, default=0.0,
                   help="Gaussian offset added to observation.state, sigma in "
                        "NORMALIZED units -- the same units the sibling trainers "
                        "augment in (train_wiltechs_moe uses 0.02). Sweep it to "
                        "find out whether the policy is brittle to being a few "
                        "millimetres off, which is what a fumbled grasp is. Flat "
                        "SR means state augmentation buys nothing; a collapse "
                        "means it does, and the collapse point is the sigma to "
                        "train at. One offset per episode-window, so the motion "
                        "differences are untouched.")
    p.add_argument("--state_noise_dims", nargs="+", type=int, default=None,
                   help="Restrict --state_noise to these state indices, e.g. "
                        "0 1 2 for end-effector position only. Default: all "
                        "dims, matching the sibling trainers.")
    p.add_argument("--image_blur", type=int, default=0,
                   help="Downsample the camera frames by this factor and "
                        "upsample back, destroying detail finer than N pixels "
                        "while keeping the token grid identical. The mirror of "
                        "--state_noise: flat SR under factor 2 means the policy "
                        "does not use fine visual detail, so a finer DINO path, "
                        "a larger --vision_input_size, or LoRA on the vision "
                        "tower is buying resolution it declines to read.")
    p.add_argument("--image_blur_cams", nargs="+", default=None,
                   help="Restrict --image_blur to camera keys containing these "
                        "substrings, e.g. image2 for the wrist view alone. "
                        "Default: every camera.")
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
    #
    # This seeds the setup; eval_task re-seeds per task so that TASK ORDER
    # cannot reach the noise. See the comment there.
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
                         a.n_action_steps, a.fixed_episode_noise,
                         a.sample_noise_scale)
    report_missing_weights(policy, ckpt, a.allow_missing_weights)
    pre, post = load_processors(ckpt, device, a.dataset_id)
    cams = _policy_cameras(policy.config)
    print(f"[eval] {ckpt}  device={device}  cameras={cams}\n"
          f"[eval] horizon={policy.config.horizon} "
          f"n_action_steps={policy.config.n_action_steps} "
          f"NFE={policy.config.num_inference_steps} "
          f"state_history={policy.config.n_obs_steps} "
          f"noise={'fixed/episode' if a.fixed_episode_noise else 'per-chunk'}")

    if a.state_noise > 0.0:
        # Report the physical size too. A sigma the arm cannot actually be off
        # by measures nothing, and one large enough to contradict the camera is
        # measuring a broken observation rather than a brittle policy.
        sc = state_scale(pre)
        dims = a.state_noise_dims if a.state_noise_dims is not None else "all"
        phys = ""
        if sc is not None:
            scale, kind = sc
            idx = (a.state_noise_dims if a.state_noise_dims is not None
                   else range(min(3, len(scale))))
            vals = [a.state_noise * float(scale[i]) for i in idx if i < len(scale)]
            if vals:
                phys = (f"  ~= {min(vals) * 1000:.1f}-{max(vals) * 1000:.1f} mm "
                        f"on dims {list(idx)} (from dataset {kind})")
        print(f"[eval] STATE NOISE sigma={a.state_noise} normalized on dims "
              f"{dims}{phys}\n"
              f"       One offset per window, so motion differences are "
              f"unchanged. This is a DIAGNOSTIC: flat SR means state "
              f"augmentation buys nothing.")

    if a.image_blur > 1:
        print(f"[eval] IMAGE BLUR x{a.image_blur} on "
              f"{a.image_blur_cams or 'every camera'} -- detail finer than "
              f"~{a.image_blur} px is gone, token grid unchanged.\n"
              f"       DIAGNOSTIC: flat SR means more vision resolution is not "
              f"the missing ingredient.")

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
                a.num_envs, device, a.max_episode_steps, a.seed, cams,
                a.policy_seed, video_cb,
                a.videos_per_task, a.heartbeat, wrong.get(tid),
                a.state_noise, a.state_noise_dims,
                a.image_blur, a.image_blur_cams, a.history_mode)
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
               "policy_seed": a.policy_seed,
               # Not cosmetic: the policy draws one noise tensor of shape
               # (num_envs, ...) per chunk, so changing num_envs changes the
               # RNG stream and silently unpairs two runs.
               "num_envs": a.num_envs,
               "max_episode_steps": a.max_episode_steps,
               "eval_commit": _git_commit(),
               "num_inference_steps": getattr(policy.config, "num_inference_steps", None),
               "n_action_steps": policy.config.n_action_steps,
               "fixed_episode_noise": bool(a.fixed_episode_noise),
               "policy_type": getattr(policy.config, "type", None),
               "sample_noise_scale": getattr(
                   policy.config, "sample_noise_scale", None),
               "state_noise": a.state_noise,
               "history_mode": a.history_mode,
               "state_noise_dims": a.state_noise_dims,
               "image_blur": a.image_blur,
               "image_blur_cams": a.image_blur_cams,
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
                    else f"eval_libero_statenoise_{a.state_noise:g}.json"
                    if a.state_noise > 0.0
                    else f"eval_libero_blur_{a.image_blur}.json"
                    if a.image_blur > 1
                    else f"eval_libero_temp_{a.sample_noise_scale:g}.json"
                    if a.sample_noise_scale is not None
                    else f"eval_libero_pseed_{a.policy_seed}.json"
                    if a.policy_seed is not None and a.policy_seed != a.seed
                    else "eval_libero.json")
    out = Path(a.out) if a.out else ckpt / default_name
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
