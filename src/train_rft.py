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

# NOTE: do NOT add `from __future__ import annotations` here. lerobot's
# parser.wrap reads the RAW function annotation (argspec.annotations[...]) to
# get the config dataclass; deferred annotations would turn `RFTConfig` into a
# string and draccus would fail with "must be called with a dataclass type".

import importlib
import os
import random
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import nullcontext

import huggingface_hub
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from safetensors.torch import load_file as load_safetensors

# JPEG-compress buffered frames to keep system RAM bounded (~10-15× vs raw
# uint8). Falls back to raw uint8 tensors if opencv is unavailable.
try:
    import cv2
    _USE_JPEG = True
except Exception:
    _USE_JPEG = False

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device

# --- Register the custom policy/config/processor types ----------------------
# Run as a fresh `python src/train_rft.py` process, nothing has imported the
# models yet, so draccus can't parse `--policy.path=<wilro ckpt>`
# ("Couldn't find a choice class for 'wilro'"). Importing the config modules
# runs their @PreTrainedConfig.register_subclass decorators; importing the
# policy/processor modules makes their classes + any custom processor steps
# available. Must happen at import time, BEFORE parser.wrap() parses argv.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # ensure src/ importable

# (config_module, policy_module, ClassName) keyed by the registered policy type.
_POLICY_CLASS: dict[str, tuple[str, str, str]] = {
    "wilro": (
        "models.wilro.wilro_config",
        "models.wilro.wilro_policy",
        "WilroPolicy",
    ),
    "interleaved_flow_matching": (
        "models.interleaved_flow_matching.interleaved_flow_matching_config",
        "models.interleaved_flow_matching.interleaved_flow_matching_policy",
        "InterleavedFlowMatchingPolicy",
    ),
    "wiltechs_vla": (
        "models.wiltechs_vla.wiltechs_vla_config",
        "models.wiltechs_vla.wiltechs_vla_policy",
        "WiltechsVLAPolicy",
    ),
}
for _cfg_mod, _pol_mod, _ in _POLICY_CLASS.values():
    try:
        importlib.import_module(_cfg_mod)   # registers the config choice for draccus
        importlib.import_module(_pol_mod)   # makes the policy class importable
    except Exception as _e:                 # e.g. wiltechs_vla's Qwen3-VL deps absent
        print(f"[train_rft] could not register '{_cfg_mod}' ({type(_e).__name__}: {_e})")


# ---------------------------------------------------------------------------
# Config: EvalPipelineConfig (env / policy / eval) + an `rft` block.
# Reuses lerobot's parser so --policy.path / --env.* behave exactly as in eval.
# ---------------------------------------------------------------------------
# --- LIBERO / robosuite 1.4.1 compatibility (mirrors benchmark_libero.py) ---
# lerobot.envs.libero builds robosuite envs that expect symbols robosuite 1.4.x
# dropped. These patches must be applied BEFORE make_env constructs the env.
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


def _start_virtual_display():
    """Headless rendering for Colab (Xvfb + EGL). No-op if a display already exists."""
    try:
        from pyvirtualdisplay import Display
        Display(visible=0, size=(1400, 900)).start()
    except Exception as e:
        print(f"[train_rft] virtual display not started ({type(e).__name__}: {e}); "
              f"assuming one already exists.")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("DISPLAY", ":99")


@dataclass
class RFTParams:
    headless: bool = True             # start Xvfb + EGL (Colab); set --rft.headless=false if you have a display
    iterations: int = 50              # collect→train cycles
    rollouts_per_task: int = 16       # episodes to roll out per task per iteration
    updates_per_iter: int = 400       # optimizer steps per iteration
    train_batch_size: int = 16
    lr: float = 1e-5
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    max_steps: int = 0                # cap rollout length (0 = use the env's per-suite default)
    buffer_size: int = 8000           # max (obs, action-chunk) samples retained (~0.4GB RAM w/ JPEG)
    min_buffer_to_train: int = 256    # don't train until the buffer has this many
    save_freq: int = 2                # save a checkpoint every N iterations (best != last, so save often)
    keep_last: int = 5                # keep only the N most recent checkpoint-<step> dirs (0 = keep all)
    output_dir: str = "outputs/rft/run"
    # --- demo anchoring (collapse prevention) ---
    # Mix expert-demo batches into training so the policy can't drift/forget while
    # self-imitating its own successes. 0 = off. ~0.3 = 30% of updates use demos.
    demo_dataset: str = ""            # e.g. "lerobot/libero" (must share the policy's obs/action space)
    demo_fraction: float = 0.0        # probability each update trains on a demo batch instead of RFT


@dataclass
class RFTConfig(EvalPipelineConfig):
    rft: RFTParams = field(default_factory=RFTParams)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _encode_img(t: torch.Tensor):
    """(C,H,W) float[0,1] → compact JPEG buffer (np.uint8 1-D). Falls back to a
    uint8 tensor when opencv is absent or the shape is unexpected."""
    if _USE_JPEG and t.dim() == 3 and t.shape[0] in (1, 3):
        arr = (t.clamp(0, 1).permute(1, 2, 0).contiguous().numpy() * 255).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if ok:
            return buf  # cv2 BGR convention round-trips to itself (channel order preserved)
    return (t.clamp(0, 1) * 255).round().to(torch.uint8)


def _decode_img(v) -> torch.Tensor:
    """Inverse of _encode_img → (C,H,W) float[0,1]."""
    if isinstance(v, np.ndarray):                      # JPEG buffer
        arr = cv2.imdecode(v, cv2.IMREAD_COLOR)        # HWC uint8
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous().float() / 255.0
    return v.float() / 255.0                            # uint8 tensor fallback


def _slice_obs(obs: dict, i: int) -> dict:
    """Extract env i's slice of a batched lerobot obs dict, on CPU.

    Image frames are JPEG-compressed (~10-15× less RAM than raw uint8) so the
    success buffer doesn't blow system memory; everything else stays float.
    `task` is a per-env list of strings.
    """
    out = {}
    for k, v in obs.items():
        if k == "task":
            out[k] = v[i] if isinstance(v, (list, tuple)) else v
        elif isinstance(v, torch.Tensor):
            t = v[i].detach().to("cpu")
            out[k] = _encode_img(t) if "image" in k else t
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
def _rft_rollout(env, policy, preprocessor, postprocessor, device, action_dim,
                 desc="", max_steps_cap=0):
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
    if max_steps_cap > 0:
        max_steps = min(max_steps, max_steps_cap)   # --rft.max_steps: stop early (episodes
        #   not done by here are dropped — they almost never succeed later for this policy)

    samples: list = []
    n_success = 0
    horizon = policy.config.horizon
    step = 0
    pbar = tqdm(total=max_steps, desc=desc, leave=False, dynamic_ncols=True)
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
        pbar.update(1)
        pbar.set_postfix(done=f"{int(done.sum())}/{B}", success=n_success)

    pbar.close()
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
                t = _decode_img(t)                      # JPEG/uint8 → float [0,1]
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


def _build_demo_dataset(demo_id, horizon, env_cam_keys, env_state_key):
    """Load the expert-demo dataset for anchoring. Returns (LeRobotDataset, info).
    Demo cameras are positionally mapped to the env's obs keys so demo batches
    flow through the SAME preprocessor as RFT obs."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.configs.types import FeatureType

    meta = LeRobotDatasetMetadata(demo_id, revision="main")
    feats = dataset_to_policy_features(meta.features)
    cam_keys = sorted(k for k, f in feats.items() if f.type == FeatureType.VISUAL)
    state_key = next(k for k, f in feats.items() if f.type == FeatureType.STATE)
    action_key = next(k for k, f in feats.items() if f.type == FeatureType.ACTION)
    fps = getattr(meta, "fps", None) or 10
    ft = 1.0 / fps
    delta = {action_key: [k * ft for k in range(horizon)], state_key: [0.0]}
    for c in cam_keys:
        delta[c] = [0.0]
    ds = LeRobotDataset(demo_id, delta_timestamps=delta, revision="main")
    cam_map = {cam_keys[i]: env_cam_keys[i] for i in range(min(len(cam_keys), len(env_cam_keys)))}
    a_mean = torch.tensor(np.asarray(meta.stats["action"]["mean"], dtype=np.float32)).reshape(-1)
    a_std = torch.tensor(np.asarray(meta.stats["action"]["std"], dtype=np.float32)).reshape(-1).clamp_min(1e-6)
    info = dict(cam_map=cam_map, state_key=state_key, action_key=action_key,
                env_state_key=env_state_key, a_mean=a_mean, a_std=a_std, n=len(ds))
    print(f"[demo] {demo_id}: {len(ds)} frames | cam_map={cam_map} | "
          f"state '{state_key}'→'{env_state_key}' | action '{action_key}' dim={a_mean.numel()}",
          flush=True)
    return ds, info


def _collate_demo(ds, info, bs, preprocessor, device, action_dim):
    """One expert-demo training batch in the SAME normalized space as _collate:
    raw obs (under env keys) → preprocessor; action normalized with demo stats."""
    cam_map, sk, ak, esk = info["cam_map"], info["state_key"], info["action_key"], info["env_state_key"]
    img_acc = {ev: [] for ev in cam_map.values()}
    state_acc, act_acc, pad_acc, tasks = [], [], [], []
    for _ in range(bs):
        item = ds[random.randint(0, info["n"] - 1)]
        for dc, ev in cam_map.items():
            img = item[dc]
            img = img[-1] if img.dim() == 4 else img          # single current frame
            img_acc[ev].append(img.float())
        st = item[sk]
        state_acc.append((st[-1] if st.dim() == 2 else st).float())
        act_acc.append(item[ak].float())                       # (horizon, action_dim) raw
        pad = item.get("action_is_pad")
        pad_acc.append(pad.bool() if isinstance(pad, torch.Tensor)
                       else torch.zeros(act_acc[-1].shape[0], dtype=torch.bool))
        t = item.get("task", item.get("task_description", ""))
        tasks.append(t if isinstance(t, str) else "")

    batch = {ev: torch.stack(lst, 0).to(device) for ev, lst in img_acc.items()}
    batch[esk] = torch.stack(state_acc, 0).to(device)
    batch["task"] = tasks
    batch = preprocessor(batch)                                # normalize obs (action absent)
    a = torch.stack(act_acc, 0).to(device)
    a = (a - info["a_mean"].to(device)) / info["a_std"].to(device)   # normalize action w/ demo stats
    batch["action"] = a
    batch["action_is_pad"] = torch.stack(pad_acc, 0).to(device)
    batch["action_dim_pad"] = torch.zeros(a.shape[0], action_dim, dtype=torch.bool, device=device)
    return batch


def _train_on_buffer(policy, optimizer, buffer, cfg: RFTParams, preprocessor,
                     device, action_dim, demo_ds=None, demo_info=None):
    policy.train()
    trainable = [p for p in policy.parameters() if p.requires_grad]
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if device.type == "cuda" else nullcontext()
    )
    running, n_demo = 0.0, 0
    for _ in range(cfg.updates_per_iter):
        # Demo anchoring: with prob demo_fraction, train on an expert-demo batch
        # instead of an RFT-success batch — keeps the policy from drifting/forgetting.
        if demo_ds is not None and cfg.demo_fraction > 0 and random.random() < cfg.demo_fraction:
            batch = _collate_demo(demo_ds, demo_info, cfg.train_batch_size, preprocessor, device, action_dim)
            n_demo += 1
        else:
            picks = random.sample(buffer, min(cfg.train_batch_size, len(buffer)))
            batch = _collate(picks, preprocessor, device, action_dim)
        with autocast_ctx:
            loss, _ = policy.forward(batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
        optimizer.step()
        running += loss.item()
    return running / max(1, cfg.updates_per_iter), n_demo


def _load_policy(cfg, device):
    """Build the custom policy from the (already CLI-overridden) cfg.policy and
    load its checkpoint weights. Bypasses lerobot's make_policy/get_policy_class,
    which is a hardcoded registry with no entry for wilro/interleaved/wiltechs.

    Constructing from cfg.policy (not from_pretrained(path)) preserves CLI
    overrides like --policy.n_action_steps=2; weights are then loaded by shape
    match (robust to token-count / minor config changes).
    """
    ptype = cfg.policy.type
    if ptype not in _POLICY_CLASS:
        raise ValueError(f"Unsupported policy type '{ptype}'. Known: {list(_POLICY_CLASS)}")
    _, pol_mod, cls_name = _POLICY_CLASS[ptype]
    PolicyCls = getattr(importlib.import_module(pol_mod), cls_name)

    policy = PolicyCls(cfg.policy)

    path = Path(str(cfg.policy.pretrained_path))
    local = path if path.exists() else Path(huggingface_hub.snapshot_download(str(path)))
    model_file = local / "model.safetensors"
    if not model_file.exists():
        cands = list(local.glob("*.safetensors"))
        if not cands:
            raise FileNotFoundError(f"No .safetensors found in {local}")
        model_file = cands[0]

    ckpt_state = load_safetensors(str(model_file), device="cpu")
    cur = policy.state_dict()
    filtered = {k: v for k, v in ckpt_state.items() if k in cur and cur[k].shape == v.shape}
    skipped = [k for k in ckpt_state if k not in filtered]
    missing = [k for k in cur if k not in filtered]
    policy.load_state_dict(filtered, strict=False)
    print(f"Loaded {cls_name}: {len(filtered)}/{len(cur)} keys "
          f"(skipped {len(skipped)}, missing {len(missing)}) from {model_file}")
    return policy


def _save(policy, preprocessor, postprocessor, out_dir: Path, name: str, step: int = 0):
    ckpt = out_dir / name
    ckpt.mkdir(parents=True, exist_ok=True)
    policy.config.training_step = step
    policy.save_pretrained(ckpt)
    preprocessor.save_pretrained(ckpt)
    postprocessor.save_pretrained(ckpt)
    print(f"  [save] {ckpt}", flush=True)


def _prune_checkpoints(out_dir: Path, keep_last: int):
    """Keep only the `keep_last` most recent checkpoint-<step> dirs (never touches
    checkpoint-best). Lets you run --rft.save_freq=1 without filling the disk."""
    if keep_last <= 0:
        return
    ckpts = []
    for p in out_dir.glob("checkpoint-*"):
        if p.is_dir() and p.name != "checkpoint-best":
            try:
                ckpts.append((int(p.name.split("-")[1]), p))
            except (IndexError, ValueError):
                pass
    ckpts.sort()
    for _, p in ckpts[:-keep_last]:
        import shutil
        shutil.rmtree(p, ignore_errors=True)
        print(f"  [prune] removed {p.name}", flush=True)


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

    # LIBERO/robosuite setup must happen BEFORE make_env builds the env.
    if "libero" in str(cfg.env.type):
        if cfg.rft.headless:
            _start_virtual_display()
        _apply_robosuite_patches()

    # Env / policy / processors — identical construction to lerobot-eval.
    envs = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)
    policy = _load_policy(cfg, device)
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

    # Demo anchoring: build the demo dataset, mapping its cameras to the env's
    # raw obs keys so demo batches flow through the SAME preprocessor as RFT obs.
    demo_ds, demo_info = None, None
    if cfg.rft.demo_fraction > 0 and cfg.rft.demo_dataset:
        _obs0, _ = task_envs[0][1].reset()
        _obs0 = preprocess_observation(_obs0)
        env_cam_keys = sorted(k for k in _obs0 if "image" in k)
        env_state_key = "observation.state" if "observation.state" in _obs0 else next(
            (k for k in _obs0 if "state" in k), "observation.state")
        demo_ds, demo_info = _build_demo_dataset(
            cfg.rft.demo_dataset, policy.config.horizon, env_cam_keys, env_state_key,
        )

    print("\n" + "=" * 64)
    print("Reward-Filtered Fine-Tuning")
    print("=" * 64)
    print(f"  policy        : {cfg.policy.type}  (path={cfg.policy.pretrained_path})")
    print(f"  device        : {device}   (if 'cpu', rollouts will be ~unusably slow — pass --policy.device=cuda)")
    print(f"  trainable     : {n_train:,}   frozen: {n_frozen:,}")
    print(f"  env/task      : {cfg.env.type}/{cfg.env.task}  ({len(task_envs)} task env(s))")
    print(f"  n_action_steps: {policy.config.n_action_steps}  horizon: {policy.config.horizon}")
    print(f"  lr={cfg.rft.lr:.1e}  updates/iter={cfg.rft.updates_per_iter}  "
          f"train_bs={cfg.rft.train_batch_size}  buffer<={cfg.rft.buffer_size}")
    print(f"  max_steps     : {cfg.rft.max_steps if cfg.rft.max_steps > 0 else 'env default per suite'}")
    print(f"  demo anchor   : {cfg.rft.demo_dataset or 'off'}  (fraction={cfg.rft.demo_fraction})")
    print(f"  save          : every {cfg.rft.save_freq} iter, keep_last={cfg.rft.keep_last} + checkpoint-best")
    print("=" * 64 + "\n")

    buffer: deque = deque(maxlen=cfg.rft.buffer_size)
    global_step = 0
    best_succ = -1.0

    for it in range(1, cfg.rft.iterations + 1):
        # ---- 1) Collect successful trajectories --------------------------------
        ep_total, ep_success, new_samples = 0, 0, 0
        for t_idx, (label, env) in enumerate(task_envs):
            n_batches = -(-cfg.rft.rollouts_per_task // env.num_envs)  # ceil
            for b in range(n_batches):
                desc = f"iter{it} collect {label} [{t_idx+1}/{len(task_envs)}] b{b+1}/{n_batches}"
                samples, n_succ, B = _rft_rollout(
                    env, policy, preprocessor, postprocessor, device, action_dim,
                    desc=desc, max_steps_cap=cfg.rft.max_steps,
                )
                buffer.extend(samples)
                ep_total += B
                ep_success += n_succ
                new_samples += len(samples)
                print(f"  {desc}: {n_succ}/{B} success  +{len(samples)} samples  "
                      f"buffer={len(buffer)}", flush=True)
        succ_rate = 100.0 * ep_success / max(1, ep_total)
        print(f"[iter {it:3d}] rollout: {ep_success}/{ep_total} success "
              f"({succ_rate:4.1f}%)  +{new_samples} samples  buffer={len(buffer)}")

        # Save the policy that achieved the best in-pool success — done BEFORE this
        # iteration's training, so checkpoint-best is exactly the policy that
        # produced succ_rate (no off-by-one). NOTE: in-pool success; for the true
        # best, eval the kept checkpoints on held-out init states.
        if succ_rate > best_succ:
            best_succ = succ_rate
            print(f"           new best success {succ_rate:.1f}% → checkpoint-best", flush=True)
            _save(policy, preprocessor, postprocessor, out_dir, "checkpoint-best", step=global_step)

        # ---- 2) Reward-filtered BC on the success buffer -----------------------
        if len(buffer) < cfg.rft.min_buffer_to_train:
            print(f"           buffer < {cfg.rft.min_buffer_to_train}; skipping update "
                  f"(policy too weak — start RFT from a stronger checkpoint).")
            continue
        avg_loss, n_demo = _train_on_buffer(
            policy, optimizer, buffer, cfg.rft, preprocessor, device, action_dim,
            demo_ds=demo_ds, demo_info=demo_info,
        )
        global_step += cfg.rft.updates_per_iter
        print(f"           train : {cfg.rft.updates_per_iter} updates  "
              f"avg_loss={avg_loss:.4f}  demo_batches={n_demo}  (global_step={global_step})")

        if it % cfg.rft.save_freq == 0 or it == cfg.rft.iterations:
            _save(policy, preprocessor, postprocessor, out_dir, f"checkpoint-{global_step}", step=global_step)
            _prune_checkpoints(out_dir, cfg.rft.keep_last)

    print("\nRFT complete.")


if __name__ == "__main__":
    main()
