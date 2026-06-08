"""
Generic fine-tuning script supporting interleaved, wilro, and wiltechs_vla models.

Loads a pretrained checkpoint produced by `train_community.py` and fine-tunes it
on ONE OR MORE LeRobot v3 datasets. The model type is auto-detected from the
checkpoint's config.json (the `model_type` field), so no --model_type flag is
needed — just point at a checkpoint and the target dataset(s).

Each downstream dataset is projected into the same canonical schema via
`DatasetAdapter` (imported from `train_community`), so every pretrained weight
transfers directly — no `state_encoder` / `action_in` / `action_out`
re-initialisation. State dims > canonical are truncated; state dims < canonical
are zero-padded. Cameras are mapped semantically (then positional fallback),
missing canonical cameras are zero-filled, and every camera is letterbox-padded
+ resized to the pretrained vision_input_size.

Multiple datasets:
  Pass several ids to `--dataset_id A B C` (e.g. RFT-collected rollouts + the
  expert demos). They are concatenated via `StitchedDataset` and sampled
  frame-uniformly by an `EpisodeAwareSampler`. Normalization is PER-DATASET:
  each set is z-scored by ITS OWN native stats inside the adapter (the global
  preprocessor is identity), so heterogeneous action/state scales mix correctly
  — exactly the scheme `train_community.py` pretrains with. `--camera_map` is
  applied to every dataset that has those native camera names (with a per-dataset
  automatic fallback), so it assumes the datasets share a camera layout.

The LIBERO 90/10 per-task split via `--train_ratio < 1.0` is applied
independently within each dataset.

Key differences vs. pretraining:
  - Lower default LR (1e-5) and shorter warmup (200 steps).
  - Optimizer-state load is guarded: any model-key shape mismatch (e.g.
    different `num_cameras`, `horizon`, or stale `state_dim`) abandons the
    saved Adam momentum and restarts fresh instead of crashing inside
    `_foreach_lerp_`.

Usage:
    # Fine-tune a wilro checkpoint on LIBERO
    python src/train_finetune.py \
        --output_dir outputs/train/libero_wilro_finetune \
        --dataset_id lerobot/libero \
        --resume_from_checkpoint outputs/train/community_wilro/checkpoint-69000 \
        --camera_map 'image:front,image2:wrist' \
        --batch_size 24 \
        --training_steps 30000

    # Mix RFT-collected rollouts WITH the expert demos
    python src/train_finetune.py \
        --output_dir outputs/train/rft_sft \
        --dataset_id outputs/rft/collected_object lerobot/libero \
        --resume_from_checkpoint ISdept/Wilro-il-comm-38k \
        --camera_map 'image:front,image2:wrist' \
        --batch_size 24 \
        --training_steps 30000
"""
from pathlib import Path
import json
import torch
from tqdm import tqdm
import huggingface_hub
from safetensors.torch import load_file as load_safetensors
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import dataset_to_policy_features
import numpy as np

from torchvision.transforms import v2
from transformers import get_cosine_schedule_with_warmup

# Import shared components from the generic pretraining script.
from train_community import (
    CANONICAL_CAMERAS,
    CANONICAL_STATE_DIM,
    CANONICAL_ACTION_DIM,
    DatasetAdapter,
    StitchedDataset,
    build_camera_mapping,
    discover_state_action_dims,
    get_sub_dataset_ep_boundaries,
    load_task_descriptions,
    get_model_components,
)


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------
def get_augmentations():
    spatial = v2.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05), fill=0)
    color = v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08)
    blur = v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))], p=0.3)
    return v2.Compose([spatial, color, blur])


def apply_joint_augmentations(batch, state_key):
    if torch.rand(1).item() > 0.5 and state_key in batch:
        noise = torch.randn_like(batch[state_key]) * 0.02
        batch[state_key] = batch[state_key] + noise
    return batch


def apply_image_augmentations(batch, camera_keys, transform):
    present_keys = [k for k in camera_keys if k in batch and isinstance(batch[k], torch.Tensor)]
    if not present_keys:
        return batch

    B = batch[present_keys[0]].shape[0]
    for b in range(B):
        sample_img = batch[present_keys[0]][b]
        has_time_dim = sample_img.dim() == 4  # (T, C, H, W)
        if has_time_dim:
            T = sample_img.shape[0]
            stacked = torch.cat([batch[k][b] for k in present_keys], dim=0)
            stacked_aug = transform(stacked)
            for i, k in enumerate(present_keys):
                batch[k][b] = stacked_aug[i * T:(i + 1) * T]
        else:
            stacked = torch.stack([batch[k][b] for k in present_keys], dim=0)
            stacked_aug = transform(stacked)
            for i, k in enumerate(present_keys):
                batch[k][b] = stacked_aug[i]
    return batch


# ---------------------------------------------------------------------------
# Optional per-task train/test split
# ---------------------------------------------------------------------------
def get_per_task_train_episodes(hf_dataset, train_ratio=0.9):
    episode_ids = np.array(hf_dataset["episode_index"])
    task_ids = np.array(hf_dataset["task_index"])

    ep_to_task: dict[int, int] = {}
    for ep_idx, task_idx in zip(episode_ids, task_ids):
        ep_to_task[int(ep_idx)] = int(task_idx)

    task_to_episodes: dict[int, list[int]] = {}
    for ep_idx, task_idx in ep_to_task.items():
        task_to_episodes.setdefault(task_idx, []).append(ep_idx)

    train_episodes: set[int] = set()
    print(f"\nPer-task train/test split (train_ratio={train_ratio}):")
    for task_idx, episodes in sorted(task_to_episodes.items()):
        episodes = sorted(episodes)
        n_train = max(1, int(len(episodes) * train_ratio))
        train_episodes.update(episodes[:n_train])
        print(f"  task {task_idx:3d}: {len(episodes):3d} demos → {n_train} train | "
              f"{len(episodes) - n_train} test")
    print(f"  Total: {len(train_episodes)} train episodes\n")
    return train_episodes


# ---------------------------------------------------------------------------
# Helper used by gradient analysis logging.
# ---------------------------------------------------------------------------
def _grad_stats(policy, prefix):
    total, count = 0.0, 0
    for name, param in policy.model.named_parameters():
        if param.requires_grad and prefix in name and param.grad is not None:
            total += param.grad.abs().mean().item() * param.numel()
            count += param.numel()
    return (total / count, count) if count > 0 else (None, 0)


# ---------------------------------------------------------------------------
# Manual camera-map parser
# ---------------------------------------------------------------------------
def _resolve_camera_mapping(
    spec: str | None,
    native_keys: list[str],
    canon_keys: list[str],
):
    """Returns {canonical_full: native_full or None}."""
    def _build_lookup(keys):
        lut = {k.lower(): k for k in keys}
        lut.update({k.split(".")[-1].lower(): k for k in keys})
        return lut

    canon_lut = _build_lookup(canon_keys)
    native_lut = _build_lookup(native_keys)

    out: dict[str, str | None] = {c: None for c in canon_keys}

    if spec:
        for pair in spec.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if ":" not in pair:
                raise ValueError(f"--camera_map entry '{pair}' missing ':'")
            native_short, canon_short = (p.strip().lower() for p in pair.split(":", 1))
            if native_short not in native_lut:
                raise ValueError(
                    f"--camera_map: native camera '{native_short}' not in dataset. "
                    f"Available: {[k.split('.')[-1] for k in native_keys]}"
                )
            if canon_short not in canon_lut:
                raise ValueError(
                    f"--camera_map: canonical slot '{canon_short}' not in pretrained set. "
                    f"Available: {[k.split('.')[-1] for k in canon_keys]}"
                )
            canon_full = canon_lut[canon_short]
            native_full = native_lut[native_short]
            if out[canon_full] is not None:
                raise ValueError(f"--camera_map: slot '{canon_full}' assigned twice")
            out[canon_full] = native_full

    assigned_natives = {v for v in out.values() if v is not None}
    remaining_natives = sorted(n for n in native_keys if n not in assigned_natives)
    for canon in canon_keys:
        if out[canon] is None and remaining_natives:
            out[canon] = remaining_natives.pop(0)
    return out


def _resolve_camera_mapping_for_dataset(
    camera_map_spec, native_camera_keys, canon_cams_for_run, dataset_id,
):
    """Per-dataset camera mapping. Try the user --camera_map spec; if a dataset
    doesn't have those native cameras, fall back to automatic semantic+positional
    matching for that dataset (so heterogeneous sets don't hard-fail)."""
    if camera_map_spec:
        try:
            return _resolve_camera_mapping(
                camera_map_spec, native_camera_keys, canon_cams_for_run
            )
        except ValueError as e:
            print(f"  [camera_map] '{dataset_id}': spec didn't apply ({e}); "
                  f"falling back to automatic mapping for this dataset.")
    return build_camera_mapping(
        {dataset_id: set(native_camera_keys)}, canon_cams_for_run
    )[dataset_id]


# ---------------------------------------------------------------------------
# Detect model type from checkpoint config
# ---------------------------------------------------------------------------
def detect_model_type_from_checkpoint(local_ckpt_path: Path) -> str:
    """Read the model_type from the checkpoint's config.json."""
    for cfg_name in ("config.json", "pretrained_config.json"):
        cfg_file = local_ckpt_path / cfg_name
        if cfg_file.exists():
            with open(cfg_file) as f:
                cfg = json.load(f)
            # LeRobot configs store the registered name in `model_type`
            model_type = cfg.get("model_type")
            if model_type:
                return model_type
    raise ValueError(
        f"Could not detect model_type from checkpoint at {local_ckpt_path}. "
        "Ensure the checkpoint has a config.json with a 'model_type' field."
    )


# ---------------------------------------------------------------------------
# Build one per-dataset adapter + (optionally split) episode boundaries.
# ---------------------------------------------------------------------------
def _build_dataset_adapter(
    dataset_id, canon_cams_for_run, canonical_state_dim, canonical_image_size,
    obs, horizon, camera_map_spec, train_ratio,
):
    """Returns (adapter, ep_boundaries). The adapter z-scores state/action by this
    dataset's OWN native stats (per-dataset normalization); ep_boundaries are
    (start, end) frame ranges in the dataset's own index space, filtered to the
    train split when train_ratio < 1.0."""
    meta = LeRobotDatasetMetadata(dataset_id, force_cache_sync=True, revision="main")
    features = dataset_to_policy_features(meta.features)

    state_dims, action_dims, state_keys, action_keys = discover_state_action_dims(
        {dataset_id: meta}
    )
    native_state_dim = state_dims[dataset_id]
    native_action_dim = action_dims[dataset_id]
    state_key = state_keys[dataset_id]
    action_key = action_keys[dataset_id]
    native_camera_keys = sorted(
        k for k, ft in features.items() if ft.type is FeatureType.VISUAL
    )
    print(f"\n[dataset] {dataset_id}")
    print(f"  state '{state_key}'(dim={native_state_dim}) → canonical {canonical_state_dim}")
    print(f"  action '{action_key}'(dim={native_action_dim}) → canonical {CANONICAL_ACTION_DIM}")
    print(f"  cameras ({len(native_camera_keys)}): {native_camera_keys}")

    cam_mapping = _resolve_camera_mapping_for_dataset(
        camera_map_spec, native_camera_keys, canon_cams_for_run, dataset_id,
    )
    for canon, src in cam_mapping.items():
        print(f"    {canon:35s} ← {src if src else '<zero-padded>'}")

    fps = meta.fps if getattr(meta, "fps", None) else 10
    frame_time = 1 / fps
    obs_temporal_window = [-i * frame_time for i in range(obs)][::-1]
    action_temporal_window = [i * frame_time for i in range(horizon)]
    delta_timestamps = {
        state_key: obs_temporal_window,
        action_key: action_temporal_window,
        **{key: [0.0] for key in native_camera_keys},
    }
    base_dataset = LeRobotDataset(
        dataset_id, delta_timestamps=delta_timestamps,
        force_cache_sync=True, revision="main", tolerance_s=0.04,
    )
    print(f"  loaded: {len(base_dataset)} frames @ {fps} fps")
    task_idx_to_desc = load_task_descriptions(base_dataset)

    # The adapter returns RAW (un-normalized) canonical-projected values; the
    # global preprocessor does the normalization, with REAL pooled stats computed
    # below. This keeps the saved preprocessor/postprocessor non-identity so the
    # finetuned checkpoint is deployable (eval un-normalizes actions correctly).
    adapter = DatasetAdapter(
        dataset=base_dataset,
        sub_dir=dataset_id,
        camera_map=cam_mapping,
        state_key=state_key,
        action_key=action_key,
        state_dim=native_state_dim,
        action_dim=native_action_dim,
        task_idx_to_desc=task_idx_to_desc,
        canonical_state_dim=canonical_state_dim,
        canonical_image_size=canonical_image_size,
        normalize_in_adapter=False,
    )

    ep_bounds = get_sub_dataset_ep_boundaries(base_dataset)
    if train_ratio < 1.0:
        train_eps = get_per_task_train_episodes(base_dataset.hf_dataset, train_ratio)
        ep_ids = np.array(base_dataset.hf_dataset["episode_index"])
        ep_bounds = [(s, e) for (s, e) in ep_bounds if int(ep_ids[s]) in train_eps]
        print(f"  train split: kept {len(ep_bounds)} episodes")

    return adapter, ep_bounds


def _compute_pooled_stats(adapters, camera_keys, state_dim, action_dim, max_samples=5000):
    """Pool state/action stats across all adapters by sampling their raw,
    canonical-projected output. Returns a stats dict (mean/std/min/max per key)
    of REAL stats for the preprocessor, so the finetuned checkpoint is deployable.
    For one dataset this is just that dataset's stats; for several it is the pooled
    normalization (correct when they share an action space, e.g. LIBERO demos +
    RFT rollouts)."""
    print("\nComputing pooled dataset statistics (real stats for the preprocessor)...")
    all_states, all_actions = [], []
    total_frames = sum(len(a) for a in adapters)
    sample_ratio = min(1.0, max_samples / max(total_frames, 1))
    for adapter in adapters:
        n = max(1, int(len(adapter) * sample_ratio))
        idxs = np.random.choice(len(adapter), n, replace=False)
        for idx in idxs:
            item = adapter[int(idx)]
            if "observation.state" in item:
                s = item["observation.state"]
                s = s.numpy() if isinstance(s, torch.Tensor) else s
                all_states.append(np.asarray(s).reshape(-1)[:state_dim])
            if "action" in item:
                a = item["action"]
                a = a.numpy() if isinstance(a, torch.Tensor) else a
                all_actions.append(np.asarray(a).reshape(-1)[:action_dim])
    if not all_states:
        all_states = [np.zeros(state_dim)]
    if not all_actions:
        all_actions = [np.zeros(action_dim)]
    all_states = np.stack(all_states).astype(np.float32)
    all_actions = np.stack(all_actions).astype(np.float32)
    stats = {
        "observation.state": {
            "mean": torch.from_numpy(all_states.mean(0)),
            "std": torch.from_numpy(all_states.std(0).clip(min=1e-6)),
            "min": torch.from_numpy(all_states.min(0)),
            "max": torch.from_numpy(all_states.max(0)),
        },
        "action": {
            "mean": torch.from_numpy(all_actions.mean(0)),
            "std": torch.from_numpy(all_actions.std(0).clip(min=1e-6)),
            "min": torch.from_numpy(all_actions.min(0)),
            "max": torch.from_numpy(all_actions.max(0)),
        },
    }
    for cam in camera_keys:
        stats[cam] = {"mean": torch.tensor([0.0]), "std": torch.tensor([1.0]),
                      "min": torch.tensor([-1.0]), "max": torch.tensor([1.0])}
    print(f"  pooled {len(all_states)} state / {len(all_actions)} action frames")
    print(f"  state  mean: {stats['observation.state']['mean'].numpy()}")
    print(f"  action mean: {stats['action']['mean'].numpy()}")
    print(f"  action std : {stats['action']['std'].numpy()}")
    return stats


# ---------------------------------------------------------------------------
# Main fine-tuning routine.
# ---------------------------------------------------------------------------
def train(
    output_dir,
    dataset_id,
    resume_from_checkpoint=None,
    train_ratio=1.0,
    batch_size=64,
    learning_rate=1e-5,
    warmup_steps=200,
    training_steps=30000,
    reset_lang_params=False,
    camera_map=None,
    state_dim=None,
    robot_encoder_tokens=49,
    gripper_encoder_tokens=100,
    noise_temporal_correlation=0.0,
):
    # Accept a single id (str) or several (list) — normalize to a list.
    dataset_ids = [dataset_id] if isinstance(dataset_id, str) else list(dataset_id)
    if not dataset_ids:
        raise ValueError("At least one --dataset_id is required.")

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    # ── Resolve model type from checkpoint ───────────────────────────────────
    if resume_from_checkpoint is None:
        raise ValueError(
            "--resume_from_checkpoint is required for fine-tuning. "
            "This script is designed to fine-tune a pretrained checkpoint "
            "produced by train_community.py."
        )

    ckpt_path = Path(resume_from_checkpoint)
    local_ckpt_path = ckpt_path if ckpt_path.exists() else Path(
        huggingface_hub.snapshot_download(resume_from_checkpoint)
    )

    model_type = detect_model_type_from_checkpoint(local_ckpt_path)
    ConfigCls, PolicyCls, processor_fn, model_defaults = get_model_components(model_type)
    canonical_image_size = model_defaults["vision_input_size"]

    print(f"\n{'='*60}")
    print(f"Fine-tuning configuration")
    print(f"{'='*60}")
    print(f"  Model type:        {model_type}")
    print(f"  Config class:      {ConfigCls.__name__}")
    print(f"  Policy class:      {PolicyCls.__name__}")
    print(f"  d_model:           {model_defaults['d_model']}")
    print(f"  vision_input_size: {canonical_image_size}")
    print(f"  Checkpoint:        {local_ckpt_path}")
    print(f"  Target dataset(s): {dataset_ids}")
    print(f"{'='*60}\n")

    # Canonical state dim for THIS run. Defaults to the pretraining canonical
    # (8). Override (e.g. --state_dim 7) to change it.
    canonical_state_dim = state_dim if state_dim is not None else CANONICAL_STATE_DIM
    state_dim_overridden = state_dim is not None and canonical_state_dim != CANONICAL_STATE_DIM
    if state_dim_overridden:
        print(f"[state_dim override] canonical state_dim {CANONICAL_STATE_DIM} → "
              f"{canonical_state_dim}: state_encoder[0] will re-init and stats "
              f"will be recomputed at dim {canonical_state_dim}.")

    progress_update_freq = 200
    checkpoint_freq = 1000
    image_transforms = get_augmentations()

    # ── Read pretrained config ONCE (camera set, robot tokens, gripper cam) ──
    canon_cams_for_run = list(CANONICAL_CAMERAS)
    pre_gripper_camera: str | None = None
    pre_cfg: dict = {}
    for cfg_name in ("config.json", "pretrained_config.json"):
        cfg_file = local_ckpt_path / cfg_name
        if cfg_file.exists():
            with open(cfg_file) as f:
                pre_cfg = json.load(f)
            pre_cams = pre_cfg.get("cameras_for_vision_state_concat")
            if pre_cams:
                canon_cams_for_run = list(pre_cams)
                print(f"Using pretrained camera set ({len(canon_cams_for_run)}): "
                      f"{canon_cams_for_run}")
            # Inherit the robot-token structure from the checkpoint so the
            # fine-tune matches what the backbone was pretrained with.
            if pre_cfg.get("robot_encoder_tokens") is not None:
                if pre_cfg["robot_encoder_tokens"] != robot_encoder_tokens:
                    print(f"[inherit] robot_encoder_tokens {robot_encoder_tokens} "
                          f"→ {pre_cfg['robot_encoder_tokens']} (from checkpoint)")
                robot_encoder_tokens = pre_cfg["robot_encoder_tokens"]
            if pre_cfg.get("gripper_encoder_tokens") is not None:
                if pre_cfg["gripper_encoder_tokens"] != gripper_encoder_tokens:
                    print(f"[inherit] gripper_encoder_tokens {gripper_encoder_tokens} "
                          f"→ {pre_cfg['gripper_encoder_tokens']} (from checkpoint)")
                gripper_encoder_tokens = pre_cfg["gripper_encoder_tokens"]
            pre_gripper_camera = pre_cfg.get("gripper_camera")
            print(f"Pretrained checkpoint was at step {pre_cfg.get('training_step', 0)}; "
                  f"fine-tune starts at step 0")
            break

    # ── Build cfg with canonical dims (matching pretraining checkpoint) ──────
    obs = 2
    horizon = 64
    n_action_steps = 64

    from lerobot.configs.types import PolicyFeature
    canon_input_features = {
        "observation.state": PolicyFeature(
            type=FeatureType.STATE,
            shape=(canonical_state_dim,),
        ),
        **{
            cam: PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, canonical_image_size, canonical_image_size),
            )
            for cam in canon_cams_for_run
        },
    }
    canon_output_features = {
        "action": PolicyFeature(
            type=FeatureType.ACTION,
            shape=(CANONICAL_ACTION_DIM,),
        ),
    }

    cfg_kwargs = dict(
        input_features=canon_input_features,
        output_features=canon_output_features,
        n_obs_steps=obs,
        horizon=horizon,
        n_action_steps=n_action_steps,
        state_dim=canonical_state_dim,
        action_dim=CANONICAL_ACTION_DIM,
        num_vlm_layers=16,
        num_cameras=len(canon_cams_for_run),
        cameras_for_vision_state_concat=list(canon_cams_for_run),
        action_dim_weights=[1.0] * CANONICAL_ACTION_DIM,
        pos_decay_lambda=0.0,
        vision_lora_num_layers=0,
        num_latent_tokens=8,
        noise_temporal_correlation=noise_temporal_correlation,
        optimizer_lr=learning_rate,
        scheduler_warmup_steps=warmup_steps,
        robot_encoder_tokens=robot_encoder_tokens,
    )

    if model_type == "interleaved":
        cfg_kwargs["vlm_attends_to_expert"] = True
        cfg_kwargs["gripper_encoder_tokens"] = gripper_encoder_tokens
    elif model_type == "wilro":
        cfg_kwargs["gripper_encoder_tokens"] = gripper_encoder_tokens
        cfg_kwargs["use_robot_cnn"] = True
        kv_strat = pre_cfg.get("kv_capture_strategy")
        if kv_strat:
            cfg_kwargs["kv_capture_strategy"] = kv_strat
            print(f"KV capture strategy (from checkpoint): {kv_strat}")
        kv_layers = pre_cfg.get("kv_capture_layers")
        if kv_layers:
            cfg_kwargs["kv_capture_layers"] = kv_layers
    elif model_type == "wiltechs_vla":
        cfg_kwargs["use_robot_cnn"] = True

    if model_type in ("interleaved", "wilro"):
        cfg_kwargs["gripper_camera"] = pre_gripper_camera or "observation.images.wrist"

    cfg = ConfigCls(**cfg_kwargs)
    print(f"Robot CNN tokens: {robot_encoder_tokens} per cam "
          f"({int(robot_encoder_tokens ** 0.5)}x{int(robot_encoder_tokens ** 0.5)} grid)")
    if model_type in ("interleaved", "wilro") and hasattr(cfg, "gripper_camera"):
        print(f"Gripper cam '{cfg.gripper_camera}': {gripper_encoder_tokens} "
              f"({int(gripper_encoder_tokens ** 0.5)}x{int(gripper_encoder_tokens ** 0.5)} grid)")
        # Cameras here are the CANONICAL names; gripper_camera should be the
        # canonical wrist slot. Warn if it matches none of the canonical set —
        # then the dense grid is inert and every camera gets robot_encoder_tokens.
        if cfg.gripper_camera in canon_cams_for_run:
            print(f"  gripper grid ACTIVE on '{cfg.gripper_camera}' "
                  f"→ {gripper_encoder_tokens} tokens; other cams → {robot_encoder_tokens}.")
        else:
            print(f"  WARNING: gripper_camera '{cfg.gripper_camera}' matches NONE of "
                  f"{canon_cams_for_run} → gripper_encoder_tokens is INERT; every camera "
                  f"gets robot_encoder_tokens={robot_encoder_tokens}. Ensure a camera maps "
                  f"to the canonical wrist slot (e.g. --camera_map 'image2:wrist').")

    # ── Build per-dataset adapters + stitch ──────────────────────────────────
    adapters, all_boundaries = [], []
    for did in dataset_ids:
        adapter, ep_bounds = _build_dataset_adapter(
            did, canon_cams_for_run, canonical_state_dim, canonical_image_size,
            obs, horizon, camera_map, train_ratio,
        )
        adapters.append(adapter)
        all_boundaries.append(ep_bounds)

    stitched = StitchedDataset(adapters, all_boundaries)
    ep_from, ep_to = stitched.get_episode_boundaries()

    # ── Build policy: resume from pretrained checkpoint ──────────────────────
    print(f"\nFine-tuning from pretrained checkpoint: {resume_from_checkpoint}")
    policy = PolicyCls(cfg)

    model_file = local_ckpt_path / "model.safetensors"
    if not model_file.exists():
        candidates = list(local_ckpt_path.glob("*.safetensors"))
        if not candidates:
            raise FileNotFoundError(f"No .safetensors file found in {local_ckpt_path}")
        model_file = candidates[0]

    step, epoch = 0, 0
    ckpt_state = load_safetensors(model_file, device=str(device))

    policy.train()
    policy.to(device)
    cur_state = policy.state_dict()
    filtered = {k: v for k, v in ckpt_state.items()
                if k in cur_state and cur_state[k].shape == v.shape}
    skipped = [k for k in ckpt_state if k not in filtered]
    missing = [k for k in cur_state if k not in ckpt_state]
    if skipped:
        print(f"Skipped {len(skipped)} keys (shape mismatch / removed): {skipped[:5]}")
    if missing:
        print(f"Missing {len(missing)} keys (will use init values): {missing[:5]}")
    policy.load_state_dict(filtered, strict=False)
    print(f"Loaded {len(filtered)}/{len(cur_state)} model keys from pretrained")

    if reset_lang_params:
        with torch.no_grad():
            if hasattr(policy.model, "lang_attn_bias"):
                policy.model.lang_attn_bias.zero_()
                print("Reset lang_attn_bias → 0")
            if hasattr(policy.model, "lang_adaptor"):
                policy.model.lang_adaptor[1].weight.fill_(1.0)
                print("Reset lang_adaptor RMSNorm gamma → 1")
    else:
        if hasattr(policy.model, "lang_attn_bias"):
            sp = torch.nn.functional.softplus(policy.model.lang_attn_bias.detach()).cpu()
            print(f"lang_attn_bias on resume — per-layer softplus: "
                  f"min={sp.min().item():.3f} max={sp.max().item():.3f} "
                  f"mean={sp.mean().item():.3f}")

    # ── Stats: REAL pooled stats → deployable preprocessor ──────────────────
    # Pool mean/std/min/max across all datasets (sampling the raw, canonical-
    # projected adapter output) and bake those REAL stats into the
    # preprocessor/postprocessor. This is what makes the finetuned checkpoint
    # deployable: at eval the preprocessor normalizes the state and the
    # postprocessor un-normalizes the action with these stats. (For same-action-
    # space datasets — e.g. LIBERO demos + RFT rollouts — pooled normalization is
    # the correct single normalization.)
    stats = _compute_pooled_stats(
        adapters, canon_cams_for_run, canonical_state_dim, CANONICAL_ACTION_DIM,
    )
    preprocessor, postprocessor = processor_fn(policy.config, dataset_stats=stats)

    trainable_params = [p for p in policy.model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate, weight_decay=1e-6)

    # ── Optimizer state: load only if zero model-key mismatches ──────────
    optimizer_state_path = local_ckpt_path / "optimizer_state.pth"
    if optimizer_state_path.exists():
        if skipped:
            print(f"Skipping optimizer state — {len(skipped)} model key(s) had "
                  f"shape mismatch, Adam momentum would be stale. Starting Adam from zero.")
        else:
            try:
                saved = torch.load(optimizer_state_path, map_location=device)
                optimizer.load_state_dict(saved)
                for pg in optimizer.param_groups:
                    pg['lr'] = learning_rate
                    pg['initial_lr'] = learning_rate
                print(f"Optimizer state loaded; LR reset to fine-tune value "
                      f"{learning_rate:.2e}")
            except (ValueError, RuntimeError) as e:
                print(f"Skipping optimizer state — mismatch ({e})")
    else:
        print("No optimizer_state.pth in checkpoint; starting Adam from zero.")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps
    )

    if isinstance(preprocessor, torch.nn.Module):
        preprocessor.to(device)

    sampler = EpisodeAwareSampler(
        dataset_from_indices=ep_from,
        dataset_to_indices=ep_to,
        drop_n_first_frames=0,
        drop_n_last_frames=0,
        shuffle=True,
    )

    print(f"\nBatch size: {batch_size}  ({model_type} model, d_model={model_defaults['d_model']}; "
          f"drop to 16 if you OOM)")
    dataloader = torch.utils.data.DataLoader(
        stitched,
        num_workers=8,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # ── Training loop ───────────────────────────────────────────────────
    print(f"\nStarting fine-tune loop: {training_steps} steps, batch_size={batch_size}, "
          f"lr={learning_rate:.2e}, warmup={warmup_steps}")
    done = False
    prog_bar = tqdm(total=training_steps, desc="Fine-tune Progress", initial=step)
    canonical_state_key = "observation.state"
    while not done:
        epoch += 1
        for batch in dataloader:
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)

            if "task" in batch and isinstance(batch["task"], (list, tuple)):
                batch["task_description"] = batch["task"]

            batch = apply_image_augmentations(batch, canon_cams_for_run, image_transforms)
            batch = apply_joint_augmentations(batch, canonical_state_key)

            if step == 0:
                # Raw state (adapter returns un-normalized); the preprocessor below
                # normalizes it with the pooled stats. Sanity check on the range.
                raw_st = batch[canonical_state_key].float()
                print(f"\nRaw (pre-norm) {canonical_state_key}: "
                      f"min={raw_st.min():.4f} max={raw_st.max():.4f} std={raw_st.std():.4f}")

            batch = preprocessor(batch)

            if step == 0:
                pad_key = next((k for k in batch if "pad" in k.lower() and "action" in k.lower()), None)
                if pad_key is None:
                    print("WARNING: no action pad key in batch")
                else:
                    print(f"Action pad key='{pad_key}', pad fraction: "
                          f"{batch[pad_key].float().mean().item():.2%}")

            if step % progress_update_freq == 0:
                policy.model._capture_attention_stats = True

            autocast_ctx = (
                torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                if device.type == "cuda"
                else torch.autocast(device_type="cpu", enabled=False)
            )
            with autocast_ctx:
                loss, _ = policy.forward(batch)

            loss.backward()

            if step % progress_update_freq == 0:
                _log_gradient_analysis(policy, step, model_type)

            trainable_params = [p for p in policy.parameters() if p.requires_grad]
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if step % progress_update_freq == 0:
                lr = optimizer.param_groups[0]['lr']
                prog_bar.set_description(f"Epoch {epoch}, Step {step}")
                prog_bar.set_postfix({
                    "loss": f"{loss.item():.3f}",
                    "lr": f"{lr:.2e}",
                    "grad_norm": f"{grad_norm:.2f}",
                })

            if step > 0 and step % checkpoint_freq == 0:
                checkpoint_dir = output_directory / f"checkpoint-{step}"
                checkpoint_dir.mkdir(exist_ok=True)
                policy.config.training_step = step
                policy.config.training_epoch = epoch
                policy.config.optimizer_lr = optimizer.param_groups[0]["lr"]
                policy.config.current_lr = optimizer.param_groups[0]["lr"]
                policy.config.training_steps_total = training_steps
                policy.save_pretrained(checkpoint_dir)
                torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer_state.pth")
                preprocessor.save_pretrained(checkpoint_dir)
                postprocessor.save_pretrained(checkpoint_dir)
                print(f"\nCheckpoint saved at step {step}")

            step += 1
            if step % progress_update_freq == 0 or step >= training_steps:
                prog_bar.update(progress_update_freq)
                prog_bar.set_description(f"Epoch {epoch}, Step {step}")

            if step >= training_steps:
                done = True
                prog_bar.close()
                break
    prog_bar.close()

    policy.config.training_step = step
    policy.config.training_epoch = epoch
    policy.config.optimizer_lr = optimizer.param_groups[0]["lr"]
    policy.config.current_lr = optimizer.param_groups[0]["lr"]
    policy.config.training_steps_total = training_steps
    policy.save_pretrained(output_directory)
    torch.save(optimizer.state_dict(), output_directory / "optimizer_state.pth")
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)


def _log_gradient_analysis(policy, step, model_type):
    print(f"\n--- Gradient Analysis at Step {step} ---")
    for label, prefix in [
        ("Vision",         "vision_model"),
        ("Vision LoRA",    "lora_"),
        ("Connector",      "connector"),
        ("State Enc",      "state_encoder"),
        ("Robot CNN",      "robot_visual_encoder"),
        ("Expert Layers",  "expert_layers"),
        ("DiT Layers",     "dit_layers"),
        ("Action In/Out",  "action_"),
        ("Final Norm",     "final_norm"),
        ("Latent Gen",     "latent_generator"),
        ("Lang Adaptor",   "lang_adaptor"),
    ]:
        grad, n = _grad_stats(policy, prefix)
        if grad is not None:
            print(f"  {label:14s} - Avg Abs Grad: {grad:.6f} ({n} params)")
        else:
            print(f"  {label:14s} - no grad")

    if hasattr(policy.model, "latent_generator"):
        gen = policy.model.latent_generator
        w_norm_sq = sum(p.detach().norm().item() ** 2 for p in gen.parameters())
        g_norm_sq = sum(p.grad.norm().item() ** 2 for p in gen.parameters() if p.grad is not None)
        out_w_norm = gen[-1].weight.detach().norm().item()
        print(f"  Latent gen     - weight_norm: {w_norm_sq ** 0.5:.4e}   "
              f"grad_norm: {g_norm_sq ** 0.5:.4e}   out_layer_w: {out_w_norm:.4e}")

    if hasattr(policy.model, "lang_attn_bias"):
        bias = policy.model.lang_attn_bias.detach()
        sp = torch.nn.functional.softplus(bias).cpu()
        grad = policy.model.lang_attn_bias.grad
        gn = f"{grad.norm().item():.4e}" if grad is not None else "None"
        sp_str = "[" + " ".join(f"{v:.2f}" for v in sp.tolist()) + "]"
        print(f"  Lang attn bias - softplus per-layer: {sp_str}")
        print(f"                   min={sp.min().item():.3f} max={sp.max().item():.3f} "
              f"mean={sp.mean().item():.3f}  grad_norm: {gn}")

    if hasattr(policy.model, "lang_adaptor"):
        w_norm = sum(p.detach().norm().item() ** 2 for p in policy.model.lang_adaptor.parameters()) ** 0.5
        g_norm_sq = sum(p.grad.norm().item() ** 2 for p in policy.model.lang_adaptor.parameters() if p.grad is not None)
        print(f"  Lang adaptor   - weight_norm: {w_norm:.4e}   grad_norm: {g_norm_sq ** 0.5:.4e}")

    # Self-attn (interleaved: one joint softmax; wilro: DiT self-attn). "sink"
    # only exists for wilro; harmless for interleaved (key absent).
    stats = getattr(policy.model, "_last_attention_stats", None)
    xstats = getattr(policy.model, "_last_cross_attention_stats", None)
    if stats:
        order = ["sink", "vision", "language", "state", "robot", "latent", "action"]
        ordered = [(k, stats[k]) for k in order if k in stats]
        cells = "  ".join(f"{k}={v*100:5.1f}%" for k, v in ordered)
        label = "self-attn" if xstats else "attn     "
        print(f"  Action→ {label} : {cells}")
    if xstats:
        cells = "  ".join(f"{k}={xstats[k]*100:5.1f}%" for k in ("vision", "language") if k in xstats)
        print(f"  Action→ x-attn     : {cells}    (cross-attn to VLM KV)")

    comps = getattr(policy.model, "_last_loss_components", None)
    cw = getattr(policy.model.config, "contrastive_loss_weight", 0.0)
    if comps is not None and cw > 0.0:
        margin = getattr(policy.model.config, "contrastive_margin", 0.05)
        contr_v = comps.get("contrastive", float("nan"))
        pct = (contr_v / margin * 100.0) if margin > 0 else float("nan")
        print(f"  Contrastive    - main: {comps.get('main', float('nan')):.4f}   "
              f"contrastive: {contr_v:.4f} ({pct:.0f}% of margin {margin:.3f})   weight: {cw}")
    print("--- End Gradient Analysis ---\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Generic fine-tuning script for interleaved, wilro, and wiltechs_vla models.",
    )
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for fine-tuned checkpoints")
    parser.add_argument("--dataset_id", type=str, required=True, nargs="+",
                        help="One or more LeRobot v3 dataset ids (HF hub or local paths). "
                             "Pass several to mix (e.g. RFT-collected rollouts + expert demos); "
                             "they're concatenated and each is normalized by its own stats.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Pretrained checkpoint produced by train_community.py. "
                             "Model type is auto-detected from config.json.")
    parser.add_argument("--train_ratio", type=float, default=1.0,
                        help="Per-task train ratio, applied within EACH dataset. 1.0 = use "
                             "everything (default). 0.9 = LIBERO-convention 90/10 episode split.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size. interleaved/wilro: 24-64, wiltechs_vla: 8-24 "
                             "(4B VLM is memory-heavy).")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Fine-tune LR. Pretraining used 1e-4; 5-10× lower is the "
                             "usual fine-tune default.")
    parser.add_argument("--warmup_steps", type=int, default=200,
                        help="Warmup steps for cosine schedule.")
    parser.add_argument("--training_steps", type=int, default=30000,
                        help="Fine-tune budget. LIBERO/single-task datasets usually "
                             "converge in 30-50k steps.")
    parser.add_argument("--reset_lang_params", action="store_true",
                        help="Zero out lang_attn_bias and reset lang_adaptor RMSNorm "
                             "gamma to 1. DO NOT use for routine fine-tuning — erases "
                             "the pretrained language pathway.")
    parser.add_argument("--state_dim", type=int, default=None,
                        help="Override the canonical state dim for this run (default: "
                             f"{CANONICAL_STATE_DIM}, inherited from pretraining). When set, "
                             "state_encoder[0] re-inits and normalization is done at the new dim.")
    parser.add_argument("--camera_map", type=str, default=None,
                        help="Manual native→canonical camera mapping, comma-separated "
                             "pairs 'native_short:canonical_short'. Applied to every dataset "
                             "that has those native cameras (per-dataset automatic fallback "
                             "otherwise). Example: --camera_map 'image:front,image2:wrist'")
    parser.add_argument("--robot_encoder_tokens", type=int, default=49,
                        help="Robot CNN tokens per non-gripper camera. Perfect square "
                             "(grid side = sqrt). Default: 49 (7x7). Overridden by checkpoint.")
    parser.add_argument("--gripper_encoder_tokens", type=int, default=100,
                        help="Robot CNN tokens for the gripper/wrist camera. Perfect square. "
                             "Used by interleaved and wilro models only. Overridden by checkpoint.")
    parser.add_argument("--noise_temporal_correlation", type=float, default=0.0,
                        help="AR(1) coefficient correlating the flow-matching source noise "
                             "along the action horizon (0=white; ~0.9=temporally smooth). "
                             "Source dist changes — resume from a rho=0 checkpoint and "
                             "fine-tune to adapt; >0.95 over-smooths sharp/contact motions.")
    args = parser.parse_args()
    for _name in ("robot_encoder_tokens", "gripper_encoder_tokens"):
        _v = getattr(args, _name)
        if int(_v ** 0.5) ** 2 != _v:
            parser.error(f"--{_name} must be a perfect square, got {_v}")
    train(**vars(args))
