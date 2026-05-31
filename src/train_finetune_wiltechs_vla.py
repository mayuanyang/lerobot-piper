"""
General-purpose fine-tuning script for WiltechsVLAPolicy (Qwen3-VL backbone).

Loads a pretrained checkpoint produced by `train_wiltechs_vla.py`
(canonical schema: state_dim=7, action_dim=7, up to 5 canonical cameras at
384×384) and fine-tunes it on a single LeRobot v3 dataset.

The downstream dataset is projected into the same canonical schema via
`DatasetAdapter` (imported from `train_wiltechs_vla`), so every pretrained
weight transfers directly — no `state_encoder` / `action_in` / `action_out`
re-initialisation. State dims > 7 are truncated; state dims < 7 are zero-
padded. Cameras are mapped semantically (then positional fallback), missing
canonical cameras are zero-filled, and every camera is letterbox-padded +
resized to 384×384.

Parallel to `train_finetune_interleaved.py` (which fine-tunes the SmolVLM-
based InterleavedFlowMatchingPolicy). Same CLI surface — `--dataset_id`,
`--resume_from_checkpoint`, `--train_ratio`, `--camera_map`.

Key differences vs. pretraining (`train_wiltechs_vla.py`):
  - Lower default LR (1e-5) and shorter warmup (200 steps).
  - Optimizer-state load is guarded: any model-key shape mismatch (e.g.
    different `num_cameras`, `horizon`, or stale `state_dim`) abandons the
    saved Adam momentum and restarts fresh instead of crashing inside
    `_foreach_lerp_`.
  - Default batch_size = 16 (Qwen3-VL-4B is much heavier than SmolVLM-500M;
    drop further to 8 or 4 if you OOM, or enable
    `policy.model.language_model.gradient_checkpointing_enable()` manually).

NOTE: Checkpoints from the OLD interleaved WiltechsVLA (pre-encoder-decoder
rewrite) are NOT compatible — the DiT replaces the joint-attention
expert_layers entirely. Old keys will be reported as "skipped" on load.
"""
from pathlib import Path
import torch
from tqdm import tqdm
import huggingface_hub
from safetensors.torch import load_file as load_safetensors
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import dataset_to_policy_features
import numpy as np
from torch.utils.data import Subset

from models.wiltechs_vla.wiltechs_vla_config import WiltechsVLAConfig
from models.wiltechs_vla.wiltechs_vla_policy import WiltechsVLAPolicy
from models.wiltechs_vla.processor_wiltechs_vla import make_pre_post_processors

from torchvision.transforms import v2
from transformers import get_cosine_schedule_with_warmup

# Canonical schema + projection adapter are imported from the WiltechsVLA
# pretraining script — single source of truth. Any change there propagates
# here. (Mirrors how train_finetune_interleaved.py imports from
# train_community_interleaved.)
from train_wiltechs_vla import (
    CANONICAL_CAMERAS,
    CANONICAL_STATE_DIM,
    CANONICAL_ACTION_DIM,
    CANONICAL_IMAGE_SIZE,
    DatasetAdapter,
    build_camera_mapping,
    compute_unified_stats,
    discover_state_action_dims,
    load_task_descriptions,
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
# Augmentations — same recipe as train_finetune_interleaved.py.
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
# Optional per-task train/test split — generic over any LeRobot dataset.
# Single-task datasets degenerate to a 90/10 episode split.
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
# Manual camera-map parser. Lets the user pin a native camera key to a
# specific canonical slot by SEMANTIC CONTENT instead of relying on the
# auto positional fallback in build_camera_mapping().
#
# Same semantics + spec form as train_finetune_interleaved._resolve_camera_mapping.
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

    # Positional fallback for whatever the user didn't pin.
    assigned_natives = {v for v in out.values() if v is not None}
    remaining_natives = sorted(n for n in native_keys if n not in assigned_natives)
    for canon in canon_keys:
        if out[canon] is None and remaining_natives:
            out[canon] = remaining_natives.pop(0)
    return out


# ---------------------------------------------------------------------------
# Main fine-tuning routine.
# ---------------------------------------------------------------------------
def train(
    output_dir,
    dataset_id,
    resume_from_checkpoint=None,
    train_ratio=1.0,
    batch_size=16,
    learning_rate=1e-5,
    warmup_steps=200,
    training_steps=30000,
    reset_lang_params=False,
    camera_map=None,
):
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    progress_update_freq = 200
    checkpoint_freq = 1000
    image_transforms = get_augmentations()

    # ── Dataset metadata + canonical projection setup ────────────────────────
    dataset_metadata = LeRobotDatasetMetadata(dataset_id, force_cache_sync=True, revision="main")
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}

    if not output_features:
        raise ValueError("No output features (actions) found.")

    print('input_features:', list(input_features.keys()))
    print('output_features:', list(output_features.keys()))

    # Native dims/keys from the downstream dataset (DatasetAdapter then maps
    # to canonical). discover_state_action_dims expects a dict keyed by
    # sub_dir; pass a single entry so we can reuse the helper unchanged.
    state_dims, action_dims, state_keys, action_keys = discover_state_action_dims(
        {dataset_id: dataset_metadata}
    )
    native_state_dim = state_dims[dataset_id]
    native_action_dim = action_dims[dataset_id]
    state_key = state_keys[dataset_id]
    action_key = action_keys[dataset_id]
    print(f"Native state_key='{state_key}' (dim={native_state_dim})  →  canonical {CANONICAL_STATE_DIM}")
    print(f"Native action_key='{action_key}' (dim={native_action_dim})  →  canonical {CANONICAL_ACTION_DIM}")

    # Native camera keys from this dataset.
    native_camera_keys = sorted(
        k for k, ft in input_features.items() if ft.type is FeatureType.VISUAL
    )
    print(f"Native cameras ({len(native_camera_keys)}): {native_camera_keys}")

    # Canonical camera subset to use. Pretraining only registers cameras that
    # at least one sub-dataset actually had (see `used_canonical` in
    # train_wiltechs_vla.py), so num_cameras and the camera list are NOT
    # guaranteed to be the full CANONICAL_CAMERAS list. If we resume from a
    # pretrained checkpoint, we MUST match its camera set exactly or the
    # robot_visual_encoder shapes diverge. Read it from saved config.json.
    canon_cams_for_run = list(CANONICAL_CAMERAS)
    if resume_from_checkpoint is not None:
        import json as _json
        ckpt_root = Path(resume_from_checkpoint)
        if not ckpt_root.exists():
            ckpt_root = Path(huggingface_hub.snapshot_download(resume_from_checkpoint))
        for cfg_name in ("config.json", "pretrained_config.json"):
            cfg_file = ckpt_root / cfg_name
            if cfg_file.exists():
                with open(cfg_file) as f:
                    pre_cfg = _json.load(f)
                pre_cams = pre_cfg.get("cameras_for_vision_state_concat")
                if pre_cams:
                    canon_cams_for_run = list(pre_cams)
                    print(f"Using pretrained camera set ({len(canon_cams_for_run)}): "
                          f"{canon_cams_for_run}")
                break

    if camera_map:
        cam_mapping = _resolve_camera_mapping(
            camera_map, native_camera_keys, canon_cams_for_run
        )
        print(f"Canonical camera mapping (user --camera_map='{camera_map}', "
              f"unspecified slots auto-filled):")
    else:
        cam_mapping = build_camera_mapping(
            {dataset_id: set(native_camera_keys)}, canon_cams_for_run
        )[dataset_id]
        print("Canonical camera mapping (automatic: semantic match + positional fallback):")
    for canon, src in cam_mapping.items():
        print(f"  {canon:35s} ← {src if src else '<zero-padded>'}")

    # ── Build cfg with canonical dims (matching pretraining checkpoint) ──────
    obs = 2
    horizon = 64
    n_action_steps = 64

    # Build input_features for cfg as if we were the canonical dataset.
    # processor_wiltechs_vla consults shapes from cfg.input_features, so they
    # need to reflect the post-adapter (canonical) shapes.
    from lerobot.configs.types import PolicyFeature
    canon_input_features = {
        "observation.state": PolicyFeature(
            type=FeatureType.STATE,
            shape=(CANONICAL_STATE_DIM,),
        ),
        **{
            cam: PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, CANONICAL_IMAGE_SIZE, CANONICAL_IMAGE_SIZE),
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

    cfg = WiltechsVLAConfig(
        input_features=canon_input_features,
        output_features=canon_output_features,
        n_obs_steps=obs,
        horizon=horizon,
        n_action_steps=n_action_steps,
        state_dim=CANONICAL_STATE_DIM,
        action_dim=CANONICAL_ACTION_DIM,
        num_vlm_layers=16,
        num_cameras=len(canon_cams_for_run),
        cameras_for_vision_state_concat=list(canon_cams_for_run),
        action_dim_weights=[1.0] * CANONICAL_ACTION_DIM,
        pos_decay_lambda=0.0,
        vision_lora_num_layers=0,
        num_latent_tokens=8,
        # Fine-tuning LR/warmup defaults; can be overridden via CLI.
        optimizer_lr=learning_rate,
        scheduler_warmup_steps=warmup_steps,
    )

    # ── Build the dataset adapter early so we have something to compute
    # canonical stats against if the checkpoint doesn't carry a saved
    # preprocessor. `LeRobotDataset` construction also needs delta_timestamps
    # in native key space.
    fps = dataset_metadata.fps if hasattr(dataset_metadata, "fps") and dataset_metadata.fps else 10
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
    print(f"Dataset loaded: {len(base_dataset)} total frames")
    task_idx_to_desc = load_task_descriptions(base_dataset)
    print(f"Loaded {len(task_idx_to_desc)} task descriptions")
    canonical_dataset = DatasetAdapter(
        dataset=base_dataset,
        sub_dir=dataset_id,
        camera_map=cam_mapping,
        state_key=state_key,
        action_key=action_key,
        state_dim=native_state_dim,
        action_dim=native_action_dim,
        task_idx_to_desc=task_idx_to_desc,
    )

    # ── Build policy: resume from pretrained checkpoint, else fresh ──────────
    if resume_from_checkpoint is not None:
        print(f"Fine-tuning from pretrained checkpoint: {resume_from_checkpoint}")
        policy = WiltechsVLAPolicy(cfg)

        ckpt_path = Path(resume_from_checkpoint)
        local_ckpt_path = ckpt_path if ckpt_path.exists() else Path(
            huggingface_hub.snapshot_download(resume_from_checkpoint)
        )

        model_file = local_ckpt_path / "model.safetensors"
        if not model_file.exists():
            candidates = list(local_ckpt_path.glob("*.safetensors"))
            if not candidates:
                raise FileNotFoundError(f"No .safetensors file found in {local_ckpt_path}")
            model_file = candidates[0]

        import json
        # Fine-tuning starts step counter at 0 — we don't fast-forward the
        # cosine schedule to the pretraining step. Treat this as a fresh run
        # that happens to start from warm weights.
        step, epoch = 0, 0
        saved_cfg_json = {}
        for config_name in ("config.json", "pretrained_config.json"):
            config_file = local_ckpt_path / config_name
            if config_file.exists():
                with open(config_file) as f:
                    saved_cfg_json = json.load(f)
                pretrain_step = saved_cfg_json.get("training_step", 0)
                print(f"Pretrained checkpoint was at step {pretrain_step} "
                      f"({config_name}); fine-tune starts at step 0")
                break

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

        # The encoder-decoder WiltechsVLA model has no `lang_attn_bias` /
        # `lang_adaptor` modules (those were interleaved-model artefacts).
        # Keep the flag for CLI compatibility; warn if the user sets it.
        if reset_lang_params:
            print("WARNING: --reset_lang_params has no effect on the encoder-decoder "
                  "WiltechsVLA (no lang_attn_bias / lang_adaptor modules).")

        # Stats: prefer the saved pretrained preprocessor (carries the
        # canonical mean/std the pretrained model was normalised against).
        # Fall back to recomputing on this dataset via the adapter — slower,
        # and introduces a normalization shift the pretrained input layers
        # weren't trained for, but works when the checkpoint pre-dates
        # preprocessor saving.
        preprocessor, postprocessor = None, None
        try:
            from lerobot.policies.factory import load_pre_post_processors
            preprocessor, postprocessor = load_pre_post_processors(str(local_ckpt_path))
            print("Loaded preprocessor + postprocessor from pretrained checkpoint "
                  "(canonical stats preserved)")
        except Exception as e:
            print(f"Could not load saved preprocessor ({e}); "
                  f"recomputing canonical stats on the fine-tune dataset")
            stats = compute_unified_stats(
                [canonical_dataset], canon_cams_for_run,
                CANONICAL_STATE_DIM, CANONICAL_ACTION_DIM,
            )
            preprocessor, postprocessor = make_pre_post_processors(
                policy.config, dataset_stats=stats
            )

        trainable_params = [p for p in policy.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=learning_rate, weight_decay=1e-6)

        # ── Optimizer state: load only if zero model-key mismatches ──────
        # Otherwise Adam's stored (exp_avg, exp_avg_sq) tensors no longer
        # align with the live params and `_foreach_lerp_` crashes on the
        # first step. For fine-tuning, losing pretraining momentum is fine —
        # we use a different LR anyway, so fresh Adam state is appropriate.
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
    else:
        # Train from scratch fallback. Useful for sanity check / ablation.
        # Stats come from the adapter (canonical-projected), not from
        # `dataset_metadata.stats` (native-keyed, dim-mismatched).
        print("No --resume_from_checkpoint given; training from scratch.")
        policy = WiltechsVLAPolicy(cfg)
        policy.train()
        policy.to(device)
        stats = compute_unified_stats(
            [canonical_dataset], canon_cams_for_run,
            CANONICAL_STATE_DIM, CANONICAL_ACTION_DIM,
        )
        preprocessor, postprocessor = make_pre_post_processors(
            cfg, dataset_stats=stats
        )
        step, epoch = 0, 0

        trainable_params = [p for p in policy.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in trainable_params)
        n_frozen = sum(p.numel() for p in policy.parameters() if not p.requires_grad)
        print(f"Total trainable parameters: {n_trainable:,}  (frozen: {n_frozen:,})")
        optimizer = torch.optim.Adam(trainable_params, lr=learning_rate, weight_decay=1e-6)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps
        )

    if isinstance(preprocessor, torch.nn.Module):
        preprocessor.to(device)
    print(f"Dataset FPS: {fps}")

    # ── Optional per-task train split ───────────────────────────────────
    all_episode_ids = np.array(base_dataset.hf_dataset["episode_index"])
    if train_ratio < 1.0:
        train_episodes = get_per_task_train_episodes(base_dataset.hf_dataset, train_ratio)
        valid_indices = [i for i, ep in enumerate(all_episode_ids) if int(ep) in train_episodes]
        train_subset = Subset(canonical_dataset, valid_indices)
        ep_ids_subset = all_episode_ids[np.array(valid_indices)]
    else:
        train_subset = canonical_dataset
        ep_ids_subset = all_episode_ids
        valid_indices = None
    print(f"Training subset: {len(train_subset)} frames "
          f"({'no split' if valid_indices is None else f'from {len(train_episodes)} episodes'})")

    # EpisodeAwareSampler boundaries against the (possibly subset) index space.
    if len(ep_ids_subset) > 0:
        ep_changes = np.where(np.diff(ep_ids_subset) != 0)[0] + 1
        ep_from = np.concatenate([[0], ep_changes]).tolist()
        ep_to = np.concatenate([ep_changes, [len(ep_ids_subset)]]).tolist()
    else:
        ep_from, ep_to = [0], [0]

    sampler = EpisodeAwareSampler(
        dataset_from_indices=ep_from,
        dataset_to_indices=ep_to,
        drop_n_first_frames=0,
        drop_n_last_frames=0,
        shuffle=True,
    )

    print(f"Batch size: {batch_size}  (Qwen3-VL-4B encoder + 16-layer DiT is "
          f"memory-heavy; drop to 8 or 4 if you OOM, or enable "
          f"`policy.model.language_model.gradient_checkpointing_enable()`)")
    dataloader = torch.utils.data.DataLoader(
        train_subset,
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

            # DatasetAdapter already sets `task_description`; if a downstream
            # dataset also provides `task` as a list, mirror it so both
            # branches in the model's contrastive code see something.
            if "task" in batch and isinstance(batch["task"], (list, tuple)):
                batch["task_description"] = batch["task"]

            batch = apply_image_augmentations(batch, canon_cams_for_run, image_transforms)
            batch = apply_joint_augmentations(batch, canonical_state_key)

            if step == 0:
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

            autocast_ctx = (
                torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                if device.type == "cuda"
                else torch.autocast(device_type="cpu", enabled=False)
            )
            with autocast_ctx:
                loss, _ = policy.forward(batch)

            loss.backward()

            if step % progress_update_freq == 0:
                _log_gradient_analysis(policy, step)

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


def _log_gradient_analysis(policy, step):
    """
    Gradient analysis tailored to WiltechsVLA's encoder-decoder structure.

    Prefix → module mapping:
      'visual'          → Qwen3-VL vision tower (frozen → no grad)
      'language_model'  → Qwen3-VL text layers (frozen → no grad)
      'lora_'           → vision LoRA adapters (if vision_lora_num_layers > 0)
      'state_encoder'   → state → token projection
      'robot_visual_encoder' → parallel ResNet-18 CNN tokens
      'dit_layers'      → trainable DiT (self-attn + cross-attn to VLM KV)
      'action_'         → action_in_proj / action_out_proj / action_pos_emb
      'time_embedder'   → flow-matching time MLP
      'latent_generator'→ task-conditional latent tokens
      'sink_token'      → attention SINK parameter
    """
    print(f"\n--- Gradient Analysis at Step {step} ---")
    for label, prefix in [
        ("Vision (VLM)",   "visual"),
        ("Language (VLM)", "language_model"),
        ("Vision LoRA",    "lora_"),
        ("State Enc",      "state_encoder"),
        ("Robot CNN",      "robot_visual_encoder"),
        ("DiT Layers",     "dit_layers"),
        ("Action In/Out",  "action_"),
        ("Time Embed",     "time_embedder"),
        ("Latent Gen",     "latent_generator"),
        ("SINK Token",     "sink_token"),
    ]:
        grad, n = _grad_stats(policy, prefix)
        if grad is not None:
            print(f"  {label:16s} - Avg Abs Grad: {grad:.6f} ({n} params)")
        else:
            print(f"  {label:16s} - no grad")

    if hasattr(policy.model, "latent_generator") and policy.model.latent_generator is not None:
        gen = policy.model.latent_generator
        w_norm_sq = sum(p.detach().norm().item() ** 2 for p in gen.parameters())
        g_norm_sq = sum(p.grad.norm().item() ** 2 for p in gen.parameters() if p.grad is not None)
        out_w_norm = gen[-1].weight.detach().norm().item()
        print(f"  Latent gen     - weight_norm: {w_norm_sq ** 0.5:.4e}   "
              f"grad_norm: {g_norm_sq ** 0.5:.4e}   out_layer_w: {out_w_norm:.4e}")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_id", type=str, required=True,
                        help="Any LeRobot v3 dataset id (HF hub or local path).")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Pretrained checkpoint produced by train_wiltechs_vla.py")
    parser.add_argument("--train_ratio", type=float, default=1.0,
                        help="Per-task train ratio. 1.0 = use everything (default for "
                             "fine-tuning). 0.9 = LIBERO-convention 90/10 episode split.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Default 16; Qwen3-VL-4B is much heavier than SmolVLM-500M. "
                             "Drop to 8 or 4 if you OOM.")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Fine-tune LR. Pretraining used 1e-4; 5-10× lower is the "
                             "usual fine-tune default.")
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--training_steps", type=int, default=30000,
                        help="Fine-tune budget. LIBERO/single-task datasets usually "
                             "converge in 30-50k steps.")
    parser.add_argument("--reset_lang_params", action="store_true",
                        help="NO-OP on encoder-decoder WiltechsVLA — kept for CLI parity "
                             "with train_finetune_interleaved.py. Will print a warning "
                             "if set.")
    parser.add_argument("--camera_map", type=str, default=None,
                        help="Manual native→canonical camera mapping, comma-separated "
                             "pairs of the form 'native_short:canonical_short'. "
                             "Pin a camera to a slot whose pretraining content prior "
                             "matches its semantic content (slot 0 'front' = scene view, "
                             "slot 1 'gripper' = wrist-mount, slot 2 'right' = side, ...). "
                             "Cameras not mentioned are auto-assigned to remaining slots "
                             "via positional fallback. Example: "
                             "--camera_map 'image:gripper,image2:front,image3:right'")
    args = parser.parse_args()
    train(**vars(args))
