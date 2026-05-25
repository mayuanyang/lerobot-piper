"""
Training script for the SmolVLA-style interleaved flow matching model on LIBERO.

This is a parallel of `train_libero.py` adapted for `InterleavedFlowMatchingPolicy`.
Notable differences from the encoder-decoder script:
  - Imports the new policy / config / processor under `models.interleaved_flow_matching`.
  - Default `batch_size=64` (was 256). The joint-attention model runs at d_model=960
    and processes the full [vlm + latent + action] sequence every layer, so per-batch
    memory is several-fold higher. Bump back up if your GPU has headroom.
  - Drops the legacy `context_proj` checkpoint migration (the model has no
    `context_proj` field) and `robot_layer_projs` references (replaced by
    parallel `expert_layers`).
  - Gradient analysis renames `action_expert` / `robot_layer_projs` →
    `expert_layers`; everything else is the same.
"""
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
import huggingface_hub
from safetensors.torch import load_file as load_safetensors
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import dataset_to_policy_features
import numpy as np
from torch.utils.data import Subset

from models.interleaved_flow_matching.interleaved_flow_matching_config import InterleavedFlowMatchingConfig
from models.interleaved_flow_matching.interleaved_flow_matching_policy import InterleavedFlowMatchingPolicy
from models.interleaved_flow_matching.processor_interleaved_flow_matching import make_pre_post_processors

from torchvision.transforms import v2
from transformers import get_cosine_schedule_with_warmup


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


def get_augmentations():
    # Same recipe as train_libero.py: small spatial + colour + occasional blur.
    spatial = v2.RandomAffine(
        degrees=0,
        translate=(0.03, 0.03),
        scale=(0.95, 1.05),
        fill=0,
    )
    color = v2.ColorJitter(
        brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08,
    )
    blur = v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))], p=0.3)
    return v2.Compose([spatial, color, blur])


def apply_joint_augmentations(batch, state_key):
    if torch.rand(1).item() > 0.5:
        if state_key in batch:
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


def _print_lora_status(policy, cfg) -> None:
    lora_params = sum(
        p.numel()
        for n, p in policy.named_parameters()
        if p.requires_grad and "lora_" in n
    )
    print("=" * 60)
    print(
        f"[LoRA] ENABLED  rank={cfg.lora_rank}  alpha={cfg.lora_alpha}  "
        f"targets={cfg.lora_target_modules}  vision_layers={cfg.vision_lora_num_layers}"
    )
    print(f"[LoRA] Trainable LoRA params: {lora_params:,}")
    if lora_params == 0:
        print("[LoRA] WARNING: vision_lora_num_layers > 0 but zero trainable LoRA params detected.")
    print("=" * 60)


def get_libero_train_episodes(hf_dataset, train_ratio=0.9):
    """Standard LIBERO 90/10 per-task split."""
    episode_ids = np.array(hf_dataset["episode_index"])
    task_ids = np.array(hf_dataset["task_index"])

    ep_to_task: dict[int, int] = {}
    for ep_idx, task_idx in zip(episode_ids, task_ids):
        ep_to_task[int(ep_idx)] = int(task_idx)

    task_to_episodes: dict[int, list[int]] = {}
    for ep_idx, task_idx in ep_to_task.items():
        task_to_episodes.setdefault(task_idx, []).append(ep_idx)

    train_episodes: set[int] = set()
    test_episodes: set[int] = set()
    print(f"\nLIBERO train/test split (train_ratio={train_ratio}):")
    for task_idx, episodes in sorted(task_to_episodes.items()):
        episodes = sorted(episodes)
        n_train = max(1, int(len(episodes) * train_ratio))
        train_episodes.update(episodes[:n_train])
        test_episodes.update(episodes[n_train:])
        print(f"  task {task_idx:3d}: {len(episodes):3d} demos → {n_train} train | {len(episodes)-n_train} test "
              f"(train ep {episodes[0]}–{episodes[n_train-1]}, "
              f"test ep {episodes[n_train] if len(episodes) > n_train else 'none'}–{episodes[-1]})")

    print(f"  Total: {len(train_episodes)} train episodes, {len(test_episodes)} test episodes\n")
    return train_episodes


def train(
    output_dir,
    dataset_id="lerobot/libero",
    resume_from_checkpoint=None,
    train_ratio=1.0,
    batch_size=64,
    reset_lang_params=False,
):
    """Train the InterleavedFlowMatching model on LIBERO."""
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    training_steps = 200000
    progress_update_freq = 200
    checkpoint_freq = 1000
    image_transforms = get_augmentations()

    dataset_metadata = LeRobotDatasetMetadata(dataset_id, force_cache_sync=True, revision="main")
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    if len(output_features) == 0:
        raise ValueError("No output features (actions) found.")

    print('input_features:', list(input_features.keys()))
    print('output_features:', list(output_features.keys()))

    camera_keys = sorted([key for key, ft in input_features.items() if ft.type is FeatureType.VISUAL])
    print(f"Detected cameras ({len(camera_keys)}): {camera_keys}")

    state_key = next(
        (k for k in ("observation.state", "state") if k in input_features),
        None,
    )
    action_key = next(iter(output_features.keys()))
    if state_key is None:
        raise ValueError(f"No state key found in input_features: {list(input_features.keys())}")

    state_dim = input_features[state_key].shape[-1]
    action_dim = output_features[action_key].shape[-1]
    print(f"State key: '{state_key}' (dim={state_dim}), Action key: '{action_key}' (dim={action_dim})")

    obs = 2
    horizon = 64
    n_action_steps = 64

    cfg = InterleavedFlowMatchingConfig(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=obs,
        horizon=horizon,
        n_action_steps=n_action_steps,
        state_dim=state_dim,
        action_dim=action_dim,
        num_vlm_layers=16,
        num_cameras=len(camera_keys),
        cameras_for_vision_state_concat=camera_keys,
        action_dim_weights=[1.0] * action_dim,
        pos_decay_lambda=0.0,
        # Joint attention is heavy; keep vision LoRA off unless you've confirmed
        # batch_size headroom on your GPU.
        vision_lora_num_layers=0,
        # Default 8 latent "thought" tokens; set to 0 to ablate.
        num_latent_tokens=8,
        # Allow VLM attention to expert tokens (SmolVLA-style true interleaving).
        vlm_attends_to_expert=True,
    )

    if resume_from_checkpoint is not None:
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        policy = InterleavedFlowMatchingPolicy(cfg)

        ckpt_path = Path(resume_from_checkpoint)
        local_ckpt_path = ckpt_path if ckpt_path.exists() else Path(huggingface_hub.snapshot_download(resume_from_checkpoint))

        model_file = local_ckpt_path / "model.safetensors"
        if not model_file.exists():
            candidates = list(local_ckpt_path.glob("*.safetensors"))
            if not candidates:
                raise FileNotFoundError(f"No .safetensors file found in {local_ckpt_path}")
            model_file = candidates[0]

        import json
        step, epoch = 0, 0
        saved_cfg_json = {}
        for config_name in ("config.json", "pretrained_config.json"):
            config_file = local_ckpt_path / config_name
            if config_file.exists():
                with open(config_file) as f:
                    saved_cfg_json = json.load(f)
                step = saved_cfg_json.get("training_step", 0)
                epoch = saved_cfg_json.get("training_epoch", 0)
                saved_total = saved_cfg_json.get("training_steps_total", 0)
                if saved_total > 0:
                    training_steps = saved_total
                print(f"Read config from {config_file.name}: step={step}, epoch={epoch}, total={training_steps}")
                break
        if step == 0 and local_ckpt_path.name.startswith("checkpoint-"):
            step = int(local_ckpt_path.name.split("-")[1])

        ckpt_state = load_safetensors(model_file, device=str(device))

        # NOTE: the interleaved model is not key-compatible with the legacy
        # encoder-decoder checkpoints (no `context_proj`, no `robot_layer_projs`,
        # no `action_expert_layers`). Resume here only works for checkpoints
        # produced by this same script.
        has_lora_in_ckpt = any("lora_" in k for k in ckpt_state)
        if has_lora_in_ckpt and cfg.vision_lora_num_layers > 0:
            print(f"Checkpoint has LoRA weights — enabling LoRA (rank={cfg.lora_rank}) before load")
            # NOTE: enable_lora is not implemented on the interleaved model yet
            # (vision_lora_num_layers defaults to 0). If you wire it later,
            # call policy.model.enable_lora(...) here as in train_libero.py.
            _print_lora_status(policy, cfg)

        policy.train()
        policy.to(device)
        cur_state = policy.state_dict()
        filtered = {k: v for k, v in ckpt_state.items() if k in cur_state and cur_state[k].shape == v.shape}
        skipped = [k for k in ckpt_state if k not in filtered]
        missing = [k for k in cur_state if k not in ckpt_state]
        if skipped:
            print(f"Skipped {len(skipped)} keys (shape mismatch / removed): {skipped[:5]}")
        if missing:
            print(f"Missing {len(missing)} keys (will use init values): {missing[:5]}")
        policy.load_state_dict(filtered, strict=False)
        print(f"Loaded {len(filtered)}/{len(cur_state)} model keys")

        # Optionally reset language-conditioning params after loading. Use
        # this when resuming from a checkpoint that learned to suppress
        # language (negative bias / shrunk adaptor gamma) and you want a
        # clean baseline to test a new forcing strategy. DO NOT use this
        # for routine resume (after a crash or to continue same run) —
        # it will erase any genuine language-conditioning progress.
        if reset_lang_params:
            with torch.no_grad():
                if hasattr(policy.model, "lang_attn_bias"):
                    old = policy.model.lang_attn_bias.item()
                    policy.model.lang_attn_bias.zero_()
                    print(f"Reset lang_attn_bias: {old:+.4f} → 0.0000")
                if hasattr(policy.model, "lang_adaptor"):
                    rms_gamma = policy.model.lang_adaptor[1].weight
                    old_norm = rms_gamma.norm().item()
                    rms_gamma.fill_(1.0)
                    print(f"Reset lang_adaptor RMSNorm gamma: norm {old_norm:.3f} → {rms_gamma.norm().item():.3f}")
        else:
            # Show current state so you know what you're resuming with.
            if hasattr(policy.model, "lang_attn_bias"):
                bias_val = policy.model.lang_attn_bias.item()
                print(f"lang_attn_bias on resume: {bias_val:+.4f} (use --reset_lang_params to zero)")
            if hasattr(policy.model, "lang_adaptor"):
                norm = policy.model.lang_adaptor[1].weight.norm().item()
                print(f"lang_adaptor RMSNorm gamma norm: {norm:.3f} (init = 30.984)")

        preprocessor, postprocessor = make_pre_post_processors(policy.config, dataset_stats=dataset_metadata.stats)

        resume_lr = saved_cfg_json.get("optimizer_lr", cfg.optimizer_lr)
        trainable_params = [p for p in policy.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=resume_lr, weight_decay=1e-6)

        optimizer_state_path = local_ckpt_path / "optimizer_state.pth"
        if optimizer_state_path.exists():
            try:
                optimizer.load_state_dict(torch.load(optimizer_state_path, map_location=device))
                for pg in optimizer.param_groups:
                    pg['lr'] = resume_lr
                    pg['initial_lr'] = resume_lr
                print(f"Optimizer state loaded. LR reset to {resume_lr}")
            except ValueError as e:
                print(f"Skipping optimizer state — mismatch ({e})")

        resume_warmup = saved_cfg_json.get("scheduler_warmup_steps", cfg.scheduler_warmup_steps)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=resume_warmup, num_training_steps=training_steps)
        for _ in range(step):
            scheduler.step()
        print(f"Scheduler fast-forwarded to step {step}, LR = {optimizer.param_groups[0]['lr']:.2e}")
    else:
        policy = InterleavedFlowMatchingPolicy(cfg)
        policy.train()
        policy.to(device)
        if cfg.vision_lora_num_layers > 0:
            print(f"[WARN] vision_lora_num_layers={cfg.vision_lora_num_layers} but interleaved "
                  f"model has no enable_lora() — ignoring.")
        preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)
        step, epoch = 0, 0

        trainable_params = [p for p in policy.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in trainable_params)
        n_frozen = sum(p.numel() for p in policy.parameters() if not p.requires_grad)
        print(f"Total trainable parameters: {n_trainable:,}  (frozen: {n_frozen:,})")
        optimizer = torch.optim.Adam(trainable_params, lr=cfg.optimizer_lr, weight_decay=cfg.optimizer_weight_decay)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=cfg.scheduler_warmup_steps, num_training_steps=training_steps)

    if isinstance(preprocessor, torch.nn.Module):
        preprocessor.to(device)

    fps = dataset_metadata.fps if hasattr(dataset_metadata, "fps") and dataset_metadata.fps else 10
    frame_time = 1 / fps
    print(f"Dataset FPS: {fps}")

    obs_temporal_window = [-i * frame_time for i in range(obs)][::-1]
    action_temporal_window = [i * frame_time for i in range(horizon)]

    delta_timestamps = {
        state_key: obs_temporal_window,
        action_key: action_temporal_window,
        **{key: [0.0] for key in camera_keys},
    }

    dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps, force_cache_sync=True, revision="main", tolerance_s=0.04)
    print(f"Dataset loaded: {len(dataset)} total frames")

    task_idx_to_description: dict[int, str] = {}
    try:
        tasks_parquet_path = dataset.root / "meta" / "tasks.parquet"
        if tasks_parquet_path.exists():
            tasks_df = pd.read_parquet(tasks_parquet_path)
            if "task_index" in tasks_df.columns:
                task_idx_to_description = {int(row["task_index"]): str(idx) for idx, row in tasks_df.iterrows()}
            print(f"Loaded {len(task_idx_to_description)} task descriptions from tasks.parquet")
    except Exception as e:
        print(f"Warning: could not load tasks.parquet: {e}")

    train_episodes = get_libero_train_episodes(dataset.hf_dataset, train_ratio=train_ratio)
    all_episode_ids = np.array(dataset.hf_dataset["episode_index"])
    valid_indices = [i for i, ep in enumerate(all_episode_ids) if int(ep) in train_episodes]
    dataset = Subset(dataset, valid_indices)
    print(f"Training subset: {len(dataset)} frames from {len(train_episodes)} episodes")

    ep_ids_subset = all_episode_ids[np.array(valid_indices)]
    ep_changes = np.where(np.diff(ep_ids_subset) != 0)[0] + 1
    ep_from = np.concatenate([[0], ep_changes]).tolist()
    ep_to = np.concatenate([ep_changes, [len(valid_indices)]]).tolist()
    sampler = EpisodeAwareSampler(
        dataset_from_indices=ep_from,
        dataset_to_indices=ep_to,
        drop_n_first_frames=0,
        drop_n_last_frames=0,
        shuffle=True,
    )

    print(f"Batch size: {batch_size}  (interleaved model is memory-heavy; "
          f"drop further if you hit OOM)")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=8,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    print("Starting training loop...")
    done = False
    prog_bar = tqdm(total=training_steps, desc="Training Progress", initial=step)
    while not done:
        epoch += 1
        for batch in dataloader:
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)

            if "task" in batch and isinstance(batch["task"], (list, tuple)):
                batch["task_description"] = batch["task"]
            elif task_idx_to_description and "task_index" in batch:
                task_indices = batch["task_index"]
                if isinstance(task_indices, torch.Tensor) and task_indices.dim() > 1:
                    task_indices = task_indices[:, 0]
                batch["task_description"] = [task_idx_to_description.get(int(ti), "") for ti in task_indices]

            batch = apply_image_augmentations(batch, camera_keys, image_transforms)
            batch = apply_joint_augmentations(batch, state_key)

            if step == 0:
                raw_st = batch[state_key].float()
                print(f"\nRaw (pre-norm) {state_key}: min={raw_st.min():.4f}  max={raw_st.max():.4f}  std={raw_st.std():.4f}")

            batch = preprocessor(batch)

            if step == 0:
                pad_key = next((k for k in batch if "pad" in k.lower() and "action" in k.lower()), None)
                if pad_key is None:
                    print("WARNING: no action pad key found in batch")
                else:
                    print(f"Action pad key='{pad_key}', pad fraction: {batch[pad_key].float().mean().item():.2%}")

            autocast_ctx = (
                torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                if device.type == "cuda"
                else torch.autocast(device_type="cpu", enabled=False)
            )
            with autocast_ctx:
                loss, _ = policy.forward(batch)

            loss.backward()

            if step % progress_update_freq == 0:
                print(f"\n--- Gradient Analysis at Step {step} ---")

                def _grad_stats(prefix):
                    total, count = 0.0, 0
                    for name, param in policy.model.named_parameters():
                        if param.requires_grad and prefix in name and param.grad is not None:
                            total += param.grad.abs().mean().item() * param.numel()
                            count += param.numel()
                    return (total / count, count) if count > 0 else (None, 0)

                # NOTE prefixes specific to the interleaved model:
                #   "expert_layers"          → the 16 trainable expert blocks
                #                              (replaces robot_layer_projs + action_expert)
                #   "action_in_proj" / etc.  → action embedding head
                #   "action_time_mlp"        → time conditioning fusion
                for label, prefix in [
                    ("Vision",         "vision_model"),
                    ("Vision LoRA",    "lora_"),
                    ("Connector",      "connector"),
                    ("State Enc",      "state_encoder"),
                    ("Robot CNN",      "robot_visual_encoder"),
                    ("Expert Layers",  "expert_layers"),
                    ("Action In/Out",  "action_"),
                    ("Final Norm",     "final_norm"),
                    ("Latent Tokens",  "latent_embs"),
                    ("Lang Adaptor",   "lang_adaptor"),
                ]:
                    grad, n = _grad_stats(prefix)
                    print(f"  {label:14s} - Avg Abs Grad: {grad:.6f} ({n} params)" if grad is not None else f"  {label:14s} - no grad")

                if hasattr(policy.model, "latent_embs"):
                    lat = policy.model.latent_embs.detach()
                    if lat.shape[1] > 0:
                        normed = torch.nn.functional.normalize(lat[0], dim=-1)
                        cos = (normed @ normed.t())
                        off_diag = cos[~torch.eye(cos.shape[0], dtype=torch.bool, device=cos.device)]
                        grad_param = policy.model.latent_embs.grad
                        grad_norm_l = grad_param.norm().item() if grad_param is not None else float("nan")
                        print(f"  Latent stats   - grad_norm: {grad_norm_l:.4e}  "
                              f"mean |cos|: {off_diag.abs().mean().item():.3f}  "
                              f"max |cos|: {off_diag.abs().max().item():.3f}")

                # Language conditioning diagnostics. We added lang_adaptor
                # (zero-init residual MLP) and lang_attn_bias (scalar) to give
                # the model a way to amplify language signal. They only matter
                # if their values move away from initialisation. Watch:
                #   - lang_attn_bias value:      0 = unused, > 0 = model uses it
                #   - lang_attn_bias gradient:   should be non-zero if signal flows
                #   - lang_adaptor norm:         tracks whether the residual MLP
                #                                has learned a non-trivial transform
                # If after several thousand steps the value stays ~0 and the
                # gradient is tiny, the "chicken-and-egg" deadlock is real and
                # we need more active forcing (vision dropout, contrastive
                # auxiliary loss, etc.).
                if hasattr(policy.model, "lang_attn_bias"):
                    bias_tensor = policy.model.lang_attn_bias.detach()
                    softplus_vals = torch.nn.functional.softplus(bias_tensor).cpu()
                    grad = policy.model.lang_attn_bias.grad
                    grad_norm_str = f"{grad.norm().item():.4e}" if grad is not None else "None (not in graph)"
                    # Per-layer values — compact one-line representation.
                    softplus_str = "[" + " ".join(f"{v:.2f}" for v in softplus_vals.tolist()) + "]"
                    sp_min = softplus_vals.min().item()
                    sp_max = softplus_vals.max().item()
                    sp_mean = softplus_vals.mean().item()
                    argmin = softplus_vals.argmin().item()
                    argmax = softplus_vals.argmax().item()
                    print(f"  Lang attn bias - softplus per-layer: {softplus_str}")
                    print(f"                   min={sp_min:.3f} (L{argmin})  "
                          f"max={sp_max:.3f} (L{argmax})  "
                          f"mean={sp_mean:.3f}  grad_norm: {grad_norm_str}")

                if hasattr(policy.model, "lang_adaptor"):
                    adaptor_w_norm = sum(
                        p.detach().norm().item() ** 2
                        for p in policy.model.lang_adaptor.parameters()
                    ) ** 0.5
                    adaptor_g_norm_sq = 0.0
                    for p in policy.model.lang_adaptor.parameters():
                        if p.grad is not None:
                            adaptor_g_norm_sq += p.grad.norm().item() ** 2
                    adaptor_g_norm = adaptor_g_norm_sq ** 0.5
                    print(f"  Lang adaptor   - weight_norm: {adaptor_w_norm:.4e}   grad_norm: {adaptor_g_norm:.4e}")

                print("--- End Gradient Analysis ---\n")

            trainable_params = [p for p in policy.parameters() if p.requires_grad]
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if step % progress_update_freq == 0:
                lr = optimizer.param_groups[0]['lr']
                prog_bar.set_description(f"Epoch {epoch}, Step {step}")
                prog_bar.set_postfix({"loss": f"{loss.item():.3f}", "lr": f"{lr:.2e}", "grad_norm": f"{grad_norm:.2f}"})

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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_id", type=str, default="lerobot/libero")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="Fraction of episodes per task used for training (standard LIBERO = 0.9)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size. Interleaved model is memory-heavy at d_model=960; "
                             "drop to 32 or 16 if you hit OOM on smaller GPUs.")
    parser.add_argument("--reset_lang_params", action="store_true",
                        help="Zero out lang_attn_bias and reset lang_adaptor RMSNorm gamma "
                             "to 1 after loading checkpoint. Use when testing a new language "
                             "forcing strategy from a checkpoint that learned to suppress "
                             "language. DO NOT use for routine resume — erases progress.")
    args = parser.parse_args()
    train(**vars(args))
