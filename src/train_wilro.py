import json
import time
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
from lerobot.datasets.compute_stats import aggregate_stats
import numpy as np
from torch.utils.data import ConcatDataset

# Wilro-specific components
from models.wilro.wilro_config import WilroConfig
from models.wilro.wilro_policy import WilroPolicy
from models.wilro.processor_wilro import make_pre_post_processors
from models.wiltechs_vla.task_rewrites import rewrite_instruction

from torchvision.transforms import v2
from transformers import get_cosine_schedule_with_warmup


# Detect the best available device
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
# Augmentation helpers (same recipe as train_transformer.py)
# ---------------------------------------------------------------------------
def get_augmentations():
    """Image augmentation transform shared across all cameras of a sample."""
    return v2.Compose([
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    ])


def apply_joint_augmentations(batch):
    """Add small Gaussian noise to observation.state (50% probability)."""
    if torch.rand(1).item() > 0.5:
        if "observation.state" in batch:
            noise = torch.randn_like(batch["observation.state"]) * 0.01
            batch["observation.state"] = batch["observation.state"] + noise
    return batch


def apply_image_augmentations(batch, camera_keys, transform):
    """Apply the same random color jitter to all cameras within each sample.

    For each sample in the batch, all camera images are stacked into a single
    tensor and passed through the transform in one call. torchvision v2 samples
    random parameters once per forward() call and applies them identically to
    every image in the tensor — so front/gripper/right cameras always receive
    the same brightness/contrast/saturation/hue shift, keeping cross-camera
    color consistency.

    Handles both (C, H, W) and (T, C, H, W) camera tensors.
    """
    present_keys = [k for k in camera_keys if k in batch and isinstance(batch[k], torch.Tensor)]
    if not present_keys:
        return batch

    B = batch[present_keys[0]].shape[0]
    for b in range(B):
        sample_img = batch[present_keys[0]][b]
        has_time_dim = sample_img.dim() == 4
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
# Gradient analysis tailored to wilro components
# ---------------------------------------------------------------------------
def _rss_gb():
    """Resident set of this process AND its dataloader workers, in GB.

    Workers are separate processes, so the parent's own RSS misses most of
    the growth that ends these runs."""
    try:
        import os
        def rss(pid):
            with open(f"/proc/{pid}/status") as f:
                for ln in f:
                    if ln.startswith("VmRSS:"):
                        return int(ln.split()[1]) / 1048576.0
            return 0.0
        me = os.getpid()
        try:
            with open(f"/proc/{me}/task/{me}/children") as f:
                kids = [int(x) for x in f.read().split()]
        except Exception:
            kids = []
        return rss(me) + sum(rss(k) for k in kids), len(kids)
    except Exception:
        return None, 0


_RSS_PREV: list = []


def _log_gradient_analysis(policy, step: int) -> None:
    print(f"\n--- Gradient Analysis at Step {step} ---")
    tot, nk = _rss_gb()
    if tot is not None:
        # The DELTA is the diagnostic, not the level. A large constant baseline
        # is just the model, the CUDA context and the Arrow table; what kills
        # the run is growth, and the OOM killer takes a WORKER, which surfaces
        # as "DataLoader worker (pid ...) is killed by signal: Killed" raised
        # from wherever the main process happened to be -- never from the loader.
        d = f"{tot - _RSS_PREV[-1]:+.2f} GB since step {_RSS_PREV[0]:.0f}" \
            if _RSS_PREV else "baseline"
        note = ""
        if _RSS_PREV and tot - _RSS_PREV[-1] > 0.3:
            note = ("   <-- GROWING; the workers are un-sharing the dataset. "
                    "Lower --num_workers / --prefetch_factor before it is "
                    "killed.")
        print(f"  Host RAM (self + {nk} worker(s)): {tot:.1f} GB  ({d}){note}")
        _RSS_PREV[:] = [step, tot]

    def _grad_stats(prefix: str):
        total, count = 0.0, 0
        for name, param in policy.model.named_parameters():
            if param.requires_grad and prefix in name and param.grad is not None:
                total += param.grad.abs().mean().item() * param.numel()
                count += param.numel()
        return (total / count, count) if count > 0 else (None, 0)

    for label, prefix in [
        ("Vision LoRA",      "vision_model.encoder.layers"),  # SigLIP ViT LoRA (trainable)
        ("Text LoRA",        "text_model.layers"),            # Text model LoRA (trainable)
        ("Connector (frzn)", "connector"),
        ("State Enc",        "state_encoder"),
        ("Robot CA K/V Proj","robot_ca_k_proj"),
        ("Robot CA V Proj",  "robot_ca_v_proj"),
        ("Robot CA Norm",    "robot_ca_norm"),
        ("DiT layers",       "dit_layers"),
        ("  ├─ Self-attn",   "sa_"),
        ("  ├─ VLM CA",      "ca_"),
        ("  ├─ Robot CA",    "robot_ca_"),
        ("  └─ FFN",         "ffn"),
        ("Action In/Out",    "action_"),
        ("Sink token",       "sink_token"),
        ("Final Norm",       "final_norm"),
        ("Time MLP",         "time_embedder"),
        ("Latent Gen",       "latent_generator"),
    ]:
        grad, n = _grad_stats(prefix)
        if grad is not None:
            print(f"  {label:22s} - Avg Abs Grad: {grad:.6f} ({n:,} params)")

    stats = getattr(policy.model, "_last_attention_stats", None)
    if stats:
        # Match DiT sequence order: [SINK, latent, state, language, prefix, robot, action]
        order = ["sink", "latent", "state", "language", "prefix", "robot", "action"]
        ordered = [(k, stats[k]) for k in order if k in stats]
        cells = "  ".join(f"{k}={v*100:5.1f}%" for k, v in ordered)
        print(f"  Action→ self-attn : {cells}    (last DiT layer)")

    x_stats = getattr(policy.model, "_last_cross_attention_stats", None)
    if x_stats:
        # VLM cross-attention: vision vs language
        vlm_order = ["vision", "language"]
        vlm_ordered = [(k, x_stats[k]) for k in vlm_order if k in x_stats]
        vlm_cells = "  ".join(f"{k}={v*100:5.1f}%" for k, v in vlm_ordered)
        print(f"  Action→ VLM x-attn  : {vlm_cells}    (cross-attn to VLM KV)")

    # Robot cross-attention stats (if captured)
    robot_ca_stats = getattr(policy.model, "_last_robot_cross_attention_stats", None)
    if robot_ca_stats:
        robot_cells = "  ".join(f"{k}={v*100:5.1f}%" for k, v in robot_ca_stats.items())
        print(f"  Action→ Robot x-attn: {robot_cells}    (cross-attn to Robot CNN)")

    comps = getattr(policy.model, "_last_loss_components", None)
    cw = getattr(policy.model.config, "contrastive_loss_weight", 0.0)
    if comps is not None and cw > 0.0:
        margin = getattr(policy.model.config, "contrastive_margin", 0.05)
        main_v = comps.get("main", float("nan"))
        contr_v = comps.get("contrastive", float("nan"))
        pct = (contr_v / margin * 100.0) if margin > 0 else float("nan")
        print(f"  Contrastive       - main: {main_v:.4f}   contrastive: {contr_v:.4f} "
              f"({pct:.0f}% of margin {margin:.3f})   weight: {cw}")

    print("--- End Gradient Analysis ---\n")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train(output_dir, dataset_id="ISdept/piper_arm", resume_from_checkpoint=None,
          gradient_checkpointing=False, max_episode_index=None, batch_size=64,
          contrastive_loss_weight=0.1, contrastive_margin=0.05,
          contrastive_hard_negatives=False,
          lock_joint_index: int | None = 3, kv_capture_strategy: str = "last",
          kv_capture_layers: list | None = None,
          cameras: list | None = None,
          rewrite_instructions: bool = False,
          rewrite_augment: bool = False,
          noise_temporal_correlation: float = 0.0,
          gripper_phase_weight: float = 1.0,
          time_sampling: str = "uniform",
          time_lognormal_mean: float = -0.5,
          time_lognormal_std: float = 1.0,
          paraphrase_augment: bool = False,
          paraphrase_limit: int = 8,
          paraphrase_file: str = "",
          paraphrase_min_variants: int = 5,
          training_steps: int | None = None,
          n_obs_steps: int | None = None,
          val_episodes: int = 0,
          val_every: int = 500,
          val_max_batches: int = 20,
          progress_update_freq: int = 200,
          num_workers: int = 8,
          start_step_override: int = -1,
          lr: float | None = None,
          warmup_steps: int | None = None,
          lora_rank: int | None = None,
          lora_alpha: float | None = None,
          vision_lora_num_layers: int | None = None,
          download_progress: bool = False,
          cache_sync: bool = False,
          load_image_size: int = 0,
          prefetch_factor: int = 2):
    """Train the Wilro (SmolVLM2 KV-cache → DiT) flow matching model.

    `dataset_id` may be a single id or a list. Multiple datasets are concatenated
    and assumed HOMOGENEOUS (same robot / cameras / state+action dims / fps) — e.g.
    several piper sets — and their normalization stats are aggregated. For
    mixed-robot data use the canonical train_finetune.py path instead.
    """
    dataset_ids = [dataset_id] if isinstance(dataset_id, str) else list(dataset_id)
    if not dataset_ids:
        raise ValueError("At least one dataset_id is required.")

    # huggingface_hub draws a per-file tqdm for every repo it touches, and this
    # trainer touches each dataset twice (metadata, then the dataset itself).
    # At ~14k files that is thousands of redrawn lines before the first step,
    # which buries the schema and normalisation output that actually needs
    # reading. One status line each instead; --download_progress restores the
    # bars for a genuinely first-time pull.
    if not download_progress:
        try:
            from huggingface_hub.utils import disable_progress_bars
            disable_progress_bars()
        except Exception:
            pass
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    # The cosine schedule spans this, so it is not a "stop whenever" ceiling:
    # interrupting a 200k run at 30k leaves the LR mid-cosine and the model
    # never annealed. Pick the number you intend to finish.
    steps_cli = training_steps                       # None unless asked for
    training_steps = 200000 if steps_cli is None else int(steps_cli)
    progress_update_freq = max(1, int(progress_update_freq))
    checkpoint_freq = 1000
    image_transforms = get_augmentations()

    # Load metadata for all datasets. Schema is taken from the first and the rest
    # are validated against it (homogeneous assumption).
    # force_cache_sync re-verifies every file in the repo against the hub on
    # each launch -- ~14k HEAD requests for a converted VLABench, before a
    # single step. It only matters when the remote may have changed under a
    # cache that already exists, so it is opt-in via --cache_sync.
    metas = {}
    for did in dataset_ids:
        _t = time.time()
        print(f"[data] reading metadata for {did}"
              + ("  (--cache_sync: re-verifying every file against the hub)"
                 if cache_sync else ""), flush=True)
        metas[did] = LeRobotDatasetMetadata(did, force_cache_sync=cache_sync,
                                            revision="main")
        print(f"[data]   ...{time.time() - _t:.0f}s", flush=True)
    ref_meta = metas[dataset_ids[0]]
    features = dataset_to_policy_features(ref_meta.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    if len(output_features) == 0:
        raise ValueError("No output features (actions) found! Check your dataset schema.")

    print('input_features:', input_features)
    print('output_features:', output_features)

    # Detect all available cameras from dataset features
    all_camera_keys = sorted([key for key, ft in input_features.items() if ft.type is FeatureType.VISUAL])
    
    # Filter cameras if --cameras is specified
    if cameras is not None and len(cameras) > 0:
        camera_keys = [c for c in cameras if c in all_camera_keys]
        missing = [c for c in cameras if c not in all_camera_keys]
        if missing:
            print(f"WARNING: Requested cameras not found in dataset: {missing}")
        if not camera_keys:
            raise ValueError(
                f"None of the requested cameras {cameras} exist in dataset. "
                f"Available cameras: {all_camera_keys}"
            )
        print(f"Camera filter applied: using {camera_keys} (from available {all_camera_keys})")
    else:
        camera_keys = all_camera_keys
    
    state_dim = input_features["observation.state"].shape[-1] if "observation.state" in input_features else 7
    action_dim = next(iter(output_features.values())).shape[-1]
    print(f"Detected cameras ({len(camera_keys)}): {camera_keys}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Validate the other datasets share the same schema.
    # Image RESOLUTION is tracked separately from the schema check below. It is
    # not a schema conflict -- the model pads to square and interpolates every
    # frame to vision_input_size regardless -- but ConcatDataset collates raw
    # tensors, so two sets at 480 and 256 pass every check here and then die in
    # torch.stack, inside a worker if num_workers > 0:
    #   "stack expects each tensor to be equal size, but got [3, 256, 256] at
    #    entry 0 and [3, 480, 480] at entry 3"
    # Resolved by resizing at LOAD time instead (see resize_to below), which is
    # the same operation the model would do later and therefore lossless.
    vis_shapes = {dataset_ids[0]: {k: tuple(ft.shape)
                                   for k, ft in input_features.items()
                                   if ft.type is FeatureType.VISUAL}}
    for did in dataset_ids[1:]:
        f = dataset_to_policy_features(metas[did].features)
        out_f = {k: ft for k, ft in f.items() if ft.type is FeatureType.ACTION}
        in_f = {k: ft for k, ft in f.items() if k not in out_f}
        cks = sorted(k for k, ft in in_f.items() if ft.type is FeatureType.VISUAL)
        sd = in_f["observation.state"].shape[-1] if "observation.state" in in_f else 7
        ad = next(iter(out_f.values())).shape[-1]
        vis_shapes[did] = {k: tuple(ft.shape) for k, ft in in_f.items()
                           if ft.type is FeatureType.VISUAL}
        if cks != camera_keys or sd != state_dim or ad != action_dim:
            raise ValueError(
                f"Dataset '{did}' schema differs from '{dataset_ids[0]}':\n"
                f"  cameras {cks} vs {camera_keys}\n"
                f"  state_dim {sd} vs {state_dim}, action_dim {ad} vs {action_dim}\n"
                f"train_wilro.py concatenation requires a homogeneous schema. For "
                f"mixed robots use the canonical train_finetune.py path."
            )

    # Aggregate normalization stats across datasets (count-weighted mean, global
    # min/max, combined std). One dataset keeps its own stats unchanged.
    if len(dataset_ids) == 1:
        combined_stats = ref_meta.stats
    else:
        combined_stats = aggregate_stats([metas[did].stats for did in dataset_ids])
        print(f"Aggregated normalization stats across {len(dataset_ids)} datasets.")

    # Training parameters — match train_transformer.py for like-for-like comparison
    obs = 2 if n_obs_steps is None else max(1, int(n_obs_steps))
    horizon = 64
    n_action_steps = 64

    # Build action_dim_weights — uniform by default. piper_arm's joint 4
    # (index 3) is always 0, so for that dataset pass --lock_joint_index 3
    # (the default) to zero out its loss contribution. For LIBERO / other
    # full-DOF robots, pass --lock_joint_index "" (None) to weight all dims.
    action_dim_weights = [1.0] * action_dim
    if lock_joint_index is not None and 0 <= lock_joint_index < action_dim:
        action_dim_weights[lock_joint_index] = 0.0
        print(f"Locking action dim {lock_joint_index} (weight=0); "
              f"action_dim_weights={action_dim_weights}")
    else:
        print(f"All {action_dim} action dims weighted equally; "
              f"action_dim_weights={action_dim_weights}")

    if paraphrase_augment:
        # Preflight, not a runtime warning. A sentence with no written variants
        # trains UNAUGMENTED while the rest vary, so the model keeps surface
        # form as a usable key for exactly those tasks -- and the run cannot
        # answer whether augmentation works. Too long a run to find that out
        # from the eval.
        from libero_paraphrase import coverage, instruction_strings, load_table
        # Union over every dataset: with several --dataset_id the task lists
        # differ, and an instruction that only appears in the second one still
        # needs variants.
        instructions, seen = [], set()
        for did in dataset_ids:
            raw = getattr(metas[did], "tasks", None)
            if raw is None:
                continue
            for ins in instruction_strings(raw):
                key = " ".join(str(ins).split())
                if key not in seen:
                    seen.add(key)
                    instructions.append(key)
        if not instructions:
            print("[paraphrase] dataset metadata exposes no task list; coverage "
                  "cannot be checked here. Run\n"
                  "  python -m libero_paraphrase --dataset_id <id> --min_variants N\n"
                  "before trusting this run.")
        else:
            table, under = coverage(
                instructions, paraphrase_limit, paraphrase_min_variants,
                load_table(paraphrase_file) if paraphrase_file else None)
            sizes = sorted(len(v) for v in table.values())
            print(f"[paraphrase] {len(table)} instructions, "
                  f"{len(table) - len(under)} at >= {paraphrase_min_variants} "
                  f"variants (min {sizes[0]}, median {sizes[len(sizes) // 2]}, "
                  f"max {sizes[-1]})")
            if under:
                shown = "\n".join(f"    {len(table[k]):>2}  {k}" for k in under[:12])
                raise SystemExit(
                    f"[paraphrase] {len(under)} instruction(s) below "
                    f"--paraphrase_min_variants {paraphrase_min_variants}:\n"
                    f"{shown}\n"
                    f"{'    ...' if len(under) > 12 else ''}\n"
                    f"  Write a table for these and pass --paraphrase_file:\n"
                    f"    python -m libero_paraphrase --dataset_id "
                    f"{dataset_ids[0]} --out para.json\n"
                    f"  then hand-edit the entries listed as UNDER. Lower "
                    f"--paraphrase_min_variants only if you accept that those "
                    f"tasks train unaugmented.")

    # LoRA sizing. alpha defaults to 2x rank because LoRALinear scales by
    # alpha/rank, and the shipped pair is 32/16 = 2.0 -- holding alpha fixed
    # while raising rank would quietly HALVE the adapter's effective strength,
    # moving two variables when only one was asked for.
    lora_kw: dict = {}
    if lora_rank is not None:
        lora_kw["lora_rank"] = int(lora_rank)
        lora_kw["lora_alpha"] = float(lora_alpha if lora_alpha is not None
                                      else 2.0 * int(lora_rank))
    elif lora_alpha is not None:
        lora_kw["lora_alpha"] = float(lora_alpha)
    if vision_lora_num_layers is not None:
        lora_kw["vision_lora_num_layers"] = int(vision_lora_num_layers)

    # Build wilro config
    cfg = WilroConfig(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=obs,
        horizon=horizon,
        n_action_steps=n_action_steps,
        state_dim=state_dim,
        action_dim=action_dim,
        num_vlm_layers=16,  # DiT depth = number of VLM KV pairs consumed
        kv_capture_strategy=kv_capture_strategy,
        kv_capture_layers=kv_capture_layers or [],
        num_cameras=len(camera_keys),
        cameras_for_vision_state_concat=camera_keys,
        action_dim_weights=action_dim_weights,
        # n_action_steps == horizon → no exponential decay needed.
        pos_decay_lambda=0.0,
        contrastive_loss_weight=contrastive_loss_weight,
        contrastive_margin=contrastive_margin,
        contrastive_hard_negatives=contrastive_hard_negatives,
        noise_temporal_correlation=noise_temporal_correlation,
        gripper_phase_weight=gripper_phase_weight,
        gripper_action_index=action_dim - 1,  # LIBERO OSC: gripper is the last dim
        time_sampling=time_sampling,
        time_lognormal_mean=time_lognormal_mean,
        time_lognormal_std=time_lognormal_std,
        **({} if lr is None else {"optimizer_lr": float(lr)}),
        **({} if warmup_steps is None else {"scheduler_warmup_steps": int(warmup_steps)}),
        **lora_kw,
        paraphrase_augment=paraphrase_augment,
        paraphrase_limit=paraphrase_limit,
        paraphrase_file=paraphrase_file,
        paraphrase_min_variants=paraphrase_min_variants,
    )

    # Model + checkpoint loading
    if resume_from_checkpoint is not None:
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        policy = WilroPolicy(cfg)

        ckpt_path = Path(resume_from_checkpoint)
        if ckpt_path.exists():
            local_ckpt_path = ckpt_path
            print(f"Using local checkpoint: {local_ckpt_path}")
        else:
            print(f"Local path not found, downloading from HuggingFace Hub: {resume_from_checkpoint}")
            local_ckpt_path = Path(huggingface_hub.snapshot_download(resume_from_checkpoint))

        model_file = local_ckpt_path / "model.safetensors"
        if not model_file.exists():
            candidates = list(local_ckpt_path.glob("*.safetensors"))
            if not candidates:
                raise FileNotFoundError(f"No .safetensors file found in {local_ckpt_path}")
            model_file = candidates[0]

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
                # An explicit --training_steps wins over the checkpoint's. The
                # opposite order is how --lr was silently thrown away on every
                # resume in this repo (4caed2d); the schedule is worth even
                # more than the peak LR, since it decides whether the run
                # anneals at all.
                if steps_cli is not None and saved_total > 0 and saved_total != steps_cli:
                    print(f"--training_steps {steps_cli} OVERRIDES the "
                          f"checkpoint's {saved_total}; the cosine schedule is "
                          f"rebuilt over {steps_cli} and the LR at step {step} "
                          f"will differ from the original run's.")
                elif saved_total > 0:
                    training_steps = saved_total
                print(f"Read config from {config_file.name}: step={step}, epoch={epoch}, training_steps_total={training_steps}")
                # Warn only on an ACTUAL geometry change. Passing --lora_rank 64
                # to continue a run that already trained at 64 is the normal way
                # to resume, and a warning there says the adapters are being
                # discarded when they load fine -- which is worth aborting over
                # if believed.
                changed = {k: (saved_cfg_json.get(k), v) for k, v in lora_kw.items()
                           if k in saved_cfg_json and saved_cfg_json[k] != v}
                if changed:
                    detail = ", ".join(f"{k}: {a} -> {b}" for k, (a, b) in changed.items())
                    print(f"\n*** LoRA geometry CHANGED ({detail}). The "
                          f"checkpoint's adapters have different shapes, so they "
                          f"will be SKIPPED and the vision adapter restarts at "
                          f"zero -- the frozen base SigLIP, i.e. every bit of "
                          f"visual adaptation this checkpoint learned is "
                          f"discarded. Check 'Skipped N checkpoint keys' below. "
                          f"***\n")
                elif lora_kw:
                    print(f"LoRA geometry matches the checkpoint "
                          f"({', '.join(f'{k}={v}' for k, v in lora_kw.items())}); "
                          f"adapters will load.")
                break
        if step == 0 and local_ckpt_path.name.startswith("checkpoint-"):
            step = int(local_ckpt_path.name.split("-")[1])
        if start_step_override >= 0:
            # Loading weights from a run on a DIFFERENT dataset is not a
            # resume, and inheriting its step counter silently does three
            # things nobody asked for: the cosine is fast-forwarded, so the
            # new run starts mid-decay with no warmup; the step budget is
            # short by however far the old run got; and the progress bar
            # reports a number that belongs to another dataset.
            print(f"--start_step_override {start_step_override}: taking the "
                  f"WEIGHTS from step {step} but restarting the counter, so "
                  f"the schedule is rebuilt from scratch (warmup included) "
                  f"over {training_steps} steps. Epoch resets too "
                  f"(checkpoint said {epoch}) -- it counts passes over THIS "
                  f"dataset, and carrying the previous run's total over makes "
                  f"the saved training_epoch unreadable: a 7-epoch run on a "
                  f"new set was saved as 21 and read back as heavy "
                  f"overtraining that never happened.")
            step = int(start_step_override)
            epoch = 0
        print(f"Resuming from step {step}, epoch {epoch}")

        print(f"Loading weights from: {model_file}")
        ckpt_state = load_safetensors(model_file, device=str(device))

        policy.train()
        policy.to(device)
        cur_state = policy.state_dict()
        filtered = {k: v for k, v in ckpt_state.items() if k in cur_state and cur_state[k].shape == v.shape}

        skipped_ckpt = [k for k in ckpt_state if k not in filtered]
        missing_from_ckpt = [k for k in cur_state if k not in ckpt_state]
        if skipped_ckpt:
            print(f"Skipped {len(skipped_ckpt)} checkpoint keys (shape mismatch / removed): {skipped_ckpt[:10]}")
        if missing_from_ckpt:
            print(f"Missing {len(missing_from_ckpt)} keys not in checkpoint (will use init values): {missing_from_ckpt[:10]}")
        policy.load_state_dict(filtered, strict=False)
        print(f"Loaded {len(filtered)}/{len(cur_state)} model keys from checkpoint ({len(ckpt_state)} keys in file)")

        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            dataset_stats=combined_stats,
        )

        # The cosine scheduler's base LR must be the PEAK (pre-decay) value: the
        # decay is reconstructed purely by fast-forwarding scheduler.step() `step`
        # times below. The checkpoint's saved "optimizer_lr" is the ALREADY-DECAYED
        # lr (overwritten at save time), so using it as the base double-applies the
        # decay → peak·cos(step)². Use the config peak (cfg.optimizer_lr — not in
        # the WilroConfig kwargs, so it's the default peak) so fast-forwarding
        # rebuilds the correct peak·cos(step). Matches train_community.py.
        base_lr = cfg.optimizer_lr
        resume_warmup = saved_cfg_json.get("scheduler_warmup_steps", cfg.scheduler_warmup_steps)
        print(f"Scheduler base (peak) LR: {base_lr:.2e}  (decay rebuilt by "
              f"fast-forwarding to step {step})")

        trainable_params = [p for p in policy.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=base_lr, weight_decay=cfg.optimizer_weight_decay)
        print(f"Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        optimizer_state_path = local_ckpt_path / "optimizer_state.pth"
        if optimizer_state_path.exists():
            try:
                optimizer.load_state_dict(torch.load(optimizer_state_path, map_location=device))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = base_lr
                    param_group['initial_lr'] = base_lr
                print(f"Optimizer state loaded. Scheduler base LR set to peak {base_lr:.2e}")
            except ValueError as e:
                print(f"Skipping optimizer state — architecture mismatch ({e})")

        warmup_steps = resume_warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=training_steps,
        )
        for _ in range(step):
            scheduler.step()
        print(f"Scheduler fast-forwarded to step {step}, LR = {optimizer.param_groups[0]['lr']:.2e}")
    else:
        policy = WilroPolicy(cfg)
        policy.train()
        policy.to(device)

        preprocessor, postprocessor = make_pre_post_processors(
            cfg,
            dataset_stats=combined_stats,
        )
        step = 0
        epoch = 0

        trainable_params = [p for p in policy.parameters() if p.requires_grad]
        n_frozen = sum(p.numel() for p in policy.parameters() if not p.requires_grad)
        print(f"Total trainable parameters: {sum(p.numel() for p in trainable_params):,}  "
              f"(frozen: {n_frozen:,})")

        fresh_lr = cfg.optimizer_lr
        fresh_warmup = cfg.scheduler_warmup_steps
        optimizer = torch.optim.Adam(trainable_params, lr=fresh_lr, weight_decay=cfg.optimizer_weight_decay)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=fresh_warmup,
            num_training_steps=training_steps,
        )

    # Optional DiT gradient checkpointing (frozen VLM is unaffected — it runs in no_grad).
    if gradient_checkpointing and hasattr(policy.model, "gradient_checkpointing_enable"):
        policy.model.gradient_checkpointing_enable()

    if isinstance(preprocessor, torch.nn.Module):
        preprocessor.to(device)

    # Dataset setup — read fps from metadata instead of hardcoding. piper_arm
    # is 30 fps but libero / community datasets are commonly 10 fps; using a
    # mismatched frame_time makes every requested delta_timestamp fall outside
    # tolerance_s and the constructor raises. All datasets must share one fps so
    # the action horizon means the same real time everywhere.
    fps = int(getattr(ref_meta, "fps", 30) or 30)
    for did in dataset_ids[1:]:
        f2 = int(getattr(metas[did], "fps", fps) or fps)
        if f2 != fps:
            raise ValueError(
                f"Dataset '{did}' fps={f2} differs from '{dataset_ids[0]}' fps={fps}. "
                f"Resample to a common fps before mixing — the chunk horizon must "
                f"cover the same real time across datasets."
            )
    frame_time = 1 / fps
    print(f"Dataset fps: {fps} (frame_time={frame_time:.4f}s)")

    # Observation window: last `obs` frames ending at t=0
    obs_temporal_window = [-i * frame_time for i in range(obs)][::-1]
    # Action window: `horizon` steps starting at t=0
    action_temporal_window = [i * frame_time for i in range(horizon)]

    delta_timestamps = {
        "observation.state": obs_temporal_window,
        "action": action_temporal_window,
        # Cameras only need the current frame — the model always uses imgs[:, -1].
        **{key: [0.0] for key in camera_keys},
    }

    # `tolerance_s` must accommodate the dataset's frame interval — too tight
    # and every delta lookup raises. Half a frame is a safe upper bound.
    tolerance_s = max(0.005, frame_time / 2)

    # Build each dataset, concatenate, and accumulate episode boundaries in the
    # concatenated index space (optionally filtered per-dataset by --max_episode_index).
    # One resize for every dataset, or none at all. Applying it to only the
    # odd one out would leave two different preprocessing paths feeding one
    # model, and the difference would be invisible in the loss.
    all_shapes = {sh for d in vis_shapes.values() for sh in d.values()}
    resize_to = None
    if len(all_shapes) > 1:
        non_square = [sh for sh in all_shapes if sh[-1] != sh[-2]]
        if non_square:
            raise ValueError(
                f"cameras differ in resolution AND some are not square "
                f"({sorted(all_shapes)}). The model pads non-square frames to "
                f"square before resizing, so resizing them here would change "
                f"the aspect handling. Convert them to a common size first.")
        resize_to = int(cfg.vision_input_size)
        print(f"\nCamera resolutions differ across datasets: "
              f"{ {d: sorted(set(v.values())) for d, v in vis_shapes.items()} }\n"
              f"  -> resizing every frame to {resize_to}x{resize_to} at LOAD "
              f"time. ConcatDataset stacks raw tensors, so mixed resolutions "
              f"would fail in collate; the model resizes to vision_input_size "
              f"anyway, so doing it here changes nothing but the timing.")
    if load_image_size:
        # The model pads to square and interpolates every frame to
        # vision_input_size anyway, so doing it in the workers costs nothing at
        # the model and cuts what the loader must hold, pin and copy by
        # (size/native)^2. A 480 source at 384 is 0.64x. This is the difference
        # between a batch of 64 costing 354 MB and 226 MB, per in-flight batch,
        # per worker.
        #
        # Opt-in rather than automatic: v2.Resize antialiases and the model's
        # F.interpolate does not, so enabling it changes the pixels slightly
        # and a run started with it is not bit-comparable to one without.
        if resize_to and resize_to != int(load_image_size):
            print(f"  --load_image_size {load_image_size} overrides the "
                  f"{resize_to} chosen to reconcile mixed resolutions.")
        resize_to = int(load_image_size)
    img_tf = v2.Resize((resize_to, resize_to), antialias=True) if resize_to else None

    sub_datasets = []
    ep_from: list[int] = []
    ep_to: list[int] = []
    ep_ds: list[str] = []          # which dataset each episode came from
    offset = 0
    first_root = None
    for did in dataset_ids:
        _t = time.time()
        print(f"[data] opening {did}"
              + ("  (syncing cache; first pull can take a long time and is "
                 "silent unless --download_progress)" if cache_sync else ""),
              flush=True)
        ds = LeRobotDataset(
            did, delta_timestamps=delta_timestamps,
            force_cache_sync=cache_sync, revision="main", tolerance_s=tolerance_s,
            image_transforms=img_tf,
        )
        print(f"[data]   ...{time.time() - _t:.0f}s", flush=True)
        if first_root is None:
            first_root = ds.root
        # Episode spans must tile the table. delta_timestamps make
        # _get_query_indices clamp every lookup to [dataset_from_index,
        # dataset_to_index - 1], and it is the ONLY consumer of those columns:
        # a set whose offsets are wrong loads, indexes and prints correctly,
        # then raises from inside a DataLoader worker on the first batch --
        #   IndexError: Invalid key: 863104 is out of bounds for size 575101
        # Published VLABench ships dataset_from_index = length * episode_index
        # instead of a running sum, so this is not hypothetical.
        E = ds.meta.episodes
        fr = np.asarray(E["dataset_from_index"], dtype=np.int64)
        to = np.asarray(E["dataset_to_index"], dtype=np.int64)
        o = np.argsort(np.asarray(E["episode_index"], dtype=np.int64))
        fr, to = fr[o], to[o]
        n_rows_ds = len(ds.hf_dataset)
        if fr[0] != 0 or to[-1] != n_rows_ds or not (to[:-1] == fr[1:]).all():
            raise ValueError(
                f"Dataset '{did}': meta/episodes row offsets do not tile the "
                f"table.\n"
                f"  spans cover [{fr[0]}, {to[-1]}), table has {n_rows_ds} rows; "
                f"{int((to[:-1] != fr[1:]).sum())} of {len(fr) - 1} boundaries "
                f"gap or overlap.\n"
                f"  Training would fail on its first batch inside a DataLoader "
                f"worker with an out-of-bounds IndexError, which does not name "
                f"this as the cause.\n"
                f"  For VLABench, src/convert_vlabench_to_libero.py rebuilds "
                f"these from the data.")
        ep_ids = np.array(ds.hf_dataset["episode_index"])
        changes = np.where(np.diff(ep_ids) != 0)[0] + 1
        starts = np.concatenate([[0], changes])
        ends = np.concatenate([changes, [len(ep_ids)]])
        kept = 0
        for s, e in zip(starts, ends):
            if max_episode_index is not None and int(ep_ids[s]) > max_episode_index:
                continue
            ep_from.append(offset + int(s))
            ep_to.append(offset + int(e))
            ep_ds.append(did)
            kept += 1
        suffix = f" (<= ep {max_episode_index})" if max_episode_index is not None else ""
        print(f"  {did}: {len(ds)} frames, {kept} episodes{suffix}")
        sub_datasets.append(ds)
        offset += len(ds)

    dataset = ConcatDataset(sub_datasets)
    print(f"Combined dataset: {len(dataset)} frames, {len(ep_from)} episodes "
          f"across {len(sub_datasets)} dataset(s)")

    # Build task_index → description mapping from the first dataset's tasks.parquet.
    # Batches carry the per-frame "task" string directly (preferred by the loop);
    # for multi-dataset, task_index is dataset-local so we rely on batch["task"].
    task_idx_to_description: dict[int, str] = {}
    try:
        tasks_parquet_path = first_root / "meta" / "tasks.parquet"
        if tasks_parquet_path.exists():
            tasks_df = pd.read_parquet(tasks_parquet_path)
            if "task_index" in tasks_df.columns:
                task_idx_to_description = {
                    int(row["task_index"]): str(idx)
                    for idx, row in tasks_df.iterrows()
                }
            print(f"Loaded {len(task_idx_to_description)} task descriptions from tasks.parquet:")
            for idx, desc in task_idx_to_description.items():
                print(f"  [{idx}] {desc}")
        else:
            print("tasks.parquet not found; task_description will not be added to batches.")
    except Exception as e:
        print(f"Warning: could not load tasks.parquet: {e}")

    if combined_stats and "observation.state" in combined_stats:
        s = combined_stats["observation.state"]
        print(f"\nNorm stats observation.state:")
        print(f"  mean={s.get('mean', 'N/A')}")
        print(f"  std ={s.get('std',  'N/A')}")
    else:
        print("WARNING: observation.state not found in combined_stats — will not be normalized!")
    if combined_stats and "action" in combined_stats:
        s = combined_stats["action"]
        print(f"Norm stats action:")
        print(f"  mean={s.get('mean', 'N/A')}")
        print(f"  std ={s.get('std',  'N/A')}")
    # ---- held-out split, by EPISODE ----
    # Splitting by frame would put frames from one episode on both sides: the
    # neighbouring frame is nearly the same image with nearly the same action,
    # so a held-out loss built that way measures interpolation and reports a
    # gap of roughly zero no matter how badly the model has memorised.
    val_ep_idx: list = []
    if val_episodes > 0:
        rng = np.random.default_rng(seed=42)
        by_ds: dict = {}
        for i, d in enumerate(ep_ds):
            by_ds.setdefault(d, []).append(i)
        # Proportional per dataset, so a mixed run does not hold out only the
        # big one and then report a number about the wrong domain.
        total = len(ep_ds)
        for d, idxs in by_ds.items():
            k = max(1, round(val_episodes * len(idxs) / total))
            k = min(k, max(0, len(idxs) - 1))
            if k:
                val_ep_idx += list(rng.choice(idxs, size=k, replace=False))
    val_set = set(val_ep_idx)
    tr_idx = [i for i in range(len(ep_ds)) if i not in val_set]

    # The `fit` slice must be drawn the SAME way the held-out set was, from the
    # same rng, or the two columns are not measuring the same thing. Taking
    # tr_idx[:n] instead looks harmless and is not: lerobot/libero is ordered by
    # SUITE, so the first n training episodes are all one suite while the
    # held-out set spans all forty tasks. The difference then reports
    # "suite A vs everything" on top of "trained vs held out", and on a real run
    # that produced a gap of +400% while held-out itself was flat.
    fit_ep_idx: list = []
    if val_ep_idx:
        by_ds_tr: dict = {}
        for i in tr_idx:
            by_ds_tr.setdefault(ep_ds[i], []).append(i)
        want: dict = {}
        for i in val_ep_idx:
            want[ep_ds[i]] = want.get(ep_ds[i], 0) + 1
        for d, k in want.items():
            pool = by_ds_tr.get(d, [])
            if pool:
                fit_ep_idx += list(rng.choice(pool, size=min(k, len(pool)),
                                              replace=False))

    def mk_sampler(idxs, shuffle):
        return EpisodeAwareSampler(
            dataset_from_indices=[ep_from[i] for i in idxs],
            dataset_to_indices=[ep_to[i] for i in idxs],
            drop_n_first_frames=0, drop_n_last_frames=0, shuffle=shuffle)

    def mk_loader(idxs, shuffle, workers):
        kw = {} if workers == 0 else {"prefetch_factor": max(1, int(prefetch_factor))}
        return torch.utils.data.DataLoader(
            dataset, num_workers=workers, batch_size=batch_size,
            sampler=mk_sampler(idxs, shuffle),
            pin_memory=device.type != "cpu", drop_last=True, **kw)

    sampler = mk_sampler(tr_idx, True)
    print(f"EpisodeAwareSampler: {len(sampler)} frames "
          f"over {len(tr_idx)} episodes")
    dataloader = mk_loader(tr_idx, True, num_workers)
    # Each worker forks a copy of the dataset, and Python refcounting
    # gradually un-shares the copy-on-write pages holding the Arrow table --
    # so host RAM climbs for hundreds of steps and then the runtime SIGINTs
    # the process. It reads as "^C" with no traceback, which looks nothing
    # like a memory error. _rss_gb below is there to make it visible.
    # Predict the in-flight cost instead of discovering it as a SIGKILL a few
    # hundred steps in. A killed worker surfaces as
    #   RuntimeError: DataLoader worker (pid ...) is killed by signal: Killed
    # raised from wherever the main process happened to be -- the traceback
    # points at the model, never at the loader.
    px = resize_to if resize_to else max(
        (sh[-1] for d in vis_shapes.values() for sh in d.values()), default=0)
    per_sample_mb = len(camera_keys) * 3 * px * px * 4 / 1048576.0
    inflight = per_sample_mb * batch_size * max(1, num_workers) * \
        max(1, int(prefetch_factor))
    print(f"DataLoader: {num_workers} worker(s), prefetch "
          f"{prefetch_factor}, batch {batch_size}, "
          f"{len(camera_keys)} cam @ {px or '?'}px\n"
          f"  ~{per_sample_mb:.1f} MB/sample -> ~{inflight / 1024:.1f} GB of "
          f"decoded frames in flight, plus an equal pinned copy"
          + ("   <-- large; lower --batch_size / --num_workers / "
             "--prefetch_factor, or pass --load_image_size 384"
             if inflight / 1024 > 3 else ""))

    val_loader = fit_loader = None
    if val_ep_idx:
        # shuffle=True on BOTH. val_max_batches x batch_size is far short of the
        # held-out set -- 20 x 60 = 1200 frames against ~6500 -- so an unshuffled
        # sampler scored the same arbitrary first ~7 episodes every pass and
        # called it the held-out loss. Whether those seven happened to be easy or
        # hard then set the absolute level for the whole run, which is how three
        # runs of the same model family came back with held-out at 0.25, 1.02 and
        # 1.30 while all three scored ~68% on the same held-out init states.
        # Shuffling spreads the same budget over all 40 episodes; run_eval_loss
        # pins the RNG, so successive passes still draw the SAME spread.
        val_loader = mk_loader(val_ep_idx, True, min(2, num_workers))
        # A same-sized slice of TRAINING episodes, scored the SAME way (eval
        # mode, no augmentation). Without it the only comparison available is
        # held-out against the running train loss, and those two differ by
        # dropout, image/state augmentation, paraphrase sampling AND the
        # contrastive term -- which is train-only here -- so their difference
        # is not a generalisation gap. fit vs held-out is.
        fit_loader = mk_loader(fit_ep_idx, True, min(2, num_workers))
        n_val_frames = sum(ep_to[i] - ep_from[i] for i in val_ep_idx)
        per_ds = {}
        for i in val_ep_idx:
            per_ds[ep_ds[i]] = per_ds.get(ep_ds[i], 0) + 1
        print(f"Validation: {len(val_ep_idx)} episodes held out "
              f"({n_val_frames} frames, {100 * n_val_frames / max(1, sum(ep_to[i] - ep_from[i] for i in range(len(ep_ds)))):.1f}%), "
              f"every {val_every} steps, <= {val_max_batches} batches\n"
              f"  per dataset: {per_ds}\n"
              f"  each pass scores {val_max_batches * batch_size} frames of "
              f"{n_val_frames} ({100 * val_max_batches * batch_size / max(1, n_val_frames):.0f}%), "
              f"sampled across all {len(val_ep_idx)} episodes"
              + ("  -- raise --val_max_batches for a steadier number"
                 if val_max_batches * batch_size < 0.5 * n_val_frames else ""))
    else:
        print("Validation: DISABLED (--val_episodes 0). Training loss alone "
              "cannot tell fitting from memorising.")

    @torch.no_grad()
    def run_eval_loss(loader):
        """Mean loss in EVAL mode with no augmentation.

        policy.eval() is what turns paraphrase sampling off (the model gates it
        on self.training), and it is also what drops the contrastive term, so
        both columns below are the same quantity measured on different
        episodes."""
        was_training = policy.training
        policy.eval()
        # Same t and same noise on every pass, and on BOTH loaders.
        # compute_loss draws a fresh flow timestep per sample and a fresh
        # source noise; without pinning them, two consecutive validations
        # differ by the draw as much as by the model. Measured on a real run:
        # the fit/held-out gap swung 8.3 -> 7.5 -> 22.2 -> 12.9 -> 21.8 -> 12.9
        # across adjacent passes while held-out itself moved by under 0.01.
        # Pinning also means `fit` and `held-out` are scored at the SAME
        # timesteps, so their difference is about the episodes and nothing else.
        cpu_state = torch.get_rng_state()
        cuda_state = (torch.cuda.get_rng_state_all()
                      if torch.cuda.is_available() else None)
        torch.manual_seed(20260829)
        tot, n = 0.0, 0
        # Magnitude probe. The flow loss says how far the predicted velocity
        # field is from the demo actions; it does not say in WHICH direction,
        # so a policy that has drifted toward larger, faster actions and one
        # that has simply got worse produce the same rising number. Sampling a
        # chunk on the first batch and comparing |a| against the target's
        # separates them: drift shows as a ratio moving away from 1.0 while the
        # loss rises, degradation shows as the loss rising with the ratio flat.
        amp_pred = amp_tgt = 0.0
        for i, b in enumerate(loader):
            if i >= val_max_batches:
                break
            for k in b:
                if isinstance(b[k], torch.Tensor):
                    b[k] = b[k].to(device, non_blocking=True)
            if "task" in b and isinstance(b["task"], (list, tuple)):
                b["task_description"] = b["task"]
            b = preprocessor(b)
            with (torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                  if device.type == "cuda"
                  else torch.autocast(device_type="cpu", enabled=False)):
                loss, _ = policy.forward(b)
            tot += float(loss.detach()); n += 1
            if i == 0 and hasattr(policy, "predict_action_chunk"):
                try:
                    policy.reset()
                    pred = policy.predict_action_chunk(b).float()
                    tgt = b["action"].float()
                    h = min(pred.shape[1], tgt.shape[1])
                    m = tgt.new_ones(tgt.shape[:2])
                    pad = b.get("action_is_pad")
                    if pad is not None:
                        m = (~pad.bool()).to(tgt.dtype)
                    m = m[:, :h].unsqueeze(-1)
                    d = tgt.shape[-1] - 1        # drop the gripper channel
                    amp_pred = float((pred[:, :h, :d].abs() * m).sum()
                                     / m.sum().clamp(min=1) / d)
                    amp_tgt = float((tgt[:, :h, :d].abs() * m).sum()
                                    / m.sum().clamp(min=1) / d)
                except Exception:
                    amp_pred = amp_tgt = 0.0
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        if was_training:
            policy.train()
        return tot / max(1, n), amp_pred, amp_tgt

    # Training loop
    print("Starting training loop...")
    done = False
    prog_bar = tqdm(total=training_steps, desc="Training Progress", initial=step)
    while not done:
        epoch += 1
        for batch in dataloader:
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)

            # Enrich batch with task description strings
            if "task" in batch and isinstance(batch["task"], (list, tuple)):
                batch["task_description"] = batch["task"]
            elif task_idx_to_description and "task_index" in batch:
                task_indices = batch["task_index"]
                if isinstance(task_indices, torch.Tensor) and task_indices.dim() > 1:
                    task_indices = task_indices[:, 0]
                batch["task_description"] = [task_idx_to_description.get(int(ti), "") for ti in task_indices]

            # Apply instruction rewriting if enabled (for LIBERO spatial grounding)
            if rewrite_instructions and "task_description" in batch:
                batch["task_description"] = [
                    rewrite_instruction(t, random_augment=rewrite_augment)
                    for t in batch["task_description"]
                ]

            batch = apply_image_augmentations(batch, camera_keys, image_transforms)
            batch = apply_joint_augmentations(batch)

            if step == 0:
                raw_st = batch["observation.state"].float()
                print(f"\nRaw (pre-norm) observation.state: min={raw_st.min():.4f}  max={raw_st.max():.4f}  std={raw_st.std():.4f}")

            batch = preprocessor(batch)

            if step == 0:
                pad_key = next((k for k in ("action_is_pad", "actions_id_pad") if k in batch), None)
                if pad_key is None:
                    print("WARNING: no action pad key found in batch — padded episode steps will pollute loss!")
                    print(f"  Available keys: {[k for k in batch.keys() if 'pad' in k.lower() or 'action' in k.lower()]}")
                else:
                    pad_frac = batch[pad_key].float().mean().item()
                    print(f"Action pad key='{pad_key}', pad fraction in first batch: {pad_frac:.2%}")

            # Forward & Backward
            # Arm the attention-mass diagnostic on the same cadence as
            # gradient analysis. The model self-disarms after one capture.
            if step % progress_update_freq == 0:
                policy.model._capture_attention_stats = True

            autocast_ctx = (
                torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                if device.type == "cuda"
                else torch.autocast(device_type="cpu", enabled=False)
            )
            with autocast_ctx:
                loss, _ = policy.forward(batch)

            if loss.item() > 100 and step < 2000:
                act = batch["action"].float()
                st = batch["observation.state"].float()
                print(f"\n[DIAG step={step}] loss={loss.item():.1f}")
                print(f"  action  : min={act.min():.2f}  max={act.max():.2f}  std={act.std():.3f}")
                print(f"  state   : min={st.min():.2f}  max={st.max():.2f}  std={st.std():.3f}")
                pad_key = next((k for k in ("action_is_pad", "actions_id_pad") if k in batch), None)
                if pad_key is not None:
                    print(f"  pad frac: {batch[pad_key].float().mean().item():.2%}")

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
                    "grad_norm": f"{grad_norm:.2f}"
                })

            if val_loader is not None and step > 0 and step % val_every == 0:
                v, ap, at = run_eval_loss(val_loader)
                f, _, _ = run_eval_loss(fit_loader)
                gap = 100.0 * (v - f) / max(abs(f), 1e-9)
                ratio = ap / at if at > 0 else float("nan")
                print(f"\n  VAL @ {step}   "
                      f"fit(train eps) {f:.4f}   held-out {v:.4f}   "
                      f"gap {gap:+.1f}%"
                      f"   [both eval mode, no augmentation]\n"
                      f"      |action| predicted {ap:.4f} vs target {at:.4f}"
                      f"   ratio {ratio:.3f}"
                      f"   (>1 = the policy is taking BIGGER steps than the demos)")
                with open(output_directory / "val_log.jsonl", "a") as fh:
                    fh.write(json.dumps({"step": step, "fit": f, "heldout": v,
                                         "gap_pct": gap, "amp_pred": ap,
                                         "amp_target": at, "amp_ratio": ratio}) + "\n")

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

    # Final save
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
    parser.add_argument("--dataset_id", type=str, nargs="+", default=["ISdept/piper_arm"],
                        help="One or more LeRobot dataset ids. Multiple are concatenated and "
                             "must share a homogeneous schema (same robot/cameras/dims/fps); "
                             "their normalization stats are aggregated.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Recompute DiT activations in backward to save memory.")
    parser.add_argument("--max_episode_index", type=int, default=None,
                        help="Filter to episodes with index <= this value "
                             "(piper_arm holdout convention; omit for full dataset).")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="DataLoader batch size (default: 64).")
    parser.add_argument("--n_obs_steps", type=int, default=None,
                        help="Frames of observation.state FETCHED per sample "
                             "(default: 2). Note that wilro then keeps only the "
                             "last one: wilro_model slices state_tok[:, -1:], so "
                             "anything above 1 is fetched, encoded and thrown "
                             "away. It changes what the dataloader carries, not "
                             "what the model sees -- 1 is strictly cheaper and "
                             "numerically identical. The model has NO velocity "
                             "input either way.")
    parser.add_argument("--val_episodes", type=int, default=0,
                        help="Hold out this many EPISODES for validation, "
                             "allocated proportionally across --dataset_id so a "
                             "mixed run does not measure only the larger set. "
                             "0 disables. Splitting by frame instead would put "
                             "neighbouring frames of one episode on both sides "
                             "and report a gap near zero however badly the "
                             "model memorised.")
    parser.add_argument("--load_image_size", type=int, default=0,
                        help="Resize frames to NxN in the DATALOADER (0 = keep "
                             "native, the default). The model pads to square "
                             "and interpolates to vision_input_size anyway, so "
                             "this costs nothing there and cuts what every "
                             "worker must decode, hold, pin and copy by "
                             "(N/native)^2 -- 480 -> 384 is 0.64x. Opt-in "
                             "because v2.Resize antialiases and the model's "
                             "interpolate does not, so a run using it is not "
                             "bit-comparable to one without.")
    parser.add_argument("--prefetch_factor", type=int, default=2,
                        help="Batches each worker keeps queued (default: 2). "
                             "Total decoded frames held is batch_size x "
                             "num_workers x this; 1 halves it.")
    parser.add_argument("--download_progress", action="store_true",
                        help="Keep huggingface_hub's per-file progress bars. "
                             "Off by default: at ~14k files they redraw "
                             "thousands of lines and bury the schema and "
                             "normalisation output printed before training. "
                             "Turn on for a genuinely first-time pull.")
    parser.add_argument("--cache_sync", action="store_true",
                        help="Re-verify every file of every dataset against the "
                             "hub at launch (the old always-on behaviour). "
                             "~14k requests for a converted VLABench before the "
                             "first step; only needed when the remote may have "
                             "changed under an existing cache.")
    parser.add_argument("--lora_rank", type=int, default=None,
                        help="LoRA rank on the SigLIP ViT (default: 16). The "
                             "vision adapter is ~0.1%% of this model's trainable "
                             "parameters -- the DiT is 99.3%% -- so this is not "
                             "a lever on OVERALL capacity, and the model already "
                             "overfits. It is a lever on where adaptation is "
                             "ALLOWED: grounding lives in the encoder, and the "
                             "encoder is where almost nothing trains.")
    parser.add_argument("--lora_alpha", type=float, default=None,
                        help="LoRA alpha (default: 2 x rank). LoRALinear scales "
                             "by alpha/rank, so leaving alpha fixed while "
                             "raising rank halves the adapter's strength. The "
                             "default tracks rank to keep that ratio at the "
                             "shipped 32/16 = 2.0.")
    parser.add_argument("--vision_lora_num_layers", type=int, default=None,
                        help="How many trailing SigLIP ViT layers get adapters "
                             "(default: 8; SmolVLM2-500M's ViT has 27). Text "
                             "LoRA stays at 0 and should: the encoder-decoder "
                             "detaches the VLM KV cache, so no gradient reaches "
                             "the text tower to train an adapter with.")
    parser.add_argument("--lr", type=float, default=None,
                        help="Peak learning rate (default: the config's 1e-4). "
                             "The cosine is built around this, and the resume "
                             "path rebuilds param_groups from it, so an "
                             "explicit value survives --resume_from_checkpoint "
                             "-- unlike the sibling trainer before 4caed2d. "
                             "Refining an already-trained policy on collected "
                             "rollouts wants ~1e-5; train_rft.py's own default "
                             "is 1e-5.")
    parser.add_argument("--warmup_steps", type=int, default=None,
                        help="Linear warmup steps before the cosine (default: "
                             "1500). Lower it for short refinement runs, where "
                             "1500 can be most of the budget.")
    parser.add_argument("--start_step_override", type=int, default=-1,
                        help="Restart the step counter at this value when "
                             "loading a checkpoint (-1 = keep the checkpoint's, "
                             "the default). Pass 0 to fine-tune on a DIFFERENT "
                             "dataset: without it the checkpoint's step is "
                             "inherited, the cosine is fast-forwarded to it, "
                             "and the run starts mid-decay with no warmup and "
                             "a step budget short by however far the previous "
                             "run got.")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader worker processes (default: 8). Each "
                             "forks a copy of the dataset; Python refcounting "
                             "then un-shares the copy-on-write pages holding "
                             "the Arrow table, so host RAM climbs for hundreds "
                             "of steps until the runtime SIGINTs the process -- "
                             "which prints a bare ^C and no traceback. On Colab "
                             "with a 575k-row set, 2-4 is the safe range.")
    parser.add_argument("--progress_update_freq", type=int, default=200,
                        help="Steps between the gradient/attention diagnostic "
                             "and the progress-bar refresh (default: 200). The "
                             "attention capture re-runs the last DiT layer's "
                             "softmax at (B, H, L, L) in fp32 under no_grad -- "
                             "transient, but the largest single allocation in "
                             "the step. Raise it to move that cost, which is "
                             "also how to test whether it is what is killing a "
                             "run at a multiple of this number.")
    parser.add_argument("--val_every", type=int, default=500,
                        help="Steps between validation passes (default: 500).")
    parser.add_argument("--val_max_batches", type=int, default=20,
                        help="Batches per validation pass (default: 20).")
    parser.add_argument("--training_steps", type=int, default=None,
                        help="Total optimizer steps (default: 200000). The "
                             "cosine LR schedule spans this, so it is not a "
                             "stop-whenever ceiling -- interrupting a 200k run "
                             "early leaves the LR mid-cosine and the model "
                             "never annealed. On resume an explicit value "
                             "overrides the checkpoint's and rebuilds the "
                             "schedule.")
    parser.add_argument("--contrastive_loss_weight", type=float, default=0.1,
                        help="Weight for the language-permute contrastive loss "
                             "(default: 0.1). Bump to ~0.5 for LIBERO / datasets "
                             "with diverse task descriptions.")
    parser.add_argument("--contrastive_margin", type=float, default=0.05,
                        help="Hinge margin on MSE between v_t and v_wrong "
                             "(default: 0.05). Bump to ~0.2 to force the model "
                             "to differentiate velocities by language.")
    parser.add_argument("--contrastive_hard_negatives", action="store_true",
                        help="Pair each sample with its hardest in-batch negative (most word "
                             "overlap, different instruction) instead of a random one, so the "
                             "contrastive hinge pressures fine-grained object grounding (the "
                             "confusable minimal pairs that fail at eval) rather than trivially-"
                             "different tasks. Expect the reported contrastive value to spike "
                             "when first enabled, then decline. Off = legacy random pairing.")
    parser.add_argument("--paraphrase_augment", action="store_true",
                        help="Draw a different phrasing of the same instruction "
                             "per sample per step, so the surface string stops "
                             "being a usable key. Measured on the sibling "
                             "(wiltechs-x-114k, libero_spatial T7): 60%% on its "
                             "own instruction, 0%% on a PARAPHRASE of that same "
                             "instruction -- it had memorised the ~40 strings "
                             "and was retrieving, not reading. Table in "
                             "src/libero_paraphrase.py; the original string is "
                             "always among the variants, since eval uses it.")
    parser.add_argument("--paraphrase_limit", type=int, default=8,
                        help="Cap on variants per instruction (0 = all).")
    parser.add_argument("--paraphrase_file", default="",
                        help="JSON table overriding the built-in one, for "
                             "instructions it does not cover. Draft it with "
                             "python -m libero_paraphrase --dataset_id <id> "
                             "--out f.json, then hand-edit -- templates never "
                             "reach the model unread.")
    parser.add_argument("--paraphrase_min_variants", type=int, default=5,
                        help="Refuse to start when any instruction has fewer "
                             "variants than this. Partial augmentation is worse "
                             "than none: the unvaried tasks keep surface form as "
                             "a key and the run answers nothing.")
    parser.add_argument("--lock_joint_index", type=int, default=3,
                        help="Action dim with weight 0 (piper_arm joint 4 = "
                             "index 3 is mechanically locked). Pass -1 to "
                             "disable for LIBERO / other full-DOF robots.")
    parser.add_argument("--kv_capture_strategy", type=str, default="last",
                        choices=["last", "stride2", "custom"],
                        help="Which VLM layers the DiT sources KV from. "
                             "'last' = trailing N layers (most refined, no "
                             "multi-scale). 'stride2' = every other layer, "
                             "end-anchored (multi-scale: shallow DiT reads "
                             "shallow VLM). 'custom' = exactly the layers given "
                             "in --kv_capture_layers (DiT depth = #layers). "
                             "NOT resume-compatible across values.")
    parser.add_argument("--kv_capture_layers", type=str, default="",
                        help="Comma-separated 0-based VLM layer indices for "
                             "--kv_capture_strategy custom, e.g. '3,7,11,15,19,"
                             "23,27,31'. Ignored for last/stride2.")
    parser.add_argument("--cameras", type=str, nargs="+", default=None,
                        help="Subset of cameras to use from the dataset. If not specified, "
                             "all available cameras are used. Example: "
                             "--cameras observation.images.chest observation.images.left_hand")
    parser.add_argument("--noise_temporal_correlation", type=float, default=0.0,
                        help="AR(1) coefficient correlating the flow-matching source "
                             "noise along the action horizon (0=white noise; ~0.9=temporally "
                             "smooth). Source dist changes, so this is NOT inference-only — "
                             "resume from a rho=0 checkpoint and fine-tune to adapt. Too high "
                             "(>0.95) over-smooths sharp/contact motions.")
    parser.add_argument("--gripper_phase_weight", type=float, default=1.0,
                        help="Up-weight the flow-matching loss on frames near a gripper "
                             "open<->close transition (grasp/release) — the precision-critical "
                             "moments uniform MSE dilutes. 1.0=off (default); try 2-4 to sharpen "
                             "placement. Gripper assumed to be the last action dim (LIBERO OSC).")
    parser.add_argument("--time_sampling", type=str, default="uniform",
                        choices=["uniform", "lognormal"],
                        help="Flow-matching timestep sampling. 'lognormal' (SD3 logit-normal) "
                             "biases toward low t (x_t≈actions), spending more capacity on the "
                             "fine-detail denoising that sets placement precision.")
    parser.add_argument("--time_lognormal_mean", type=float, default=-0.5,
                        help="Mean of the logit-normal (only if --time_sampling lognormal). "
                             "More negative => more mass at low t (finer detail).")
    parser.add_argument("--time_lognormal_std", type=float, default=1.0,
                        help="Std of the logit-normal (only if --time_sampling lognormal).")
    parser.add_argument("--rewrite_instructions", action="store_true",
                        help="Apply instruction rewriting from task_rewrites.py for LIBERO "
                             "spatial grounding (e.g., ramekin -> visual description, "
                             "'between' -> 'closer to'). Rewritten instructions are used "
                             "for both VLM encoding and contrastive hard negatives.")
    parser.add_argument("--rewrite_augment", action="store_true",
                        help="When --rewrite_instructions is enabled, randomly choose between "
                             "original and rewritten instruction (50/50) for each sample. "
                             "This trains the model to understand BOTH phrasings.")
    args = parser.parse_args()
    # Argparse can't express None for an int, so use -1 sentinel.
    if args.lock_joint_index is not None and args.lock_joint_index < 0:
        args.lock_joint_index = None
    # Parse the comma-separated custom layer list into ints.
    args.kv_capture_layers = [
        int(tok) for tok in args.kv_capture_layers.split(",") if tok.strip() != ""
    ]
    if args.kv_capture_strategy == "custom" and not args.kv_capture_layers:
        parser.error("--kv_capture_strategy custom requires --kv_capture_layers")
    train(**vars(args))
