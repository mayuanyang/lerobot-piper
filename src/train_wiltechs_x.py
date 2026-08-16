"""Stage-A SFT trainer for WiltechsX.

    python src/train_wiltechs_x.py \
        --dataset_ids physical-intelligence/libero \
        --output_dir ./outputs/wx_a --training_steps 60000 --batch_size 8

WHAT TO WATCH, and it is not the average loss:

  * `min` in the per-task success report at eval time. Stage B (RL) recovers a
    task sitting at 10%; it can do nothing with one sitting at 0, because a
    binary reward gives no gradient where every rollout fails. 93% average with
    a floor of 15% is a better stage-A checkpoint than 95% with two zeros.
  * `discrete` in the loss breakdown. If it does not fall, the VLM is not
    learning the task and knowledge insulation is costing you the backbone for
    nothing -- run `--no_discrete_head` and compare.
  * `shortcut`. If it stays high, few-step inference is not valid and
    `--num_inference_steps 4` is silently under-integrating.

This trainer deliberately does NOT do image preprocessing in DataLoader
workers (the model's `_encode_images` fallback runs the Qwen processor inline).
That is slower per step and simpler to get right; port the worker path from
train_wiltechs_vla.py if throughput becomes the binding constraint.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.utils.utils import init_logging

from models.wiltechs_x.processor_wiltechs_x import make_pre_post_processors
from models.wiltechs_x.wiltechs_x_config import WiltechsXConfig
from models.wiltechs_x.wiltechs_x_policy import WiltechsXPolicy

try:
    from lerobot.datasets.utils import aggregate_stats
except ImportError:                                            # older lerobot
    aggregate_stats = None


WRIST_HINTS = ("image2", "wrist", "gripper", "eye_in_hand", "hand")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


class ProgressDataset(torch.utils.data.Dataset):
    """Adds a `progress` scalar (normalized time-to-completion) per frame.

    The progress head needs a target and LeRobot exposes frame_index but not
    episode length, so it is computed once here from the episode boundaries.
    Without this the model SKIPS the progress term with a warning rather than
    training it against zeros.
    """

    def __init__(self, base, ep_from, ep_to):
        self.base = base
        prog = np.zeros(len(base), dtype=np.float32)
        for s, e in zip(ep_from, ep_to):
            n = max(e - s - 1, 1)
            prog[s:e] = np.arange(e - s, dtype=np.float32) / n
        self.prog = prog

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        item["progress"] = torch.tensor(self.prog[i], dtype=torch.float32)
        return item


def build_datasets(dataset_ids, obs_steps, horizon, max_episode_index):
    metas = {d: LeRobotDatasetMetadata(d, force_cache_sync=True, revision="main")
             for d in dataset_ids}
    ref = metas[dataset_ids[0]]
    feats = dataset_to_policy_features(ref.features)
    out_f = {k: v for k, v in feats.items() if v.type is FeatureType.ACTION}
    in_f = {k: v for k, v in feats.items() if k not in out_f}
    if not out_f:
        raise ValueError("no action features in the dataset schema")

    cameras = sorted(k for k, v in in_f.items() if v.type is FeatureType.VISUAL)
    state_dim = in_f["observation.state"].shape[-1]
    action_dim = next(iter(out_f.values())).shape[-1]
    print(f"cameras={cameras}  state_dim={state_dim}  action_dim={action_dim}")

    for d in dataset_ids[1:]:
        f = dataset_to_policy_features(metas[d].features)
        o = {k: v for k, v in f.items() if v.type is FeatureType.ACTION}
        i = {k: v for k, v in f.items() if k not in o}
        if sorted(k for k, v in i.items() if v.type is FeatureType.VISUAL) != cameras \
           or i["observation.state"].shape[-1] != state_dim \
           or next(iter(o.values())).shape[-1] != action_dim:
            raise ValueError(f"dataset {d!r} schema differs from {dataset_ids[0]!r}")

    stats = ref.stats if len(dataset_ids) == 1 else (
        aggregate_stats([metas[d].stats for d in dataset_ids]) if aggregate_stats
        else ref.stats)

    fps = ref.fps
    ft = 1.0 / fps
    print(f"fps={fps}")
    delta = {
        "observation.state": [-i * ft for i in range(obs_steps)][::-1],
        "action": [i * ft for i in range(horizon)],
        **{c: [0.0] for c in cameras},
    }

    subs, ep_from, ep_to, offset = [], [], [], 0
    for d in dataset_ids:
        ds = LeRobotDataset(d, delta_timestamps=delta, force_cache_sync=True,
                            revision="main", tolerance_s=max(0.005, ft / 2))
        ep = np.array(ds.hf_dataset["episode_index"])
        cuts = np.where(np.diff(ep) != 0)[0] + 1
        starts = np.concatenate([[0], cuts])
        ends = np.concatenate([cuts, [len(ep)]])
        for s, e in zip(starts, ends):
            if max_episode_index is not None and int(ep[s]) > max_episode_index:
                continue
            ep_from.append(offset + int(s))
            ep_to.append(offset + int(e))
        subs.append(ds)
        offset += len(ds)

    base = subs[0] if len(subs) == 1 else torch.utils.data.ConcatDataset(subs)
    return {
        "dataset": base, "ep_from": ep_from, "ep_to": ep_to, "cameras": cameras,
        "state_dim": state_dim, "action_dim": action_dim, "stats": stats,
        "fps": fps, "input_features": in_f, "output_features": out_f,
    }


def report_memory_budget(model, counts, device) -> None:
    """Print the FIXED memory cost and compare it to the card, before step 1.

    Everything here is allocated regardless of batch size, so if it alone does
    not fit, no --batch_size will help. This exists because the first OOM
    landed inside `torch._foreach_sqrt` at opt.step() -- 40 minutes of dataset
    download and model load to learn something computable in a millisecond.
    """
    if device != "cuda":
        return
    w = sum(p.numel() * p.element_size() for p in model.parameters()) / 2 ** 30
    tr = counts["trainable"]
    grads = tr * 4 / 2 ** 30
    adam = tr * 8 / 2 ** 30
    fixed = w + grads + adam
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30

    groups = {"expert": model.expert_layers, "wrist": model.wrist_encoder,
              "motion": model.motion_encoder, "discrete": model.discrete_head}
    print("\nmemory budget (fixed, independent of batch size)")
    for name, mod in groups.items():
        n = sum(p.numel() for p in mod.parameters() if p.requires_grad) if mod else 0
        if n:
            print(f"  {name:9s} {n / 1e6:>7.1f}M params  ({100 * n / tr:.0f}% of trainable)")
    lora = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    if lora:
        print(f"  {'lora':9s} {lora / 1e6:>7.1f}M params  ({100 * lora / tr:.0f}% of trainable)")
    print(f"  weights {w:.2f} + grads {grads:.2f} + Adam {adam:.2f} = "
          f"{fixed:.2f} GiB of {total:.1f} GiB")
    print(f"  -> {total - fixed:.2f} GiB left for activations")

    if fixed > 0.80 * total:
        print(
            "\n  *** this will very likely OOM ***\n"
            "  The fixed cost alone is >80% of the card. Batch size will not\n"
            "  save it. In order of effect:\n"
            "    --expert_num_layers 12       expert params scale linearly in this\n"
            "    --expert_hidden_size 512     and roughly quadratically in this\n"
            "    --ada_rank 32                adaLN factorisation rank\n"
            "    --freeze_wrist_encoder       drops the DINO backward\n"
            "    --gradient_checkpointing     activations only, not the fixed cost\n")


def calibrate_gripper_threshold(stats, gripper_dim):
    """Threshold in NORMALIZED action units.

    The loss sees MEAN_STD-normalized actions; the stats are raw. The gripper
    column is bimodal with the median floating in the empty gap, so a guessed
    threshold can land inside a mode and mislabel a large fraction of frames.
    The q10/q90 midpoint sits in the gap by construction.
    """
    try:
        a = stats["action"]
        g = int(gripper_dim) % len(a["mean"])
        q10, q90 = float(a["q10"][g]), float(a["q90"][g])
        mu, sd = float(a["mean"][g]), float(a["std"][g])
        raw = 0.5 * (q10 + q90)
        norm = (raw - mu) / max(sd, 1e-8)
        print(f"gripper dim {g}: q10={q10:.4f} q90={q90:.4f} -> "
              f"{raw:.4f} raw / {norm:+.3f} normalized")
        return norm
    except (KeyError, IndexError, TypeError, ValueError) as e:
        print(f"gripper threshold NOT calibrated ({e}); the BCE term is DISABLED")
        return float("nan")


def train(
    dataset_ids: list[str],
    output_dir: str = "./outputs/wiltechs_x",
    vlm_model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
    # OPTIMIZER steps, and the cosine schedule is sized from it, so this is not
    # a "stop whenever" budget -- shortening it later changes the LR curve
    # rather than truncating it. 20000 is ~7 epochs on LIBERO's ~273k frames at
    # batch 12 x grad_accum 8 (2848 optimizer steps/epoch). The previous 60000
    # was 21 epochs and ~95 wall-clock hours at the 256-token wrist setting.
    training_steps: int = 20000,
    batch_size: int = 8,
    grad_accum: int = 1,
    lr: float = 1e-4,
    weight_decay: float = 1e-6,
    warmup_steps: int = 1000,
    horizon: int = 16,
    n_action_steps: int = 8,
    expert_hidden_size: int = 1024,
    expert_num_layers: int = 0,
    expert_intermediate_size: int = 0,
    ada_rank: int = 64,
    num_register_tokens: int = 8,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.0,
    lora_on_vision_tower: bool = False,
    freeze_vlm: bool = False,
    bidirectional_prefix: bool = True,
    knowledge_insulation: bool = True,
    discrete_head: bool = True,
    fast_token_loss_weight: float = 0.5,
    wrist_encoder: bool = True,
    wrist_encoder_id: str = "facebook/dinov2-small",
    wrist_cameras: list[str] | None = None,
    wrist_tokens: int = 256,
    wrist_input_size: int = 256,
    freeze_wrist_encoder: bool = False,
    motion_vectors: bool = True,
    motion_history_len: int = 8,
    motion_vector_tokens: int = 8,
    progress_head: bool = True,
    progress_loss_weight: float = 0.1,
    flow_objective: str = "shortcut",
    shortcut_consistency_frac: float = 0.25,
    num_inference_steps: int = 4,
    sample_noise_scale: float = 1.0,
    noise_temporal_correlation: float = 0.0,
    vision_input_size: int = 0,
    lang_max_len: int = 48,
    instruction_template: str = "",
    action_loss_weight: float = 1.0,
    loss_exec_steps: int = 0,
    future_steps_weight: float = 1.0,
    gripper_bce_weight: float = 0.05,
    gripper_action_dim: int = -1,
    gripper_bce_temp: float = 0.25,
    no_gripper_class_balance: bool = False,
    use_descriptive_objects: bool = False,
    num_workers: int = 4,
    save_every: int = 5000,
    log_every: int = 20,
    max_episode_index: int | None = None,
    gradient_checkpointing: bool = False,
    resume_from_checkpoint: str | None = None,
    seed: int = 42,
):
    init_logging()
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = pick_device()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  output={out}")

    # The state window must cover the motion history, or the encoder gets a
    # single frame left-padded into a constant and the term is a no-op.
    obs_steps = max(1, motion_history_len if motion_vectors else 1)

    D = build_datasets(dataset_ids, obs_steps, horizon, max_episode_index)
    cameras, stats = D["cameras"], D["stats"]
    dataset = (ProgressDataset(D["dataset"], D["ep_from"], D["ep_to"])
               if progress_head else D["dataset"])

    if wrist_cameras:
        missing = [c for c in wrist_cameras if c not in cameras]
        if missing:
            raise ValueError(f"--wrist_cameras {missing} not in {cameras}")
        wrist_keys = list(wrist_cameras)
    else:
        wrist_keys = [c for c in cameras
                      if any(h in c.rsplit(".", 1)[-1].lower() for h in WRIST_HINTS)]
        if wrist_encoder and not wrist_keys:
            raise ValueError(
                f"no wrist-like camera among {cameras}; pass --wrist_cameras "
                f"explicitly or --no_wrist_encoder. The wrist path is worth ~34 "
                f"points in this repo's own measurements — do not drop it silently.")
    print(f"wrist cameras: {wrist_keys}")

    gthr = calibrate_gripper_threshold(stats, gripper_action_dim)

    cfg = WiltechsXConfig(
        input_features=D["input_features"], output_features=D["output_features"],
        n_obs_steps=obs_steps, horizon=horizon, n_action_steps=n_action_steps,
        state_dim=D["state_dim"], action_dim=D["action_dim"],
        vlm_model_id=vlm_model_id, freeze_vlm=freeze_vlm,
        lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        lora_on_vision_tower=lora_on_vision_tower,
        bidirectional_prefix=bidirectional_prefix,
        num_cameras=len(cameras), cameras_for_vlm=cameras,
        vision_input_size=vision_input_size, lang_max_len=lang_max_len,
        instruction_template=instruction_template,
        use_descriptive_objects=use_descriptive_objects,
        knowledge_insulation=knowledge_insulation,
        fast_token_head=discrete_head,
        fast_token_loss_weight=fast_token_loss_weight,
        expert_hidden_size=expert_hidden_size,
        expert_intermediate_size=expert_intermediate_size,
        ada_rank=ada_rank,
        expert_num_layers=expert_num_layers,
        num_register_tokens=num_register_tokens,
        use_wrist_encoder=wrist_encoder, wrist_encoder_id=wrist_encoder_id,
        wrist_cameras=wrist_keys, wrist_tokens=wrist_tokens,
        wrist_input_size=wrist_input_size,
        freeze_wrist_encoder=freeze_wrist_encoder,
        use_motion_vectors=motion_vectors, motion_history_len=motion_history_len,
        motion_vector_tokens=motion_vector_tokens,
        progress_head=progress_head, progress_loss_weight=progress_loss_weight,
        flow_objective=flow_objective,
        shortcut_consistency_frac=shortcut_consistency_frac,
        num_inference_steps=num_inference_steps,
        sample_noise_scale=sample_noise_scale,
        noise_temporal_correlation=noise_temporal_correlation,
        action_loss_weight=action_loss_weight,
        loss_exec_steps=loss_exec_steps,
        future_steps_weight=future_steps_weight,
        gripper_bce_weight=gripper_bce_weight,
        gripper_action_dim=gripper_action_dim,
        gripper_bce_temp=gripper_bce_temp,
        gripper_class_balance=not no_gripper_class_balance,
        gripper_threshold_norm=gthr,
        optimizer_lr=lr, optimizer_weight_decay=weight_decay,
        scheduler_warmup_steps=warmup_steps,
        scheduler_decay_steps=training_steps,
        training_steps_total=training_steps,
        device=device,
    )
    cfg.validate_features()

    policy = WiltechsXPolicy(cfg).to(device)
    if gradient_checkpointing:
        policy.model.gradient_checkpointing_enable()

    start_step = 0
    if resume_from_checkpoint:
        ck = Path(resume_from_checkpoint)
        print(f"resuming from {ck}")
        sd = torch.load(ck / "training_state.pth", map_location=device)
        policy.load_state_dict(sd["model"])
        start_step = sd.get("step", 0)

    counts = policy.model.count_parameters()
    print(f"params: trainable={counts['trainable']:,}  frozen={counts['frozen']:,}")
    print(f"prefix gradient needed: {policy.model.needs_prefix_grad} "
          f"(False = the 36-layer prefix runs under no_grad, which is most of "
          f"the training memory)")
    report_memory_budget(policy.model, counts, device)
    if counts["trainable"] == 0:
        raise RuntimeError("nothing is trainable — check --freeze_vlm / LoRA targets")

    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=stats)

    params = [p for p in policy.parameters() if p.requires_grad]
    # Read betas/eps/weight_decay off the config rather than hardcoding them:
    # the values happened to match the defaults, so changing the config would
    # have silently had no effect.
    #
    # fused, NOT the default foreach. The multi-tensor path allocates a
    # full-size temporary per operation -- `torch._foreach_sqrt` on the
    # exp_avg_sq list is another 4 bytes/param, 3 GiB at 800M trainable, and
    # it is where this OOM'd on a 22 GiB card. The fused kernel does the same
    # arithmetic in place.
    adam_kw = dict(lr=lr, betas=tuple(cfg.optimizer_betas), eps=cfg.optimizer_eps,
                   weight_decay=cfg.optimizer_weight_decay)
    try:
        opt = torch.optim.AdamW(params, fused=True, **adam_kw)
        print("optimizer: fused AdamW")
    except (RuntimeError, ValueError, TypeError):
        opt = torch.optim.AdamW(params, foreach=False, **adam_kw)
        print("optimizer: AdamW with foreach=False (fused unavailable) — "
              "slower per step, but it does not allocate the full-size "
              "temporaries the default path does")

    def lr_at(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        p = (step - warmup_steps) / max(training_steps - warmup_steps, 1)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    for _ in range(start_step):
        sched.step()

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=(device == "cuda"), drop_last=True,
        persistent_workers=num_workers > 0)
    steps_per_epoch = max(len(loader) // max(grad_accum, 1), 1)
    print(f"{len(loader)} batches/epoch, batch_size={batch_size}, "
          f"grad_accum={grad_accum}")
    print(f"{steps_per_epoch} OPTIMIZER steps/epoch -> {training_steps} steps = "
          f"{training_steps / steps_per_epoch:.1f} epochs "
          f"(warmup {warmup_steps} = {warmup_steps / steps_per_epoch:.2f} epochs)")

    # The batch MUST go through the preprocessor before the loss sees it.
    # Everything downstream is written for MEAN_STD-normalized state and
    # action: the flow target, the discrete head's binning (clip=3.0, i.e.
    # +-3 sigma), and the threshold `calibrate_gripper_threshold` returns
    # (which it explicitly documents as being in normalized units). And
    # `select_action` applies this same pipeline at inference, so skipping it
    # here trains a different model than the one that gets deployed.
    #
    # Measured cost of the omission on this dataset's action stats: the three
    # rotation dims have std 0.04-0.08 against the gripper's 1.0, so raw
    # training gave rotation 0.8% of the flow loss where normalization gives
    # it 43%. Grasp pose was effectively not being trained.
    #
    # The restore loop is not optional. `transition_to_batch` rebuilds the
    # dict from observation.* / action / *_is_pad / task / index / task_index
    # ONLY, so every other key is dropped on the round trip -- including
    # `progress` from ProgressDataset, whose absence makes the progress head
    # silently skip its term instead of failing.
    def prepare(batch):
        out = preprocessor(batch)
        for k, v in batch.items():
            out.setdefault(k, v)
        return out

    # `step` counts OPTIMIZER steps, not dataloader iterations. The scheduler
    # advances once per optimizer step, so counting iterations instead makes
    # --training_steps and --warmup_steps mean different things at different
    # --grad_accum: at grad_accum=8 a 60000-"step" run would take only 7500
    # scheduler steps, i.e. finish at ~peak LR with the cosine barely started,
    # and warmup would silently last 8x longer than requested.
    policy.train()
    step, micro, t0, acc, n_acc = start_step, 0, time.time(), {}, 0
    done, checked = False, False
    while not done:
        for batch in loader:
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            batch = prepare(batch)
            if not checked:
                checked = True
                # One line that would have caught all of the above. RMS 1.0 is
                # what MEAN_STD normalization means; the predicted flow value
                # is exact because action_out_proj is zero-init, so v(0) = 0.
                a2 = float(batch["action"].float().pow(2).mean())
                print(f"[wiltechs_x] first batch after preprocessor: action "
                      f"RMS={a2 ** 0.5:.3f} (1.000 = normalized), "
                      f"progress={'present' if 'progress' in batch else 'MISSING'}"
                      f" -> flow should start at "
                      f"{cfg.sample_noise_scale ** 2 + a2:.2f}")
            loss, parts = policy.model.compute_loss(batch, return_parts=True)
            (loss / grad_accum).backward()

            for k, v in parts.items():
                acc[k] = acc.get(k, 0.0) + v
            acc["total"] = acc.get("total", 0.0) + float(loss.detach())
            n_acc += 1
            micro += 1
            if micro % grad_accum:
                continue

            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1

            if step % log_every == 0:
                msg = "  ".join(f"{k}={v / max(n_acc, 1):.4f}"
                                for k, v in sorted(acc.items()))
                mem = ""
                if device == "cuda":
                    mem = (f"  mem={torch.cuda.max_memory_allocated() / 2**30:.1f}/"
                           f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f}GiB")
                    torch.cuda.reset_peak_memory_stats()
                print(f"step {step}/{training_steps}  lr={sched.get_last_lr()[0]:.2e}  "
                      f"{msg}  {(time.time() - t0) / log_every:.2f}s/step{mem}")
                acc, n_acc, t0 = {}, 0, time.time()

            if step % save_every == 0 or step >= training_steps:
                ck = out / f"checkpoint-{step}"
                ck.mkdir(parents=True, exist_ok=True)
                cfg.training_step = step
                policy.save_pretrained(ck)
                preprocessor.save_pretrained(ck)
                postprocessor.save_pretrained(ck)
                torch.save({"model": policy.state_dict(), "opt": opt.state_dict(),
                            "step": step}, ck / "training_state.pth")
                print(f"saved {ck}")

            if step >= training_steps:
                done = True
                break

    (out / "run_config.json").write_text(json.dumps(
        {"dataset_ids": dataset_ids, "steps": training_steps,
         "cameras": cameras, "wrist": wrist_keys,
         "trainable": counts["trainable"]}, indent=2))
    print("done")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_ids", nargs="+", required=True)
    p.add_argument("--output_dir", default="./outputs/wiltechs_x")
    p.add_argument("--vlm_model_id", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--training_steps", type=int, default=60000,
                   help="OPTIMIZER steps, not dataloader iterations. With "
                        "--grad_accum 8 this is 8x that many batches. "
                        "--warmup_steps is in the same unit.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--n_action_steps", type=int, default=8,
                   help="Executed in full before replanning (the OFT setting).")
    p.add_argument("--expert_hidden_size", type=int, default=1024)
    p.add_argument("--expert_intermediate_size", type=int, default=0,
                   help="0 = match --expert_hidden_size.")
    p.add_argument("--expert_num_layers", type=int, default=0,
                   help="0 = one expert block per VLM layer. Fewer = a shallower "
                        "expert attached to the deepest N layers only.")
    p.add_argument("--ada_rank", type=int, default=64,
                   help="Rank of the adaLN modulation factorisation. A full "
                        "Linear(d,6d) is 32%% of the expert (226M at 36L/1024) "
                        "and OOM'd a 22 GiB card. 0 = full rank.")
    p.add_argument("--num_register_tokens", type=int, default=8)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_on_vision_tower", action="store_true",
                   help="Also adapt the ViT. Off by default: the vision tower is "
                        "the part most likely to lose general features, and the "
                        "wrist encoder is the designated trainable visual path.")
    p.add_argument("--freeze_vlm", action="store_true",
                   help="ABLATION ONLY. This is the configuration that produced "
                        "this repo's vision collapse; no top-10 LIBERO method "
                        "freezes its backbone.")
    p.add_argument("--causal_prefix", dest="bidirectional_prefix",
                   action="store_false", default=True,
                   help="Keep Qwen's causal mask over the prefix. The default is "
                        "bidirectional, which is what pi0/PaliGemma/X-VLA train.")
    p.add_argument("--no_knowledge_insulation", dest="knowledge_insulation",
                   action="store_false", default=True,
                   help="Let flow-matching gradients into the VLM. Expect language "
                        "grounding to degrade; this is the ablation that shows it.")
    p.add_argument("--no_discrete_head", dest="discrete_head",
                   action="store_false", default=True,
                   help="THE FIRST ABLATION TO RUN. The knowledge-insulation "
                        "result comes from large cross-embodiment corpora; at "
                        "LIBERO's 50 demos/task LoRA's rank constraint may "
                        "already supply the insulation and this head may be "
                        "dead weight.")
    p.add_argument("--fast_token_loss_weight", type=float, default=0.5)
    p.add_argument("--no_wrist_encoder", dest="wrist_encoder",
                   action="store_false", default=True)
    p.add_argument("--wrist_encoder_id", default="facebook/dinov2-small",
                   help="DINOv3 ids on HF were NOT verified when this was "
                        "written; confirm before switching off the v2 default.")
    p.add_argument("--wrist_cameras", nargs="+", default=None)
    p.add_argument("--wrist_tokens", type=int, default=256,
                   help="Perfect square. Granularity is --wrist_input_size / "
                        "sqrt(this) px per token and MUST come out below the Qwen "
                        "grid's 32, or this path supplies nothing the VLM tokens "
                        "do not already have. 256px/sqrt(64) = 32.0 is a no-op. "
                        "COST: these sit in the prefix, so they lengthen the K/V "
                        "every expert layer attends to — the first knob to turn "
                        "down under memory pressure.")
    p.add_argument("--wrist_input_size", type=int, default=256,
                   help="Raising this instead of --wrist_tokens buys the same "
                        "px/token without lengthening the prefix; the cost moves "
                        "into the DINO forward. 512 + 64 tokens = 16 px/token at "
                        "a quarter of the prefix length of 256 tokens @ 256px.")
    p.add_argument("--freeze_wrist_encoder", action="store_true",
                   help="Skips the DINO backward. Saves memory, but this is the "
                        "path this repo measured at 34 points — freeze it only "
                        "to fit, not as a default.")
    p.add_argument("--no_motion_vectors", dest="motion_vectors",
                   action="store_false", default=True)
    p.add_argument("--motion_history_len", type=int, default=8)
    p.add_argument("--motion_vector_tokens", type=int, default=8)
    p.add_argument("--no_progress_head", dest="progress_head",
                   action="store_false", default=True)
    p.add_argument("--progress_loss_weight", type=float, default=0.1)
    p.add_argument("--flow_objective", default="shortcut",
                   choices=["flow", "shortcut"],
                   help="'meanflow' is declared in the config but not implemented.")
    p.add_argument("--shortcut_consistency_frac", type=float, default=0.25)
    p.add_argument("--num_inference_steps", type=int, default=4)
    p.add_argument("--sample_noise_scale", type=float, default=1.0,
                   help="Temperature on the initial noise. Matters for stage B: a "
                        "flow policy annealed to near-determinism cannot explore "
                        "and RL dies silently.")
    p.add_argument("--noise_temporal_correlation", type=float, default=0.0,
                   help="AR(1) correlation across the horizon. 0 = iid.")
    p.add_argument("--vision_input_size", type=int, default=0)
    p.add_argument("--lang_max_len", type=int, default=48)
    p.add_argument("--instruction_template", type=str, default="",
                   help="Must contain the literal '{instruction}'. Empty = the "
                        "bare task string.")
    p.add_argument("--action_loss_weight", type=float, default=1.0)
    p.add_argument("--loss_exec_steps", type=int, default=0,
                   help="Steps the loss treats as executed; the tail beyond is "
                        "scaled by --future_steps_weight. 0 = full horizon. "
                        "Deliberately NOT tied to --n_action_steps: coupling the "
                        "loss to an inference knob is the bug wiltechs_vla had "
                        "to back out.")
    p.add_argument("--future_steps_weight", type=float, default=1.0)
    p.add_argument("--gripper_bce_weight", type=float, default=0.05)
    p.add_argument("--gripper_action_dim", type=int, default=-1)
    p.add_argument("--gripper_bce_temp", type=float, default=0.25)
    p.add_argument("--no_gripper_class_balance", action="store_true",
                   help="Without balancing this term sits in the majority-class "
                        "optimum (~89% open) and transition-time agreement stays "
                        "at chance. Ablation only.")
    p.add_argument("--use_descriptive_objects", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--max_episode_index", type=int, default=None)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--resume_from_checkpoint", default=None)
    p.add_argument("--seed", type=int, default=42)
    train(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
