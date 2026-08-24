"""Stage-A SFT trainer for WiltechsX.

    python src/train_wiltechs_x.py \
        --dataset_ids physical-intelligence/libero \
        --output_dir ./outputs/wx_a --batch_size 12 --grad_accum 8

--training_steps is in OPTIMIZER steps. At batch 12 x grad_accum 8 one epoch
over LIBERO's ~273k frames is ~2848 of them, and the trainer prints the epoch
count at startup -- read it before committing to a number. The step rate is
dominated by the 452-token prefix through 36 layers; --profile_steps attributes
it, and the first thing to check is whether "data wait" is large.

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

Qwen image preprocessing runs in the DataLoader workers by default
(`VlmPixelDataset`); `--no_preprocess_in_workers` puts it back on the critical
path inside `_encode_images`, which is useful only for ruling that path out as
a suspect. Both produce identical vision grids -- the `vision grid <cam>` line
at startup is printed from whichever one ran.
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

from models.wiltechs_vla.wiltechs_vla_model import (
    preprocess_camera_to_pixels,
    vlm_grid_key,
    vlm_pixels_key,
)
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


class VlmPixelDataset(torch.utils.data.Dataset):
    """Runs the Qwen image processor in the DataLoader worker rather than on
    the training loop's critical path.

    `_encode_images` already prefers `vlm_pixels_key(cam)` / `vlm_grid_key(cam)`
    when present and falls back to running the processor inline, so this moves
    WHERE the work happens without changing what is computed. Confirm with the
    `vision grid <cam>: [1, 16, 16]` line at startup: it is printed from
    whichever path produced the grid, so a mismatch between the two shows up
    there rather than as a silent difference.

    The raw camera tensors are kept — the wrist encoder consumes those, not the
    Qwen pixels.

    Ported from `train_wiltechs_vla.VLMImagePreprocDataset`, minus the image
    augmentation this trainer does not do.
    """

    def __init__(self, base, image_processor, camera_keys, target_size=0):
        self.base = base
        self.image_processor = image_processor
        self.camera_keys = list(camera_keys)
        # Must match what _encode_images would have passed inline, or the two
        # paths build different vision grids for the same frame.
        self.target_size = int(target_size or 0)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        s = self.base[i]
        for k in self.camera_keys:
            v = s.get(k)
            if not isinstance(v, torch.Tensor):
                continue
            # delta_timestamps gives cameras a leading T=1; the inline path
            # takes [:, -1] for the same reason.
            img = v[-1] if v.dim() == 4 else v
            pv, thw = preprocess_camera_to_pixels(self.image_processor, img,
                                                  target_size=self.target_size)
            s[vlm_pixels_key(k)] = pv                          # (P, dim)
            s[vlm_grid_key(k)] = thw[0]                        # (3,)
        return s


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
        # fps is NOT cosmetic here. `delta` below is derived once from the FIRST
        # dataset's fps and then applied to every dataset, so a mismatch changes
        # the STRIDE of the action chunk and state history in the others: a set
        # stamped at 20 Hz mixed under a 10 Hz reference gets every second frame,
        # i.e. a 64-step chunk covering 128 real steps, and nothing raises. The
        # opposite order lands between frames and fails the tolerance instead.
        # `train_rft.py --rft.collect_only` defaults --rft.save_fps to 20 while
        # LIBERO runs at 10, so this is the expected way to get bitten.
        if abs(float(metas[d].fps) - float(ref.fps)) > 1e-6:
            raise ValueError(
                f"dataset {d!r} is {metas[d].fps} fps but {dataset_ids[0]!r} is "
                f"{ref.fps}. Mixing them would silently re-stride one of the two. "
                f"Re-stamp the collected set (its frames are unchanged -- only "
                f"info.json's fps is wrong) or collect again with "
                f"--rft.save_fps={ref.fps}.")

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

    subs, ep_from, ep_to, ep_task, offset = [], [], [], [], 0
    for d in dataset_ids:
        ds = LeRobotDataset(d, delta_timestamps=delta, force_cache_sync=True,
                            revision="main", tolerance_s=max(0.005, ft / 2))
        ep = np.array(ds.hf_dataset["episode_index"])
        # Per-episode task label, for stratifying the validation holdout. Keyed
        # by dataset too: the same task_index means different things in two
        # datasets, and a holdout that took every episode of one task would
        # measure generalisation to an unseen TASK, not to unseen episodes.
        cols = getattr(ds.hf_dataset, "column_names", [])
        ti = np.array(ds.hf_dataset["task_index"]) if "task_index" in cols else None
        cuts = np.where(np.diff(ep) != 0)[0] + 1
        starts = np.concatenate([[0], cuts])
        ends = np.concatenate([cuts, [len(ep)]])
        for s, e in zip(starts, ends):
            if max_episode_index is not None and int(ep[s]) > max_episode_index:
                continue
            ep_from.append(offset + int(s))
            ep_to.append(offset + int(e))
            ep_task.append(f"{d}#{int(ti[s])}" if ti is not None else d)
        subs.append(ds)
        offset += len(ds)

    base = subs[0] if len(subs) == 1 else torch.utils.data.ConcatDataset(subs)
    return {
        "dataset": base, "ep_from": ep_from, "ep_to": ep_to,
        "ep_task": ep_task, "cameras": cameras,
        "state_dim": state_dim, "action_dim": action_dim, "stats": stats,
        "fps": fps, "input_features": in_f, "output_features": out_f,
        # Carried out so the paraphrase preflight can enumerate the instruction
        # strings without re-fetching the metadata.
        "meta": ref, "metas": metas,
    }


def split_episodes(ep_task, n_val: int, seed: int):
    """-> (train_ep, val_ep, alloc) holding out whole EPISODES, stratified.

    Episode-level, never frame-level. Neighbouring frames of one episode share
    almost all of their pixels and an action chunk overlapping by horizon-1
    steps, so a frame split leaks the answer and the val loss then measures
    nothing but memorisation of the same episode.

    Stratified over `ep_task`, and never takes a whole group: a task left with
    no training episodes would be scored on something the model was never
    shown, which is generalisation to an unseen TASK -- a different question
    than the one this is here to answer.
    """
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = {}
    for i, t in enumerate(ep_task):
        groups.setdefault(t, []).append(i)
    keys = sorted(groups)
    alloc = {k: 0 for k in keys}
    left = int(n_val)
    while left > 0:                       # round-robin, so small groups are not
        moved = False                     # squeezed out by large ones
        for k in keys:
            if left and alloc[k] < len(groups[k]) - 1:
                alloc[k] += 1
                left -= 1
                moved = True
        if not moved:
            break
    val: list[int] = []
    for k in keys:
        if alloc[k]:
            val += [int(i) for i in rng.choice(groups[k], size=alloc[k],
                                               replace=False)]
    vs = set(val)
    train = [i for i in range(len(ep_task)) if i not in vs]
    return train, sorted(val), alloc


@torch.no_grad()
def run_validation(policy, loader, prepare, device, max_batches: int, seed: int):
    """-> (mean loss parts over held-out episodes, n_batches).

    Two things make the number comparable across steps rather than just
    smaller-looking:

      * The flow draws are PINNED. compute_loss samples `t` and the noise fresh
        every call, so two passes over identical data differ by the sampling
        alone -- at n=20 batches that is easily the size of the effect being
        looked for. The RNG state is saved and restored, so measuring does not
        perturb the training stream.
      * policy.eval() disables paraphrase augmentation, because
        _resolve_descs gates on self.training. The val loss is therefore always
        against the CANONICAL instruction; a random paraphrase per pass would
        move the target between measurements.
    """
    was_training = policy.training
    policy.eval()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if device == "cuda" else None
    torch.manual_seed(seed)
    acc: dict[str, float] = {}
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        _, parts = policy.model.compute_loss(prepare(batch), return_parts=True)
        for k, v in parts.items():
            acc[k] = acc.get(k, 0.0) + float(v)
        n += 1
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    if was_training:
        policy.train()
    return {k: v / max(n, 1) for k, v in acc.items()}, n


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


# Two sides, because knowledge insulation makes them structurally different.
# Everything in PREFIX_GROUPS lives upstream of the detached K/V cache, so the
# flow loss cannot reach it -- the discrete head is its ONLY gradient source.
# Everything in EXPERT_GROUPS is trained by the flow/shortcut/gripper/progress
# terms and never sees the discrete CE.
PREFIX_GROUPS = [
    ("LoRA (q/k/v/o)", "lora_"),
    ("wrist encoder", "wrist_encoder"),
    ("  wrist proj", "wrist_encoder.proj"),
    ("motion encoder", "motion_encoder"),
    ("discrete head", "discrete_head"),
]
EXPERT_GROUPS = [
    ("expert layers", "expert_layers"),
    ("  adaLN", ".ada."),
    ("state encoder", "state_encoder"),
    ("registers", "register_tokens"),
    ("action pos emb", "action_pos_emb"),
    ("action in", "action_in_proj"),
    ("action out", "action_out_proj"),
    ("time embedder", "time_embedder"),
    ("final norm", "final_norm"),
    ("progress head", "progress_head"),
]


def _grad_stats(model, needle: str):
    """-> (mean|g|, g_rms/param, g/w, n_with_grad, n_trainable).

    `n_trainable` separates the two ways a group reports nothing: DISABLED (the
    module does not exist -- no wrist encoder, no discrete head) from BROKEN
    (parameters are there and got no gradient). Printing "no grad" for both
    buries the one that matters.
    """
    total, g_sq, w_sq, n, present = 0.0, 0.0, 0.0, 0, 0
    for name, p in model.named_parameters():
        if not (p.requires_grad and needle in name):
            continue
        present += p.numel()
        if p.grad is not None:
            # norm(1), not abs().mean(): this runs at PEAK memory (every
            # gradient is live, opt.zero_grad has not happened yet) and
            # .abs() would materialise a full-size copy of the largest
            # tensor to compute one scalar.
            total += p.grad.norm(1).item()
            g_sq += p.grad.norm().item() ** 2
            w_sq += p.detach().norm().item() ** 2
            n += p.numel()
    if n == 0:
        return None, None, None, 0, present
    g_rms = (g_sq ** 0.5) / (n ** 0.5)
    w_rms = (w_sq ** 0.5) / (n ** 0.5)
    return total / n, g_rms, (g_rms / w_rms if w_rms > 0 else None), n, present


def log_gradient_analysis(model, step: int, knowledge_insulation: bool) -> None:
    """Per-component gradient health, on the RAW accumulated gradient.

    Called BEFORE clip_grad_norm_, so the magnitudes are what the model
    produced rather than what survived rescaling.

    Read the `g/w` column. `avg|g|` alone rounds to zero for a large module
    whose gradient concentrates in a few sub-parameters, and `rms/param` is
    fair in COUNT but not in SCALE -- a pretrained DINOv2 weight is far larger
    than a freshly initialised projection, so the same absolute gradient is a
    much smaller relative step. g/w (gradient RMS over weight RMS) is the
    scale-free version and the only column that compares across modules.
    """
    print(f"\n--- gradients at step {step} (raw, pre-clip) ---")

    seen: dict = {}                 # label -> g/w, for the ratio line below

    def section(title, groups):
        tot = 0.0
        print(f"  {title}")
        for label, needle in groups:
            g, rms, gw, n, present = _grad_stats(model, needle)
            if gw is not None:
                seen[label.strip()] = gw
            if g is None:
                if present:
                    print(f"    {label:<16s} *** {present:,} trainable params, "
                          f"NO GRAD ***")
                continue                       # present == 0 -> disabled, be quiet
            if not label.startswith("  "):     # sub-rows must not double-count
                tot += rms * (n ** 0.5)
            gw_s = f"  g/w {gw:.2e}" if gw is not None else ""
            print(f"    {label:<16s} avg|g| {g:.3e}  rms/param {rms:.2e}"
                  f"{gw_s}  ({n / 1e6:.1f}M)")
        return tot

    # The header states the GRADIENT PATH, which knowledge insulation changes.
    # It used to read "reachable ONLY via the discrete CE head" unconditionally,
    # so a --no_knowledge_insulation run printed a header denying the very
    # thing it was there to measure.
    pre = section(
        "PREFIX side — reachable ONLY via the discrete CE head:"
        if knowledge_insulation else
        "PREFIX side — discrete CE head AND flow/shortcut/gripper/progress "
        "(KI off):",
        PREFIX_GROUPS)
    exp = section("EXPERT side — trained by flow / shortcut / gripper / progress:",
                  EXPERT_GROUPS)

    if pre > 0 and exp > 0:
        print(f"  prefix/expert gradient L2 = {pre / exp:.3f}")
        # Comparing g/w across REPORTS is unsafe when the batch changed --
        # raw gradient magnitude carries the batch's noise scale. Ratios taken
        # inside one report do not, so print the two that matter. Measured
        # across the KI switch at matched steps 14000/14050: wrist 0.025 ->
        # 0.036 while motion went 1.43 -> 6.12, i.e. the gradient a segment
        # gains is proportional to the attention it already had.
        base = seen.get("expert layers")
        if base:
            parts = [f"{k}/expert {seen[k] / base:.4f}"
                     for k in ("wrist encoder", "motion encoder", "LoRA (q/k/v/o)",
                               "discrete head") if k in seen]
            if parts:
                print("  g/w relative to expert layers (batch-independent, so "
                      "these compare across runs):\n    " + "   ".join(parts))
    if knowledge_insulation and pre == 0.0:
        print("  *** PREFIX SIDE IS DEAD ***\n"
              "  knowledge_insulation detaches the K/V cache, so the discrete "
              "head is the only\n  gradient path into LoRA, the wrist encoder "
              "and the motion encoder. Zero here\n  means the backbone is "
              "frozen in practice while still costing full forward time.\n"
              "  Check that --fast_token_loss_weight is non-zero and that "
              "`discrete` is falling.")


def resolve_checkpoint(path_or_repo: str, *, for_resume: bool = True) -> Path:
    """Accept a local directory OR a Hugging Face repo id.

    `--resume_from_checkpoint ISdept/wiltech-x-6k` used to fail with a bare
    FileNotFoundError on `<repo>/training_state.pth`, which reads like a
    corrupt checkpoint rather than "that is a Hub id and this function only
    knew about disk".

    `for_resume` picks what to download. Resuming reads training_state.pth,
    which already carries the model, so model.safetensors would be ~9.5 GiB of
    waste. Loading a policy for eval or a probe needs exactly the opposite.
    """
    p = Path(path_or_repo)
    if p.exists():
        return p
    if "/" not in path_or_repo or path_or_repo.startswith((".", "/", "~")):
        raise SystemExit(f"no such checkpoint directory: {path_or_repo}")
    from huggingface_hub import list_repo_files, snapshot_download

    print(f"'{path_or_repo}' is not a local path; fetching from the Hub...")
    files = set(list_repo_files(path_or_repo))
    if for_resume and "training_state.pth" in files:
        # NOT lerobot's allow_patterns (*.safetensors/*.json): that filter
        # drops training_state.pth and would silently downgrade a full resume
        # to a weights-only one. And since the .pth already carries the model,
        # pulling model.safetensors too is ~9.5 GiB of redundant download.
        patterns = ["training_state.pth", "*.json"]
        print("  repo has training_state.pth -> full resume "
              "(skipping the redundant model.safetensors)")
    elif for_resume:
        patterns = ["*.safetensors", "*.json"]
        print("  repo has no training_state.pth -> weights-only resume")
    else:
        patterns = ["*.safetensors", "*.json"]
    local = Path(snapshot_download(repo_id=path_or_repo, allow_patterns=patterns))
    print(f"  -> {local}")
    return local


def _report_key_diff(policy, missing, unexpected, allow_new: bool, src: str):
    """Account for every parameter the checkpoint did not supply.

    Adding a module to a resumed run (the wrist encoder is the case this was
    written for) is legitimate, but it means part of the model starts from
    random init inside a converged one. Both load paths used to be STRICT, so
    it raised; making them lenient without saying what got initialised would be
    worse -- half a model silently freshly-initialised is a 25-hour mistake
    that looks like a training failure.

    `unexpected` is the opposite case: the checkpoint has parameters this model
    does not. That means something was REMOVED, which is almost never intended
    on a resume, so it is always reported.
    """
    if unexpected:
        by = {}
        for k in unexpected:
            by[k.split(".")[0]] = by.get(k.split(".")[0], 0) + 1
        print(f"*** {len(unexpected)} tensor(s) in {src} have NO home in this "
              f"model — something was removed since it was written: "
              f"{dict(sorted(by.items(), key=lambda x: -x[1])[:6])}")
    if not missing:
        return
    shapes = dict(policy.named_parameters())
    shapes.update(dict(policy.named_buffers()))
    by_mod, n_fresh = {}, 0
    for k in missing:
        n = shapes[k].numel() if k in shapes else 0
        n_fresh += n
        # Group at the module that owns it, not the leaf.
        pre = ".".join(k.split(".")[:2]) if "." in k else k
        by_mod[pre] = by_mod.get(pre, 0) + n
    total = sum(p.numel() for p in policy.parameters())
    lines = "\n".join(f"      {m:<44} {n / 1e6:>8.1f}M"
                      for m, n in sorted(by_mod.items(), key=lambda x: -x[1])[:12])
    msg = (f"{len(missing)} tensor(s) / {n_fresh / 1e6:.1f}M parameters "
           f"({100 * n_fresh / max(total, 1):.1f}% of the model) are NOT in "
           f"{src} and would start from RANDOM INIT:\n{lines}")
    if not allow_new:
        raise SystemExit(
            f"*** {msg}\n"
            f"  If that is the point -- adding a module to a resumed run -- pass "
            f"--allow_new_modules.\n"
            f"  If it is not, the config does not match the checkpoint: check "
            f"--expert_hidden_size / --expert_intermediate_size / "
            f"--expert_num_layers / --horizon, which change parameter SHAPES "
            f"and cannot be resumed into.")
    print(f"*** --allow_new_modules: {msg}\n"
          f"    A freshly-initialised module inside a converged model starts at "
          f"a disadvantage: attention has already learned to route around\n"
          f"    tokens that were not there, and this repo has measured that\n"
          f"    throttle to be self-reinforcing. Give it warmup and read the\n"
          f"    per-component gradient report before trusting the eval.")


def load_resume_state(policy, ck: Path, device: str, allow_new: bool = False):
    """-> (resume_state | None, start_step | None).

    Two kinds of checkpoint reach this. `training_state.pth` is what this
    trainer writes: weights + optimizer + step, a real resume. A directory
    holding only `model.safetensors` is what `save_pretrained`/`push_to_hub`
    produces -- the weights are all that survived, and BOTH the Adam moments
    and the step counter are gone. The second case is silently destructive:
    start_step 0 restarts the LR schedule from warmup on a model that is
    thousands of steps in, so it is reported loudly and --start_step exists to
    repair it.

    Loading is non-strict so a module can be ADDED on resume, but every tensor
    the checkpoint did not supply is accounted for; see _report_key_diff.
    """
    state_file = ck / "training_state.pth"
    if state_file.exists():
        # CPU, not `device`: this file holds the full model AND the Adam
        # moments (~9.5 + 2.6 GiB here), and mapping it straight onto the GPU
        # doubles the weights for as long as the dict is alive.
        st = torch.load(state_file, map_location="cpu")
        r = policy.load_state_dict(st["model"], strict=False)
        _report_key_diff(policy, r.missing_keys, r.unexpected_keys, allow_new,
                         "training_state.pth")
        return st, st.get("step", 0)

    from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
    from safetensors.torch import load_file as load_safetensor_file

    weights = ck / SAFETENSORS_SINGLE_FILE
    if not weights.exists():
        found = sorted(f.name for f in ck.iterdir()) if ck.is_dir() else []
        raise SystemExit(
            f"{ck} has neither training_state.pth nor {SAFETENSORS_SINGLE_FILE}.\n"
            f"  contents: {found}")
    sd = load_safetensor_file(str(weights), device="cpu")
    r = policy.load_state_dict(sd, strict=False)
    _report_key_diff(policy, r.missing_keys, r.unexpected_keys, allow_new,
                     SAFETENSORS_SINGLE_FILE)
    print(f"*** WEIGHTS-ONLY resume from {weights.name}.\n"
          f"    No optimizer state: the Adam moments restart at zero, which is "
          f"a transient of ~20 steps at betas=(0.9, 0.95).\n"
          f"    No step counter: the LR schedule would RESTART FROM WARMUP "
          f"unless you pass --start_step.\n"
          f"    A full resume needs training_state.pth, which "
          f"save_pretrained/push_to_hub does not write.")
    return None, None


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
    wrist_gate_init: float = 1.0,
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
    contrastive_loss_weight: float = 0.0,
    contrastive_margin: float = 0.05,
    contrastive_frac: float = 0.5,
    contrastive_suite_jaccard: float = 0.5,
    allow_new_modules: bool = False,
    paraphrase_augment: bool = False,
    paraphrase_limit: int = 8,
    paraphrase_file: str = "",
    paraphrase_min_variants: int = 5,
    use_descriptive_objects: bool = False,
    preprocess_in_workers: bool = True,
    profile_steps: int = 20,
    grad_log_every: int = 1000,
    num_workers: int = 4,
    val_episodes: int = 0,
    val_every: int = 500,
    val_max_batches: int = 20,
    save_every: int = 5000,
    log_every: int = 20,
    max_episode_index: int | None = None,
    gradient_checkpointing: bool = False,
    resume_from_checkpoint: str | None = None,
    start_step_override: int = -1,
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

    if paraphrase_augment:
        # Preflight, not a runtime warning. A sentence the templates decline to
        # restructure trains UNAUGMENTED while the rest vary, so the model keeps
        # surface form as a usable key for exactly those tasks -- and the run
        # cannot answer whether augmentation works. Twenty hours is too long to
        # find that out from the eval.
        from models.wiltechs_x.paraphrase import (
            coverage, instruction_strings, load_table)
        # Union over every dataset, not just the reference one: with several
        # --dataset_ids the task lists differ, and an instruction that only
        # appears in the second one still needs variants.
        instructions, seen = [], set()
        for m in D.get("metas") or ([D["meta"]] if "meta" in D else []):
            raw = getattr(m, "tasks", None)
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
                  "  python -m models.wiltechs_x.paraphrase --dataset_id <id> "
                  "--min_variants N\nbefore trusting this run.")
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
                    f"    python -m models.wiltechs_x.paraphrase "
                    f"--dataset_id {dataset_ids[0]} --out para.json\n"
                    f"  then hand-edit the entries listed as UNDER. Lower "
                    f"--paraphrase_min_variants only if you accept that those "
                    f"tasks train unaugmented.")

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
        wrist_gate_init=wrist_gate_init,
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
        contrastive_loss_weight=contrastive_loss_weight,
        contrastive_margin=contrastive_margin,
        contrastive_frac=contrastive_frac,
        contrastive_suite_jaccard=contrastive_suite_jaccard,
        paraphrase_augment=paraphrase_augment,
        paraphrase_limit=paraphrase_limit,
        paraphrase_file=paraphrase_file,
        paraphrase_min_variants=paraphrase_min_variants,
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

    start_step, resume_state = 0, None
    if resume_from_checkpoint:
        ck = resolve_checkpoint(resume_from_checkpoint)
        print(f"resuming from {ck}")
        resume_state, ckpt_step = load_resume_state(policy, ck, device,
                                                    allow_new_modules)
        start_step = ckpt_step if ckpt_step is not None else 0
        if start_step_override >= 0:
            print(f"start_step: {start_step} -> {start_step_override} (--start_step)")
            start_step = start_step_override
        if start_step == 0:
            print("*** starting the LR schedule from step 0 on a resumed model. "
                  "If this checkpoint is not from step 0, pass --start_step.")

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

    # Restore the Adam moments. The checkpoint has always written them and the
    # resume path has never read them, so every resume so far restarted
    # exp_avg/exp_avg_sq at zero. At betas=(0.9, 0.95) the second moment is
    # rebuilt over ~20 steps, and until it is the updates are effectively
    # unnormalised -- a loss spike on resume that looks like a real regression
    # and is not.
    if resume_state is not None:
        if "opt" in resume_state:
            try:
                opt.load_state_dict(resume_state["opt"])
                print("optimizer state restored (Adam moments continue)")
            except (ValueError, KeyError) as e:
                # Param groups differ -- a flag that changes what is trainable
                # was toggled between the two runs. Continue rather than kill a
                # multi-day resume, but do not let it pass silently.
                print(f"*** optimizer state NOT restored: {e}\n"
                      f"    The parameter set differs from the checkpoint's, so "
                      f"a flag affecting which modules train was changed. "
                      f"Expect a transient for ~20 steps.")
        else:
            print("*** checkpoint has no optimizer state; Adam restarts cold")
        resume_state = None                    # free the CPU copy

    def lr_at(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        p = (step - warmup_steps) / max(training_steps - warmup_steps, 1)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    for _ in range(start_step):
        sched.step()

    if preprocess_in_workers:
        dataset = VlmPixelDataset(dataset, policy.model.processor.image_processor,
                                  cameras, target_size=vision_input_size)
        where = (f"{num_workers} DataLoader workers (parallel, overlapped with "
                 f"GPU)" if num_workers > 0
                 else "the MAIN process — --num_workers is 0, so this buys "
                      "nothing; raise it or pass --no_preprocess_in_workers")
        print(f"Qwen image preprocessing runs in {where}.")
    else:
        print("Qwen image preprocessing runs INLINE in _encode_images, on the "
              "critical path. --profile_steps will show what that costs.")

    # Subset the OUTERMOST wrapper: ProgressDataset and VlmPixelDataset are
    # both index-preserving maps, so a positional Subset over either reaches
    # the same frames as one over the base.
    val_loader, train_ds, n_val_frames = None, dataset, 0
    if val_episodes > 0:
        tr_ep, va_ep, alloc = split_episodes(D["ep_task"], val_episodes, seed)
        frames = lambda eps: [i for e in eps
                              for i in range(D["ep_from"][e], D["ep_to"][e])]
        tr_idx, va_idx = frames(tr_ep), frames(va_ep)
        n_val_frames = len(va_idx)
        empty = [k for k, v in alloc.items() if v == 0]
        print(f"Validation: {len(va_ep)} episodes / {n_val_frames} frames held "
              f"out over {len(alloc)} group(s) (dataset#task), "
              f"{min(alloc.values())}-{max(alloc.values())} each"
              + (f"  *** {len(empty)} group(s) got NONE — raise "
                 f"--val_episodes to at least {len(alloc)} ***" if empty else ""))
        if n_val_frames < batch_size:
            raise SystemExit(
                f"--val_episodes {val_episodes} holds out {n_val_frames} "
                f"frames, fewer than one batch ({batch_size}). Raise it.")
        # The TRAIN sampler must not see them. Dropping the val frames from the
        # val loader alone would leave them in the training stream and the
        # split would measure nothing.
        train_ds = torch.utils.data.Subset(dataset, tr_idx)
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, va_idx), batch_size=batch_size,
            shuffle=False, num_workers=max(num_workers // 2, 1),
            pin_memory=(device == "cuda"), drop_last=True)
    else:
        print("Validation: DISABLED (--val_episodes 0). Training loss alone "
              "cannot separate a better model from a better-memorised one.")

    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
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
    # ---- stage timing ---------------------------------------------------
    # Attribution, not a total: `s/step` in the log already gives the total,
    # and it cannot tell a dataloader stall from a slow backward. Each lap
    # needs a cuda synchronize to be meaningful, which itself costs time, so
    # this runs for a bounded window and then turns itself off -- the steady
    # state must not pay for the measurement. The first two optimizer steps
    # are discarded (allocator warmup, cuDNN autotune, worker spin-up).
    prof: dict[str, float] = {}
    prof_done, profiling = 0, profile_steps > 0
    PROF_WARMUP = 2

    def _sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def _lap(name, t_from):
        _sync()
        t_now = time.perf_counter()
        prof[name] = prof.get(name, 0.0) + (t_now - t_from)
        return t_now

    def _report_profile(n_steps):
        total = sum(prof.values())
        per = total / max(n_steps, 1)
        print(f"\nstage timing over {n_steps} optimizer steps "
              f"({grad_accum} micro-batches each, batch_size={batch_size})")
        for name in sorted(prof):
            v = prof[name]
            print(f"  {name:<22s} {v / max(n_steps, 1):7.3f} s/step "
                  f"({100 * v / max(total, 1e-9):5.1f}%)")
        print(f"  {'TOTAL':<22s} {per:7.3f} s/step "
              f"= {batch_size * grad_accum / max(per, 1e-9):.1f} samples/s")
        print(f"  {training_steps} steps -> {training_steps * per / 3600:.1f} h "
              f"({training_steps / max(steps_per_epoch, 1):.1f} epochs)")
        print("  'data wait' is the loop blocked on the DataLoader. High = the "
              "workers cannot keep up;\n  raise --num_workers or check "
              "--preprocess_in_workers. It is NOT the cost of preprocessing,\n"
              "  only the part that failed to overlap.\n")

    policy.train()
    step, micro, t0, acc, n_acc = start_step, 0, time.time(), {}, 0
    gn_sum, gn_clipped, gn_n = 0.0, 0, 0
    done, checked = False, False
    t_prev = time.perf_counter()
    while not done:
        for batch in loader:
            if profiling:
                t_mark = _lap("1 data wait", t_prev)
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            batch = prepare(batch)
            if profiling:
                t_mark = _lap("2 H2D + normalize", t_mark)
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
            if profiling:
                t_mark = _lap("3 forward", t_mark)
            (loss / grad_accum).backward()
            if profiling:
                t_mark = _lap("4 backward", t_mark)

            for k, v in parts.items():
                acc[k] = acc.get(k, 0.0) + v
            acc["total"] = acc.get("total", 0.0) + float(loss.detach())
            n_acc += 1
            micro += 1
            if micro % grad_accum:
                t_prev = time.perf_counter()
                continue

            if grad_log_every and (step + 1) % grad_log_every == 0:
                log_gradient_analysis(policy.model, step + 1, knowledge_insulation)

            # The pre-clip norm is free -- clip_grad_norm_ already computes it
            # and the old code threw it away. It is the cheapest way to see
            # that the LR you set is the LR you get: once this sits above
            # max_norm every step, every update is rescaled by max_norm/‖g‖ and
            # the effective step size is set by the clip, not by --lr.
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            gn_sum += gnorm
            gn_clipped += int(gnorm > 1.0)
            gn_n += 1

            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1

            if profiling:
                _lap("5 clip + opt.step", t_mark)
                prof_done += 1
                if prof_done == PROF_WARMUP:
                    prof.clear()                       # discard warmup
                elif prof_done >= PROF_WARMUP + profile_steps:
                    _report_profile(profile_steps)
                    profiling = False                  # stop paying for syncs

            if step % log_every == 0:
                msg = "  ".join(f"{k}={v / max(n_acc, 1):.4f}"
                                for k, v in sorted(acc.items()))
                mem = ""
                if device == "cuda":
                    mem = (f"  mem={torch.cuda.max_memory_allocated() / 2**30:.1f}/"
                           f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f}GiB")
                    torch.cuda.reset_peak_memory_stats()
                # gnorm is per OPTIMIZER step, so it cannot go in `acc` (which
                # is averaged over micro-batches). clip% is the part that
                # matters: at 100% the update direction is still yours but the
                # magnitude is the clip's, not --lr's.
                gn = (f"  gnorm={gn_sum / max(gn_n, 1):.2f}"
                      f"(clip {100 * gn_clipped / max(gn_n, 1):.0f}%)")
                print(f"step {step}/{training_steps}  lr={sched.get_last_lr()[0]:.2e}  "
                      f"{msg}{gn}  {(time.time() - t0) / log_every:.2f}s/step{mem}")
                acc, n_acc, t0 = {}, 0, time.time()
                gn_sum, gn_clipped, gn_n = 0.0, 0, 0

            if val_loader is not None and (step % val_every == 0
                                           or step >= training_steps):
                vp, nb = run_validation(policy, val_loader, prepare, device,
                                        val_max_batches, seed)
                vmsg = "  ".join(f"{k}={v:.4f}" for k, v in sorted(vp.items()))
                print(f"  VAL @ {step}  {vmsg}  "
                      f"({nb} batches of {batch_size} over {n_val_frames} "
                      f"held-out frames)")

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

            # LAST, so logging and checkpoint writes are not charged to the
            # next iteration's "data wait".
            t_prev = time.perf_counter()

            if step >= training_steps:
                done = True
                break

    # Every field below CHANGES WHAT THE RUN OPTIMISED. Recording only the
    # dataset and the step count left two runs that differ in their objective
    # looking identical on disk, which is how "is this one KI or not?" became
    # unanswerable after the fact.
    (out / "run_config.json").write_text(json.dumps(
        {"dataset_ids": dataset_ids, "steps": training_steps,
         "cameras": cameras, "wrist": wrist_keys,
         "trainable": counts["trainable"],
         "batch_size": batch_size, "grad_accum": grad_accum,
         "start_step": start_step,
         "resumed_from": str(resume_from_checkpoint) if resume_from_checkpoint else None,
         "knowledge_insulation": knowledge_insulation,
         "discrete_head": bool(policy.model.discrete_head is not None),
         "contrastive_loss_weight": contrastive_loss_weight,
         "contrastive_margin": contrastive_margin,
         "contrastive_frac": contrastive_frac,
         "contrastive_suite_jaccard": contrastive_suite_jaccard,
         "allow_new_modules": allow_new_modules,
         "val_episodes": val_episodes, "val_every": val_every,
         "val_max_batches": val_max_batches,
         "paraphrase_augment": paraphrase_augment,
         "paraphrase_limit": paraphrase_limit,
         "paraphrase_file": paraphrase_file,
         "paraphrase_min_variants": paraphrase_min_variants,
         "freeze_wrist_encoder": policy.config.freeze_wrist_encoder,
         "wrist_gate_init": policy.config.wrist_gate_init,
         "gradient_checkpointing": gradient_checkpointing}, indent=2))
    print("done")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_ids", nargs="+", required=True)
    p.add_argument("--output_dir", default="./outputs/wiltechs_x")
    p.add_argument("--vlm_model_id", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--training_steps", type=int, default=20000,
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
                   help="Perfect square, and it MUST exceed the VLM's own "
                        "per-camera token count — (grid_h/2)*(grid_w/2), which is "
                        "64 at a 16x16 patch grid — or this path resolves nothing "
                        "the prefix already has. The startup banner prints the "
                        "verdict FINER/IDENTICAL/COARSER once the real grid is "
                        "known. COST: these sit in the prefix, so they lengthen "
                        "the K/V every expert layer attends to — the first knob "
                        "to turn down under memory or throughput pressure.")
    p.add_argument("--wrist_input_size", type=int, default=256,
                   help="Feeds the DINO forward only. It does NOT substitute for "
                        "--wrist_tokens: the features are adaptive-avg-pooled to "
                        "a sqrt(wrist_tokens) grid, so a larger input is averaged "
                        "away rather than resolved. Raise it to give DINO more to "
                        "look at, not to buy token resolution.")
    p.add_argument("--wrist_gate_init", type=float, default=1.0,
                   help="Initial gain on the wrist tokenizer's output. 1.0 for "
                        "a fresh run. Use 1e-3 when ADDING the wrist encoder to "
                        "a resumed checkpoint, so its tokens start inert and "
                        "grow in rather than dumping full-magnitude noise into "
                        "a converged prefix.")
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
    p.add_argument("--contrastive_loss_weight", type=float, default=0.0,
                   help="Hinge forcing the action to DEPEND on the instruction: "
                        "with another sample's instruction the predicted "
                        "velocity must differ by at least --contrastive_margin. "
                        "0 = off, which is what shipped until 2026-08-17 and "
                        "what the language probe and the rollout ablation "
                        "showed leaves the policy ignoring the instruction. "
                        "This repo measured 0.1 as sufficient on the sibling "
                        "model. Costs one extra SUFFIX pass, not a prefix one.")
    p.add_argument("--contrastive_margin", type=float, default=0.05,
                   help="A floor to raise, not a ceiling: the sibling model's "
                        "hinge saturated around 15k steps at this value.")
    p.add_argument("--contrastive_frac", type=float, default=0.5,
                   help="Fraction of the batch given the extra suffix pass.")
    p.add_argument("--contrastive_suite_jaccard", type=float, default=0.5,
                   help="Token-Jaccard above which two instructions count as "
                        "the same suite and so as HARD negatives for each "
                        "other. The hinge keeps sample i's image and swaps in "
                        "j's instruction, so a cross-suite j names objects "
                        "absent from the scene and can be rejected on object "
                        "presence alone -- no relation parsing. Drawn "
                        "uniformly within the bucket, not argmax-similar, "
                        "which would collapse to one fixed partner per task. "
                        "0 = uniform-random negatives (behaviour before "
                        "2026-08-18).")
    p.add_argument("--allow_new_modules", action="store_true",
                   help="Permit --resume_from_checkpoint to leave parameters "
                        "randomly initialised because the checkpoint has no "
                        "entry for them -- i.e. you are ADDING a module (the "
                        "wrist encoder) to a resumed run. Without it a resume "
                        "that would silently initialise part of the model "
                        "refuses to start and prints which part.")
    p.add_argument("--paraphrase_augment", action="store_true",
                   help="Train on several phrasings per instruction, resampled "
                        "every step. Measured motivation: this model scores 60% "
                        "on its own instruction and 0% on a PARAPHRASE of it, "
                        "i.e. it keys on surface form, not content. Unlike "
                        "--use_descriptive_objects (one fixed rewrite = a "
                        "second table to memorise) the variant changes per "
                        "sample, so surface form stops being a usable key. The "
                        "original string is always in the set; eval uses it.")
    p.add_argument("--paraphrase_limit", type=int, default=8,
                   help="Variants per instruction, including the original.")
    p.add_argument("--paraphrase_file", default="",
                   help="JSON of instruction -> [variants], overriding the "
                        "templates. For the sentences they decline to "
                        "restructure (libero_goal, libero_10). Generate a "
                        "starting table with: python -m "
                        "models.wiltechs_x.paraphrase --dataset_id <id> --out f.json")
    p.add_argument("--paraphrase_min_variants", type=int, default=5,
                   help="Refuse to start if any instruction has fewer variants "
                        "than this. Partial augmentation is worse than none: "
                        "the untouched tasks keep surface form as a usable key "
                        "and the run cannot say whether augmentation worked.")
    p.add_argument("--use_descriptive_objects", action="store_true")
    p.add_argument("--no_preprocess_in_workers", dest="preprocess_in_workers",
                   action="store_false", default=True,
                   help="Run the Qwen image processor inline in _encode_images "
                        "instead of in the DataLoader workers. Inline is on the "
                        "critical path with the GPU idle; the only reason to "
                        "pick it is to rule the worker path out as a suspect.")
    p.add_argument("--grad_log_every", type=int, default=1000,
                   help="Per-component gradient report, on the raw pre-clip "
                        "gradient. Read the g/w column: it is the only one that "
                        "compares fairly across a pretrained DINOv2 and a fresh "
                        "projection. The load-bearing check is that the PREFIX "
                        "side is non-zero -- under knowledge insulation the "
                        "discrete head is its only gradient path. 0 = off.")
    p.add_argument("--profile_steps", type=int, default=20,
                   help="Optimizer steps to attribute across data wait / "
                        "forward / backward / optimizer, after a 2-step warmup. "
                        "Prints once and then disables itself -- each lap needs "
                        "a cuda synchronize, so the steady state must not pay "
                        "for it. 0 = off.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_episodes", type=int, default=0,
                   help="Hold out this many whole EPISODES, stratified over "
                        "dataset#task, and report the loss on them every "
                        "--val_every steps. 0 = off. Training loss alone cannot "
                        "tell an under-capacity model from an under-trained one "
                        "from an over-fitting one, which is the question every "
                        "'should I make the model bigger' decision needs "
                        "answered. 40 gives one episode per LIBERO task.")
    p.add_argument("--val_every", type=int, default=500)
    p.add_argument("--val_max_batches", type=int, default=20,
                   help="Batches per validation pass. Capped so a pass stays a "
                        "rounding error against --val_every training steps.")
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--max_episode_index", type=int, default=None)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--resume_from_checkpoint", default=None,
                   help="Local checkpoint directory OR a Hugging Face repo id. "
                        "A full resume needs training_state.pth (weights + "
                        "optimizer + step); a directory holding only "
                        "model.safetensors resumes the WEIGHTS alone.")
    p.add_argument("--start_step", dest="start_step_override", type=int, default=-1,
                   help="Override the step the LR schedule resumes at. Needed "
                        "for a weights-only checkpoint, where the step counter "
                        "did not survive and the schedule would otherwise "
                        "restart from warmup. -1 = take it from the checkpoint.")
    p.add_argument("--seed", type=int, default=42)
    train(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
