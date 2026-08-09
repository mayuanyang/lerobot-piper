"""
LoRA finetune Qwen3-VL on a LeRobot-format robot dataset.

WHY THIS EXISTS
---------------
WiltechsMoE keeps Qwen3-VL frozen and consumes its per-layer KV caches. Those
caches are produced by an encoder pretrained on internet images, never on
LIBERO's renderer, its prop set, or its camera poses. LoRA adapts them.

WHAT IT OPTIMISES, AND THE CATCH
--------------------------------
The VLM never generates text inside WiltechsMoE -- it is a pure feature
extractor. So a next-token objective optimises generation, which is only
*indirectly* the thing that gets consumed (the KV caches). The bet is the
standard one: if the model can produce a scene-grounded description from the
pixels, the caches it produced it from demonstrably contain that grounding.

Read the measured picture before spending GPU-hours here: on libero_spatial
task 0 the residual failures at 92% are grasp PRECISION, not object selection,
and the frozen encoder already localises both candidate bowls to ~8mm against a
150mm separation. `--target cot` sharpens selection, which is not the measured
bottleneck. `--target pose` is the option aimed at geometry. Neither is free --
see the KV-drift warning below.

TARGETS (--target)
------------------
  cot          rewrite_instruction(task) -- the descriptive/CoT string from
               task_rewrites.py. Same string the MoE will later encode, so
               success here means the caches carry that trace. (default)
  instruction  the raw task string, unrewritten.
  pose         a templated end-effector pose read from observation.state:
               "gripper at x=-0.047 y=+0.034 z=+0.765, fingers open". Aimed at
               fine geometry rather than naming. Uses no extra annotation --
               the state column is already in the dataset.

KV DRIFT -- READ BEFORE RESUMING THE MoE ON THE RESULT
------------------------------------------------------
The 92% checkpoint's experts learned ca_q projections against the ORIGINAL
encoder's feature geometry. Swapping in a LoRA-modified VLM moves every KV the
experts cross-attend to, so resuming is NOT a warm start for the cross-attention
-- expect a large transient and possibly a partial re-learn. Start at low rank
(8) and low alpha, and use --report_kv_drift to measure how far the caches
actually moved before committing to a long MoE run.

USAGE
-----
    python src/lora_finetune_qwen.py \
        --dataset_id lerobot/libero \
        --output_dir ./outputs/qwen_lora \
        --target cot --frame_stride 10 \
        --lora_r 8 --batch_size 4 --training_steps 4000

    # then bake the adapter into a standalone model directory
    python src/lora_finetune_qwen.py --merge_and_save ./outputs/qwen_lora \
        --merged_dir ./outputs/qwen_lora_merged

    # and point the MoE trainer at it
    python src/train_wiltechs_moe.py --vlm_model_id ./outputs/qwen_lora_merged ...
"""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)

from models.wiltechs_vla.task_rewrites import rewrite_instruction  # noqa: E402

VLM_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

PROMPT = (
    "Describe the scene and the manipulation task, naming the target object and "
    "how to tell it apart from the other objects present."
)
# The question has to match the answer format, or the model is being asked one
# thing and graded on another.
POSE_PROMPT_BINARY = (
    "Report the robot gripper's current end-effector position and whether its "
    "fingers are open or closed."
)
POSE_PROMPT_NUMERIC = (
    "Report the robot gripper's current end-effector position and how far apart "
    "its fingers are."
)


# Same convention the MoE trainer uses to auto-detect the wrist view
# (--robot_cnn_wrist_only), kept identical so the two never disagree about which
# camera is which.
WRIST_HINTS = ("image2", "wrist", "gripper", "eye_in_hand", "hand")


def is_wrist_cam(key: str) -> bool:
    return any(h in key.rsplit(".", 1)[-1].lower() for h in WRIST_HINTS)


def order_cameras(keys, max_cameras=0):
    """Order cameras by ROLE -- scene views first, wrist views last.

    Datasets do not agree on naming, and sorting alphabetically makes the
    ordering incidental rather than meaningful: 'top' before 'wrist' puts the
    scene first, but 'gripper' before 'top', 'hand' before 'side', and
    'eye_in_hand' before 'front' all put the WRIST first. Half the common
    conventions invert. Since the images enter the prompt in list order, that
    hands the model a convention that flips per dataset for no reason.

    max_cameras caps the count so a 4-camera dataset does not contribute 4x the
    image tokens of a 1-camera one; when it bites, one scene and one wrist view
    are kept first so neither role is dropped entirely.

    Returns (ordered_keys, scene_keys, wrist_keys).
    """
    scene = sorted(k for k in keys if not is_wrist_cam(k))
    wrist = sorted(k for k in keys if is_wrist_cam(k))
    ordered = scene + wrist
    if max_cameras > 0 and len(ordered) > max_cameras:
        keep = []
        for k in (scene[:1] + wrist[:1] + scene[1:] + wrist[1:]):
            if len(keep) >= max_cameras:
                break
            if k not in keep:
                keep.append(k)
        ordered = [k for k in ordered if k in keep]
        scene = [k for k in scene if k in ordered]
        wrist = [k for k in wrist if k in ordered]
    return ordered, scene, wrist


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def to_pil(img: torch.Tensor) -> Image.Image:
    """LeRobot hands back (C,H,W) float in [0,1]; the processor wants PIL."""
    if img.dim() == 4:
        img = img[-1]
    arr = (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def frame_episode_indices(ds):
    """Per-frame episode index, or None if this build will not surface it.

    A validation split MUST be made on episodes, never on frames. Neighbouring
    frames of one episode differ by 0.1s and are all but identical, so a frame
    split puts near-copies of training samples in the validation set: val loss
    then tracks train loss down forever and reports no overfitting no matter how
    long the run goes -- which is precisely the failure it exists to detect.
    Returning None here is treated as fatal rather than silently degrading.
    """
    hf = getattr(getattr(ds, "reader", None), "hf_dataset", None)
    if hf is None:
        hf = getattr(ds, "hf_dataset", None)
    if hf is None:
        return None
    try:
        return np.asarray(hf.data.column("episode_index").to_numpy())
    except Exception:
        pass
    try:
        return np.asarray(hf["episode_index"])
    except Exception:
        return None


def split_episodes(ep_ids, val_spec, seed):
    """Choose validation episodes deterministically.

    val_spec < 1 is a fraction of episodes, >= 1 an absolute count. Seeded so a
    restart or a resume reuses the same split -- a reshuffled split would move
    the validation set into what the model has already trained on and quietly
    flatter every later number.
    """
    uniq = np.unique(ep_ids)
    n_val = int(round(len(uniq) * val_spec)) if val_spec < 1 else int(val_spec)
    n_val = max(1, min(n_val, len(uniq) - 1))
    order = np.random.default_rng(seed).permutation(len(uniq))
    return set(uniq[order[:n_val]].tolist())


GRIPPER_DIM = 6


def gripper_threshold_from_stats(stats, state_key="observation.state"):
    """Midpoint of the gripper dim's two modes, read from the dataset's own stats.

    The finger position is bimodal -- on LIBERO the modes sit at ~0.011 (closed)
    and ~0.040 (open) -- so the open/closed cut belongs at their midpoint. A
    hardcoded constant does not survive a change of robot or of units, and gets
    it wrong even here: 0.02 labelled 31% of frames closed, while the ACTION
    column (bimodal at -1.0/+0.9194, mean -0.0496) independently implies 50.5%.
    That is ~19% of frames carrying the wrong word for the one categorical token
    in the target.

    q10/q90 rather than min/max: the tails are where stray frames live, and
    q01-q10 on this column spans only 0.002 while q10-q50 spans 0.014.
    Returns None if the stats cannot supply it, leaving the caller to decide.
    """
    st = (stats or {}).get(state_key) or {}
    lo, hi = st.get("q10"), st.get("q90")
    try:
        if lo is not None and hi is not None and len(lo) > GRIPPER_DIM:
            return (float(lo[GRIPPER_DIM]) + float(hi[GRIPPER_DIM])) / 2.0
    except (TypeError, ValueError, IndexError):
        pass
    return None


def pose_target_numeric(state: torch.Tensor) -> str:
    """Same pose, with the finger APERTURE as a number instead of open/closed.

    How far the fingers are apart is set by the width of whatever is being
    held, so it is not two states but a continuum, and the free-space modes
    (~0.011 closed, ~0.040 open on LIBERO) are only its endpoints. A binary cut
    at their midpoint therefore lands exactly where a grasped object puts the
    aperture, and the label flips on millimetre changes during the grasp --
    contradictory supervision on the one phase that is still failing.

    Reporting the number removes the cut entirely and keeps the object-width
    signal the binary form discards. Aperture is the finger GAP where both
    finger dims exist, since they mirror each other.
    """
    s = state.flatten().tolist()
    if len(s) > GRIPPER_DIM + 1:
        gap = abs(s[GRIPPER_DIM] - s[GRIPPER_DIM + 1])
    elif len(s) > GRIPPER_DIM:
        gap = abs(s[GRIPPER_DIM])
    else:
        gap = 0.0
    return (f"gripper at x={s[0]:+.3f} y={s[1]:+.3f} z={s[2]:+.3f}, "
            f"fingers {gap:.3f} apart")


def pose_target(state: torch.Tensor, grip_threshold: float) -> str:
    """Template an end-effector pose from observation.state.

    LIBERO's state is [xyz(3), axis-angle(3), finger_qpos(2)]. Only xyz and the
    finger aperture are templated: the axis-angle components sit near the +-pi
    wrap (dim 4 spans [-3.64, 3.56] with a q01-q99 width of only 1.06), so their
    decimal form is discontinuous for orientations that are physically adjacent
    and would teach the model a boundary that is an artefact of the encoding.
    """
    s = state.flatten().tolist()
    grip = ("open" if len(s) <= GRIPPER_DIM or abs(s[GRIPPER_DIM]) > grip_threshold
            else "closed")
    return (f"gripper at x={s[0]:+.3f} y={s[1]:+.3f} z={s[2]:+.3f}, "
            f"fingers {grip}")


class LeRobotVLMDataset(torch.utils.data.Dataset):
    """Frames -> (images, prompt, target text).

    Subsamples with `frame_stride`: consecutive frames of an episode share one
    instruction and near-identical pixels, so training on all of them mostly
    buys duplicate gradients and a fast route to memorising the instruction set.
    """

    def __init__(self, ds, camera_keys, target, frame_stride, state_key,
                 grip_threshold=0.02, grip_format="numeric", ds_id="",
                 decode_retries=8, ep_ids=None, keep_episodes=None):
        self.ds = ds
        self.ds_id = ds_id
        self.grip_format = grip_format
        self.decode_retries = decode_retries
        self._decode_failures = 0
        self.camera_keys = camera_keys
        self.target = target
        self.state_key = state_key
        self.grip_threshold = grip_threshold
        idx = range(0, len(ds), max(1, frame_stride))
        if keep_episodes is not None and ep_ids is not None:
            self.indices = [i for i in idx if int(ep_ids[i]) in keep_episodes]
        else:
            self.indices = list(idx)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        # A single undecodable video packet used to kill the whole run from
        # inside a DataLoader worker, hours in. One corrupt frame out of tens of
        # thousands is not a reason to lose the training; substitute a
        # neighbouring frame, which is an equally valid self-contained sample
        # (its own images, own state, own task). Failures are counted and
        # announced so a systematically broken dataset still cannot pass as
        # healthy.
        item = None
        for attempt in range(self.decode_retries + 1):
            j = self.indices[(i + attempt) % len(self.indices)]
            try:
                item = self.ds[j]
                break
            except Exception as exc:
                self._decode_failures += 1
                if self._decode_failures in (1, 10, 100, 1000, 10000):
                    print(f"[decode] {self.ds_id}: frame {j} unreadable "
                          f"({type(exc).__name__}: {str(exc)[:90]}); using a neighbour. "
                          f"{self._decode_failures} failure(s) so far in this worker.",
                          flush=True)
        if item is None:
            raise RuntimeError(
                f"{self.ds_id}: {self.decode_retries + 1} consecutive frames from index "
                f"{i} were undecodable. That is a broken dataset, not a stray bad packet.")
        images = [to_pil(item[k]) for k in self.camera_keys if k in item]
        task = item.get("task", "")
        if isinstance(task, (list, tuple)):
            task = task[0] if task else ""
        if self.target == "pose":
            st = item[self.state_key].float()
            if self.grip_format == "numeric":
                return images, POSE_PROMPT_NUMERIC, pose_target_numeric(st)
            return images, POSE_PROMPT_BINARY, pose_target(st, self.grip_threshold)
        if self.target == "cot":
            return images, PROMPT, rewrite_instruction(str(task))
        return images, PROMPT, str(task)


def make_collate(processor, max_len):
    def collate(batch):
        images, prompts, targets = zip(*batch)
        prompt_texts, full_texts, img_lists = [], [], []
        for imgs, prompt, tgt in zip(images, prompts, targets):
            content = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": prompt}]
            msg = [{"role": "user", "content": content}]
            # add_generation_prompt=True gives the exact prefix the answer
            # follows, so its token length is the boundary to mask at.
            prefix = processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True)
            prompt_texts.append(prefix)
            full_texts.append(prefix + tgt + processor.tokenizer.eos_token)
            img_lists.append(list(imgs))

        full = processor(text=full_texts, images=img_lists, return_tensors="pt",
                         padding=True, truncation=True, max_length=max_len)
        # Tokenised separately with the SAME images, so image-token expansion is
        # identical and the prompt is a genuine prefix of `full`.
        prompt_only = processor(text=prompt_texts, images=img_lists,
                                return_tensors="pt", padding=True,
                                truncation=True, max_length=max_len)

        labels = full["input_ids"].clone()
        pad_id = processor.tokenizer.pad_token_id
        for b in range(labels.shape[0]):
            n_prompt = int(prompt_only["attention_mask"][b].sum())
            # Left padding would put the prompt at the END of the row, so mask
            # by counting real tokens from whichever side the padding is on.
            if processor.tokenizer.padding_side == "left":
                start = int(labels.shape[1] - full["attention_mask"][b].sum())
                labels[b, : start + n_prompt] = -100
            else:
                labels[b, :n_prompt] = -100
        labels[full["attention_mask"] == 0] = -100
        if pad_id is not None:
            labels[full["input_ids"] == pad_id] = -100
        full["labels"] = labels
        return full

    return collate


# The two towers do NOT share Linear naming, and matching only the text names
# against the vision tower silently selects nothing:
#   Qwen3VLTextAttention    q_proj, k_proj, v_proj, o_proj
#   Qwen3VLVisionAttention  qkv, proj              <- fused QKV, different names
#   Qwen3VLTextMLP          gate_proj, up_proj, down_proj
#   Qwen3VLVisionMLP        linear_fc1, linear_fc2 (same for the patch merger)
# Because the text names still match, the module list comes back non-empty and
# the run proceeds with the vision tower untouched.
TEXT_ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
TEXT_MLP = ("gate_proj", "up_proj", "down_proj")
VIS_ATTN = ("qkv", "proj")
VIS_MLP = ("linear_fc1", "linear_fc2")


def build_lora(model, args):
    from peft import LoraConfig, get_peft_model

    text_names = set(TEXT_ATTN) | (set(TEXT_MLP) if args.lora_mlp else set())
    vis_names = set(VIS_ATTN) | (set(VIS_MLP) if args.lora_mlp else set())

    def is_visual(name: str) -> bool:
        return ".visual." in name or name.startswith("visual.")

    text_mods, vis_mods = set(), set()
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        # Exact LEAF match, not endswith: "q_proj".endswith("proj") is True, so a
        # suffix test would pull every text projection in under the vision rule.
        leaf = name.rsplit(".", 1)[-1]
        if is_visual(name):
            if args.lora_vision and leaf in vis_names:
                vis_mods.add(name)
        elif leaf in text_names:
            text_mods.add(name)

    modules = sorted(text_mods | vis_mods)
    if not modules:
        raise RuntimeError("No LoRA target modules matched.")
    if args.lora_vision and not vis_mods:
        raise RuntimeError(
            "--lora_vision matched ZERO modules in the vision tower. Its Linear "
            f"names are expected to be {VIS_ATTN + VIS_MLP}; this build must differ. "
            "Refusing to train, because the text tower still matched and the run "
            "would look normal while adapting no vision at all.")

    cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=modules,
    )
    model = get_peft_model(model, cfg)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"LoRA: r={args.lora_r} alpha={args.lora_alpha}  mlp={'yes' if args.lora_mlp else 'no'}")
    print(f"  language tower: {len(text_mods)} modules")
    print(f"  vision tower  : {len(vis_mods)} modules"
          + ("" if args.lora_vision else "   (--lora_vision not set)"))
    print(f"  trainable: {n_train:,} / {n_all:,} ({100 * n_train / n_all:.3f}%)")
    # The MoE reads the LANGUAGE layers' KV caches, and vision tokens occupy
    # positions inside those same caches (the x-attn diagnostic counts ~128 vis
    # vs ~96 lang). So adapting the text tower alone already changes how vision
    # is represented in exactly the tensors the experts consume; --lora_vision
    # additionally changes the patch features before they enter that sequence.
    return model


@torch.no_grad()
def evaluate(model, loader, device, max_batches):
    """Token-weighted mean loss on held-out EPISODES.

    Weighted by supervised-token count, not batch count: HF returns a per-token
    mean within each batch, so averaging those directly would over-weight short
    batches. Without this number, training loss falls forever and there is no
    way to tell learned geometry from memorised coordinates -- which matter very
    differently for the KV caches the MoE will consume.
    """
    was_training = model.training
    model.eval()
    tot, ntok = 0.0, 0
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        n = int((batch["labels"] != -100).sum())
        if n == 0:
            continue
        tot += float(model(**batch).loss) * n
        ntok += n
    if was_training:
        model.train()
    return tot / ntok if ntok else float("nan")


@torch.no_grad()
def report_kv_drift(model, loader, device, image_token_id,
                    n_batches=4, n_bands=4, layers_per_band=9):
    """How far the LoRA moved the hidden states the MoE's experts read.

    Reported per EXPERT BAND and split by VISION vs LANGUAGE token positions,
    because a single scalar cannot answer the question that matters. Each expert
    cross-attends to its own contiguous block of VLM layers, so drift
    concentrated in the deep layers hits E3 and leaves E0 nearly untouched. And
    vision and language occupy positions in the SAME per-layer KV cache (the
    MoE's x-attn diagnostic counts ~128 vis vs ~96 lang), so adapting both
    towers can move one and not the other.

    The baseline comes from peft's disable_adapter() rather than a second loaded
    model: same weights, adapter off, so the comparison is exact and costs no
    extra memory -- a second 4B copy would be ~8GB of bf16 for nothing.
    """
    model.eval()
    stats = {}  # layer -> [vis_num, vis_den, lang_num, lang_den]
    for bi, batch in enumerate(loader):
        if bi >= n_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        batch.pop("labels", None)
        with model.disable_adapter():
            hb = model(**batch, output_hidden_states=True).hidden_states
        ht = model(**batch, output_hidden_states=True).hidden_states

        vis = (batch["input_ids"] == image_token_id)
        real = batch["attention_mask"].bool()
        lang = real & ~vis
        for li, (a, b) in enumerate(zip(hb, ht)):
            a = a.float(); b = b.float()
            d2 = (a - b).pow(2).sum(-1)
            a2 = a.pow(2).sum(-1)
            s = stats.setdefault(li, [0.0, 0.0, 0.0, 0.0])
            s[0] += float(d2[vis].sum()); s[1] += float(a2[vis].sum())
            s[2] += float(d2[lang].sum()); s[3] += float(a2[lang].sum())

    def rel(num, den):
        return math.sqrt(num / den) if den > 0 else float("nan")

    # hidden_states[0] is the embedding output; layer i is at index i+1.
    print(f"\n  {'expert band':<22} {'vision':>9} {'language':>9}")
    worst = 0.0
    for e in range(n_bands):
        lo, hi = 1 + e * layers_per_band, (e + 1) * layers_per_band
        agg = [0.0, 0.0, 0.0, 0.0]
        for li in range(lo, hi + 1):
            if li in stats:
                for j in range(4):
                    agg[j] += stats[li][j]
        v, l = rel(agg[0], agg[1]), rel(agg[2], agg[3])
        worst = max(worst, max(x for x in (v, l) if not math.isnan(x)) if (agg[1] or agg[3]) else 0.0)
        print(f"  E{e} (VLM layers {lo - 1:>2}-{hi - 1:<2})    {v:>9.4f} {l:>9.4f}")
    return worst


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_id", type=str, nargs="+", default=["lerobot/libero"])
    p.add_argument("--output_dir", type=str, default="./outputs/qwen_lora")
    p.add_argument("--vlm_model_id", type=str, default=VLM_MODEL_ID)
    p.add_argument("--target", type=str, default="cot",
                   choices=["cot", "instruction", "pose"],
                   help="What the model is trained to produce. See module docstring.")
    p.add_argument("--frame_stride", type=int, nargs="+", default=[10],
                   help="Keep 1 frame in N. Neighbouring frames share an instruction "
                        "and nearly the same pixels, so a stride of 1 mostly buys "
                        "duplicate gradients. Pass one value, or one per dataset to "
                        "rebalance the mix -- ConcatDataset samples proportionally to "
                        "size, so a 10x larger dataset otherwise supplies 10x the "
                        "gradient.")
    p.add_argument("--val_episodes", type=float, default=0.05,
                   help="Held-out EPISODES for validation: <1 a fraction, >=1 a count, "
                        "0 disables. Split is on episodes, never frames -- neighbouring "
                        "frames differ by 0.1s and are near copies, so a frame split "
                        "makes val loss track train loss forever and detect nothing.")
    p.add_argument("--val_every", type=int, default=200,
                   help="Steps between validation passes.")
    p.add_argument("--val_batches", type=int, default=20,
                   help="Validation batches per pass.")
    p.add_argument("--val_seed", type=int, default=0,
                   help="Seeds the episode split. Fixed so a restart reuses the same "
                        "held-out set; a reshuffle would move validation onto episodes "
                        "the model already trained on.")
    p.add_argument("--gripper_format", type=str, default="numeric",
                   choices=["numeric", "binary"],
                   help="How --target pose reports the fingers. numeric (default): "
                        "'fingers 0.027 apart'. binary: 'fingers open/closed', which "
                        "needs a cut that a grasped object sits right on top of, so "
                        "the label flips during the grasp -- see pose_target_numeric.")
    p.add_argument("--decode_retries", type=int, default=8,
                   help="Neighbouring frames to try when a video packet fails to "
                        "decode, before treating the dataset as broken.")
    p.add_argument("--gripper_threshold", type=float, default=None,
                   help="Open/closed cut on observation.state[6] for --target pose. "
                        "Default: auto, the midpoint of that dim's q10/q90 in the "
                        "dataset's own stats -- the column is bimodal and a hardcoded "
                        "constant does not survive a change of robot or units.")
    p.add_argument("--balance", action="store_true",
                   help="Auto-pick per-dataset strides so every dataset contributes "
                        "roughly equally. ConcatDataset samples proportionally to "
                        "length, so without this the largest dataset IS the run. "
                        "Overrides --frame_stride; reference is the smallest dataset "
                        "at stride 1, since upsampling is not possible here.")
    p.add_argument("--max_cameras", type=int, default=0,
                   help="Cap cameras per dataset (0 = all). Keeps a 4-camera dataset "
                        "from contributing 4x the image tokens of a 1-camera one. When "
                        "it bites, one scene and one wrist view are kept first.")
    p.add_argument("--cameras", type=str, nargs="+", default=None,
                   help="Camera keys to feed the VLM. Default: all detected, resolved "
                        "PER DATASET -- names need not match across datasets, since the "
                        "VLM only receives a list of images.")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--training_steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_vision", action="store_true",
                   help="Also adapt the vision tower. Moves the KV caches hardest; "
                        "read the KV-drift note in the module docstring first.")
    p.add_argument("--lora_mlp", action="store_true",
                   help="Also adapt gate/up/down projections, not just attention.")
    p.add_argument("--drift_bands", type=int, default=4,
                   help="Expert bands to report drift over. Match --num_experts.")
    p.add_argument("--drift_layers_per_band", type=int, default=9,
                   help="VLM layers each expert reads. Match --expert_num_layers.")
    p.add_argument("--report_kv_drift", action="store_true",
                   help="After training, measure relative hidden-state drift vs the "
                        "base model -- the size of the shock a resumed MoE absorbs.")
    p.add_argument("--merge_and_save", type=str, default=None,
                   help="Adapter dir to bake into a standalone model. Skips training.")
    p.add_argument("--merged_dir", type=str, default=None,
                   help="Where --merge_and_save writes. Default: <adapter>_merged")
    args = p.parse_args()

    device = pick_device()

    # ── merge-only path ──────────────────────────────────────────────────
    if args.merge_and_save:
        from peft import PeftModel
        out = args.merged_dir or (args.merge_and_save.rstrip("/") + "_merged")
        print(f"Loading base {args.vlm_model_id} ...")
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            args.vlm_model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        print(f"Applying adapter {args.merge_and_save} ...")
        merged = PeftModel.from_pretrained(base, args.merge_and_save).merge_and_unload()
        Path(out).mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(out)
        # The processor must travel with it: WiltechsMoETransformer does
        # AutoProcessor.from_pretrained(<same id>), so a model dir without one
        # fails at load rather than silently falling back.
        AutoProcessor.from_pretrained(args.vlm_model_id).save_pretrained(out)
        print(f"\nMerged model written to {out}")
        print(f"Use it with:  python src/train_wiltechs_moe.py --vlm_model_id {out} ...")
        return

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ── data ─────────────────────────────────────────────────────────────
    print(f"Loading dataset(s): {args.dataset_id}")
    state_key = "observation.state"

    strides = args.frame_stride
    if len(strides) == 1:
        strides = strides * len(args.dataset_id)
    elif len(strides) != len(args.dataset_id):
        raise ValueError(
            f"--frame_stride takes 1 value or one per dataset, got {len(strides)} "
            f"for {len(args.dataset_id)} datasets")

    # Load first, decide strides second: --balance needs every raw frame count
    # before it can pick any of them.
    loaded = []
    for did in args.dataset_id:
        ds = LeRobotDataset(did, force_cache_sync=True, revision="main")
        loaded.append((did, ds, len(ds)))

    if args.balance:
        # ConcatDataset samples proportionally to length, so an equal mix means
        # equal post-stride counts. Reference is the SMALLEST dataset at stride
        # 1 -- anything larger would need to upsample, which this cannot do.
        target = min(n for _, _, n in loaded)
        strides = [max(1, round(n / target)) for _, _, n in loaded]
        print(f"--balance: equalising to the smallest dataset ({target:,} frames); "
              f"--frame_stride overridden -> {strides}")

    subsets, val_subsets, rows, state_dims = [], [], [], {}
    for (did, ds, raw), stride in zip(loaded, strides):
        # Camera keys are resolved PER DATASET and deliberately not required to
        # match. The VLM is handed a list of images and nothing else -- unlike
        # the MoE, where a camera's identity fixes its slot in the DiT sequence,
        # here the key name carries no meaning. Requiring identical names was
        # the one thing blocking a LIBERO + community-dataset mix.
        avail = [k for k in ds.meta.features if k.startswith("observation.images.")]
        if not avail:
            raise ValueError(f"{did}: no observation.images.* features found")
        if args.cameras:
            # An explicit list is global, so a key absent from THIS dataset used
            # to be filtered out silently in __getitem__ -- and a sample whose
            # every key missed produced an image-free prompt, training the VLM
            # to answer from text alone while looking entirely normal.
            missing = [c for c in args.cameras if c not in avail]
            if missing:
                raise ValueError(
                    f"{did}: --cameras {missing} not present. Available: {sorted(avail)}. "
                    f"Drop --cameras to auto-select per dataset (names need not match).")
            sel = list(args.cameras)
        else:
            sel = avail
        cks, scene, wrist = order_cameras(sel, args.max_cameras)
        sd = ds.meta.features.get(state_key, {}).get("shape", [None])[-1]
        state_dims[did] = sd
        grip_t = args.gripper_threshold
        if grip_t is None:
            grip_t = gripper_threshold_from_stats(getattr(ds.meta, "stats", None), state_key)
        if grip_t is None:
            grip_t = 0.02
            if args.target == "pose":
                print(f"  WARNING {did}: no q10/q90 for {state_key}; falling back to a "
                      f"hardcoded gripper threshold of {grip_t}. Check the open/closed "
                      f"split below -- a misplaced cut mislabels the one categorical "
                      f"token in every target.")
        keep_tr = keep_va = None
        ep_ids = None
        if args.val_episodes > 0:
            ep_ids = frame_episode_indices(ds)
            if ep_ids is None:
                raise RuntimeError(
                    f"{did}: cannot read per-frame episode_index, so a leak-free "
                    f"validation split is impossible. Splitting on frames instead "
                    f"would make val loss meaningless (neighbouring frames are near "
                    f"copies). Pass --val_episodes 0 to train without validation.")
            keep_va = split_episodes(ep_ids, args.val_episodes, args.val_seed)
            keep_tr = set(np.unique(ep_ids).tolist()) - keep_va

        def _mk(keep):
            return LeRobotVLMDataset(ds, cks, args.target, stride, state_key, grip_t,
                                     args.gripper_format, did, args.decode_retries,
                                     ep_ids, keep)
        sub = _mk(keep_tr)
        subsets.append(sub)
        if keep_va is not None:
            val_subsets.append(_mk(keep_va))
        eps = getattr(ds, "num_episodes", None)
        rows.append((did, cks, scene, wrist, sd, stride, len(sub), raw, eps, grip_t,
                     len(val_subsets[-1]) if keep_va is not None else 0,
                     len(keep_va) if keep_va is not None else 0))

    # pose templates observation.state positionally as LIBERO's
    # [xyz(3), axis-angle(3), fingers(2)]. On a dataset whose state is joint
    # angles, s[0:3] are joint positions and every target would be confidently
    # wrong -- a silent label corruption, not a crash.
    if args.target == "pose":
        dims = set(state_dims.values())
        if len(dims) > 1:
            raise ValueError(
                f"--target pose reads observation.state positionally as LIBERO's "
                f"[xyz(3), axis-angle(3), fingers(2)], but the datasets disagree on "
                f"state dim: {state_dims}. Mixing them would produce confidently "
                f"wrong targets with no error. Train pose on one layout at a time.")
        if dims and next(iter(dims)) not in (7, 8):
            raise ValueError(
                f"--target pose expects an 8-dim LIBERO-style state (or 7), got "
                f"{next(iter(dims))}. Check the layout before trusting the targets.")
        print(f"--target pose: reading state dim {next(iter(dims), '?')} as "
              f"[xyz(3), axis-angle(3), fingers(2)]; only xyz + finger aperture "
              f"are templated.")

    # `used` is ceil(raw / stride) -- the raw column is here so the number can be
    # checked against what the dataset is known to contain, rather than only
    # being back-computable by multiplying by the stride.
    print(f"\n  {'dataset':<30} {'cams':>4} {'state':>5} {'episodes':>8} {'raw':>9} "
          f"{'stride':>6} {'used':>9} {'mix':>6}")
    total = sum(r[6] for r in rows)
    for ri, (did, cks, scene, wrist, sd, stride, n, raw, eps, grip_t,
             n_val, n_val_eps) in enumerate(rows):
        print(f"  {did[:30]:<30} {len(cks):>4} {str(sd):>5} {str(eps or '?'):>8} {raw:>9,} "
              f"{stride:>6} {n:>9,} {100 * n / max(1, total):>5.1f}%")
        # Print the ROLE beside each key. Camera names do not match across
        # datasets and are never matched to each other -- they are ordered
        # scene-first so the prompt has one convention regardless of naming.
        order = "  ".join(f"{k.rsplit('.', 1)[-1]}[{'wrist' if k in wrist else 'scene'}]"
                          for k in cks)
        print(f"      {order}")
        if args.target == "pose" and args.gripper_format == "binary":
            # Sample the labels this cut actually produces instead of trusting
            # it. The gripper column is bimodal and roughly balanced in these
            # demos, so a split far from 50/50 means the cut landed inside a
            # mode -- which is exactly how a hardcoded 0.02 read 31/69 here.
            sub = subsets[ri]
            step = max(1, len(sub) // 400)
            idxs = range(0, len(sub), step)
            opened = sum(
                1 for i in idxs
                if abs(sub.ds[sub.indices[i]][state_key].float().flatten().tolist()
                       [GRIPPER_DIM]) > grip_t)
            pct = 100 * opened / max(1, len(idxs))
            flag = "   <-- far from 50/50, check the cut" if not 30 <= pct <= 70 else ""
            print(f"      gripper cut {grip_t:.4f} -> {pct:.1f}% open / "
                  f"{100 - pct:.1f}% closed  (n={len(idxs)}){flag}")
        if n_val:
            print(f"      val split: {n_val_eps} held-out episodes -> {n_val:,} frames "
                  f"({100 * n_val / max(1, n + n_val):.1f}% of this dataset)")
        if not scene:
            print("      WARNING: every camera classified as WRIST. A scene view is "
                  "what a 'describe the scene' target needs.")
    # ConcatDataset + shuffle samples proportionally to size, so the mix column
    # IS the sampling distribution. A dataset 10x the size of another silently
    # supplies 10x the gradient; tune it with per-dataset --frame_stride or
    # --balance.
    print(f"  {'TOTAL':<30} {'':>4} {'':>5} {'':>8} {'':>9} {'':>6} {total:>9,}")
    if len(rows) > 1:
        top = max(rows, key=lambda r: r[6])
        share = 100 * top[6] / max(1, total)
        if share >= 80.0:
            print(f"\n  WARNING: {top[0]} is {share:.1f}% of the sampling distribution. "
                  f"The other {len(rows) - 1} dataset(s) contribute "
                  f"{100 - share:.1f}% of the gradient between them, so this is close "
                  f"to a single-dataset run. Use --balance, or raise that dataset's "
                  f"--frame_stride.")
    print()
    dataset = subsets[0] if len(subsets) == 1 else torch.utils.data.ConcatDataset(subsets)

    processor = AutoProcessor.from_pretrained(args.vlm_model_id)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=(device == "cuda"),
        collate_fn=make_collate(processor, args.max_len),
    )

    val_loader = None
    if val_subsets:
        val_ds = (val_subsets[0] if len(val_subsets) == 1
                  else torch.utils.data.ConcatDataset(val_subsets))
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=max(1, args.num_workers // 2), drop_last=False,
            pin_memory=(device == "cuda"),
            collate_fn=make_collate(processor, args.max_len))
        print(f"Validation: {len(val_ds):,} frames from held-out episodes "
              f"({100 * len(val_ds) / (len(dataset) + len(val_ds)):.1f}% of all frames), "
              f"every {args.val_every} steps over {args.val_batches} batches\n")

    # Show one fully-assembled example. A silently-empty or mis-masked target is
    # the failure mode that looks exactly like normal training.
    sample = next(iter(loader))
    n_sup = int((sample["labels"] != -100).sum())
    print(f"\n--- sample batch ---")
    print(f"  input_ids {tuple(sample['input_ids'].shape)}   "
          f"supervised tokens: {n_sup} / {sample['labels'].numel()}")
    row0 = sample["labels"][0]
    print(f"  target[0]: {processor.tokenizer.decode(row0[row0 != -100])!r}")
    if n_sup == 0:
        raise RuntimeError("No supervised tokens -- every label is masked. "
                           "The prompt/target split is wrong; fix before training.")
    print("--- end sample ---\n")

    # ── model ────────────────────────────────────────────────────────────
    print(f"Loading {args.vlm_model_id} ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.vlm_model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = build_lora(model, args)
    model.to(device)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = get_cosine_schedule_with_warmup(opt, args.warmup_steps, args.training_steps)

    print(f"Training {args.training_steps} steps on {device}, "
          f"batch {args.batch_size} x accum {args.grad_accum}\n")
    step, run_loss, it = 0, 0.0, iter(loader)
    best_val, best_step = None, 0
    while step < args.training_steps:
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            run_loss += float(loss) * args.grad_accum
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        step += 1

        if step % args.log_every == 0:
            # run_loss accumulates ONE entry per micro-batch, i.e. grad_accum
            # entries per optimiser step -- so the divisor is the micro-batch
            # count, not the step count. Dividing by log_every alone reported
            # grad_accum x the true loss.
            n_micro = args.log_every * args.grad_accum
            print(f"step {step:6d}/{args.training_steps}  loss {run_loss / n_micro:.4f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  grad_norm {float(gn):.2f}")
            run_loss = 0.0
        if val_loader is not None and (step % args.val_every == 0
                                       or step == args.training_steps):
            v = evaluate(model, val_loader, device, args.val_batches)
            gap = ""
            if best_val is None or v < best_val - 1e-4:
                best_val, best_step = v, step
            else:
                gap = f"   (best {best_val:.4f} @ {best_step}; {step - best_step} steps ago)"
            print(f"  val {v:.4f}{gap}")
        if step % args.save_every == 0 or step == args.training_steps:
            out = Path(args.output_dir) / f"checkpoint-{step}"
            model.save_pretrained(out)
            print(f"  saved adapter -> {out}")

    final = Path(args.output_dir) / "final"
    model.save_pretrained(final)
    print(f"\nAdapter saved to {final}")

    if args.report_kv_drift:
        print("\nMeasuring KV drift vs the base encoder (adapter disabled) ...")
        img_tok = getattr(model.config, "image_token_id", None)
        if img_tok is None:
            img_tok = getattr(getattr(model, "base_model", model).config, "image_token_id", None)
        if img_tok is None:
            print("  skipped: could not resolve image_token_id, so vision and "
                  "language positions cannot be told apart.")
        else:
            worst = report_kv_drift(model, loader, device, img_tok,
                                    n_bands=args.drift_bands,
                                    layers_per_band=args.drift_layers_per_band)
            print(f"\n  worst band/tower drift: {worst:.4f}")
            print("  This is the shock a resumed MoE's cross-attention must absorb: each")
            print("  expert's ca_q was fit against the ORIGINAL geometry of ITS band.")
            print("  Above ~0.3 in a band, expect that expert to need real re-adaptation")
            print("  rather than a warm start. Drift concentrated in the deep bands hits")
            print("  E3 hardest; drift on the vision positions is what --lora_vision buys.")

    print(f"\nNext:\n"
          f"  python src/lora_finetune_qwen.py --merge_and_save {final} \\\n"
          f"      --merged_dir {args.output_dir}/merged\n"
          f"  python src/train_wiltechs_moe.py --vlm_model_id {args.output_dir}/merged ...")


if __name__ == "__main__":
    main()
