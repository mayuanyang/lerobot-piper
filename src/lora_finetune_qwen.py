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
POSE_PROMPT = (
    "Report the robot gripper's current end-effector position and whether its "
    "fingers are open or closed."
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


def pose_target(state: torch.Tensor) -> str:
    """Template an end-effector pose from observation.state.

    LIBERO's state is [xyz(3), axis-angle(3), finger_qpos(2)]. Only xyz and the
    finger aperture are templated: the axis-angle components sit near the +-pi
    wrap (dim 4 spans [-3.64, 3.56] with a q01-q99 width of only 1.06), so their
    decimal form is discontinuous for orientations that are physically adjacent
    and would teach the model a boundary that is an artefact of the encoding.
    """
    s = state.flatten().tolist()
    grip = "open" if len(s) < 7 or abs(s[6]) > 0.02 else "closed"
    return (f"gripper at x={s[0]:+.3f} y={s[1]:+.3f} z={s[2]:+.3f}, "
            f"fingers {grip}")


class LeRobotVLMDataset(torch.utils.data.Dataset):
    """Frames -> (images, prompt, target text).

    Subsamples with `frame_stride`: consecutive frames of an episode share one
    instruction and near-identical pixels, so training on all of them mostly
    buys duplicate gradients and a fast route to memorising the instruction set.
    """

    def __init__(self, ds, camera_keys, target, frame_stride, state_key):
        self.ds = ds
        self.camera_keys = camera_keys
        self.target = target
        self.state_key = state_key
        self.indices = list(range(0, len(ds), max(1, frame_stride)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        item = self.ds[self.indices[i]]
        images = [to_pil(item[k]) for k in self.camera_keys if k in item]
        task = item.get("task", "")
        if isinstance(task, (list, tuple)):
            task = task[0] if task else ""
        if self.target == "pose":
            return images, POSE_PROMPT, pose_target(item[self.state_key].float())
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

    subsets, rows, state_dims = [], [], {}
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
        sub = LeRobotVLMDataset(ds, cks, args.target, stride, state_key)
        subsets.append(sub)
        eps = getattr(ds, "num_episodes", None)
        rows.append((did, cks, scene, wrist, sd, stride, len(sub), raw, eps))

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
    for did, cks, scene, wrist, sd, stride, n, raw, eps in rows:
        print(f"  {did[:30]:<30} {len(cks):>4} {str(sd):>5} {str(eps or '?'):>8} {raw:>9,} "
              f"{stride:>6} {n:>9,} {100 * n / max(1, total):>5.1f}%")
        # Print the ROLE beside each key. Camera names do not match across
        # datasets and are never matched to each other -- they are ordered
        # scene-first so the prompt has one convention regardless of naming.
        order = "  ".join(f"{k.rsplit('.', 1)[-1]}[{'wrist' if k in wrist else 'scene'}]"
                          for k in cks)
        print(f"      {order}")
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
            print(f"step {step:6d}/{args.training_steps}  loss {run_loss / args.log_every:.4f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  grad_norm {float(gn):.2f}")
            run_loss = 0.0
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
