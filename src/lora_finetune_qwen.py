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


def build_lora(model, args):
    from peft import LoraConfig, get_peft_model

    targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if args.lora_mlp:
        targets += ["gate_proj", "up_proj", "down_proj"]

    # By default LoRA touches the LANGUAGE tower only. The vision tower is what
    # produces the patch features the experts ultimately read, so adapting it
    # moves the KV caches hardest -- opt in deliberately with --lora_vision.
    def in_scope(name: str) -> bool:
        is_visual = ".visual." in name or name.startswith("visual.")
        return args.lora_vision or not is_visual

    modules = sorted({
        name for name, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear)
        and any(name.endswith(t) for t in targets)
        and in_scope(name)
    })
    if not modules:
        raise RuntimeError(f"No LoRA target modules matched {targets}")

    cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=modules,
    )
    model = get_peft_model(model, cfg)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"LoRA: r={args.lora_r} alpha={args.lora_alpha} on {len(modules)} modules "
          f"(vision={'yes' if args.lora_vision else 'no'}, mlp={'yes' if args.lora_mlp else 'no'})")
    print(f"Trainable: {n_train:,} / {n_all:,} ({100 * n_train / n_all:.3f}%)")
    return model


@torch.no_grad()
def report_kv_drift(base, tuned, processor, loader, device, n_batches=4):
    """How far the LoRA moved the hidden states the MoE's experts read.

    The experts' ca_q projections were fit against the ORIGINAL geometry, so
    this number is the size of the shock a resumed MoE run has to absorb. A
    relative drift near 1.0 means the caches are unrecognisable and the
    cross-attention is effectively re-initialised.
    """
    base.eval(); tuned.eval()
    num, den, seen = 0.0, 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        batch.pop("labels", None)
        hb = base(**batch, output_hidden_states=True).hidden_states
        ht = tuned(**batch, output_hidden_states=True).hidden_states
        for a, b in zip(hb, ht):
            num += float((a.float() - b.float()).pow(2).sum())
            den += float(a.float().pow(2).sum())
        seen += 1
        if seen >= n_batches:
            break
    if den == 0:
        return float("nan")
    return math.sqrt(num / den)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_id", type=str, nargs="+", default=["lerobot/libero"])
    p.add_argument("--output_dir", type=str, default="./outputs/qwen_lora")
    p.add_argument("--vlm_model_id", type=str, default=VLM_MODEL_ID)
    p.add_argument("--target", type=str, default="cot",
                   choices=["cot", "instruction", "pose"],
                   help="What the model is trained to produce. See module docstring.")
    p.add_argument("--frame_stride", type=int, default=10,
                   help="Keep 1 frame in N. Neighbouring frames share an instruction "
                        "and nearly the same pixels, so a stride of 1 mostly buys "
                        "duplicate gradients.")
    p.add_argument("--cameras", type=str, nargs="+", default=None,
                   help="Camera keys to feed the VLM. Default: all detected.")
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
    subsets = []
    camera_keys, state_key = None, "observation.state"
    for did in args.dataset_id:
        ds = LeRobotDataset(did, force_cache_sync=True, revision="main")
        cks = args.cameras or sorted(
            k for k in ds.meta.features if k.startswith("observation.images."))
        if camera_keys is None:
            camera_keys = cks
            print(f"Cameras: {camera_keys}")
        elif cks != camera_keys:
            raise ValueError(f"{did} cameras {cks} != {camera_keys}")
        subsets.append(LeRobotVLMDataset(ds, camera_keys, args.target,
                                         args.frame_stride, state_key))
    dataset = subsets[0] if len(subsets) == 1 else torch.utils.data.ConcatDataset(subsets)
    print(f"Frames after stride {args.frame_stride}: {len(dataset):,}")

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
        print("\nMeasuring KV drift vs the base encoder ...")
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            args.vlm_model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)
        drift = report_kv_drift(base, model, processor, loader, device)
        print(f"  relative hidden-state drift: {drift:.4f}")
        print("  This is the shock a resumed MoE's cross-attention must absorb: its")
        print("  ca_q projections were fit against the ORIGINAL geometry. Above ~0.3,")
        print("  expect the experts to need substantial re-adaptation, not a warm start.")

    print(f"\nNext:\n"
          f"  python src/lora_finetune_qwen.py --merge_and_save {final} \\\n"
          f"      --merged_dir {args.output_dir}/merged\n"
          f"  python src/train_wiltechs_moe.py --vlm_model_id {args.output_dir}/merged ...")


if __name__ == "__main__":
    main()
