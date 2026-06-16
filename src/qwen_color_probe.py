"""Does the frozen Qwen3-VL-4B actually resolve the T0 can colors?

The decisive diagnostic for the libero_10 T0 wrong-can failures: feed the REAL
LIBERO front-camera frame to the SAME VLM the policy uses (Qwen3-VL-4B-Instruct,
bf16) and ask it, in plain language, to identify the cans by color and side.

  - If Qwen answers correctly  -> the color IS in the VLM tokens. The fix is to
    make the DiT USE them: raise RobotCNN dropout (vision_dropout_prob) to break
    the CNN shortcut, or add detection. NOT resolution.
  - If Qwen gets it wrong       -> the encoder is color-blind at this resolution.
    No attention/dropout change helps; the fix is RESOLUTION (vision_input_size /
    camera render size) or explicit detection.

We test at the policy's NATIVE camera resolution and at 2x/4x upscales. If native
fails but an upscale succeeds, the image_processor's downsampling is the
bottleneck -> raising resolution will help.

Run on Colab:
    python src/qwen_color_probe.py --suite libero_10 --task_id 0

Saves the probed frames to /content (or cwd) so you can eyeball them too.
"""
from __future__ import annotations

import argparse


QUESTIONS = [
    "Look at the table. List each item you can see and its color.",
    "There are two cans on the table: a can of alphabet soup and a can of "
    "tomato sauce. What color is the alphabet soup can, and what color is the "
    "tomato sauce can?",
    "One can is blue and one can is red. Is the BLUE can on the left or the "
    "right? Is the RED can on the left or the right?",
    "Describe the exact location of the blue can relative to the other objects.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--task_id", type=int, default=0)
    ap.add_argument("--camera", default="image",
                    help="env pixel key to probe (image=agentview, image2=wrist)")
    ap.add_argument("--max_new_tokens", type=int, default=160)
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    from lerobot.envs.libero import LiberoEnv, _get_suite
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    # 1. Real env frame -------------------------------------------------------
    suite = _get_suite(args.suite)
    env = LiberoEnv(task_suite=suite, task_id=args.task_id, task_suite_name=args.suite,
                    obs_type="pixels_agent_pos", init_states=True, episode_index=0)
    obs, _ = env.reset(seed=0)
    print("task:", getattr(env, "task_description", "?"))
    print("cameras:", list(obs["pixels"].keys()))
    frame = np.asarray(obs["pixels"][args.camera])
    if frame.dtype != np.uint8:
        frame = (frame.clip(0, 1) * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
    env.close()
    base = Image.fromarray(frame).convert("RGB")
    print(f"native frame: {base.size} (W,H) from camera '{args.camera}'")

    variants = {
        f"native_{base.size[0]}px": base,
        f"up2x_{base.size[0]*2}px": base.resize((base.size[0]*2, base.size[1]*2), Image.LANCZOS),
        f"up4x_{base.size[0]*4}px": base.resize((base.size[0]*4, base.size[1]*4), Image.LANCZOS),
    }
    for name, im in variants.items():
        path = f"qwen_probe_{args.suite}_t{args.task_id}_{name}.png"
        im.save(path)
        print("saved", path)

    # 2. Load the SAME VLM the policy uses ------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "Qwen/Qwen3-VL-4B-Instruct"
    print(f"\nloading {model_id} (bf16) ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def ask(img: Image.Image, question: str) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": question},
        ]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors="pt").to(device)
        gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        trimmed = gen[:, inputs.input_ids.shape[1]:]
        return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    # 3. Ask, at each resolution ---------------------------------------------
    for vname, im in variants.items():
        print("\n" + "=" * 70 + f"\nRESOLUTION: {vname}\n" + "=" * 70)
        for q in QUESTIONS:
            print(f"\nQ: {q}\nA: {ask(im, q)}")

    print("\nDONE. Read the answers:")
    print("  - colors correct at native  -> info is THERE; fix = make DiT use it "
          "(RobotCNN dropout / detection), NOT resolution.")
    print("  - native wrong but upscale right -> processor downsampling is the "
          "bottleneck -> raise vision_input_size / camera render size.")
    print("  - wrong at all resolutions  -> encoder can't resolve it; use explicit "
          "detection (box_encoder.py).")


if __name__ == "__main__":
    main()
