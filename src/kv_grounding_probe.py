"""Is the target bowl's identity/position linearly readable from the frozen
VLM features the DiT actually cross-attends to?

WHY THIS EXISTS. qwen_color_probe.py showed the encoder CAN do the libero_spatial
discrimination -- with perceptual vocabulary and a plate anchor it answered
consistently and correctly from 256 vision tokens up. But that probe uses Qwen's
FULL GENERATIVE pathway: 36 layers of attention plus autoregressive decoding.
The policy uses nothing of the sort. It takes the K/V cache at the captured
layers and reads it with cross-attention from a trainable DiT. "Present enough
for Qwen to verbalise" does not imply "extractable by that readout".

This script measures the gap directly, in minutes instead of a 10-hour run.

  probe succeeds -> the information IS in the representation the DiT reads.
                    The failure is downstream: training signal, readout, or the
                    RobotCNN shortcut. Keep working on the policy.
  probe fails    -> the information is NOT in there in a readable form. More
                    prompt wording and more pixels cannot help; go to explicit
                    detection (object_detector.py + box_encoder.py), which hands
                    the DiT coordinates instead of asking it to infer them.

WHY HIDDEN STATES AND NOT K/V. K = k_proj(h) and V = v_proj(h) are LINEAR maps of
the layer's hidden state. A linear probe on h therefore has exactly the same
information available as a linear probe on K or V, and reading hidden_states
avoids hand-reimplementing the capture path (and any chance of it drifting from
the model's).

THE CONTROL MATTERS MORE THAN THE SCORE. A probe that locates the target may
just be locating "a bowl". So we fit the SAME probe against the distractor. If
both score alike, the representation encodes bowl positions but not which one
the instruction selects -- that is a grounding failure, and it is the result
that sends you to detection. A label-shuffled fit is also reported as a noise
floor, because with ~50 initial states an unregularised high-dimensional probe
will happily fit pure noise.

Usage:
    python src/kv_grounding_probe.py --suite libero_spatial --list_bodies
    python src/kv_grounding_probe.py --suite libero_spatial --task_id 3 \
        --target_body akita_black_bowl_1_main --distractor_body akita_black_bowl_2_main \
        --n_states 50 --vision_input_size 512
"""
from __future__ import annotations

import argparse


def ridge_cv(X, Y, alphas, folds=5, seed=0):
    """Leave-fold-out ridge. Returns (best_alpha, cv_r2, per-dim cv_r2).

    Closed form, numpy only -- no sklearn dependency. R^2 is computed against a
    predict-the-training-mean baseline, so 0 means "no better than guessing the
    average layout" and negative means actively worse.
    """
    import numpy as np

    n = X.shape[0]
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    cuts = np.array_split(order, folds)
    best = (None, -1e9, None)
    for a in alphas:
        preds = np.zeros_like(Y)
        for f in range(folds):
            te = cuts[f]
            tr = np.concatenate([cuts[g] for g in range(folds) if g != f])
            Xtr, Ytr = X[tr], Y[tr]
            mu, my = Xtr.mean(0, keepdims=True), Ytr.mean(0, keepdims=True)
            Xc, Yc = Xtr - mu, Ytr - my
            d = Xc.shape[1]
            if d <= Xc.shape[0]:
                W = np.linalg.solve(Xc.T @ Xc + a * np.eye(d), Xc.T @ Yc)
            else:  # dual form: cheaper and identical when features outnumber samples
                K = Xc @ Xc.T
                W = Xc.T @ np.linalg.solve(K + a * np.eye(K.shape[0]), Yc)
            preds[te] = (X[te] - mu) @ W + my
        ss_res = ((Y - preds) ** 2).sum(0)
        ss_tot = ((Y - Y.mean(0, keepdims=True)) ** 2).sum(0)
        r2_dim = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
        r2 = float(r2_dim.mean())
        if r2 > best[1]:
            best = (a, r2, r2_dim)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task_id", type=int, default=0)
    ap.add_argument("--camera", default="image")
    ap.add_argument("--n_states", type=int, default=50,
                    help="initial states to sample. This is the REAL sample size: "
                         "object positions are constant within an episode, so extra "
                         "timesteps add no label diversity.")
    ap.add_argument("--vision_input_size", type=int, default=0,
                    help="0 = processor default (matches the policy's current 8x8 "
                         "grid); 512 = the 16x16 grid --vision_input_size 512 gives it.")
    ap.add_argument("--layer", type=int, default=-1,
                    help="LM layer to read hidden states from. -1 = last. Use one of "
                         "the policy's vlm_capture_layers to match what an expert sees.")
    ap.add_argument("--target_body", default=None)
    ap.add_argument("--distractor_body", default=None)
    ap.add_argument("--list_bodies", action="store_true",
                    help="print the sim body names for this task and exit")
    ap.add_argument("--instruction", default=None,
                    help="text placed BEFORE the image, as text_first=True does. "
                         "Defaults to the task's own language.")
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    from lerobot.envs.libero import LiberoEnv, _get_suite

    suite = _get_suite(args.suite)
    env = LiberoEnv(task_suite=suite, task_id=args.task_id, task_suite_name=args.suite,
                    obs_type="pixels_agent_pos", init_states=True, episode_index=0)
    task_text = args.instruction or getattr(env, "task_description", "")
    print(f"task: {task_text!r}")

    def sim_of(e):
        for path in ("sim", "env.sim", "env.env.sim", "unwrapped.env.sim"):
            o = e
            try:
                for part in path.split("."):
                    o = getattr(o, part)
                if hasattr(o, "data") and hasattr(o, "model"):
                    return o
            except AttributeError:
                continue
        raise RuntimeError("could not reach the robosuite sim through the LiberoEnv wrapper")

    sim = sim_of(env)
    names = [sim.model.body_id2name(i) for i in range(sim.model.nbody)]
    if args.list_bodies:
        print("\nbody names (pick --target_body / --distractor_body from these):")
        for n in names:
            if n and not n.startswith(("robot", "gripper", "world", "table")):
                print("   ", n)
        env.close()
        return
    if not args.target_body or not args.distractor_body:
        ap.error("--target_body and --distractor_body are required (see --list_bodies)")
    for b in (args.target_body, args.distractor_body):
        if b not in names:
            ap.error(f"body {b!r} not in this task's sim. Run --list_bodies.")

    # 1. Collect frames + ground-truth object positions -----------------------
    frames, pos_t, pos_d = [], [], []
    for i in range(args.n_states):
        try:
            env.set_init_state_index(i)  # some wrappers expose this
        except AttributeError:
            pass
        obs, _ = env.reset(seed=i)
        s = sim_of(env)
        f = np.asarray(obs["pixels"][args.camera])
        if f.dtype != np.uint8:
            f = (f.clip(0, 1) * 255).astype(np.uint8) if f.max() <= 1.0 else f.astype(np.uint8)
        frames.append(Image.fromarray(f).convert("RGB"))
        pos_t.append(np.array(s.data.body_xpos[s.model.body_name2id(args.target_body)]))
        pos_d.append(np.array(s.data.body_xpos[s.model.body_name2id(args.distractor_body)]))
    env.close()
    pos_t, pos_d = np.stack(pos_t), np.stack(pos_d)

    spread_t = pos_t.std(0)
    print(f"\ncollected {len(frames)} layouts, frame {frames[0].size}")
    print(f"  target     xyz std: {np.round(spread_t, 4)}")
    print(f"  distractor xyz std: {np.round(pos_d.std(0), 4)}")
    if spread_t.max() < 1e-3:
        print("  !! the target never moves across initial states. A position probe has "
              "nothing to fit -- regression R^2 will be meaningless. Check that "
              "init_states is actually varying the layout.")

    # 2. Frozen VLM features, text FIRST (matches text_first=True) ------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    model_id = "Qwen/Qwen3-VL-4B-Instruct"
    print(f"\nloading {model_id} (bf16) ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    merge = int(getattr(getattr(model.config, "vision_config", None),
                        "spatial_merge_size", 2) or 2)

    @torch.no_grad()
    def features(img):
        if args.vision_input_size:
            ts = args.vision_input_size
            img = img.resize((ts, ts), Image.BICUBIC)
        # Instruction BEFORE the image: under a causal mask this is what makes
        # the vision positions language-conditioned, which is the whole premise
        # of the layout the policy now uses.
        messages = [{"role": "user", "content": [
            {"type": "text", "text": task_text},
            {"type": "image", "image": img},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        px = img.size[0] * img.size[1]
        try:
            inputs = processor(text=[text], images=[img], return_tensors="pt",
                               min_pixels=px, max_pixels=px)
        except (TypeError, ValueError):
            inputs = processor(text=[text], images=[img], return_tensors="pt")
        thw = inputs["image_grid_thw"][0].tolist()
        n_vis = (int(thw[1]) // merge) * (int(thw[2]) // merge)
        ids = inputs["input_ids"][0]
        out = model(**inputs.to(device), output_hidden_states=True)
        h = out.hidden_states[args.layer][0].float().cpu().numpy()  # (L, d)
        # Keep ONLY the image positions. Slicing the whole sequence would mix in
        # the instruction tokens, which are identical across layouts and would
        # dilute the very signal being measured.
        img_tok = getattr(model.config, "image_token_id", None)
        if img_tok is not None:
            vis_idx = (ids == img_tok).nonzero().flatten().cpu().numpy()
        else:
            vis_idx = np.arange(h.shape[0])
        if len(vis_idx) != n_vis:
            print(f"  note: {len(vis_idx)} image-token positions vs {n_vis} expected "
                  f"from grid_thw; using the token mask.")
        return h[vis_idx], n_vis

    feats, n_vis = [], None
    for i, im in enumerate(frames):
        h, nv = features(im)
        n_vis = nv if n_vis is None else n_vis
        # A plain mean over the image positions throws away WHERE things are,
        # which is the whole question. Vision tokens arrive in raster order, so
        # pooling into horizontal bands keeps coarse vertical structure and the
        # band means keep coarse horizontal structure -- enough for a linear
        # probe to express "the target is over there" without blowing the
        # feature count past what ~50 layouts can support.
        L = h.shape[0]
        bands = np.stack([h[j * L // 4:(j + 1) * L // 4].mean(0) for j in range(4)])
        feats.append(np.concatenate([h.mean(0), bands.reshape(-1)]))
        if i == 0:
            print(f"  vision tokens/frame: {n_vis}   pooled positions: {L}   "
                  f"feature dim: {feats[0].shape[0]}")
    X = np.stack(feats)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)

    # 3. Fit ------------------------------------------------------------------
    alphas = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
    rng = np.random.default_rng(0)
    print("\n" + "=" * 68)
    print(f"{'probe':<28}{'best alpha':>12}{'CV R^2':>10}")
    print("=" * 68)
    a1, r1, _ = ridge_cv(X, pos_t, alphas)
    print(f"{'TARGET bowl xyz':<28}{a1:>12.0e}{r1:>10.3f}")
    a2, r2, _ = ridge_cv(X, pos_d, alphas)
    print(f"{'DISTRACTOR bowl xyz':<28}{a2:>12.0e}{r2:>10.3f}")
    a3, r3, _ = ridge_cv(X, pos_t[rng.permutation(len(pos_t))], alphas)
    print(f"{'TARGET, labels shuffled':<28}{a3:>12.0e}{r3:>10.3f}   <- noise floor")
    print("=" * 68)

    print("\nHOW TO READ THIS")
    print("  target >> shuffled  AND  target >> distractor")
    print("     -> the representation encodes WHICH bowl the instruction selects.")
    print("        The information is there; the failure is downstream (training")
    print("        signal / readout / RobotCNN shortcut). Keep fixing the policy.")
    print("  target ~ distractor, both >> shuffled")
    print("     -> it encodes where bowls ARE but not which one is meant. This is")
    print("        the grounding failure itself, and no amount of prompt wording or")
    print("        resolution fixes it -> explicit detection (box_encoder.py).")
    print("  target ~ shuffled")
    print("     -> nothing linearly readable at all. Check --layer (try a capture")
    print("        layer, not just the last) and --vision_input_size before")
    print("        concluding; then go to detection.")
    print("\n  Re-run with --vision_input_size 0 vs 512 to see whether resolution")
    print("  changes the readability, which is the question the last 10k steps")
    print("  were supposed to answer and could not.")


if __name__ == "__main__":
    main()
