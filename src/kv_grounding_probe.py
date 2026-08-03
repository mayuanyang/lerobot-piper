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
    # 1. object names + the BDDL goal state, which names the real target
    python src/kv_grounding_probe.py --suite libero_spatial --task_id 3 --list_bodies

    # 2. label the objects on the frame, so identity is not guessed by eye
    python src/kv_grounding_probe.py --suite libero_spatial --task_id 3 \
        --annotate --init_state_id 0

    # 3. the probe, once at the policy's current grid and once at 512
    python src/kv_grounding_probe.py --suite libero_spatial --task_id 3 \
        --target_body akita_black_bowl_1 --distractor_body akita_black_bowl_2 \
        --n_states 50 [--vision_input_size 512]

Layouts differ between LIBERO's ~50 initial states and eval sweeps all of them,
so --init_state_id decides which one --list_bodies/--annotate describe, and the
probe cycles through them for its sample.
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
    ap.add_argument("--annotate", action="store_true",
                    help="save the frame with every object's sim position projected "
                         "onto it and labelled, then exit. Settles which blob in the "
                         "picture is which sim object instead of guessing.")
    ap.add_argument("--camera_name", default="agentview",
                    help="robosuite camera name for --annotate projection "
                         "(agentview / robot0_eye_in_hand)")
    ap.add_argument("--annotate_detect", action="store_true",
                    help="annotate by asking Qwen3-VL for 2D boxes instead of "
                         "projecting sim coordinates. Uses the convention from "
                         "precompute_video_bounding_boxes_standalone.py: bbox_2d = "
                         "[x1,y1,x2,y2], x=col, y=row, origin top-left, normalised to "
                         "[0,1000]. No camera matrices, so no axis ambiguity -- and it "
                         "shows whether Qwen can localise the bowls at all.")
    ap.add_argument("--init_state_id", type=int, default=0,
                    help="which LIBERO init state to use for --list_bodies/--annotate. "
                         "LIBERO ships ~50 per task and eval sweeps them, so layouts "
                         "differ between them -- pass the one you are looking at.")
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

    # Reuse the walker from rl_staged_reward rather than guessing attribute
    # paths: it descends LiberoEnv -> OffScreenRenderEnv -> the robosuite
    # problem env, 12 levels with cycle detection, and is already load-bearing
    # for the staged-reward RL runs.
    try:
        from rl_staged_reward import get_sim_env
    except ImportError:
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from rl_staged_reward import get_sim_env

    sim_env = get_sim_env(env)
    if sim_env is None:
        raise RuntimeError(
            "get_sim_env() could not find the robosuite problem env. Run "
            "`python src/libero_reward_probe.py` to dump the wrapper chain.")

    def n_init_states(e):
        """Count of LIBERO initial states. Deliberately no truthiness test:
        _init_states is a numpy array, so `x or []` raises "truth value of an
        array is ambiguous"."""
        s = getattr(e, "_init_states", None)
        if s is None:
            return 0
        try:
            return len(s)
        except TypeError:
            return 0

    def obj_positions(e):
        """{object_name: xyz}. obj_body_id is the env's own object registry, so
        it lists exactly the task objects -- far better than dumping every
        MuJoCo body and guessing which are props."""
        se = get_sim_env(e)
        bid = getattr(se, "obj_body_id", None) or {}
        return {k: np.array(se.sim.data.body_xpos[v]) for k, v in bid.items()}

    if args.list_bodies:
        n_init = n_init_states(env)
        if n_init:
            env._init_state_id = args.init_state_id % n_init
            env.reset()
        print(f"\ninit state {args.init_state_id} of {n_init or '?'} "
              f"(layouts differ between them -- these coordinates are for THIS one)")
        pos0 = obj_positions(env)
        print("\nobjects:")
        for k, v in pos0.items():
            print(f"    {k:<34} xyz={np.round(v, 3)}")
        print("\ntask language:", repr(getattr(env, 'task_description', '')))

        # AUTHORITATIVE target. The BDDL goal state names the object the task
        # actually scores on -- do NOT infer the target by reading the English
        # relation geometrically off the coordinates above. Two identical bowls
        # make that inference look reasonable and be wrong.
        pp = getattr(sim_env, "parsed_problem", None)
        goal = (pp or {}).get("goal_state") if isinstance(pp, dict) else None
        print("\nBDDL goal_state (this is what defines the target):")
        if goal:
            for conj in goal:
                print("   ", conj)
            objs = {c[1] for c in goal if len(c) > 1 and c[1] in pos0}
            if len(objs) == 1:
                tgt = objs.pop()
                other = [k for k in pos0 if k.startswith(tgt.rsplit("_", 1)[0]) and k != tgt]
                print(f"\n  => --target_body {tgt}")
                if other:
                    print(f"     --distractor_body {other[0]}")
            elif objs:
                print(f"\n  goal mentions several objects: {sorted(objs)} -- pick by hand")
        else:
            print("    (not exposed; try `python src/libero_reward_probe.py` "
                  "which dumps parsed_problem in full)")
        ooi = getattr(sim_env, "obj_of_interest", None)
        if ooi:
            print("\nobj_of_interest:", list(ooi))
        env.close()
        return
    # --annotate labels every object, so it needs no target/distractor: keep it
    # ahead of the required-args check.
    if args.annotate_detect:
        # Sim-projection-free annotation. Qwen reports boxes directly in image
        # coordinates, so there is no camera matrix and no axis convention to
        # get wrong -- which is the entire reason --annotate kept mislabelling.
        # Convention lifted from precompute_video_bounding_boxes_standalone.py.
        import json as _json
        from PIL import ImageDraw
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        n_init = n_init_states(env)
        if n_init:
            env._init_state_id = args.init_state_id % n_init
        obs, _ = env.reset()
        f = np.asarray(obs["pixels"][args.camera])
        if f.dtype != np.uint8:
            f = (f.clip(0, 1) * 255).astype(np.uint8) if f.max() <= 1.0 else f.astype(np.uint8)
        env.close()
        im = Image.fromarray(f).convert("RGB")
        H, W = f.shape[:2]
        if args.vision_input_size:
            im = im.resize((args.vision_input_size,) * 2, Image.BICUBIC)
            W = H = args.vision_input_size

        device = "cuda" if torch.cuda.is_available() else "cpu"
        mid = "Qwen/Qwen3-VL-4B-Instruct"
        print(f"loading {mid} ...")
        mdl = Qwen3VLForConditionalGeneration.from_pretrained(
            mid, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device).eval()
        proc = AutoProcessor.from_pretrained(mid)
        sysmsg = ("You are a precise 2D object detector. For each object, provide its "
                  "2D bounding box in JSON format with 'bbox_2d' field containing "
                  "[x1, y1, x2, y2] where (x1, y1) is the top-left corner and (x2, y2) "
                  "is the bottom-right corner. All coordinates should be normalized to "
                  "[0, 1000]. Also provide a 'label' field indicating the object category.")
        usermsg = ('locate every instance that belongs to the following categories: '
                   '"bowl, plate, small shallow cup, box, cabinet, stove". '
                   'Report bbox coordinates in JSON format.')
        msgs = [{"role": "system", "content": [{"type": "text", "text": sysmsg}]},
                {"role": "user", "content": [{"type": "image", "image": im},
                                             {"type": "text", "text": usermsg}]}]
        with torch.no_grad():
            inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt").to(device)
            gen = mdl.generate(**inp, max_new_tokens=512)
            txt = proc.batch_decode(gen[:, inp.input_ids.shape[1]:],
                                    skip_special_tokens=True)[0]
        print("\nraw:", txt[:1200])
        s = txt.find("[")
        e = txt.rfind("]")
        boxes = []
        if s != -1 and e != -1:
            try:
                boxes = _json.loads(txt[s:e + 1])
            except _json.JSONDecodeError as ex:
                print("could not parse JSON:", ex)
        dr = ImageDraw.Draw(im)
        print(f"\n{len(boxes)} box(es), converted to pixels on a {W}x{H} frame:")
        for b in boxes:
            bb, lab = b.get("bbox_2d"), str(b.get("label", "?"))
            if not bb or len(bb) != 4:
                continue
            x1, y1, x2, y2 = (bb[0] / 1000 * W, bb[1] / 1000 * H,
                              bb[2] / 1000 * W, bb[3] / 1000 * H)
            dr.rectangle([x1, y1, x2, y2], outline=(255, 60, 60), width=2)
            dr.text((x1 + 2, y1 + 2), lab, fill=(255, 60, 60))
            print(f"  {lab:<24} x=[{x1:6.1f},{x2:6.1f}]  y=[{y1:6.1f},{y2:6.1f}]")
        path = f"detected_t{args.task_id}_s{args.init_state_id}.png"
        im.save(path)
        print(f"\nsaved {path}")
        print("Qwen's own image coordinates -- if the boxes land on the objects, the "
              "convention is confirmed and this is also the readout box_encoder.py "
              "would consume. If it finds only one bowl, or none, that is the "
              "grounding answer without needing the probe.")
        return

    if args.annotate:
        # Project each object's sim xyz into the frame and label it. Every
        # confusion in this investigation so far has come from reading object
        # identity off a screenshot by eye; this replaces that with the sim's
        # own answer.
        from PIL import ImageDraw
        from robosuite.utils.camera_utils import (
            get_camera_transform_matrix, project_points_from_world_to_camera)

        n_init = n_init_states(env)
        if n_init:
            env._init_state_id = args.init_state_id % n_init
        print(f"init state {args.init_state_id} of {n_init or '?'}")
        obs, _ = env.reset()
        se = get_sim_env(env)
        pos = obj_positions(env)
        f = np.asarray(obs["pixels"][args.camera])
        if f.dtype != np.uint8:
            f = (f.clip(0, 1) * 255).astype(np.uint8) if f.max() <= 1.0 else f.astype(np.uint8)
        H, W = f.shape[:2]
        T = get_camera_transform_matrix(sim=se.sim, camera_name=args.camera_name,
                                        camera_height=H, camera_width=W)
        names = list(pos)
        pix = project_points_from_world_to_camera(
            np.stack([pos[n] for n in names]), T, H, W)  # (N, 2) as [row, col]

        try:
            cams = [se.sim.model.camera_id2name(i)
                    for i in range(se.sim.model.ncam)]
            print("cameras in this sim:", cams)
        except Exception:
            pass

        # Save every row/col mirror combination rather than reasoning about the
        # convention. robosuite renders bottom-up, LIBERO may flip before handing
        # the frame over, and the projection helper's own axis order is one more
        # place to be wrong -- four small PNGs settle it by eye in one look,
        # which guessing demonstrably did not.
        for fr in (False, True):
            for fc in (False, True):
                im = Image.fromarray(f).convert("RGB")
                dr = ImageDraw.Draw(im)
                rows = []
                for n, (r, c) in zip(names, pix):
                    r = (H - 1 - r) if fr else r
                    c = (W - 1 - c) if fc else c
                    r, c = int(round(r)), int(round(c))
                    colr = (255, 40, 40) if "bowl_1" in n else (
                        (40, 120, 255) if "bowl_2" in n else (40, 200, 40))
                    dr.ellipse([c - 5, r - 5, c + 5, r + 5], outline=colr, width=2)
                    dr.text((c + 8, r - 6), n.replace("akita_black_", ""), fill=colr)
                    rows.append((n, r, c))
                tag = ("_flipR" if fr else "") + ("_flipC" if fc else "")
                path = f"annotated_t{args.task_id}_s{args.init_state_id}{tag or '_raw'}.png"
                im.save(path)
                print(f"saved {path}   " + "  ".join(f"{n.split('_')[-2][:4]}=({r},{c})"
                                                     for n, r, c in rows[:3]))
        print("\nFour variants saved: raw, flipR (rows mirrored), flipC (columns "
              "mirrored), flipR_flipC (both). Open them and keep the one whose "
              "circles land on the objects; tell me which and I will pin it.")
        print("RED = akita_black_bowl_1 (the BDDL target).  BLUE = bowl_2 (distractor).")
        print("If NONE line up, --camera_name is probably wrong -- try another "
              "entry from the camera list above.")
        env.close()
        return

    if not args.target_body or not args.distractor_body:
        ap.error("--target_body and --distractor_body are required (see --list_bodies)")
    known = set(obj_positions(env))
    for b in (args.target_body, args.distractor_body):
        if b not in known:
            ap.error(f"object {b!r} not in this task. Known: {sorted(known)}")

    # 1. Collect frames + ground-truth object positions -----------------------
    frames, pos_t, pos_d = [], [], []
    n_init = n_init_states(env)
    if n_init:
        print(f"LIBERO init states available for this task: {n_init}")
        if args.n_states > n_init:
            print(f"  capping --n_states {args.n_states} -> {n_init}")
            args.n_states = n_init
    else:
        print("!! could not read env._init_states -- every sample may be the SAME "
              "layout, which makes the probe meaningless. Check the wrapper.")

    for i in range(args.n_states):
        # LiberoEnv.reset() reads _init_state_id and calls
        # set_init_state(_init_states[id]). reset(seed=...) does NOT select the
        # layout -- getting this wrong pins every sample to one init state.
        # (Mechanism per train_rft.py's init-state sweep.)
        if n_init:
            env._init_state_id = i % n_init
        obs, _ = env.reset()
        p = obj_positions(env)
        f = np.asarray(obs["pixels"][args.camera])
        if f.dtype != np.uint8:
            f = (f.clip(0, 1) * 255).astype(np.uint8) if f.max() <= 1.0 else f.astype(np.uint8)
        frames.append(Image.fromarray(f).convert("RGB"))
        pos_t.append(p[args.target_body])
        pos_d.append(p[args.distractor_body])
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
