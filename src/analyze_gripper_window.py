#!/usr/bin/env python3
"""Is `gripper_transition_window` wide enough?

WHAT THE KNOB DOES
------------------
`compute_loss` up-weights the flow loss near a gripper open<->close transition:

    dg[:, 1:] = (g[:, 1:] - g[:, :-1]).abs()        # |delta gripper| per position
    trans     = dg > gripper_transition_thresh
    trans     = max_pool1d(trans, 2*win+1, padding=win)   # dilate by +/- win
    phase_w   = 1.0 + (gripper_phase_weight - 1.0) * trans

At win=2 that is 5 chunk positions, 0.5 s at 10 Hz. Whether 5 is the right
number has never been measured; it was shipped inside the precision bundle with
three other changes and never isolated.

WHAT THIS ANSWERS, IN ONE PASS
------------------------------
1. DETECTION. The distribution of |delta gripper| in NORMALISED units (which is
   what the threshold sees), how many cells the threshold flags, and how many
   consecutive positions a transition spans. A ramp that takes 4 positions with
   |dg| ~ 0.4 each is invisible to a 0.5 threshold, and then the window is being
   centred on the wrong place before its width matters at all.

2. COVERAGE. For each candidate win, the fraction of valid cells inside the
   dilated mask and the share of total loss WEIGHT that lands on them. If win=2
   covers 2% of cells, gripper_phase_weight is nearly inert and its width is a
   second-order question; if it covers 20%, the knob is reshaping the loss.

3. THE ACTUAL PROFILE. flow / ambiguity / bias^2 bucketed by DISTANCE TO THE
   NEAREST TRANSITION, normalised by E[u^2] so the buckets are comparable.
   **This is the measurement that sets win**: if the error is still elevated at
   +/-6 the window is covering a fraction of the problem, and if it is flat by
   +/-2 then 2 is enough -- or the whole up-weighting is aimed at a region that
   was never harder than the rest.

Distance is computed per SAMPLE: the transition sits at a different position in
every chunk, so the buckets cannot be read off the horizon profile.

    python analyze_gripper_window.py \
        --checkpoint ISdept/wilro-wilromoe-8x4-22k-obs2 \
        --dataset_id lerobot/libero --batches 60 --batch_size 8
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from models.wilro_moe.wilro_moe_policy import WilroMoEPolicy
from models.wilro_moe.processor_wilro_moe import make_pre_post_processors

BUCKETS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 10 ** 6)]


def pick_device(req):
    if req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def dist_to_transition(trans: np.ndarray) -> np.ndarray:
    """(B, H) bool -> (B, H) int distance to the nearest True, -1 if the row has none."""
    B, H = trans.shape
    out = np.full((B, H), -1, dtype=np.int64)
    pos = np.arange(H)
    for i in range(B):
        idx = np.flatnonzero(trans[i])
        if idx.size:
            out[i] = np.abs(pos[:, None] - idx[None, :]).min(axis=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset_id", required=True)
    ap.add_argument("--batches", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--full_chunks_only", action="store_true",
                    help="Drop samples whose chunk is padded, so every bucket is "
                         "scored on one population. Recommended.")
    ap.add_argument("--thresh", type=float, default=None,
                    help="Override gripper_transition_thresh for the scan.")
    a = ap.parse_args()

    device = pick_device(a.device)
    dev_t = torch.device(device)
    torch.manual_seed(a.seed)

    print(f"[load] policy  : {a.checkpoint}", flush=True)
    policy = WilroMoEPolicy.from_pretrained(a.checkpoint)
    cfg = policy.config
    policy.to(dev_t)
    policy.eval()
    policy.model._record_position_loss = True

    H = int(cfg.horizon)
    n_obs = int(getattr(cfg, "n_obs_steps", 1) or 1)
    gidx = int(getattr(cfg, "gripper_action_index", -1))
    thresh = float(a.thresh if a.thresh is not None
                   else getattr(cfg, "gripper_transition_thresh", 0.5))
    win_cfg = int(getattr(cfg, "gripper_transition_window", 2))
    gpw = float(getattr(cfg, "gripper_phase_weight", 1.0))
    print(f"        horizon={H}  gripper_action_index={gidx}  thresh={thresh}  "
          f"window={win_cfg}  phase_weight={gpw}")
    if gpw == 1.0:
        print("        NOTE gripper_phase_weight is 1.0 -- the whole mechanism is "
              "OFF in this checkpoint; the profile below still says whether it "
              "would have anything to bite on.")

    print(f"[load] dataset : {a.dataset_id}", flush=True)
    probe = LeRobotDataset(a.dataset_id, revision="main")
    fps = int(getattr(probe.meta, "fps", 10) or 10)
    del probe
    ft = 1.0 / fps
    ds = LeRobotDataset(
        a.dataset_id, revision="main", tolerance_s=max(0.005, ft / 2),
        delta_timestamps={
            "observation.state": [-i * ft for i in range(n_obs)][::-1],
            "action": [i * ft for i in range(H)],
            **{k: [0.0] for k in cfg.input_features if k.startswith("observation.images.")},
        },
    )
    preprocessor, _ = make_pre_post_processors(cfg, dataset_stats=ds.meta.stats)
    if hasattr(preprocessor, "to"):
        preprocessor.to(dev_t)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers,
        drop_last=True, generator=torch.Generator().manual_seed(a.seed))

    n_exp = int(getattr(cfg, "num_experts", 1) or 1)
    # accumulators
    dg_all: list = []
    span_hist: Counter = Counter()
    n_cells = n_flagged = n_rows = n_rows_with = 0
    num = {b: 0.0 for b in BUCKETS}
    u2 = {b: 0.0 for b in BUCKETS}
    amb = {b: 0.0 for b in BUCKETS}
    cnt = {b: 0.0 for b in BUCKETS}
    none_num = none_u2 = none_cnt = 0.0
    kept = seen = 0

    torch.manual_seed(20260829)   # same pin as validate(): t and source noise
    autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if dev_t.type == "cuda" else torch.autocast(device_type="cpu", enabled=False))
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= a.batches:
                break
            batch = {k: (v.to(dev_t, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            if "task" in batch and isinstance(batch["task"], (list, tuple)):
                batch["task_description"] = batch["task"]
            if a.full_chunks_only:
                pad = batch.get("action_is_pad")
                if pad is not None:
                    keep = ~pad.bool().any(dim=1)
                    if not bool(keep.any()):
                        continue
                    kl = keep.tolist()
                    batch = {k: (v[keep] if torch.is_tensor(v) and v.shape[:1] == keep.shape
                                 else [x for x, m in zip(v, kl) if m]
                                 if isinstance(v, (list, tuple)) and len(v) == len(kl)
                                 else v) for k, v in batch.items()}
            seen += a.batch_size
            proc = preprocessor(batch)
            with autocast:
                policy.model.compute_loss(proc)
            L, U, C, A = policy.model._cell_loss              # each (B, H)
            L, U, C, A = (x.numpy() for x in (L, U, C, A))

            # Transitions, computed exactly as compute_loss does -- on the
            # NORMALISED actions the model sees, not the raw dataset units.
            g = proc["action"].float().nan_to_num(0.0).clamp(-10.0, 10.0)[:, :, gidx]
            g = g.cpu().numpy()
            dg = np.zeros_like(g)
            dg[:, 1:] = np.abs(g[:, 1:] - g[:, :-1])
            trans = dg > thresh
            dg_all.append(dg[:, 1:].ravel())
            n_cells += trans.size
            n_flagged += int(trans.sum())
            n_rows += trans.shape[0]
            n_rows_with += int(trans.any(axis=1).sum())
            for r in trans:                                    # contiguous run lengths
                run = 0
                for v in r:
                    if v:
                        run += 1
                    elif run:
                        span_hist[run] += 1
                        run = 0
                if run:
                    span_hist[run] += 1

            d = dist_to_transition(trans)
            kept += trans.shape[0]
            for lo, hi in BUCKETS:
                m = (d >= lo) & (d <= hi)
                if not m.any():
                    continue
                num[(lo, hi)] += float(L[m].sum())
                u2[(lo, hi)] += float(U[m].sum())
                amb[(lo, hi)] += float(A[m].sum())
                cnt[(lo, hi)] += float(C[m].sum())
            m0 = d < 0
            if m0.any():
                none_num += float(L[m0].sum()); none_u2 += float(U[m0].sum())
                none_cnt += float(C[m0].sum())
            print(f"  batch {i + 1}/{a.batches}", end="\r", flush=True)

    print(f"\n[done] {kept} chunks scored"
          + (f" ({kept}/{seen} unpadded)" if a.full_chunks_only else "") + "\n")

    # ---- 1. detection -------------------------------------------------
    dgv = np.concatenate(dg_all)
    print("=== 1. DETECTION: |delta gripper| in NORMALISED units ===")
    qs = [50, 90, 95, 99, 99.9]
    print("  percentiles: " + "  ".join(
        f"p{q}={np.percentile(dgv, q):.3f}" for q in qs) + f"  max={dgv.max():.3f}")
    print(f"  cells over thresh {thresh}: {n_flagged}/{n_cells} = "
          f"{n_flagged / max(n_cells, 1) * 100:.2f}%")
    print(f"  chunks containing >=1 transition: {n_rows_with}/{n_rows} = "
          f"{n_rows_with / max(n_rows, 1) * 100:.1f}%")
    if span_hist:
        tot = sum(span_hist.values())
        spans = " ".join(f"{k}:{v / tot * 100:.0f}%" for k, v in sorted(span_hist.items())[:6])
        print(f"  transition RUN LENGTH (consecutive flagged positions): {spans}")
        print("    a run of 1 means the gripper flips in a single step and the "
              "threshold sees it cleanly;\n    longer runs mean a ramp, and then "
              "the window is dilated around several centres.")

    # ---- 2. coverage --------------------------------------------------
    print(f"\n=== 2. COVERAGE: what fraction of the loss gripper_phase_weight "
          f"{gpw} actually touches ===")
    print(f"{'win':>5} {'positions':>10} {'seconds':>8} {'cells covered':>14} "
          f"{'share of weight':>16}")
    covered_frac = {}
    for w in (0, 1, 2, 3, 4, 6, 8):
        # A cell is covered iff its distance <= w. Buckets are contiguous
        # ranges, so only whole buckets with hi <= w count; a bucket straddling
        # w (e.g. 3-4 at w=3) is left out rather than split, which makes this a
        # LOWER bound at those w. The bucket edges 0,1,2,3-4,5-8 line up exactly
        # with w = 0,1,2,4,8.
        c = sum(v for (lo, hi), v in cnt.items() if hi <= w)
        tot = sum(cnt.values()) + none_cnt
        f = c / max(tot, 1e-9)
        covered_frac[w] = f
        share = f * gpw / ((1 - f) + f * gpw) if gpw > 0 else f
        mark = "  <- config" if w == win_cfg else ""
        print(f"{w:5d} {2 * w + 1:10d} {(2 * w + 1) / fps:7.1f}s "
              f"{f * 100:13.1f}% {share * 100:15.1f}%{mark}")

    # ---- 3. the profile ----------------------------------------------
    print(f"\n=== 3. THE PROFILE: error vs distance to the nearest transition ===")
    print(f"{'distance':>10} {'flow':>8} {'amb':>8} {'bias^2':>8} "
          f"{'flow/E[u2]':>11} {'cells':>10}")
    print("-" * 62)
    rows = []
    for lo, hi in BUCKETS:
        c = cnt[(lo, hi)]
        if c <= 0:
            continue
        f = num[(lo, hi)] / c
        e = u2[(lo, hi)] / c
        m = amb[(lo, hi)] / c
        t2 = m * n_exp / (n_exp - 1) if n_exp > 1 else 0.0
        lbl = f"{lo}" if lo == hi else (f"{lo}+" if hi > 10 ** 5 else f"{lo}-{hi}")
        rows.append((lo, hi, f / e if e else float("nan")))
        print(f"{lbl:>10} {f:8.4f} {m:8.4f} {f - t2 / n_exp:8.4f} "
              f"{f / e if e else float('nan'):11.4f} {c:10.0f}")
    if none_cnt > 0:
        f = none_num / none_cnt
        e = none_u2 / none_cnt
        print(f"{'no trans':>10} {f:8.4f} {'-':>8} {'-':>8} "
              f"{f / e if e else float('nan'):11.4f} {none_cnt:10.0f}")

    if rows:
        far = rows[-1][2]
        print(f"\nReading it (flow/E[u2] column, baseline = the farthest bucket "
              f"{far:.4f}):")
        print(f"  Find the largest distance whose ratio is still clearly above "
              f"{far:.4f}.")
        print(f"  That distance IS the window this loss term should use. The "
              f"config uses {win_cfg}.")
        print("  If NO bucket is elevated, the transition neighbourhood was never "
              "harder than\n  the rest of the chunk, and gripper_phase_weight is "
              "up-weighting a region that\n  did not need it -- which would make "
              "its contribution to the precision bundle\n  something other than "
              "what the bundle assumed.")


if __name__ == "__main__":
    main()
