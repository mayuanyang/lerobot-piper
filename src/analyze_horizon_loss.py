#!/usr/bin/env python3
"""Per-horizon-position flow loss for a wilro_moe checkpoint.

WHY THIS EXISTS
---------------
The training log prints ONE flow number, averaged over all `horizon` positions.
Inference executes only `n_action_steps` of them (2 in every eval in this repo),
so that number cannot answer the question it is always used to answer: is the
error in the part that RUNS, or in the far horizon, which is intrinsically less
predictable 6.4 s out?

The distinction is not cosmetic. `wilro_moe_model.compute_loss` sets
`pos_w[n_action_steps:] = future_steps_weight`, so with n_action_steps == horizon
(the shipped configs) that slice is EMPTY and future_steps_weight is inert --
position 63 is trained as hard as position 0 while only 0 and 1 are ever
executed. Whether that is worth changing depends entirely on the profile this
script prints, and on nothing else.

It also decides whether the `bias^2 ~ 0.20` reading in the benchmark tracker
measures a model deficiency or the far horizon's own entropy.

WHAT IT REPORTS
---------------
Per position bucket, pad-masked:

    flow          mean (v_theta - u)^2, UNWEIGHTED -- no pos_w, no gripper phase
    E[u^2]        mean squared target. Varies with position, so comparing raw
                  flow across buckets is not scale-free; this is the normalizer.
    flow/E[u^2]   1.0 means "predicted nothing at all". THIS is the column to read.
    valid%        fraction of cells not masked by action_is_pad. Late positions
                  are padded far more often (episodes end), and ignoring that
                  alone would manufacture a horizon profile.

t is drawn per SAMPLE and applied to every position of that sample, so the
large t-dependence of the flow loss is common-mode across buckets and cancels
in the ratios -- which is why the between-bucket comparison is trustworthy at
a few hundred samples while the absolute numbers still move with --batches.

Runs the real `compute_loss`, so the numbers cannot drift from training.

    python src/analyze_horizon_loss.py \
        --checkpoint ISdept/wilro-wilromoe-8x4-30k \
        --dataset_id ISdept/libero_10_lerobot \
        --batches 20 --batch_size 8
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from models.wilro_moe.wilro_moe_policy import WilroMoEPolicy
from models.wilro_moe.processor_wilro_moe import make_pre_post_processors


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def buckets_for(horizon: int, n_exec: int) -> list[tuple[int, int]]:
    """Bucket edges that put the EXECUTED prefix in a bucket of its own.

    n_exec comes from the checkpoint, but every eval overrides it to 2, so 0..1
    is always split out separately -- that is the only slice whose accuracy
    reaches the environment.
    """
    edges = sorted({0, 2, min(n_exec, horizon), 8, 16, 32, horizon}
                   & set(range(horizon + 1)) | {0, horizon})
    edges = [e for e in sorted(edges) if 0 <= e <= horizon]
    return [(a, b) for a, b in zip(edges, edges[1:]) if b > a]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="HF id or local dir holding policy.safetensors + config.json")
    ap.add_argument("--dataset_id", required=True,
                    help="LeRobot dataset to draw validation batches from")
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--full_chunks_only", action="store_true",
                    help="Drop any sample whose 64-step chunk is padded. THE "
                         "CONTROL for the horizon profile: without it the far "
                         "bucket is averaged over the ~71%% of samples that are "
                         "not near an episode end, while the executed bucket "
                         "sees all of them -- so the two buckets are scored on "
                         "different subpopulations, and end-of-episode is "
                         "exactly where placement happens.")
    a = ap.parse_args()

    device = pick_device(a.device)
    torch.manual_seed(a.seed)

    print(f"[load] policy   : {a.checkpoint}", flush=True)
    policy = WilroMoEPolicy.from_pretrained(a.checkpoint)
    cfg = policy.config
    policy.to(device)
    # eval(): the router adds randn*0.5 and paraphrase augmentation fires in
    # train mode. Both would land in this measurement as position-independent
    # noise and neither is present at inference.
    policy.eval()
    policy.model._record_position_loss = True

    horizon = int(cfg.horizon)
    n_exec = int(cfg.n_action_steps)
    n_obs = int(getattr(cfg, "n_obs_steps", 1) or 1)
    print(f"        horizon={horizon}  n_action_steps={n_exec}  n_obs_steps={n_obs}")
    if n_exec >= horizon:
        print(f"        NOTE pos_w[{n_exec}:] is an EMPTY slice -- "
              f"future_steps_weight={cfg.future_steps_weight} is inert in this "
              f"checkpoint; all {horizon} positions trained at weight 1.0.")

    print(f"[load] dataset  : {a.dataset_id}", flush=True)
    meta_probe = LeRobotDataset(a.dataset_id, revision="main")
    fps = int(getattr(meta_probe.meta, "fps", 10) or 10)
    del meta_probe
    ft = 1.0 / fps
    ds = LeRobotDataset(
        a.dataset_id, revision="main",
        tolerance_s=max(0.005, ft / 2),
        delta_timestamps={
            "observation.state": [-i * ft for i in range(n_obs)][::-1],
            "action": [i * ft for i in range(horizon)],
            **{k: [0.0] for k in cfg.input_features if k.startswith("observation.images.")},
        },
    )
    print(f"        fps={fps}  frames={len(ds)}", flush=True)

    preprocessor, _ = make_pre_post_processors(cfg, dataset_stats=ds.meta.stats)
    if hasattr(preprocessor, "to"):
        preprocessor.to(device)

    loader = torch.utils.data.DataLoader(
        ds, batch_size=a.batch_size, shuffle=True,
        num_workers=a.num_workers, drop_last=True,
        generator=torch.Generator().manual_seed(a.seed),
    )

    num = torch.zeros(horizon)
    u2 = torch.zeros(horizon)
    cnt = torch.zeros(horizon)
    amb = torch.zeros(horizon)
    seen = 0
    kept = 0
    # Same pin as train_wilro_moe's validate(): compute_loss draws a fresh t and
    # a fresh source noise per sample, and on that run two adjacent UNPINNED
    # passes moved the fit/held-out gap 8.3 -> 22.2 while the model itself did
    # not change. Pinning makes two invocations of this script comparable.
    torch.manual_seed(20260829)
    dev_t = torch.device(device)
    autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if dev_t.type == "cuda"
                else torch.autocast(device_type="cpu", enabled=False))
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= a.batches:
                break
            batch = {k: (v.to(dev_t, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            # The VLM reads the instruction from task_description; the dataset
            # ships it as "task". Without this the encoder runs unconditioned and
            # the whole profile is measured on the wrong model.
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
                                 else v)
                             for k, v in batch.items()}
                    kept += int(keep.sum())
            with autocast:
                policy.model.compute_loss(preprocessor(batch))
            n, u, c, m = policy.model._position_loss
            num += n; u2 += u; cnt += c; amb += m
            seen += a.batch_size
            print(f"  batch {i + 1}/{a.batches}", end="\r", flush=True)
    if a.full_chunks_only:
        print(f"\n[done] {kept}/{seen} samples kept (unpadded chunks only) over "
              f"{min(a.batches, i + 1)} batches\n")
    else:
        print(f"\n[done] {seen} samples over {min(a.batches, i + 1)} batches\n")
        print("NOTE valid% below is the confound: buckets are scored on different\n"
              "     subpopulations. Re-run with --full_chunks_only to control it.\n")

    if cnt.sum() == 0:
        raise SystemExit("No valid cells -- every action cell was masked as padding.")

    n_exp = int(getattr(cfg, "num_experts", 1) or 1)
    cells_per_pos = cnt.max().clamp(min=1)

    def decompose(lo, hi):
        """-> flow, E[u^2], ambiguity, bias^2, valid%. Krogh-Vedelsby, per bucket.

        bias^2 = flow - tau^2/n with tau^2 = amb*n/(n-1): the error component
        every expert shares, which no amount of ensemble diversity removes.
        """
        c = cnt[lo:hi].sum()
        if c == 0:
            return None
        f = (num[lo:hi].sum() / c).item()
        e = (u2[lo:hi].sum() / c).item()
        m = (amb[lo:hi].sum() / c).item()
        tau2 = m * n_exp / (n_exp - 1) if n_exp > 1 else 0.0
        return f, e, m, f - tau2 / max(n_exp, 1), (c / (cells_per_pos * (hi - lo))).item() * 100

    print(f"{'positions':>10} {'seconds':>11} {'flow':>7} {'amb':>7} {'bias^2':>7}"
          f" {'flow/E[u2]':>10} {'amb/flow':>9} {'bias2/flow':>11} {'valid%':>7}")
    print("-" * 90)
    rows = []
    for lo, hi in buckets_for(horizon, n_exec):
        r = decompose(lo, hi)
        if r is None:
            continue
        f, e, m, b2, v = r
        rows.append((lo, hi, f, e, f / e if e else float("nan"), m / f if f else 0.0, b2 / f if f else 0.0))
        print(f"{f'{lo}-{hi - 1}':>10} {f'{lo / fps:.1f}-{hi / fps:.1f}s':>11} "
              f"{f:7.4f} {m:7.4f} {b2:7.4f} {f / e if e else float('nan'):10.4f} "
              f"{m / f if f else 0:9.3f} {b2 / f if f else 0:11.3f} {v:6.1f}%"
              + ("  <- EXECUTED" if lo == 0 else ""))
    r = decompose(0, horizon)
    f_all, e_all, m_all, b2_all, _ = r
    print("-" * 90)
    print(f"{'ALL':>10} {f'0-{horizon / fps:.1f}s':>11} {f_all:7.4f} {m_all:7.4f} "
          f"{b2_all:7.4f} {f_all / e_all:10.4f} {m_all / f_all:9.3f} {b2_all / f_all:11.3f}")
    print(f"\n(n_experts={n_exp}; ensemble removes at most 1 - 1/n = "
          f"{(1 - 1 / n_exp) * 100:.1f}% of the average individual expert loss)")

    if rows:
        near = rows[0]
        far = rows[-1]
        ratio = far[4] / near[4] if near[4] else float("nan")
        print(f"\nExecuted prefix (pos {near[0]}-{near[1] - 1}) sits at "
              f"{near[4]:.3f} of 'predict nothing'; the far bucket "
              f"(pos {far[0]}-{far[1] - 1}) at {far[4]:.3f}. Ratio {ratio:.2f}x.")
        print("\nWHERE the error is (flow/E[u2] column):")
        print("  ratio >> 1  far-horizon entropy dominates; the executed steps are")
        print("              already accurate and reweighting buys little.")
        print("  ratio <= 1  the executed steps are the WORST part of the chunk, so")
        print("              n_action_steps currently spends the gradient on the")
        print("              positions that need it least.")
        print("\nWHETHER reweighting can fix it (amb/flow column, executed row):")
        print("  amb/flow LOW vs the other buckets")
        print("     -> the experts agree and are wrong together. The target is")
        print("        genuinely uncertain given the conditioning (demonstrator")
        print("        timing jitter at the contact phase). IRREDUCIBLE -- more")
        print("        weight there spends capacity on noise and can cost the far")
        print("        horizon, which is currently the healthy part.")
        print("  amb/flow COMPARABLE or higher")
        print("     -> estimation error, which averaging removes at 1 - 1/n and")
        print("        more gradient genuinely reduces. n_action_steps=8 with")
        print("        future_steps_weight 0.3 is then correctly aimed.")
        print("\nBoth columns are needed: the first says where to aim, the second")
        print("says whether aiming there does anything.")


if __name__ == "__main__":
    main()
