"""Train Q(o, a) -> steps-to-success on a collected corpus. The policy is untouched.

    python src/train_scorer.py \
        --dataset_id ISdept/libero-awr-goal \
        --rewards /content/awr_goal_rewards.json \
        --output_dir ./outputs/scorer_goal

THE GO/NO-GO IS `rank_acc`, NOT `auc`. A scorer that predicts the episode
outcome beautifully but ignores its `action` argument is USELESS for best-of-N:
selection asks it to rank K candidates against ONE observation, so anything it
knows from the observation alone cancels. `rank_acc` is measured against a
hard negative -- the action chunk from the SAME observation Delta steps later,
i.e. the right plan at the wrong phase -- and if it sits near 0.50 the model
has learned V(o) and the whole approach is dead before any eval is spent.

Reported each validation pass:
  rank_acc   P(score(a_true) < score(a_shifted))  -- action sensitivity
  spearman   rank correlation with true steps-to-success
  auc        first-N-chunk mean score separating failed from successful episodes
             (restricted to the head of the episode ON PURPOSE: failures run
             149 chunks against successes' 31, so an episode-mean measures
             LENGTH, not prediction)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from models.scorer import ActionScorer


class _RowIndexed(torch.utils.data.Dataset):
    """Attach the CONCATENATED row id to every sample.

    Not `b["index"]`: that column is the row's position inside its OWN dataset
    and restarts at 0 for the second corpus, so with several --dataset_id it
    would look up the wrong target and the wrong episode -- silently, since
    both arrays are the right length. Module level so DataLoader workers can
    pickle it.
    """

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        s = self.ds[i]
        s["_row"] = torch.tensor(int(i), dtype=torch.long)
        return s


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """P(score[label=1] > score[label=0]), ties counted as half."""
    pos, neg = int(labels.sum()), int((1 - labels).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks over ties so a constant predictor reads 0.5, not 1.0
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2
        i = j + 1
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_id", nargs="+", required=True)
    ap.add_argument("--rewards", nargs="+", required=True,
                    help="awr_rewards.json per dataset, positionally paired; "
                         "only the `success` flag is read from it -- episode "
                         "LENGTH comes from the dataset itself, so a sidecar "
                         "whose `steps` disagrees cannot corrupt the targets.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--horizon", type=int, default=50,
                    help="MUST match the policy's horizon: the scorer is fed "
                         "the policy's own chunks at selection time.")
    ap.add_argument("--max_shift", type=int, default=20,
                    help="Hard negative = the action chunk from the same "
                         "observation, shifted this many steps at most. Right "
                         "plan, wrong phase -- closing the gripper before the "
                         "hand has arrived. A negative drawn from another "
                         "scene would be trivially separable and would teach "
                         "nothing about timing.")
    ap.add_argument("--min_shift", type=int, default=4)
    ap.add_argument("--cap", type=int, default=300,
                    help="Steps-to-success is normalised by this and clamped. "
                         "Failed episodes take the value 1.0 throughout.")
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--vis_tokens", type=int, default=36)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--rank_weight", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_batches", type=int, default=40)
    ap.add_argument("--auc_frames", type=int, default=40,
                    help="The episode-outcome AUC uses only this many frames "
                         "from the START of each episode. Failures run 149 "
                         "chunks against successes' 31, so an episode-mean "
                         "score measures LENGTH; restricting to the head asks "
                         "whether the signal PREDICTS the outcome before "
                         "anything has gone wrong. 40 frames = 20 chunks at "
                         "n_action_steps 2.")
    ap.add_argument("--val_every_nth_episode", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--video_backend", default=None)
    args = ap.parse_args()

    if len(args.rewards) != len(args.dataset_id):
        print("ERROR: --rewards and --dataset_id are positional pairs and must "
              "be the same length", file=sys.stderr)
        return 1

    device = pick_device()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"device {device}")

    window = args.horizon + args.max_shift
    datasets, tr_idx, va_idx, meta_rows = [], [], [], []
    offset = 0
    for did, rpath in zip(args.dataset_id, args.rewards):
        probe = LeRobotDataset(did, revision="main",
                               **({} if args.video_backend is None
                                  else {"video_backend": args.video_backend}))
        fps = int(getattr(probe.meta, "fps", 10) or 10)
        cams = sorted(k for k in probe.meta.features if k.startswith("observation.images."))
        dt = {
            "action": [i / fps for i in range(window)],
            "observation.state": [0.0],
            **{c: [0.0] for c in cams},
        }
        ds = LeRobotDataset(did, delta_timestamps=dt, revision="main",
                            **({} if args.video_backend is None
                               else {"video_backend": args.video_backend}))
        side = json.loads(Path(rpath).read_text())
        ok = {int(e["episode_index"]): bool(e["success"]) for e in side["episodes"]}

        E = ds.meta.episodes
        eid = np.asarray(E["episode_index"], dtype=np.int64)
        fr = np.asarray(E["dataset_from_index"], dtype=np.int64)
        to = np.asarray(E["dataset_to_index"], dtype=np.int64)
        o = np.argsort(eid); eid, fr, to = eid[o], fr[o], to[o]
        missing = [int(e) for e in eid if int(e) not in ok]
        if missing:
            print(f"ERROR: {rpath} has no entry for {len(missing)} episode(s) of "
                  f"{did}, e.g. {missing[:5]}. The sidecar and the dataset are "
                  f"not from the same collection run.", file=sys.stderr)
            return 1

        n_ok = 0
        for rank, (e, s, t) in enumerate(zip(eid, fr, to)):
            succ = ok[int(e)]; n_ok += succ
            T = int(t - s)
            for i, row in enumerate(range(int(s), int(t))):
                tgt = 1.0 if not succ else min(1.0, max(0.0, (T - 1 - i) / args.cap))
                meta_rows.append((offset + row, tgt, i, len(datasets), int(e), int(succ)))
            (va_idx if rank % args.val_every_nth_episode == 0 else tr_idx).extend(
                range(offset + int(s), offset + int(t)))
        print(f"[{did}] {len(eid)} episodes ({n_ok} success / {len(eid) - n_ok} "
              f"failure), {len(ds)} frames, fps {fps}, cams {len(cams)}")
        datasets.append(ds); offset += len(ds)
        if len(datasets) == 1:
            first_cams, first_stats = cams, ds.meta.stats

    full = _RowIndexed(datasets[0] if len(datasets) == 1
                       else torch.utils.data.ConcatDataset(datasets))
    tgt = np.zeros(offset, dtype=np.float32)
    fidx = np.zeros(offset, dtype=np.int64)
    epix = np.zeros(offset, dtype=np.int64)
    succ_of = np.zeros(offset, dtype=np.int64)
    for row, t, i, dsi, e, sc in meta_rows:
        tgt[row], fidx[row], epix[row] = t, i, e + 100000 * dsi
        succ_of[row] = sc
    print(f"train {len(tr_idx)} frames / val {len(va_idx)} frames "
          f"({len(va_idx) / offset:.1%} held out, split BY EPISODE)")

    st = first_stats
    a_mean = torch.tensor(np.asarray(st["action"]["mean"], np.float32))
    a_std = torch.tensor(np.asarray(st["action"]["std"], np.float32)).clamp_min(1e-6)
    s_mean = torch.tensor(np.asarray(st["observation.state"]["mean"], np.float32))
    s_std = torch.tensor(np.asarray(st["observation.state"]["std"], np.float32)).clamp_min(1e-6)

    model = ActionScorer(horizon=args.horizon, action_dim=a_mean.numel(),
                         state_dim=s_mean.numel(), n_cams=len(first_cams),
                         input_size=args.input_size, vis_tokens=args.vis_tokens,
                         d_model=args.d_model, n_layers=args.n_layers).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ActionScorer {n_par / 1e6:.1f}M trainable params, "
          f"seq len {len(first_cams) * args.vis_tokens + 2 + args.horizon}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (
        (s + 1) / args.warmup if s < args.warmup
        else 0.5 * (1 + np.cos(np.pi * (s - args.warmup) / max(1, args.steps - args.warmup)))))

    def loaders():
        mk = lambda idx, sh: DataLoader(
            Subset(full, idx), batch_size=args.batch_size, shuffle=sh,
            num_workers=args.num_workers, drop_last=sh, persistent_workers=args.num_workers > 0)
        return mk(tr_idx, True), mk(va_idx, False)

    train_dl, val_dl = loaders()

    # A single-element delta_timestamps window comes back WITHOUT a time
    # dimension, so a camera is (B, 3, H, W) and not (B, 1, 3, H, W). The repo
    # already handles this -- smolvlm_encoder.py:291 is `imgs[:, -1] if
    # imgs.dim() == 5 else imgs`. Indexing [:, 0] unconditionally takes the
    # CHANNEL off an image (which at least crashes) and takes one SCALAR off
    # the state vector (which does not: it trains silently on garbage).
    _last = lambda x, nd: (x[:, -1] if x.dim() == nd + 1 else x)
    shapes_reported = []

    def unpack(b):
        imgs = torch.stack([_last(b[c], 4) for c in first_cams], dim=1)
        state = _last(b["observation.state"], 2)
        A = b["action"]
        if not shapes_reported:
            shapes_reported.append(True)
            print(f"[shapes] images {tuple(imgs.shape)}  state {tuple(state.shape)}  "
                  f"action {tuple(A.shape)}")
            if imgs.dim() != 5 or state.dim() != 2 or A.dim() != 3:
                raise ValueError(
                    f"expected images (B,C,3,H,W), state (B,D), action (B,W,A); "
                    f"got {tuple(imgs.shape)}, {tuple(state.shape)}, {tuple(A.shape)}")
            if A.shape[1] != window:
                raise ValueError(
                    f"action window is {A.shape[1]}, expected {window} "
                    f"(--horizon {args.horizon} + --max_shift {args.max_shift}); "
                    f"the hard negative would read past the end.")
        rows = b["_row"].numpy()
        return (imgs.to(device), ((state - s_mean) / s_std).to(device),
                ((A - a_mean) / a_std).to(device),
                torch.tensor(tgt[rows], device=device), rows)

    def evaluate():
        model.eval()
        P, T, R, ok_pair, n_pair = [], [], [], 0, 0
        with torch.no_grad():
            for i, b in enumerate(val_dl):
                if i >= args.eval_batches:
                    break
                imgs, state, A, y, rows = unpack(b)
                obs = model.encode_obs(imgs, state)
                s_pos = model.score(obs, A[:, :args.horizon])
                # Deterministic shift schedule: a random one makes rank_acc
                # jitter between validation passes for reasons that have
                # nothing to do with the model.
                d = args.min_shift + (i * 7) % max(1, args.max_shift - args.min_shift + 1)
                s_neg = model.score(obs, A[:, d:d + args.horizon])
                ok_pair += int((s_pos < s_neg).sum()); n_pair += len(s_pos)
                P.append(s_pos.cpu().numpy()); T.append(y.cpu().numpy()); R.append(rows)
        P, T, R = np.concatenate(P), np.concatenate(T), np.concatenate(R)
        head = fidx[R] < args.auc_frames
        per_ep, lab = {}, {}
        for p, r in zip(P[head], R[head]):
            per_ep.setdefault(epix[r], []).append(p)
            lab[epix[r]] = 1 - succ_of[r]          # label 1 = the episode FAILED
        keys = sorted(per_ep)
        a = (auc(np.array([np.mean(per_ep[k]) for k in keys]),
                 np.array([lab[k] for k in keys])) if len(keys) > 1 else float("nan"))
        model.train()
        return ok_pair / max(n_pair, 1), spearman(P, T), a, len(keys)

    model.train()
    step, t0 = 0, time.time()
    while step < args.steps:
        for b in train_dl:
            if step >= args.steps:
                break
            imgs, state, A, y, _ = unpack(b)
            obs = model.encode_obs(imgs, state)
            s_pos = model.score(obs, A[:, :args.horizon])
            d = int(np.random.randint(args.min_shift, args.max_shift + 1))
            s_neg = model.score(obs, A[:, d:d + args.horizon])
            l_reg = F.mse_loss(s_pos, y)
            l_rank = F.relu(args.margin - (s_neg - s_pos)).mean()
            loss = l_reg + args.rank_weight * l_rank
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1

            if step % 100 == 0:
                print(f"step {step:6d}  loss {loss.item():.4f}  "
                      f"reg {l_reg.item():.4f}  rank {l_rank.item():.4f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  "
                      f"{(time.time() - t0) / step:.2f}s/step", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                ra, sp, au, ne = evaluate()
                verdict = ("ACTION-SENSITIVE" if ra >= 0.65 else
                           "WEAK -- selection will be near-random" if ra >= 0.55 else
                           "DEAD: it has learned V(o) and ignores the action")
                print(f"  [val @ {step}]  rank_acc {ra:.3f}  <- {verdict}\n"
                      f"                 spearman {sp:.3f}   episode auc {au:.3f} "
                      f"(n={ne}, first {args.auc_frames} frames)", flush=True)
                torch.save({"model": model.state_dict(), "args": vars(args),
                            "a_mean": a_mean, "a_std": a_std,
                            "s_mean": s_mean, "s_std": s_std,
                            "cams": first_cams, "step": step},
                           out / "scorer.pt")
    print(f"\nsaved {out / 'scorer.pt'}")
    print("Wire it into eval only if rank_acc >= 0.65. Below that the scorer "
          "cannot tell two candidate chunks apart and best-of-N reduces to "
          "picking at random -- with the added risk that a biased ranker makes "
          "the policy quasi-deterministic and destroys the re-draw rescue that "
          "is worth 25 points.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
