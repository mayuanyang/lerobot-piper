#!/usr/bin/env python
"""Does `observation.state` history alone predict the action chunk?

ARCHITECTURE.md 8.2 lists this as a known risk and names the control:

  > Motion vectors may leak the demonstrator's action, reintroducing causal
  > confusion through the back door -- the exact failure frame stacking has.
  > Check that a motion-vector-only model does *not* score above chance.

That control has never been run. This is its cheap form: no VLM, no GPU, no
images -- just least squares from the state window to the action chunk, which
upper-bounds what ANY architecture could extract from that channel alone.

THE MECHANISM. Training requests `obs_steps = motion_history_len` frames of
`observation.state` (train_wiltechs_x.build_datasets), so the model can form
`s_t - s_{t-1}` internally. Under a position controller that difference IS the
previously executed action, as an identity rather than something learned. The
label starts one step later at `a_t`, so nothing leaks directly -- but demos
are smooth, `a_t ~ a_{t-1}`, and momentum extrapolation then explains most of
the near horizon for free, with no image and no instruction.

At `n_action_steps=2` only `a_t` and `a_{t+1}` are ever executed, and those are
exactly where extrapolation is strongest. Hence the report is broken down BY
HORIZON POSITION -- an aggregate would hide the whole point.

    python probe_state_history_leak.py --dataset_id lerobot/libero
    python probe_state_history_leak.py --self_test     # no dataset needed

Read the three rows against each other:

  diff1    only s_t - s_{t-1}      pure momentum
  state1   only s_t                what --no_motion_vectors leaves
  hist8    the whole window        what the model gets today

`hist8` minus `state1` is what the history buys. If that gap is large at
k=0..1 the back door is real and load-bearing on the executed steps.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ---------------------------------------------------------------------------
# flow <-> conditional spread, so the probe's MSE is comparable to the log
# ---------------------------------------------------------------------------
def flow_of(sigma2: float, n: int = 20000) -> float:
    """Uniform-t average of the OPTIMAL flow residual at spread sigma2.

    `flow` in the training log is not an action-space MSE, so a probe MSE
    cannot be read against it directly. This is the bridge, from
    x_t = t*noise + (1-t)*a with target u_t = noise - a:

        E||u - E[u|x_t,obs]||^2 = 1 + s2 - [t - (1-t)s2]^2 / (t^2 + (1-t)^2 s2)

    At sigma2 = 0.0097 it returns 0.155, which is what 236k converged to.
    """
    t = np.linspace(1e-4, 1 - 1e-4, n)
    S = t**2 + (1 - t) ** 2 * sigma2
    return float(np.mean(1 + sigma2 - (t - (1 - t) * sigma2) ** 2 / S))


def ridge(X: np.ndarray, Y: np.ndarray, lam: float = 1e-3):
    """Closed-form, with an intercept. X (N,F), Y (N,O) -> predict(X) -> (N,O)."""
    X1 = np.concatenate([X, np.ones((len(X), 1), dtype=X.dtype)], axis=1)
    A = X1.T @ X1
    A[np.diag_indices_from(A)] += lam * len(X)
    W = np.linalg.solve(A, X1.T @ Y)
    return lambda Z: np.concatenate(
        [Z, np.ones((len(Z), 1), dtype=Z.dtype)], axis=1) @ W


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
def build_windows(state, action, ep_index, obs_steps, horizon, stride=1):
    """-> S (N, obs_steps, Ds), A (N, horizon, Da), ep (N,).

    Left-pads the state window by repeating the first frame and right-pads the
    action chunk by repeating the last, which is LeRobot's own convention and
    what the trainer's `action_is_pad` marks. Padded action steps are excluded
    from the score via the returned mask.
    """
    S, A, M, E = [], [], [], []
    for e in np.unique(ep_index):
        idx = np.flatnonzero(ep_index == e)
        s, a, T = state[idx], action[idx], len(idx)
        for t in range(0, T, stride):
            lo = t - obs_steps + 1
            w = s[max(lo, 0): t + 1]
            if lo < 0:                                   # left-pad
                w = np.concatenate([np.repeat(w[:1], -lo, 0), w])
            hi = min(t + horizon, T)
            c = a[t:hi]
            m = np.ones(horizon, dtype=np.float32)
            if len(c) < horizon:                         # right-pad
                m[len(c):] = 0.0
                c = np.concatenate([c, np.repeat(c[-1:], horizon - len(c), 0)])
            S.append(w); A.append(c); M.append(m); E.append(e)
    return (np.asarray(S, np.float64), np.asarray(A, np.float64),
            np.asarray(M, np.float32), np.asarray(E))


def load_columns(dataset_id: str, max_episodes: int | None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    p = Path(dataset_id)
    kw = dict(root=str(p)) if p.exists() else dict(revision="main")
    rid = "local/probe" if p.exists() else dataset_id
    meta = LeRobotDatasetMetadata(rid, **kw)
    # No delta_timestamps and no item access: the parquet columns are all this
    # needs, and touching __getitem__ would decode video for every frame.
    ds = LeRobotDataset(rid, **kw)
    hf = ds.hf_dataset
    state = np.asarray(hf["observation.state"], dtype=np.float64)
    action = np.asarray(hf["action"], dtype=np.float64)
    ep = np.asarray(hf["episode_index"])
    if max_episodes:
        keep = ep < (np.unique(ep)[:max_episodes].max() + 1)
        state, action, ep = state[keep], action[keep], ep[keep]
    st = meta.stats
    return state, action, ep, st


def normalize(x, st, key):
    """MEAN_STD, the same normalization prepare() applies before the loss.

    Not optional: the trainer's own risk log records that skipping it gave the
    three rotation dims 0.79% of the flow loss instead of 42.9%, so an
    unnormalized probe would measure a different quantity than `flow`.
    """
    m = np.asarray(st[key]["mean"], dtype=np.float64).reshape(-1)
    s = np.asarray(st[key]["std"], dtype=np.float64).reshape(-1)
    return (x - m) / np.maximum(s, 1e-8)


# ---------------------------------------------------------------------------
def report(S, A, M, E, args):
    Ds, Da, H = S.shape[-1], A.shape[-1], A.shape[1]
    rng = np.random.default_rng(args.seed)
    eps = np.unique(E)
    rng.shuffle(eps)
    n_te = max(1, int(0.2 * len(eps)))
    te = np.isin(E, eps[:n_te])
    tr = ~te
    print(f"\n{len(S):,} windows  ({tr.sum():,} fit / {te.sum():,} test, "
          f"split by EPISODE)   state {Ds}d  action {Da}d  horizon {H}")

    d1 = S[:, -1] - S[:, -2] if S.shape[1] > 1 else np.zeros_like(S[:, -1])
    feats = {
        "diff1   (s_t - s_{t-1})": d1,
        "state1  (s_t only)": S[:, -1],
        f"hist{S.shape[1]}   (whole window)": S.reshape(len(S), -1),
    }
    Y = A.reshape(len(A), -1)

    # cells: the flow loss's own position weighting, so the aggregate row is
    # weighted the same way `flow` is.
    pos_w = np.ones(H)
    pos_w[args.loss_exec_steps:] = args.future_steps_weight

    ks = [k for k in (0, 1, 3, 7, 15, 31, H - 1) if k < H]
    print("\n  per-position MSE in NORMALIZED action units "
          "(= sigma^2 of what the channel cannot explain)")
    print(" " * 28 + "".join(f"{'k=' + str(k):>9s}" for k in ks)
          + f"{'cells-wtd':>12s}{'~flow':>8s}")
    rows = {}
    for name, X in feats.items():
        f = ridge(X[tr], Y[tr], args.ridge)
        P = f(X[te]).reshape(-1, H, Da)
        se = ((P - A[te]) ** 2).mean(-1) * M[te]              # (N, H)
        per_k = se.sum(0) / np.maximum(M[te].sum(0), 1e-9)
        agg = float((se * pos_w).sum() / np.maximum((M[te] * pos_w).sum(), 1e-9))
        rows[name] = per_k
        print(f"  {name:<26s}" + "".join(f"{per_k[k]:9.4f}" for k in ks)
              + f"{agg:12.4f}{flow_of(agg):8.3f}")

    print(f"\n  {'the history buys':<26s}"
          + "".join(f"{rows[list(feats)[1]][k] - rows[list(feats)[2]][k]:9.4f}"
                    for k in ks)
          + "     <- state1 minus hist (positive = history helps)")
    print(f"  reference: 236k converged to flow 0.155, i.e. sigma^2 ~ 0.0097.")

    # What IS observation.state? Answer it from the data instead of guessing.
    if S.shape[1] > 1:
        print("\n  corr( s_t - s_{t-1} , a_{t-1} )  -- best state-diff dim per "
              "action dim")
        prev = A[:, 0]                       # a_t; a_{t-1} is the diff's partner
        # a_{t-1} is not in the chunk, so use the identity the other way: the
        # diff should match the PREVIOUS action, which for a window ending at t
        # is what produced s_t. Compare against a_t as the smoothness proxy and
        # say so.
        def corr(x, y):
            # A constant dim (a state channel that never moves) gives std 0 and
            # numpy warns its way to nan. Report it as "no relationship", which
            # is what it is, rather than letting a warning look like a fault.
            sx, sy = x.std(), y.std()
            return 0.0 if sx < 1e-12 or sy < 1e-12 else \
                abs(float(np.corrcoef(x, y)[0, 1]))

        for j in range(Da):
            c = [corr(d1[:, i], prev[:, j]) for i in range(Ds)]
            i = int(np.nanargmax(c))
            print(f"    action dim {j}  <- state-diff dim {i}   |r| = {c[i]:.3f}"
                  + ("   *** near-identity ***" if c[i] > 0.9 else ""))
        print("    (vs a_t, one step AFTER the diff -- so this measures the "
              "identity AND the smoothness together, which is exactly the\n"
              "     path the shortcut takes.)")


def self_test(args):
    """Synthetic data where the answer is known, so the probe can be trusted.

    Actions are pure momentum plus noise: a_{t+k} = v_t + eps, with v_t the
    current velocity. `diff1` should recover it almost exactly and `state1`
    should fail, because a single position carries no velocity.
    """
    rng = np.random.default_rng(0)
    T, E, D = 200, 40, 7
    st, ac, ep = [], [], []
    for e in range(E):
        v = rng.normal(0, 0.3, D)
        s = np.zeros(D)
        for t in range(T):
            v = 0.97 * v + rng.normal(0, 0.02, D)          # smooth velocity
            s = s + v
            st.append(np.concatenate([s, [0.0]]))          # 8d state, last dim dead
            ac.append(v + rng.normal(0, 0.05, D))          # action IS the velocity
            ep.append(e)
    S, A, M, Ep = build_windows(np.asarray(st), np.asarray(ac),
                                np.asarray(ep), args.obs_steps, args.horizon)
    print("SELF TEST -- actions are pure momentum + noise(0.05).")
    print("Expect: diff1 ~ 0.0025 at k=0 (the noise floor), state1 far worse.")
    report(S, A, M, Ep, args)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_id", default="lerobot/libero")
    ap.add_argument("--obs_steps", type=int, default=8,
                    help="motion_history_len; the window the model gets")
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--loss_exec_steps", type=int, default=16)
    ap.add_argument("--future_steps_weight", type=float, default=0.3)
    ap.add_argument("--max_episodes", type=int, default=None)
    ap.add_argument("--stride", type=int, default=2,
                    help="subsample windows; 1 is every frame")
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--self_test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test(a)

    state, action, ep, st = load_columns(a.dataset_id, a.max_episodes)
    print(f"{a.dataset_id}: {len(state):,} frames over "
          f"{len(np.unique(ep)):,} episodes")
    state = normalize(state, st, "observation.state")
    action = normalize(action, st, "action")
    S, A, M, E = build_windows(state, action, ep, a.obs_steps, a.horizon, a.stride)
    report(S, A, M, E, a)


if __name__ == "__main__":
    main()
