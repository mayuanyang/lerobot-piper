#!/usr/bin/env python
"""Rewrite VLABench into LIBERO's observation and action space.

WHY. The two datasets look compatible -- both panda, both 10 fps, both a
7-vector called `actions` -- and they are not. Measured on the real files:

    VLABench   corr(a_t, s_{t+1})[xyz] = 0.99  0.98  0.91
               mean |a_t - s_{t+1}|    = 1mm   6mm   10mm
    LIBERO     corr(a_t, s_{t+1})[xyz] = -0.08 0.24  0.00
               corr(a_t, ds_t)[xyz]    = 0.96  0.99  0.99

VLABench's action is the ABSOLUTE next end-effector pose. LIBERO's is a
controller-normalised DELTA. Pretraining an action head on one and
fine-tuning it on the other trains the decoder on the wrong distribution --
and nothing raises, because both are float32 (7,).

Four more differences that each fail silently rather than loudly:

  * KEY NAMES. VLABench calls them `image` / `state` / `actions`, with no
    `observation.` prefix. lerobot's dataset_to_policy_features drops any key
    that does not start with one -- `state` never becomes a policy feature,
    train_wilro falls back to its default state_dim=7, and the model then
    raises KeyError on batch["observation.state"] far from the cause.
  * ROTATION WRAPS. rpy is stored in (-pi, pi] and flips sign mid-episode
    (+3.141 -> -3.142 is the SAME angle). A plain difference turns that into a
    -6.28 delta. VLABench's state[3] has std 2.87, nearly all of it wrap.
  * GRIPPER SIGN. VLABench action 1 = open, LIBERO action -1 = open. The two
    are anti-correlated (corr(a[6], s[6]) = -0.957 on the source).
  * STATE WIDTH. VLABench 7 = (xyz, rpy, gripper in {0,1}); LIBERO 8 =
    (xyz, rpy, finger1, finger2) with the fingers at +-0.040 when open.

WHAT IT DOES NOT DO. It does not re-encode video. Both datasets are v3.0 with
one video file per episode per camera, so `image` -> `observation.images.image`
and `wrist_image` -> `observation.images.image2` are directory renames, and
`second_image` is simply not carried over (LIBERO has two cameras; wilro has no
per-camera weights, so this is about matching the token budget, not the shapes).
That keeps the whole conversion to metadata and parquet.

    python src/convert_vlabench_to_libero.py \
        --src VLABench/vlabench_primitive_ft_lerobot_video \
        --out /content/vlabench_libero_space

Read the ACCEPTANCE block it prints at the end. It re-runs the same two
correlations on the OUTPUT: corr(a, ds) must land near LIBERO's 0.96-0.99 and
corr(a, s_next) near zero. If it does not, the conversion is wrong and the
number to look at is the one that moved.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

# Source -> destination camera keys. `second_image` is deliberately absent.
CAMERA_MAP = {
    "image": "observation.images.image",
    "wrist_image": "observation.images.image2",
}
# LIBERO's finger separation when the gripper is open, from its own stats
# (finger1 max 0.0402, finger2 min -0.0405).
FINGER_OPEN = 0.040


def wrap(x):
    """Angle difference into (-pi, pi]. The whole reason rotation needs care."""
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def convert_frame_block(state, actions, scale_t, scale_r, clip=1.0):
    """(N,7) VLABench state+action -> (N,8) LIBERO state, (N,7) LIBERO action.

    The action is `a_t - s_t`, NOT `s_{t+1} - s_t`: the commanded delta is what
    a policy must output, and it is defined on the last frame too. They differ
    by the controller's tracking error, which is exactly the part a policy
    cannot be asked to predict from the previous state.
    """
    state = np.asarray(state, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)

    d_t = (actions[:, 0:3] - state[:, 0:3]) * scale_t
    d_r = wrap(actions[:, 3:6] - state[:, 3:6]) * scale_r
    # VLABench gripper is binary {0,1} with 1 = OPEN on the action channel
    # (verified: state goes 0->1 one step AFTER the action goes 1->0).
    # LIBERO uses -1 = open, +1 = close.
    g = 1.0 - 2.0 * actions[:, 6:7]

    out_a = np.concatenate([d_t, d_r, g], axis=1)
    n_clipped = int((np.abs(out_a[:, :6]) > clip).sum())
    out_a[:, :6] = np.clip(out_a[:, :6], -clip, clip)

    # state gripper: 0 = open, 1 = closed -> two fingers, LIBERO's convention.
    sg = state[:, 6:7]
    f1 = FINGER_OPEN * (1.0 - sg)
    out_s = np.concatenate([state[:, 0:6], f1, -f1], axis=1)
    return out_s.astype(np.float32), out_a.astype(np.float32), n_clipped


def _data_files(root: Path):
    """Sorted the way lerobot concatenates them: by (chunk_index, file_index).

    Lexicographic path order agrees only while the indices stay zero-padded to
    the same width. Sorting on the integers removes that dependency, and the
    row offsets computed below are only correct in lerobot's own order.
    """
    def key(p: Path):
        return (int(p.parent.name.split("-")[-1]), int(p.stem.split("-")[-1]))
    return sorted((root / "data").rglob("*.parquet"), key=key)


def fit_scale(root: Path, ref_std, max_files: int = 40):
    """Pick the two scales so converted actions match the reference's spread.

    A single gearing constant does not exist: LIBERO's own |a|/|ds| ranges
    73-125 across episodes because the controller is proportional and the
    achieved motion lags the command. What CAN be matched is the marginal
    spread, and that is also what matters downstream -- both trainers
    normalise actions MEAN_STD per dataset, so only the distribution SHAPE
    survives into the model.
    """
    import pandas as pd
    files = _data_files(root)[:max_files]
    if not files:
        raise SystemExit(f"no data/*.parquet under {root}")
    dt, dr = [], []
    for f in files:
        df = pd.read_parquet(f, columns=["state", "actions"])
        S = np.stack(df["state"].to_numpy()).astype(np.float64)
        A = np.stack(df["actions"].to_numpy()).astype(np.float64)
        dt.append(A[:, 0:3] - S[:, 0:3])
        dr.append(wrap(A[:, 3:6] - S[:, 3:6]))
    dt = np.concatenate(dt); dr = np.concatenate(dr)
    st = float(np.mean(ref_std[0:3]) / max(np.mean(dt.std(0)), 1e-9))
    sr = float(np.mean(ref_std[3:6]) / max(np.mean(dr.std(0)), 1e-9))
    print(f"[scale] fitted on {len(files)} file(s): translation x{st:.1f}, "
          f"rotation x{sr:.1f}\n"
          f"        raw delta std  t={dt.std(0).round(5)}  r={dr.std(0).round(5)}\n"
          f"        reference std  t={np.round(ref_std[0:3], 4)}  "
          f"r={np.round(ref_std[3:6], 4)}")
    return st, sr


def reference_action_std(ref: str):
    """Per-dim action std of the LIBERO set we are matching."""
    from huggingface_hub import hf_hub_download
    s = json.load(open(hf_hub_download(ref, "meta/stats.json", repo_type="dataset")))
    return np.asarray(s["action"]["std"], dtype=np.float64).ravel()


def link_or_copy(src: Path, dst: Path):
    """Hardlink, else symlink, else copy. 5000 videos per camera is tens of GB
    and none of it changes -- duplicating the bytes is the one avoidable cost
    in this conversion."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "skip"
    real = src.resolve()          # snapshot_download hands back symlinks
    try:
        os.link(real, dst); return "hardlink"
    except OSError:
        pass
    try:
        os.symlink(real, dst); return "symlink"
    except OSError:
        shutil.copy2(real, dst); return "copy"


def acceptance(out_root: Path, n_files: int = 3):
    """Re-run the diagnostic that found the problem, on the OUTPUT.

    Not a unit test -- the actual claim. If corr(a, ds) is not high and
    corr(a, s_next) is not near zero, the output is still in the wrong space.
    """
    import pandas as pd
    files = _data_files(out_root)[:n_files]
    ca_ds, ca_sn, gvals = [], [], set()
    for f in files:
        df = pd.read_parquet(f)
        for e in df.episode_index.unique():
            ep = df[df.episode_index == e]
            if len(ep) < 8:
                continue
            S = np.stack(ep["observation.state"].to_numpy()).astype(np.float64)
            A = np.stack(ep["action"].to_numpy()).astype(np.float64)
            ds = S[1:, :3] - S[:-1, :3]
            for i in range(3):
                if ds[:, i].std() > 1e-9 and A[:-1, i].std() > 1e-9:
                    ca_ds.append(np.corrcoef(A[:-1, i], ds[:, i])[0, 1])
                if S[1:, i].std() > 1e-9 and A[:-1, i].std() > 1e-9:
                    ca_sn.append(np.corrcoef(A[:-1, i], S[1:, i])[0, 1])
            gvals |= set(np.unique(A[:, 6]).tolist())
    print("\n" + "=" * 68)
    print("ACCEPTANCE  (LIBERO reference: corr(a,ds) 0.96-0.99, corr(a,s_next) ~0)")
    print(f"  corr(a_t, ds_t)     mean {np.mean(ca_ds):+.3f}   "
          f"median {np.median(ca_ds):+.3f}   n={len(ca_ds)}")
    print(f"  corr(a_t, s_t+1)    mean {np.mean(ca_sn):+.3f}   "
          f"median {np.median(ca_sn):+.3f}   n={len(ca_sn)}")
    print(f"  gripper action values: {sorted(gvals)}   (LIBERO: [-1.0, 1.0])")
    ok = np.median(ca_ds) > 0.85 and abs(np.median(ca_sn)) < 0.35
    print(f"  action space: {'PASS' if ok else 'FAIL -- NOT in LIBERO action space'}")
    ok = _load_with_deltas(out_root) and ok
    print("=" * 68)
    return ok


def _load_with_deltas(out_root: Path, horizon: int = 16, obs: int = 8):
    """Open the output the way the TRAINER does, with delta_timestamps.

    Not the same test as loading it plainly. _get_query_indices is only reached
    when deltas are configured, and it is the sole consumer of
    dataset_from_index / dataset_to_index -- so a dataset with broken episode
    spans loads, indexes and prints perfectly right up until a trainer asks for
    an action chunk. That is exactly how the source's bad offsets reached a
    training run here.
    """
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as e:
        print(f"  span check: SKIPPED (lerobot not importable: {e})")
        return True
    info = json.load(open(out_root / "meta" / "info.json"))
    ft = 1.0 / float(info["fps"])
    dt = {"observation.state": [-i * ft for i in range(obs)][::-1],
          "action": [i * ft for i in range(horizon)],
          **{k: [0.0] for k in info["features"] if k.startswith("observation.images")}}
    try:
        ds = LeRobotDataset("local/acceptance", root=str(out_root),
                            delta_timestamps=dt, tolerance_s=max(0.005, ft / 2))
        n = len(ds)
        # STRUCTURAL, not sampled. Sampling a few frames does not detect this:
        # on a small subset `length * episode_index` stays inside the table and
        # every probe loads fine, so the first version of this check passed on
        # deliberately corrupted metadata. The spans must TILE the table.
        E = ds.meta.episodes
        fr = np.asarray(E["dataset_from_index"], dtype=np.int64)
        to = np.asarray(E["dataset_to_index"], dtype=np.int64)
        ln = np.asarray(E["length"], dtype=np.int64)
        order = np.argsort(np.asarray(E["episode_index"], dtype=np.int64))
        fr, to, ln = fr[order], to[order], ln[order]
        bad = []
        if fr[0] != 0:
            bad.append(f"first episode starts at {fr[0]}, not 0")
        if not (to[:-1] == fr[1:]).all():
            k = int((to[:-1] != fr[1:]).sum())
            bad.append(f"{k} of {len(fr) - 1} episode boundaries leave a gap or overlap")
        if to[-1] != n:
            bad.append(f"last episode ends at {to[-1]}, table has {n} rows")
        if not (to - fr == ln).all():
            bad.append("length disagrees with to - from")
        if not bad:
            # And the spans must hold the rows they claim. Metadata can tile
            # perfectly while pointing at the wrong rows.
            for i in list(range(min(3, len(fr)))) + [len(fr) - 1]:
                eis = ds.hf_dataset.select(range(int(fr[i]), int(to[i])))["episode_index"]
                if len(set(int(x) for x in eis)) != 1:
                    bad.append(f"span {i} spills across episodes")
                    break
        if bad:
            print("  span check: FAIL -- " + "; ".join(bad) + "\n"
                  "    _get_query_indices clamps to [from, to-1], so this "
                  "kills the trainer on its first batch with an IndexError "
                  "from a DataLoader worker.")
            return False
        # Cheap smoke test on top of the structural one.
        for i in (0, n // 2, n - 1):
            item = ds[int(i)]
            assert item["action"].shape[0] == horizon, item["action"].shape
            assert item["observation.state"].shape[0] == obs
        print(f"  span check: PASS -- {len(fr)} episode spans tile "
              f"[0, {n}) and load with delta_timestamps")
        return True
    except Exception as e:
        print(f"  span check: FAIL -- {type(e).__name__}: {str(e)[:180]}")
        return False


def convert(src: str, out: str, reference: str, scale_t=None, scale_r=None,
            clip: float = 1.0, src_root: str = "", limit_files: int = 0):
    import pandas as pd
    from huggingface_hub import snapshot_download

    if src_root:
        root = Path(src_root)
        if not root.exists():
            raise SystemExit(f"--src_root {root} does not exist")
        print(f"[src] {root}  (--src_root)")
    else:
        # Try the cache first. snapshot_download would otherwise stat all
        # ~19k files against the hub before deciding it has everything, and
        # this dataset is normally already on disk by the time anyone converts
        # it. local_files_only never touches the network.
        try:
            root = Path(snapshot_download(src, repo_type="dataset",
                                          local_files_only=True))
            print(f"[src] {root}  (cache, no download)")
        except Exception:
            print(f"[src] not fully cached; downloading {src} "
                  f"(videos included; this is the slow part)")
            root = Path(snapshot_download(src, repo_type="dataset"))
            print(f"[src] {root}")

    # A cache can be complete enough to open and still be missing files. Say so
    # here rather than letting the episode count quietly come out short.
    _info = json.load(open(root / "meta" / "info.json"))
    n_data = len(_data_files(root))
    n_vid = {k: len(list((root / "videos" / k).rglob("*.mp4")))
             for k in CAMERA_MAP if (root / "videos" / k).exists()}
    print(f"[src] {n_data} data file(s), videos {n_vid}, "
          f"info.json declares {_info.get('total_episodes')} episodes / "
          f"{_info.get('total_frames')} frames")
    if any(v < _info.get("total_episodes", 0) for v in n_vid.values()):
        print("      NOTE: fewer videos than episodes -- the cache is partial. "
              "Episodes without data are dropped and any kept episode with a "
              "missing video aborts the run below.")
    out_root = Path(out)
    out_root.mkdir(parents=True, exist_ok=True)

    info = json.load(open(root / "meta" / "info.json"))
    feats = info["features"]
    missing = [k for k in list(CAMERA_MAP) + ["state", "actions"] if k not in feats]
    if missing:
        raise SystemExit(
            f"source is not the expected VLABench layout: missing {missing}.\n"
            f"  has: {sorted(feats)}")

    if scale_t is None or scale_r is None:
        ref_std = reference_action_std(reference)
        ft, fr = fit_scale(root, ref_std)
        scale_t = ft if scale_t is None else scale_t
        scale_r = fr if scale_r is None else scale_r
    print(f"[scale] using translation x{scale_t:.1f}, rotation x{scale_r:.1f}")

    # ---- data parquet ----
    files = _data_files(root)
    if limit_files:
        files = files[:limit_files]
    n_rows = n_clip = 0
    covered: set = set()          # episode_index values actually written
    # TRUE row count per episode, in lerobot's concatenation order. The source
    # ships dataset_from_index = length * episode_index instead of a running
    # sum -- see the repair below -- so nothing downstream may trust the
    # published offsets, and counting the rows here is the only authority.
    ep_rows: dict = {}
    ep_order: list = []
    last_ep = -1
    acc = {"observation.state": [], "action": []}
    for i, f in enumerate(files):
        df = pd.read_parquet(f)
        S = np.stack(df["state"].to_numpy())
        A = np.stack(df["actions"].to_numpy())
        s_out, a_out, nc = convert_frame_block(S, A, scale_t, scale_r, clip)
        df = df.drop(columns=["state", "actions"])
        df["observation.state"] = list(s_out)
        df["action"] = list(a_out)
        rel = f.relative_to(root)
        (out_root / rel).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_root / rel, index=False)
        n_rows += len(df); n_clip += nc
        acc["observation.state"].append(s_out)
        acc["action"].append(a_out)
        for e, c in df["episode_index"].value_counts().sort_index().items():
            e = int(e)
            if e not in ep_rows:
                ep_rows[e] = 0
                ep_order.append(e)
                if e < last_ep:
                    raise SystemExit(
                        f"episode {e} appears after {last_ep} in the data "
                        f"concatenation; row offsets cannot be derived from "
                        f"file order for this source.")
                last_ep = e
            ep_rows[e] += int(c)
        covered.update(ep_rows)
        if i % 200 == 0 or i == len(files) - 1:
            print(f"  data {i + 1}/{len(files)}  rows={n_rows}", flush=True)
    pct = 100.0 * n_clip / max(n_rows * 6, 1)
    print(f"[clip] {n_clip} of {n_rows * 6} pose components hit +-{clip} ({pct:.2f}%)"
          + ("  <-- high; the fitted scale is too large" if pct > 5 else ""))

    # ---- meta/episodes: rename the video columns, and DROP any episode whose
    # data was not converted. Without this, --limit_files leaves a dataset
    # whose metadata promises 5000 episodes over data that covers a handful,
    # and the failure surfaces later as a missing-file error during training
    # rather than here.
    #
    # It ALSO rebuilds dataset_from_index / dataset_to_index, because the
    # source's are wrong. Published VLABench ships
    #     dataset_from_index = length * episode_index
    # instead of a running sum, so the spans do not tile the table: 4945 of
    # 4999 boundaries have gaps and the maximum reaches 947551 against 575101
    # actual rows. _get_query_indices clamps to [from, to-1], so training dies
    # on the first batch with
    #     IndexError: Invalid key: 863104 is out of bounds for size 575101
    # -- and ONLY when delta_timestamps are in play. A plain
    # LeRobotDataset[i] never consults the span, which is why this survived an
    # end-to-end load test and surfaced in the trainer instead.
    #
    # The lengths themselves are correct and the rows are in episode order, so
    # the true offsets are the prefix sums of the counts observed above.
    starts, n = {}, 0
    for e in ep_order:
        starts[e] = n
        n += ep_rows[e]
    keep_videos: set = set()
    n_ep_in = n_ep_out = n_span_fixed = n_len_fixed = 0
    for f in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        df = pd.read_parquet(f)
        n_ep_in += len(df)
        df = df[df["episode_index"].isin(covered)].copy()
        n_ep_out += len(df)
        ei = df["episode_index"].astype(int)
        new_from = ei.map(starts)
        new_len = ei.map(ep_rows)
        n_span_fixed += int((df["dataset_from_index"].to_numpy() != new_from.to_numpy()).sum())
        n_len_fixed += int((df["length"].to_numpy() != new_len.to_numpy()).sum())
        df["dataset_from_index"] = new_from.to_numpy()
        df["length"] = new_len.to_numpy()
        df["dataset_to_index"] = (new_from + new_len).to_numpy()
        for dst_key, src_key in ((d, s) for s, d in CAMERA_MAP.items()):
            ci, fi = f"videos/{src_key}/chunk_index", f"videos/{src_key}/file_index"
            if ci in df.columns:
                for c, v in zip(df[ci].to_numpy(), df[fi].to_numpy()):
                    keep_videos.add((src_key, int(c), int(v)))
        drop = [c for c in df.columns if c.startswith("videos/second_image/")]
        ren = {c: c.replace(f"videos/{s}/", f"videos/{d}/")
               for c in df.columns for s, d in CAMERA_MAP.items()
               if c.startswith(f"videos/{s}/")}
        df = df.drop(columns=drop).rename(columns=ren)
        rel = f.relative_to(root)
        (out_root / rel).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_root / rel, index=False)
    print(f"[meta] episodes {n_ep_out}/{n_ep_in} kept "
          f"(those whose data was converted); "
          f"{n_span_fixed} row-offset(s) and {n_len_fixed} length(s) rebuilt "
          f"from the actual data")
    if n != n_rows:
        raise SystemExit(
            f"internal: episode row counts sum to {n} but {n_rows} rows were "
            f"written. The rebuilt offsets would not match the table.")
    shutil.copy2(root / "meta" / "tasks.parquet", out_root / "meta" / "tasks.parquet")

    # ---- videos: rename the directory, do not touch the bytes ----
    modes, skipped = {}, 0
    for src_key, dst_key in CAMERA_MAP.items():
        sd = root / "videos" / src_key
        if not sd.exists():
            raise SystemExit(f"missing video dir {sd}")
        for v in sorted(sd.rglob("*.mp4")):
            ci = int(v.parent.name.split("-")[-1])
            fi = int(v.stem.split("-")[-1])
            if (src_key, ci, fi) not in keep_videos:
                skipped += 1
                continue
            m = link_or_copy(v, out_root / "videos" / dst_key / v.relative_to(sd))
            modes[m] = modes.get(m, 0) + 1
    print(f"[videos] {modes}, {skipped} not referenced "
          f"(second_image not carried over)")
    missing = sorted(k for k in keep_videos
                     if not (out_root / "videos" / CAMERA_MAP[k[0]] /
                             f"chunk-{k[1]:03d}" / f"file-{k[2]:03d}.mp4").exists())
    if missing:
        raise SystemExit(
            f"{len(missing)} video file(s) referenced by the kept episodes are "
            f"absent from the source, e.g. {missing[:3]}. The output would "
            f"train fine until the loader reached one. Re-download the source.")

    # ---- meta/info.json ----
    new_feats = {}
    for k, v in feats.items():
        if k == "second_image":
            continue
        if k in CAMERA_MAP:
            new_feats[CAMERA_MAP[k]] = v
        elif k == "state":
            new_feats["observation.state"] = {**v, "shape": [8], "names": ["state"]}
        elif k == "actions":
            new_feats["action"] = {**v, "names": ["action"]}
        else:
            new_feats[k] = v
    info["features"] = new_feats
    info["total_episodes"] = n_ep_out
    info["total_frames"] = n_rows
    info["splits"] = {"train": f"0:{n_ep_out}"}
    info["conversion"] = {
        "source": src, "action_scale_translation": scale_t,
        "action_scale_rotation": scale_r, "clip": clip,
        "note": "absolute EEF pose -> LIBERO-style delta; rpy pi-wrapped; "
                "gripper 1-2g; state 7 -> 8 fingers; second_image dropped",
    }
    json.dump(info, open(out_root / "meta" / "info.json", "w"), indent=2)

    # ---- meta/stats.json: recomputed, NOT copied ----
    stats = json.load(open(root / "meta" / "stats.json"))
    for k in ("state", "actions"):
        stats.pop(k, None)
    stats.pop("second_image", None)
    for s, d in CAMERA_MAP.items():
        if s in stats:
            stats[d] = stats.pop(s)
    for k, arrs in acc.items():
        x = np.concatenate(arrs).astype(np.float64)
        stats[k] = {"mean": x.mean(0).tolist(), "std": x.std(0).tolist(),
                    "min": x.min(0).tolist(), "max": x.max(0).tolist(),
                    "count": [int(len(x))]}
    json.dump(stats, open(out_root / "meta" / "stats.json", "w"), indent=2)
    print(f"[meta] info.json + stats.json rewritten; "
          f"state dim {len(stats['observation.state']['mean'])}, "
          f"action dim {len(stats['action']['mean'])}")

    return acceptance(out_root)


def self_test():
    """Synthetic round trip, so the transform is checked without the network."""
    print("convert_vlabench_to_libero self-test")
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    rng = np.random.default_rng(0)
    n = 200
    # A trajectory that crosses the +-pi seam, which is the case a naive
    # difference gets wrong.
    xyz = np.cumsum(rng.normal(0, 0.005, (n, 3)), axis=0) + 0.3
    rpy = np.zeros((n, 3))
    rpy[:, 0] = np.linspace(3.10, 3.30, n)          # walks past pi
    rpy[:, 0] = wrap(rpy[:, 0])
    g = (np.arange(n) > n // 2).astype(float)[:, None]
    S = np.concatenate([xyz, rpy, g], axis=1)
    # The action must be the NEXT pose plus a small tracking error, not the
    # current one: the source has |a_t - s_{t+1}| ~ 1mm against |ds| ~ 2mm.
    # Generating a_t = s_t + noise instead makes a_t - s_t pure noise, and the
    # corr(a, ds) check below then fails on the GENERATOR while the transform
    # is fine -- which is exactly what it did the first time.
    a_xyz = np.vstack([xyz[1:], xyz[-1:]]) + rng.normal(0, 0.0005, (n, 3))
    a_rpy = wrap(np.vstack([rpy[1:], rpy[-1:]]))
    A = np.concatenate([a_xyz, a_rpy, 1.0 - g], axis=1)

    s_out, a_out, nc = convert_frame_block(S, A, 80.0, 5.0)
    check("state widens 7 -> 8", s_out.shape[1] == 8)
    check("action stays 7", a_out.shape[1] == 7)
    check("fingers open when state gripper is 0",
          abs(s_out[0, 6] - FINGER_OPEN) < 1e-6 and abs(s_out[0, 7] + FINGER_OPEN) < 1e-6)
    check("fingers closed when state gripper is 1",
          abs(s_out[-1, 6]) < 1e-6 and abs(s_out[-1, 7]) < 1e-6)
    check("gripper action is exactly {-1,+1}",
          set(np.unique(a_out[:, 6]).tolist()) <= {-1.0, 1.0})
    check("action gripper -1 (open) where VLABench said 1",
          a_out[0, 6] == -1.0 and A[0, 6] == 1.0)
    # The seam: a plain difference would produce ~-2pi somewhere.
    naive = (A[:, 3:6] - S[:, 3:6]) * 5.0
    check("naive difference DOES blow up at the seam", np.abs(naive).max() > 25)
    check("wrapped rotation delta stays small", np.abs(a_out[:, 3:6]).max() < 1.0)
    check("clip counter agrees with the data",
          nc == int((np.abs(np.concatenate([
              (A[:, 0:3] - S[:, 0:3]) * 80.0,
              wrap(A[:, 3:6] - S[:, 3:6]) * 5.0], axis=1)) > 1.0).sum()))
    # The property the whole conversion exists for.
    ds = s_out[1:, :3] - s_out[:-1, :3]
    c = [np.corrcoef(a_out[:-1, i], ds[:, i])[0, 1] for i in range(3)]
    check(f"corr(a, ds) is high on all three axes {np.round(c, 2)}", min(c) > 0.7)
    print("ALL PASS" if ok else "FAILURES ABOVE")
    return ok


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="VLABench/vlabench_primitive_ft_lerobot_video")
    ap.add_argument("--src_root", default="",
                    help="Already-downloaded copy; skips snapshot_download.")
    ap.add_argument("--out", help="Output dataset root.")
    ap.add_argument("--reference", default="HuggingFaceVLA/libero",
                    help="LIBERO set whose action spread the scales are fitted "
                         "to. Only its meta/stats.json is read.")
    ap.add_argument("--action_scale", type=float, default=None,
                    help="Override the fitted translation scale.")
    ap.add_argument("--rotation_scale", type=float, default=None,
                    help="Override the fitted rotation scale.")
    ap.add_argument("--clip", type=float, default=1.0,
                    help="Clip pose deltas to +-this, as LIBERO's are.")
    ap.add_argument("--limit_files", type=int, default=0,
                    help="Convert only the first N data parquet files -- for a "
                         "small feasibility run before committing to all 5000 "
                         "episodes.")
    ap.add_argument("--self_test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if not a.out:
        ap.error("--out is required (or pass --self_test)")
    ok = convert(a.src, a.out, a.reference, a.action_scale, a.rotation_scale,
                 a.clip, a.src_root, a.limit_files)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
