#!/usr/bin/env python
"""Check a `train_rft.py --rft.collect_only` dataset against what train_wiltechs_x needs.

The collected data is model-agnostic -- raw frames, raw state, ENV-SPACE actions,
task string -- so a dataset harvested with any policy is usable as SFT material
for any other. What is not automatic is whether it will MIX with the expert set:
`build_datasets` derives one set of delta_timestamps from a reference fps and
applies it to every dataset, so a dataset stamped at a different fps silently
gets a different stride for its action chunk and state history.

    python inspect_collected_dataset.py <dataset_root_or_repo_id> [--compare lerobot/libero]

Also reports the episode-LENGTH distribution, which is the thing to look at
before training on this: LIBERO scores a success the same whether it took 100
steps or 240, and this policy's failure mode is succeeding slowly. Fitting BC to
the slow successes teaches the fumbling.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_meta(spec: str):
    """revision="main" is not optional. lerobot defaults the revision to its own
    codebase version tag, which a dataset written by `--rft.collect_only` does
    not carry; the lookup then raises RevisionNotFoundError, and on current
    huggingface_hub that raise ITSELF fails with "missing keyword-only argument
    'response'" -- so the real cause is buried under a TypeError. Same value
    train_wiltechs_x.build_datasets uses.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    p = Path(spec)
    if p.exists():
        return LeRobotDatasetMetadata("local/inspect", root=str(p))
    return LeRobotDatasetMetadata(spec, force_cache_sync=True, revision="main")


def describe(meta, label: str) -> dict:
    from lerobot.datasets.utils import dataset_to_policy_features
    from models.wiltechs_x.paraphrase import instruction_strings

    feats = meta.features
    cams = sorted(k for k in feats if "image" in k)
    state = next((k for k in feats if "state" in k), None)
    # The POLICY shape, not the raw metadata shape. dataset_to_policy_features
    # rewrites (h, w, c) -> (c, h, w) when names[2] is "channel"/"channels", so
    # a set declaring (256, 256, 3) and one declaring (3, 256, 256) can be the
    # same thing downstream -- and build_datasets only ever sees this side.
    # Comparing the raw shapes reports a conflict that does not exist.
    pol = dataset_to_policy_features(feats)
    info = {
        "fps": getattr(meta, "fps", None),
        "cams": cams,
        "raw_shape": {c: tuple(feats[c]["shape"]) for c in cams},
        "names": {c: list(feats[c].get("names") or []) for c in cams},
        "cam_shape": {c: tuple(pol[c].shape) for c in cams if c in pol},
        "state_dim": int(np.prod(feats[state]["shape"])) if state else None,
        "action_dim": int(np.prod(feats["action"]["shape"])) if "action" in feats else None,
        "episodes": int(getattr(meta, "total_episodes", 0) or 0),
        "frames": int(getattr(meta, "total_frames", 0) or 0),
        "tasks": instruction_strings(meta.tasks) if getattr(meta, "tasks", None) is not None else [],
    }
    print(f"\n=== {label} ===")
    print(f"  fps            {info['fps']}")
    print(f"  episodes       {info['episodes']}   frames {info['frames']}")
    if info["episodes"]:
        print(f"  mean length    {info['frames'] / info['episodes']:.0f} frames")
    print(f"  cameras        {cams}")
    for c in cams:
        print(f"                 {c}  declared {info['raw_shape'][c]} "
              f"names={info['names'][c]}  ->  policy {info['cam_shape'].get(c)}")
    print(f"  state '{state}' dim {info['state_dim']}")
    print(f"  action dim     {info['action_dim']}")
    print(f"  tasks          {len(info['tasks'])}")
    return info


def episode_lengths(meta) -> np.ndarray | None:
    """Per-episode frame counts, if the metadata exposes them.

    lerobot 0.4.0's `meta.episodes` is an HF `datasets.Dataset` with a "length"
    column (and dataset_from_index/dataset_to_index). It is neither a DataFrame
    nor a dict, which is how the first version of this returned None and printed
    nothing.
    """
    d = getattr(meta, "episodes", None)
    if d is None:
        return None
    cols = set(getattr(d, "column_names", None) or getattr(d, "columns", []) or [])
    try:
        if "length" in cols:
            return np.asarray(d["length"], dtype=float)
        if {"dataset_from_index", "dataset_to_index"} <= cols:
            return (np.asarray(d["dataset_to_index"], dtype=float)
                    - np.asarray(d["dataset_from_index"], dtype=float))
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="collected dataset root, or a hub repo id")
    ap.add_argument("--compare", default=None,
                    help="expert dataset to mix with, e.g. lerobot/libero")
    a = ap.parse_args()

    got = describe(load_meta(a.dataset), a.dataset)

    lens = episode_lengths(load_meta(a.dataset))
    if lens is not None and len(lens):
        q = np.percentile(lens, [10, 25, 50, 75, 90])
        print(f"\n  episode length  min {lens.min():.0f}  "
              f"p10 {q[0]:.0f}  p25 {q[1]:.0f}  MEDIAN {q[2]:.0f}  "
              f"p75 {q[3]:.0f}  p90 {q[4]:.0f}  max {lens.max():.0f}")
        print("  ^ these are all SUCCESSES. A long tail here is the fumbling this\n"
              "    policy already does; training on it teaches more of it. Consider\n"
              "    re-collecting with --rft.max_steps at about the median.")

    if not a.compare:
        return

    want = describe(load_meta(a.compare), a.compare)

    print("\n=== mixing verdict ===")
    bad = []
    if got["fps"] != want["fps"]:
        bad.append(
            f"fps {got['fps']} vs {want['fps']}. build_datasets derives ONE set of\n"
            f"      delta_timestamps from a reference fps, so mixing these gives one of\n"
            f"      them the wrong stride for its action chunk and state history. This is\n"
            f"      the --rft.save_fps default (20) not matching LIBERO's 10.")
    if got["cams"] != want["cams"]:
        bad.append(f"camera keys {got['cams']} vs {want['cams']} -- build_datasets "
                   f"raises on a mismatch")
    if got["state_dim"] != want["state_dim"]:
        bad.append(f"state dim {got['state_dim']} vs {want['state_dim']}")
    if got["action_dim"] != want["action_dim"]:
        bad.append(f"action dim {got['action_dim']} vs {want['action_dim']}")

    for c in got["cams"]:
        if c in want["cam_shape"] and got["cam_shape"][c] != want["cam_shape"][c]:
            bad.append(f"{c} POLICY shape {got['cam_shape'][c]} vs "
                       f"{want['cam_shape'][c]} (declared {got['raw_shape'][c]} vs "
                       f"{want['raw_shape'][c]})")
        elif c in want["raw_shape"] and got["raw_shape"][c] != want["raw_shape"][c]:
            print(f"  [ok] {c} declared differently ({got['raw_shape'][c]} vs "
                  f"{want['raw_shape'][c]}) but both resolve to "
                  f"{got['cam_shape'][c]} -- not a conflict")

    unknown = [t for t in got["tasks"] if t not in set(want["tasks"])]
    if unknown:
        print(f"  [WARN] {len(unknown)} task string(s) not present in {a.compare}; "
              f"--paraphrase_augment keys off the exact string, so these would train\n"
              f"         UNAUGMENTED and the preflight will refuse to start:")
        for t in unknown[:5]:
            print(f"           {t!r}")

    if bad:
        print("  NOT directly mixable:")
        for b in bad:
            print(f"    - {b}")
    else:
        print("  compatible: same fps, cameras, shapes, state and action dims.")
        print(f"  python train_wiltechs_x.py --dataset_ids {a.compare} {a.dataset} ...")


if __name__ == "__main__":
    main()
