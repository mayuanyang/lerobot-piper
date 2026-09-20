"""Compare normalization stats across datasets, and show what mixing them does.

The trainer uses `ref_meta.stats` for a single --dataset_id but
`aggregate_stats([...])` for several.  This prints the difference that swap
makes, per dimension, in the units that matter:

  action        std ratio  -> the factor every unnormalized action is scaled by
  observation.state         -> the shift injected at the encoder input, in
                               units of the *reference* std

Usage:
  python src/compare_norm_stats.py lerobot/libero /path/to/awr_corpus
  python src/compare_norm_stats.py <ref> <extra> [<extra> ...]

The first dataset is the reference (what the checkpoint was pretrained with).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

KEYS = ("action", "observation.state")


def load_stats(spec: str) -> dict:
    p = Path(spec)
    for cand in (p / "meta" / "stats.json", p / "stats.json", p):
        if cand.is_file():
            return json.loads(cand.read_text())
    # fall back to the hub cache, then the hub itself
    cache = Path.home() / ".cache" / "huggingface" / "lerobot" / spec / "meta" / "stats.json"
    if cache.is_file():
        return json.loads(cache.read_text())
    from huggingface_hub import hf_hub_download

    return json.loads(
        Path(hf_hub_download(spec, "meta/stats.json", repo_type="dataset")).read_text()
    )


def aggregate(stats_list: list[dict], key: str) -> dict:
    """Mirror of lerobot's aggregate_feature_stats for one feature."""
    means = np.stack([np.asarray(s[key]["mean"], float) for s in stats_list])
    variances = np.stack([np.asarray(s[key]["std"], float) ** 2 for s in stats_list])
    counts = np.asarray([np.asarray(s[key]["count"], float).reshape(-1)[0] for s in stats_list])
    total = counts.sum()
    c = counts[:, None]
    total_mean = (means * c).sum(0) / total
    total_var = ((variances + (means - total_mean) ** 2) * c).sum(0) / total
    return {"mean": total_mean, "std": np.sqrt(total_var), "count": total}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+", help="first one is the reference")
    ap.add_argument("--warn_pct", type=float, default=5.0)
    args = ap.parse_args()

    stats = [load_stats(d) for d in args.datasets]
    names = args.datasets

    worst = 0.0
    for key in KEYS:
        if any(key not in s for s in stats):
            print(f"\n{key}: missing from one of the datasets — skipped")
            continue
        counts = [np.asarray(s[key]["count"], float).reshape(-1)[0] for s in stats]
        total = sum(counts)
        print(f"\n{'=' * 78}\n{key}\n{'=' * 78}")
        for n, c in zip(names, counts):
            print(f"  {n:<44} {int(c):>9,} frames  ({c / total:5.1%} of the mix)")

        ref = {k: np.asarray(stats[0][key][k], float) for k in ("mean", "std")}
        agg = aggregate(stats, key)

        print(f"\n  {'dim':>3}  {'ref mean':>10} {'ref std':>10} │ "
              f"{'agg mean':>10} {'agg std':>10} │ {'std x':>7} {'mean shift':>11}")
        print(f"  {'-' * 74}")
        for i in range(len(ref["std"])):
            rs, rm = ref["std"][i], ref["mean"][i]
            as_, am = agg["std"][i], agg["mean"][i]
            ratio = as_ / rs if rs else float("nan")
            shift = (am - rm) / rs if rs else float("nan")   # in reference sigmas
            flag = "  <-- " if abs(ratio - 1) * 100 > args.warn_pct or abs(shift) > 0.05 else ""
            worst = max(worst, abs(ratio - 1) * 100)
            print(f"  {i:>3}  {rm:>10.4f} {rs:>10.4f} │ {am:>10.4f} {as_:>10.4f} │ "
                  f"{ratio:>7.3f} {shift:>+10.3f}σ{flag}")

        if key == "action":
            r = agg["std"] / np.where(ref["std"] == 0, np.nan, ref["std"])
            print(f"\n  every unnormalized action is scaled by  {np.nanmin(r):.3f}..{np.nanmax(r):.3f}")
            print(f"  (< 1.0 = the pretrained policy now UNDERSHOOTS by that factor)")

    print(f"\n{'=' * 78}")
    if worst > args.warn_pct:
        print(f"VERDICT: normalization moved by up to {worst:.1f}% — the mix is NOT a")
        print("         no-op for a checkpoint pretrained on the reference alone.")
    else:
        print(f"VERDICT: largest std change {worst:.1f}% — normalization is not the story;")
        print("         look at AWR itself (credit assignment) instead.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
