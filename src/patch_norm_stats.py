"""Rewrite a checkpoint's normalizer with the stats a different dataset mix produces.

This exists to test one thing without training: when a run mixes several
--dataset_id, the trainer swaps `ref_meta.stats` for `aggregate_stats([...])`,
so every state and action is normalized differently than it was during
pretraining. Patching a known-good checkpoint to the mixed stats and evaluating
it isolates that swap from everything else the mix changes.

  python src/patch_norm_stats.py \
      --checkpoint outputs/obs2-22k \
      --stats_from ISdept/libero-awr-goal lerobot/libero \
      --out outputs/obs2-22k-aggstats

The checkpoint's weights are copied untouched; only `action` and
`observation.state` entries in the normalizer are replaced. Image stats are
left alone -- they are not what the mix moved, and the vision path normalizes
against fixed constants.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

PATCH_KEYS = ("action", "observation.state")


def load_stats(spec: str) -> dict:
    p = Path(spec)
    for cand in (p / "meta" / "stats.json", p / "stats.json", p):
        if cand.is_file():
            return json.loads(cand.read_text())
    cache = Path.home() / ".cache" / "huggingface" / "lerobot" / spec / "meta" / "stats.json"
    if cache.is_file():
        return json.loads(cache.read_text())
    from huggingface_hub import hf_hub_download

    return json.loads(
        Path(hf_hub_download(spec, "meta/stats.json", repo_type="dataset")).read_text()
    )


def aggregate(stats_list: list[dict], key: str) -> dict:
    """lerobot's aggregate_feature_stats, for one feature."""
    means = np.stack([np.asarray(s[key]["mean"], float) for s in stats_list])
    variances = np.stack([np.asarray(s[key]["std"], float) ** 2 for s in stats_list])
    counts = np.asarray([np.asarray(s[key]["count"], float).reshape(-1)[0] for s in stats_list])
    total = counts.sum()
    c = counts[:, None]
    total_mean = (means * c).sum(0) / total
    total_var = ((variances + (means - total_mean) ** 2) * c).sum(0) / total
    out = {
        "mean": total_mean,
        "std": np.sqrt(total_var),
        "count": np.array([total]),
        "min": np.min(np.stack([np.asarray(s[key]["min"], float) for s in stats_list]), 0),
        "max": np.max(np.stack([np.asarray(s[key]["max"], float) for s in stats_list]), 0),
    }
    # Quantiles aggregate as count-weighted averages of the per-dataset
    # quantiles -- not as true quantiles. That is lerobot's behaviour and the
    # point here is to reproduce what training actually used, not to improve it.
    for q in [k for k in stats_list[0][key] if k.startswith("q") and k[1:].isdigit()]:
        if all(q in s[key] for s in stats_list):
            vals = np.stack([np.asarray(s[key][q], float) for s in stats_list])
            out[q] = (vals * c).sum(0) / total
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="local checkpoint directory")
    ap.add_argument("--stats_from", nargs="+", required=True,
                    help="the --dataset_id list whose aggregate to install")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src, out = Path(args.checkpoint), Path(args.out)
    hits = sorted(src.glob("*normalizer_processor.safetensors"))
    if not hits:
        print(f"ERROR: no *normalizer_processor.safetensors under {src}", file=sys.stderr)
        return 1
    if out.exists():
        print(f"ERROR: {out} exists; refusing to overwrite", file=sys.stderr)
        return 1

    stats = [load_stats(d) for d in args.stats_from]
    shutil.copytree(src, out)
    print(f"copied {src} -> {out}")

    for f in hits:
        tgt = out / f.name
        d = load_file(str(tgt))
        n = 0
        for key in PATCH_KEYS:
            if not all(key in s for s in stats):
                print(f"  {key}: absent from one of --stats_from, left as-is")
                continue
            agg = aggregate(stats, key)
            for sub, val in agg.items():
                name = f"{key}.{sub}"
                if name not in d:
                    continue
                old = d[name]
                new = np.asarray(val, dtype=old.dtype).reshape(old.shape)
                if sub == "std":
                    with np.errstate(divide="ignore", invalid="ignore"):
                        rel = np.abs(new - old) / np.where(old == 0, np.nan, np.abs(old))
                    print(f"  {name:28s} max |delta| {np.nanmax(rel) * 100:6.2f}%")
                elif sub == "mean":
                    # In sigma, not percent: several of these means sit near
                    # zero, where a relative change is large and meaningless.
                    sd = np.asarray(d.get(f"{key}.std", np.ones_like(old)), float)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        sig = np.abs(new - old) / np.where(sd == 0, np.nan, sd)
                    print(f"  {name:28s} max shift  {np.nanmax(sig):6.3f} sigma")
                d[name] = new
                n += 1
        save_file(d, str(tgt))
        print(f"  wrote {n} tensors into {tgt.name}")

    print("\nEvaluate this directory against the unpatched one. Same weights, "
          "same everything, only the normalization differs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
