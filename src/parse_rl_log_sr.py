"""
Parse RL training log (JSONL) and compute average success rate across all tasks.

Usage:
    python src/parse_rl_log_sr.py <log_path> [--last_n N] [--window W]

Examples:
    # Parse full log
    python src/parse_rl_log_sr.py outputs/rl/wilro_10_4/rl_log.jsonl

    # Show only last 5 iterations
    python src/parse_rl_log_sr.py outputs/rl/wilro_10_4/rl_log.jsonl --last_n 5

    # Compute rolling average over last 3 iterations
    python src/parse_rl_log_sr.py outputs/rl/wilro_10_4/rl_log.jsonl --window 3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def parse_log(log_path: str):
    """Parse JSONL log file and return list of iteration records."""
    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: skipping malformed line: {line[:80]}...")
    return records


def compute_avg_sr(records, last_n=None, window=None):
    """Compute per-task and overall average success rates."""
    if last_n:
        records = records[-last_n:]

    if window and len(records) >= window:
        records = records[-window:]

    if not records:
        print("No records to analyze.")
        return

    # Collect all task IDs
    all_task_ids = set()
    for rec in records:
        sr = rec.get("sr", {})
        all_task_ids.update(int(k) for k in sr.keys() if sr[k] is not None)
    all_task_ids = sorted(all_task_ids)

    # Per-task SR across iterations
    task_srs = {tid: [] for tid in all_task_ids}
    for rec in records:
        sr = rec.get("sr", {})
        for tid in all_task_ids:
            val = sr.get(str(tid))
            if val is not None:
                task_srs[tid].append(float(val))

    # Print header
    n_iters = len(records)
    print(f"\n{'='*70}")
    print(f"RL Training Log Analysis — {n_iters} iteration(s)")
    if last_n:
        print(f"  (showing last {last_n} iterations)")
    if window:
        print(f"  (rolling window: last {window} iterations)")
    print(f"{'='*70}")

    # Per-iteration summary
    print(f"\n{'Iter':>5}  |  {'Avg SR':>8}  |  Per-Task SR (%)")
    print(f"{'-'*5}-+-{'-'*8}-+-{'-'*50}")
    for rec in records:
        it = rec.get("iter", "?")
        sr = rec.get("sr", {})
        vals = [float(v) for v in sr.values() if v is not None]
        avg = np.mean(vals) * 100 if vals else float("nan")
        per_task = "  ".join(
            f"T{k}={float(v)*100:5.1f}" if v is not None else f"T{k}=  –  "
            for k, v in sorted(sr.items(), key=lambda x: int(x[0]))
        )
        print(f"{it:>5}  |  {avg:>7.1f}%  |  {per_task}")

    # Overall average
    print(f"\n{'='*70}")
    print("Per-Task Average SR (across selected iterations):")
    print(f"{'-'*70}")

    overall_all = []
    for tid in all_task_ids:
        vals = task_srs[tid]
        if vals:
            avg = np.mean(vals) * 100
            std = np.std(vals) * 100
            overall_all.extend(vals)
            print(f"  Task {tid:>2}: {avg:6.1f}%  (±{std:.1f}%,  n={len(vals)} iters)")
        else:
            print(f"  Task {tid:>2}:    –    (no data)")

    if overall_all:
        overall_avg = np.mean(overall_all) * 100
        overall_std = np.std(overall_all) * 100
        print(f"\n{'='*70}")
        print(f"  Overall Average SR: {overall_avg:.1f}%  (±{overall_std:.1f}%)")
        print(f"  Tasks with data:    {len([v for v in task_srs.values() if v])}/{len(all_task_ids)}")
        print(f"{'='*70}\n")
    else:
        print("\nNo success rate data found.\n")


def main():
    parser = argparse.ArgumentParser(description="Parse RL log and compute average SR")
    parser.add_argument("log_path", type=str, help="Path to rl_log.jsonl")
    parser.add_argument("--last_n", type=int, default=None,
                        help="Only analyze last N iterations")
    parser.add_argument("--window", type=int, default=None,
                        help="Compute rolling average over last N iterations")
    args = parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        print(f"Error: log file not found: {log_path}")
        sys.exit(1)

    records = parse_log(str(log_path))
    if not records:
        print(f"Error: no valid records in {log_path}")
        sys.exit(1)

    print(f"Loaded {len(records)} iterations from {log_path}")
    compute_avg_sr(records, last_n=args.last_n, window=args.window)


if __name__ == "__main__":
    main()