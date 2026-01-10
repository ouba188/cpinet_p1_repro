#!/usr/bin/env python3
# cv/summarize_repeated.py
"""
Summarize repeated split runs:
- For each split, take median across repeats
- Across splits, compute mean ± std
"""

import argparse
from pathlib import Path
import json
import statistics
import csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, required=True, help="runs/repeated_splits/<model>")
    ap.add_argument("--metric", type=str, default="test.acc", help="e.g., test.acc or test.macro_f1")
    ap.add_argument("--out_csv", type=str, default=None)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    split_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("split_")])
    if not split_dirs:
        raise ValueError(f"No split dirs found in {runs_dir}")

    def get_metric(d: dict, key: str):
        cur = d
        for part in key.split("."):
            cur = cur[part]
        return float(cur)

    split_medians = []
    details = []

    for sd in split_dirs:
        reps = sorted([p for p in sd.iterdir() if p.is_dir() and p.name.startswith("rep_")])
        vals = []
        for rp in reps:
            mp = rp / "metrics.json"
            if not mp.exists():
                continue
            with mp.open("r", encoding="utf-8") as f:
                m = json.load(f)
            vals.append(get_metric(m, args.metric))
        if not vals:
            continue
        med = statistics.median(vals)
        split_medians.append(med)
        details.append((sd.name, med, len(vals)))

    if not split_medians:
        raise ValueError("No metrics found. Did you run with --save_json?")

    mean = statistics.mean(split_medians)
    std = statistics.pstdev(split_medians) if len(split_medians) > 1 else 0.0

    print(f"Metric={args.metric}")
    print(f"Per-split median (n={len(split_medians)} splits):")
    for name, med, nrep in details:
        print(f"  {name}: median={med:.6f} (reps={nrep})")
    print(f"\nFinal: mean={mean:.6f}, std={std:.6f}")

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["split", "median", "repeats"])
            for name, med, nrep in details:
                w.writerow([name, f"{med:.6f}", nrep])
            w.writerow([])
            w.writerow(["mean", f"{mean:.6f}"])
            w.writerow(["std", f"{std:.6f}"])
        print(f"Wrote: {out_csv}")

if __name__ == "__main__":
    main()
