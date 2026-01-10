#!/usr/bin/env python3
# cv/make_repeated_splits.py
"""
Create repeated random (stratified) splits for dual-pol CSV samples.

This matches CPINet paper protocol more closely than fixed train/val/test:
- Randomly split train/test for K times (default K=5)
- For each split, further split train into train/val (for model selection)
Outputs:
  out_dir/split_00/train.csv, val.csv, test.csv
  ...
"""

import argparse
from pathlib import Path
import csv
import random
from collections import defaultdict

def read_csv(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if "label" in row:
                try:
                    row["label"] = int(row["label"])
                except ValueError:
                    pass # handled as string or kept as is
            elif "label_name" in row:
                # Use label_name as label for stratification
                row["label"] = row["label_name"]
            
            rows.append(row)
    return rows

def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Empty rows for {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def stratified_split(indices_by_class, test_ratio, rng: random.Random):
    train_idx, test_idx = [], []
    for c, idxs in indices_by_class.items():
        idxs = idxs[:]  # copy
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = int(round(n * test_ratio))
        # keep at least 1 in train if possible
        if n >= 2:
            n_test = max(1, min(n - 1, n_test))
        else:
            n_test = 0
        test_idx.extend(idxs[:n_test])
        train_idx.extend(idxs[n_test:])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", type=str, nargs="+", required=True,
                    help="One or more csv files to merge (e.g., original train/val/test.csv).")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--k", type=int, default=5, help="Number of random splits (paper uses 5).")
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--val_ratio", type=float, default=0.1, help="Val ratio within train set.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = []
    seen = set()
    for p in args.csvs:
        for row in read_csv(Path(p)):
            key = (row["vv_path"], row["vh_path"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)

    if not rows:
        raise ValueError("No samples loaded.")

    # build indices by class
    by_class = defaultdict(list)
    for i, r in enumerate(rows):
        by_class[r["label"]].append(i)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for split_id in range(args.k):
        rng = random.Random(args.seed + split_id)
        train_idx, test_idx = stratified_split(by_class, args.test_ratio, rng)

        # build train rows
        train_rows = [rows[i] for i in train_idx]
        # stratified val within train
        by_class_tr = defaultdict(list)
        for j, r in enumerate(train_rows):
            by_class_tr[r["label"]].append(j)
        tr2_idx, va2_idx = stratified_split(by_class_tr, args.val_ratio, rng)
        tr_rows = [train_rows[j] for j in tr2_idx]
        va_rows = [train_rows[j] for j in va2_idx]
        te_rows = [rows[i] for i in test_idx]

        split_dir = out_root / f"split_{split_id:02d}"
        write_csv(split_dir / "train.csv", tr_rows)
        write_csv(split_dir / "val.csv", va_rows)
        write_csv(split_dir / "test.csv", te_rows)

    print(f"Done. Wrote {args.k} splits to: {out_root}")

if __name__ == "__main__":
    main()
