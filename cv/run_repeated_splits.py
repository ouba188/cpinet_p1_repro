#!/usr/bin/env python3
# cv/run_repeated_splits.py
"""
Run repeated random splits with repeated trainings per split (paper: 5 splits, 3 trainings each).
Calls train_cpinet.py and collects metrics.json.
"""

import argparse
from pathlib import Path
import subprocess
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="runs/repeated_splits_cpinet")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--base_seed", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=64)
    ap.add_argument("--num_classes", type=int, default=3)
    ap.add_argument("--device", type=str, default="cuda:0")

    # model-specific knobs (keep aligned with train_cpinet.py)
    ap.add_argument("--gbp_groups", type=int, default=3)
    ap.add_argument("--fbc_k", type=int, default=2048)
    ap.add_argument("--fbc_lam", type=float, default=1e-3)
    ap.add_argument("--assemble", type=str, default="logit_mean", choices=["mean", "logit_mean"])
    ap.add_argument("--loss_balance", type=str, default="gradnorm", choices=["sum", "gradnorm"])
    ap.add_argument("--gradnorm_alpha", type=float, default=1.5)

    ap.add_argument("--python", type=str, default=sys.executable)
    args = ap.parse_args()

    splits_dir = Path(args.splits_dir)
    split_dirs = sorted([p for p in splits_dir.iterdir() if p.is_dir() and p.name.startswith("split_")])
    if not split_dirs:
        raise ValueError(f"No split_* dirs found in {splits_dir}")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for s_i, sd in enumerate(split_dirs):
        for r_i in range(args.repeats):
            seed = args.base_seed + s_i * 100 + r_i
            run_dir = out_root / sd.name / f"rep_{r_i:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                args.python, str(Path(__file__).resolve().parents[1] / "train_cpinet.py"),
                "--train_csv", str(sd / "train.csv"),
                "--val_csv", str(sd / "val.csv"),
                "--test_csv", str(sd / "test.csv"),
                "--run_dir", str(run_dir),
                "--save_json",
                "--seed", str(seed),
                "--epochs", str(args.epochs),
                "--batch", str(args.batch),
                "--workers", str(args.workers),
                "--imgsz", str(args.imgsz),
                "--num_classes", str(args.num_classes),
                "--device", args.device,
                "--gbp_groups", str(args.gbp_groups),
                "--fbc_k", str(args.fbc_k),
                "--fbc_lam", str(args.fbc_lam),
                "--assemble", args.assemble,
                "--loss_balance", args.loss_balance,
                "--gradnorm_alpha", str(args.gradnorm_alpha),
            ]
            print("\n>>>", " ".join(cmd))
            subprocess.check_call(cmd)

    print(f"Done. Results in: {out_root}")

if __name__ == "__main__":
    main()
