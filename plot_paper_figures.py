# plot_paper_figures.py
import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import csv


def read_history(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    cols = {k: [] for k in reader.fieldnames or []}
    for row in rows:
        for k in cols:
            cols[k].append(float(row[k]))
    return cols


def plot_sensitivity(run_dirs: List[str], labels: List[str], metric: str, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for run_dir, label in zip(run_dirs, labels):
        df = read_history(Path(run_dir) / "history.csv")
        ax.plot(df["epoch"], df[metric], label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_gradnorm(run_dir: str, out_path: Path) -> None:
    df = read_history(Path(run_dir) / "history.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    weight_cols = [c for c in df.keys() if c.startswith("w_")]
    for c in weight_cols:
        axes[0].plot(df["epoch"], df[c], label=c.replace("w_", "w_"))
    axes[0].set_title("Weight Magnitude")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Weight")
    axes[0].legend()

    loss_cols = [c for c in df.keys() if c.startswith("loss_") and c != "loss_total"]
    for c in loss_cols:
        axes[1].plot(df["epoch"], df[c], label=c.replace("loss_", "L_"))
    axes[1].set_title("Loss Value")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    ratio_cols = [c for c in df.keys() if c.startswith("ratio_")]
    for c in ratio_cols:
        axes[2].plot(df["epoch"], df[c], label=c.replace("ratio_", "r_"))
    axes[2].set_title("Loss Ratio")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Ratio")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k_runs", nargs="*", default=[], help="Run dirs for different k values.")
    ap.add_argument("--k_labels", nargs="*", default=[], help="Labels for k runs.")
    ap.add_argument("--lam_runs", nargs="*", default=[], help="Run dirs for different lambda values.")
    ap.add_argument("--lam_labels", nargs="*", default=[], help="Labels for lambda runs.")
    ap.add_argument("--metric", type=str, default="val_acc")
    ap.add_argument("--gradnorm_run", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="runs/plots")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.k_runs:
        labels = args.k_labels or [Path(r).name for r in args.k_runs]
        plot_sensitivity(args.k_runs, labels, args.metric, out_dir / "fig6_k.png", "Sensitivity: k")
    if args.lam_runs:
        labels = args.lam_labels or [Path(r).name for r in args.lam_runs]
        plot_sensitivity(args.lam_runs, labels, args.metric, out_dir / "fig6_lambda.png", "Sensitivity: lambda")
    if args.gradnorm_run:
        plot_gradnorm(args.gradnorm_run, out_dir / "fig7_gradnorm.png")


if __name__ == "__main__":
    main()
