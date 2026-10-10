"""Plot accuracy / macro-F1 vs perturbation fraction for every target model."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.plotting.style import use_paper_style
from src.utils.paths import FIGURES, METRICS, ensure_dirs


def plot(df: pd.DataFrame, dataset: str, surrogate: str, out_path: Path):
    use_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharex=True)
    for tgt, sub in df[df["surrogate"] == surrogate].groupby("target"):
        sub = sub.sort_values("fraction")
        axes[0].plot(sub["fraction"], sub["accuracy"], "o-", label=tgt)
        axes[1].plot(sub["fraction"], sub["macro_f1"], "o-", label=tgt)
    for ax, title in zip(axes, ("accuracy", "macro F1")):
        ax.set_xlabel("perturbation fraction")
        ax.set_ylabel(title)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle(f"{dataset} — surrogate={surrogate}")
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--surrogate", default="cnn")
    args = p.parse_args()
    ensure_dirs()
    src = METRICS / args.dataset / f"robustness_surrogate_{args.surrogate}.csv"
    df = pd.read_csv(src)
    out_dir = FIGURES / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    plot(df, args.dataset, args.surrogate, out_dir / f"robustness_{args.surrogate}.png")


if __name__ == "__main__":
    main()
