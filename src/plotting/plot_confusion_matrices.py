"""Render confusion matrices saved alongside each trained model."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.plotting.style import use_paper_style
from src.utils.paths import CHECKPOINTS, FIGURES, ensure_dirs


def _classes(config: Path):
    with open(config) as f:
        cfg = yaml.safe_load(f)
    return list(cfg["dataset"]["classes"])


def _plot(cm: np.ndarray, classes, title, out_path):
    use_paper_style()
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    cm_n = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(cm_n, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm_n[i,j]:.2f}", ha="center", va="center",
                    color="white" if cm_n[i, j] > 0.5 else "black", fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_path); fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", required=True)
    args = p.parse_args()
    ensure_dirs()
    classes = _classes(Path(args.config))
    base = CHECKPOINTS / args.dataset
    out = FIGURES / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    for model_dir in sorted([d for d in base.iterdir() if d.is_dir()]):
        cm_path = model_dir / "confusion.npy"
        if cm_path.exists():
            cm = np.load(cm_path)
            _plot(cm, classes, f"{args.dataset} — {model_dir.name}",
                  out / f"confusion_{model_dir.name}.png")


if __name__ == "__main__":
    main()
