"""Histogram + box plot of |delta_i| / e_i per perturbation fraction."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plotting.style import use_paper_style
from src.utils.paths import ADVERSARIAL, FIGURES, METRICS, ensure_dirs


def _abs_delta(npz_path: Path) -> np.ndarray:
    d = np.load(npz_path)
    mask = d["perturb_mask"].astype(bool) & (d["e_raw"] > 0)
    delta = (d["m_adv"] - d["m_clean"])[mask] / (d["e_raw"][mask] + 1e-12)
    return np.abs(delta)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--surrogate", default="cnn")
    args = p.parse_args()
    ensure_dirs()
    use_paper_style()

    adv_root = ADVERSARIAL / args.dataset / args.surrogate
    fractions, samples = [], []
    for npz in sorted(adv_root.glob("adv_frac_*.npz")):
        frac = float(npz.stem.split("_")[-1])
        if frac == 0.0:
            continue
        fractions.append(frac)
        samples.append(_abs_delta(npz))

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.boxplot(samples, labels=[f"{f:.1f}" for f in fractions], showfliers=False)
    # Bound k from the validity table.
    k = pd.read_csv(METRICS / args.dataset / "perturbation_validity.csv")["bound_k"].iloc[0]
    ax.axhline(k, color="red", lw=1.0, ls="--", label=f"k = {k}")
    ax.set_xlabel("perturbation fraction")
    ax.set_ylabel(r"$|\Delta m_i| / e_i$")
    ax.legend(frameon=False)
    ax.set_title(f"{args.dataset} — perturbation magnitudes")

    out = FIGURES / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"perturbation_validation_{args.surrogate}.png")
    fig.savefig(out / f"perturbation_validation_{args.surrogate}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
