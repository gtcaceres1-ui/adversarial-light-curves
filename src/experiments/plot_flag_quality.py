"""Photometric-error distribution of flagged vs stable sources.

Shows that the sources the undefended CNN flips under the transferred attack
(the robustness flag) have systematically larger reported photometric errors
than the stable ones, i.e. the flag tracks genuine data quality.

Usage:
    python -m src.experiments.plot_flag_quality
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "paper/flag_quality.pdf"


def main():
    pe = np.load(REPO / "outputs/metrics/ogle/per_example_preds.npz",
                 allow_pickle=True)
    sp = np.load(REPO / "runs/adversarial_dataset/test_split.npz",
                 allow_pickle=True)
    valid = sp["valid_te"].astype(bool)
    n_obs = sp["n_sel_te"].astype(int)
    e_raw = sp["E_te"].squeeze(-1)
    point_valid = np.arange(e_raw.shape[1])[None, :] < n_obs[:, None]
    med_err = np.array([np.median(e_raw[i, point_valid[i]])
                        if point_valid[i].any() else np.nan
                        for i in range(len(n_obs))])

    clean, adv = pe["cnn_clean"], pe["cnn_adv"]
    flip = (clean != adv) & valid & np.isfinite(med_err)
    stable = (clean == adv) & valid & np.isfinite(med_err)

    plt.rcParams.update({
        "font.size": 9, "axes.linewidth": 0.7,
        "xtick.direction": "in", "ytick.direction": "in",
    })
    fig, ax = plt.subplots(figsize=(3.5, 2.7))  # A&A single column
    lo = max(1e-3, np.nanmin(med_err[valid]))
    hi = np.nanpercentile(med_err[valid], 99.5)
    bins = np.logspace(np.log10(lo), np.log10(hi), 40)

    med_st, med_fl = np.median(med_err[stable]), np.median(med_err[flip])
    ax.hist(med_err[stable], bins=bins, density=True, histtype="stepfilled",
            color="#4C72B0", alpha=0.55,
            label=f"stable ({stable.sum()})\nmedian {med_st:.3f}")
    ax.hist(med_err[flip], bins=bins, density=True, histtype="stepfilled",
            color="#C44E52", alpha=0.55,
            label=f"flagged ({flip.sum()})\nmedian {med_fl:.3f}")
    ax.axvline(med_st, color="#2f4a75", lw=1.1, ls="--")
    ax.axvline(med_fl, color="#8f2f33", lw=1.1, ls="--")

    ax.set_xscale("log")
    ax.set_xlabel(r"median photometric error $\tilde{e}$ (mag)")
    ax.set_ylabel("normalized density")
    # Single colour key (counts + medians) in the empty upper region between
    # the two median lines, clear of the tall stable-source bars on the left.
    ax.legend(frameon=False, fontsize=7.5, loc="upper left",
              bbox_to_anchor=(0.34, 0.80), handlelength=1.2,
              labelspacing=0.8,
              borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print(f"wrote {OUT}")
    print(f"stable median={np.median(med_err[stable]):.4f}  "
          f"flagged median={np.median(med_err[flip]):.4f}")


if __name__ == "__main__":
    main()
