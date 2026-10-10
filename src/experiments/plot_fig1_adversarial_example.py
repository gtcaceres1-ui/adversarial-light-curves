"""Regenerate Fig. 1 of the paper from the adversarial test set.

Picks an RR Lyrae whose CNN prediction flips rrlyr -> cep under the
transferred attack (per runs/adversarial_dataset/test_attack_summary.csv),
and plots the clean and adversarial versions phase-folded with the reported
+-e_i error bars.

Both panels show the same first-500 epochs that enter the classifier, so
the comparison is exact. The script asserts the pointwise photometric
bound |m_adv - m| <= k e_i before plotting.

Usage:
    python -m src.experiments.plot_fig1_adversarial_example
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
LCS_DIR = REPO / "Data/OGLE-III/LCs"
ADV_DIR = REPO / "runs/adversarial_dataset/adversarial_lcs"
SUMMARY = REPO / "runs/adversarial_dataset/test_attack_summary.csv"
PERIODS = REPO / "runs/periods_ogle.npz"
OUT_DIR = REPO / "paper"

TARGET_ID = "OGLE-LMC-RRLYR-13159"
K = 0.5
SEQ_LEN = 500


def load_dat(path):
    lc = pd.read_csv(
        path, sep=r"\s+", comment="#", header=None,
        names=["time", "magnitude", "error"],
    )
    lc = lc.apply(pd.to_numeric, errors="coerce")  # drop header row if any
    return (
        lc.dropna().sort_values("time")
        .drop_duplicates(subset="time", keep="first")
        .reset_index(drop=True)
    )


def main():
    summary = pd.read_csv(SUMMARY).set_index("id")
    row = summary.loc[TARGET_ID]
    assert row["true_label"] == "rrlyr" and row["pred_clean"] == "rrlyr" \
        and row["pred_adv"] == "cep" and bool(row["valid"]), (
        f"{TARGET_ID} is not a canonical rrlyr->cep flip: {dict(row)}"
    )

    per = np.load(PERIODS, allow_pickle=True)
    period = float(dict(zip(per["IDs"], per["period"]))[TARGET_ID])

    clean = load_dat(LCS_DIR / f"{TARGET_ID}.dat").iloc[:SEQ_LEN]
    adv = load_dat(ADV_DIR / f"{TARGET_ID}_adv.dat")
    n = min(len(clean), len(adv))
    clean, adv = clean.iloc[:n], adv.iloc[:n]
    assert np.allclose(clean["time"].values, adv["time"].values), \
        "clean/adv epochs differ"
    delta = np.abs(adv["magnitude"].values - clean["magnitude"].values)
    bound = K * clean["error"].values
    assert (delta <= bound + 1e-9).all(), "photometric bound violated"
    print(f"{TARGET_ID}: n={n}, P={period:.4f} d, "
          f"max |delta|/(k e) = {(delta / bound).max():.3f}, "
          f"median e = {clean['error'].median():.3f} mag")

    panels = [
        (clean, "fig1_clean_errorbars.pdf"),
        (adv, "fig1_adversarial_errorbars.pdf"),
    ]
    ymin = min(df["magnitude"].min() - df["error"].max() for df, _ in panels)
    ymax = max(df["magnitude"].max() + df["error"].max() for df, _ in panels)
    for df, fname in panels:
        phase = (df["time"].values / period) % 1.0
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        for shift in (0.0, 1.0):
            ax.errorbar(
                phase + shift, df["magnitude"], yerr=df["error"],
                fmt="o", ms=2.4, color="black", ecolor="0.55",
                elinewidth=0.6, capsize=0, alpha=0.85, zorder=2,
            )
        ax.set_xlim(-0.02, 2.02)
        ax.set_ylim(ymax + 0.02, ymin - 0.02)  # brighter up
        ax.set_xlabel("Phase")
        ax.set_ylabel("magnitude")
        ax.set_title(f"{TARGET_ID} — P = {period:.4f} d")
        fig.tight_layout()
        out = OUT_DIR / fname
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
