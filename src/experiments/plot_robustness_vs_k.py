"""Accuracy vs photometric budget k for the OGLE-III evaluations.

Tailored attacks (solid): white-box PGD for CNN/RNN, black-box Square Attack
for both RF variants (the 16-feature FATS RF with per-query feature
recomputation, and the simplified 7-feature Delta-m variant), adaptive PGD
for L5-raw. Transferred CNN-surrogate attack on L5-raw (dashed) covers the
full range as upper context.

Sources:
  outputs/metrics/ogle/rf_attack_sweep.csv      (cnn_pgd on CNN; clean rows)
  outputs/metrics/ogle/rnn_attack_sweep.csv     (rnn_pgd on RNN; L5 transfer)
  outputs/metrics/ogle/rf_fats_square_attack_k{0.50,1.00}.csv  (RF-FATS16)
  outputs/metrics/ogle/rf_square_attack{,_k1}.csv             (RF-Dm7)
  outputs/metrics/ogle/adaptive_pgd_l5_raw.csv  (L5-raw adaptive, k=0.5)

Output: paper/robustness_vs_k.pdf
        outputs/figures/robustness_vs_k.pdf
"""
from __future__ import annotations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
M = REPO / "outputs/metrics/ogle"
OUTS = [REPO / "outputs/figures/robustness_vs_k.pdf",
        REPO / "paper/robustness_vs_k.pdf"]


def load_series():
    rf_sweep = pd.read_csv(M / "rf_attack_sweep.csv")
    rnn_sweep = pd.read_csv(M / "rnn_attack_sweep.csv")

    def sweep_get(df, attack, model, k):
        if k == 0.0:
            r = df[(df.attack == "clean") & (df.model == model)]
        else:
            r = df[(df.attack == attack) & (df.model == model) & (df.k == k)]
        return float(r.acc.iloc[0]) if len(r) else None

    ks = [0.0, 0.5, 1.0]
    series = {}
    series["CNN (PGD, white-box)"] = [
        sweep_get(rf_sweep, "cnn_pgd", "CNN", k) for k in ks]
    series["RNN (PGD, white-box)"] = [
        sweep_get(rnn_sweep, "rnn_pgd", "RNN", k) for k in ks]

    fats = {}
    for kk in (0.5, 1.0):
        df = pd.read_csv(M / f"rf_fats_square_attack_k{kk:.2f}.csv")
        fats[0.0] = float(df[df.attack == "clean"].acc.iloc[0])
        fats[kk] = float(df[df.attack == "square_fats"].acc.iloc[0])
    series["RF-FATS16 (Square, per-query features)"] = [fats[k] for k in ks]

    dm7 = {}
    for name, kk in (("rf_square_attack.csv", 0.5),
                     ("rf_square_attack_k1.csv", 1.0)):
        df = pd.read_csv(M / name)
        dm7[0.0] = float(df[df.attack == "clean"].acc.iloc[0])
        dm7[kk] = float(df[df.attack == "rf_square"].acc.iloc[0])
    series["RF-$\\Delta m$7 (Square)"] = [dm7[k] for k in ks]

    ada = pd.read_csv(M / "adaptive_pgd_l5_raw.csv")
    row = ada[ada.scope == "all"].iloc[0]
    l5_k1 = None
    k1_path = M / "adaptive_pgd_l5_raw_k1.00.csv"
    if k1_path.exists():
        r1 = pd.read_csv(k1_path)
        r1 = r1[r1.scope == "all"]
        if len(r1):
            l5_k1 = float(r1.acc_adv.iloc[0])
    series["RRE (adaptive PGD)"] = [float(row.acc_clean),
                                       float(row.acc_adv), l5_k1]

    series["RRE (transferred CNN attack)"] = [
        sweep_get(rf_sweep, "cnn_pgd", "L5", k) for k in ks]
    return ks, series


STYLE = {
    "CNN (PGD, white-box)": dict(color="#d62728", marker="o"),
    "RNN (PGD, white-box)": dict(color="#ff7f0e", marker="s"),
    "RF-FATS16 (Square, per-query features)": dict(color="#2ca02c",
                                                   marker="D"),
    "RF-$\\Delta m$7 (Square)": dict(color="#98df8a", marker="d"),
    "RRE (adaptive PGD)": dict(color="#1f77b4", marker="^"),
    "RRE (transferred CNN attack)": dict(color="#1f77b4", marker="v",
                                            linestyle="--", alpha=0.6),
}


def main() -> None:
    ks, series = load_series()
    fig, ax = plt.subplots(figsize=(4.0, 3.2), dpi=200)
    for name, vals in series.items():
        kv = [(k, v) for k, v in zip(ks, vals) if v is not None]
        st = dict(STYLE[name])
        ls = st.pop("linestyle", "-")
        alpha = st.pop("alpha", 1.0)
        ax.plot([k for k, _ in kv], [v for _, v in kv], linestyle=ls,
                alpha=alpha, markersize=4.5, linewidth=1.4, label=name, **st)
    ax.axhline(0.25, color="gray", linewidth=0.8, linestyle=":",
               label="random guess")
    ax.axhline(4000 / 13568, color="gray", linewidth=0.8, linestyle="--",
               label="majority-class rate")
    ax.set_xlabel(r"perturbation budget $k$  ($|\delta_i| \leq k e_i$)")
    ax.set_ylabel("test accuracy (OGLE-III)")
    ax.set_xticks(ks)
    ax.set_ylim(0.2, 1.02)
    ax.legend(fontsize=6.1, loc="lower left", framealpha=0.9)
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    for out in OUTS:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        print(f"saved {out}")


if __name__ == "__main__":
    main()
