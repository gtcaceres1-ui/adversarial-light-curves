"""Latent-space stability figure for the raw pipeline.

Visualizes the two-tower raw encoder at two ablation levels trained in
``train_robust_raw_ablation.py``:

  * top row    -- L1 (clean cross-entropy only, no adversarial component),
                  the "undefended encoder";
  * bottom row -- L5-raw (full objective), the proposed classifier.

For a representative class (LPV) it plots kernel-density estimates of the
first three latent means (mu_1, mu_2, mu_3) for the clean test curves and
their CNN-targeted adversarial counterparts. An encoder without adversarial
training (L1) maps the perturbed input to a shifted latent region, whereas
the full L5 objective keeps clean and adversarial inputs superimposed in
latent space.

A quantitative diagnostic (per-dimension KS statistic and a mean
standardized clean->adv latent shift, over *all* latent dimensions) is
printed so the qualitative reading of the figure can be checked.

Output:
  outputs/figures/latent_comparison_l1_vs_l5_raw.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.stats import gaussian_kde, ks_2samp

from src.experiments.build_adversarials_ogle import LABEL_TO_INT
from src.models.robust_raw_encoder import _Sampling

REPO = Path(__file__).resolve().parents[2]
ADV_ROOT = REPO / "runs/adversarial_dataset"
L1_DIR = REPO / "runs/robust_raw_l1_ce"          # undefended (CE only)
L5_DIR = REPO / "runs/robust_raw_l5_ogle"        # proposed full L5-raw (headline)
OUT_DIR = REPO / "outputs/figures"

TARGET_CLASS = "lpv"
CLASS_INDEX = LABEL_TO_INT[TARGET_CLASS]


def _make_xm(X2, sigma_std):
    return np.concatenate([X2, sigma_std[..., None]], axis=-1).astype(np.float32)


def _encode(enc, xt, xm, bs=512):
    """Return the deterministic latent mean mu for every curve."""
    n = len(xt)
    mu0, _, _ = enc([xt[:1], xm[:1]], training=False)
    d = int(mu0.shape[-1])
    Z = np.zeros((n, d), np.float32)
    for s in range(0, n, bs):
        e = min(s + bs, n)
        mu, _, _ = enc([xt[s:e], xm[s:e]], training=False)
        Z[s:e] = mu.numpy() if hasattr(mu, "numpy") else mu
    return Z


def _kde_xy(values, x_min, x_max, n_pts=400):
    kde = gaussian_kde(values)
    xs = np.linspace(x_min, x_max, n_pts)
    return xs, kde(xs)


def _shift_diagnostic(name, Z_cl, Z_ad):
    """Mean standardized clean->adv shift and per-dim KS over all dims."""
    sd = Z_cl.std(axis=0) + 1e-8
    std_shift = np.abs(Z_cl.mean(axis=0) - Z_ad.mean(axis=0)) / sd
    ks = np.array([ks_2samp(Z_cl[:, d], Z_ad[:, d]).statistic
                   for d in range(Z_cl.shape[1])])
    print(f"  [{name}] mean |Δμ|/σ over all dims = {std_shift.mean():.4f} "
          f"(max {std_shift.max():.4f}); mean KS = {ks.mean():.4f} "
          f"(max {ks.max():.4f})")
    print(f"        first 3 dims |Δμ|/σ = {np.round(std_shift[:3], 4)}, "
          f"KS = {np.round(ks[:3], 4)}")
    return std_shift, ks


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw test tensors from test_split.npz")
    sp = np.load(ADV_ROOT / "test_split.npz", allow_pickle=True)
    X1_te = sp["X1_te"].astype(np.float32)
    X2_te = sp["X2_te"].astype(np.float32)
    X2_te_adv = sp["X2_te_adv"].astype(np.float32)
    DE_te = sp["de_te"].astype(np.float32)
    y_te = sp["y_te"].astype(np.int64)
    valid = sp["valid"].astype(bool)
    sd_m = float(sp["sd_m"])

    sigma_std = (DE_te.squeeze(-1) / sd_m).astype(np.float32)
    xt = X1_te
    xm_cl = _make_xm(X2_te, sigma_std)
    xm_ad = _make_xm(X2_te_adv, sigma_std)

    mask = valid & (y_te == CLASS_INDEX)
    print(f"  valid {TARGET_CLASS} curves: {int(mask.sum())}")
    xt_m = xt[mask]; xm_cl_m = xm_cl[mask]; xm_ad_m = xm_ad[mask]

    print(f"Loading L1 encoder  ({L1_DIR.name})")
    l1_enc = tf.keras.models.load_model(
        L1_DIR / "encoder_backbone.keras",
        custom_objects={"_Sampling": _Sampling}, compile=False, safe_mode=False)
    print(f"Loading L5 encoder  ({L5_DIR.name})")
    l5_enc = tf.keras.models.load_model(
        L5_DIR / "encoder_backbone.keras",
        custom_objects={"_Sampling": _Sampling}, compile=False, safe_mode=False)

    print("Encoding 4 cases (latent means)")
    Z_l1_cl = _encode(l1_enc, xt_m, xm_cl_m)
    Z_l1_ad = _encode(l1_enc, xt_m, xm_ad_m)
    Z_l5_cl = _encode(l5_enc, xt_m, xm_cl_m)
    Z_l5_ad = _encode(l5_enc, xt_m, xm_ad_m)

    print("Latent clean->adv shift diagnostic (all 48 dims):")
    _shift_diagnostic("L1 (undefended)", Z_l1_cl, Z_l1_ad)
    _shift_diagnostic("RRE (ours)", Z_l5_cl, Z_l5_ad)

    # 3x2 layout (rows = mu_1..mu_3, columns = V1 / RRE): vertical format
    # sized for a single A&A column.
    print("Plotting KDE 3x2 (single-column layout)")
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.7})
    fig, axes = plt.subplots(3, 2, figsize=(3.4, 5.4), sharex="row")
    titles = ["$\\mu_1$", "$\\mu_2$", "$\\mu_3$"]
    blue = "#1f77b4"    # clean
    orange = "#ff7f0e"  # adversarial
    cols = [("V1 (undefended)", Z_l1_cl, Z_l1_ad),
            ("RRE (ours)", Z_l5_cl, Z_l5_ad)]

    for c, (col_name, Z_cl, Z_ad) in enumerate(cols):
        for d in range(3):
            ax = axes[d, c]
            cl_vals = Z_cl[:, d]; ad_vals = Z_ad[:, d]
            all_vals = np.concatenate([cl_vals, ad_vals])
            xmin = np.percentile(all_vals, 1) - 0.5
            xmax = np.percentile(all_vals, 99) + 0.5
            xs, k_cl = _kde_xy(cl_vals, xmin, xmax)
            _, k_ad = _kde_xy(ad_vals, xmin, xmax)
            ax.fill_between(xs, k_cl, color=blue, alpha=0.45, label="clean")
            ax.fill_between(xs, k_ad, color=orange, alpha=0.45,
                            label="adversarial")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=7)
            ax.set_yticks([])
            if d == 0:
                ax.set_title(col_name, fontsize=8.5)
            ax.set_xlabel(titles[d], fontsize=8)
            if c == 0:
                ax.set_ylabel("density", fontsize=8)
            # Color key in the empty right half of panel (a), where it
            # does not overlap any density curve.
            if d == 0 and c == 0:
                ax.legend(loc="center right", frameon=False, fontsize=6.5,
                          handlelength=1.2, borderaxespad=0.2)

    labels = ["(a)", "(d)", "(b)", "(e)", "(c)", "(f)"]
    for ax, lab in zip(axes.flat, labels):
        ax.text(0.03, 0.95, lab, transform=ax.transAxes,
                fontsize=8, va="top", ha="left", fontweight="bold")

    plt.tight_layout(pad=0.4, h_pad=0.8, w_pad=0.6)
    out_pdf = OUT_DIR / "latent_comparison_l1_vs_l5_raw.pdf"
    out_png = OUT_DIR / "latent_comparison_l1_vs_l5_raw.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    print(f"Wrote {out_pdf}\nWrote {out_png}")


if __name__ == "__main__":
    main()
