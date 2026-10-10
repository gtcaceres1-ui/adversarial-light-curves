"""Per-source robustness-flag audit.

For each (model, survey) computes the fraction of test predictions that
change under the transferred attack (the "flip rate"), and how that fraction
depends on the clean-prediction margin (top1 - top2 probability). Flipped
sources concentrate at low margin, so margin + flip-under-attack identify the
predictions to treat with caution.

Evaluation-only: forward passes of the trained CNN and L5-raw checkpoints.

Outputs:
    outputs/metrics/flip_rate_audit.csv         (per model/survey summary)
    outputs/metrics/flip_rate_bins.csv          (per margin-decile bins)
    paper/flip_rate_vs_margin.pdf

Usage:
    python -m src.experiments.flip_rate_audit
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from src.models.robust_raw_encoder import _Sampling

REPO = Path(__file__).resolve().parents[2]
METRICS = REPO / "outputs/metrics"
FIG_OUT = REPO / "paper/flip_rate_vs_margin.pdf"
BATCH = 512

SURVEYS = {
    "ogle": {"adv": "runs/adversarial_dataset",
             "l5": "runs/robust_raw_l5_ogle", "cnn": "runs/cnn_ogle"},
    "gaia": {"adv": "runs/gaia_adversarial_dataset",
             "l5": "runs/gaia_robust_raw_l5", "cnn": "runs/gaia_cnn"},
    "wise": {"adv": "runs/wise_adversarial_dataset",
             "l5": "runs/wise_robust_raw_l5", "cnn": "runs/wise_cnn"},
}
N_BINS = 10


def batched(fn, n):
    parts = [fn(s, min(s + BATCH, n)) for s in range(0, n, BATCH)]
    return np.concatenate(parts, axis=0)


def survey_probs(name, cfg):
    sp = np.load(REPO / cfg["adv"] / "test_split.npz", allow_pickle=True)
    x1 = sp["X1_te"].astype(np.float32)
    x2 = sp["X2_te"].astype(np.float32)
    x2a = sp["X2_te_adv"].astype(np.float32)
    sd_m = float(sp["sd_m"])
    sigma = (sp["DE_te"].squeeze(-1) / max(abs(sd_m), 1e-8)).astype(np.float32)
    y = sp["y_te"].astype(np.int64)
    valid = sp["valid_te"].astype(bool)
    n = len(y)

    cnn = tf.keras.models.load_model(
        REPO / cfg["cnn"] / "cnn_two_tower.keras", compile=False,
        safe_mode=False,
    )
    p_cnn_cl = batched(lambda s, e: cnn(
        [x1[s:e, :, 0][..., None], x2[s:e, :, 0][..., None]],
        training=False).numpy(), n)
    p_cnn_ad = batched(lambda s, e: cnn(
        [x1[s:e, :, 0][..., None], x2a[s:e, :, 0][..., None]],
        training=False).numpy(), n)

    encoder = tf.keras.models.load_model(
        REPO / cfg["l5"] / "encoder_backbone.keras",
        custom_objects={"_Sampling": _Sampling}, compile=False,
        safe_mode=False,
    )
    classifier = tf.keras.models.load_model(
        REPO / cfg["l5"] / "classifier.keras", compile=False, safe_mode=False,
    )

    def l5_probs(x2_arr):
        def fn(s, e):
            tower = np.concatenate(
                [x2_arr[s:e], sigma[s:e][..., None]], axis=-1
            ).astype(np.float32)
            mu, _, _ = encoder([x1[s:e], tower], training=False)
            return classifier(mu, training=False).numpy()
        return batched(fn, n)

    p_l5_cl = l5_probs(x2)
    p_l5_ad = l5_probs(x2a)
    return {
        "y": y, "valid": valid,
        "CNN": (p_cnn_cl, p_cnn_ad),
        "RRE": (p_l5_cl, p_l5_ad),
    }


def margin(p):
    top = np.sort(p, axis=1)[:, ::-1]
    return top[:, 0] - top[:, 1]


def main():
    summary, bins_rows = [], []
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, (name, cfg) in zip(axes, SURVEYS.items()):
        data = survey_probs(name, cfg)
        y, valid = data["y"], data["valid"]
        for model, color in [("CNN", "#c44"), ("RRE", "#246")]:
            p_cl, p_ad = data[model]
            pred_cl, pred_ad = p_cl.argmax(1), p_ad.argmax(1)
            flip = (pred_cl != pred_ad) & valid
            m = margin(p_cl)
            flip_rate = float(flip.sum()) / float(valid.sum())
            correct_cl = (pred_cl == y) & valid
            summary.append({
                "survey": name, "model": model,
                "flip_rate": flip_rate,
                "n_flipped": int(flip.sum()),
                "n_valid": int(valid.sum()),
                "median_margin_flipped": float(np.median(m[flip]))
                if flip.any() else np.nan,
                "median_margin_stable": float(
                    np.median(m[valid & ~flip])),
                "clean_acc_flipped": float(
                    (pred_cl[flip] == y[flip]).mean())
                if flip.any() else np.nan,
            })
            # margin-decile bins over valid sources
            edges = np.quantile(m[valid], np.linspace(0, 1, N_BINS + 1))
            edges[0], edges[-1] = -1e-9, 1.0 + 1e-9
            centers, rates = [], []
            for b in range(N_BINS):
                sel = valid & (m > edges[b]) & (m <= edges[b + 1])
                if sel.sum() == 0:
                    continue
                rate = float((flip & sel).sum()) / float(sel.sum())
                centers.append(float(np.median(m[sel])))
                rates.append(rate)
                bins_rows.append({
                    "survey": name, "model": model, "bin": b,
                    "margin_median": centers[-1], "flip_rate": rate,
                    "n": int(sel.sum()),
                })
            ax.plot(centers, np.asarray(rates) * 100, "o-", color=color,
                    label=f"{model} (total {flip_rate*100:.1f}%)", ms=4)
        ax.set_yscale("symlog", linthresh=0.1)
        ticks = [0, 0.1, 1, 10, 50]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:g}" for t in ticks])
        ax.set_ylim(-0.02, 80)
        ax.set_xlabel("clean-prediction margin ($p_1 - p_2$)")
        ax.set_title(name.upper() if name != "gaia" else "Gaia DR2")
        ax.legend(fontsize=8)
        print(f"{name}: done", flush=True)
    axes[0].set_ylabel("flip rate under attack [%]")
    fig.tight_layout()
    fig.savefig(FIG_OUT, dpi=200)
    pd.DataFrame(summary).to_csv(METRICS / "flip_rate_audit.csv", index=False)
    pd.DataFrame(bins_rows).to_csv(METRICS / "flip_rate_bins.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"Wrote {FIG_OUT}", flush=True)


if __name__ == "__main__":
    main()
