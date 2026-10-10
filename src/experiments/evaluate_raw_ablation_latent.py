"""Quantify clean-to-adversarial latent shifts for OGLE L1--L5."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import ks_2samp

from src.models.robust_raw_encoder import _Sampling

REPO = Path(__file__).resolve().parents[2]
ADV_ROOT = REPO / "runs/adversarial_dataset"
OUT = REPO / "outputs/metrics/ogle/raw_ablation_latent_stability.csv"
MODELS = {
    "L1": "runs/robust_raw_l1_ce",
    "L2": "runs/robust_raw_l2_advce",
    "L3": "runs/robust_raw_l3_trades",
    "L4": "runs/robust_raw_l4_supcon",
    "L5": "runs/robust_raw_l5_ogle",
}


def encode(encoder, x_t, x_m, batch=512):
    output = []
    for start in range(0, len(x_t), batch):
        end = min(start + batch, len(x_t))
        mu, _, _ = encoder(
            [x_t[start:end], x_m[start:end]], training=False
        )
        output.append(mu.numpy())
    return np.concatenate(output)


def main():
    sp = np.load(ADV_ROOT / "test_split.npz", allow_pickle=True)
    valid = sp["valid_te"].astype(bool)
    x_t = sp["X1_te"][valid].astype(np.float32)
    sigma = (
        sp["DE_te"][valid].squeeze(-1) / float(sp["sd_m"])
    ).astype(np.float32)
    x_m_clean = np.concatenate(
        [sp["X2_te"][valid], sigma[..., None]], axis=-1
    ).astype(np.float32)
    x_m_adv = np.concatenate(
        [sp["X2_te_adv"][valid], sigma[..., None]], axis=-1
    ).astype(np.float32)

    rows = []
    for name, relative in MODELS.items():
        tf.keras.backend.clear_session()
        encoder = tf.keras.models.load_model(
            REPO / relative / "encoder_backbone.keras",
            custom_objects={"_Sampling": _Sampling},
            compile=False,
            safe_mode=False,
        )
        clean = encode(encoder, x_t, x_m_clean)
        adv = encode(encoder, x_t, x_m_adv)
        delta = adv - clean
        l2 = np.linalg.norm(delta, axis=1)
        denominator = (
            np.linalg.norm(clean, axis=1) * np.linalg.norm(adv, axis=1)
        )
        cosine = np.divide(
            np.sum(clean * adv, axis=1),
            denominator,
            out=np.ones_like(denominator),
            where=denominator > 0,
        )
        scale = clean.std(axis=0) + 1e-8
        centroid_shift = np.abs(adv.mean(axis=0) - clean.mean(axis=0)) / scale
        ks = np.array([
            ks_2samp(clean[:, dim], adv[:, dim]).statistic
            for dim in range(clean.shape[1])
        ])
        rows.append({
            "model": name,
            "n_test": len(clean),
            "mean_l2_shift": float(l2.mean()),
            "median_l2_shift": float(np.median(l2)),
            "mean_cosine_similarity": float(cosine.mean()),
            "mean_centroid_shift_sigma": float(centroid_shift.mean()),
            "max_centroid_shift_sigma": float(centroid_shift.max()),
            "mean_ks": float(ks.mean()),
            "max_ks": float(ks.max()),
        })
        print(rows[-1])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
