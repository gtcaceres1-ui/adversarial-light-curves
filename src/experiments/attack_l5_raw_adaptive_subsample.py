"""Adaptive white-box PGD on the exact Square-attack subsample.

Makes the black-box vs white-box comparison apples-to-apples:
runs the canonical adaptive CE-PGD (T=40, 5 restarts, k=0.5) on the same
800-curve stratified subsample selected by attack_l5_raw_square (same
seed and selection logic).

Usage:
    python3.12 -m src.experiments.attack_l5_raw_adaptive_subsample
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score

from src.attacks.photometric_pgd import encode_magnitude_budget, pgd_magnitude_tf
from src.experiments.attack_l5_raw_square import (
    ADV, MODELS, N_PER_CLASS, SEED, K,
)
from src.models.robust_raw_encoder import _Sampling

REPO = Path(__file__).resolve().parents[2]
STEPS = 40
STEP_FRAC = 0.1
RESTARTS = 5
BATCH = 256


def main():
    survey = "ogle"
    sp = np.load(REPO / ADV[survey] / "test_split.npz", allow_pickle=True)
    sd_m = float(sp["sd_m"])
    x_t = sp["X1_te"].astype(np.float32)
    x2 = sp["X2_te"][..., 0].astype(np.float32)
    sigma = (sp["DE_te"].squeeze(-1) / max(abs(sd_m), 1e-8)).astype(np.float32)
    eps = encode_magnitude_budget(
        sp["E_te"].squeeze(-1), sp["n_sel_te"], sd_m, K
    )
    y = sp["y_te"].astype(np.int64)
    valid = sp["valid_te"].astype(bool)

    # EXACT same subsample as attack_l5_raw_square
    rng = np.random.default_rng(SEED)
    take = []
    for c in np.unique(y):
        pool = np.where(valid & (y == c))[0]
        take.append(rng.choice(pool, min(N_PER_CLASS, len(pool)),
                               replace=False))
    idx = np.sort(np.concatenate(take))
    x_t, x2, sigma, eps, y = x_t[idx], x2[idx], sigma[idx], eps[idx], y[idx]

    encoder = tf.keras.models.load_model(
        REPO / MODELS[survey] / "encoder_backbone.keras",
        custom_objects={"_Sampling": _Sampling}, compile=False,
        safe_mode=False,
    )
    classifier = tf.keras.models.load_model(
        REPO / MODELS[survey] / "classifier.keras", compile=False,
        safe_mode=False,
    )
    y_oh = tf.keras.utils.to_categorical(y, 4).astype(np.float32)
    pred_clean = np.zeros(len(y), np.int64)
    pred_adv = np.zeros(len(y), np.int64)

    for s in range(0, len(y), BATCH):
        e = min(s + BATCH, len(y))
        dt = tf.constant(x_t[s:e, :, 0])
        dm = tf.constant(x2[s:e])
        sig = tf.constant(sigma[s:e])
        labels = tf.constant(y_oh[s:e])
        budget = tf.constant(eps[s:e])

        def forward(dt_v, dm_v):
            tower = tf.stack([dm_v, sig], axis=-1)
            mu, _, _ = encoder([dt_v[..., None], tower], training=False)
            return classifier(mu, training=False)

        pred_clean[s:e] = forward(dt, dm).numpy().argmax(1)
        best_loss = np.full(e - s, -np.inf, np.float32)
        best_pred = np.zeros(e - s, np.int64)
        for r in range(RESTARTS):
            tf.random.set_seed(10_000 * r + s)
            dm_adv, _ = pgd_magnitude_tf(
                forward, dt, dm, labels, budget,
                tf.constant(STEPS, tf.int32),
                tf.constant(STEP_FRAC, tf.float32),
                tf.constant(True),
                tf.constant(0.0, tf.float32),
            )
            prob = forward(dt, dm_adv).numpy()
            loss = tf.keras.losses.categorical_crossentropy(
                labels, prob).numpy()
            keep = loss > best_loss
            best_loss[keep] = loss[keep]
            best_pred[keep] = prob.argmax(1)[keep]
        pred_adv[s:e] = best_pred
        print(f"attacked {e}/{len(y)}", flush=True)

    clean = float(accuracy_score(y, pred_clean))
    adv = float(accuracy_score(y, pred_adv))
    row = {
        "model": "RRE", "attack": "adaptive_pgd_subsample",
        "survey": survey, "k": K, "steps": STEPS, "restarts": RESTARTS,
        "n_subsample": len(y), "acc_clean_sub": clean, "acc_adv_sub": adv,
        "drop": adv - clean,
    }
    out = (REPO / "outputs/metrics/ogle" /
           "adaptive_pgd_subsample_k0.50.csv")
    pd.DataFrame([row]).to_csv(out, index=False)
    print(row, flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
