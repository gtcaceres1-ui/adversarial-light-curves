"""Control: does freeing delta_1 strengthen the adaptive attack on RRE?

The inner and adaptive PGD anchor the first observation (delta_1 = 0) to
remove the global-offset gauge that differencing makes invisible. With
heteroscedastic budgets, however, anchoring delta_1 explores a strict subset
of the admissible set. This script re-runs the OGLE-III adaptive white-box
attack on RRE with the anchor removed (delta_1 free within its own budget
k*e_1), same protocol otherwise (T=40, 5 restarts, k=0.5), so the accuracy
can be compared against the anchored run.

Usage:
    python -m src.experiments.attack_l5_raw_adaptive_free_delta1
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

from src.attacks.photometric_pgd import (
    apply_magnitude_delta_tf,
    encode_magnitude_budget,
)
from src.models.robust_raw_encoder import _Sampling

REPO = Path(__file__).resolve().parents[2]
ADV = REPO / "runs/adversarial_dataset"
MODEL = REPO / "runs/robust_raw_l5_ogle"
K, STEPS, STEP_FRAC, RESTARTS, BATCH = 0.5, 40, 0.1, 5, 256


def pgd_free_delta1(forward, x_dt, x_dm, y_oh, eps_encoded, seed):
    """PGD on magnitude displacements WITHOUT anchoring delta_1."""
    encoded = tf.cast(eps_encoded, tf.float32)
    point_valid = encoded >= 0.0
    eps = tf.maximum(encoded, 0.0)
    perturbable = tf.cast(point_valid, tf.float32)   # <-- no anchor mask
    eps = eps * perturbable
    tf.random.set_seed(seed)
    delta = tf.random.uniform(tf.shape(eps), -1.0, 1.0) * eps
    step = tf.maximum(STEP_FRAC * eps, 1e-8)
    for _ in range(STEPS):
        with tf.GradientTape() as tape:
            tape.watch(delta)
            dm_adv = apply_magnitude_delta_tf(x_dm, delta, point_valid)
            ce = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(y_oh, forward(x_dt, dm_adv)))
        g = tape.gradient(ce, delta)
        g = tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
        delta = tf.clip_by_value(delta + step * tf.sign(g) * perturbable, -eps, eps)
    return apply_magnitude_delta_tf(x_dm, delta, point_valid)


def main():
    sp = np.load(ADV / "test_split.npz", allow_pickle=True)
    x_t = sp["X1_te"].astype(np.float32)
    x2 = sp["X2_te"][..., 0].astype(np.float32)
    sd_m = float(sp["sd_m"])
    sigma = (sp["DE_te"].squeeze(-1) / max(abs(sd_m), 1e-8)).astype(np.float32)
    eps = encode_magnitude_budget(sp["E_te"].squeeze(-1), sp["n_sel_te"], sd_m, K)
    y = sp["y_te"].astype(np.int64)
    valid = sp["valid_te"].astype(bool)

    enc = tf.keras.models.load_model(MODEL / "encoder_backbone.keras",
        custom_objects={"_Sampling": _Sampling}, compile=False, safe_mode=False)
    clf = tf.keras.models.load_model(MODEL / "classifier.keras",
        compile=False, safe_mode=False)
    y_oh = tf.keras.utils.to_categorical(y, 4).astype(np.float32)
    pred_clean = np.zeros(len(y), np.int64)
    pred_adv = np.zeros(len(y), np.int64)

    for s in range(0, len(y), BATCH):
        e = min(s + BATCH, len(y))
        dt = tf.constant(x_t[s:e, :, 0]); dm = tf.constant(x2[s:e])
        sig = tf.constant(sigma[s:e]); yb = tf.constant(y_oh[s:e])
        eb = tf.constant(eps[s:e])

        def forward(dt_v, dm_v):
            tower = tf.stack([dm_v, sig], axis=-1)
            mu, _, _ = enc([dt_v[..., None], tower], training=False)
            return clf(mu, training=False)

        pred_clean[s:e] = forward(dt, dm).numpy().argmax(1)
        best_loss = np.full(e - s, -np.inf, np.float32)
        best = np.zeros(e - s, np.int64)
        for r in range(RESTARTS):
            dm_adv = pgd_free_delta1(forward, dt, dm, yb, eb, 10_000 * r + s)
            prob = forward(dt, dm_adv).numpy()
            loss = tf.keras.losses.categorical_crossentropy(yb, prob).numpy()
            take = loss > best_loss
            best_loss[take] = loss[take]; best[take] = prob.argmax(1)[take]
        pred_adv[s:e] = best
        if (s // BATCH) % 10 == 0:
            print(f"attacked {e}/{len(y)}", flush=True)

    clean = float(accuracy_score(y[valid], pred_clean[valid]))
    adv = float(accuracy_score(y[valid], pred_adv[valid]))
    print(f"RESULT (delta_1 FREE): clean={clean:.4f} adaptive={adv:.4f} "
          f"(anchored reference: 0.973)", flush=True)


if __name__ == "__main__":
    main()
