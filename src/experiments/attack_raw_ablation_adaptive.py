"""Adaptive magnitude-space PGD for every OGLE L1--L5 ablation level."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score

from src.attacks.photometric_pgd import encode_magnitude_budget, pgd_magnitude_tf
from src.models.robust_raw_encoder import _Sampling

REPO = Path(__file__).resolve().parents[2]
ADV_ROOT = REPO / "runs/adversarial_dataset"
OUT = REPO / "outputs/metrics/ogle/adaptive_pgd_raw_ablation.csv"
K = 0.5
STEPS = 20
STEP_FRAC = 0.2
RESTARTS = 3
BATCH = 256
MODELS = {
    "L1": "runs/robust_raw_l1_ce",
    "L2": "runs/robust_raw_l2_advce",
    "L3": "runs/robust_raw_l3_trades",
    "L4": "runs/robust_raw_l4_supcon",
    "L5": "runs/robust_raw_l5_ogle",
}


def run_model(name, model_path, x_t, x2, sigma, eps, y, valid):
    tf.keras.backend.clear_session()
    encoder = tf.keras.models.load_model(
        model_path / "encoder_backbone.keras",
        custom_objects={"_Sampling": _Sampling},
        compile=False,
        safe_mode=False,
    )
    classifier = tf.keras.models.load_model(
        model_path / "classifier.keras", compile=False, safe_mode=False
    )
    pred_clean = np.zeros(len(y), np.int64)
    pred_adv = np.zeros(len(y), np.int64)
    y_onehot = tf.keras.utils.to_categorical(y, 4).astype(np.float32)

    for start in range(0, len(y), BATCH):
        end = min(start + BATCH, len(y))
        dt = tf.constant(x_t[start:end, :, 0], tf.float32)
        dm = tf.constant(x2[start:end, :, 0], tf.float32)
        sig = tf.constant(sigma[start:end], tf.float32)
        labels = tf.constant(y_onehot[start:end], tf.float32)
        budget = tf.constant(eps[start:end], tf.float32)

        def forward(dt_value, dm_value):
            tower = tf.stack([dm_value, sig], axis=-1)
            mu, _, _ = encoder([dt_value[..., None], tower], training=False)
            return classifier(mu, training=False)

        clean_probability = forward(dt, dm).numpy()
        pred_clean[start:end] = clean_probability.argmax(1)
        best_loss = np.full(end - start, -np.inf, np.float32)
        best_pred = np.zeros(end - start, np.int64)
        for restart in range(RESTARTS):
            tf.random.set_seed(start + 1000 * restart)
            dm_adv, _ = pgd_magnitude_tf(
                forward, dt, dm, labels, budget,
                tf.constant(STEPS, tf.int32),
                tf.constant(STEP_FRAC, tf.float32),
                tf.constant(True),
                tf.constant(0.0, tf.float32),
            )
            probability = forward(dt, dm_adv).numpy()
            loss = tf.keras.losses.categorical_crossentropy(
                labels, probability
            ).numpy()
            take = loss > best_loss
            best_loss[take] = loss[take]
            best_pred[take] = probability.argmax(1)[take]
        pred_adv[start:end] = best_pred
    clean = float(accuracy_score(y[valid], pred_clean[valid]))
    adversarial = float(accuracy_score(y[valid], pred_adv[valid]))
    print(f"{name}: clean={clean:.4f} adaptive={adversarial:.4f}")
    return {
        "model": name,
        "acc_clean": clean,
        "acc_adv": adversarial,
        "drop": clean - adversarial,
        "k": K,
        "steps": STEPS,
        "step_frac": STEP_FRAC,
        "restarts": RESTARTS,
        "n_test": int(valid.sum()),
    }


def main():
    sp = np.load(ADV_ROOT / "test_split.npz", allow_pickle=True)
    x_t = sp["X1_te"].astype(np.float32)
    x2 = sp["X2_te"].astype(np.float32)
    sigma = (
        sp["DE_te"].squeeze(-1) / float(sp["sd_m"])
    ).astype(np.float32)
    eps = encode_magnitude_budget(
        sp["E_te"].squeeze(-1),
        sp["n_sel_te"],
        float(sp["sd_m"]),
        K,
    )
    y = sp["y_te"].astype(np.int64)
    valid = sp["valid_te"].astype(bool)
    rows = [
        run_model(
            name, REPO / path, x_t, x2, sigma, eps, y, valid
        )
        for name, path in MODELS.items()
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
