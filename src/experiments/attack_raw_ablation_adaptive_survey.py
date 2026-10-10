"""Adaptive magnitude-space PGD for the L1/L2/L5 ablation on Gaia or WISE.

Attacks every ablation variant of a survey with the *same* protocol used for
the canonical L5 adaptive evaluation (``attack_l5_raw_adaptive_multi``:
T=40 steps, step 0.1*eps_i, 5 random restarts, raw-magnitude budget k=0.5),
so the resulting cross-survey ablation rows are directly comparable.

Usage:
    python3.12 -m src.experiments.attack_raw_ablation_adaptive_survey \
        --survey gaia --levels 1,2,5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score

from src.attacks.photometric_pgd import encode_magnitude_budget, pgd_magnitude_tf
from src.models.robust_raw_encoder import _Sampling

REPO = Path(__file__).resolve().parents[2]
METRICS = REPO / "outputs/metrics"
K = 0.5
STEPS = 40
STEP_FRAC = 0.1
RESTARTS = 5
BATCH = 256

MODEL_DIRS = {
    "gaia": {
        "L1": "runs/gaia_robust_raw_l1",
        "L2": "runs/gaia_robust_raw_l2",
        "L5": "runs/gaia_robust_raw_l5",
    },
    "wise": {
        "L1": "runs/wise_robust_raw_l1",
        "L2": "runs/wise_robust_raw_l2",
        "L5": "runs/wise_robust_raw_l5",
    },
}


def run_model(name, model_path, x_t, x2, sigma, eps, y, valid, n_classes):
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
    y_onehot = tf.keras.utils.to_categorical(y, n_classes).astype(np.float32)

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

        pred_clean[start:end] = forward(dt, dm).numpy().argmax(1)
        best_loss = np.full(end - start, -np.inf, np.float32)
        best_pred = np.zeros(end - start, np.int64)
        for restart in range(RESTARTS):
            tf.random.set_seed(10_000 * restart + start)
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
        if (start // BATCH) % 10 == 0:
            print(f"{name}: attacked {end}/{len(y)}", flush=True)
    clean = float(accuracy_score(y[valid], pred_clean[valid]))
    adversarial = float(accuracy_score(y[valid], pred_adv[valid]))
    print(f"{name}: clean={clean:.4f} adaptive={adversarial:.4f}", flush=True)
    return {
        "model": name,
        "acc_clean": clean,
        "acc_adv": adversarial,
        "drop": clean - adversarial,
        "k": K,
        "steps": STEPS,
        "step_frac": STEP_FRAC,
        "restarts": RESTARTS,
        "parameterization": "raw_magnitude_displacement",
        "n_test": int(valid.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--survey", required=True, choices=["gaia", "wise"])
    parser.add_argument("--levels", default="1,2,5")
    args = parser.parse_args()
    wanted = {f"L{x.strip()}" for x in args.levels.split(",")}

    adv_root = REPO / f"runs/{args.survey}_adversarial_dataset"
    sp = np.load(adv_root / "test_split.npz", allow_pickle=True)
    x_t = sp["X1_te"].astype(np.float32)
    x2 = sp["X2_te"].astype(np.float32)
    sigma = (sp["DE_te"].squeeze(-1) / float(sp["sd_m"])).astype(np.float32)
    eps = encode_magnitude_budget(
        sp["E_te"].squeeze(-1),
        sp["n_sel_te"],
        float(sp["sd_m"]),
        K,
    )
    y = sp["y_te"].astype(np.int64)
    valid = sp["valid_te"].astype(bool)

    rows = []
    for name, rel_path in MODEL_DIRS[args.survey].items():
        if name not in wanted:
            continue
        rows.append(run_model(
            name, REPO / rel_path, x_t, x2, sigma, eps, y, valid, 4
        ))
    out = METRICS / args.survey / "adaptive_pgd_raw_ablation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
