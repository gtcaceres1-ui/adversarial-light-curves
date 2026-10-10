"""Margin-based (DLR) adaptive PGD against L5-raw.

Same protocol as attack_l5_raw_adaptive_multi (raw-magnitude displacement,
k=0.5, multi-restart) but the attack objective is the untargeted DLR loss
of APGD-DLR (Croce & Hein 2020) instead of cross-entropy. Because softmax
is shift-invariant, logit differences equal log-probability differences,
so DLR is computed exactly from the classifier probabilities.

Also supports --steps to produce an iteration-convergence curve.

Usage:
    python -m src.experiments.attack_l5_raw_adaptive_dlr --surveys ogle
    python -m src.experiments.attack_l5_raw_adaptive_dlr --surveys ogle --steps 100 --restarts 1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score

from src.attacks.photometric_pgd import (
    apply_magnitude_delta_tf,
    encode_magnitude_budget,
)
from src.models.robust_raw_encoder import _Sampling

REPO = Path(__file__).resolve().parents[2]
METRICS = REPO / "outputs/metrics"
K = 0.5
BATCH = 256

SURVEYS = {
    "ogle": {"adv": "runs/adversarial_dataset",
             "model": "runs/robust_raw_l5_ogle"},
    "gaia": {"adv": "runs/gaia_adversarial_dataset",
             "model": "runs/gaia_robust_raw_l5"},
    "wise": {"adv": "runs/wise_adversarial_dataset",
             "model": "runs/wise_robust_raw_l5"},
}


def dlr_objective(y_oh, p):
    """Untargeted DLR from probabilities (shift-invariance of softmax)."""
    logp = tf.math.log(tf.clip_by_value(p, 1e-12, 1.0))
    z_y = tf.reduce_sum(logp * y_oh, axis=1)
    z_other = tf.reduce_max(logp - 1e9 * y_oh, axis=1)
    top = tf.sort(logp, axis=1, direction="DESCENDING")
    denom = top[:, 0] - top[:, 2] + 1e-12
    return tf.reduce_mean(-(z_y - z_other) / denom)


def pgd_magnitude_dlr(forward, x_dt, x_dm, y_oh, eps_encoded,
                      steps, step_frac):
    """PGD on magnitude displacements maximizing the DLR objective."""
    x_dt = tf.cast(x_dt, tf.float32)
    x_dm = tf.cast(x_dm, tf.float32)
    y_oh = tf.cast(y_oh, tf.float32)
    encoded = tf.cast(eps_encoded, tf.float32)
    point_valid = encoded >= 0.0
    eps = tf.maximum(encoded, 0.0)
    anchor = tf.concat(
        [tf.zeros_like(eps[:, :1]), tf.ones_like(eps[:, 1:])], axis=1
    )
    perturbable = tf.cast(point_valid, tf.float32) * anchor
    eps = eps * perturbable
    delta = tf.random.uniform(tf.shape(eps), -1.0, 1.0) * eps
    step = tf.maximum(step_frac * eps, 1e-8)
    for _ in range(int(steps)):
        with tf.GradientTape() as tape:
            tape.watch(delta)
            dm_adv = apply_magnitude_delta_tf(x_dm, delta, point_valid)
            objective = dlr_objective(y_oh, forward(x_dt, dm_adv))
        grad = tape.gradient(objective, delta)
        grad = tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(grad))
        delta = tf.clip_by_value(delta + step * tf.sign(grad) * perturbable,
                                 -eps, eps)
    return apply_magnitude_delta_tf(x_dm, delta, point_valid)


def run_survey(name, cfg, steps, restarts, k_budget):
    sp = np.load(REPO / cfg["adv"] / "test_split.npz", allow_pickle=True)
    x_t = sp["X1_te"].astype(np.float32)
    x2 = sp["X2_te"][..., 0].astype(np.float32)
    sd_m = float(sp["sd_m"])
    sigma = (sp["DE_te"].squeeze(-1) / max(abs(sd_m), 1e-8)).astype(np.float32)
    eps = encode_magnitude_budget(
        sp["E_te"].squeeze(-1), sp["n_sel_te"], sd_m, k_budget
    )
    y = sp["y_te"].astype(np.int64)
    valid = sp["valid_te"].astype(bool)

    encoder = tf.keras.models.load_model(
        REPO / cfg["model"] / "encoder_backbone.keras",
        custom_objects={"_Sampling": _Sampling}, compile=False,
        safe_mode=False,
    )
    classifier = tf.keras.models.load_model(
        REPO / cfg["model"] / "classifier.keras", compile=False,
        safe_mode=False,
    )
    y_oh = tf.keras.utils.to_categorical(y, 4).astype(np.float32)
    pred_clean = np.zeros(len(y), np.int64)
    pred_adv = np.zeros(len(y), np.int64)

    for s in range(0, len(y), BATCH):
        e = min(s + BATCH, len(y))
        dt_b = tf.constant(x_t[s:e, :, 0])
        dm_b = tf.constant(x2[s:e])
        sig_b = tf.constant(sigma[s:e])
        y_b = tf.constant(y_oh[s:e])
        eps_b = tf.constant(eps[s:e])

        def forward(dt, dm):
            tower = tf.stack([dm, sig_b], axis=-1)
            mu, _, _ = encoder([dt[..., None], tower], training=False)
            return classifier(mu, training=False)

        pred_clean[s:e] = forward(dt_b, dm_b).numpy().argmax(1)
        best_obj = np.full(e - s, -np.inf, np.float32)
        best_pred = np.zeros(e - s, np.int64)
        for r in range(restarts):
            tf.random.set_seed(10_000 * r + s)
            dm_adv = pgd_magnitude_dlr(
                forward, dt_b, dm_b, y_b, eps_b, steps,
                tf.constant(0.1, tf.float32),
            )
            prob = forward(dt_b, dm_adv).numpy()
            logp = np.log(np.clip(prob, 1e-12, 1.0))
            z_y = logp[np.arange(e - s), y[s:e]]
            masked = logp.copy()
            masked[np.arange(e - s), y[s:e]] = -1e9
            z_other = masked.max(1)
            top = np.sort(logp, 1)[:, ::-1]
            obj = -(z_y - z_other) / (top[:, 0] - top[:, 2] + 1e-12)
            take = obj > best_obj
            best_obj[take] = obj[take]
            best_pred[take] = prob.argmax(1)[take]
        pred_adv[s:e] = best_pred
        if (s // BATCH) % 10 == 0:
            print(f"{name}: attacked {e}/{len(y)}", flush=True)

    clean = float(accuracy_score(y[valid], pred_clean[valid]))
    adv = float(accuracy_score(y[valid], pred_adv[valid]))
    print(f"{name}: clean={clean:.4f} dlr_adaptive={adv:.4f}", flush=True)
    return {
        "model": "L5-raw", "attack": "adaptive_pgd_dlr", "survey": name,
        "k": k_budget, "steps": steps, "restarts": restarts,
        "acc_clean": clean, "acc_adv": adv, "drop": adv - clean,
        "n_valid": int(valid.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surveys", default="ogle")
    ap.add_argument("--k", type=float, default=K)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--restarts", type=int, default=5)
    args = ap.parse_args()
    rows = []
    for s in args.surveys.split(","):
        s = s.strip()
        rows.append(run_survey(s, SURVEYS[s], args.steps, args.restarts,
                               args.k))
    out = (METRICS / rows[0]["survey"] /
           f"adaptive_pgd_dlr_k{args.k:.2f}_T{args.steps}.csv"
           if len(rows) == 1 else
           METRICS / f"adaptive_pgd_dlr_all_k{args.k:.2f}_T{args.steps}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
