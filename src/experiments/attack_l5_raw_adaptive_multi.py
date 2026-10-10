"""Multi-restart adaptive PGD against L5-raw on all three surveys."""
from __future__ import annotations

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

SURVEYS = {
    "ogle": {
        "adv": "runs/adversarial_dataset",
        "model": "runs/robust_raw_l5_ogle",
        "labels": ["ecl", "rrlyr", "cep", "lpv"],
    },
    "gaia": {
        "adv": "runs/gaia_adversarial_dataset",
        "model": "runs/gaia_robust_raw_l5",
        "labels": ["rrlyr", "lpv", "cep", "dsct"],
    },
    "wise": {
        "adv": "runs/wise_adversarial_dataset",
        "model": "runs/wise_robust_raw_l5",
        "labels": ["rrlyr", "lpv", "cep", "dsct"],
    },
}


def predict(encoder, classifier, x_t, x_m, batch_size=BATCH):
    labels = np.zeros(len(x_t), np.int64)
    for start in range(0, len(x_t), batch_size):
        end = min(start + batch_size, len(x_t))
        mu, _, _ = encoder(
            [x_t[start:end], x_m[start:end]], training=False
        )
        labels[start:end] = classifier(
            mu, training=False
        ).numpy().argmax(1)
    return labels


def run_survey(name, config, k_budget=K):
    sp = np.load(
        REPO / config["adv"] / "test_split.npz", allow_pickle=True
    )
    x_t = sp["X1_te"].astype(np.float32)
    x2 = sp["X2_te"].astype(np.float32)
    de = sp["DE_te"].astype(np.float32)
    e_obs = sp["E_te"].squeeze(-1).astype(np.float32)
    n_sel = sp["n_sel_te"].astype(np.int32)
    y = sp["y_te"].astype(np.int64)
    valid = sp["valid_te"].astype(bool)
    sd_m = float(sp["sd_m"])
    sigma = (de.squeeze(-1) / max(abs(sd_m), 1e-8)).astype(np.float32)
    x_m = np.concatenate([x2, sigma[..., None]], axis=-1)
    eps = encode_magnitude_budget(e_obs, n_sel, sd_m, k_budget)

    encoder = tf.keras.models.load_model(
        REPO / config["model"] / "encoder_backbone.keras",
        custom_objects={"_Sampling": _Sampling},
        compile=False,
        safe_mode=False,
    )
    classifier = tf.keras.models.load_model(
        REPO / config["model"] / "classifier.keras",
        compile=False,
        safe_mode=False,
    )
    pred_clean = predict(encoder, classifier, x_t, x_m)
    pred_adv = np.zeros(len(y), np.int64)
    y_onehot = tf.keras.utils.to_categorical(
        y, len(config["labels"])
    ).astype(np.float32)

    for start in range(0, len(y), BATCH):
        end = min(start + BATCH, len(y))
        dt_batch = tf.constant(x_t[start:end, :, 0], tf.float32)
        dm_batch = tf.constant(x2[start:end, :, 0], tf.float32)
        sigma_batch = tf.constant(sigma[start:end], tf.float32)
        labels_batch = tf.constant(y_onehot[start:end], tf.float32)
        eps_batch = tf.constant(eps[start:end], tf.float32)

        def forward(dt, dm):
            tower = tf.stack([dm, sigma_batch], axis=-1)
            mu, _, _ = encoder([dt[..., None], tower], training=False)
            return classifier(mu, training=False)

        best_loss = np.full(end - start, -np.inf, np.float32)
        best_pred = np.zeros(end - start, np.int64)
        for restart in range(RESTARTS):
            tf.random.set_seed(10_000 * restart + start)
            dm_adv, _ = pgd_magnitude_tf(
                forward,
                dt_batch,
                dm_batch,
                labels_batch,
                eps_batch,
                tf.constant(STEPS, tf.int32),
                tf.constant(STEP_FRAC, tf.float32),
                tf.constant(True),
                tf.constant(0.0, tf.float32),
            )
            probabilities = forward(dt_batch, dm_adv)
            losses = tf.keras.losses.categorical_crossentropy(
                labels_batch, probabilities
            ).numpy()
            take = losses > best_loss
            best_loss[take] = losses[take]
            best_pred[take] = probabilities.numpy().argmax(1)[take]
        pred_adv[start:end] = best_pred
        if (start // BATCH) % 10 == 0:
            print(f"{name}: attacked {end}/{len(y)}")

    rows = []
    for class_id, scope in [(-1, "all")] + list(
        enumerate(config["labels"])
    ):
        mask = valid if class_id == -1 else valid & (y == class_id)
        clean = float(accuracy_score(y[mask], pred_clean[mask]))
        adversarial = float(accuracy_score(y[mask], pred_adv[mask]))
        rows.append({
            "model": "L5-raw",
            "attack": "adaptive_pgd",
            "scope": scope,
            "k": k_budget,
            "steps": STEPS,
            "step_frac": STEP_FRAC,
            "restarts": RESTARTS,
            "parameterization": "raw_magnitude_displacement",
            "acc_clean": clean,
            "acc_adv": adversarial,
            "drop": adversarial - clean,
            "n_valid": int(mask.sum()),
        })
    suffix = "" if abs(k_budget - 0.5) < 1e-9 else f"_k{k_budget:.2f}"
    out = METRICS / name / f"adaptive_pgd_l5_raw{suffix}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(
        f"{name}: clean={rows[0]['acc_clean']:.4f} "
        f"adaptive={rows[0]['acc_adv']:.4f}"
    )
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, default=K,
                    help="photometric budget (default 0.5; use 1.0 for M5)")
    ap.add_argument("--surveys", default="ogle,gaia,wise")
    args = ap.parse_args()
    wanted = set(s.strip() for s in args.surveys.split(","))
    combined = []
    for survey, config in SURVEYS.items():
        if survey not in wanted:
            continue
        for row in run_survey(survey, config, k_budget=args.k):
            combined.append({"survey": survey, **row})
    suffix = "" if abs(args.k - 0.5) < 1e-9 else f"_k{args.k:.2f}"
    pd.DataFrame(combined).to_csv(
        METRICS / f"adaptive_pgd_l5_raw_all{suffix}.csv", index=False
    )


if __name__ == "__main__":
    main()
