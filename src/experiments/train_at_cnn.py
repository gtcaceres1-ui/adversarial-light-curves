"""Adversarially trained CNN baseline.

Trains the *same* two-tower (dt, dm) surrogate CNN architecture used as a
baseline, but with Madry-style adversarial training: at every step the batch
is attacked with the same raw-magnitude PGD used by the robust encoder's
inner attack (S=3 steps, step 0.4*eps, k=0.5), and the loss is
CE(clean) + CE(adv), i.e. the recipe of the L2 ablation variant applied to
the baseline architecture.

Evaluates on: clean test set, transferred CNN-targeted attack (the
adversarial test set), and an adaptive white-box PGD on the AT-CNN's own
gradient (T=40, 5 restarts), so the numbers are directly comparable to both
the baseline table and the ablation tables.

Usage:
    python -m src.experiments.train_at_cnn --survey ogle
    python -m src.experiments.train_at_cnn --survey gaia
    python -m src.experiments.train_at_cnn --survey wise
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score

from src.attacks.photometric_pgd import encode_magnitude_budget, pgd_magnitude_tf
from src.data.raw_sequences import build_array_bundle

REPO = Path(__file__).resolve().parents[2]
K = 0.5
TRAIN_STEPS = 3
TRAIN_STEP_FRAC = 0.4
EVAL_STEPS = 40
EVAL_STEP_FRAC = 0.1
EVAL_RESTARTS = 5
BATCH = 256
LR = 1e-3
DEFAULT_EPOCHS = 15
PATIENCE = 5


def survey_paths(survey):
    if survey == "ogle":
        return {
            "adv": REPO / "runs/adversarial_dataset",
            "cnn": REPO / "runs/cnn_ogle",
            "out": REPO / "runs/ogle_at_cnn",
            "metrics": REPO / "outputs/metrics/ogle",
        }
    return {
        "adv": REPO / f"runs/{survey}_adversarial_dataset",
        "cnn": REPO / f"runs/{survey}_cnn",
        "out": REPO / f"runs/{survey}_at_cnn",
        "metrics": REPO / f"outputs/metrics/{survey}",
    }


def load_training_arrays(survey, sp):
    stats = (
        float(sp["mu_t"]), float(sp["sd_t"]),
        float(sp["mu_m"]), float(sp["sd_m"]),
    )
    if survey == "ogle":
        from src.experiments.build_adversarials_ogle import (
            LABEL_TO_INT, build_catalog, load_lightcurve, subsample_balanced,
        )
        catalog = subsample_balanced(build_catalog())
        load = load_lightcurve
    else:
        from src.data.survey_config import LABEL_TO_INT
        from src.data.survey_loader import (
            build_catalog, load_lightcurve, subsample_balanced,
        )
        catalog = subsample_balanced(build_catalog(survey))
        load = lambda path: load_lightcurve(Path(path), survey)
    y_all = catalog["label4"].map(LABEL_TO_INT).to_numpy(np.int64)
    idx_train = sp["idx_train"].astype(np.int64)
    train_df = catalog.iloc[idx_train].reset_index(drop=True)
    train = build_array_bundle(train_df, load, stats)
    return train, y_all[idx_train]


def make_forward(model, rank3):
    def forward(dt, dm):
        if rank3:
            return model([dt[..., None], dm[..., None]], training=False)
        return model([dt, dm], training=False)
    return forward


def adaptive_attack(model, rank3, dt, dm, eps, y, n_classes):
    """Multi-restart adaptive PGD on the model's own gradient."""
    pred = np.zeros(len(y), np.int64)
    y_oh = tf.keras.utils.to_categorical(y, n_classes).astype(np.float32)
    forward = make_forward(model, rank3)
    for start in range(0, len(y), BATCH):
        end = min(start + BATCH, len(y))
        dt_b = tf.constant(dt[start:end], tf.float32)
        dm_b = tf.constant(dm[start:end], tf.float32)
        y_b = tf.constant(y_oh[start:end], tf.float32)
        eps_b = tf.constant(eps[start:end], tf.float32)
        best_loss = np.full(end - start, -np.inf, np.float32)
        best_pred = np.zeros(end - start, np.int64)
        for restart in range(EVAL_RESTARTS):
            tf.random.set_seed(10_000 * restart + start)
            dm_adv, _ = pgd_magnitude_tf(
                forward, dt_b, dm_b, y_b, eps_b,
                tf.constant(EVAL_STEPS, tf.int32),
                tf.constant(EVAL_STEP_FRAC, tf.float32),
                tf.constant(True),
                tf.constant(0.0, tf.float32),
            )
            prob = forward(dt_b, dm_adv).numpy()
            loss = tf.keras.losses.categorical_crossentropy(y_b, prob).numpy()
            take = loss > best_loss
            best_loss[take] = loss[take]
            best_pred[take] = prob.argmax(1)[take]
        pred[start:end] = best_pred
        if (start // BATCH) % 10 == 0:
            print(f"  adaptive: {end}/{len(y)}", flush=True)
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--survey", required=True,
                        choices=["ogle", "gaia", "wise"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    args = parser.parse_args()
    paths = survey_paths(args.survey)
    paths["out"].mkdir(parents=True, exist_ok=True)
    paths["metrics"].mkdir(parents=True, exist_ok=True)

    sp = np.load(paths["adv"] / "test_split.npz", allow_pickle=True)
    sd_m = float(sp["sd_m"])
    n_classes = 4

    train, y_train = load_training_arrays(args.survey, sp)
    tv = train["valid"]
    dt_tr = train["X1"][tv, :, 0].astype(np.float32)
    dm_tr = train["X2"][tv, :, 0].astype(np.float32)
    eps_tr = encode_magnitude_budget(
        train["E"][tv].squeeze(-1), train["n_sel"][tv], sd_m, K
    )
    y_tr = y_train[tv]

    vv = sp["valid_val"].astype(bool)
    dt_val = sp["X1_val"][vv, :, 0].astype(np.float32)
    dm_val = sp["X2_val"][vv, :, 0].astype(np.float32)
    eps_val = encode_magnitude_budget(
        sp["E_val"][vv].squeeze(-1), sp["n_sel_val"][vv], sd_m, K
    )
    y_val = sp["y_val"].astype(np.int64)[vv]

    dt_te = sp["X1_te"][..., 0].astype(np.float32)
    dm_te = sp["X2_te"][..., 0].astype(np.float32)
    dm_te_adv = sp["X2_te_adv"][..., 0].astype(np.float32)
    eps_te = encode_magnitude_budget(
        sp["E_te"].squeeze(-1), sp["n_sel_te"], sd_m, K
    )
    y_te = sp["y_te"].astype(np.int64)
    valid_te = sp["valid_te"].astype(bool)

    # Same architecture as the canonical surrogate: clone with fresh weights.
    canonical = tf.keras.models.load_model(
        paths["cnn"] / "cnn_two_tower.keras", compile=False, safe_mode=False
    )
    rank3 = len(canonical.inputs[0].shape) == 3
    tf.keras.utils.set_random_seed(42)
    model = tf.keras.models.clone_model(canonical)
    optimizer = tf.keras.optimizers.Adam(LR)
    ce = tf.keras.losses.CategoricalCrossentropy()
    forward = make_forward(model, rank3)

    def eval_acc(dt, dm, y):
        pred = np.zeros(len(y), np.int64)
        for s in range(0, len(y), 512):
            e = min(s + 512, len(y))
            pred[s:e] = forward(dt[s:e], dm[s:e]).numpy().argmax(1)
        return float(accuracy_score(y, pred)), pred

    y_tr_oh = tf.keras.utils.to_categorical(y_tr, n_classes).astype(np.float32)
    n_train = len(y_tr)
    best_val, best_weights, best_epoch, wait = -np.inf, None, -1, 0
    rng = np.random.default_rng(42)
    start_t = time.time()
    for epoch in range(args.epochs):
        order = rng.permutation(n_train)
        losses = []
        for s in range(0, n_train, BATCH):
            idx = order[s:s + BATCH]
            dt_b = tf.constant(dt_tr[idx], tf.float32)
            dm_b = tf.constant(dm_tr[idx], tf.float32)
            y_b = tf.constant(y_tr_oh[idx], tf.float32)
            eps_b = tf.constant(eps_tr[idx], tf.float32)
            dm_adv, _ = pgd_magnitude_tf(
                forward, dt_b, dm_b, y_b, eps_b,
                tf.constant(TRAIN_STEPS, tf.int32),
                tf.constant(TRAIN_STEP_FRAC, tf.float32),
                tf.constant(True),
                tf.constant(0.0, tf.float32),
            )
            with tf.GradientTape() as tape:
                if rank3:
                    p_cl = model([dt_b[..., None], dm_b[..., None]],
                                 training=True)
                    p_ad = model([dt_b[..., None], dm_adv[..., None]],
                                 training=True)
                else:
                    p_cl = model([dt_b, dm_b], training=True)
                    p_ad = model([dt_b, dm_adv], training=True)
                loss = ce(y_b, p_cl) + ce(y_b, p_ad)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            losses.append(float(loss))
        # validation: clean + short-attack adversarial accuracy
        acc_cl_val, _ = eval_acc(dt_val, dm_val, y_val)
        dm_val_adv = np.zeros_like(dm_val)
        for s in range(0, len(y_val), BATCH):
            e = min(s + BATCH, len(y_val))
            adv, _ = pgd_magnitude_tf(
                forward,
                tf.constant(dt_val[s:e], tf.float32),
                tf.constant(dm_val[s:e], tf.float32),
                tf.constant(tf.keras.utils.to_categorical(
                    y_val[s:e], n_classes).astype(np.float32)),
                tf.constant(eps_val[s:e], tf.float32),
                tf.constant(TRAIN_STEPS, tf.int32),
                tf.constant(TRAIN_STEP_FRAC, tf.float32),
                tf.constant(True),
                tf.constant(0.0, tf.float32),
            )
            dm_val_adv[s:e] = adv.numpy()
        acc_ad_val, _ = eval_acc(dt_val, dm_val_adv, y_val)
        print(f"epoch {epoch+1}/{args.epochs}: loss={np.mean(losses):.4f} "
              f"val_cl={acc_cl_val:.4f} val_ad={acc_ad_val:.4f}", flush=True)
        if acc_ad_val > best_val:
            best_val, best_epoch, wait = acc_ad_val, epoch, 0
            best_weights = model.get_weights()
        else:
            wait += 1
            if wait >= PATIENCE:
                print("early stop", flush=True)
                break
    model.set_weights(best_weights)
    wall = time.time() - start_t
    model.save(paths["out"] / "at_cnn_two_tower.keras")

    acc_clean, pred_clean = eval_acc(dt_te, dm_te, y_te)
    acc_transf, pred_transf = eval_acc(dt_te, dm_te_adv, y_te)
    pred_adapt = adaptive_attack(
        model, rank3, dt_te, dm_te, eps_te, y_te, n_classes
    )
    acc_clean = float(accuracy_score(y_te[valid_te], pred_clean[valid_te]))
    acc_transf = float(accuracy_score(y_te[valid_te], pred_transf[valid_te]))
    acc_adapt = float(accuracy_score(y_te[valid_te], pred_adapt[valid_te]))

    row = {
        "model": "AT-CNN",
        "survey": args.survey,
        "acc_clean": acc_clean,
        "acc_transferred": acc_transf,
        "acc_adaptive": acc_adapt,
        "k": K,
        "train_steps": TRAIN_STEPS,
        "eval_steps": EVAL_STEPS,
        "eval_restarts": EVAL_RESTARTS,
        "best_epoch": best_epoch + 1,
        "wall_time_s": wall,
        "n_test": int(valid_te.sum()),
    }
    out_csv = paths["metrics"] / "at_cnn_eval.csv"
    pd.DataFrame([row]).to_csv(out_csv, index=False)
    (paths["out"] / "cfg.json").write_text(json.dumps(row, indent=2))
    print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
