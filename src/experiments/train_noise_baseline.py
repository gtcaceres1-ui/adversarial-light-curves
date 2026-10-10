"""Random-noise-augmentation control baseline.

Trains the L1 raw encoder (clean cross-entropy only, no PGD) with input
augmentation drawn inside the photometric budget at every batch:

    uniform:  delta_mag ~ U(-k e_i, +k e_i)
    gauss:    delta_mag ~ N(0, (k e_i / 2)^2), clipped to +/- k e_i

The displacement is applied in raw-magnitude space and converted to the
standardized Delta-m channel with the same parameterization used by the PGD
attack (magnitude_delta_to_dm_tf), so the augmentation explores exactly the
same perturbation set as the adversary, but sampled at random instead of
optimized. If this baseline matched L5-raw's robustness, adversarial
training would be unnecessary; the comparison quantifies how much of the
robustness requires worst-case (rather than average-case) training.

Run (CPU, ~30-50 min like an L1 run):
    python -m src.experiments.train_noise_baseline --dist uniform
    python -m src.experiments.train_noise_baseline --dist gauss

Outputs:
    runs/robust_raw_l1_noise_{dist}[_s{seed}]/   (encoder/classifier)
    outputs/metrics/ogle/noise_baseline.csv      (appended row per run)

Evaluation: clean test + transferred CNN-targeted adversarial test set (same
protocol as the ablation). For the adaptive white-box evaluation use
attack_raw_ablation_adaptive.py pointing at the saved run directory.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score

from src.attacks.photometric_pgd import magnitude_delta_to_dm_tf
from src.data.raw_sequences import make_robust_inputs
from src.experiments.build_adversarials_ogle import LABELS, SEED, SEQ_LEN
from src.experiments.train_robust_raw_ablation import (
    ADV_K, BATCH, LATENT, LR,
    build_training_data, bundle_from_split,
    make_adversarial_tower, predict_labels,
)
from src.models.robust_raw_encoder import RobustRawEncoder

REPO = Path(__file__).resolve().parents[2]
ADV_ROOT = REPO / "runs/adversarial_dataset"
METRICS = REPO / "outputs/metrics/ogle"


def noise_map(dist: str, seed: int):
    """Return a tf.data map that perturbs the Delta-m channel within k*e."""

    def _map(x_t, x_m, eps, y):
        valid = eps >= 0.0
        budget = tf.maximum(eps, 0.0)
        if dist == "uniform":
            delta = tf.random.uniform(
                tf.shape(budget), -1.0, 1.0, seed=seed) * budget
        else:  # gauss, sigma = budget/2, clipped to the budget
            delta = tf.random.normal(
                tf.shape(budget), 0.0, 1.0, seed=seed) * budget / 2.0
            delta = tf.clip_by_value(delta, -budget, budget)
        dm_delta = magnitude_delta_to_dm_tf(
            delta, tf.cast(valid, tf.float32))
        x_m_noised = tf.concat(
            [x_m[..., :1] + dm_delta[..., None], x_m[..., 1:]], axis=-1)
        return x_t, x_m_noised, eps, y

    return _map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", choices=["uniform", "gauss"], default="uniform")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--k", type=float, default=ADV_K)
    args = ap.parse_args()

    suffix = "" if args.seed == SEED else f"_s{args.seed}"
    run_name = f"robust_raw_l1_noise_{args.dist}{suffix}"
    out_dir = REPO / "runs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] data")
    sp = np.load(ADV_ROOT / "test_split.npz", allow_pickle=True)
    sd_m = float(sp["sd_m"])
    train, y_train = build_training_data(sp)
    val = bundle_from_split(sp, "val")
    test = bundle_from_split(sp, "te")
    y_val = sp["y_val"].astype(np.int64)
    y_test = sp["y_te"].astype(np.int64)

    x_t_tr, x_m_tr, eps_tr = make_robust_inputs(train, sd_m, args.k)
    x_t_va, x_m_va, eps_va = make_robust_inputs(val, sd_m, args.k)
    x_t_te, x_m_te, _ = make_robust_inputs(test, sd_m, args.k)
    x_m_te_adv = make_adversarial_tower(test, sp["X2_te_adv"], sd_m)

    print("[2/3] train", run_name)
    tf.keras.utils.set_random_seed(args.seed)
    model = RobustRawEncoder(
        seq_len=SEQ_LEN, latent_dim=LATENT, n_classes=len(LABELS),
        use_adv_ce=False, use_trades=False, use_supcon=False, use_vib=False,
        adv_k=args.k, adv_steps=0, adv_step_frac=0.0, attack_mode="ce",
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(LR))
    tv, vv = train["valid"], val["valid"]
    train_ds = (
        tf.data.Dataset.from_tensor_slices(
            (x_t_tr[tv], x_m_tr[tv], eps_tr[tv], y_train[tv]))
        .shuffle(8192, seed=args.seed)
        .batch(BATCH)
        .map(noise_map(args.dist, args.seed),  # per-batch: helpers want (B,L)
             num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE))
    val_ds = (
        tf.data.Dataset.from_tensor_slices(
            (x_t_va[vv], x_m_va[vv], eps_va[vv], y_val[vv]))
        .batch(BATCH).prefetch(tf.data.AUTOTUNE))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_acc_cl", patience=8,
            restore_best_weights=True, mode="max"),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_acc_cl", patience=4, factor=0.5,
            min_lr=1e-6, mode="max", verbose=1),
    ]
    t0 = time.time()
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
              callbacks=callbacks, verbose=2)
    wall = time.time() - t0
    model.encoder_backbone.save(out_dir / "encoder_backbone.keras")
    model.classifier.save(out_dir / "classifier.keras")

    print("[3/3] eval")
    pred_c = predict_labels(model, x_t_te, x_m_te)
    pred_a = predict_labels(model, x_t_te, x_m_te_adv)
    valid = test["valid"]
    row = {
        "model": run_name, "dist": args.dist, "k": args.k,
        "seed": args.seed,
        "acc_clean": float(accuracy_score(y_test[valid], pred_c[valid])),
        "acc_adv": float(accuracy_score(y_test[valid], pred_a[valid])),
        "n_test": int(valid.sum()), "wall_s": round(wall, 1),
    }
    row["drop"] = row["acc_clean"] - row["acc_adv"]
    for cid, lab in enumerate(LABELS):
        m = valid & (y_test == cid)
        row[f"acc_cl_{lab}"] = float(accuracy_score(y_test[m], pred_c[m]))
        row[f"acc_ad_{lab}"] = float(accuracy_score(y_test[m], pred_a[m]))
    out_csv = METRICS / "noise_baseline.csv"
    METRICS.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if out_csv.exists():
        df = pd.concat([pd.read_csv(out_csv), df], ignore_index=True)
    df.to_csv(out_csv, index=False)
    print(df.tail(1).to_string(index=False))


if __name__ == "__main__":
    main()
