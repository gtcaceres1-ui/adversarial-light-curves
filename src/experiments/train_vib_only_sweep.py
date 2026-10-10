"""VIB-only beta sweep on OGLE without adversarial training."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score

from src.data.raw_sequences import make_robust_inputs
from src.experiments.build_adversarials_ogle import LABELS, SEED, SEQ_LEN
from src.experiments.train_robust_raw_ablation import (
    ADV_K,
    ADV_ROOT,
    ADV_STEP_FRAC,
    ADV_STEPS,
    BATCH,
    LATENT,
    LAMBDA_ADV_CE,
    LAMBDA_SUPCON,
    LAMBDA_TRADES,
    LR,
    METRICS,
    SUPCON_TAU,
    SaveBestSubmodels,
    build_training_data,
    bundle_from_split,
    make_adversarial_tower,
    predict_labels,
)
from src.models.robust_raw_encoder import RobustRawEncoder

REPO = Path(__file__).resolve().parents[2]
BETAS = [5e-4, 1e-3, 1e-2, 1e-1, 1.0]
T_STOCH = 16


def predict_stochastic(model, x_t, x_m, samples=T_STOCH, batch_size=512):
    probabilities = np.zeros((len(x_t), len(LABELS)), np.float32)
    for start in range(0, len(x_t), batch_size):
        end = min(start + batch_size, len(x_t))
        mu, log_var, _ = model.encoder_backbone(
            [x_t[start:end], x_m[start:end]], training=False
        )
        std = tf.exp(0.5 * log_var)
        total = np.zeros((end - start, len(LABELS)), np.float32)
        for _ in range(samples):
            z = mu + std * tf.random.normal(tf.shape(mu))
            total += model.classifier(z, training=False).numpy()
        probabilities[start:end] = total / samples
    return probabilities.argmax(1)


def train_beta(
    beta,
    train,
    y_train,
    val,
    y_val,
    test,
    y_test,
    x2_test_adv,
    sd_m,
    epochs,
):
    tf.keras.backend.clear_session()
    tag = f"{beta:.0e}"
    out_dir = REPO / f"runs/vib_only_b{tag}"
    x_t_train, x_m_train, eps_train = make_robust_inputs(
        train, sd_m, ADV_K
    )
    x_t_val, x_m_val, eps_val = make_robust_inputs(val, sd_m, ADV_K)
    x_t_test, x_m_test, _ = make_robust_inputs(test, sd_m, ADV_K)
    x_m_test_adv = make_adversarial_tower(test, x2_test_adv, sd_m)
    train_valid = train["valid"]
    val_valid = val["valid"]

    tf.keras.utils.set_random_seed(SEED)
    model = RobustRawEncoder(
        seq_len=SEQ_LEN,
        latent_dim=LATENT,
        n_classes=len(LABELS),
        use_adv_ce=False,
        use_trades=False,
        use_supcon=False,
        use_vib=True,
        lambda_adv_ce=LAMBDA_ADV_CE,
        lambda_trades=LAMBDA_TRADES,
        lambda_supcon=LAMBDA_SUPCON,
        beta_vib=beta,
        supcon_tau=SUPCON_TAU,
        adv_k=ADV_K,
        adv_steps=ADV_STEPS,
        adv_step_frac=ADV_STEP_FRAC,
        attack_mode="ce",
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(LR))
    train_ds = (
        tf.data.Dataset.from_tensor_slices((
            x_t_train[train_valid],
            x_m_train[train_valid],
            eps_train[train_valid],
            y_train[train_valid],
        ))
        .shuffle(8192, seed=SEED)
        .batch(BATCH)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((
            x_t_val[val_valid],
            x_m_val[val_valid],
            eps_val[val_valid],
            y_val[val_valid],
        ))
        .batch(BATCH)
        .prefetch(tf.data.AUTOTUNE)
    )
    callbacks = [
        SaveBestSubmodels(model, out_dir, "val_acc_cl"),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_acc_cl", patience=8,
            restore_best_weights=True, mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_acc_cl", patience=4, factor=0.5,
            min_lr=1e-6, mode="max", verbose=1,
        ),
    ]
    start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )
    wall = time.time() - start
    model.encoder_backbone.save(out_dir / "encoder_backbone.keras")
    model.classifier.save(out_dir / "classifier.keras")

    valid = test["valid"]
    pred_clean = predict_labels(model, x_t_test, x_m_test)
    pred_adv = predict_labels(model, x_t_test, x_m_test_adv)
    tf.keras.utils.set_random_seed(SEED)
    pred_clean_stoch = predict_stochastic(model, x_t_test, x_m_test)
    pred_adv_stoch = predict_stochastic(model, x_t_test, x_m_test_adv)
    row = {
        "model": "vib_only",
        "beta": beta,
        "acc_clean": float(accuracy_score(y_test[valid], pred_clean[valid])),
        "acc_adv": float(accuracy_score(y_test[valid], pred_adv[valid])),
        "acc_clean_stoch": float(
            accuracy_score(y_test[valid], pred_clean_stoch[valid])
        ),
        "acc_adv_stoch": float(
            accuracy_score(y_test[valid], pred_adv_stoch[valid])
        ),
        "n_test": int(valid.sum()),
        "epochs_run": len(history.history["loss"]),
        "wall_s": wall,
        "best_val_acc_cl": float(max(history.history["val_acc_cl"])),
    }
    row["drop"] = row["acc_clean"] - row["acc_adv"]
    row["drop_stoch"] = (
        row["acc_clean_stoch"] - row["acc_adv_stoch"]
    )
    for class_id, label in enumerate(LABELS):
        mask = valid & (y_test == class_id)
        row[f"acc_cl_{label}"] = float(
            accuracy_score(y_test[mask], pred_clean[mask])
        )
        row[f"acc_ad_{label}"] = float(
            accuracy_score(y_test[mask], pred_adv[mask])
        )
    (out_dir / "cfg.json").write_text(json.dumps({
        **row,
        "split": "70/10/20",
        "test_used_for_selection": False,
        "use_adv_ce": False,
        "use_trades": False,
        "use_supcon": False,
        "use_vib": True,
        "T_stoch": T_STOCH,
    }, indent=2))
    print(
        f"beta={beta:g}: clean={row['acc_clean']:.4f} "
        f"adv={row['acc_adv']:.4f} stochastic_adv={row['acc_adv_stoch']:.4f}"
    )
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument(
        "--betas", default=",".join(f"{value:g}" for value in BETAS)
    )
    parser.add_argument("--out", default="vib_only_beta_sweep.csv")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    betas = [float(value) for value in args.betas.split(",")]
    METRICS.mkdir(parents=True, exist_ok=True)

    sp = np.load(ADV_ROOT / "test_split.npz", allow_pickle=True)
    train, y_train = build_training_data(sp)
    val = bundle_from_split(sp, "val")
    test = bundle_from_split(sp, "te")
    y_val = sp["y_val"].astype(np.int64)
    y_test = sp["y_te"].astype(np.int64)
    sd_m = float(sp["sd_m"])

    out_csv = METRICS / args.out
    results = {}
    if out_csv.exists() and not args.force:
        for _, old in pd.read_csv(out_csv).iterrows():
            results[f"{float(old['beta']):g}"] = dict(old)
    for beta in betas:
        key = f"{beta:g}"
        if key in results and not args.force:
            print(f"beta={key} already present; skipping")
            continue
        results[key] = train_beta(
            beta,
            train,
            y_train,
            val,
            y_val,
            test,
            y_test,
            sp["X2_te_adv"],
            sd_m,
            args.epochs,
        )
        pd.DataFrame(
            [results[key] for key in sorted(results, key=float)]
        ).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
