"""Train and evaluate the incremental L1--L5 raw-encoder ablation on OGLE."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score

from src.data.raw_sequences import build_array_bundle, make_robust_inputs
from src.experiments.build_adversarials_ogle import (
    LABELS,
    LABEL_TO_INT,
    SEED,
    SEQ_LEN,
    build_catalog,
    load_lightcurve,
    subsample_balanced,
)
from src.models.robust_raw_encoder import RobustRawEncoder

REPO = Path(__file__).resolve().parents[2]
ADV_ROOT = REPO / "runs/adversarial_dataset"
CNN_DIR = REPO / "runs/cnn_ogle"
METRICS = REPO / "outputs/metrics/ogle"

LATENT = 48
LR = 5e-4
DEFAULT_EPOCHS = 25
BATCH = 256
ADV_K = 0.5
ADV_STEPS = 3
ADV_STEP_FRAC = 0.4
LAMBDA_ADV_CE = 1.0
LAMBDA_TRADES = 2.0
LAMBDA_SUPCON = 0.5
BETA_VIB = 5e-4
SUPCON_TAU = 0.1

ABLATIONS = [
    dict(level=1, run="robust_raw_l1_ce", use_adv_ce=False,
         use_trades=False, use_supcon=False, use_vib=False),
    dict(level=2, run="robust_raw_l2_advce", use_adv_ce=True,
         use_trades=False, use_supcon=False, use_vib=False),
    dict(level=3, run="robust_raw_l3_trades", use_adv_ce=True,
         use_trades=True, use_supcon=False, use_vib=False),
    dict(level=4, run="robust_raw_l4_supcon", use_adv_ce=True,
         use_trades=True, use_supcon=True, use_vib=False),
    dict(level=5, run="robust_raw_l5_ogle", use_adv_ce=True,
         use_trades=True, use_supcon=True, use_vib=True),
]


class SaveBestSubmodels(tf.keras.callbacks.Callback):
    def __init__(self, wrapped_model, out_dir, monitor, mode="max"):
        super().__init__()
        self.wrapped_model = wrapped_model
        self.out_dir = Path(out_dir)
        self.monitor = monitor
        self.best = -np.inf if mode == "max" else np.inf
        self.mode = mode
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        value = (logs or {}).get(self.monitor)
        if value is None:
            return
        improved = value > self.best if self.mode == "max" else value < self.best
        if not improved:
            return
        self.best = float(value)
        self.wrapped_model.encoder_backbone.save(
            self.out_dir / "encoder_backbone.keras"
        )
        self.wrapped_model.classifier.save(self.out_dir / "classifier.keras")
        if self.wrapped_model.proj is not None:
            self.wrapped_model.proj.save(self.out_dir / "projection.keras")
        print(
            f"  [SaveBest] epoch {epoch + 1}: "
            f"{self.monitor}={value:.4f}"
        )


def bundle_from_split(sp, suffix: str) -> dict:
    return {
        "X1": sp[f"X1_{suffix}"],
        "X2": sp[f"X2_{suffix}"],
        "DE": sp[f"DE_{suffix}"],
        "E": sp[f"E_{suffix}"],
        "n_sel": sp[f"n_sel_{suffix}"],
        "valid": sp[f"valid_{suffix}"].astype(bool),
    }


def make_adversarial_tower(bundle: dict, x2_adv: np.ndarray, sd_m: float):
    sigma = (
        bundle["DE"].squeeze(-1) / max(abs(float(sd_m)), 1e-8)
    ).astype(np.float32)
    return np.concatenate(
        [x2_adv, sigma[..., None]], axis=-1
    ).astype(np.float32)


def predict_labels(model, x_t, x_m, batch_size=512):
    out = np.zeros(len(x_t), np.int64)
    for start in range(0, len(x_t), batch_size):
        end = min(start + batch_size, len(x_t))
        mu, _, _ = model.encoder_backbone(
            [x_t[start:end], x_m[start:end]], training=False
        )
        out[start:end] = model.classifier(
            mu, training=False
        ).numpy().argmax(1)
    return out


def build_training_data(sp):
    stats = (
        float(sp["mu_t"]), float(sp["sd_t"]),
        float(sp["mu_m"]), float(sp["sd_m"]),
    )
    catalog = subsample_balanced(build_catalog())
    y_all = catalog["label4"].map(LABEL_TO_INT).to_numpy(np.int64)
    idx_train = sp["idx_train"].astype(np.int64)
    train_df = catalog.iloc[idx_train].reset_index(drop=True)
    train = build_array_bundle(train_df, load_lightcurve, stats)
    return train, y_all[idx_train]


def train_level(
    cfg,
    train,
    y_train,
    val,
    y_val,
    test,
    y_test,
    x2_test_adv,
    sd_m,
    epochs,
    seed: int = SEED,
):
    tf.keras.backend.clear_session()
    suffix = "" if seed == SEED else f"_s{seed}"
    out_dir = REPO / "runs" / (cfg["run"] + suffix)
    print(f"\n=== L{cfg['level']} (seed {seed}) -> {out_dir.name} ===")
    x_t_train, x_m_train, eps_train = make_robust_inputs(
        train, sd_m, ADV_K
    )
    x_t_val, x_m_val, eps_val = make_robust_inputs(val, sd_m, ADV_K)
    x_t_test, x_m_test, _ = make_robust_inputs(test, sd_m, ADV_K)
    x_m_test_adv = make_adversarial_tower(test, x2_test_adv, sd_m)
    train_valid = train["valid"]
    val_valid = val["valid"]

    tf.keras.utils.set_random_seed(seed)
    model = RobustRawEncoder(
        seq_len=SEQ_LEN,
        latent_dim=LATENT,
        n_classes=len(LABELS),
        use_adv_ce=cfg["use_adv_ce"],
        use_trades=cfg["use_trades"],
        use_supcon=cfg["use_supcon"],
        use_vib=cfg["use_vib"],
        lambda_adv_ce=LAMBDA_ADV_CE,
        lambda_trades=LAMBDA_TRADES,
        lambda_supcon=LAMBDA_SUPCON,
        beta_vib=BETA_VIB,
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
        .shuffle(8192, seed=seed)
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
    monitor = "val_acc_ad" if cfg["use_adv_ce"] else "val_acc_cl"
    callbacks = [
        SaveBestSubmodels(model, out_dir, monitor),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=8, restore_best_weights=True, mode="max"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, patience=4, factor=0.5,
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
    if model.proj is not None:
        model.proj.save(out_dir / "projection.keras")

    pred_clean = predict_labels(model, x_t_test, x_m_test)
    pred_adv = predict_labels(model, x_t_test, x_m_test_adv)
    valid = test["valid"]
    acc_clean = float(accuracy_score(y_test[valid], pred_clean[valid]))
    acc_adv = float(accuracy_score(y_test[valid], pred_adv[valid]))
    row = {
        "model": f"robust_raw_l{cfg['level']}" + (
            "" if seed == SEED else f"_s{seed}"),
        "seed": seed,
        "acc_clean": acc_clean,
        "acc_adv": acc_adv,
        "drop": acc_clean - acc_adv,
        "n_test": int(valid.sum()),
    }
    for class_id, label in enumerate(LABELS):
        mask = valid & (y_test == class_id)
        row[f"acc_cl_{label}"] = float(
            accuracy_score(y_test[mask], pred_clean[mask])
        )
        row[f"acc_ad_{label}"] = float(
            accuracy_score(y_test[mask], pred_adv[mask])
        )
        row[f"drop_{label}"] = (
            row[f"acc_cl_{label}"] - row[f"acc_ad_{label}"]
        )

    config = {
        **cfg,
        "split": "70/10/20",
        "test_used_for_selection": False,
        "attack_parameterization": "raw_magnitude_displacement",
        "attack_mode": "ce",
        "seq_len": SEQ_LEN,
        "latent_dim": LATENT,
        "lambda_adv_ce": LAMBDA_ADV_CE,
        "lambda_trades": LAMBDA_TRADES,
        "lambda_supcon": LAMBDA_SUPCON,
        "beta_vib": BETA_VIB,
        "supcon_tau": SUPCON_TAU,
        "adv_k": ADV_K,
        "adv_steps": ADV_STEPS,
        "adv_step_frac": ADV_STEP_FRAC,
        "lr": LR,
        "batch": BATCH,
        "epochs_run": len(history.history["loss"]),
        "wall_time_s": wall,
        "monitor": monitor,
        "best_val": float(max(history.history[monitor])),
        **row,
    }
    (out_dir / "cfg.json").write_text(json.dumps(config, indent=2))
    print(
        f"  test clean={acc_clean:.4f} adv={acc_adv:.4f} "
        f"drop={acc_clean - acc_adv:+.4f} ({wall:.0f}s)"
    )
    return row


def write_l5_eval(row: dict):
    rows = [{
        "model": "L5-raw",
        "scope": "all",
        "acc_clean": row["acc_clean"],
        "acc_adv": row["acc_adv"],
        "drop": row["acc_adv"] - row["acc_clean"],
        "n_valid": row["n_test"],
    }]
    for label in LABELS:
        rows.append({
            "model": "L5-raw",
            "scope": label,
            "acc_clean": row[f"acc_cl_{label}"],
            "acc_adv": row[f"acc_ad_{label}"],
            "drop": row[f"acc_ad_{label}"] - row[f"acc_cl_{label}"],
            "n_valid": np.nan,
        })
    pd.DataFrame(rows).to_csv(METRICS / "robust_raw_eval.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--levels", default="1,2,3,4,5")
    parser.add_argument("--out", default="robust_raw_ablation.csv")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="training seed (multi-seed runs)")
    args = parser.parse_args()
    levels = {int(value) for value in args.levels.split(",")}
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
            results[str(old["model"])] = dict(old)

    for cfg in ABLATIONS:
        key = f"robust_raw_l{cfg['level']}" + (
            "" if args.seed == SEED else f"_s{args.seed}")
        if cfg["level"] not in levels:
            continue
        if key in results and not args.force:
            print(f"{key} already present; skipping")
            continue
        results[key] = train_level(
            cfg,
            train,
            y_train,
            val,
            y_val,
            test,
            y_test,
            sp["X2_te_adv"],
            sd_m,
            args.epochs,
            seed=args.seed,
        )
        ordered = sorted(results.values(), key=lambda row: str(row["model"]))
        pd.DataFrame(ordered).to_csv(out_csv, index=False)
        if cfg["level"] == 5 and args.seed == SEED:
            write_l5_eval(results[key])
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
