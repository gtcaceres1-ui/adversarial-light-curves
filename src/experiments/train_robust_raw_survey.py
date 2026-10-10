"""Train the full L5 raw encoder on Gaia DR2 or WISE."""
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
from src.data.survey_config import LABELS_COMMON, LABEL_TO_INT, SEED
from src.data.survey_loader import build_catalog, load_lightcurve, subsample_balanced
from src.experiments.train_robust_raw_ablation import (
    ADV_K,
    ADV_STEP_FRAC,
    ADV_STEPS,
    BATCH,
    BETA_VIB,
    LATENT,
    LAMBDA_ADV_CE,
    LAMBDA_SUPCON,
    LAMBDA_TRADES,
    LR,
    SUPCON_TAU,
    SaveBestSubmodels,
    bundle_from_split,
    make_adversarial_tower,
    predict_labels,
)
from src.models.robust_raw_encoder import RobustRawEncoder

REPO = Path(__file__).resolve().parents[2]
DEFAULT_EPOCHS = 25
SEQ_LEN = 500


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--survey", required=True, choices=["gaia", "wise"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    args = parser.parse_args()
    survey = args.survey
    adv_root = REPO / f"runs/{survey}_adversarial_dataset"
    out_dir = REPO / f"runs/{survey}_robust_raw_l5"
    metrics_dir = REPO / f"outputs/metrics/{survey}"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    sp = np.load(adv_root / "test_split.npz", allow_pickle=True)
    stats = (
        float(sp["mu_t"]), float(sp["sd_t"]),
        float(sp["mu_m"]), float(sp["sd_m"]),
    )
    sd_m = stats[3]
    catalog = subsample_balanced(build_catalog(survey))
    y_all = catalog["label4"].map(LABEL_TO_INT).to_numpy(np.int64)
    idx_train = sp["idx_train"].astype(np.int64)
    train_df = catalog.iloc[idx_train].reset_index(drop=True)
    train = build_array_bundle(
        train_df,
        lambda path: load_lightcurve(Path(path), survey),
        stats,
    )
    y_train = y_all[idx_train]
    val = bundle_from_split(sp, "val")
    test = bundle_from_split(sp, "te")
    y_val = sp["y_val"].astype(np.int64)
    y_test = sp["y_te"].astype(np.int64)

    x_t_train, x_m_train, eps_train = make_robust_inputs(
        train, sd_m, ADV_K
    )
    x_t_val, x_m_val, eps_val = make_robust_inputs(val, sd_m, ADV_K)
    x_t_test, x_m_test, _ = make_robust_inputs(test, sd_m, ADV_K)
    x_m_test_adv = make_adversarial_tower(
        test, sp["X2_te_adv"], sd_m
    )
    train_valid = train["valid"]
    val_valid = val["valid"]

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    model = RobustRawEncoder(
        seq_len=SEQ_LEN,
        latent_dim=LATENT,
        n_classes=len(LABELS_COMMON),
        use_adv_ce=True,
        use_trades=True,
        use_supcon=True,
        use_vib=True,
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
        SaveBestSubmodels(model, out_dir, "val_acc_ad"),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_acc_ad", patience=8,
            restore_best_weights=True, mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_acc_ad", patience=4, factor=0.5,
            min_lr=1e-6, mode="max", verbose=1,
        ),
    ]
    start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )
    wall = time.time() - start
    model.encoder_backbone.save(out_dir / "encoder_backbone.keras")
    model.classifier.save(out_dir / "classifier.keras")
    model.proj.save(out_dir / "projection.keras")

    pred_clean = predict_labels(model, x_t_test, x_m_test)
    pred_adv = predict_labels(model, x_t_test, x_m_test_adv)
    valid = test["valid"]
    rows = []
    for class_id, scope in [(-1, "all")] + list(enumerate(LABELS_COMMON)):
        mask = valid if class_id == -1 else valid & (y_test == class_id)
        acc_clean = float(accuracy_score(y_test[mask], pred_clean[mask]))
        acc_adv = float(accuracy_score(y_test[mask], pred_adv[mask]))
        rows.append({
            "model": "L5-raw",
            "scope": scope,
            "acc_clean": acc_clean,
            "acc_adv": acc_adv,
            "drop": acc_adv - acc_clean,
            "n_valid": int(mask.sum()),
        })
    pd.DataFrame(rows).to_csv(
        metrics_dir / "robust_raw_eval.csv", index=False
    )
    overall = rows[0]
    config = {
        "tag": f"{survey}_robust_raw_l5",
        "survey": survey,
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
        "best_val_acc_ad": float(max(history.history["val_acc_ad"])),
        "wall_time_s": wall,
        **overall,
    }
    (out_dir / "cfg.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
