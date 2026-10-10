"""Train the L1/L2 raw-encoder ablation variants on Gaia DR2 or WISE.

Extends the OGLE-only ablation of ``train_robust_raw_ablation`` to the other
two surveys, so the cross-survey ablation table can compare L1 (clean CE),
L2 (Madry AdvCE) and the full L5 (already trained by
``train_robust_raw_survey``) under identical protocols.

Usage:
    python -m src.experiments.train_robust_raw_survey_ablation --survey gaia --levels 1,2
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

from src.data.raw_sequences import build_array_bundle, make_robust_inputs
from src.data.survey_config import LABELS_COMMON, LABEL_TO_INT, SEED
from src.data.survey_loader import build_catalog, load_lightcurve, subsample_balanced
from src.experiments.train_robust_raw_ablation import (
    ABLATIONS,
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


def train_level(cfg, survey, data, epochs, suffix=""):
    """Train one ablation level on a survey; mirrors train_robust_raw_survey."""
    tf.keras.backend.clear_session()
    out_dir = REPO / "runs" / f"{survey}_robust_raw_l{cfg['level']}{suffix}"
    print(f"\n=== {survey} L{cfg['level']}{suffix} -> {out_dir.name} ===",
          flush=True)

    tf.keras.utils.set_random_seed(SEED)
    model = RobustRawEncoder(
        seq_len=SEQ_LEN,
        latent_dim=LATENT,
        n_classes=len(LABELS_COMMON),
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
            data["x_t_train"], data["x_m_train"],
            data["eps_train"], data["y_train"],
        ))
        .shuffle(8192, seed=SEED)
        .batch(BATCH)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((
            data["x_t_val"], data["x_m_val"],
            data["eps_val"], data["y_val"],
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

    pred_clean = predict_labels(model, data["x_t_test"], data["x_m_test"])
    pred_adv = predict_labels(model, data["x_t_test"], data["x_m_test_adv"])
    valid = data["test_valid"]
    y_test = data["y_test"]
    row = {
        "model": f"robust_raw_l{cfg['level']}{suffix}",
        "seed": SEED,
        "acc_clean": float(accuracy_score(y_test[valid], pred_clean[valid])),
        "acc_adv": float(accuracy_score(y_test[valid], pred_adv[valid])),
        "n_test": int(valid.sum()),
    }
    row["drop"] = row["acc_clean"] - row["acc_adv"]
    from sklearn.metrics import f1_score
    for class_id, label in enumerate(LABELS_COMMON):
        mask = valid & (y_test == class_id)
        row[f"acc_clean_{label}"] = float(
            accuracy_score(y_test[mask], pred_clean[mask])
        )
        row[f"acc_adv_{label}"] = float(
            accuracy_score(y_test[mask], pred_adv[mask])
        )
        row[f"f1_clean_{label}"] = float(f1_score(
            y_test[valid] == class_id, pred_clean[valid] == class_id
        ))
        row[f"f1_adv_{label}"] = float(f1_score(
            y_test[valid] == class_id, pred_adv[valid] == class_id
        ))
    config = {
        "tag": f"{survey}_robust_raw_l{cfg['level']}{suffix}",
        "survey": survey,
        "level": cfg["level"],
        "use_adv_ce": cfg["use_adv_ce"],
        "use_trades": cfg["use_trades"],
        "use_supcon": cfg["use_supcon"],
        "use_vib": cfg["use_vib"],
        "attack_parameterization": "raw_magnitude_displacement",
        "adv_k": ADV_K,
        "adv_steps": ADV_STEPS,
        "adv_step_frac": ADV_STEP_FRAC,
        "lr": LR,
        "batch": BATCH,
        "epochs_run": len(history.history["loss"]),
        "best_val_metric": float(max(history.history[monitor])),
        "monitor": monitor,
        "wall_time_s": wall,
        **{k: row[k] for k in ("acc_clean", "acc_adv", "drop", "n_test")},
    }
    (out_dir / "cfg.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2), flush=True)
    return row


def load_survey_data(survey):
    adv_root = REPO / f"runs/{survey}_adversarial_dataset"
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

    x_t_train, x_m_train, eps_train = make_robust_inputs(train, sd_m, ADV_K)
    x_t_val, x_m_val, eps_val = make_robust_inputs(val, sd_m, ADV_K)
    x_t_test, x_m_test, _ = make_robust_inputs(test, sd_m, ADV_K)
    x_m_test_adv = make_adversarial_tower(test, sp["X2_te_adv"], sd_m)
    tv, vv = train["valid"], val["valid"]
    return {
        "x_t_train": x_t_train[tv], "x_m_train": x_m_train[tv],
        "eps_train": eps_train[tv], "y_train": y_train[tv],
        "x_t_val": x_t_val[vv], "x_m_val": x_m_val[vv],
        "eps_val": eps_val[vv], "y_val": y_val[vv],
        "x_t_test": x_t_test, "x_m_test": x_m_test,
        "x_m_test_adv": x_m_test_adv,
        "test_valid": test["valid"], "y_test": y_test,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--survey", required=True, choices=["gaia", "wise"])
    parser.add_argument("--levels", default="1,2")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--oversample", action="store_true",
                        help="balance training classes by resampling "
                             "minority classes with replacement (WISE "
                             "imbalance diagnostic)")
    args = parser.parse_args()
    levels = {int(x) for x in args.levels.split(",")}
    configs = [c for c in ABLATIONS if c["level"] in levels]
    if not configs:
        raise SystemExit(f"No ablation levels matched {sorted(levels)}")

    data = load_survey_data(args.survey)
    suffix = ""
    if args.oversample:
        suffix = "_oversampled"
        rng = np.random.default_rng(SEED)
        y = data["y_train"]
        counts = np.bincount(y)
        target = counts.max()
        take = []
        for c in range(len(counts)):
            pool = np.where(y == c)[0]
            take.append(pool)
            if len(pool) < target:
                take.append(rng.choice(pool, target - len(pool),
                                       replace=True))
        idx = rng.permutation(np.concatenate(take))
        for key in ("x_t_train", "x_m_train", "eps_train", "y_train"):
            data[key] = data[key][idx]
        print(f"oversampled train: {counts.tolist()} -> "
              f"{np.bincount(data['y_train']).tolist()}", flush=True)
    metrics_dir = REPO / f"outputs/metrics/{args.survey}"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_csv = metrics_dir / "robust_raw_ablation.csv"

    rows = [train_level(cfg, args.survey, data, args.epochs, suffix=suffix)
            for cfg in configs]
    frame = pd.DataFrame(rows)
    if out_csv.exists():
        prior = pd.read_csv(out_csv)
        prior = prior[~prior["model"].isin(frame["model"])]
        frame = pd.concat([prior, frame], ignore_index=True)
    frame.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
