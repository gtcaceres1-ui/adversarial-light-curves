"""Component-wise raw-encoder ablations on OGLE.

This complements the incremental L1->L5 ablation.  The incremental ladder
answers "what happens when terms are added in this order?".  This script
answers more direct attribution questions:

* single-component: CE + one auxiliary component;
* leave-one-out: full objective with one component removed.

Existing rows from robust_raw_ablation.csv and vib_only_beta_sweep.csv should
be combined downstream; this script trains only the missing configurations by
default to avoid duplicating expensive runs.
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

from src.data.raw_sequences import make_robust_inputs
from src.experiments.build_adversarials_ogle import LABELS, SEED, SEQ_LEN
from src.experiments.train_robust_raw_ablation import (
    ADV_K,
    ADV_ROOT,
    ADV_STEP_FRAC,
    ADV_STEPS,
    BATCH,
    BETA_VIB,
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
DEFAULT_EPOCHS = 25


CONFIGS = [
    dict(
        name="single_trades",
        suite="single_component",
        description="CE + TRADES only",
        use_adv_ce=False,
        use_trades=True,
        use_supcon=False,
        use_vib=False,
    ),
    dict(
        name="single_supcon",
        suite="single_component",
        description="CE + SupCon only",
        use_adv_ce=False,
        use_trades=False,
        use_supcon=True,
        use_vib=False,
    ),
    dict(
        name="loo_no_adv_ce",
        suite="leave_one_out",
        description="Full objective without adversarial CE",
        use_adv_ce=False,
        use_trades=True,
        use_supcon=True,
        use_vib=True,
    ),
    dict(
        name="loo_no_trades",
        suite="leave_one_out",
        description="Full objective without TRADES",
        use_adv_ce=True,
        use_trades=False,
        use_supcon=True,
        use_vib=True,
    ),
    dict(
        name="loo_no_supcon",
        suite="leave_one_out",
        description="Full objective without SupCon",
        use_adv_ce=True,
        use_trades=True,
        use_supcon=False,
        use_vib=True,
    ),
]


def _monitor_for(cfg: dict) -> str:
    if cfg["use_adv_ce"] or cfg["use_trades"] or cfg["use_supcon"]:
        return "val_acc_ad"
    return "val_acc_cl"


def train_component(
    cfg: dict,
    train: dict,
    y_train: np.ndarray,
    val: dict,
    y_val: np.ndarray,
    test: dict,
    y_test: np.ndarray,
    x2_test_adv: np.ndarray,
    sd_m: float,
    epochs: int,
):
    tf.keras.backend.clear_session()
    out_dir = REPO / "runs" / f"robust_raw_component_{cfg['name']}"
    print(f"\n=== {cfg['name']} -> {out_dir.name} ===")

    x_t_train, x_m_train, eps_train = make_robust_inputs(train, sd_m, ADV_K)
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
    monitor = _monitor_for(cfg)
    callbacks = [
        SaveBestSubmodels(model, out_dir, monitor),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=8,
            restore_best_weights=True,
            mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            patience=4,
            factor=0.5,
            min_lr=1e-6,
            mode="max",
            verbose=1,
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

    valid = test["valid"]
    pred_clean = predict_labels(model, x_t_test, x_m_test)
    pred_adv = predict_labels(model, x_t_test, x_m_test_adv)
    row = {
        "model": cfg["name"],
        "suite": cfg["suite"],
        "description": cfg["description"],
        "use_adv_ce": cfg["use_adv_ce"],
        "use_trades": cfg["use_trades"],
        "use_supcon": cfg["use_supcon"],
        "use_vib": cfg["use_vib"],
        "acc_clean": float(accuracy_score(y_test[valid], pred_clean[valid])),
        "acc_adv": float(accuracy_score(y_test[valid], pred_adv[valid])),
        "n_test": int(valid.sum()),
        "epochs_run": len(history.history["loss"]),
        "wall_s": wall,
        "monitor": monitor,
        "best_val": float(max(history.history[monitor])),
    }
    row["drop"] = row["acc_clean"] - row["acc_adv"]
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
        "attack_parameterization": "raw_magnitude_displacement",
        "attack_mode": "ce",
        "adv_k": ADV_K,
        "adv_steps": ADV_STEPS,
        "adv_step_frac": ADV_STEP_FRAC,
        "lambda_adv_ce": LAMBDA_ADV_CE,
        "lambda_trades": LAMBDA_TRADES,
        "lambda_supcon": LAMBDA_SUPCON,
        "beta_vib": BETA_VIB,
        "supcon_tau": SUPCON_TAU,
    }, indent=2))
    print(
        f"{cfg['name']}: clean={row['acc_clean']:.4f} "
        f"adv={row['acc_adv']:.4f} drop={row['drop']:.4f}"
    )
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--out", default="robust_raw_component_ablation.csv")
    parser.add_argument(
        "--configs",
        default=",".join(cfg["name"] for cfg in CONFIGS),
        help="Comma-separated config names to run.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    selected = set(args.configs.split(","))
    configs = [cfg for cfg in CONFIGS if cfg["name"] in selected]
    unknown = selected - {cfg["name"] for cfg in CONFIGS}
    if unknown:
        raise ValueError(f"Unknown component configs: {sorted(unknown)}")

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

    for cfg in configs:
        if cfg["name"] in results and not args.force:
            print(f"{cfg['name']} already present; skipping")
            continue
        results[cfg["name"]] = train_component(
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
        )
        ordered_names = [cfg["name"] for cfg in CONFIGS]
        ordered = sorted(
            results.values(),
            key=lambda row: ordered_names.index(str(row["model"])),
        )
        pd.DataFrame(ordered).to_csv(out_csv, index=False)

    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
