"""Shared CNN/RNN/RF baseline training and evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.data.raw_sequences import build_array_bundle
from src.metrics.lightcurve_features import features_from_arrays
from src.models.rf import RFModel
from src.models.rnn import build_rnn


def _raw_features(curves: list[tuple[np.ndarray, np.ndarray]]):
    rows = [features_from_arrays(m, e) for m, e in curves]
    keys = sorted(rows[0])
    return np.array(
        [[row[key] for key in keys] for row in rows], dtype=np.float64
    ), keys


def _load_raw_fold(df, load_curve: Callable[[Path], object]):
    curves, valid = [], np.zeros(len(df), bool)
    for i, row in df.iterrows():
        try:
            curve = load_curve(Path(row["lc_file"]))
            if len(curve) < 3:
                raise ValueError("too few observations")
            curves.append((
                curve["magnitude"].to_numpy(float),
                curve["error"].to_numpy(float),
            ))
            valid[i] = True
        except Exception:
            curves.append((np.zeros(3), np.ones(3) * 1e-3))
    return curves, valid


def _per_class(row, labels, y, pred_clean, pred_adv, valid):
    for class_id, label in enumerate(labels):
        mask = valid & (y == class_id)
        row[f"acc_cl_{label}"] = float(
            accuracy_score(y[mask], pred_clean[mask])
        )
        row[f"acc_ad_{label}"] = float(
            accuracy_score(y[mask], pred_adv[mask])
        )


def run_baselines(
    *,
    catalog,
    labels: list[str],
    label_to_int: dict[str, int],
    load_curve: Callable[[Path], object],
    adv_root: Path,
    cnn_dir: Path,
    rnn_dir: Path,
    rf_dir: Path,
    metrics_dir: Path,
    seed: int,
    rnn_epochs: int = 12,
) -> None:
    sp = np.load(adv_root / "test_split.npz", allow_pickle=True)
    stats = (
        float(sp["mu_t"]), float(sp["sd_t"]),
        float(sp["mu_m"]), float(sp["sd_m"]),
    )
    idx_train = sp["idx_train"].astype(np.int64)
    idx_val = sp["idx_val"].astype(np.int64)
    idx_test = sp["idx_test"].astype(np.int64)
    y_all = catalog["label4"].map(label_to_int).to_numpy(np.int64)
    train_df = catalog.iloc[idx_train].reset_index(drop=True)
    test_df = catalog.iloc[idx_test].reset_index(drop=True)
    y_train = y_all[idx_train]
    y_val = sp["y_val"].astype(np.int64)
    y_test = sp["y_te"].astype(np.int64)
    train = build_array_bundle(train_df, load_curve, stats)
    train_valid = train["valid"]
    val_valid = sp["valid_val"].astype(bool)
    test_valid = sp["valid_te"].astype(bool)
    x1_val = sp["X1_val"][val_valid, :, 0]
    x2_val = sp["X2_val"][val_valid, :, 0]

    rnn_dir.mkdir(parents=True, exist_ok=True)
    rf_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(seed)
    rnn = build_rnn(
        train["X1"].shape[1],
        len(labels),
        {"hidden": 64, "cell": "gru"},
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4,
            restore_best_weights=True, mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", patience=2,
            factor=0.5, min_lr=1e-6, mode="max",
        ),
    ]
    rnn.fit(
        [
            train["X1"][train_valid, :, 0],
            train["X2"][train_valid, :, 0],
        ],
        tf.keras.utils.to_categorical(
            y_train[train_valid], len(labels)
        ),
        validation_data=(
            [x1_val, x2_val],
            tf.keras.utils.to_categorical(y_val[val_valid], len(labels)),
        ),
        epochs=rnn_epochs,
        batch_size=256,
        callbacks=callbacks,
        verbose=2,
    )
    rnn.save(rnn_dir / "model.keras")

    print("Building raw features for RF")
    raw_train, raw_train_valid = _load_raw_fold(train_df, load_curve)
    features_train, keys = _raw_features(raw_train)
    rf_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        random_state=seed,
        n_jobs=-1,
    )
    rf_keep = train_valid & raw_train_valid
    rf_clf.fit(features_train[rf_keep], y_train[rf_keep])
    rf = RFModel(rf_clf, keys, len(labels))
    rf.save(rf_dir / "model.joblib")

    x1_test = sp["X1_te"][..., 0].astype(np.float32)
    x2_test = sp["X2_te"][..., 0].astype(np.float32)
    x2_adv = sp["X2_te_adv"][..., 0].astype(np.float32)
    cnn = tf.keras.models.load_model(
        cnn_dir / "cnn_two_tower.keras", compile=False
    )
    cnn_clean = cnn.predict(
        [x1_test[..., None], x2_test[..., None]],
        batch_size=512, verbose=0,
    ).argmax(1)
    cnn_adv = cnn.predict(
        [x1_test[..., None], x2_adv[..., None]],
        batch_size=512, verbose=0,
    ).argmax(1)
    rnn_clean = rnn.predict(
        [x1_test, x2_test], batch_size=512, verbose=0
    ).argmax(1)
    rnn_adv = rnn.predict(
        [x1_test, x2_adv], batch_size=512, verbose=0
    ).argmax(1)

    raw_test, raw_test_valid = _load_raw_fold(test_df, load_curve)
    raw_adv = []
    for i, (clean_m, clean_e) in enumerate(raw_test):
        path = adv_root / "adversarial_lcs" / (
            f"{str(sp['ids_te'][i]).strip()}_adv.dat"
        )
        try:
            curve = pd.read_csv(path, sep=r"\s+")
            raw_adv.append((
                curve["magnitude"].to_numpy(float),
                clean_e,
            ))
        except Exception:
            raw_adv.append((clean_m, clean_e))
    f_test, _ = _raw_features(raw_test)
    f_adv, _ = _raw_features(raw_adv)
    rf_clean = rf_clf.predict(f_test)
    rf_adv = rf_clf.predict(f_adv)

    rows = []
    for name, pred_clean, pred_adv, valid in [
        ("CNN", cnn_clean, cnn_adv, test_valid),
        ("RNN", rnn_clean, rnn_adv, test_valid),
        ("RF", rf_clean, rf_adv, test_valid & raw_test_valid),
    ]:
        row = {
            "model": name,
            "acc_clean": float(
                accuracy_score(y_test[valid], pred_clean[valid])
            ),
            "acc_adv": float(
                accuracy_score(y_test[valid], pred_adv[valid])
            ),
            "n_test": int(valid.sum()),
        }
        row["drop"] = row["acc_clean"] - row["acc_adv"]
        _per_class(
            row, labels, y_test, pred_clean, pred_adv, valid
        )
        rows.append(row)
        print(
            f"{name}: clean={row['acc_clean']:.4f} "
            f"adv={row['acc_adv']:.4f}"
        )
    pd.DataFrame(rows).to_csv(
        metrics_dir / "baselines_clean_vs_adv.csv", index=False
    )
