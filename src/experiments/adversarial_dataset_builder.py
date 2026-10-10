"""Shared train/validation/test and transfer-attack dataset builder."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, layers

from src.attacks.photometric_pgd import encode_magnitude_budget, pgd_magnitude_tf
from src.data.raw_sequences import (
    SEQ_LEN,
    build_array_bundle,
    compute_standardization_stats,
)
from src.data.splits import canonical_object_split


K_MAG = 0.5
PGD_STEPS = 20
PGD_STEP_FRAC = 0.2
PGD_RESTARTS = 3
SMOOTH_LAMBDA = 1e-3
BATCH_SIZE = 256
CNN_EPOCHS = 12
CNN_BATCH = 256


def build_cnn(seq_len: int, n_classes: int) -> Model:
    inp_t = layers.Input((seq_len, 1), name="dt")
    inp_m = layers.Input((seq_len, 1), name="dm")
    conv1 = layers.Conv1D(32, 15, strides=2, activation="relu", padding="same")
    conv2 = layers.Conv1D(16, 11, strides=2, activation="relu", padding="same")
    x1 = conv2(conv1(inp_t))
    x2 = conv2(conv1(inp_m))
    x = layers.Concatenate()([x1, x2])
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="tanh")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = Model([inp_t, inp_m], out, name="two_tower_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _attack_test(
    cnn: Model,
    test: dict,
    y_test: np.ndarray,
    sd_m: float,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    eps_all = encode_magnitude_budget(
        test["E"].squeeze(-1), test["n_sel"], sd_m, K_MAG
    )
    y_oh = tf.keras.utils.to_categorical(y_test, n_classes).astype(np.float32)
    x2_adv = test["X2"].copy()
    delta_best = np.zeros_like(test["X2"][..., 0])

    def forward(dt, dm):
        return cnn([dt[..., None], dm[..., None]], training=False)

    for start in range(0, len(y_test), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(y_test))
        dt = tf.constant(test["X1"][start:end, :, 0], tf.float32)
        dm = tf.constant(test["X2"][start:end, :, 0], tf.float32)
        labels = tf.constant(y_oh[start:end], tf.float32)
        eps = tf.constant(eps_all[start:end], tf.float32)
        best_loss = np.full(end - start, -np.inf, np.float32)
        best_dm = test["X2"][start:end, :, 0].copy()
        best_delta = np.zeros((end - start, SEQ_LEN), np.float32)

        for restart in range(PGD_RESTARTS):
            tf.random.set_seed(42 + restart + start)
            dm_adv, delta = pgd_magnitude_tf(
                forward,
                dt,
                dm,
                labels,
                eps,
                tf.constant(PGD_STEPS, tf.int32),
                tf.constant(PGD_STEP_FRAC, tf.float32),
                tf.constant(True),
                tf.constant(SMOOTH_LAMBDA, tf.float32),
            )
            pred = forward(dt, dm_adv)
            loss = tf.keras.losses.categorical_crossentropy(
                labels, pred
            ).numpy()
            take = loss > best_loss
            best_loss[take] = loss[take]
            best_dm[take] = dm_adv.numpy()[take]
            best_delta[take] = delta.numpy()[take]

        x2_adv[start:end, :, 0] = best_dm
        delta_best[start:end] = best_delta
        if (start // BATCH_SIZE) % 10 == 0:
            print(f"      attacked {end}/{len(y_test)}")
    return x2_adv, delta_best


def _save_adversarial_curves(
    full_curves: list,
    ids: np.ndarray,
    delta_std: np.ndarray,
    sd_m: float,
    adv_folder: Path,
) -> int:
    count = 0
    for i, curve in enumerate(full_curves):
        if curve is None:
            continue
        n_sel = curve["n_sel"]
        delta_sel = delta_std[i, :n_sel] * float(sd_m)
        t_sel = curve["t_sel"]
        t_full = curve["t_full"]
        if n_sel >= 2:
            delta_full = np.interp(
                t_full, t_sel, delta_sel,
                left=delta_sel[0], right=delta_sel[-1],
            )
        else:
            delta_full = np.zeros_like(t_full)
        m_full = curve["m_full"]
        e_full = curve["e_full"]
        m_adv = np.clip(
            m_full + delta_full,
            m_full - K_MAG * e_full,
            m_full + K_MAG * e_full,
        )
        pd.DataFrame({
            "time": t_full,
            "magnitude": m_adv,
            "error": e_full,
        }).to_csv(
            adv_folder / f"{str(ids[i]).strip()}_adv.dat",
            sep=" ",
            index=False,
        )
        count += 1
    return count


def _npz_fold(prefix: str, bundle: dict, y: np.ndarray, ids: np.ndarray) -> dict:
    return {
        f"X1_{prefix}": bundle["X1"],
        f"X2_{prefix}": bundle["X2"],
        f"DE_{prefix}": bundle["DE"],
        f"E_{prefix}": bundle["E"],
        f"n_sel_{prefix}": bundle["n_sel"],
        f"valid_{prefix}": bundle["valid"],
        f"y_{prefix}": np.asarray(y, dtype=np.int64),
        f"ids_{prefix}": np.asarray(ids, dtype=str),
    }


def run_builder(
    *,
    survey: str,
    df,
    labels: list[str],
    label_to_int: dict[str, int],
    load_curve: Callable[[Path], object],
    cnn_dir: Path,
    out_root: Path,
    seed: int,
) -> None:
    """Train the transfer CNN and build a canonical bounded test attack."""
    cnn_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    adv_folder = out_root / "adversarial_lcs"
    adv_folder.mkdir(parents=True, exist_ok=True)
    for stale in adv_folder.glob("*_adv.dat"):
        stale.unlink()

    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    y_all = df["label4"].map(label_to_int).to_numpy(np.int64)
    ids_all = df["ID"].astype(str).to_numpy()
    idx_train, idx_val, idx_test = canonical_object_split(y_all, seed)
    folds = {
        "train": (idx_train, df.iloc[idx_train].reset_index(drop=True)),
        "val": (idx_val, df.iloc[idx_val].reset_index(drop=True)),
        "test": (idx_test, df.iloc[idx_test].reset_index(drop=True)),
    }
    print(
        f"[1/7] {survey}: split train={len(idx_train)} "
        f"val={len(idx_val)} test={len(idx_test)}"
    )

    print("[2/7] Computing preprocessing statistics from TRAIN only")
    stats = compute_standardization_stats(
        folds["train"][1], load_curve, seed=seed, seq_len=SEQ_LEN
    )
    mu_t, sd_t, mu_m, sd_m = stats
    np.savez(
        cnn_dir / "preprocess_stats.npz",
        mu_t=mu_t, sd_t=sd_t, mu_m=mu_m, sd_m=sd_m,
        split_seed=seed,
    )
    print(
        f"      mu_t={mu_t:.5f} sd_t={sd_t:.5f} "
        f"mu_m={mu_m:.5f} sd_m={sd_m:.5f}"
    )

    print("[3/7] Building TRAIN tensors")
    train = build_array_bundle(folds["train"][1], load_curve, stats)
    print("[4/7] Building VALIDATION tensors")
    val = build_array_bundle(folds["val"][1], load_curve, stats)
    train_keep = train["valid"]
    val_keep = val["valid"]
    y_train = y_all[idx_train]
    y_val = y_all[idx_val]

    print("[5/7] Training transfer CNN with untouched validation fold")
    cnn = build_cnn(SEQ_LEN, len(labels))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4,
            restore_best_weights=True, mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", patience=2, factor=0.5,
            min_lr=1e-6, mode="max",
        ),
    ]
    t0 = time.time()
    history = cnn.fit(
        [train["X1"][train_keep], train["X2"][train_keep]],
        y_train[train_keep],
        validation_data=(
            [val["X1"][val_keep], val["X2"][val_keep]],
            y_val[val_keep],
        ),
        epochs=CNN_EPOCHS,
        batch_size=CNN_BATCH,
        callbacks=callbacks,
        verbose=2,
    )
    cnn.save(cnn_dir / "cnn_two_tower.keras")

    print("[6/7] Building untouched TEST tensors")
    test = build_array_bundle(
        folds["test"][1], load_curve, stats, keep_full=True
    )
    y_test = y_all[idx_test]
    ids_test = ids_all[idx_test]

    print(
        f"[7/7] Transfer PGD in magnitude space "
        f"(steps={PGD_STEPS}, restarts={PGD_RESTARTS}, k={K_MAG})"
    )
    x2_adv, delta_std = _attack_test(
        cnn, test, y_test, sd_m, len(labels)
    )
    p_clean = cnn.predict(
        [test["X1"], test["X2"]], batch_size=512, verbose=0
    )
    p_adv = cnn.predict(
        [test["X1"], x2_adv], batch_size=512, verbose=0
    )
    valid = test["valid"]
    pred_clean = p_clean.argmax(1)
    pred_adv = p_adv.argmax(1)
    acc_clean = float(np.mean(pred_clean[valid] == y_test[valid]))
    acc_adv = float(np.mean(pred_adv[valid] == y_test[valid]))
    print(f"      CNN clean={acc_clean:.4f} adversarial={acc_adv:.4f}")

    saved = _save_adversarial_curves(
        test["full"], ids_test, delta_std, sd_m, adv_folder
    )
    print(f"      saved {saved} bounded adversarial curves")

    summary = pd.DataFrame({
        "id": ids_test,
        "true_label": [labels[i] for i in y_test],
        "pred_clean": [labels[i] for i in pred_clean],
        "pred_adv": [labels[i] for i in pred_adv],
        "valid": valid,
    })
    summary.to_csv(out_root / "test_attack_summary.csv", index=False)

    payload = {}
    payload.update(_npz_fold("val", val, y_val, ids_all[idx_val]))
    payload.update(_npz_fold("te", test, y_test, ids_test))
    payload["X2_te_adv"] = x2_adv
    payload["delta_mag_std_te"] = delta_std
    payload.update({
        "idx_train": idx_train,
        "idx_val": idx_val,
        "idx_test": idx_test,
        "y_train": y_all[idx_train],
        "ids_train": ids_all[idx_train],
        "mu_t": mu_t,
        "sd_t": sd_t,
        "mu_m": mu_m,
        "sd_m": sd_m,
        "seed": seed,
        "k_mag": K_MAG,
        "seq_len": SEQ_LEN,
        "p_cl": p_clean,
        "p_ad": p_adv,
        # Flat aliases read by the evaluation scripts.
        "de_te": test["DE"],
        "valid": test["valid"],
        "n_sel": test["n_sel"],
    })
    np.savez_compressed(out_root / "test_split.npz", **payload)
    with open(out_root / "summary.json", "w") as handle:
        json.dump({
            "survey": survey,
            "n_train": int(train_keep.sum()),
            "n_val": int(val_keep.sum()),
            "n_test": int(valid.sum()),
            "cnn_acc_clean": acc_clean,
            "cnn_acc_adv": acc_adv,
            "k_mag": K_MAG,
            "pgd_steps": PGD_STEPS,
            "pgd_restarts": PGD_RESTARTS,
            "pgd_step_frac": PGD_STEP_FRAC,
            "smooth_lambda": SMOOTH_LAMBDA,
            "epochs_run": len(history.history["loss"]),
            "best_val_accuracy": float(max(history.history["val_accuracy"])),
            "wall_time_s": time.time() - t0,
            "split": "70/10/20",
        }, handle, indent=2)
