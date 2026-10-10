"""Two-tower CNN classifier on (Δt, Δm).

Mirrors the surrogate from Aguirre et al. 2019 used in the paper but with
configurable depth/width.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


def build_two_tower_cnn(seq_len: int, n_classes: int, cfg: Dict[str, Any]) -> Model:
    """Two-tower CNN with shared weights between the Δt and Δm branches."""
    f1 = int(cfg.get("filters1", 32))
    f2 = int(cfg.get("filters2", 16))
    k1 = int(cfg.get("kernel1", 15))
    k2 = int(cfg.get("kernel2", 11))
    s = int(cfg.get("stride", 2))
    drop = float(cfg.get("dropout", 0.5))
    h = int(cfg.get("hidden", 64))
    activation = cfg.get("activation", "tanh")

    conv1 = layers.Conv1D(f1, k1, strides=s, activation="relu", padding="same")
    conv2 = layers.Conv1D(f2, k2, strides=s, activation="relu", padding="same")

    inp_t = layers.Input((seq_len,), name="dt")
    inp_m = layers.Input((seq_len,), name="dm")
    x1 = conv2(conv1(layers.Reshape((seq_len, 1))(inp_t)))
    x2 = conv2(conv1(layers.Reshape((seq_len, 1))(inp_m)))
    x = layers.Concatenate()([x1, x2])
    x = layers.Flatten()(x)
    x = layers.Dropout(drop)(x)
    x = layers.Dense(h, activation=activation)(x)
    x = layers.Dropout(drop)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = Model([inp_t, inp_m], out, name="two_tower_cnn")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_cnn(
    dt_tr: np.ndarray, dm_tr: np.ndarray, y_tr: np.ndarray,
    dt_val: np.ndarray, dm_val: np.ndarray, y_val: np.ndarray,
    n_classes: int, cfg: Dict[str, Any], seed: int = 42,
):
    tf.keras.utils.set_random_seed(seed)
    seq_len = dt_tr.shape[1]
    model = build_two_tower_cnn(seq_len, n_classes, cfg)
    y_tr_oh = tf.keras.utils.to_categorical(y_tr, n_classes)
    y_val_oh = tf.keras.utils.to_categorical(y_val, n_classes) if len(y_val) else None
    val_data = ([dt_val, dm_val], y_val_oh) if y_val_oh is not None else None
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss" if val_data else "loss",
                                                   patience=int(cfg.get("patience", 15)),
                                                   restore_best_weights=True)]
    model.fit([dt_tr, dm_tr], y_tr_oh,
              validation_data=val_data,
              epochs=int(cfg.get("epochs", 5)),
              batch_size=int(cfg.get("batch_size", 128)),
              callbacks=callbacks, verbose=0)
    return model
