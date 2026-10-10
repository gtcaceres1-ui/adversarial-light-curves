"""GRU/LSTM classifier on (Δt, Δm).

Same input representation as the CNN so adversarial transferability is well
defined: both models consume scaled Δt and Δm with identical preprocessing.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


def build_rnn(seq_len: int, n_classes: int, cfg: Dict[str, Any]) -> Model:
    h = int(cfg.get("hidden", 32))
    cell = cfg.get("cell", "gru").lower()
    inp_t = layers.Input((seq_len,), name="dt")
    inp_m = layers.Input((seq_len,), name="dm")
    x = layers.Concatenate(axis=-1)([
        layers.Reshape((seq_len, 1))(inp_t),
        layers.Reshape((seq_len, 1))(inp_m),
    ])  # (B, L, 2)
    rnn_layer = (layers.GRU if cell == "gru" else layers.LSTM)(h)
    x = rnn_layer(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = Model([inp_t, inp_m], out, name=f"rnn_{cell}")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_rnn(
    dt_tr: np.ndarray, dm_tr: np.ndarray, y_tr: np.ndarray,
    dt_val: np.ndarray, dm_val: np.ndarray, y_val: np.ndarray,
    n_classes: int, cfg: Dict[str, Any], seed: int = 42,
):
    tf.keras.utils.set_random_seed(seed)
    seq_len = dt_tr.shape[1]
    model = build_rnn(seq_len, n_classes, cfg)
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
