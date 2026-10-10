"""Regression tests for custom-loop epoch metric aggregation."""
from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from src.models.robust_raw_encoder import RobustRawEncoder


def test_metric_mean_is_weighted_by_batch_size():
    model = RobustRawEncoder(
        seq_len=8,
        latent_dim=4,
        n_classes=2,
        use_adv_ce=False,
        use_trades=False,
        use_supcon=False,
        use_vib=False,
    )
    values_a = tuple(tf.constant(value, tf.float32) for value in
                     (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0))
    values_b = tuple(tf.constant(value, tf.float32) for value in
                     (3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0))
    model._update_metrics(values_a, tf.constant(4))
    result = model._update_metrics(values_b, tf.constant(1))
    assert np.isclose(float(result["loss"]), 1.4)
    assert np.isclose(float(result["acc_cl"]), 0.4)


def test_fit_reports_full_validation_accuracy_not_last_batch():
    tf.keras.utils.set_random_seed(7)
    model = RobustRawEncoder(
        seq_len=8,
        latent_dim=4,
        n_classes=2,
        use_adv_ce=False,
        use_trades=False,
        use_supcon=False,
        use_vib=False,
    )
    model.compile(optimizer=tf.keras.optimizers.SGD(0.0))
    x_t = np.zeros((5, 8, 1), np.float32)
    x_m = np.zeros((5, 8, 2), np.float32)
    eps = np.zeros((5, 8), np.float32)
    probabilities, _ = model.predict_proba(x_t, x_m, training=False)
    predicted = probabilities.numpy().argmax(1)
    labels = predicted.copy()
    labels[:3] = 1 - labels[:3]
    expected = float(np.mean(predicted == labels))
    dataset = tf.data.Dataset.from_tensor_slices(
        (x_t, x_m, eps, labels)
    ).batch(4)
    history = model.fit(
        dataset, validation_data=dataset, epochs=1, verbose=0
    )
    assert np.isclose(history.history["val_acc_cl"][0], expected)
