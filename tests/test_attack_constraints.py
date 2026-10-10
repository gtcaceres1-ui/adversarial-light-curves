"""Tests for the photometric PGD constraints.

These tests do not require a trained model: a tiny 1-layer Keras model serves
as the surrogate so we can assert geometric properties of the attack
independently of any classifier behavior.
"""
from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from src.attacks.photometric_pgd import (
    AttackConfig,
    attack_batch,
    encode_magnitude_budget,
    integrate_dm,
    project_raw_magnitude,
    propagated_sigma_dm,
    select_perturbation_mask,
)


def _make_dummy_model(seq_len: int, n_classes: int):
    inp_t = tf.keras.Input((seq_len,))
    inp_m = tf.keras.Input((seq_len,))
    x = tf.keras.layers.Concatenate()([inp_t, inp_m])
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model([inp_t, inp_m], out)


def _toy_batch(B=4, L=20, seed=0):
    rng = np.random.default_rng(seed)
    m_raw = rng.normal(18.0, 0.2, (B, L)).astype(np.float32)
    e_raw = rng.uniform(0.01, 0.05, (B, L)).astype(np.float32)
    n_used = np.full(B, L, dtype=np.int32)
    dm = np.zeros_like(m_raw)
    dm[:, 1:] = np.diff(m_raw, axis=1)
    dt = np.ones_like(m_raw)
    dt[:, 0] = 0.0
    mu_m, sd_m = float(dm.mean()), float(dm.std() + 1e-8)
    dm_scaled = ((dm - mu_m) / sd_m).astype(np.float32)
    dt_scaled = dt.astype(np.float32)
    y_oh = np.eye(3, dtype=np.float32)[rng.integers(0, 3, B)]
    return dt_scaled, dm_scaled, y_oh, m_raw, e_raw, n_used, mu_m, sd_m


def test_propagated_sigma_dm_zero_at_index_zero():
    e = np.array([0.05, 0.04, 0.03])
    s = propagated_sigma_dm(e)
    assert s[0] == 0.0
    assert np.isclose(s[1], np.sqrt(0.04 ** 2 + 0.05 ** 2))
    assert np.isclose(s[2], np.sqrt(0.03 ** 2 + 0.04 ** 2))


def test_integrate_dm_inverse_of_diff():
    m = np.array([18.0, 18.1, 18.05, 17.9])
    dm = np.concatenate(([0.0], np.diff(m)))
    rec = integrate_dm(dm, m[0])
    assert np.allclose(rec, m)


def test_project_raw_magnitude_clips_only_masked_indices():
    m_clean = np.array([18.0, 18.0, 18.0, 18.0])
    m_adv = np.array([18.0, 18.5, 18.0, 19.0])  # 18.5 and 19.0 are out of range
    e = np.array([0.01, 0.1, 0.01, 0.1])
    mask = np.array([False, True, True, True])
    out = project_raw_magnitude(m_adv, m_clean, e, k=0.5, valid_mask=mask)
    assert out[0] == 18.0  # never perturbed
    assert out[1] == 18.05  # clipped to m + 0.5*0.1
    assert out[2] == 18.0  # already inside
    assert out[3] == 18.05


def test_perturbation_fraction_respected():
    rng = np.random.default_rng(0)
    mask = select_perturbation_mask(n_used=100, seq_len=100, fraction=0.4, rng=rng)
    # Index 0 is never perturbed; expect ~40 perturbed points among indices 1..99
    assert mask[0] == False
    assert abs(mask.sum() - 40) <= 1


def test_attack_keeps_dt_unchanged_and_respects_bound():
    L = 16
    dt, dm, y, m_raw, e_raw, n_used, mu_m, sd_m = _toy_batch(B=3, L=L, seed=1)
    model = _make_dummy_model(L, 3)
    cfg = AttackConfig(k=0.5, steps=5, step_frac=0.5, random_start=True, perturbation_fraction=1.0, seed=42)

    dm_adv_s, m_adv, dm_adv_r, mask = attack_batch(
        forward=lambda a, b: model([a, b], training=False),
        x_dt_scaled=dt, x_dm_scaled=dm, y_oh=y,
        m_raw=m_raw, e_raw=e_raw, n_used=n_used,
        sd_m=sd_m, mu_m=mu_m, cfg=cfg,
    )

    # Δt is never updated — the attack only returns dm_adv tensors. Caller keeps dt.
    assert dm_adv_s.shape == dm.shape
    # Hard bound check on every raw magnitude that was perturbed.
    over = np.abs(m_adv - m_raw)[mask]
    bound = (cfg.k * e_raw)[mask]
    assert (over <= bound + 1e-5).all(), "raw magnitude bound violated"

    # Δm_adv must be the diff of m_adv (recomputed after projection).
    for i in range(m_adv.shape[0]):
        nu = int(n_used[i])
        expected = np.concatenate(([0.0], np.diff(m_adv[i, :nu])))
        assert np.allclose(dm_adv_r[i, :nu], expected, atol=1e-5)


def test_attack_handles_zero_fraction():
    L = 10
    dt, dm, y, m_raw, e_raw, n_used, mu_m, sd_m = _toy_batch(B=2, L=L, seed=2)
    model = _make_dummy_model(L, 3)
    cfg = AttackConfig(k=0.5, steps=3, perturbation_fraction=0.0)
    _, m_adv, _, mask = attack_batch(
        forward=lambda a, b: model([a, b], training=False),
        x_dt_scaled=dt, x_dm_scaled=dm, y_oh=y,
        m_raw=m_raw, e_raw=e_raw, n_used=n_used,
        sd_m=sd_m, mu_m=mu_m, cfg=cfg,
    )
    assert mask.sum() == 0
    assert np.allclose(m_adv, m_raw, atol=1e-6)


def test_encoded_budget_marks_padding_and_keeps_anchor_fixed():
    errors = np.array([[0.02, 0.03, 0.04, 0.0, 0.0]], dtype=np.float32)
    encoded = encode_magnitude_budget(
        errors, np.array([3]), sd_m=0.2, k=0.5
    )
    assert np.all(encoded[0, :3] >= 0.0)
    assert np.all(encoded[0, 3:] == -1.0)


def test_evaluated_scaled_dm_matches_bounded_magnitudes():
    dt, dm, y, m_raw, e_raw, n_used, mu_m, sd_m = _toy_batch(
        B=2, L=12, seed=3
    )
    model = _make_dummy_model(12, 3)
    dm_adv_s, m_adv, dm_adv_r, _ = attack_batch(
        forward=lambda a, b: model([a, b], training=False),
        x_dt_scaled=dt,
        x_dm_scaled=dm,
        y_oh=y,
        m_raw=m_raw,
        e_raw=e_raw,
        n_used=n_used,
        sd_m=sd_m,
        mu_m=mu_m,
        cfg=AttackConfig(k=0.5, steps=4, step_frac=0.5),
    )
    expected = ((dm_adv_r - mu_m) / sd_m).astype(np.float32)
    assert np.allclose(dm_adv_s, expected, atol=2e-5)
    assert np.all(np.abs(m_adv - m_raw) <= 0.5 * e_raw + 1e-7)
