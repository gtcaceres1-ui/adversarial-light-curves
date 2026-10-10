"""Photometrically bounded PGD in raw-magnitude displacement space.

The classifier consumes standardized first differences, but the threat model
is defined on observations:

    |m_adv[i] - m_clean[i]| <= k * e[i].

We therefore optimize a displacement for each raw magnitude and derive the
adversarial Delta-m sequence by differencing that displacement. This keeps the
constraint true at every PGD iteration and guarantees that the tensor evaluated
by a model is the tensor reconstructed from the bounded magnitudes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import tensorflow as tf


@dataclass
class AttackConfig:
    k: float = 0.5
    steps: int = 20
    step_frac: float = 0.25
    random_start: bool = True
    perturbation_fraction: float = 1.0
    seed: int = 42
    raw_projection: bool = True
    smooth_lambda: float = 0.0


def propagated_sigma_dm(e: np.ndarray) -> np.ndarray:
    """Return propagated uncertainty for first differences."""
    e = np.asarray(e, dtype=np.float64)
    out = np.zeros_like(e)
    if len(e) >= 2:
        out[1:] = np.sqrt(e[1:] ** 2 + e[:-1] ** 2)
    return out


def integrate_dm(dm: np.ndarray, m0: float) -> np.ndarray:
    """Reconstruct magnitudes from Delta-m, anchoring the first observation."""
    dm = np.asarray(dm, dtype=np.float64)
    m = np.empty_like(dm)
    m[0] = m0
    if len(dm) >= 2:
        m[1:] = m0 + np.cumsum(dm[1:])
    return m


def project_raw_magnitude(
    m_adv: np.ndarray,
    m_clean: np.ndarray,
    e: np.ndarray,
    k: float,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Clip valid observations to the pointwise photometric budget."""
    m_adv = np.asarray(m_adv, dtype=np.float64)
    m_clean = np.asarray(m_clean, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    bound = k * e
    return np.where(
        valid_mask,
        np.clip(m_adv, m_clean - bound, m_clean + bound),
        m_clean,
    )


def select_perturbation_mask(
    n_used: int,
    seq_len: int,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select raw observations to perturb; the anchor at index zero is fixed."""
    mask = np.zeros(seq_len, dtype=bool)
    if n_used <= 1 or fraction <= 0.0:
        return mask
    candidates = np.arange(1, n_used)
    n_pick = int(round(fraction * len(candidates)))
    if n_pick > 0:
        mask[rng.choice(candidates, size=n_pick, replace=False)] = True
    return mask


def point_mask_from_n_used(n_used: np.ndarray, seq_len: int) -> np.ndarray:
    """Return a (B,L) mask for real, non-padded observations."""
    n_used = np.asarray(n_used, dtype=np.int32)
    return np.arange(seq_len)[None, :] < n_used[:, None]


def encode_magnitude_budget(
    e_raw: np.ndarray,
    n_used: np.ndarray,
    sd_m: float,
    k: float,
) -> np.ndarray:
    """Encode standardized magnitude budgets, using -1 for padded positions."""
    e_raw = np.asarray(e_raw, dtype=np.float32)
    valid = point_mask_from_n_used(n_used, e_raw.shape[1])
    sd = max(abs(float(sd_m)), 1e-8)
    eps = np.full(e_raw.shape, -1.0, dtype=np.float32)
    eps[valid] = (k * np.abs(e_raw[valid]) / sd).astype(np.float32)
    return eps


def magnitude_delta_to_dm_tf(
    delta_mag_std: tf.Tensor,
    point_valid: tf.Tensor,
) -> tf.Tensor:
    """Convert standardized magnitude displacement to Delta-m displacement."""
    delta_mag_std = tf.cast(delta_mag_std, tf.float32)
    point_valid = tf.cast(point_valid, tf.float32)
    dm_delta = tf.concat(
        [tf.zeros_like(delta_mag_std[:, :1]),
         delta_mag_std[:, 1:] - delta_mag_std[:, :-1]],
        axis=1,
    )
    index_mask = tf.concat(
        [tf.zeros_like(point_valid[:, :1]), tf.ones_like(point_valid[:, 1:])],
        axis=1,
    )
    return dm_delta * point_valid * index_mask


def apply_magnitude_delta_tf(
    x_dm: tf.Tensor,
    delta_mag_std: tf.Tensor,
    point_valid: tf.Tensor,
) -> tf.Tensor:
    """Apply a raw-magnitude displacement to a standardized Delta-m tensor."""
    return tf.cast(x_dm, tf.float32) + magnitude_delta_to_dm_tf(
        delta_mag_std, point_valid
    )


@tf.function(reduce_retracing=True)
def pgd_magnitude_tf(
    forward: Callable,
    x_dt: tf.Tensor,
    x_dm: tf.Tensor,
    y_oh: tf.Tensor,
    eps_mag_encoded: tf.Tensor,
    steps: tf.Tensor,
    step_frac: tf.Tensor,
    random_start: tf.Tensor,
    smooth_lambda: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Untargeted PGD on per-observation magnitude displacements.

    ``eps_mag_encoded`` is in standardized-magnitude units. Real observations
    have a nonnegative budget; padded positions are encoded as -1. The first
    observation is held fixed because differencing removes any global offset.
    """
    x_dt = tf.cast(x_dt, tf.float32)
    x_dm = tf.cast(x_dm, tf.float32)
    y_oh = tf.cast(y_oh, tf.float32)
    encoded = tf.cast(eps_mag_encoded, tf.float32)
    point_valid = encoded >= 0.0
    eps = tf.maximum(encoded, 0.0)
    anchor_mask = tf.concat(
        [tf.zeros_like(eps[:, :1]), tf.ones_like(eps[:, 1:])], axis=1
    )
    perturbable = tf.cast(point_valid, tf.float32) * anchor_mask
    eps = eps * perturbable

    def random_delta():
        return tf.random.uniform(tf.shape(eps), -1.0, 1.0) * eps

    delta = tf.cond(
        tf.cast(random_start, tf.bool),
        random_delta,
        lambda: tf.zeros_like(eps),
    )
    step = tf.maximum(tf.cast(step_frac, tf.float32) * eps, 1e-8)
    smooth_lambda = tf.cast(smooth_lambda, tf.float32)

    def body(i, current):
        with tf.GradientTape() as tape:
            tape.watch(current)
            dm_adv = apply_magnitude_delta_tf(x_dm, current, point_valid)
            p_adv = forward(x_dt, dm_adv)
            ce = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(y_oh, p_adv)
            )
            d2 = current[:, 2:] - 2.0 * current[:, 1:-1] + current[:, :-2]
            smooth = tf.reduce_mean(tf.square(d2))
            objective = ce - smooth_lambda * smooth
        grad = tape.gradient(objective, current)
        grad = tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(grad))
        updated = current + step * tf.sign(grad) * perturbable
        updated = tf.clip_by_value(updated, -eps, eps)
        return i + 1, updated

    _, delta = tf.while_loop(
        lambda i, _delta: i < steps,
        body,
        (tf.constant(0), delta),
        parallel_iterations=1,
    )
    dm_adv = apply_magnitude_delta_tf(x_dm, delta, point_valid)
    return dm_adv, delta


def attack_batch(
    forward: Callable,
    x_dt_scaled: np.ndarray,
    x_dm_scaled: np.ndarray,
    y_oh: np.ndarray,
    m_raw: np.ndarray,
    e_raw: np.ndarray,
    n_used: np.ndarray,
    sd_m: float,
    mu_m: float,
    cfg: AttackConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Attack a batch and return self-consistent bounded model inputs."""
    del mu_m  # Delta displacements are invariant to the standardization mean.
    rng = np.random.default_rng(cfg.seed)
    batch, seq_len = x_dm_scaled.shape
    valid_points = point_mask_from_n_used(n_used, seq_len)
    perturb_mask = np.zeros((batch, seq_len), dtype=bool)
    for i in range(batch):
        perturb_mask[i] = select_perturbation_mask(
            int(n_used[i]), seq_len, cfg.perturbation_fraction, rng
        )

    sd = max(abs(float(sd_m)), 1e-8)
    eps = np.full((batch, seq_len), -1.0, dtype=np.float32)
    eps[valid_points] = 0.0
    eps[perturb_mask] = (
        cfg.k * np.abs(np.asarray(e_raw, dtype=np.float32)[perturb_mask]) / sd
    )

    dm_adv_scaled, delta_std = pgd_magnitude_tf(
        forward,
        tf.constant(x_dt_scaled, tf.float32),
        tf.constant(x_dm_scaled, tf.float32),
        tf.constant(y_oh, tf.float32),
        tf.constant(eps, tf.float32),
        tf.constant(cfg.steps, tf.int32),
        tf.constant(cfg.step_frac, tf.float32),
        tf.constant(cfg.random_start, tf.bool),
        tf.constant(cfg.smooth_lambda, tf.float32),
    )
    dm_adv_scaled = dm_adv_scaled.numpy()
    delta_raw = delta_std.numpy().astype(np.float64) * sd
    m_adv = np.asarray(m_raw, dtype=np.float64).copy()
    m_adv[valid_points] += delta_raw[valid_points]

    # A final numerical clip protects against float round-off at the boundary.
    if cfg.raw_projection:
        m_adv = project_raw_magnitude(
            m_adv, m_raw, e_raw, cfg.k, valid_points
        )

    dm_adv_raw = np.zeros_like(m_adv, dtype=np.float64)
    for i, count in enumerate(np.asarray(n_used, dtype=np.int32)):
        if count >= 2:
            dm_adv_raw[i, 1:count] = np.diff(m_adv[i, :count])

    dm_adv_scaled_check = np.asarray(x_dm_scaled, dtype=np.float32).copy()
    dm_delta_raw = dm_adv_raw - np.concatenate(
        [np.zeros((batch, 1)), np.diff(np.asarray(m_raw), axis=1)], axis=1
    )
    dm_adv_scaled_check += (dm_delta_raw / sd).astype(np.float32)
    return (
        dm_adv_scaled_check,
        m_adv,
        dm_adv_raw.astype(np.float32),
        perturb_mask,
    )
