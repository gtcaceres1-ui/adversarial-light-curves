"""Robust raw-curve encoder for supervised adversarial training.

Five-term objective (clean CE, adversarial CE, TRADES consistency, supervised
contrastive, VIB) on the raw (Delta t, Delta m, sigma_dm) representation of
length SEQ_LEN. No period estimation or phase folding is involved, so there
is no class-conditional frequency search that could leak label information.

Input shape: (N, SEQ_LEN, 3): channels are standardized Delta t, Delta m,
and per-point sigma of Delta m.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, layers

from src.attacks.photometric_pgd import apply_magnitude_delta_tf


SEQ_LEN_DEFAULT = 500
LATENT_DEFAULT = 48
N_CLASSES_DEFAULT = 4


def supcon_loss(z, y, tau=0.1):
    """Supervised contrastive loss; z must be L2-normalized."""
    sim = tf.matmul(z, z, transpose_b=True) / tau
    pos_mask = tf.cast(y[:, None] == y[None, :], tf.float32)
    self_mask = tf.eye(tf.shape(z)[0])
    pos_mask = pos_mask - self_mask
    neg_mask = 1.0 - self_mask
    sim_for_max = sim - 1e9 * self_mask
    max_sim = tf.reduce_max(sim_for_max, axis=-1, keepdims=True)
    sim_stable = sim - max_sim
    exp_sim = tf.exp(sim_stable) * neg_mask
    log_denom = tf.math.log(tf.reduce_sum(exp_sim, axis=-1, keepdims=True) + 1e-12)
    log_prob = sim_stable - log_denom
    pos_count = tf.reduce_sum(pos_mask, axis=-1) + 1e-12
    per_anchor = -tf.reduce_sum(pos_mask * log_prob, axis=-1) / pos_count
    has_pos = tf.cast(tf.reduce_sum(pos_mask, axis=-1) > 0, tf.float32)
    return tf.reduce_sum(per_anchor * has_pos) / (tf.reduce_sum(has_pos) + 1e-12)


class _Sampling(layers.Layer):
    def call(self, inputs):
        mu, lv = inputs
        return mu + tf.exp(0.5 * lv) * tf.random.normal(tf.shape(mu))


def build_encoder_raw(seq_len: int, latent_dim: int) -> Model:
    """Two-tower Conv1D backbone inspired by the CNN baseline of
    \\citet{aguirre2019deep}: a time-difference tower and a
    magnitude+uncertainty tower whose feature maps are concatenated
    before the joint convolutional block.

    Inputs:
      - dt_in        : (seq_len, 1) standardized Delta t
      - dm_sigma_in  : (seq_len, 2) standardized (Delta m, sigma_dm)

    Output: (mu, log_var, z) of dimension latent_dim.
    """
    inp_t = layers.Input((seq_len, 1), name="dt_in")
    inp_m = layers.Input((seq_len, 2), name="dm_sigma_in")

    # Tower 1: cadence
    xt = layers.Conv1D(32, 7, padding="same", activation="relu")(inp_t)
    xt = layers.Conv1D(64, 5, strides=2, padding="same", activation="relu")(xt)

    # Tower 2: magnitude + uncertainty
    xm = layers.Conv1D(32, 7, padding="same", activation="relu")(inp_m)
    xm = layers.Conv1D(64, 5, strides=2, padding="same", activation="relu")(xm)

    # Fuse and process jointly
    x = layers.Concatenate(axis=-1)([xt, xm])
    x = layers.Conv1D(192, 5, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv1D(256, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation="relu")(x)
    mu = layers.Dense(latent_dim, name="z_mean")(x)
    lv = layers.Dense(latent_dim, name="z_log_var")(x)
    z  = _Sampling(name="z")([mu, lv])
    return Model([inp_t, inp_m], [mu, lv, z], name="encoder_raw_twotower")


def build_classifier_head(latent_dim, n_classes, dropout=0.2):
    inp = layers.Input((latent_dim,))
    x = layers.Dropout(dropout)(inp)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return Model(inp, out, name="classifier_head_raw")


def build_projection_head(latent_dim, proj_dim=64):
    inp = layers.Input((latent_dim,))
    x = layers.Dense(128, activation="relu")(inp)
    x = layers.Dense(proj_dim)(x)
    return Model(inp, x, name="projection_head_raw")


class RobustRawEncoder(Model):
    """L5 trained on raw (Delta t, Delta m, sigma) input."""

    def __init__(self, seq_len: int = SEQ_LEN_DEFAULT,
                 latent_dim: int = LATENT_DEFAULT,
                 n_classes: int = N_CLASSES_DEFAULT,
                 use_adv_ce: bool = True, use_trades: bool = True,
                 use_supcon: bool = True, use_vib: bool = True,
                 lambda_adv_ce: float = 1.0, lambda_trades: float = 2.0,
                 lambda_supcon: float = 0.5, beta_vib: float = 5e-4,
                 supcon_tau: float = 0.1,
                 adv_k: float = 0.5, adv_steps: int = 3,
                 adv_step_frac: float = 0.4,
                 attack_mode: str = "ce", **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.use_adv_ce = bool(use_adv_ce)
        self.use_trades = bool(use_trades)
        self.use_supcon = bool(use_supcon)
        self.use_vib = bool(use_vib)
        self.lambda_adv_ce = float(lambda_adv_ce)
        self.lambda_trades = float(lambda_trades)
        self.lambda_supcon = float(lambda_supcon)
        self.beta_vib = float(beta_vib)
        self.supcon_tau = float(supcon_tau)
        self.adv_k = float(adv_k)
        self.adv_steps = int(adv_steps)
        self.adv_step_frac = float(adv_step_frac)
        if attack_mode != "ce":
            raise ValueError("RobustRawEncoder uses a fixed CE inner attack")
        self.attack_mode = attack_mode

        self.encoder_backbone = build_encoder_raw(seq_len, latent_dim)
        self.classifier = build_classifier_head(latent_dim, n_classes)
        self.proj = build_projection_head(latent_dim) if self.use_supcon else None
        names = ("loss", "ce_cl", "ce_ad", "kl_trades", "supcon",
                 "kl_vib", "acc_cl", "acc_ad")
        self._metric_trackers = {
            name: tf.keras.metrics.Mean(name=name) for name in names
        }

    @property
    def metrics(self):
        return list(self._metric_trackers.values())

    def encode(self, x_t, x_m, training: bool):
        mu, lv, z = self.encoder_backbone([x_t, x_m], training=training)
        if self.use_vib:
            return mu, lv, z
        return mu, lv, mu

    def predict_proba(self, x_t, x_m, training: bool):
        _, _, z = self.encode(x_t, x_m, training)
        return self.classifier(z, training=training), z

    def _attack(self, x_t, x_m, eps_mag_encoded, y_onehot):
        """Inner PGD over bounded per-observation magnitude displacements.

        x_t  : (B, L, 1) standardized Delta t  (kept fixed)
        x_m  : (B, L, 2) standardized (Delta m, sigma_dm)  (only dm is moved)
        eps_mag_encoded: (B, L) magnitude budget in standardized space;
                         padded positions are encoded as -1.
        """
        encoded = tf.cast(eps_mag_encoded, tf.float32)
        point_valid = encoded >= 0.0
        eps = tf.maximum(encoded, 0.0)
        anchor_mask = tf.concat(
            [tf.zeros_like(eps[:, :1]), tf.ones_like(eps[:, 1:])], axis=1
        )
        eps = eps * tf.cast(point_valid, tf.float32) * anchor_mask
        delta = tf.random.uniform(tf.shape(eps), -1.0, 1.0) * eps
        step = tf.maximum(self.adv_step_frac * eps, 1e-8)

        def _apply(xm, d):
            dm = apply_magnitude_delta_tf(xm[..., 0], d, point_valid)
            return tf.stack([dm, xm[..., 1]], axis=-1)

        for _ in range(self.adv_steps):
            with tf.GradientTape() as t:
                t.watch(delta)
                xm_try = _apply(x_m, delta)
                p_ad, _ = self.predict_proba(x_t, xm_try, training=False)
                obj = tf.reduce_mean(
                    tf.keras.losses.categorical_crossentropy(y_onehot, p_ad))
            g = t.gradient(obj, delta)
            g = tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
            delta = tf.clip_by_value(delta + step * tf.sign(g), -eps, eps)
        return tf.stop_gradient(_apply(x_m, delta))

    def _compute(self, data, training: bool):
        x_t, x_m, eps_mag_encoded, y = data
        # Ensure all inputs are tf.Tensor so the inner attack does not mix
        # numpy and tensor types when called from the dummy forward in train.
        x_t = tf.cast(x_t, tf.float32)
        x_m = tf.cast(x_m, tf.float32)
        eps_mag_encoded = tf.cast(eps_mag_encoded, tf.float32)
        y = tf.cast(y, tf.int32)
        y_onehot = tf.one_hot(y, self.n_classes)

        any_adv = self.use_adv_ce or self.use_trades or self.use_supcon
        if any_adv:
            x_m_adv = self._attack(x_t, x_m, eps_mag_encoded, y_onehot)
        else:
            x_m_adv = x_m

        mu_cl, lv_cl, z_cl = self.encode(x_t, x_m, training)
        p_cl = self.classifier(z_cl, training=training)
        ce_cl = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(y_onehot, p_cl))

        total = ce_cl
        ce_ad = tf.constant(0.0); kl_trades = tf.constant(0.0)
        supcon = tf.constant(0.0); kl_vib = tf.constant(0.0)

        if any_adv:
            mu_ad, _, z_ad = self.encode(x_t, x_m_adv, training)
            p_ad = self.classifier(z_ad, training=training)
            if self.use_adv_ce:
                ce_ad = tf.reduce_mean(
                    tf.keras.losses.categorical_crossentropy(y_onehot, p_ad))
                total = total + self.lambda_adv_ce * ce_ad
            if self.use_trades:
                kl_trades = tf.reduce_mean(tf.reduce_sum(
                    p_cl * (tf.math.log(p_cl + 1e-12)
                             - tf.math.log(p_ad + 1e-12)), axis=-1))
                total = total + self.lambda_trades * kl_trades
            if self.use_supcon:
                z_proj_cl = tf.math.l2_normalize(self.proj(mu_cl), axis=-1)
                z_proj_ad = tf.math.l2_normalize(self.proj(mu_ad), axis=-1)
                z_all = tf.concat([z_proj_cl, z_proj_ad], axis=0)
                y_all = tf.concat([y, y], axis=0)
                supcon = supcon_loss(z_all, y_all, tau=self.supcon_tau)
                total = total + self.lambda_supcon * supcon

        if self.use_vib:
            kl_vib = -0.5 * tf.reduce_mean(tf.reduce_sum(
                1.0 + lv_cl - tf.square(mu_cl) - tf.exp(lv_cl), axis=-1))
            total = total + self.beta_vib * kl_vib

        acc_cl = tf.reduce_mean(tf.cast(
            tf.argmax(p_cl, axis=-1) == tf.cast(y, tf.int64), tf.float32))
        if any_adv:
            acc_ad = tf.reduce_mean(tf.cast(
                tf.argmax(p_ad, axis=-1) == tf.cast(y, tf.int64), tf.float32))
        else:
            acc_ad = tf.constant(0.0)

        return total, ce_cl, ce_ad, kl_trades, supcon, kl_vib, acc_cl, acc_ad

    def _update_metrics(self, values, batch_size):
        names = ("loss", "ce_cl", "ce_ad", "kl_trades", "supcon",
                 "kl_vib", "acc_cl", "acc_ad")
        for name, value in zip(names, values):
            self._metric_trackers[name].update_state(
                value, sample_weight=tf.cast(batch_size, tf.float32)
            )
        return {name: tracker.result()
                for name, tracker in self._metric_trackers.items()}

    def train_step(self, data):
        with tf.GradientTape() as tape:
            values = self._compute(data, training=True)
        grads = tape.gradient(values[0], self.trainable_variables)
        pairs = [(g, v) for g, v in zip(grads, self.trainable_variables)
                 if g is not None]
        self.optimizer.apply_gradients(pairs)
        return self._update_metrics(values, tf.shape(data[3])[0])

    def test_step(self, data):
        return self._update_metrics(
            self._compute(data, training=False), tf.shape(data[3])[0]
        )
