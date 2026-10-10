"""Adaptive white-box PGD attack against the Robust Raw Encoder (L5-raw).

Generates adversarial inputs whose gradients are computed on the L5-raw model
itself (encoder backbone + classifier head), not on the surrogate CNN. The
robustness measured with the CNN-targeted transferred attack could in
principle be a transferability artifact (a classifier that the CNN's gradient
does not "see" still appears robust). An attack that follows the L5-raw's own
gradient is the strongest first-order threat we can pose to the model, and is
the adaptive evaluation recommended by Carlini et al. (2019), "On Evaluating
Adversarial Robustness".

The attack perturbs only the Delta m channel of the (Delta t, Delta m, sigma)
two-tower input. The per-point budget is k * sigma_i (standardized), matching
the photometric constraint used everywhere else.

Output:
  outputs/metrics/ogle/adaptive_pgd_l5_raw.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score

from src.experiments.build_adversarials_ogle import LABELS
from src.models.robust_raw_encoder import _Sampling

REPO = Path(__file__).resolve().parents[2]
ADV_ROOT = REPO / "runs/adversarial_dataset"
L5_DIR = REPO / "runs/robust_raw_l5_ogle"
METRICS = REPO / "outputs/metrics/ogle"

K = 0.5            # per-point photometric budget
T = 20             # number of PGD iterations (stronger than the inner train attack)
ALPHA_FRAC = 0.2   # step size as a fraction of the per-point budget
BS = 256           # batch size during the attack


def _make_two_tower(X1, X2, DE, sd_m):
    """Build the two-tower input plus the per-point Delta m budget."""
    sigma_std = (DE.squeeze(-1) / sd_m).astype(np.float32)
    xt = X1.astype(np.float32)
    xm = np.concatenate([X2, sigma_std[..., None]], axis=-1).astype(np.float32)
    return xt, xm, sigma_std


def _predict_clean(encoder, clf, xt, xm, bs=BS):
    n = len(xt)
    preds = np.zeros(n, np.int64)
    for s in range(0, n, bs):
        e = min(s + bs, n)
        mu, _, _ = encoder([xt[s:e], xm[s:e]], training=False)
        p = clf(mu, training=False).numpy()
        preds[s:e] = p.argmax(axis=1)
    return preds


def _attack_batch(encoder, clf, xt_b, xm_b, y_b, eps_b, n_steps=T,
                   alpha_frac=ALPHA_FRAC):
    """Adaptive PGD on the Delta m channel of xm.

    eps_b has shape (B, L, 1) and gives the per-point budget. Step size is
    alpha_frac * eps_b. delta is clipped to [-eps_b, +eps_b] every iteration.
    """
    delta = tf.Variable(
        (tf.random.uniform(eps_b.shape, minval=-1.0, maxval=1.0) * eps_b),
        trainable=True,
    )
    y_b_t = tf.constant(y_b.astype(np.int32))
    for _ in range(n_steps):
        with tf.GradientTape() as tape:
            tape.watch(delta)
            dm_pert = xm_b[..., 0:1] + delta
            xm_pert = tf.concat([dm_pert, xm_b[..., 1:2]], axis=-1)
            mu, _, _ = encoder([xt_b, xm_pert], training=False)
            p = clf(mu, training=False)
            loss = tf.reduce_sum(
                tf.keras.losses.sparse_categorical_crossentropy(
                    y_b_t, p, from_logits=False)
            )
        grad = tape.gradient(loss, delta)
        sgn = tf.sign(grad)
        delta.assign(delta + alpha_frac * eps_b * sgn)
        delta.assign(tf.clip_by_value(delta, -eps_b, eps_b))
    # Final prediction on the perturbed input
    dm_pert = xm_b[..., 0:1] + delta
    xm_pert = tf.concat([dm_pert, xm_b[..., 1:2]], axis=-1)
    mu, _, _ = encoder([xt_b, xm_pert], training=False)
    p = clf(mu, training=False).numpy()
    return p.argmax(axis=1)


def main():
    METRICS.mkdir(parents=True, exist_ok=True)
    print("[1/4] Loading split + L5-raw weights")
    sp = np.load(ADV_ROOT / "test_split.npz", allow_pickle=True)
    X1 = sp["X1_te"]; X2 = sp["X2_te"]; DE = sp["de_te"]
    y = sp["y_te"]; valid = sp["valid"]
    sd_m = float(sp["sd_m"])

    encoder = tf.keras.models.load_model(
        L5_DIR / "encoder_backbone.keras",
        custom_objects={"_Sampling": _Sampling},
        compile=False, safe_mode=False)
    clf = tf.keras.models.load_model(
        L5_DIR / "classifier.keras",
        custom_objects={"_Sampling": _Sampling},
        compile=False, safe_mode=False)

    print("[2/4] Assembling two-tower inputs")
    xt, xm, sigma_std = _make_two_tower(X1, X2, DE, sd_m)
    eps_dm = (K * np.abs(sigma_std)).astype(np.float32)  # (N, L)

    print("[3/4] Clean predictions")
    preds_cl = _predict_clean(encoder, clf, xt, xm)

    print(f"[4/4] Adaptive PGD (T={T} steps, alpha=0.2*eps)  — perturbing Delta m only")
    n = len(xt)
    preds_ad = np.zeros(n, np.int64)
    tf.random.set_seed(0)
    n_batches = (n + BS - 1) // BS
    for bi, s in enumerate(range(0, n, BS)):
        e = min(s + BS, n)
        xt_b = tf.constant(xt[s:e])
        xm_b = tf.constant(xm[s:e])
        eps_b = tf.constant(eps_dm[s:e, :, None])  # (B, L, 1)
        preds_ad[s:e] = _attack_batch(encoder, clf, xt_b, xm_b, y[s:e], eps_b)
        if (bi % 10) == 0:
            print(f"   batch {bi+1}/{n_batches}  (n={e})")

    m = valid
    acc_cl = float(accuracy_score(y[m], preds_cl[m]))
    acc_ad = float(accuracy_score(y[m], preds_ad[m]))
    drop = acc_ad - acc_cl
    print(f"\n  L5-raw OGLE  ADAPTIVE PGD (T={T}, k={K})")
    print(f"    clean={acc_cl:.4f}  adv={acc_ad:.4f}  drop={drop*100:+.1f} pp")

    rows = [{"model": "L5-raw", "attack": "adaptive_pgd", "scope": "all",
             "k": K, "T": T, "acc_clean": acc_cl, "acc_adv": acc_ad,
             "drop": drop, "n_valid": int(m.sum())}]
    print("  Per-class:")
    for c, lab in enumerate(LABELS):
        mc = m & (y == c)
        if mc.sum() == 0:
            continue
        ac = float(accuracy_score(y[mc], preds_cl[mc]))
        aa = float(accuracy_score(y[mc], preds_ad[mc]))
        d = aa - ac
        print(f"    {lab:6s}: clean={ac:.4f}  adv={aa:.4f}  drop={d*100:+.1f} pp")
        rows.append({"model": "L5-raw", "attack": "adaptive_pgd", "scope": lab,
                     "k": K, "T": T, "acc_clean": ac, "acc_adv": aa,
                     "drop": d, "n_valid": int(mc.sum())})

    out = METRICS / "adaptive_pgd_l5_raw.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
