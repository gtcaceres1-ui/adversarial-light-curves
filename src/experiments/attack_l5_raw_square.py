"""Score-based black-box Square-style attack on L5-raw.

Gradient-free random search in raw-magnitude displacement space, mirroring
the RF-FATS16 Square protocol: stratified subsample of test curves, boundary
proposals (+/- k e_i) on shrinking random subsets of points, a proposal is
kept when the true-class probability decreases (or the prediction flips).

Purpose: check against residual gradient masking. The expected (healthy)
outcome is black-box adversarial accuracy above the white-box adaptive PGD
accuracy, i.e. the gradient-free attack is weaker.

Usage:
    python -m src.experiments.attack_l5_raw_square --survey ogle
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score

from src.attacks.photometric_pgd import (
    apply_magnitude_delta_tf,
    encode_magnitude_budget,
)
from src.models.robust_raw_encoder import _Sampling

REPO = Path(__file__).resolve().parents[2]
METRICS = REPO / "outputs/metrics"
K = 0.5
N_PER_CLASS = 200
N_ITER = 1000
P_INIT = 0.05
SEED = 42
BATCH = 400

MODELS = {
    "ogle": "runs/robust_raw_l5_ogle",
    "gaia": "runs/gaia_robust_raw_l5",
    "wise": "runs/wise_robust_raw_l5",
}
ADV = {
    "ogle": "runs/adversarial_dataset",
    "gaia": "runs/gaia_adversarial_dataset",
    "wise": "runs/wise_adversarial_dataset",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--survey", default="ogle",
                        choices=["ogle", "gaia", "wise"])
    parser.add_argument("--k", type=float, default=K)
    parser.add_argument("--iters", type=int, default=N_ITER)
    args = parser.parse_args()

    sp = np.load(REPO / ADV[args.survey] / "test_split.npz",
                 allow_pickle=True)
    sd_m = float(sp["sd_m"])
    x_t = sp["X1_te"].astype(np.float32)
    x2 = sp["X2_te"][..., 0].astype(np.float32)
    sigma = (sp["DE_te"].squeeze(-1) / max(abs(sd_m), 1e-8)).astype(np.float32)
    eps = encode_magnitude_budget(
        sp["E_te"].squeeze(-1), sp["n_sel_te"], sd_m, args.k
    )
    y = sp["y_te"].astype(np.int64)
    valid = sp["valid_te"].astype(bool)

    # stratified subsample of valid test curves (same spirit as RF protocol)
    rng = np.random.default_rng(SEED)
    take = []
    for c in np.unique(y):
        pool = np.where(valid & (y == c))[0]
        take.append(rng.choice(pool, min(N_PER_CLASS, len(pool)),
                               replace=False))
    idx = np.sort(np.concatenate(take))
    x_t, x2, sigma, eps, y = x_t[idx], x2[idx], sigma[idx], eps[idx], y[idx]
    B, L = x2.shape

    encoder = tf.keras.models.load_model(
        REPO / MODELS[args.survey] / "encoder_backbone.keras",
        custom_objects={"_Sampling": _Sampling}, compile=False,
        safe_mode=False,
    )
    classifier = tf.keras.models.load_model(
        REPO / MODELS[args.survey] / "classifier.keras",
        compile=False, safe_mode=False,
    )

    point_valid = eps >= 0.0
    eps_pos = np.maximum(eps, 0.0)
    eps_pos[:, 0] = 0.0  # first observation anchored, as in the PGD

    def query(delta):
        """Class probabilities for magnitude displacement delta (B, L)."""
        probs = np.zeros((B, classifier.output_shape[-1]), np.float32)
        for s in range(0, B, BATCH):
            e = min(s + BATCH, B)
            dm_adv = apply_magnitude_delta_tf(
                tf.constant(x2[s:e]), tf.constant(delta[s:e]),
                tf.constant(point_valid[s:e]),
            )
            tower = tf.stack([dm_adv, tf.constant(sigma[s:e])], axis=-1)
            mu, _, _ = encoder(
                [tf.constant(x_t[s:e]), tower], training=False
            )
            probs[s:e] = classifier(mu, training=False).numpy()
        return probs

    # clean accuracy on the subsample
    probs = query(np.zeros_like(x2))
    pred_clean = probs.argmax(1)
    acc_clean = float(accuracy_score(y, pred_clean))

    # iteration 0: random boundary init
    rng = np.random.default_rng(SEED * 100003)
    delta = rng.choice([-1.0, 1.0], size=(B, L)).astype(np.float32) * eps_pos
    probs = query(delta)
    true_p = probs[np.arange(B), y]
    preds = probs.argmax(1)
    n_query = np.ones(B, np.int64)
    t_success = np.where(preds != y, 0, -1).astype(np.int64)

    n_valid_pts = point_valid.sum(1)
    start = time.time()
    for t in range(1, args.iters + 1):
        active = np.where(preds == y)[0]
        if len(active) == 0:
            break
        p_t = P_INIT * (1.0 - t / args.iters)
        prop = delta.copy()
        for i in active:
            avail = np.where(point_valid[i])[0]
            avail = avail[avail > 0]
            npert = max(1, int(round(p_t * n_valid_pts[i])))
            pos = rng.choice(avail, min(npert, len(avail)), replace=False)
            prop[i, pos] = (rng.choice([-1.0, 1.0], size=len(pos))
                            * eps_pos[i, pos])
        probs = query(prop)
        tp = probs[np.arange(B), y]
        pr = probs.argmax(1)
        improved = np.zeros(B, bool)
        improved[active] = ((tp[active] < true_p[active])
                            | (pr[active] != y[active]))
        delta[improved] = prop[improved]
        true_p[improved] = tp[improved]
        preds[improved] = pr[improved]
        n_query[active] += 1
        newly = improved & (pr != y) & (t_success < 0)
        t_success[newly] = t
        if t % 50 == 0:
            acc = float(np.mean(preds == y))
            print(f"iter {t}/{args.iters}: acc={acc:.4f} "
                  f"active={len(active)} elapsed={time.time()-start:.0f}s",
                  flush=True)

    acc_adv = float(accuracy_score(y, preds))
    assert (np.abs(delta) <= eps_pos + 1e-6).all(), "budget violated"
    row = {
        "model": "L5-raw",
        "attack": "square_blackbox",
        "survey": args.survey,
        "k": args.k,
        "n_subsample": B,
        "iters_run": t,
        "acc_clean_sub": acc_clean,
        "acc_adv_sub": acc_adv,
        "drop": acc_adv - acc_clean,
        "median_queries_to_success": float(
            np.median(t_success[t_success >= 0])
        ) if (t_success >= 0).any() else -1.0,
        "wall_time_s": time.time() - start,
    }
    out = METRICS / args.survey / f"l5_raw_square_attack_k{args.k:.2f}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out, index=False)
    print(row, flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
