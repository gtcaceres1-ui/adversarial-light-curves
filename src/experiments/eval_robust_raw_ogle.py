"""Evaluate the trained Robust Raw Encoder (L5-raw) on the OGLE-III
adversarial test set, comparing against the CNN baseline.

Outputs:
  outputs/metrics/ogle/robust_raw_eval.csv
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
CNN_DIR = REPO / "runs/cnn_ogle"
L5_DIR = REPO / "runs/robust_raw_l5_ogle"
METRICS = REPO / "outputs/metrics/ogle"


def _make_two_tower(X1, X2, DE, sd_m):
    """Stack into (XT, XM_sigma) where XM_sigma has Delta m and sigma_dm."""
    sigma_std = (DE.squeeze(-1) / sd_m).astype(np.float32)
    xt = X1.astype(np.float32)
    xm = np.concatenate([X2, sigma_std[..., None]], axis=-1).astype(np.float32)
    return xt, xm


def main():
    METRICS.mkdir(parents=True, exist_ok=True)
    print("[1/3] Loading split + L5-raw")
    sp = np.load(ADV_ROOT/"test_split.npz", allow_pickle=True)
    X1_te = sp["X1_te"]; X2_te = sp["X2_te"]; X2_te_adv = sp["X2_te_adv"]
    DE_te = sp["de_te"]; y_te = sp["y_te"]; valid = sp["valid"]
    sd_m = float(sp["sd_m"])

    encoder = tf.keras.models.load_model(L5_DIR/"encoder_backbone.keras",
                                          custom_objects={"_Sampling": _Sampling},
                                          compile=False, safe_mode=False)
    clf = tf.keras.models.load_model(L5_DIR/"classifier.keras",
                                      custom_objects={"_Sampling": _Sampling},
                                      compile=False, safe_mode=False)

    print("[2/3] Building two-tower inputs")
    XT, XM_cl = _make_two_tower(X1_te, X2_te, DE_te, sd_m)
    _,  XM_ad = _make_two_tower(X1_te, X2_te_adv, DE_te, sd_m)

    def _predict(xt, xm, bs=512):
        n = len(xt); preds = np.zeros(n, np.int64)
        for s in range(0, n, bs):
            e = min(s+bs, n)
            mu, _, _ = encoder([xt[s:e], xm[s:e]], training=False)
            p = clf(mu, training=False).numpy()
            preds[s:e] = p.argmax(axis=1)
        return preds

    print("[3/3] Predicting clean + adv")
    p_cl = _predict(XT, XM_cl)
    p_ad = _predict(XT, XM_ad)

    m = valid
    acc_cl = float(accuracy_score(y_te[m], p_cl[m]))
    acc_ad = float(accuracy_score(y_te[m], p_ad[m]))
    drop = acc_ad - acc_cl
    print(f"  L5-raw OGLE  clean={acc_cl:.4f}  adv={acc_ad:.4f}  drop={drop*100:+.1f} pp")

    # Per-class
    rows = [{"model": "L5-raw", "scope": "all",
             "acc_clean": acc_cl, "acc_adv": acc_ad,
             "drop": drop, "n_valid": int(m.sum())}]
    print("  Per-class:")
    for c, lab in enumerate(LABELS):
        mc = m & (y_te == c)
        if mc.sum() == 0:
            continue
        ac = float(accuracy_score(y_te[mc], p_cl[mc]))
        aa = float(accuracy_score(y_te[mc], p_ad[mc]))
        d = aa - ac
        print(f"    {lab:6s}: clean={ac:.4f}  adv={aa:.4f}  drop={d*100:+.1f} pp")
        rows.append({"model": "L5-raw", "scope": lab,
                     "acc_clean": ac, "acc_adv": aa,
                     "drop": d, "n_valid": int(mc.sum())})

    pd.DataFrame(rows).to_csv(METRICS/"robust_raw_eval.csv", index=False)
    print(f"\nWrote {METRICS/'robust_raw_eval.csv'}")


if __name__ == "__main__":
    main()
