"""Square Attack (Andriushchenko et al. 2020) against the Random Forest
baseline on the OGLE-III adversarial test set.

This is a score-based, decision-free black-box attack that does not
require a differentiable surrogate. It iteratively flips a small random
subset of the per-point perturbation to its bound (+/- k*sigma_dm[i]),
queries the RF, and keeps the change whenever the true-class confidence
decreases. Curves that already flip class are removed from the active
set to speed up convergence.

The implementation supports the same heterogeneous per-point budget as
the PGD attacks elsewhere in this paper, so the resulting adversarial
test set is directly comparable to the CNN/RNN white-box attacks.

Output: outputs/metrics/ogle/rf_square_attack.csv
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.experiments.build_adversarials_ogle import (
    LABEL_TO_INT, LABELS, SEED, SEQ_LEN,
    build_catalog, load_lightcurve, subsample_balanced, build_inputs,
)

REPO = Path(__file__).resolve().parents[2]
ADV_ROOT = REPO / "runs/adversarial_dataset"
CNN_DIR = REPO / "runs/cnn_ogle"
PERIODS_NPZ = REPO / "runs/periods_ogle.npz"
METRICS = REPO / "outputs/metrics/ogle"

K_BUDGET = 0.5
N_ITER = 4000
P_INIT = 0.05   # initial fraction of positions perturbed per step
SEED_RNG = 1234


def _features(X2: np.ndarray, periods_arr: np.ndarray) -> np.ndarray:
    """7 hand-engineered features used by the RF baseline."""
    M = X2.squeeze(-1)
    f = np.column_stack([
        M.mean(axis=1),
        M.std(axis=1),
        skew(M, axis=1, bias=False),
        kurtosis(M, axis=1, bias=False),
        M.max(axis=1) - M.min(axis=1),
        np.log10(np.clip(np.abs(M).mean(axis=1), 1e-6, None)),
        np.log10(np.clip(periods_arr, 1e-3, None)),
    ])
    return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)


def square_attack_rf(X2_clean: np.ndarray, y: np.ndarray, periods: np.ndarray,
                      sigma_dm: np.ndarray, rf, k: float, n_iter: int,
                      p_init: float, rng_seed: int) -> tuple:
    """Square Attack on the RF baseline with per-point budget k*sigma_dm.

    Returns (X2_adv, n_iters_run).
    """
    rng = np.random.default_rng(rng_seed)
    N, L, _ = X2_clean.shape
    eps = (k * np.abs(sigma_dm)).squeeze(-1)            # (N, L)
    M_clean = X2_clean.squeeze(-1)                        # (N, L) in Delta-m space
    # Initialize with +/- eps random sign per position
    signs = rng.choice([-1.0, 1.0], size=(N, L)).astype(np.float32)
    M_adv = (M_clean + signs * eps).astype(np.float32)
    # Bound to clean +/- eps
    M_adv = np.clip(M_adv, M_clean - eps, M_clean + eps)

    # Evaluate initial state
    feats = _features(M_adv[:, :, None], periods)
    probs = rf.predict_proba(feats)
    true_p = probs[np.arange(N), y]
    preds = probs.argmax(axis=1)
    success = preds != y

    print(f"  initial success rate after random init: {success.mean():.4f}")

    for t in range(1, n_iter + 1):
        # Decay number of perturbed positions
        p_step = p_init * (1.0 - t / n_iter)
        n_perturb = max(1, int(L * p_step))

        # Only update curves still active (not yet flipped)
        active = np.where(~success)[0]
        if len(active) == 0:
            print(f"  iter {t}: all curves attacked, stopping early")
            break

        # Build candidate proposals for ACTIVE curves only
        M_proposal = M_adv[active].copy()
        for i_local, i_glob in enumerate(active):
            pos = rng.choice(L, n_perturb, replace=False)
            new_signs = rng.choice([-1.0, 1.0], size=n_perturb).astype(np.float32)
            M_proposal[i_local, pos] = M_clean[i_glob, pos] + new_signs * eps[i_glob, pos]
        # Reproject to bound (in case)
        M_proposal = np.clip(M_proposal,
                              M_clean[active] - eps[active],
                              M_clean[active] + eps[active])

        # Batch evaluate proposals
        feats_p = _features(M_proposal[:, :, None], periods[active])
        probs_p = rf.predict_proba(feats_p)
        true_p_p = probs_p[np.arange(len(active)), y[active]]
        preds_p = probs_p.argmax(axis=1)

        # Accept if true-class probability decreased OR class already flipped
        improved = (true_p_p < true_p[active]) | (preds_p != y[active])
        accept_idx = active[improved]
        M_adv[accept_idx] = M_proposal[improved]
        true_p[accept_idx] = true_p_p[improved]
        preds[accept_idx] = preds_p[improved]
        success = preds != y

        if t % 50 == 0 or t == n_iter:
            print(f"  iter {t}/{n_iter}  active={len(active)}  "
                  f"success_rate={success.mean():.4f}  n_perturb={n_perturb}")

    return M_adv[:, :, None].astype(np.float32), t


def main(k_budget=K_BUDGET, out_name="rf_square_attack.csv"):
    METRICS.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED); rng_seed = SEED_RNG

    print("[1/4] Loading splits + CNN stats + RF")
    sp = np.load(ADV_ROOT/"test_split.npz", allow_pickle=True)
    X1_te = sp["X1_te"]; X2_te = sp["X2_te"]
    y_te = sp["y_te"]; ids_te = sp["ids_te"]; valid = sp["valid"]
    DE_te = sp["de_te"]; mu_m = float(sp["mu_m"]); sd_m = float(sp["sd_m"])

    periods_npz = np.load(PERIODS_NPZ, allow_pickle=True)
    periods_dict = dict(zip(periods_npz["IDs"], periods_npz["period"]))
    P_te = np.array([periods_dict.get(str(i), np.nan) for i in ids_te])

    print("[2/4] Rebuilding TRAIN to fit RF")
    cat = build_catalog(); df = subsample_balanced(cat)
    y_all = df["label4"].map(LABEL_TO_INT).to_numpy()
    ids_all = df["ID"].to_numpy()
    idx_tr, _ = train_test_split(np.arange(len(df)), test_size=0.2,
                                  random_state=SEED, stratify=y_all)
    df_tr = df.iloc[idx_tr].reset_index(drop=True)
    y_tr = y_all[idx_tr]; ids_tr = ids_all[idx_tr]
    P_tr = np.array([periods_dict.get(str(i), np.nan) for i in ids_tr])

    stats = np.load(CNN_DIR/"preprocess_stats.npz")
    mu_t = float(stats["mu_t"].squeeze()); sd_t = float(stats["sd_t"].squeeze())
    mu_m_cnn = float(stats["mu_m"].squeeze()); sd_m_cnn = float(stats["sd_m"].squeeze())
    n_tr = len(df_tr)
    X2_tr = np.zeros((n_tr, SEQ_LEN, 1), np.float32)
    keep_tr = np.zeros(n_tr, bool)
    for i, row in df_tr.iterrows():
        try:
            lc = load_lightcurve(Path(row["lc_file"]))
            if len(lc) < 3: continue
            t = lc["time"].to_numpy(float); m = lc["magnitude"].to_numpy(float); e = lc["error"].to_numpy(float)
            _, dm_std, _, _, _, _, _ = build_inputs(t, m, e, mu_t, sd_t, mu_m_cnn, sd_m_cnn)
            X2_tr[i, :, 0] = dm_std; keep_tr[i] = True
        except Exception:
            pass
        if (i+1) % 10000 == 0:
            print(f"  train built {i+1}/{n_tr}")
    Ftr = _features(X2_tr[keep_tr], P_tr[keep_tr])
    rf = RandomForestClassifier(n_estimators=300, max_depth=None,
                                 n_jobs=-1, random_state=SEED).fit(Ftr, y_tr[keep_tr])
    rf_acc_clean = float(accuracy_score(y_te[valid],
                                          rf.predict(_features(X2_te, P_te))[valid]))
    print(f"  RF clean acc on test: {rf_acc_clean:.4f}")

    print(f"[3/4] Running Square Attack (k={k_budget}, n_iter={N_ITER}, p_init={P_INIT})")
    # sigma_dm budget = de_te / sd_m_cnn (same convention as PGD)
    sigma_dm = DE_te / sd_m_cnn
    t0 = time.time()
    X2_adv, iters_run = square_attack_rf(X2_te, y_te, P_te, sigma_dm, rf,
                                           k_budget, N_ITER, P_INIT, rng_seed)
    dt = time.time() - t0
    print(f"  finished in {dt/60:.1f} min ({iters_run} iters)")

    print("[4/4] Final evaluation")
    rf_adv = rf.predict(_features(X2_adv, P_te))
    acc_adv = float(accuracy_score(y_te[valid], rf_adv[valid]))
    drop = rf_acc_clean - acc_adv
    print(f"  RF acc_clean = {rf_acc_clean:.4f}")
    print(f"  RF acc_adv   = {acc_adv:.4f}")
    print(f"  drop         = {drop*100:+.1f} pp")

    rows = [
        {"attack": "clean", "k": 0.0, "model": "RF", "acc": rf_acc_clean,
         "n_eval": int(valid.sum()), "n_iter": 0},
        {"attack": "rf_square", "k": k_budget, "model": "RF", "acc": acc_adv,
         "n_eval": int(valid.sum()), "n_iter": iters_run},
    ]
    pd.DataFrame(rows).to_csv(METRICS/out_name, index=False)
    print(f"Wrote {METRICS/out_name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, default=K_BUDGET,
                    help="per-point photometric budget in units of sigma_dm")
    ap.add_argument("--out", type=str, default="rf_square_attack.csv",
                    help="output CSV filename (written under outputs/metrics/ogle)")
    args = ap.parse_args()
    main(k_budget=args.k, out_name=args.out)
