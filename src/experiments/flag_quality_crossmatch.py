"""Do adversarially-flipped sources have worse photometric quality?

Cross-matches the per-source robustness flag (prediction changes under the
transferred attack) against instrument-level quality proxies computed from
the light curves themselves (median photometric error, per-curve S/N as
amplitude / median error, and number of observations) for the CNN and
for RRE on OGLE-III. If flipped sources are systematically noisier, the
flag tracks genuine data quality, not just classifier idiosyncrasy.

Usage:
    python -m src.experiments.flag_quality_crossmatch
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]


def main():
    pe = np.load(REPO / "outputs/metrics/ogle/per_example_preds.npz",
                 allow_pickle=True)
    sp = np.load(REPO / "runs/adversarial_dataset/test_split.npz",
                 allow_pickle=True)

    valid = sp["valid_te"].astype(bool)
    n_obs = sp["n_sel_te"].astype(int)
    e_raw = sp["E_te"].squeeze(-1)          # reported errors (standardized-free)
    x2 = sp["X2_te"].squeeze(-1)            # standardized Delta-m sequence

    # per-source quality proxies over the real (unpadded) points
    point_valid = np.arange(e_raw.shape[1])[None, :] < n_obs[:, None]
    med_err = np.array([np.median(e_raw[i, point_valid[i]])
                        if point_valid[i].any() else np.nan
                        for i in range(len(n_obs))])
    # amplitude proxy from the differential channel (robust range)
    amp = np.array([np.subtract(*np.percentile(x2[i, point_valid[i]], [95, 5]))
                    if point_valid[i].sum() > 2 else np.nan
                    for i in range(len(n_obs))])
    snr = np.abs(amp) / np.maximum(med_err, 1e-6)

    for model, clean, adv in [("CNN", pe["cnn_clean"], pe["cnn_adv"]),
                              ("RRE", pe["l5_clean"], pe["l5_adv"])]:
        flip = (clean != adv) & valid
        stable = (clean == adv) & valid
        print(f"\n=== {model} ===")
        print(f"flipped: {flip.sum()}  stable: {stable.sum()}")
        for name, q in [("median error (mag)", med_err),
                        ("S/N (amp/err)", snr),
                        ("n_obs", n_obs.astype(float))]:
            qf = q[flip]; qf = qf[np.isfinite(qf)]
            qs = q[stable]; qs = qs[np.isfinite(qs)]
            if len(qf) < 5:
                print(f"  {name:20s}: too few flips ({len(qf)})")
                continue
            u, p = stats.mannwhitneyu(qf, qs, alternative="two-sided")
            print(f"  {name:20s}: flipped median={np.median(qf):.3f}  "
                  f"stable median={np.median(qs):.3f}  "
                  f"Mann-Whitney p={p:.2e}")


if __name__ == "__main__":
    main()
