"""Hand-crafted light-curve features for the RF baseline and feature tables.

Features are intentionally simple and survey-agnostic so they can be computed
from raw magnitudes + photometric errors alone, with no period information.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def _safe(x: float) -> float:
    return float(x) if np.isfinite(x) else 0.0


def features_from_arrays(m: np.ndarray, e: np.ndarray) -> Dict[str, float]:
    """Compute a small dict of per-curve descriptors.

    Parameters
    ----------
    m, e : 1D arrays of valid (un-padded) magnitudes and photometric errors.
    """
    m = np.asarray(m, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    n = int(m.size)
    if n < 2:
        return {k: 0.0 for k in (
            "n_obs", "amplitude", "weighted_mean", "weighted_std",
            "stetson_k", "eta", "kurtosis", "skew",
            "slope_p10", "slope_p90", "beyond1std", "mad",
            "median", "iqr", "mean_err", "median_err",
        )}

    w = 1.0 / np.maximum(e * e, 1e-12)
    wm = float(np.sum(w * m) / np.sum(w))
    wvar = float(np.sum(w * (m - wm) ** 2) / np.sum(w))
    wstd = float(np.sqrt(max(wvar, 0.0)))

    delta = (m - wm) / max(e, 1e-12) if False else (m - wm) / np.maximum(e, 1e-12)
    # Stetson K
    num = np.mean(np.abs(delta))
    den = np.sqrt(np.mean(delta * delta) + 1e-12)
    stetson_k = float(num / den) if den > 0 else 0.0

    # Eta (von Neumann ratio): variance of consecutive differences / total variance
    diffs = np.diff(m)
    var_total = float(np.var(m))
    eta = float(np.mean(diffs * diffs) / var_total) if var_total > 0 else 0.0

    # Kurtosis / skew (Fisher-Pearson)
    mu = float(np.mean(m))
    sd = float(np.std(m)) + 1e-12
    z = (m - mu) / sd
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4) - 3.0)

    amp = float(0.5 * (np.max(m) - np.min(m)))
    p10, p50, p90 = np.percentile(m, [10, 50, 90])
    iqr = float(np.percentile(m, 75) - np.percentile(m, 25))
    mad = float(np.median(np.abs(m - p50)))

    # Magnitude slope at the 10/90 percentiles (robust trend descriptors).
    idx = np.argsort(m)
    m_sorted = m[idx]
    n10 = max(int(0.1 * n), 1)
    slope_p10 = float(m_sorted[n10] - m_sorted[0])
    slope_p90 = float(m_sorted[-1] - m_sorted[-1 - n10])

    beyond1std = float(np.mean(np.abs(m - mu) > sd))

    return {
        "n_obs": float(n),
        "amplitude": _safe(amp),
        "weighted_mean": _safe(wm),
        "weighted_std": _safe(wstd),
        "stetson_k": _safe(stetson_k),
        "eta": _safe(eta),
        "kurtosis": _safe(kurt),
        "skew": _safe(skew),
        "slope_p10": _safe(slope_p10),
        "slope_p90": _safe(slope_p90),
        "beyond1std": _safe(beyond1std),
        "mad": _safe(mad),
        "median": _safe(p50),
        "iqr": _safe(iqr),
        "mean_err": _safe(float(np.mean(e))),
        "median_err": _safe(float(np.median(e))),
    }
