"""Validation metrics for adversarial light curves.

Given clean and adversarial raw magnitude tensors, compute the normalized
displacement delta_i = (m_adv_i - m_i) / e_i used to populate the perturbation
validity table in the paper.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def perturbation_stats(
    m_clean: np.ndarray,
    m_adv: np.ndarray,
    e_raw: np.ndarray,
    perturb_mask: np.ndarray,
    k: float,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Summary statistics of |delta_i| over actually perturbed observations."""
    m_clean = np.asarray(m_clean, dtype=np.float64)
    m_adv = np.asarray(m_adv, dtype=np.float64)
    e_raw = np.asarray(e_raw, dtype=np.float64)
    mask = np.asarray(perturb_mask, dtype=bool) & (e_raw > 0)

    if mask.sum() == 0:
        return {
            "n_perturbed": 0,
            "mean_abs_delta": 0.0,
            "median_abs_delta": 0.0,
            "p95_abs_delta": 0.0,
            "max_abs_delta": 0.0,
            "bound_k": float(k),
            "violations": 0,
        }

    delta = (m_adv[mask] - m_clean[mask]) / (e_raw[mask] + eps)
    abs_d = np.abs(delta)
    violations = int(np.sum(abs_d > k + 1e-6))
    return {
        "n_perturbed": int(mask.sum()),
        "mean_abs_delta": float(np.mean(abs_d)),
        "median_abs_delta": float(np.median(abs_d)),
        "p95_abs_delta": float(np.percentile(abs_d, 95)),
        "max_abs_delta": float(np.max(abs_d)),
        "bound_k": float(k),
        "violations": violations,
    }
