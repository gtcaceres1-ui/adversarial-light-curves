"""Latent-space stability metrics under perturbation."""
from __future__ import annotations

from typing import Dict

import numpy as np


def latent_shift(z_clean: np.ndarray, z_adv: np.ndarray) -> Dict[str, float]:
    """Per-batch L2 / cosine shifts between clean and adversarial latent means."""
    diff = z_adv - z_clean
    l2 = np.linalg.norm(diff, axis=1)
    nc = np.linalg.norm(z_clean, axis=1) + 1e-12
    na = np.linalg.norm(z_adv, axis=1) + 1e-12
    cos = np.sum(z_clean * z_adv, axis=1) / (nc * na)
    return {
        "mean_l2_shift": float(np.mean(l2)),
        "median_l2_shift": float(np.median(l2)),
        "p95_l2_shift": float(np.percentile(l2, 95)),
        "mean_cosine": float(np.mean(cos)),
        "median_cosine": float(np.median(cos)),
    }


def per_class_latent_shift(z_clean: np.ndarray, z_adv: np.ndarray, y: np.ndarray) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for c in np.unique(y):
        m = y == c
        if m.any():
            out[int(c)] = float(np.mean(np.linalg.norm(z_adv[m] - z_clean[m], axis=1)))
    return out
