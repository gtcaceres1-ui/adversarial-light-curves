"""Smoke tests for the data pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.preprocessing import lightcurve_to_deltas, log1p_dt, standardize


def test_lightcurve_to_deltas_basic():
    lc = pd.DataFrame({
        "time": [0.0, 1.0, 3.0, 6.0],
        "magnitude": [18.0, 18.1, 18.05, 17.9],
        "magnitude_error": [0.01, 0.02, 0.03, 0.04],
    })
    dt, dm, sdm, m_raw, e_raw, nu = lightcurve_to_deltas(lc, seq_len=8)
    assert nu == 4
    assert dt.shape == (8,)
    assert dt[0] == 0.0
    assert dt[1] == 1.0
    assert np.isclose(dm[2], -0.05, atol=1e-6)
    # propagated sigma_dm at index 1 = sqrt(0.02^2 + 0.01^2)
    assert np.isclose(sdm[1], np.hypot(0.02, 0.01), atol=1e-6)
    assert np.isclose(m_raw[0], 18.0)
    assert m_raw[5] == 0.0  # padding


def test_standardize_ignores_padding_zeros():
    arr = np.array([[1.0, 2.0, 3.0, 0.0, 0.0]])
    z, mu, sd = standardize(arr)
    assert np.isclose(mu, 2.0, atol=1e-6)
    assert sd > 0
    # Padding should be remapped via z = (0 - mu)/sd
    assert z[0, 3] == pytest_approx(-mu / sd)


def pytest_approx(value, tol=1e-6):
    class _A:
        def __eq__(self, other):
            return abs(other - value) < tol
    return _A()


def test_log1p_dt_nonnegative():
    dt = np.array([-1.0, 0.0, 1.0, 5.0])
    out = log1p_dt(dt)
    assert (out >= 0).all()
