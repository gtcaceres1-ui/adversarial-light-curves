"""Detectability analysis, phase B: Lomb-Scargle period shift clean vs adv.

For each test curve, computes the LS periodogram of the clean and the
adversarial magnitude sequence on the *same* class-specific frequency grid
(grids follow compute_periods.py) and records the peak period and power.

Times come from runs/<dataset>_adv_test_times.npz (see cache_test_times.py);
alignment with the adversarial npz rows is guaranteed by construction and
checked via m_clean there.

Chunked execution: --start/--stop curve range writes a partial CSV under
outputs/detectability/ls_partials/; merge later.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle

# class-specific period ranges (days) and grid sizes — compute_periods.py
P_MIN = {"ecl": 0.2, "rrlyr": 0.2, "cep": 0.5, "lpv": 30.0}
P_MAX = {"ecl": 30.0, "rrlyr": 1.5, "cep": 60.0, "lpv": 600.0}
N_FREQ = {"ecl": 12000, "rrlyr": 16000, "cep": 8000, "lpv": 4000}
CLASSES = ["ecl", "rrlyr", "cep", "lpv"]
GAIA_CLASSES = ["ecl", "rrlyr", "cep", "lpv"]  # same int coding as ogle

MIN_POINTS = 20


def grids() -> dict:
    g = {}
    for c in CLASSES:
        f = np.linspace(1.0 / P_MAX[c], 1.0 / P_MIN[c], N_FREQ[c])
        g[c] = f
    return g


def peak(t, m, e, freq):
    ls = LombScargle(t, m, dy=e)
    power = ls.power(freq, method="fast", assume_regular_frequency=True)
    i = int(np.nanargmax(power))
    return 1.0 / freq[i], float(power[i])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--times", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=None)
    p.add_argument("--out-dir", default="outputs/detectability/ls_partials")
    args = p.parse_args()

    npz = Path(args.npz)
    dataset, model = npz.parent.parent.name, npz.parent.name
    frac = float(npz.stem.replace("adv_frac_", ""))

    d = np.load(npz)
    tt = np.load(args.times)
    assert bool(tt["m_check_ok"]), "times cache failed alignment check"
    m_c = d["m_clean"].astype(np.float64)
    m_a = d["m_adv"].astype(np.float64)
    e = d["e_raw"].astype(np.float64)
    nu = d["n_used"].astype(int)
    y = d["y"].astype(int)
    assert (nu == tt["n_used"]).all() and (y == tt["y"]).all(), "row mismatch"
    dt = tt["dt"]

    B = len(y)
    stop = min(args.stop if args.stop is not None else B, B)
    g = grids()

    rows = []
    for i in range(args.start, stop):
        n = nu[i]
        if n < MIN_POINTS:
            continue
        cls = CLASSES[y[i]]
        t = np.cumsum(dt[i, :n])
        freq = g[cls]
        ec = e[i, :n]
        pc, wc = peak(t, m_c[i, :n], ec, freq)
        pa, wa = peak(t, m_a[i, :n], ec, freq)
        rows.append({
            "dataset": dataset, "model": model, "fraction": frac,
            "curve": i, "y": y[i], "cls": cls, "n": n,
            "P_clean": pc, "P_adv": pa,
            "pow_clean": wc, "pow_adv": wa,
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (f"ls_{dataset}_{model}_f{frac:.2f}_"
                     f"{args.start:05d}_{stop:05d}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
