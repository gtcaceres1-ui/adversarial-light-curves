"""Lomb-Scargle period estimation for any survey in survey_config.

Restricts to the SEED=42 4-class subsample so we only pay for the curves
the rest of the pipeline will use. Saves to runs/<survey>_periods.npz.
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle

from src.data.survey_config import (
    LABELS_COMMON, MAX_PER_CLASS, PERIOD_RANGES_DEFAULT, SEED, get_cfg,
)
from src.data.survey_loader import build_catalog, subsample_balanced, load_lightcurve

REPO = Path(__file__).resolve().parents[2]

N_FREQ_PER_CLASS = {"rrlyr": 16000, "cep": 8000, "lpv": 4000, "dsct": 12000}


# Workers run as separate processes — make sure they only use lightweight
# pandas/numpy operations (no tensorflow import at module top).

_SURVEY_CACHE = None  # populated per process via initializer


def _init_worker(survey: str):
    global _SURVEY_CACHE
    _SURVEY_CACHE = survey


def _ls_one(args):
    ID, lc_file, lab, period_ranges = args
    try:
        if lab is None or lab not in period_ranges:
            return ID, float("nan"), 0.0
        d = load_lightcurve(Path(lc_file), _SURVEY_CACHE)
        if len(d) < 10:
            return ID, float("nan"), 0.0
        t = d["time"].to_numpy(float)
        m = d["magnitude"].to_numpy(float)
        slope = np.polyfit(t, m, 1)
        m = m - (slope[0] * t + slope[1])
        p_min, p_max = period_ranges[lab]
        f_min, f_max = 1.0 / p_max, 1.0 / p_min
        nf = N_FREQ_PER_CLASS[lab]
        freq = np.linspace(f_min, f_max, nf)
        power = LombScargle(t, m).power(freq)
        k = int(np.argmax(power))
        return ID, float(1.0 / freq[k]), float(power[k])
    except Exception:
        return ID, float("nan"), 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--survey", required=True, choices=["ogle", "gaia", "wise"])
    args = ap.parse_args()
    survey = args.survey
    cfg = get_cfg(survey)
    out = REPO / f"runs/{survey}_periods.npz"
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Loading catalog ({survey})")
    cat = build_catalog(survey)
    sub = subsample_balanced(cat)
    print(f"  subsample: {len(sub)}  classes={sub['label4'].value_counts().to_dict()}")

    args_list = [(row["ID"], str(row["lc_file"]), row["label4"],
                  cfg["period_ranges"]) for _, row in sub.iterrows()]
    n_workers = max(1, cpu_count() - 1)
    print(f"[2/3] Lomb-Scargle on {len(args_list)} curves "
          f"({n_workers} workers)")

    t0 = time.time()
    results = {}
    with ProcessPoolExecutor(max_workers=n_workers,
                              initializer=_init_worker,
                              initargs=(survey,)) as ex:
        for i, res in enumerate(ex.map(_ls_one, args_list, chunksize=32)):
            ID, P, pw = res
            results[ID] = (P, pw)
            if (i + 1) % 2000 == 0:
                rate = (i + 1) / (time.time() - t0)
                eta = (len(args_list) - i - 1) / max(rate, 1e-9)
                print(f"  {i+1}/{len(args_list)}  rate={rate:.1f}/s  eta={eta/60:.1f} min")
    dt = time.time() - t0
    print(f"  done in {dt/60:.1f} min")

    IDs = np.array([row["ID"] for _, row in sub.iterrows()])
    P = np.array([results.get(ID, (np.nan, 0.0))[0] for ID in IDs])
    PW = np.array([results.get(ID, (np.nan, 0.0))[1] for ID in IDs])
    np.savez(out, IDs=IDs, period=P, power=PW)
    print(f"[3/3] Saved {out}  "
          f"finite={int(np.isfinite(P).sum())}/{len(P)}  "
          f"median_power={float(np.nanmedian(PW)):.3f}")


if __name__ == "__main__":
    main()
