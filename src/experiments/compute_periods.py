"""Estimate Lomb-Scargle periods for every OGLE light curve in the catalog.

For each star: load lc, run LS on a class-specific frequency grid, save
(ID, P_LS, peak_power). The periods are used only for catalog-level quality
filtering and for phase-folded figures; no classifier consumes them.

Runs in parallel via ProcessPoolExecutor (no TensorFlow imports in the
worker, so start-up is fast).

Output: runs/periods_ogle.npz with arrays IDs (str), period (float), power (float).
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle

REPO = Path(__file__).resolve().parents[2]
CATALOG = REPO / "Data/OGLE-III/OGLE_dataset.dat"
LCS_DIR = REPO / "Data/OGLE-III/LCs"
OUT = REPO / "runs/periods_ogle.npz"

LABELS = ["ecl", "rrlyr", "cep", "lpv"]
# Class-specific frequency ranges (cycles/day). We use linear grids dense
# enough that the LS peak width (≈1/T_obs ≈ 3e-4 cyc/day for OGLE-III) is
# well sampled.
P_MIN = {"ecl": 0.2, "rrlyr": 0.2, "cep": 0.5, "lpv": 30.0}
P_MAX = {"ecl": 30.0, "rrlyr": 1.5, "cep": 60.0, "lpv": 600.0}
N_FREQ_PER_CLASS = {"ecl": 12000, "rrlyr": 16000, "cep": 8000, "lpv": 4000}


def _infer_label(path_str: str):
    p = str(path_str).lower().replace("\\", "/")
    for k in LABELS:
        if f"/{k}/" in p:
            return k
    return None


def _load_lc(path: Path):
    d = pd.read_csv(path, sep=r"\s+", comment="#", header=None,
                    names=["time", "magnitude", "error"])
    d = d.dropna().sort_values("time").drop_duplicates(subset="time", keep="first")
    return d


def _ls_one(args):
    ID, lc_file, lab = args
    try:
        if lab is None or lab not in P_MIN:
            return ID, float("nan"), 0.0
        d = _load_lc(Path(lc_file))
        if len(d) < 10:
            return ID, float("nan"), 0.0
        t = d["time"].to_numpy(float)
        m = d["magnitude"].to_numpy(float)
        # Remove very long-term linear trend per curve (helps LS for LPV)
        slope = np.polyfit(t, m, 1)
        m = m - (slope[0] * t + slope[1])
        f_min = 1.0 / P_MAX[lab]
        f_max = 1.0 / P_MIN[lab]
        nf = N_FREQ_PER_CLASS[lab]
        freq = np.linspace(f_min, f_max, nf)
        power = LombScargle(t, m).power(freq)
        k = int(np.argmax(power))
        return ID, float(1.0 / freq[k]), float(power[k])
    except Exception:
        return ID, float("nan"), 0.0


def main():
    # Restrict to the IDs used by the pipeline (subsample_balanced of 80k +
    # any extra target IDs needed for figures). Importing
    # build_adversarials_ogle would pull in TensorFlow, so we replicate
    # the small subsample step here using only pandas.
    SEED = 42; MAX_PER_CLASS = 20000
    cat = pd.read_csv(CATALOG)
    cat.columns = [c.strip() for c in cat.columns]
    cat["label4"] = cat["Path"].apply(_infer_label)
    cat["lc_file"] = cat["ID"].apply(lambda x: LCS_DIR / f"{x}.dat")
    cat = cat.dropna(subset=["label4"])
    cat["exists"] = cat["lc_file"].apply(lambda p: p.exists())
    cat = cat[cat["exists"]].reset_index(drop=True)
    print(f"Catalog: {len(cat)} curves with files on disk")
    parts = []
    for lab in LABELS:
        d = cat[cat["label4"] == lab]
        if len(d) > MAX_PER_CLASS:
            d = d.sample(MAX_PER_CLASS, random_state=SEED)
        parts.append(d)
    sub = pd.concat(parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    # Add specific extra targets we want figures for, even if dropped above.
    EXTRA_IDS = ["OGLE-LMC-RRLYR-00231"]
    extras = cat[cat["ID"].isin(EXTRA_IDS)]
    cat = pd.concat([sub, extras]).drop_duplicates(subset="ID").reset_index(drop=True)
    print(f"After subsample + extras: {len(cat)} curves to process")

    args = [(row["ID"], str(row["lc_file"]), row["label4"])
            for _, row in cat.iterrows()]
    n_workers = max(1, cpu_count() - 1)
    print(f"Using {n_workers} worker processes")
    t0 = time.time()
    results = {}
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for i, res in enumerate(ex.map(_ls_one, args, chunksize=32)):
            ID, P, pw = res
            results[ID] = (P, pw)
            if (i + 1) % 2000 == 0:
                rate = (i + 1) / (time.time() - t0 + 1e-9)
                eta = (len(args) - i - 1) / max(rate, 1e-9)
                print(f"  {i+1}/{len(args)}  rate={rate:.1f}/s  eta={eta/60:.1f} min")
    dt = time.time() - t0
    print(f"Done in {dt/60:.1f} min")
    IDs = np.array([row["ID"] for _, row in cat.iterrows()])
    P = np.array([results[row["ID"]][0] for _, row in cat.iterrows()])
    PW = np.array([results[row["ID"]][1] for _, row in cat.iterrows()])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, IDs=IDs, period=P, power=PW)
    print(f"Saved {OUT}")
    print(f"  finite periods: {int(np.isfinite(P).sum())} / {len(P)}")
    print(f"  median power: {np.median(PW):.3f}")


if __name__ == "__main__":
    main()
