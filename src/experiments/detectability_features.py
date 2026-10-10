"""Detectability analysis, phase A: time-independent features clean vs adv.

For every *valid* adversarial npz (bounded attack), computes per curve and per
version (clean / adv) a set of standard variability statistics that a survey
pipeline could evaluate without knowing the clean curve:

  std        sample std of magnitudes
  amp        robust amplitude (p95 - p5)
  skew       magnitude skewness
  stetson_k  Stetson K (uses reported errors e_raw)
  eta_vn     von Neumann ratio  sum(diff(m)^2) / ((n-1) var(m))
  acf1_m     lag-1 autocorrelation of the magnitude sequence
  chi2_med   mean of ((m - median(m)) / e)^2

Emits:
  outputs/detectability/features_per_curve.csv
  outputs/detectability/features_summary.csv   (per config: median clean, median
      adv, median paired shift, KS statistic + p, fraction |z|>3 per feature)

Configurations produced by an earlier, unbounded version of the attack are
excluded via an explicit denylist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DENYLIST = {
    ("ogle", "cnn", 0.2), ("ogle", "cnn", 0.4), ("ogle", "cnn", 0.6),
}
FEATURES = ["std", "amp", "skew", "stetson_k", "eta_vn", "acf1_m", "chi2_med"]


def feature_block(m: np.ndarray, e: np.ndarray, valid: np.ndarray) -> dict:
    """Vectorized per-row features; invalid entries masked as NaN."""
    x = np.where(valid, m, np.nan)
    er = np.where(valid, e, np.nan)
    n = valid.sum(axis=1).astype(float)

    mean = np.nanmean(x, axis=1)
    sd = np.nanstd(x, axis=1, ddof=1)
    p95 = np.nanpercentile(x, 95, axis=1)
    p5 = np.nanpercentile(x, 5, axis=1)
    med = np.nanmedian(x, axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        z = (x - mean[:, None]) / sd[:, None]
        skew = np.nanmean(z ** 3, axis=1)

        # Stetson K
        d = np.sqrt(n / (n - 1.0))[:, None] * (x - mean[:, None]) / er
        k_num = np.nanmean(np.abs(d), axis=1)
        k_den = np.sqrt(np.nanmean(d ** 2, axis=1))
        stetson_k = k_num / k_den

        # von Neumann ratio over consecutive valid pairs
        pair = valid[:, 1:] & valid[:, :-1]
        dm = np.where(pair, np.diff(np.where(valid, m, 0.0), axis=1), np.nan)
        eta = np.nansum(dm ** 2, axis=1) / (
            (pair.sum(axis=1)) * np.nanvar(x, axis=1, ddof=1)
        )

        # lag-1 ACF of m
        a = x[:, :-1]
        b = x[:, 1:]
        ok = ~np.isnan(a) & ~np.isnan(b)
        a = np.where(ok, a, np.nan)
        b = np.where(ok, b, np.nan)
        cov = np.nanmean(
            (a - np.nanmean(a, axis=1)[:, None])
            * (b - np.nanmean(b, axis=1)[:, None]), axis=1)
        acf1 = cov / (np.nanstd(a, axis=1) * np.nanstd(b, axis=1))

        chi2 = np.nanmean(((x - med[:, None]) / er) ** 2, axis=1)

    return {
        "std": sd, "amp": p95 - p5, "skew": skew, "stetson_k": stetson_k,
        "eta_vn": eta, "acf1_m": acf1, "chi2_med": chi2,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--skip", default="ogle_smoke")
    p.add_argument("--only", default=None, help="dataset/model filter")
    p.add_argument("--frac", default=None)
    args = p.parse_args()

    root = Path(args.root)
    out_dir = root / "outputs" / "detectability"
    out_dir.mkdir(parents=True, exist_ok=True)
    skip = set(s.strip() for s in args.skip.split(",") if s.strip())

    frames = []
    for npz in sorted((root / "outputs" / "adversarial").glob(
            "*/*/adv_frac_*.npz")):
        dataset, model = npz.parent.parent.name, npz.parent.name
        frac = float(npz.stem.replace("adv_frac_", ""))
        if dataset in skip or frac == 0.0:
            continue
        if (dataset, model, round(frac, 2)) in DENYLIST:
            print(f"SKIP contaminated {dataset}/{model} f={frac}")
            continue
        if args.only and f"{dataset}/{model}" != args.only:
            continue
        if args.frac is not None and abs(frac - float(args.frac)) > 1e-6:
            continue

        d = np.load(npz)
        m_c = d["m_clean"].astype(np.float64)
        m_a = d["m_adv"].astype(np.float64)
        e = d["e_raw"].astype(np.float64)
        nu = d["n_used"].astype(int)
        y = d["y"].astype(int)
        L = m_c.shape[1]
        valid = np.arange(L)[None, :] < nu[:, None]

        fc = feature_block(m_c, e, valid)
        fa = feature_block(m_a, e, valid)
        row = {"dataset": dataset, "model": model, "fraction": frac,
               "curve": np.arange(len(y)), "y": y, "n_valid": nu}
        for f in FEATURES:
            row[f + "_clean"] = fc[f]
            row[f + "_adv"] = fa[f]
        frames.append(pd.DataFrame(row))
        print(f"done {dataset}/{model} f={frac:.2f} n={len(y)}")

    per_curve = pd.concat(frames, ignore_index=True)
    tag = ""
    if args.only:
        tag += "_" + args.only.replace("/", "_")
    if args.frac is not None:
        tag += f"_f{float(args.frac):.2f}"
    per_curve.to_csv(out_dir / f"features_per_curve{tag}.csv", index=False)

    rows = []
    for (ds, mo, fr), g in per_curve.groupby(["dataset", "model", "fraction"]):
        for f in FEATURES:
            c, a = g[f + "_clean"].values, g[f + "_adv"].values
            ok = np.isfinite(c) & np.isfinite(a)
            c, a = c[ok], a[ok]
            ks = stats.ks_2samp(c, a)
            shift = a - c
            iqr_c = np.subtract(*np.percentile(c, [75, 25]))
            rows.append({
                "dataset": ds, "model": mo, "fraction": fr, "feature": f,
                "n": ok.sum(),
                "med_clean": np.median(c), "med_adv": np.median(a),
                "med_shift": np.median(shift),
                "shift_in_iqr": np.median(shift) / iqr_c if iqr_c > 0 else np.nan,
                "ks_stat": ks.statistic, "ks_p": ks.pvalue,
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / f"features_summary{tag}.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
