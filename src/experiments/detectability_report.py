"""Merge detectability outputs (phases A and B) and emit the paper table.

Inputs:
  outputs/detectability/features_per_curve_*.csv  (phase A partials)
  outputs/detectability/ls_partials/ls_*.csv      (phase B partials, optional)

Outputs:
  outputs/detectability/features_per_curve.csv / features_summary.csv (merged)
  outputs/detectability/ls_periods.csv                       (merged, if any)
  outputs/detectability/detectability_summary.csv            (one row/config)
  outputs/detectability/table_detectability.tex              (A&A longtable-free)

Period-shift criterion: relative |P_adv - P_clean| / P_clean > 1e-3 counts as
a period change (well above LS grid resolution for every class grid).
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

FEATURES = ["std", "amp", "skew", "stetson_k", "eta_vn", "acf1_m", "chi2_med"]
REL_P_THRESH = 1e-3


def merge_features(out_dir: Path) -> pd.DataFrame:
    parts = sorted(glob.glob(str(out_dir / "features_per_curve_*.csv")))
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df = df.drop_duplicates(subset=["dataset", "model", "fraction", "curve"],
                            keep="last")
    df.to_csv(out_dir / "features_per_curve.csv", index=False)
    rows = []
    for (ds, mo, fr), g in df.groupby(["dataset", "model", "fraction"]):
        for f in FEATURES:
            c, a = g[f + "_clean"].values, g[f + "_adv"].values
            ok = np.isfinite(c) & np.isfinite(a)
            c, a = c[ok], a[ok]
            ks = stats.ks_2samp(c, a)
            try:
                w = stats.wilcoxon(a, c, zero_method="wilcox")
                w_p = w.pvalue
            except ValueError:
                w_p = np.nan
            rows.append({
                "dataset": ds, "model": mo, "fraction": fr, "feature": f,
                "n": int(ok.sum()),
                "med_clean": np.median(c), "med_adv": np.median(a),
                "med_shift": np.median(a - c),
                "ks_stat": ks.statistic, "ks_p": ks.pvalue,
                "wilcoxon_p": w_p,
            })
    s = pd.DataFrame(rows)
    s.to_csv(out_dir / "features_summary.csv", index=False)
    return s


def merge_periods(out_dir: Path) -> pd.DataFrame | None:
    parts = sorted(glob.glob(str(out_dir / "ls_partials" / "ls_*.csv")))
    if not parts:
        return None
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df = df.drop_duplicates(subset=["dataset", "model", "fraction", "curve"],
                            keep="last")
    df["rel_dP"] = np.abs(df["P_adv"] - df["P_clean"]) / df["P_clean"]
    df["changed"] = df["rel_dP"] > REL_P_THRESH
    df.to_csv(out_dir / "ls_periods.csv", index=False)
    return df


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "--"
    if p < 1e-99:
        return r"$<10^{-99}$"
    if p < 1e-3:
        exp = int(np.floor(np.log10(p)))
        return rf"$10^{{{exp}}}$"
    return f"{p:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    out_dir = Path(args.root) / "outputs" / "detectability"

    feats = merge_features(out_dir)
    pers = merge_periods(out_dir)

    rows = []
    for (ds, mo, fr), g in feats.groupby(["dataset", "model", "fraction"]):
        row = {"dataset": ds, "model": mo, "fraction": fr}
        for f in FEATURES:
            gg = g[g.feature == f].iloc[0]
            row[f"{f}_ks"] = gg.ks_stat
            row[f"{f}_ksp"] = gg.ks_p
        if pers is not None:
            sel = pers[(pers.dataset == ds) & (pers.model == mo)
                       & (pers.fraction == fr)]
            if len(sel):
                row["n_ls"] = len(sel)
                row["frac_P_changed"] = float(sel["changed"].mean())
                row["med_rel_dP_changed"] = (
                    float(sel.loc[sel.changed, "rel_dP"].median())
                    if sel.changed.any() else 0.0)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "detectability_summary.csv", index=False)

    # paper table: frac = 1.0 configs
    t = summary[summary.fraction == 1.0]
    lines = [
        r"\begin{table*}",
        r"\caption{Detectability of bounded adversarial perturbations at"
        r" $f=1.0$: two-sample KS statistic (clean vs. adversarial test"
        r" distributions) for standard variability features, and fraction of"
        r" curves whose Lomb--Scargle period changes by more than"
        r" $10^{-3}$ in relative terms.}",
        r"\label{tab:detectability}",
        r"\centering",
        r"\begin{tabular}{llcccccc}",
        r"\hline\hline",
        r"Survey & Surrogate & $\sigma$ & Amp. & $K$ & $\eta_{\rm vN}$ &"
        r" ACF$_1$ & $P_{\rm LS}$ changed \\",
        r"\hline",
    ]
    for _, r in t.iterrows():
        pl = (f"{100*r['frac_P_changed']:.1f}\\%"
              if "frac_P_changed" in r and np.isfinite(
                  r.get("frac_P_changed", np.nan)) else "--")
        lines.append(
            f"{r.dataset.upper()} & {r.model.replace('_', ' ')} & "
            f"{r.std_ks:.3f} ({fmt_p(r.std_ksp)}) & "
            f"{r.amp_ks:.3f} ({fmt_p(r.amp_ksp)}) & "
            f"{r.stetson_k_ks:.3f} ({fmt_p(r.stetson_k_ksp)}) & "
            f"{r.eta_vn_ks:.3f} ({fmt_p(r.eta_vn_ksp)}) & "
            f"{r.acf1_m_ks:.3f} ({fmt_p(r.acf1_m_ksp)}) & "
            f"{pl} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}", ""]
    (out_dir / "table_detectability.tex").write_text("\n".join(lines))
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nLaTeX -> {out_dir/'table_detectability.tex'}")


if __name__ == "__main__":
    main()
