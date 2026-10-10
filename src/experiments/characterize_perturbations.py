"""Characterize the temporal structure of adversarial perturbations.

Measures whether the perturbation delta = m_adv - m_clean is temporally
correlated (smooth) or white-noise-like. Photometric noise is uncorrelated
point to point, so a *correlated* delta is structurally distinguishable from
noise; this script quantifies that.

Per curve (valid segment, excluding the fixed anchor at index 0):
  * lag-1..5 autocorrelation of delta (index-lag, i.e. adjacent observations,
    matching how the smoothness penalty in photometric_pgd.py is defined);
  * permutation null: same marginal distribution, shuffled order;
  * curvature ratio: sum((d2 delta)^2) / sum(delta^2), observed vs shuffled
    (d2 = second difference, the exact quantity penalized by smooth_lambda);
  * sign agreement between adjacent perturbed points (0.5 under whiteness);
  * budget usage |delta|/(k*e): median and saturation fraction (>0.99).

Outputs (under outputs/perturbation_stats/):
  delta_structure_per_curve.csv   one row per curve per (dataset, model, frac)
  delta_structure_summary.csv     aggregated medians/IQRs + null comparison
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

K_BY_DATASET = {"ogle": 1.0, "gaia": 1.0, "wise": 1.0, "ogle_smoke": 0.5}
N_SHUFFLES = 20
MAX_LAG = 5


def k_for_dir(npz_dir: Path, dataset: str) -> float:
    """Attack budget k actually used, from the run manifest if available."""
    mf = npz_dir / "manifest.json"
    if mf.exists():
        try:
            summary = json.loads(mf.read_text()).get("summary", [])
            if summary and "k" in summary[0]:
                return float(summary[0]["k"])
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
    return K_BY_DATASET.get(dataset, 1.0)


def masked_acf_lag(x: np.ndarray, valid: np.ndarray, lag: int) -> np.ndarray:
    """Per-row Pearson correlation between (x_t, x_{t+lag}) over valid pairs."""
    a = np.where(valid, x, np.nan)[:, :-lag]
    b = np.where(valid, x, np.nan)[:, lag:]
    pair = ~np.isnan(a) & ~np.isnan(b)
    a = np.where(pair, a, np.nan)
    b = np.where(pair, b, np.nan)
    n = pair.sum(axis=1).astype(float)
    with np.errstate(invalid="ignore"):
        ma = np.nanmean(a, axis=1)
        mb = np.nanmean(b, axis=1)
        cov = np.nanmean((a - ma[:, None]) * (b - mb[:, None]), axis=1)
        sa = np.nanstd(a, axis=1)
        sb = np.nanstd(b, axis=1)
        r = cov / (sa * sb)
    r[(n < 8) | ~np.isfinite(r)] = np.nan
    return r


def shuffle_within_valid(x: np.ndarray, valid: np.ndarray,
                         rng: np.random.Generator) -> np.ndarray:
    """Independently permute each row's valid entries among themselves."""
    B, L = x.shape
    keys = rng.random(x.shape)
    keys[~valid] = np.inf  # keep invalid slots parked at the end
    order = np.argsort(keys, axis=1)
    shuffled_vals = np.take_along_axis(x, order, axis=1)
    # valid positions per row, original order first
    pos = np.argsort(np.where(valid, 0, 1), axis=1, kind="stable")
    nval = valid.sum(axis=1)
    col_mask = np.arange(L)[None, :] < nval[:, None]
    rows = np.arange(B)[:, None]
    out = np.zeros_like(x)
    out.ravel()[((rows * L) + pos)[col_mask]] = shuffled_vals[col_mask]
    return out


def curvature_ratio(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """sum((second difference)^2) / sum(x^2) over rows, valid triples only."""
    v3 = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]
    d2 = x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]
    num = np.where(v3, d2, 0.0)
    num = (num ** 2).sum(axis=1)
    den = np.where(valid, x, 0.0)
    den = (den ** 2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / den
    r[(den <= 0) | (v3.sum(axis=1) < 8)] = np.nan
    return r


def sign_agreement(x: np.ndarray, both: np.ndarray) -> np.ndarray:
    """P(adjacent same-sign) among pairs where both points are perturbed."""
    s = np.sign(x)
    agree = (s[:, :-1] == s[:, 1:]) & both
    n = both.sum(axis=1).astype(float)
    with np.errstate(invalid="ignore"):
        p = agree.sum(axis=1) / n
    p[n < 8] = np.nan
    return p


def analyze_npz(path: Path, dataset: str, model: str, frac: float,
                rng: np.random.Generator, k: float | None = None
                ) -> pd.DataFrame | None:
    d = np.load(path)
    m_clean = d["m_clean"].astype(np.float64)
    m_adv = d["m_adv"].astype(np.float64)
    e_raw = d["e_raw"].astype(np.float64)
    n_used = d["n_used"].astype(int)
    pmask = d["perturb_mask"].astype(bool)
    y = d["y"].astype(int)

    B, L = m_clean.shape
    delta = m_adv - m_clean
    valid = (np.arange(L)[None, :] < n_used[:, None])
    valid[:, 0] = False  # anchor is held fixed by construction
    if not pmask.any():
        return None

    if k is None:
        k = K_BY_DATASET.get(dataset, 1.0)
    # budget usage on perturbed points
    with np.errstate(divide="ignore", invalid="ignore"):
        usage = np.abs(delta) / (k * np.abs(e_raw))
    usage = np.where(pmask & valid & (e_raw > 0), usage, np.nan)
    budget_med = np.nanmedian(usage, axis=1)
    sat_frac = np.nanmean(usage > 0.99, axis=1)

    # observed ACF
    acf = {f"acf{l}": masked_acf_lag(delta, valid, l) for l in range(1, MAX_LAG + 1)}
    curv = curvature_ratio(delta, valid)

    # permutation null (same marginals, shuffled order)
    null1 = np.zeros((N_SHUFFLES, B))
    nullc = np.zeros((N_SHUFFLES, B))
    for s in range(N_SHUFFLES):
        xs = shuffle_within_valid(delta, valid, rng)
        null1[s] = masked_acf_lag(xs, valid, 1)
        nullc[s] = curvature_ratio(xs, valid)
    null1_mean = np.nanmean(null1, axis=0)
    null1_std = np.nanstd(null1, axis=0)
    nullc_med = np.nanmedian(nullc, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        z1 = (acf["acf1"] - null1_mean) / null1_std

    both = pmask[:, :-1] & pmask[:, 1:] & valid[:, :-1] & valid[:, 1:]
    sagree = sign_agreement(delta, both)

    df = pd.DataFrame({
        "dataset": dataset, "model": model, "fraction": frac,
        "curve": np.arange(B), "y": y, "n_valid": valid.sum(axis=1),
        "n_perturbed": (pmask & valid).sum(axis=1),
        **acf,
        "acf1_null_mean": null1_mean, "acf1_null_std": null1_std,
        "acf1_z": z1,
        "curv_ratio": curv, "curv_ratio_null": nullc_med,
        "sign_agree_adj": sagree,
        "budget_med": budget_med, "sat_frac": sat_frac,
    })
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".", help="repo root")
    p.add_argument("--skip", default="ogle_smoke",
                   help="comma-separated dataset names to skip")
    p.add_argument("--only", default=None,
                   help="restrict to 'dataset/model' (e.g. ogle/cnn)")
    p.add_argument("--frac", default=None,
                   help="restrict to one fraction (e.g. 0.60)")
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    root = Path(args.root)
    adv_root = root / "outputs" / "adversarial"
    out_dir = root / "outputs" / "perturbation_stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    skip = set(s.strip() for s in args.skip.split(",") if s.strip())
    rng = np.random.default_rng(args.seed)

    frames = []
    for npz in sorted(adv_root.glob("*/*/adv_frac_*.npz")):
        dataset, model = npz.parent.parent.name, npz.parent.name
        if dataset in skip:
            continue
        if args.only and f"{dataset}/{model}" != args.only:
            continue
        frac = float(npz.stem.replace("adv_frac_", ""))
        if frac == 0.0:
            continue
        if args.frac is not None and abs(frac - float(args.frac)) > 1e-6:
            continue
        k = k_for_dir(npz.parent, dataset)
        df = analyze_npz(npz, dataset, model, frac, rng, k=k)
        if df is not None:
            df["k"] = k
            frames.append(df)
            print(f"done {dataset}/{model} frac={frac:.2f} k={k}  n={len(df)}")

    tag = f"_{args.only.replace('/', '_')}" if args.only else ""
    if args.frac is not None:
        tag += f"_f{float(args.frac):.2f}"
    per_curve = pd.concat(frames, ignore_index=True)
    per_curve.to_csv(out_dir / f"delta_structure_per_curve{tag}.csv",
                     index=False)

    g = per_curve.groupby(["dataset", "model", "fraction"])
    summary = g.agg(
        n_curves=("curve", "size"),
        acf1_med=("acf1", "median"),
        acf1_q25=("acf1", lambda s: s.quantile(0.25)),
        acf1_q75=("acf1", lambda s: s.quantile(0.75)),
        acf2_med=("acf2", "median"),
        acf3_med=("acf3", "median"),
        acf1_null_med=("acf1_null_mean", "median"),
        frac_z_gt3=("acf1_z", lambda s: float(np.mean(s > 3.0))),
        curv_ratio_med=("curv_ratio", "median"),
        curv_ratio_null_med=("curv_ratio_null", "median"),
        sign_agree_med=("sign_agree_adj", "median"),
        budget_med=("budget_med", "median"),
        sat_frac_med=("sat_frac", "median"),
    ).reset_index()
    summary["curv_suppression"] = (
        summary["curv_ratio_med"] / summary["curv_ratio_null_med"]
    )
    summary.to_csv(out_dir / f"delta_structure_summary{tag}.csv", index=False)

    print("\n=== summary ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    (out_dir / f"delta_structure_summary{tag}.json").write_text(
        json.dumps(json.loads(summary.to_json(orient="records")), indent=2)
    )


if __name__ == "__main__":
    main()
