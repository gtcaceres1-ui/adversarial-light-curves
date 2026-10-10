"""Validate both evaluated tensors and saved curves against the threat model."""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/metrics/perturbation_bounds.csv"
K = 0.5
SURVEYS = {
    "ogle": ("runs/adversarial_dataset", "Data/OGLE-III/LCs"),
    "gaia": ("runs/gaia_adversarial_dataset", "Data/Gaia/LCs"),
    "wise": ("runs/wise_adversarial_dataset", "Data/WISE/LCs"),
}


def read_curve(path: Path):
    for delimiter in (None, ","):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                arr = np.genfromtxt(
                    path,
                    comments="#",
                    delimiter=delimiter,
                    usecols=(0, 1, 2),
                    dtype=float,
                    invalid_raise=False,
                )
        except Exception:
            continue
        if arr.size == 0:
            continue
        arr = np.atleast_2d(arr)
        arr = arr[np.isfinite(arr).all(axis=1)]
        if len(arr) == 0:
            continue
        arr = arr[np.argsort(arr[:, 0])]
        _, unique_idx = np.unique(np.round(arr[:, 0], 6), return_index=True)
        return arr[np.sort(unique_idx)]
    return None


def validate_tensor(survey, adv_root):
    sp = np.load(adv_root / "test_split.npz", allow_pickle=True)
    delta = sp["delta_mag_std_te"] * float(sp["sd_m"])
    errors = sp["E_te"].squeeze(-1)
    n_sel = sp["n_sel_te"].astype(np.int32)
    valid_points = np.arange(delta.shape[1])[None, :] < n_sel[:, None]
    ratio = np.zeros_like(delta, dtype=np.float64)
    good = valid_points & (errors > 0)
    ratio[good] = np.abs(delta[good]) / errors[good]

    expected = sp["X2_te"].copy()
    dm_delta = np.zeros_like(delta)
    dm_delta[:, 1:] = np.diff(delta, axis=1)
    dm_valid = valid_points.copy()
    dm_valid[:, 0] = False
    expected[..., 0][dm_valid] += (
        dm_delta[dm_valid] / float(sp["sd_m"])
    ).astype(np.float32)
    consistency = np.max(np.abs(expected - sp["X2_te_adv"]))
    values = ratio[good]
    return {
        "survey": survey,
        "representation": "evaluated_tensor",
        "n_obs": int(good.sum()),
        "mean_abs_delta": float(values.mean()),
        "median_abs_delta": float(np.median(values)),
        "p95_abs_delta": float(np.percentile(values, 95)),
        "max_abs_delta": float(values.max()),
        "bound_k": K,
        "violations": int((values > K + 1e-6).sum()),
        "max_tensor_inconsistency": float(consistency),
    }


def validate_files(survey, adv_root, clean_root):
    ratios = []
    for adv_path in sorted((adv_root / "adversarial_lcs").glob("*.dat")):
        identifier = adv_path.stem.removesuffix("_adv")
        clean = read_curve(clean_root / f"{identifier}.dat")
        adv = read_curve(adv_path)
        if clean is None or adv is None:
            continue
        clean_key = np.round(clean[:, 0], 6)
        adv_key = np.round(adv[:, 0], 6)
        _, adv_idx, clean_idx = np.intersect1d(
            adv_key, clean_key, assume_unique=True, return_indices=True
        )
        if len(adv_idx) == 0:
            continue
        error = clean[clean_idx, 2]
        good = error > 0
        ratios.append(
            np.abs(adv[adv_idx, 1][good] - clean[clean_idx, 1][good])
            / error[good]
        )
    if not ratios:
        raise RuntimeError(f"no saved curves could be validated for {survey}")
    values = np.concatenate(ratios)
    return {
        "survey": survey,
        "representation": "saved_full_curve",
        "n_obs": int(len(values)),
        "mean_abs_delta": float(values.mean()),
        "median_abs_delta": float(np.median(values)),
        "p95_abs_delta": float(np.percentile(values, 95)),
        "max_abs_delta": float(values.max()),
        "bound_k": K,
        "violations": int((values > K + 1e-6).sum()),
        "max_tensor_inconsistency": np.nan,
    }


def main():
    rows = []
    for survey, (adv_rel, clean_rel) in SURVEYS.items():
        adv_root = REPO / adv_rel
        clean_root = REPO / clean_rel
        rows.append(validate_tensor(survey, adv_root))
        rows.append(validate_files(survey, adv_root, clean_root))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    if any(row["violations"] for row in rows):
        raise RuntimeError("photometric bound violations detected")
    if any(
        row["representation"] == "evaluated_tensor"
        and row["max_tensor_inconsistency"] > 1e-5
        for row in rows
    ):
        raise RuntimeError("evaluated tensors are not self-consistent")


if __name__ == "__main__":
    main()
