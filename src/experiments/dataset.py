"""Dataset assembly used by every experiment script.

Loads a survey config, materializes the catalog + light curves, builds Δt/Δm
tensors with shared standardization, and produces stratified train/val/test
splits. Returned bundle is intentionally plain so it can be cached or pickled.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.data.loading import (
    iterate_lightcurves,
    load_catalog,
    subsample_catalog,
)
from src.data.preprocessing import build_arrays, log1p_dt, standardize
from src.data.splits import stratified_split


@dataclass
class DatasetBundle:
    name: str
    classes: List[str]
    class_to_int: Dict[str, int]
    int_to_class: Dict[int, str]
    seq_len: int
    # Raw arrays (Δt, Δm, sigma_dm) — un-standardized
    dt: np.ndarray
    dm: np.ndarray
    sigma_dm: np.ndarray
    m_raw: np.ndarray
    e_raw: np.ndarray
    n_used: np.ndarray
    y: np.ndarray
    object_id: np.ndarray
    # Standardized inputs aligned with the raw arrays
    dt_s: np.ndarray
    dm_s: np.ndarray
    mu_dt: float
    sd_dt: float
    mu_dm: float
    sd_dm: float
    # Splits (indices into the arrays above)
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray

    @property
    def n_classes(self) -> int:
        return len(self.classes)


def build_dataset(cfg: Dict[str, Any], repo_root: Path, seed: int) -> DatasetBundle:
    ds = cfg["dataset"]
    cat = load_catalog(ds, repo_root)
    if "max_per_class" in ds and ds["max_per_class"]:
        cat = subsample_catalog(cat, int(ds["max_per_class"]), seed=seed)

    lcs, labels, ids = iterate_lightcurves(
        cat, file_format=ds["format"], columns=ds["columns"], min_obs=int(ds.get("min_obs", 30)),
    )
    if not lcs:
        raise RuntimeError(f"No light curves loaded for dataset {ds['name']}")

    classes = list(ds["classes"])
    class_to_int = {c: i for i, c in enumerate(classes)}
    int_to_class = {i: c for c, i in class_to_int.items()}

    arrs = build_arrays(lcs, labels, ids, class_to_int, int(ds["seq_len"]))

    dt_log = log1p_dt(arrs["dt"])
    dt_s, mu_dt, sd_dt = standardize(dt_log)
    dm_s, mu_dm, sd_dm = standardize(arrs["dm"])

    sp = cfg.get("split", {})
    idx_tr, idx_val, idx_te = stratified_split(
        n=len(arrs["y"]), y=arrs["y"],
        test_size=float(sp.get("test_size", 0.2)),
        val_size=float(sp.get("val_size", 0.15)),
        seed=int(sp.get("random_state", seed)),
    )

    return DatasetBundle(
        name=ds["name"],
        classes=classes, class_to_int=class_to_int, int_to_class=int_to_class,
        seq_len=int(ds["seq_len"]),
        dt=arrs["dt"], dm=arrs["dm"], sigma_dm=arrs["sigma_dm"],
        m_raw=arrs["m_raw"], e_raw=arrs["e_raw"], n_used=arrs["n_used"],
        y=arrs["y"], object_id=arrs["object_id"],
        dt_s=dt_s, dm_s=dm_s,
        mu_dt=float(mu_dt), sd_dt=float(sd_dt),
        mu_dm=float(mu_dm), sd_dm=float(sd_dm),
        idx_train=idx_tr, idx_val=idx_val, idx_test=idx_te,
    )
