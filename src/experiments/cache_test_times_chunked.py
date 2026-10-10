"""Chunk-safe version of cache_test_times.py.

Replicates build_dataset() deterministically but loads light-curve files in
row ranges so each invocation stays short. Steps:

  --chunk START STOP   process catalog rows [START, STOP) -> partial npz
  --assemble           merge partials, apply stratified split, verify against
                       the adversarial npz m_clean, write the times cache

The catalog order (load_catalog + subsample_catalog with the config seed) is
rebuilt identically on every call; partials store a `kept` flag per catalog
row so skipped curves (load error / < min_obs) reproduce iterate_lightcurves.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import os

import pandas as pd

from src.data.loading import (
    _infer_label_ogle_path,
    load_lightcurve,
    subsample_catalog,
)
from src.data.preprocessing import lightcurve_to_deltas
from src.data.splits import stratified_split
from src.utils.config import load_config
from src.utils.paths import REPO_ROOT


def fast_load_catalog(ds: dict) -> pd.DataFrame:
    """load_catalog() clone without the per-row Path.exists() filter.

    Assumes every catalog row has its LC file (true for OGLE-III here:
    runs/periods_ogle.npz covers all 67,837 IDs). Missing files would surface
    as kept=False during chunk loading, and any ordering mismatch is caught
    by the final m_clean alignment check in --assemble.
    """
    catalog_path = REPO_ROOT / ds["catalog"]
    lcs_dir = REPO_ROOT / ds["lcs_dir"]
    cat = pd.read_csv(catalog_path)
    cat.columns = [c.strip() for c in cat.columns]
    if ds["label_inference"] != "ogle_path_token":
        raise NotImplementedError("only ogle_path_token supported here")
    cat["label"] = cat[ds["catalog_path_col"]].map(_infer_label_ogle_path)
    cat = cat.dropna(subset=["label"])
    cat = cat[cat["label"].isin(ds["classes"])].copy()
    cat["object_id"] = cat[ds["catalog_id_col"]].astype(str)
    cat["lc_path"] = cat["object_id"].map(lambda i: lcs_dir / f"{i}.dat")
    return cat[["object_id", "label", "lc_path"]].reset_index(drop=True)


def ordered_catalog(cfg: dict, seed: int):
    ds = cfg["dataset"]
    cat = fast_load_catalog(ds)
    if "max_per_class" in ds and ds["max_per_class"]:
        cat = subsample_catalog(cat, int(ds["max_per_class"]), seed=seed)
    return cat, ds


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--chunk", nargs=2, type=int, default=None)
    p.add_argument("--assemble", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", args.seed))
    cat, ds = ordered_catalog(cfg, seed)
    name = ds["name"]
    seq_len = int(ds["seq_len"])
    min_obs = int(ds.get("min_obs", 30))
    part_dir = REPO_ROOT / "runs" / f"{name}_times_partials"
    part_dir.mkdir(parents=True, exist_ok=True)

    if args.chunk:
        start, stop = args.chunk
        stop = min(stop, len(cat))
        rows = cat.iloc[start:stop]
        n = len(rows)
        kept = np.zeros(n, dtype=bool)
        dt = np.zeros((n, seq_len), np.float64)
        m_raw = np.zeros((n, seq_len), np.float32)
        n_used = np.zeros(n, np.int32)
        labels = np.array(rows["label"].astype(str))
        ids = np.array(rows["object_id"].astype(str))
        for j, (_, row) in enumerate(rows.iterrows()):
            try:
                lc = load_lightcurve(Path(row["lc_path"]), ds["format"],
                                     ds["columns"])
            except Exception:
                continue
            if len(lc) < min_obs:
                continue
            a, _b, _c, mr, _er, nu = lightcurve_to_deltas(lc, seq_len)
            kept[j] = True
            dt[j] = a.astype(np.float64)
            m_raw[j] = mr
            n_used[j] = nu
        out = part_dir / f"part_{start:06d}_{stop:06d}.npz"
        np.savez(out, kept=kept, dt=dt, m_raw=m_raw, n_used=n_used,
                 labels=labels, ids=ids, start=start, stop=stop)
        print(f"saved {out} kept={kept.sum()}/{n}")
        return

    if args.assemble:
        parts = sorted(part_dir.glob("part_*.npz"))
        starts = [int(p.stem.split("_")[1]) for p in parts]
        stops = [int(p.stem.split("_")[2]) for p in parts]
        assert starts and starts[0] == 0, "missing first chunk"
        for a, b in zip(stops[:-1], starts[1:]):
            assert a == b, f"gap in chunks at {a}"
        assert stops[-1] >= len(cat), "missing tail chunk"

        classes = list(ds["classes"])
        class_to_int = {c: i for i, c in enumerate(classes)}
        dts, mraws, nus, ys, idss = [], [], [], [], []
        for pth in parts:
            z = np.load(pth, allow_pickle=False)
            k = z["kept"]
            dts.append(z["dt"][k])
            mraws.append(z["m_raw"][k])
            nus.append(z["n_used"][k])
            ys.append(np.array([class_to_int[l] for l in z["labels"][k]]))
            idss.append(z["ids"][k])
        dt = np.concatenate(dts)
        m_raw = np.concatenate(mraws)
        n_used = np.concatenate(nus)
        y = np.concatenate(ys)
        ids = np.concatenate(idss)
        print(f"assembled {len(y)} curves")

        sp = cfg.get("split", {})
        _tr, _va, idx_te = stratified_split(
            n=len(y), y=y,
            test_size=float(sp.get("test_size", 0.2)),
            val_size=float(sp.get("val_size", 0.15)),
            seed=int(sp.get("random_state", seed)),
        )

        ok = None
        adv_dir = REPO_ROOT / "outputs" / "adversarial" / name
        for npz in sorted(adv_dir.glob("*/adv_frac_*.npz")):
            ref = np.load(npz)["m_clean"]
            if ref.shape == m_raw[idx_te].shape:
                ok = bool(np.allclose(ref, m_raw[idx_te], atol=1e-5))
                print(f"alignment vs {npz.parent.name}/{npz.name}: {ok}")
                break
        out = REPO_ROOT / "runs" / f"{name}_adv_test_times.npz"
        np.savez(out, dt=dt[idx_te], n_used=n_used[idx_te], y=y[idx_te],
                 ids=ids[idx_te], m_check_ok=np.array(bool(ok)))
        print(f"saved {out} ({len(idx_te)} test curves) alignment_ok={ok}")
        return

    print(f"catalog rows: {len(cat)} (use --chunk START STOP / --assemble)")


if __name__ == "__main__":
    main()
