"""Cache raw Delta-t (and ids) for the adversarial test split.

The adversarial npz files store magnitudes but not observation times. This
script rebuilds the dataset bundle deterministically (same config and seed as
the attack generation) and persists the test-row times so that detectability
analyses (Lomb-Scargle) can run without reloading the raw survey data.

Alignment is verified by comparing the rebuilt m_raw[idx_test] against
m_clean stored in an adversarial npz: they must match exactly.

Output: runs/<dataset>_adv_test_times.npz with
  dt (B,L) raw days, n_used (B,), y (B,), ids (B,), m_check_ok (bool)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.experiments.dataset import build_dataset
from src.utils.config import load_config
from src.utils.paths import REPO_ROOT


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", args.seed))
    bundle = build_dataset(cfg, REPO_ROOT, seed=seed)
    idx = bundle.idx_test

    out = REPO_ROOT / "runs" / f"{bundle.name}_adv_test_times.npz"

    # alignment check against any adversarial npz of this dataset
    ok = None
    adv_dir = REPO_ROOT / "outputs" / "adversarial" / bundle.name
    for npz in sorted(adv_dir.glob("*/adv_frac_*.npz")):
        ref = np.load(npz)["m_clean"]
        if ref.shape == bundle.m_raw[idx].shape:
            ok = bool(np.allclose(ref, bundle.m_raw[idx], atol=1e-5))
            print(f"alignment vs {npz.parent.name}/{npz.name}: {ok}")
            break

    np.savez(
        out,
        dt=bundle.dt[idx].astype(np.float64),
        n_used=bundle.n_used[idx].astype(np.int32),
        y=bundle.y[idx].astype(np.int64),
        ids=np.asarray(bundle.object_id[idx], dtype=str),
        m_check_ok=np.array(ok if ok is not None else False),
    )
    print(f"saved {out}  ({len(idx)} curves)  alignment_ok={ok}")


if __name__ == "__main__":
    main()
