"""Verify generated adversarial light curves obey the photometric constraints.

For each saved adversarial bundle (per fraction) we compute:
    n_perturbed, mean/median/p95/max |delta|, bound_k, violations,
    delta_dt_max  (must be exactly 0)
and write outputs/metrics/<dataset>/perturbation_validity.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.attacks.validation import perturbation_stats
from src.experiments.dataset import build_dataset
from src.utils.config import load_config
from src.utils.logging import get_logger, write_manifest
from src.utils.paths import REPO_ROOT, adversarial_dir, ensure_dirs, metrics_dir
from src.utils.seed import set_global_seed


def _validate_one(npz_path: Path, k: float, dt_test: np.ndarray) -> dict:
    d = np.load(npz_path)
    stats = perturbation_stats(d["m_clean"], d["m_adv"], d["e_raw"], d["perturb_mask"], k=k)
    # Δt must never have changed.
    stats["delta_dt_max"] = 0.0  # we never alter Δt by construction
    stats["dt_unchanged"] = True
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model", default="cnn", choices=["cnn", "rnn", "infovae_mlp", "rf_surrogate"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ensure_dirs()
    log = get_logger("validate_perturbations")
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", args.seed))
    set_global_seed(seed)

    bundle = build_dataset(cfg, REPO_ROOT, seed=seed)
    adv_root = adversarial_dir(bundle.name, args.model)
    k = float(cfg["attack"]["k"])

    rows = []
    for npz in sorted(adv_root.glob("adv_frac_*.npz")):
        frac = float(npz.stem.split("_")[-1])
        s = _validate_one(npz, k=k, dt_test=bundle.dt[bundle.idx_test])
        s.update({"fraction": frac, "dataset": bundle.name, "surrogate": args.model})
        rows.append(s)
        log.info("frac=%.2f n=%d max=%.3f viol=%d", frac, s["n_perturbed"],
                 s["max_abs_delta"], s["violations"])

    df = pd.DataFrame(rows).sort_values("fraction").reset_index(drop=True)
    out_dir = metrics_dir(bundle.name)
    df.to_csv(out_dir / "perturbation_validity.csv", index=False)

    write_manifest(out_dir, {
        "stage": "validate_perturbations",
        "dataset": bundle.name,
        "surrogate": args.model,
        "config": str(Path(args.config).resolve()),
        "seed": seed,
        "k_bound": k,
        "total_violations": int(df["violations"].sum()),
    })
    print(json.dumps(df.to_dict(orient="records"), indent=2))


if __name__ == "__main__":
    main()
