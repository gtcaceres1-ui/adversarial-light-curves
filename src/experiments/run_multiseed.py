"""Multi-seed re-training of the headline models (run-to-run variance).

Trains L1 and L5 with several seeds under the fixed transfer protocol and
reports mean +/- std across seeds, so that the single-seed numbers in the
tables can be read against the run-to-run spread.

Usage (each L5 run ~30-50 min on CPU, L1 faster):
    python -m src.experiments.run_multiseed --seeds 42,1,2,3,4 --levels 1,5
    python -m src.experiments.run_multiseed --aggregate-only

Also trains the noise baseline per seed with --include-noise.

Outputs:
    outputs/metrics/ogle/robust_raw_ablation.csv   (one row per run, seed col)
    outputs/metrics/ogle/multiseed_summary.csv     (mean/std per level)
    outputs/tables/multiseed_summary.tex           (paper-ready snippet)

The CNN surrogate is not re-seeded here: that would require regenerating the
adversarial test set per seed, and the attack test set is kept fixed across
model seeds on purpose.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
METRICS = REPO / "outputs/metrics/ogle"
TABLES = REPO / "outputs/tables"


def aggregate() -> None:
    df = pd.read_csv(METRICS / "robust_raw_ablation.csv")
    if "seed" not in df.columns:
        df["seed"] = 42
    df["level"] = df["model"].str.extract(r"robust_raw_l(\d)").astype(int)
    rows = []
    for lvl, g in df.groupby("level"):
        rows.append({
            "level": lvl, "n_seeds": len(g),
            "acc_clean_mean": g.acc_clean.mean(),
            "acc_clean_std": g.acc_clean.std(ddof=1),
            "acc_adv_mean": g.acc_adv.mean(),
            "acc_adv_std": g.acc_adv.std(ddof=1),
            "drop_mean": g["drop"].mean(),
            "drop_std": g["drop"].std(ddof=1),
            "seeds": ",".join(str(s) for s in sorted(g.seed)),
        })
    out = pd.DataFrame(rows)
    out.to_csv(METRICS / "multiseed_summary.csv", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    TABLES.mkdir(parents=True, exist_ok=True)
    lines = [
        r"% Per-seed variance of the raw-encoder ablation (transfer protocol)",
        r"\begin{table}",
        r"\centering",
        r"\caption{Run-to-run variability of the raw encoder under the fixed"
        r" transferred attack: mean $\pm$ standard deviation over"
        f" {int(out.n_seeds.max())} training seeds.}}",
        r"\label{tab:multiseed}",
        r"\begin{tabular}{lcc}",
        r"\hline\hline",
        r"Variant & Clean & Adversarial \\",
        r"\hline",
    ]
    for _, r in out.iterrows():
        cs = 0.0 if np.isnan(r.acc_clean_std) else r.acc_clean_std
        as_ = 0.0 if np.isnan(r.acc_adv_std) else r.acc_adv_std
        lines.append(
            f"L{int(r.level)} & ${r.acc_clean_mean:.3f} \\pm {cs:.3f}$ & "
            f"${r.acc_adv_mean:.3f} \\pm {as_:.3f}$ \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    (TABLES / "multiseed_summary.tex").write_text("\n".join(lines))
    print(f"LaTeX -> {TABLES/'multiseed_summary.tex'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,1,2,3,4")
    ap.add_argument("--levels", default="1,5")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--include-noise", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    if not args.aggregate_only:
        seeds = [int(s) for s in args.seeds.split(",")]
        for seed in seeds:
            cmd = [sys.executable, "-m",
                   "src.experiments.train_robust_raw_ablation",
                   "--levels", args.levels,
                   "--epochs", str(args.epochs),
                   "--seed", str(seed)]
            print(">>", " ".join(cmd))
            subprocess.run(cmd, check=True, cwd=REPO)
            if args.include_noise:
                for dist in ("uniform", "gauss"):
                    cmd = [sys.executable, "-m",
                           "src.experiments.train_noise_baseline",
                           "--dist", dist, "--seed", str(seed),
                           "--epochs", str(args.epochs)]
                    print(">>", " ".join(cmd))
                    subprocess.run(cmd, check=True, cwd=REPO)
    aggregate()


if __name__ == "__main__":
    main()
