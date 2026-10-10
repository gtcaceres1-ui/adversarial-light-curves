"""Sequentially rebuild all artifacts used by the publication."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYTHON = sys.executable


def run(module: str, *arguments: str, stdout_path: Path | None = None):
    command = [PYTHON, "-m", module, *arguments]
    print("\n>>>", " ".join(command), flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    if stdout_path is None:
        subprocess.run(command, cwd=REPO, env=env, check=True)
    else:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("w") as handle:
            subprocess.run(
                command, cwd=REPO, env=env, check=True,
                stdout=handle, text=True,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    epochs = str(args.epochs)

    if not args.skip_training:
        run("src.experiments.build_adversarials_ogle")
        for survey in ("gaia", "wise"):
            run(
                "src.experiments.build_adversarials_survey",
                "--survey", survey,
            )
        run("src.experiments.eval_baselines_ogle")
        for survey in ("gaia", "wise"):
            run(
                "src.experiments.train_eval_baselines_survey",
                "--survey", survey,
            )
        ablation_args = ["--epochs", epochs, "--levels", "1,2,3,4,5"]
        vib_args = ["--epochs", epochs]
        if args.force:
            ablation_args.append("--force")
            vib_args.append("--force")
        run("src.experiments.train_robust_raw_ablation", *ablation_args)
        for survey in ("gaia", "wise"):
            run(
                "src.experiments.train_robust_raw_survey",
                "--survey", survey, "--epochs", epochs,
            )
        run("src.experiments.train_vib_only_sweep", *vib_args)

    run("src.experiments.validate_perturbation_bounds")
    run("src.experiments.attack_l5_raw_adaptive_multi")
    run("src.experiments.attack_raw_ablation_adaptive")
    run("src.experiments.evaluate_raw_ablation_latent")
    run("src.experiments.bootstrap_metrics")
    run("src.experiments.dataset_counts")
    run("src.experiments.export_paper_tables_multisurvey")
    run("src.experiments.plot_baseline_vulnerability")
    run("src.experiments.plot_latent_comparison_l1_vs_l5_raw")
    run(
        "src.experiments.emit_latex_tables",
        stdout_path=REPO / "outputs/tables/publication_tables.tex",
    )
    print("\nPublication pipeline complete.")


if __name__ == "__main__":
    main()
