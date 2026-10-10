"""Train and evaluate Gaia DR2 or WISE CNN/RNN/RF baselines."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.survey_config import LABELS_COMMON, LABEL_TO_INT, SEED
from src.data.survey_loader import build_catalog, load_lightcurve, subsample_balanced
from src.experiments.baseline_trainer import run_baselines

REPO = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--survey", required=True, choices=["gaia", "wise"])
    args = parser.parse_args()
    survey = args.survey
    run_baselines(
        catalog=subsample_balanced(build_catalog(survey)),
        labels=LABELS_COMMON,
        label_to_int=LABEL_TO_INT,
        load_curve=lambda path: load_lightcurve(Path(path), survey),
        adv_root=REPO / f"runs/{survey}_adversarial_dataset",
        cnn_dir=REPO / f"runs/{survey}_cnn",
        rnn_dir=REPO / f"runs/{survey}_rnn",
        rf_dir=REPO / f"runs/{survey}_rf",
        metrics_dir=REPO / f"outputs/metrics/{survey}",
        seed=SEED,
    )


if __name__ == "__main__":
    main()
