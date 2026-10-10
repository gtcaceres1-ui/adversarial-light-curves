"""Train and evaluate OGLE CNN/RNN/RF baselines."""
from pathlib import Path

from src.experiments.baseline_trainer import run_baselines
from src.experiments.build_adversarials_ogle import (
    CNN_DIR,
    LABELS,
    LABEL_TO_INT,
    OUT_ROOT,
    SEED,
    build_catalog,
    load_lightcurve,
    subsample_balanced,
)

REPO = Path(__file__).resolve().parents[2]


def main():
    run_baselines(
        catalog=subsample_balanced(build_catalog()),
        labels=LABELS,
        label_to_int=LABEL_TO_INT,
        load_curve=load_lightcurve,
        adv_root=OUT_ROOT,
        cnn_dir=CNN_DIR,
        rnn_dir=REPO / "runs/rnn_ogle",
        rf_dir=REPO / "runs/rf_ogle",
        metrics_dir=REPO / "outputs/metrics/ogle",
        seed=SEED,
    )


if __name__ == "__main__":
    main()
