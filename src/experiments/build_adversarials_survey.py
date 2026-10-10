"""Build canonical Gaia DR2 or WISE transfer attacks and data splits."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.raw_sequences import SEQ_LEN, build_sequence_inputs
from src.data.survey_config import LABELS_COMMON, LABEL_TO_INT, SEED
from src.data.survey_loader import (
    build_catalog,
    load_lightcurve,
    subsample_balanced,
)
from src.experiments.adversarial_dataset_builder import run_builder

REPO = Path(__file__).resolve().parents[2]


def _build_inputs(t_full, m_full, e_full, mu_t, sd_t, mu_m, sd_m):
    """Flat-tuple wrapper around build_sequence_inputs."""
    dt, dm, de, _e, count, ts, ms, es = build_sequence_inputs(
        t_full, m_full, e_full, mu_t, sd_t, mu_m, sd_m, SEQ_LEN
    )
    return dt, dm, de, count, ts, ms, es


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--survey", required=True, choices=["gaia", "wise"])
    args = parser.parse_args()
    survey = args.survey
    catalog = subsample_balanced(build_catalog(survey))
    run_builder(
        survey=survey,
        df=catalog,
        labels=LABELS_COMMON,
        label_to_int=LABEL_TO_INT,
        load_curve=lambda path: load_lightcurve(Path(path), survey),
        cnn_dir=REPO / f"runs/{survey}_cnn",
        out_root=REPO / f"runs/{survey}_adversarial_dataset",
        seed=SEED,
    )


if __name__ == "__main__":
    main()
