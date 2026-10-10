"""Build the canonical OGLE-III transfer attack and data splits."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.raw_sequences import SEQ_LEN, build_sequence_inputs
from src.data.survey_loader import uniform_indices
from src.experiments.adversarial_dataset_builder import (
    BATCH_SIZE,
    K_MAG,
    PGD_STEP_FRAC,
    PGD_STEPS,
    SMOOTH_LAMBDA,
    run_builder,
)

SEED = 42
MAX_PER_CLASS = 20000
LABELS = ["ecl", "rrlyr", "cep", "lpv"]
LABEL_TO_INT = {label: i for i, label in enumerate(LABELS)}

REPO = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPO / "Data/OGLE-III/OGLE_dataset.dat"
LCS_DIR = REPO / "Data/OGLE-III/LCs"
CNN_DIR = REPO / "runs/cnn_ogle"
OUT_ROOT = REPO / "runs/adversarial_dataset"


def infer_label_from_path(path_str: str):
    path = str(path_str).lower().replace("\\", "/")
    return next((label for label in LABELS if f"/{label}/" in path), None)


def build_catalog() -> pd.DataFrame:
    cat = pd.read_csv(CATALOG_PATH)
    cat.columns = [column.strip() for column in cat.columns]
    cat["label4"] = cat["Path"].apply(infer_label_from_path)
    cat["lc_file"] = cat["ID"].apply(lambda value: LCS_DIR / f"{value}.dat")
    cat = cat.dropna(subset=["label4"]).copy()
    return cat[cat["lc_file"].apply(Path.exists)].reset_index(drop=True)


def subsample_balanced(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for label in LABELS:
        part = df[df["label4"] == label]
        if len(part) > MAX_PER_CLASS:
            part = part.sample(MAX_PER_CLASS, random_state=SEED)
        parts.append(part)
    return pd.concat(parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def load_lightcurve(path: Path) -> pd.DataFrame:
    curve = pd.read_csv(
        path, sep=r"\s+", comment="#", header=None,
        names=["time", "magnitude", "error"],
    )
    return (
        curve.dropna()
        .sort_values("time")
        .drop_duplicates(subset="time", keep="first")
    )


def build_inputs(t_full, m_full, e_full, mu_t, sd_t, mu_m, sd_m):
    """Flat-tuple wrapper around build_sequence_inputs."""
    dt, dm, de, _e, count, ts, ms, es = build_sequence_inputs(
        t_full, m_full, e_full, mu_t, sd_t, mu_m, sd_m, SEQ_LEN
    )
    return dt, dm, de, count, ts, ms, es


def main():
    catalog = subsample_balanced(build_catalog())
    run_builder(
        survey="ogle",
        df=catalog,
        labels=LABELS,
        label_to_int=LABEL_TO_INT,
        load_curve=load_lightcurve,
        cnn_dir=CNN_DIR,
        out_root=OUT_ROOT,
        seed=SEED,
    )


if __name__ == "__main__":
    main()
