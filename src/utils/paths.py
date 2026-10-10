"""Centralized output-path helpers so every experiment writes to the same tree."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = REPO_ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
METRICS = OUTPUTS / "metrics"
ADVERSARIAL = OUTPUTS / "adversarial"
CHECKPOINTS = OUTPUTS / "checkpoints"
LOGS = OUTPUTS / "logs"


def ensure_dirs() -> None:
    for d in (FIGURES, TABLES, METRICS, ADVERSARIAL, CHECKPOINTS, LOGS):
        d.mkdir(parents=True, exist_ok=True)


def metrics_dir(dataset: str) -> Path:
    p = METRICS / dataset
    p.mkdir(parents=True, exist_ok=True)
    return p


def checkpoint_dir(dataset: str, model: str) -> Path:
    p = CHECKPOINTS / dataset / model
    p.mkdir(parents=True, exist_ok=True)
    return p


def adversarial_dir(dataset: str, model: str) -> Path:
    p = ADVERSARIAL / dataset / model
    p.mkdir(parents=True, exist_ok=True)
    return p
