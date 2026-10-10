"""Classification metrics shared by all evaluation scripts."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None,
                           n_classes: int | None = None) -> Dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_proba is not None and n_classes is not None and n_classes > 1:
        try:
            if n_classes == 2:
                out["auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                out["auc_ovr_macro"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                )
        except ValueError:
            pass
    return out


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray, classes: Sequence[int]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for c in classes:
        m = y_true == c
        out[int(c)] = float((y_pred[m] == c).mean()) if m.any() else 0.0
    return out


def confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=labels)
