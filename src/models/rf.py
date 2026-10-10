"""Random forest baseline.

The RF works on a small fixed feature vector extracted from each light curve
(amplitude, weighted mean/std, Stetson K, eta, kurtosis, slopes, fraction of
points beyond 1 std). This stays close to the feature-engineering baselines
historically used for variable star classification (e.g. FATS).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.metrics.lightcurve_features import features_from_arrays


def featurize(m_raw: np.ndarray, e_raw: np.ndarray, n_used: np.ndarray) -> np.ndarray:
    """Per-curve feature matrix used by both the RF baseline and the feature tables."""
    rows = []
    for i in range(m_raw.shape[0]):
        nu = int(n_used[i])
        rows.append(features_from_arrays(m_raw[i, :nu], e_raw[i, :nu]))
    keys = sorted(rows[0].keys()) if rows else []
    return np.array([[r[k] for k in keys] for r in rows], dtype=np.float64), keys


@dataclass
class RFModel:
    clf: RandomForestClassifier
    feature_keys: list
    n_classes: int

    def predict(self, m_raw: np.ndarray, e_raw: np.ndarray, n_used: np.ndarray) -> np.ndarray:
        X, _ = featurize(m_raw, e_raw, n_used)
        return self.clf.predict(X)

    def predict_proba(self, m_raw: np.ndarray, e_raw: np.ndarray, n_used: np.ndarray) -> np.ndarray:
        """Return probabilities aligned to canonical class indices [0..n_classes).

        sklearn's predict_proba returns columns in `clf.classes_` order — which
        only spans classes seen at fit time. We pad missing classes with zeros
        so downstream argmax over columns yields the correct integer label.
        """
        X, _ = featurize(m_raw, e_raw, n_used)
        raw = self.clf.predict_proba(X)
        out = np.zeros((raw.shape[0], self.n_classes), dtype=raw.dtype)
        for j, c in enumerate(self.clf.classes_):
            out[:, int(c)] = raw[:, j]
        return out

    def save(self, path):
        joblib.dump({"clf": self.clf, "feature_keys": self.feature_keys,
                     "n_classes": self.n_classes}, path)

    @staticmethod
    def load(path) -> "RFModel":
        d = joblib.load(path)
        n_classes = int(d.get("n_classes", len(d["clf"].classes_)))
        return RFModel(clf=d["clf"], feature_keys=d["feature_keys"], n_classes=n_classes)


def train_rf(
    m_raw: np.ndarray, e_raw: np.ndarray, n_used: np.ndarray, y: np.ndarray,
    cfg: Dict[str, Any], seed: int = 42, n_classes: int | None = None,
) -> RFModel:
    X, keys = featurize(m_raw, e_raw, n_used)
    clf = RandomForestClassifier(
        n_estimators=int(cfg.get("n_estimators", 100)),
        max_depth=cfg.get("max_depth", None),
        random_state=seed, n_jobs=-1,
    )
    clf.fit(X, y)
    if n_classes is None:
        n_classes = int(np.max(y)) + 1
    return RFModel(clf=clf, feature_keys=keys, n_classes=int(n_classes))
