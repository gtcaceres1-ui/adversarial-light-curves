"""Emit exact train/validation/test counts and test epoch distributions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/tables"
SURVEYS = {
    "ogle": {
        "adv": "runs/adversarial_dataset",
        "labels": ["ecl", "rrlyr", "cep", "lpv"],
    },
    "gaia": {
        "adv": "runs/gaia_adversarial_dataset",
        "labels": ["rrlyr", "lpv", "cep", "dsct"],
    },
    "wise": {
        "adv": "runs/wise_adversarial_dataset",
        "labels": ["rrlyr", "lpv", "cep", "dsct"],
    },
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    count_rows, distribution_rows = [], []
    for survey, config in SURVEYS.items():
        sp = np.load(
            REPO / config["adv"] / "test_split.npz", allow_pickle=True
        )
        y_train = sp["y_train"].astype(np.int64)
        y_val = sp["y_val"].astype(np.int64)
        y_test = sp["y_te"].astype(np.int64)
        labels = config["labels"]
        for class_id, label in enumerate(labels):
            train = int((y_train == class_id).sum())
            val = int((y_val == class_id).sum())
            test = int((y_test == class_id).sum())
            count_rows.append({
                "survey": survey,
                "klass": label,
                "train": train,
                "validation": val,
                "test": test,
                "total": train + val + test,
            })
        count_rows.append({
            "survey": survey,
            "klass": "TOTAL",
            "train": len(y_train),
            "validation": len(y_val),
            "test": len(y_test),
            "total": len(y_train) + len(y_val) + len(y_test),
        })

        epochs = sp["n_sel_te"].astype(np.int64)
        q = np.percentile(epochs, [0, 25, 50, 75, 100])
        row = {
            "survey": survey,
            "n_obj": len(epochs),
            "ep_min": int(q[0]),
            "ep_p25": int(q[1]),
            "ep_median": int(q[2]),
            "ep_p75": int(q[3]),
            "ep_max": int(q[4]),
            "ep_mean": float(epochs.mean()),
            "capped_at_500": bool((epochs == 500).any()),
        }
        for class_id, label in enumerate(labels):
            row[f"median_{label}"] = int(
                np.median(epochs[y_test == class_id])
            )
        distribution_rows.append(row)

    pd.DataFrame(count_rows).to_csv(
        OUT / "dataset_counts.csv", index=False
    )
    pd.DataFrame(distribution_rows).to_csv(
        OUT / "obs_count_distribution.csv", index=False
    )
    print(f"Wrote {OUT / 'dataset_counts.csv'}")
    print(f"Wrote {OUT / 'obs_count_distribution.csv'}")


if __name__ == "__main__":
    main()
