"""Export compact multi-survey LaTeX tables from authoritative CSV files."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
METRICS = REPO / "outputs/metrics"
TABLES = REPO / "outputs/tables"
OUT = TABLES / "multisurvey"
SURVEYS = {
    "ogle": "OGLE-III",
    "gaia": r"\textit{Gaia} DR2",
    "wise": "WISE",
}
LABELS = {
    "ogle": ["ecl", "rrlyr", "cep", "lpv"],
    "gaia": ["rrlyr", "lpv", "cep", "dsct"],
    "wise": ["rrlyr", "lpv", "cep", "dsct"],
}


def main_table():
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Clean and transferred-attack test accuracy for the CNN surrogate, feature-based RF baseline, and L5-raw. Drop is clean minus adversarial accuracy in percentage points.}",
        r"\label{tab:main_multisurvey}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Survey & Model & Clean & Adversarial & Drop (pp)\\",
        r"\midrule",
    ]
    for survey, survey_tex in SURVEYS.items():
        baseline = pd.read_csv(
            METRICS / survey / "baselines_clean_vs_adv.csv"
        )
        robust = pd.read_csv(
            METRICS / survey / "robust_raw_eval.csv"
        )
        rows = []
        for model in ("CNN", "RF"):
            row = baseline[baseline.model == model].iloc[0]
            rows.append((model, row.acc_clean, row.acc_adv))
        row = robust[robust.scope == "all"].iloc[0]
        rows.append(("L5-raw", row.acc_clean, row.acc_adv))
        for index, (model, clean, adversarial) in enumerate(rows):
            survey_cell = (
                f"\\multirow{{3}}{{*}}{{{survey_tex}}}" if index == 0 else ""
            )
            model_cell = (
                r"\textbf{L5-raw}" if model == "L5-raw" else model
            )
            lines.append(
                f"{survey_cell} & {model_cell} & {clean:.3f} & "
                f"{adversarial:.3f} & {100*(clean-adversarial):.1f}\\\\"
            )
        lines.append(
            r"\midrule" if survey != "wise" else r"\bottomrule"
        )
    lines.extend([r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def per_class_table():
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Per-class clean and transferred-attack accuracy of L5-raw.}",
        r"\label{tab:per_class_l5_multi}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Survey & Class & Clean & Adversarial & Drop (pp)\\",
        r"\midrule",
    ]
    for survey, survey_tex in SURVEYS.items():
        robust = pd.read_csv(
            METRICS / survey / "robust_raw_eval.csv"
        )
        labels = LABELS[survey]
        for index, label in enumerate(labels):
            row = robust[robust.scope == label].iloc[0]
            survey_cell = (
                f"\\multirow{{4}}{{*}}{{{survey_tex}}}" if index == 0 else ""
            )
            lines.append(
                f"{survey_cell} & {label.upper()} & {row.acc_clean:.3f} & "
                f"{row.acc_adv:.3f} & "
                f"{100*(row.acc_clean-row.acc_adv):.1f}\\\\"
            )
        lines.append(
            r"\midrule" if survey != "wise" else r"\bottomrule"
        )
    lines.extend([r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def counts_table():
    counts = pd.read_csv(TABLES / "dataset_counts.csv")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Exact object counts in the 70/10/20 split.}",
        r"\label{tab:counts}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Survey & Class & Train & Validation & Test & Total\\",
        r"\midrule",
    ]
    for survey, survey_tex in SURVEYS.items():
        subset = counts[counts.survey == survey]
        for index, (_, row) in enumerate(subset.iterrows()):
            survey_cell = (
                f"\\multirow{{{len(subset)}}}{{*}}{{{survey_tex}}}"
                if index == 0 else ""
            )
            label = "All" if row.klass == "TOTAL" else row.klass.upper()
            lines.append(
                f"{survey_cell} & {label} & {int(row.train):,} & "
                f"{int(row.validation):,} & {int(row.test):,} & "
                f"{int(row.total):,}\\\\"
            )
        lines.append(
            r"\midrule" if survey != "wise" else r"\bottomrule"
        )
    lines.extend([r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "main_multisurvey.tex": main_table(),
        "per_class_l5_multi.tex": per_class_table(),
        "dataset_counts.tex": counts_table(),
    }
    for name, text in artifacts.items():
        path = OUT / name
        path.write_text(text)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
