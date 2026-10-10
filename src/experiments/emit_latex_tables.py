"""Emit paste-ready LaTeX table bodies from the saved bootstrap and count CSVs.

Reads:
  outputs/metrics/<survey>/bootstrap_metrics.csv
  outputs/metrics/<survey>/bootstrap_diffs.csv
  outputs/metrics/<survey>/adaptive_pgd_l5_raw.csv      (if present)
  outputs/tables/dataset_counts.csv
  outputs/tables/obs_count_distribution.csv

Prints each table to stdout. Numbers are read from disk, never transcribed.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
M = REPO / "outputs/metrics"
T = REPO / "outputs/tables"
SURVEYS = ["ogle", "gaia", "wise"]
SURVEY_TEX = {"ogle": "OGLE-III", "gaia": "Gaia DR2", "wise": "WISE"}
LABELS = {"ogle": ["ecl", "rrlyr", "cep", "lpv"],
          "gaia": ["rrlyr", "lpv", "cep", "dsct"],
          "wise": ["rrlyr", "lpv", "cep", "dsct"]}
CLASS_TEX = {"ecl": "ECL", "rrlyr": "RRL", "cep": "CEP", "lpv": "LPV", "dsct": "DSCT"}


def pct(p, lo, hi):
    """Format a [0,1] metric as a percentage with a 95% CI."""
    return f"{100*p:.1f} ({100*lo:.1f}\\,\\text{{--}}\\,{100*hi:.1f})"


def f3(p, lo, hi):
    return f"{p:.3f} ({lo:.3f}\\,\\text{{--}}\\,{hi:.3f})"


def get(df, model, split, scope, metric):
    r = df[(df.model == model) & (df.split == split) &
           (df.scope == scope) & (df.metric == metric)]
    r = r.iloc[0]
    return r.point, r.lo, r.hi


# --------------------------------------------------------------------------- #
def table_main():
    print(r"% ===================== TABLE A: bootstrap CIs (C5/C6) =====================")
    print(r"\begin{table*}[t]\centering")
    print(r"\caption{Clean and adversarial test performance with 95\% bootstrap "
          r"confidence intervals ($B=2000$). Adversarial inputs use the photometric "
          r"budget $k\,\sigma_i$ ($k=0.5$). The attack optimizes bounded raw-magnitude "
          r"displacements and recomputes $\Delta m$. Intervals are percentile "
          r"bootstrap over the test set; the "
          r"clean$\to$adv drop uses a paired bootstrap. All values are percentages.}")
    print(r"\label{tab:bootstrap_main}")
    print(r"\begin{tabular}{llcccccc}")
    print(r"\toprule")
    print(r" & & \multicolumn{2}{c}{Accuracy} & \multicolumn{2}{c}{Balanced acc.} "
          r"& \multicolumn{2}{c}{Macro-$F_1$}\\")
    print(r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}")
    print(r"Survey & Model & Clean & Adv. & Clean & Adv. & Clean & Adv.\\")
    print(r"\midrule")
    for s in SURVEYS:
        df = pd.read_csv(M / s / "bootstrap_metrics.csv")
        for j, model in enumerate(["CNN", "L5-raw"]):
            cells = []
            for metric in ["acc", "balanced_acc", "macro_f1"]:
                for split in ["clean", "adv"]:
                    cells.append("$" + pct(*get(df, model, split, "all", metric)) + "$")
            sv = f"\\multirow{{2}}{{*}}{{{SURVEY_TEX[s]}}}" if j == 0 else ""
            mdl = r"\textbf{L5-raw}" if model == "L5-raw" else model
            print(f"{sv} & {mdl} & " + " & ".join(cells) + r"\\")
        print(r"\midrule" if s != SURVEYS[-1] else r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table*}")
    print()


def table_drops():
    print(r"% ============== TABLE A2: paired drop & L5-vs-CNN gap (C5) ==============")
    print(r"\begin{table}[t]\centering")
    print(r"\caption{Paired bootstrap 95\% CIs for the accuracy quantities that the "
          r"robustness claims rest on. The clean$\to$adv \emph{drop} and the "
          r"L5-raw$-$CNN \emph{gap} are computed within each bootstrap replicate "
          r"(paired), so the intervals account for the correlation between estimators. "
          r"All quantities are in percentage points.}")
    print(r"\label{tab:bootstrap_diffs}")
    print(r"\begin{tabular}{llc}")
    print(r"\toprule")
    print(r"Survey & Quantity (accuracy) & Value (pp), 95\% CI\\")
    print(r"\midrule")
    for s in SURVEYS:
        d = pd.read_csv(M / s / "bootstrap_diffs.csv")
        rows = [("CNN clean$-$adv drop", "CNN clean-adv (drop)"),
                ("L5-raw clean$-$adv drop", "L5-raw clean-adv (drop)"),
                ("L5-raw $-$ CNN (clean)", "L5-raw - CNN (clean)"),
                ("L5-raw $-$ CNN (adv)", "L5-raw - CNN (adv)")]
        for j, (lab, comp) in enumerate(rows):
            r = d[(d.comparison == comp) & (d.metric == "acc")].iloc[0]
            sv = f"\\multirow{{4}}{{*}}{{{SURVEY_TEX[s]}}}" if j == 0 else ""
            print(f"{sv} & {lab} & "
                  f"${100*r.point:+.1f}\\ [{100*r.lo:+.1f},\\,{100*r.hi:+.1f}]$\\\\")
        print(r"\midrule" if s != SURVEYS[-1] else r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()


def table_perclass():
    print(r"% ============== TABLE B: per-class F1 with CIs (C6) ==============")
    print(r"\begin{table}[t]\centering")
    print(r"\caption{Per-class $F_1$ of L5-raw (clean and adversarial) with 95\% "
          r"bootstrap CIs. Per-class results expose minority-class behavior that "
          r"aggregate accuracy can mask. All values are percentages.}")
    print(r"\label{tab:perclass_f1}")
    print(r"\begin{tabular}{llcc}")
    print(r"\toprule")
    print(r"Survey & Class & $F_1$ clean & $F_1$ adv.\\")
    print(r"\midrule")
    for s in SURVEYS:
        df = pd.read_csv(M / s / "bootstrap_metrics.csv")
        labs = LABELS[s]
        for j, lab in enumerate(labs):
            cl = "$" + pct(*get(df, "L5-raw", "clean", lab, "f1")) + "$"
            ad = "$" + pct(*get(df, "L5-raw", "adv", lab, "f1")) + "$"
            sv = f"\\multirow{{{len(labs)}}}{{*}}{{{SURVEY_TEX[s]}}}" if j == 0 else ""
            print(f"{sv} & {CLASS_TEX[lab]} & {cl} & {ad}\\\\")
        print(r"\midrule" if s != SURVEYS[-1] else r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()


def table_counts():
    print(r"% ============== TABLE D: dataset counts + epochs (C11) ==============")
    c = pd.read_csv(T / "dataset_counts.csv")
    o = pd.read_csv(T / "obs_count_distribution.csv")
    print(r"\begin{table}[t]\centering")
    print(r"\caption{Final per-class train/validation/test object counts and the per-object "
          r"observation-count (epochs) distribution. Counts follow a deterministic "
          r"stratified 70/10/20 split (seed~42) after capping each class at "
          r"$20{,}000$ objects. Epoch statistics are median [IQR] over the test set; "
          r"OGLE light curves are truncated to 500 epochs.}")
    print(r"\label{tab:counts}")
    print(r"\begin{tabular}{llrrrr}")
    print(r"\toprule")
    print(r"Survey & Class & Train & Val. & Test & Epochs (med.\ [IQR])\\")
    print(r"\midrule")
    for s in SURVEYS:
        labs = LABELS[s]
        orow = o[o.survey == s].iloc[0]
        for j, lab in enumerate(labs):
            r = c[(c.survey == s) & (c.klass == lab)].iloc[0]
            med = orow[f"median_{lab}"]
            sv = f"\\multirow{{{len(labs)+1}}}{{*}}{{{SURVEY_TEX[s]}}}" if j == 0 else ""
            print(f"{sv} & {CLASS_TEX[lab]} & {int(r.train):,} & "
                  f"{int(r.validation):,} & {int(r.test):,} & {int(med)}\\\\")
        tot = c[(c.survey == s) & (c.klass == 'TOTAL')].iloc[0]
        eps = f"{int(orow.ep_median)} [{int(orow.ep_p25)},{int(orow.ep_p75)}]"
        print(f" & \\emph{{All}} & {int(tot.train):,} & "
              f"{int(tot.validation):,} & {int(tot.test):,} & {eps}\\\\")
        print(r"\midrule" if s != SURVEYS[-1] else r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()


def table_adaptive():
    # only if all three adaptive CSVs exist
    paths = {s: M / s / "adaptive_pgd_l5_raw.csv" for s in SURVEYS}
    if not all(p.exists() for p in paths.values()):
        print("% (adaptive_pgd table skipped: attack CSVs not all present yet)")
        return
    print(r"% ============== TABLE E: adaptive white-box PGD (C2) ==============")
    print(r"\begin{table}[t]\centering")
    first = pd.read_csv(paths[SURVEYS[0]]).iloc[0]
    print(r"\caption{Adaptive white-box PGD evaluated on L5-raw's own gradient "
          f"({int(first.steps)} steps, {int(first.restarts)} random restarts, "
          f"step ${float(first.step_frac):.2f}\\,\\epsilon$, budget $k=0.5$). "
          r"The attack operates in bounded raw-magnitude displacement space. "
          r"These results complement, but do not certify against, other attacks. "
          r"All values are percentages.}")
    print(r"\label{tab:adaptive_pgd}")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"Survey & Clean & Adaptive PGD & Drop (pp)\\")
    print(r"\midrule")
    for s in SURVEYS:
        a = pd.read_csv(paths[s])
        r = a[a.scope == "all"].iloc[0]
        print(f"{SURVEY_TEX[s]} & {100*r.acc_clean:.1f} & {100*r.acc_adv:.1f} & "
              f"${100*r['drop']:+.1f}$\\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()


def table_ablation():
    path = M / "ogle" / "robust_raw_ablation.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    print(r"% ============== TABLE F: incremental objective ablation ==============")
    print(r"\begin{table}[t]\centering")
    print(r"\caption{Incremental OGLE-III ablation under the same transferred "
          r"photometrically bounded attack. L2 adds adversarial cross-entropy; "
          r"L3--L5 add TRADES, supervised contrastive learning, and VIB. "
          r"The comparison identifies which terms affect test robustness rather "
          r"than presuming that every auxiliary term is necessary.}")
    print(r"\label{tab:ablation_robust}")
    print(r"\begin{tabular}{lccc}\toprule")
    print(r"Model & Clean & Adversarial & Drop (pp)\\\midrule")
    for _, row in frame.iterrows():
        print(f"{row.model} & {100*row.acc_clean:.1f} & "
              f"{100*row.acc_adv:.1f} & {100*row['drop']:.1f}\\\\")
    print(r"\bottomrule\end{tabular}\end{table}")
    print()


def table_vib_only():
    path = M / "ogle" / "vib_only_beta_sweep.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    print(r"% ============== TABLE G: VIB-only negative control ==============")
    print(r"\begin{table}[t]\centering")
    print(r"\caption{VIB-only control without adversarial training on OGLE-III. "
          r"Deterministic inference uses the posterior mean; stochastic inference "
          r"averages 16 posterior samples. This directly tests whether the "
          r"bottleneck alone improves robustness.}")
    print(r"\label{tab:vib_only}")
    print(r"\begin{tabular}{lccc}\toprule")
    print(r"$\beta$ & Clean & Adversarial & Adv. stochastic\\\midrule")
    for _, row in frame.iterrows():
        print(f"{row.beta:g} & {100*row.acc_clean:.1f} & "
              f"{100*row.acc_adv:.1f} & {100*row.acc_adv_stoch:.1f}\\\\")
    print(r"\bottomrule\end{tabular}\end{table}")
    print()


def main():
    table_main()
    table_drops()
    table_perclass()
    table_counts()
    table_adaptive()
    table_ablation()
    table_vib_only()


if __name__ == "__main__":
    main()
