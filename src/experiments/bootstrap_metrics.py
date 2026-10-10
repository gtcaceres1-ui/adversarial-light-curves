"""Bootstrap confidence intervals, macro-F1 / balanced accuracy / weighted-F1
and confusion matrices for the L5-raw model and the CNN baseline, across
OGLE / Gaia / WISE, on clean and photometrically-bounded adversarial test
inputs.

Rationale:
  * Statistical significance: the headline robustness claims hinge on
    sub-percentage-point differences, so every reported accuracy / F1 is given
    a 95% bootstrap confidence interval. The clean->adv drop and the L5-vs-CNN
    gap use a paired bootstrap (the same resampled indices are applied to every
    model and split within a replicate), which is the correct way to put a CI
    on a difference of correlated estimators.
  * Beyond accuracy: balanced accuracy, macro-F1 and weighted-F1 are reported
    in addition to accuracy, plus full per-class precision / recall / F1 and the
    confusion matrices, because the surveys are class-imbalanced (cep/dsct are
    minority classes).

Predictions are recomputed by reloading the trained models; the recomputed
L5-raw accuracies reproduce outputs/metrics/<survey>/robust_raw_eval.csv to
machine precision (asserted below). The `p_cl`/`p_ad` arrays stored inside the
Gaia/WISE test_split.npz are the CNN baseline's probabilities, not L5-raw, and
the "L5" rows of baselines_clean_vs_adv.csv for Gaia/WISE are stale; both are
ignored.

Outputs (per survey, under outputs/metrics/<survey>/):
  bootstrap_metrics.csv          tidy: model,split,scope,metric,point,lo,hi
  bootstrap_diffs.csv            paired CIs for drop (clean-adv) and L5-CNN gap
  per_example_preds.npz          y_true, valid, {l5,cnn}_{clean,adv} preds
  confusion_<model>_<split>.npy  4x4 confusion matrices (rows=true, cols=pred)
And a combined outputs/metrics/bootstrap_metrics_all.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.models.robust_raw_encoder import _Sampling

REPO = Path(__file__).resolve().parents[2]
METRICS = REPO / "outputs/metrics"

B = 2000          # bootstrap replicates
SEED = 0
CI = (2.5, 97.5)  # 95% percentile interval
K_CLASSES = 4

# Per-survey configuration. Note the OGLE vs Gaia/WISE key-naming difference
# (de_te/valid vs DE_te/valid_te) and the survey-specific class label order.
SURVEYS = {
    "ogle": dict(
        adv="runs/adversarial_dataset", l5="runs/robust_raw_l5_ogle",
        cnn="runs/cnn_ogle",
        labels=["ecl", "rrlyr", "cep", "lpv"]),
    "gaia": dict(
        adv="runs/gaia_adversarial_dataset", l5="runs/gaia_robust_raw_l5",
        cnn="runs/gaia_cnn",
        labels=["rrlyr", "lpv", "cep", "dsct"]),
    "wise": dict(
        adv="runs/wise_adversarial_dataset", l5="runs/wise_robust_raw_l5",
        cnn="runs/wise_cnn",
        labels=["rrlyr", "lpv", "cep", "dsct"]),
}


# --------------------------------------------------------------------------- #
# Metric helpers (fast, confusion-matrix based)
# --------------------------------------------------------------------------- #
def confusion(y_true, y_pred, k=K_CLASSES):
    """k x k confusion matrix (rows=true, cols=pred) via a single bincount."""
    code = y_true.astype(np.int64) * k + y_pred.astype(np.int64)
    return np.bincount(code, minlength=k * k).reshape(k, k)


def metrics_from_cm(cm):
    """Return scalar metrics + per-class precision/recall/F1 from a conf. matrix."""
    cm = cm.astype(np.float64)
    n = cm.sum()
    tp = np.diag(cm)
    support = cm.sum(axis=1)          # true count per class
    pred_tot = cm.sum(axis=0)         # predicted count per class
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    prec = np.divide(tp, pred_tot, out=np.zeros_like(tp), where=pred_tot > 0)
    denom = prec + recall
    f1 = np.divide(2 * prec * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    acc = tp.sum() / n if n > 0 else 0.0
    # balanced accuracy = mean recall over classes that are present in y_true
    bal = recall[support > 0].mean() if (support > 0).any() else 0.0
    macro_f1 = f1.mean()                       # over all k classes
    weighted_f1 = (support * f1).sum() / n if n > 0 else 0.0
    return dict(acc=acc, balanced_acc=bal, macro_f1=macro_f1,
                weighted_f1=weighted_f1, precision=prec, recall=recall, f1=f1)


def scalar_metric_vector(y_true, y_pred, k=K_CLASSES):
    """Return [acc, balanced_acc, macro_f1, weighted_f1] for a bootstrap sample."""
    m = metrics_from_cm(confusion(y_true, y_pred, k))
    return np.array([m["acc"], m["balanced_acc"], m["macro_f1"], m["weighted_f1"]])


SCALAR_NAMES = ["acc", "balanced_acc", "macro_f1", "weighted_f1"]


# --------------------------------------------------------------------------- #
# Model prediction
# --------------------------------------------------------------------------- #
def _two_tower(X1, X2, DE, sd_m):
    sigma = (DE.squeeze(-1) / sd_m).astype(np.float32)
    xt = X1.astype(np.float32)
    xm = np.concatenate([X2, sigma[..., None]], axis=-1).astype(np.float32)
    return xt, xm


def _predict_l5(enc, clf, xt, xm, bs=512):
    out = np.zeros(len(xt), np.int64)
    for s in range(0, len(xt), bs):
        e = min(s + bs, len(xt))
        mu, _, _ = enc([xt[s:e], xm[s:e]], training=False)
        out[s:e] = clf(mu, training=False).numpy().argmax(1)
    return out


def _predict_cnn(cnn, xt, dm, bs=512):
    """CNN baseline takes two single-channel towers [dt, dm] (no sigma)."""
    out = np.zeros(len(xt), np.int64)
    for s in range(0, len(xt), bs):
        e = min(s + bs, len(xt))
        p = cnn([xt[s:e], dm[s:e]], training=False).numpy()
        out[s:e] = p.argmax(1)
    return out


# --------------------------------------------------------------------------- #
# Per-survey driver
# --------------------------------------------------------------------------- #
def run_survey(name, cfg):
    print(f"\n{'='*64}\n{name.upper()}\n{'='*64}")
    adv = REPO / cfg["adv"]
    sp = np.load(adv / "test_split.npz", allow_pickle=True)
    X1 = sp["X1_te"]; X2 = sp["X2_te"]; X2a = sp["X2_te_adv"]
    DE = sp["DE_te"]; y = sp["y_te"].astype(np.int64)
    valid = sp["valid_te"].astype(bool); sd_m = float(sp["sd_m"])
    labels = cfg["labels"]

    xt, xm_cl = _two_tower(X1, X2, DE, sd_m)
    _,  xm_ad = _two_tower(X1, X2a, DE, sd_m)

    # ---- L5-raw ----
    print("  reloading L5-raw ...")
    enc = tf.keras.models.load_model(
        REPO / cfg["l5"] / "encoder_backbone.keras",
        custom_objects={"_Sampling": _Sampling}, compile=False, safe_mode=False)
    clf = tf.keras.models.load_model(
        REPO / cfg["l5"] / "classifier.keras",
        custom_objects={"_Sampling": _Sampling}, compile=False, safe_mode=False)
    l5_cl = _predict_l5(enc, clf, xt, xm_cl)
    l5_ad = _predict_l5(enc, clf, xt, xm_ad)

    # sanity: recomputed L5 accuracy must match robust_raw_eval.csv
    acc_cl = (l5_cl[valid] == y[valid]).mean()
    acc_ad = (l5_ad[valid] == y[valid]).mean()
    eval_df = pd.read_csv(METRICS / name / "robust_raw_eval.csv")
    ref = eval_df[eval_df.scope == "all"].iloc[0]
    ref_cl, ref_ad = float(ref.acc_clean), float(ref.acc_adv)
    print(f"  L5 clean={acc_cl:.6f} (ref {ref_cl})  adv={acc_ad:.6f} (ref {ref_ad})")
    assert abs(acc_cl - ref_cl) < 1e-4 and abs(acc_ad - ref_ad) < 1e-4, \
        "L5 recomputation does not match robust_raw_eval.csv!"

    # ---- CNN baseline ----
    print("  reloading CNN baseline ...")
    cnn = tf.keras.models.load_model(
        REPO / cfg["cnn"] / "cnn_two_tower.keras", compile=False, safe_mode=False)
    dt = X1.astype(np.float32)
    cnn_cl = _predict_cnn(cnn, dt, X2.astype(np.float32))
    cnn_ad = _predict_cnn(cnn, dt, X2a.astype(np.float32))
    print(f"  CNN clean={(cnn_cl[valid]==y[valid]).mean():.6f}  "
          f"adv={(cnn_ad[valid]==y[valid]).mean():.6f}")

    # ---- restrict to valid ----
    yt = y[valid]
    preds = {
        ("L5-raw", "clean"): l5_cl[valid], ("L5-raw", "adv"): l5_ad[valid],
        ("CNN", "clean"): cnn_cl[valid], ("CNN", "adv"): cnn_ad[valid],
    }
    n = len(yt)

    # ---- point estimates + confusion matrices ----
    mout = METRICS / name
    mout.mkdir(parents=True, exist_ok=True)
    point = {}
    for (mdl, split), yp in preds.items():
        cm = confusion(yt, yp)
        np.save(mout / f"confusion_{mdl}_{split}.npy", cm)
        point[(mdl, split)] = metrics_from_cm(cm)

    # ---- paired bootstrap ----
    print(f"  bootstrap B={B} ...")
    rng = np.random.default_rng(SEED)
    # storage: scalar metrics (B x 4) per (model,split); per-class f1 (B x k) for L5
    scal = {key: np.zeros((B, 4)) for key in preds}
    pc = {key: {m: np.zeros((B, K_CLASSES)) for m in ("precision", "recall", "f1")}
          for key in preds}
    # differences
    drop = {mdl: np.zeros((B, 4)) for mdl in ("L5-raw", "CNN")}   # clean - adv
    gap = np.zeros((B, 4))   # L5 - CNN (clean)
    gap_adv = np.zeros((B, 4))  # L5 - CNN (adv)

    for b in range(B):
        idx = rng.integers(0, n, size=n)           # shared across all models/splits
        yt_b = yt[idx]
        cur = {}
        for key, yp in preds.items():
            m = metrics_from_cm(confusion(yt_b, yp[idx]))
            scal[key][b] = [m["acc"], m["balanced_acc"], m["macro_f1"], m["weighted_f1"]]
            cur[key] = scal[key][b]
            for mm in ("precision", "recall", "f1"):
                pc[key][mm][b] = m[mm]
        drop["L5-raw"][b] = cur[("L5-raw", "clean")] - cur[("L5-raw", "adv")]
        drop["CNN"][b] = cur[("CNN", "clean")] - cur[("CNN", "adv")]
        gap[b] = cur[("L5-raw", "clean")] - cur[("CNN", "clean")]
        gap_adv[b] = cur[("L5-raw", "adv")] - cur[("CNN", "adv")]

    lo_hi = lambda arr: (np.percentile(arr, CI[0], axis=0),
                         np.percentile(arr, CI[1], axis=0))

    # ---- assemble tidy rows ----
    rows = []
    for (mdl, split), m in point.items():
        plo, phi = lo_hi(scal[(mdl, split)])
        for i, nm in enumerate(SCALAR_NAMES):
            rows.append(dict(survey=name, model=mdl, split=split, scope="all",
                             metric=nm, point=m[nm], lo=plo[i], hi=phi[i]))
        # per-class precision/recall/f1
        for mm in ("precision", "recall", "f1"):
            clo, chi = lo_hi(pc[(mdl, split)][mm])
            for c, lab in enumerate(labels):
                rows.append(dict(survey=name, model=mdl, split=split, scope=lab,
                                 metric=mm, point=m[mm][c], lo=clo[c], hi=chi[c]))
    df = pd.DataFrame(rows)
    df.to_csv(mout / "bootstrap_metrics.csv", index=False)

    # ---- differences (paired) ----
    drows = []
    for mdl in ("L5-raw", "CNN"):
        pt = (np.array([point[(mdl, "clean")][k] for k in SCALAR_NAMES])
              - np.array([point[(mdl, "adv")][k] for k in SCALAR_NAMES]))
        dlo, dhi = lo_hi(drop[mdl])
        for i, nm in enumerate(SCALAR_NAMES):
            drows.append(dict(survey=name, comparison=f"{mdl} clean-adv (drop)",
                              metric=nm, point=pt[i], lo=dlo[i], hi=dhi[i]))
    for label, arr, sa, sb in [("L5-raw - CNN (clean)", gap, "clean", "clean"),
                               ("L5-raw - CNN (adv)", gap_adv, "adv", "adv")]:
        pt = (np.array([point[("L5-raw", sa)][k] for k in SCALAR_NAMES])
              - np.array([point[("CNN", sb)][k] for k in SCALAR_NAMES]))
        glo, ghi = lo_hi(arr)
        for i, nm in enumerate(SCALAR_NAMES):
            drows.append(dict(survey=name, comparison=label, metric=nm,
                              point=pt[i], lo=glo[i], hi=ghi[i]))
    ddf = pd.DataFrame(drows)
    ddf.to_csv(mout / "bootstrap_diffs.csv", index=False)

    # ---- per-example preds for reuse ----
    np.savez_compressed(
        mout / "per_example_preds.npz",
        y_true=yt, labels=np.array(labels),
        l5_clean=preds[("L5-raw", "clean")], l5_adv=preds[("L5-raw", "adv")],
        cnn_clean=preds[("CNN", "clean")], cnn_adv=preds[("CNN", "adv")])

    # ---- console summary ----
    print(f"\n  {'metric':14s}  {'L5 clean':>22s}  {'L5 adv':>22s}")
    for i, nm in enumerate(SCALAR_NAMES):
        c = point[("L5-raw", "clean")][nm]
        a = point[("L5-raw", "adv")][nm]
        clo, chi = lo_hi(scal[("L5-raw", "clean")])
        alo, ahi = lo_hi(scal[("L5-raw", "adv")])
        print(f"  {nm:14s}  {c:.4f} [{clo[i]:.4f},{chi[i]:.4f}]  "
              f"{a:.4f} [{alo[i]:.4f},{ahi[i]:.4f}]")
    return df, ddf


def main():
    all_df = []
    for name, cfg in SURVEYS.items():
        df, _ = run_survey(name, cfg)
        all_df.append(df)
    combined = pd.concat(all_df, ignore_index=True)
    combined.to_csv(METRICS / "bootstrap_metrics_all.csv", index=False)
    print(f"\nWrote {METRICS/'bootstrap_metrics_all.csv'} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
