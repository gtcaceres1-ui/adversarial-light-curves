"""Resumable, chunked version of bootstrap_metrics.py.

Same metrics, RNG and output schema as src.experiments.bootstrap_metrics, but
(a) split into short resumable stages so each invocation stays under a
wall-clock budget, and (b) using standalone Keras 3 (`import keras`) to load
the .keras checkpoints, which were saved with Keras 3 and cannot be read by
tf.keras 2.x. A small `from_config` shim drops the `quantization_config`
kwarg written by newer Keras versions.

  python -m src.experiments.bootstrap_canonical_chunked predict --survey ogle
      Advances the four prediction streams (L5 clean/adv, CNN clean/adv),
      saving progress to outputs/metrics/<survey>/_pred_progress.npz.
      Re-run until it prints DONE; it then asserts the recomputed L5
      accuracies match robust_raw_eval.csv and writes
      per_example_preds.npz + confusion matrices.

  python -m src.experiments.bootstrap_canonical_chunked bootstrap --survey ogle
      Numpy-only paired bootstrap (B=2000, seed 0) from per_example_preds.npz.
      Writes bootstrap_metrics.csv and bootstrap_diffs.csv (same schema).

  python -m src.experiments.bootstrap_canonical_chunked combine
      Rebuilds outputs/metrics/bootstrap_metrics_all.csv.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

T0 = time.monotonic()
WALL_BUDGET = 34.0  # seconds of work per invocation (keras import counts)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
METRICS = REPO / "outputs/metrics"

B = 2000
SEED = 0
CI = (2.5, 97.5)
K_CLASSES = 4
SCALAR_NAMES = ["acc", "balanced_acc", "macro_f1", "weighted_f1"]
BS = 512

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


def _left() -> float:
    return WALL_BUDGET - (time.monotonic() - T0)


# ---- metric helpers (same logic as bootstrap_metrics.py) ----------------- #
def confusion(y_true, y_pred, k=K_CLASSES):
    code = y_true.astype(np.int64) * k + y_pred.astype(np.int64)
    return np.bincount(code, minlength=k * k).reshape(k, k)


def metrics_from_cm(cm):
    cm = cm.astype(np.float64)
    n = cm.sum()
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    pred_tot = cm.sum(axis=0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    prec = np.divide(tp, pred_tot, out=np.zeros_like(tp), where=pred_tot > 0)
    denom = prec + recall
    f1 = np.divide(2 * prec * recall, denom, out=np.zeros_like(tp),
                   where=denom > 0)
    acc = tp.sum() / n if n > 0 else 0.0
    bal = recall[support > 0].mean() if (support > 0).any() else 0.0
    macro_f1 = f1.mean()
    weighted_f1 = (support * f1).sum() / n if n > 0 else 0.0
    return dict(acc=acc, balanced_acc=bal, macro_f1=macro_f1,
                weighted_f1=weighted_f1, precision=prec, recall=recall, f1=f1)


def _two_tower(X1, X2, DE, sd_m):
    sigma = (DE.squeeze(-1) / sd_m).astype(np.float32)
    xt = X1.astype(np.float32)
    xm = np.concatenate([X2, sigma[..., None]], axis=-1).astype(np.float32)
    return xt, xm


# ---- Keras 3 loading shim ------------------------------------------------- #
def _keras3():
    import keras

    @keras.saving.register_keras_serializable(name="_Sampling")
    class _Sampling(keras.layers.Layer):
        def call(self, inputs):
            mu, logv = inputs
            eps = keras.random.normal(shape=keras.ops.shape(mu))
            return mu + keras.ops.exp(0.5 * logv) * eps

    for L in (keras.layers.Dense, keras.layers.Conv1D,
              keras.layers.EinsumDense):
        orig = L.from_config.__func__

        def mk(orig):
            def fc(cls, config):
                config = dict(config)
                config.pop("quantization_config", None)
                return orig(cls, config)
            return classmethod(fc)

        L.from_config = mk(orig)
    return keras, {"_Sampling": _Sampling}


# --------------------------------------------------------------------------- #
# Stage 1: chunked prediction
# --------------------------------------------------------------------------- #
def stage_predict(survey: str) -> None:
    keras, CO = _keras3()
    cfg = SURVEYS[survey]
    mout = METRICS / survey
    mout.mkdir(parents=True, exist_ok=True)
    prog_f = mout / "_pred_progress.npz"

    sp = np.load(REPO / cfg["adv"] / "test_split.npz", allow_pickle=True)
    X1 = sp["X1_te"]; X2 = sp["X2_te"]; X2a = sp["X2_te_adv"]
    DE = sp["DE_te"]; y = sp["y_te"].astype(np.int64)
    valid = sp["valid_te"].astype(bool); sd_m = float(sp["sd_m"])
    n_tot = len(X1)

    xt, xm_cl = _two_tower(X1, X2, DE, sd_m)
    _,  xm_ad = _two_tower(X1, X2a, DE, sd_m)
    dm_cl = X2.astype(np.float32); dm_ad = X2a.astype(np.float32)

    streams = ["l5_clean", "l5_adv", "cnn_clean", "cnn_adv"]
    if prog_f.exists():
        z = np.load(prog_f)
        preds = {s: z[s].copy() for s in streams}
        pos = {s: int(z[f"pos_{s}"]) for s in streams}
    else:
        preds = {s: np.full(n_tot, -1, np.int64) for s in streams}
        pos = {s: 0 for s in streams}

    if any(pos[s] < n_tot for s in streams):
        load = keras.saving.load_model
        enc = load(REPO / cfg["l5"] / "encoder_backbone.keras",
                   custom_objects=CO, compile=False, safe_mode=False)
        clf = load(REPO / cfg["l5"] / "classifier.keras",
                   custom_objects=CO, compile=False, safe_mode=False)
        cnn = load(REPO / cfg["cnn"] / "cnn_two_tower.keras",
                   custom_objects=CO, compile=False, safe_mode=False)
        print(f"[{survey}] models loaded t={time.monotonic()-T0:.1f}s "
              f"resume={ {s: pos[s] for s in streams} }")

        def step(stream: str) -> None:
            s = pos[stream]
            e = min(s + BS, n_tot)
            if stream == "l5_clean":
                mu, _, _ = enc([xt[s:e], xm_cl[s:e]], training=False)
                p = clf(mu, training=False)
            elif stream == "l5_adv":
                mu, _, _ = enc([xt[s:e], xm_ad[s:e]], training=False)
                p = clf(mu, training=False)
            elif stream == "cnn_clean":
                p = cnn([xt[s:e], dm_cl[s:e]], training=False)
            else:
                p = cnn([xt[s:e], dm_ad[s:e]], training=False)
            preds[stream][s:e] = np.asarray(p).argmax(1)
            pos[stream] = e

        while _left() > 4.0 and any(pos[s] < n_tot for s in streams):
            for s in streams:
                if pos[s] < n_tot and _left() > 4.0:
                    step(s)

        np.savez_compressed(
            prog_f, **preds, **{f"pos_{s}": pos[s] for s in streams})

    if any(pos[s] < n_tot for s in streams):
        print(f"[{survey}] PARTIAL "
              f"{ {s: f'{pos[s]}/{n_tot}' for s in streams} } — re-run.")
        return

    # ---- complete: validate against robust_raw_eval.csv, then persist ----
    yt = y[valid]
    l5_cl = preds["l5_clean"][valid]; l5_ad = preds["l5_adv"][valid]
    cnn_cl = preds["cnn_clean"][valid]; cnn_ad = preds["cnn_adv"][valid]
    acc_cl = (l5_cl == yt).mean(); acc_ad = (l5_ad == yt).mean()
    ref = pd.read_csv(mout / "robust_raw_eval.csv")
    ref = ref[ref.scope == "all"].iloc[0]
    print(f"[{survey}] L5 clean={acc_cl:.6f} (ref {ref.acc_clean:.6f})  "
          f"adv={acc_ad:.6f} (ref {ref.acc_adv:.6f})")
    assert abs(acc_cl - float(ref.acc_clean)) < 1e-4, "clean mismatch vs canonical"
    assert abs(acc_ad - float(ref.acc_adv)) < 1e-4, "adv mismatch vs canonical"
    print(f"[{survey}] CNN clean={(cnn_cl == yt).mean():.6f}  "
          f"adv={(cnn_ad == yt).mean():.6f}")

    np.savez_compressed(
        mout / "per_example_preds.npz",
        y_true=yt, labels=np.array(cfg["labels"]),
        l5_clean=l5_cl, l5_adv=l5_ad, cnn_clean=cnn_cl, cnn_adv=cnn_ad)
    for mdl, split, yp in [("L5-raw", "clean", l5_cl), ("L5-raw", "adv", l5_ad),
                           ("CNN", "clean", cnn_cl), ("CNN", "adv", cnn_ad)]:
        np.save(mout / f"confusion_{mdl}_{split}.npy", confusion(yt, yp))
    prog_f.unlink()
    print(f"[{survey}] DONE predict — per_example_preds.npz written.")


# --------------------------------------------------------------------------- #
# Stage 2: bootstrap from saved predictions (identical math/RNG)
# --------------------------------------------------------------------------- #
def stage_bootstrap(survey: str) -> None:
    mout = METRICS / survey
    z = np.load(mout / "per_example_preds.npz", allow_pickle=True)
    yt = z["y_true"]; labels = [str(x) for x in z["labels"]]
    preds = {("L5-raw", "clean"): z["l5_clean"], ("L5-raw", "adv"): z["l5_adv"],
             ("CNN", "clean"): z["cnn_clean"], ("CNN", "adv"): z["cnn_adv"]}
    n = len(yt)

    point = {k: metrics_from_cm(confusion(yt, yp)) for k, yp in preds.items()}

    rng = np.random.default_rng(SEED)
    scal = {key: np.zeros((B, 4)) for key in preds}
    pc = {key: {m: np.zeros((B, K_CLASSES))
                for m in ("precision", "recall", "f1")} for key in preds}
    drop = {mdl: np.zeros((B, 4)) for mdl in ("L5-raw", "CNN")}
    gap = np.zeros((B, 4)); gap_adv = np.zeros((B, 4))

    for b in range(B):
        idx = rng.integers(0, n, size=n)
        yt_b = yt[idx]
        cur = {}
        for key, yp in preds.items():
            m = metrics_from_cm(confusion(yt_b, yp[idx]))
            scal[key][b] = [m["acc"], m["balanced_acc"], m["macro_f1"],
                            m["weighted_f1"]]
            cur[key] = scal[key][b]
            for mm in ("precision", "recall", "f1"):
                pc[key][mm][b] = m[mm]
        drop["L5-raw"][b] = cur[("L5-raw", "clean")] - cur[("L5-raw", "adv")]
        drop["CNN"][b] = cur[("CNN", "clean")] - cur[("CNN", "adv")]
        gap[b] = cur[("L5-raw", "clean")] - cur[("CNN", "clean")]
        gap_adv[b] = cur[("L5-raw", "adv")] - cur[("CNN", "adv")]

    lo_hi = lambda a: (np.percentile(a, CI[0], axis=0),
                       np.percentile(a, CI[1], axis=0))

    rows = []
    for (mdl, split), m in point.items():
        plo, phi = lo_hi(scal[(mdl, split)])
        for i, nm in enumerate(SCALAR_NAMES):
            rows.append(dict(survey=survey, model=mdl, split=split, scope="all",
                             metric=nm, point=m[nm], lo=plo[i], hi=phi[i]))
        for mm in ("precision", "recall", "f1"):
            clo, chi = lo_hi(pc[(mdl, split)][mm])
            for c, lab in enumerate(labels):
                rows.append(dict(survey=survey, model=mdl, split=split,
                                 scope=lab, metric=mm, point=m[mm][c],
                                 lo=clo[c], hi=chi[c]))
    pd.DataFrame(rows).to_csv(mout / "bootstrap_metrics.csv", index=False)

    drows = []
    for mdl in ("L5-raw", "CNN"):
        pt = (np.array([point[(mdl, "clean")][k] for k in SCALAR_NAMES])
              - np.array([point[(mdl, "adv")][k] for k in SCALAR_NAMES]))
        dlo, dhi = lo_hi(drop[mdl])
        for i, nm in enumerate(SCALAR_NAMES):
            drows.append(dict(survey=survey,
                              comparison=f"{mdl} clean-adv (drop)",
                              metric=nm, point=pt[i], lo=dlo[i], hi=dhi[i]))
    for label, arr, sa, sb in [("L5-raw - CNN (clean)", gap, "clean", "clean"),
                               ("L5-raw - CNN (adv)", gap_adv, "adv", "adv")]:
        pt = (np.array([point[("L5-raw", sa)][k] for k in SCALAR_NAMES])
              - np.array([point[("CNN", sb)][k] for k in SCALAR_NAMES]))
        glo, ghi = lo_hi(arr)
        for i, nm in enumerate(SCALAR_NAMES):
            drows.append(dict(survey=survey, comparison=label, metric=nm,
                              point=pt[i], lo=glo[i], hi=ghi[i]))
    pd.DataFrame(drows).to_csv(mout / "bootstrap_diffs.csv", index=False)

    a = point[("L5-raw", "clean")]["acc"]; b_ = point[("L5-raw", "adv")]["acc"]
    print(f"[{survey}] bootstrap done: L5 acc {100*a:.1f} -> {100*b_:.1f}; "
          f"CNN {100*point[('CNN','clean')]['acc']:.1f} -> "
          f"{100*point[('CNN','adv')]['acc']:.1f}  (B={B})")


def stage_combine() -> None:
    frames = [pd.read_csv(METRICS / s / "bootstrap_metrics.csv")
              for s in SURVEYS]
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(METRICS / "bootstrap_metrics_all.csv", index=False)
    print(f"combined: {len(out)} rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["predict", "bootstrap", "combine"])
    ap.add_argument("--survey", default="ogle")
    args = ap.parse_args()
    if args.stage == "predict":
        stage_predict(args.survey)
    elif args.stage == "bootstrap":
        stage_bootstrap(args.survey)
    else:
        stage_combine()


if __name__ == "__main__":
    main()
