"""Component-attribution analysis for the OGLE robust raw encoder.

This script is intentionally evaluation-only: it does not train models.
It loads existing checkpoints, evaluates them on the canonical OGLE clean
and transferred-adversarial test split, and quantifies whether different
loss components fail on the same objects or rescue different subsets.

Outputs:
  outputs/metrics/ogle/component_attribution_summary.csv
  outputs/metrics/ogle/component_attribution_by_class.csv
  outputs/metrics/ogle/component_pairwise_overlap.csv
  outputs/metrics/ogle/component_l2_rescue_by_class.csv
  outputs/metrics/ogle/component_l2_rescue_context.csv
  outputs/predictions/ogle/component_attribution_predictions.csv
  outputs/tables/ogle/component_attribution_summary.tex
  outputs/tables/ogle/component_l2_overlap.tex
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score

from src.models.robust_raw_encoder import _Sampling


REPO = Path(__file__).resolve().parents[2]
ADV_ROOT = REPO / "runs/adversarial_dataset"
METRICS = REPO / "outputs/metrics/ogle"
TABLES = REPO / "outputs/tables/ogle"
PREDICTIONS = REPO / "outputs/predictions/ogle"
BATCH = 512
LABELS = ["ecl", "rrlyr", "cep", "lpv"]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    path: str
    suite: str
    description: str
    use_adv_ce: bool
    use_trades: bool
    use_supcon: bool
    use_vib: bool


MODEL_SPECS: dict[str, ModelSpec] = {
    "l1": ModelSpec(
        "l1", "L1", "runs/robust_raw_l1_ce",
        "incremental", "CE only",
        False, False, False, False,
    ),
    "l2": ModelSpec(
        "l2", "L2", "runs/robust_raw_l2_advce",
        "incremental", "CE + AdvCE",
        True, False, False, False,
    ),
    "l3": ModelSpec(
        "l3", "L3", "runs/robust_raw_l3_trades",
        "incremental", "CE + AdvCE + TRADES",
        True, True, False, False,
    ),
    "l4": ModelSpec(
        "l4", "L4", "runs/robust_raw_l4_supcon",
        "incremental", "CE + AdvCE + TRADES + SupCon",
        True, True, True, False,
    ),
    "l5": ModelSpec(
        "l5", "L5", "runs/robust_raw_l5_ogle",
        "incremental", "CE + AdvCE + TRADES + SupCon + VIB",
        True, True, True, True,
    ),
    "single_trades": ModelSpec(
        "single_trades", "TRADES-only", "runs/robust_raw_component_single_trades",
        "single_component", "CE + TRADES, no AdvCE",
        False, True, False, False,
    ),
    "single_supcon": ModelSpec(
        "single_supcon", "SupCon-only", "runs/robust_raw_component_single_supcon",
        "single_component", "CE + SupCon, no AdvCE/TRADES",
        False, False, True, False,
    ),
    "loo_no_adv_ce": ModelSpec(
        "loo_no_adv_ce", "Aux-no-AdvCE",
        "runs/robust_raw_component_loo_no_adv_ce",
        "auxiliary_without_adv_ce", "CE + TRADES + SupCon + VIB, no AdvCE",
        False, True, True, True,
    ),
    "vib_only_b5e-04": ModelSpec(
        "vib_only_b5e-04", "VIB-only", "runs/vib_only_b5e-04",
        "single_component", "CE + VIB (beta=5e-4), no adversarial component",
        False, False, False, True,
    ),
}


DEFAULT_MODELS = (
    "l1,l2,l3,l4,l5,"
    "single_trades,single_supcon,loo_no_adv_ce,vib_only_b5e-04"
)


def _valid_mask(sp: np.lib.npyio.NpzFile) -> np.ndarray:
    if "valid_te" in sp.files:
        return sp["valid_te"].astype(bool)
    return sp["valid"].astype(bool)


def _make_inputs(sp: np.lib.npyio.NpzFile):
    sd_m = float(sp["sd_m"])
    x_t = sp["X1_te"].astype(np.float32)
    sigma = (sp["DE_te"].squeeze(-1) / max(abs(sd_m), 1e-8)).astype(np.float32)
    x_m_clean = np.concatenate(
        [sp["X2_te"], sigma[..., None]], axis=-1
    ).astype(np.float32)
    x_m_adv = np.concatenate(
        [sp["X2_te_adv"], sigma[..., None]], axis=-1
    ).astype(np.float32)
    y = sp["y_te"].astype(np.int64)
    valid = _valid_mask(sp)
    return x_t, x_m_clean, x_m_adv, y, valid


def _load_checkpoint(spec: ModelSpec):
    model_dir = REPO / spec.path
    encoder_path = model_dir / "encoder_backbone.keras"
    classifier_path = model_dir / "classifier.keras"
    missing = [
        str(path) for path in (encoder_path, classifier_path) if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint files for {spec.key}: {missing}")
    encoder = tf.keras.models.load_model(
        encoder_path,
        custom_objects={"_Sampling": _Sampling},
        compile=False,
        safe_mode=False,
    )
    classifier = tf.keras.models.load_model(
        classifier_path,
        custom_objects={"_Sampling": _Sampling},
        compile=False,
        safe_mode=False,
    )
    return encoder, classifier


def _predict(encoder, classifier, x_t, x_m, batch_size: int):
    n = len(x_t)
    pred = np.zeros(n, np.int64)
    conf = np.zeros(n, np.float32)
    margin = np.zeros(n, np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        mu, _, _ = encoder([x_t[start:end], x_m[start:end]], training=False)
        prob = classifier(mu, training=False).numpy()
        order = np.sort(prob, axis=1)
        pred[start:end] = prob.argmax(axis=1)
        conf[start:end] = prob.max(axis=1)
        margin[start:end] = order[:, -1] - order[:, -2]
    return pred, conf, margin


def _context_features(sp: np.lib.npyio.NpzFile, valid: np.ndarray) -> pd.DataFrame:
    n_sel = sp["n_sel_te"].astype(np.int64)
    e = sp["E_te"].squeeze(-1).astype(np.float32)
    x2 = sp["X2_te"].squeeze(-1).astype(np.float32)
    x2_adv = sp["X2_te_adv"].squeeze(-1).astype(np.float32)
    ids = sp["ids_te"] if "ids_te" in sp.files else np.arange(len(n_sel))

    idx = np.arange(e.shape[1])[None, :]
    mask = idx < n_sel[:, None]
    denom = np.maximum(n_sel, 1).astype(np.float32)

    mean_error = (np.where(mask, e, 0.0).sum(axis=1) / denom).astype(np.float32)
    rms_dm = np.sqrt((np.where(mask, x2, 0.0) ** 2).sum(axis=1) / denom)
    mean_abs_adv_delta = (
        np.where(mask, np.abs(x2_adv - x2), 0.0).sum(axis=1) / denom
    )

    out = pd.DataFrame({
        "test_index": np.arange(len(n_sel), dtype=np.int64),
        "object_id": ids.astype(str),
        "valid": valid,
        "n_obs": n_sel,
        "mean_error": mean_error,
        "rms_delta_m_std": rms_dm.astype(np.float32),
        "mean_abs_adv_delta_std": mean_abs_adv_delta.astype(np.float32),
    })
    return out


def _safe_qcut(series: pd.Series, prefix: str) -> pd.Series:
    try:
        bins = pd.qcut(series, q=4, duplicates="drop").astype(str)
        return prefix + "_" + bins
    except ValueError:
        return pd.Series([f"{prefix}_constant"] * len(series), index=series.index)


def _summaries(preds: dict, y: np.ndarray, valid: np.ndarray):
    rows = []
    class_rows = []
    for key, result in preds.items():
        spec: ModelSpec = result["spec"]
        clean_correct = result["pred_clean"] == y
        adv_correct = result["pred_adv"] == y
        acc_clean = float(accuracy_score(y[valid], result["pred_clean"][valid]))
        acc_adv = float(accuracy_score(y[valid], result["pred_adv"][valid]))
        rows.append({
            "model": key,
            "label": spec.label,
            "suite": spec.suite,
            "description": spec.description,
            "use_adv_ce": spec.use_adv_ce,
            "use_trades": spec.use_trades,
            "use_supcon": spec.use_supcon,
            "use_vib": spec.use_vib,
            "acc_clean": acc_clean,
            "acc_adv": acc_adv,
            "drop": acc_clean - acc_adv,
            "n_test": int(valid.sum()),
        })
        for class_id, label in enumerate(LABELS):
            mask = valid & (y == class_id)
            class_rows.append({
                "model": key,
                "label": spec.label,
                "class": label,
                "n_test": int(mask.sum()),
                "acc_clean": float(accuracy_score(y[mask], result["pred_clean"][mask])),
                "acc_adv": float(accuracy_score(y[mask], result["pred_adv"][mask])),
                "drop": float(clean_correct[mask].mean() - adv_correct[mask].mean()),
            })
    return pd.DataFrame(rows), pd.DataFrame(class_rows)


def _pairwise_overlap(preds: dict, y: np.ndarray, valid: np.ndarray):
    rows = []
    keys = list(preds)
    for ref in keys:
        ref_correct = (preds[ref]["pred_adv"] == y) & valid
        ref_wrong = (~ref_correct) & valid
        for other in keys:
            if other == ref:
                continue
            other_correct = (preds[other]["pred_adv"] == y) & valid
            other_wrong = (~other_correct) & valid
            both_correct = int((ref_correct & other_correct).sum())
            ref_only = int((ref_correct & other_wrong).sum())
            other_only = int((ref_wrong & other_correct).sum())
            both_wrong = int((ref_wrong & other_wrong).sum())
            union_errors = int((ref_wrong | other_wrong).sum())
            shared_errors = int((ref_wrong & other_wrong).sum())
            rows.append({
                "reference": ref,
                "comparator": other,
                "n_test": int(valid.sum()),
                "both_correct_adv": both_correct,
                "reference_only_correct_adv": ref_only,
                "comparator_only_correct_adv": other_only,
                "both_wrong_adv": both_wrong,
                "reference_errors_adv": int(ref_wrong.sum()),
                "comparator_errors_adv": int(other_wrong.sum()),
                "shared_errors_adv": shared_errors,
                "error_jaccard": (
                    shared_errors / union_errors if union_errors else np.nan
                ),
                "comparator_rescues_reference": other_only,
                "comparator_breaks_reference": ref_only,
                "net_rescue_vs_reference": other_only - ref_only,
            })
    return pd.DataFrame(rows)


def _l2_rescue_tables(preds: dict, y: np.ndarray, valid: np.ndarray,
                      context: pd.DataFrame):
    if "l2" not in preds:
        return pd.DataFrame(), pd.DataFrame()

    l2_correct = (preds["l2"]["pred_adv"] == y) & valid
    valid_context = context.loc[valid].copy()
    valid_context["n_obs_bin"] = _safe_qcut(valid_context["n_obs"], "nobs")
    valid_context["mean_error_bin"] = _safe_qcut(
        valid_context["mean_error"], "err"
    )
    valid_context["rms_delta_m_bin"] = _safe_qcut(
        valid_context["rms_delta_m_std"], "rmsdm"
    )

    by_class = []
    by_context = []
    for key, result in preds.items():
        if key == "l2":
            continue
        other_correct = (result["pred_adv"] == y) & valid
        rescued = (~l2_correct) & other_correct & valid
        broken = l2_correct & (~other_correct) & valid
        for class_id, label in enumerate(LABELS):
            class_mask = valid & (y == class_id)
            by_class.append({
                "comparator": key,
                "class": label,
                "n_class": int(class_mask.sum()),
                "l2_errors": int((class_mask & ~l2_correct).sum()),
                "rescued_l2_errors": int((class_mask & rescued).sum()),
                "broken_l2_correct": int((class_mask & broken).sum()),
                "net_rescue": int((class_mask & rescued).sum())
                - int((class_mask & broken).sum()),
            })

        tmp = valid_context.copy()
        tmp["rescued"] = rescued[valid]
        tmp["broken"] = broken[valid]
        tmp["l2_error"] = (~l2_correct)[valid]
        tmp["comparator"] = key
        for col in ("n_obs_bin", "mean_error_bin", "rms_delta_m_bin"):
            grouped = tmp.groupby(["comparator", col], dropna=False)
            for (comparator, bin_name), group in grouped:
                by_context.append({
                    "comparator": comparator,
                    "context": col,
                    "bin": bin_name,
                    "n_bin": int(len(group)),
                    "l2_errors": int(group["l2_error"].sum()),
                    "rescued_l2_errors": int(group["rescued"].sum()),
                    "broken_l2_correct": int(group["broken"].sum()),
                    "rescue_rate_among_l2_errors": (
                        float(group["rescued"].sum() / group["l2_error"].sum())
                        if group["l2_error"].sum() else np.nan
                    ),
                })
    return pd.DataFrame(by_class), pd.DataFrame(by_context)


def _prediction_frame(preds: dict, y: np.ndarray, valid: np.ndarray,
                      context: pd.DataFrame) -> pd.DataFrame:
    out = context.copy()
    out["y_true"] = y
    out["label_true"] = [LABELS[int(v)] for v in y]
    out["valid"] = valid
    for key, result in preds.items():
        out[f"{key}_pred_clean"] = result["pred_clean"]
        out[f"{key}_pred_adv"] = result["pred_adv"]
        out[f"{key}_correct_clean"] = result["pred_clean"] == y
        out[f"{key}_correct_adv"] = result["pred_adv"] == y
        out[f"{key}_conf_clean"] = result["conf_clean"]
        out[f"{key}_conf_adv"] = result["conf_adv"]
        out[f"{key}_margin_clean"] = result["margin_clean"]
        out[f"{key}_margin_adv"] = result["margin_adv"]
    return out


def _write_latex(summary: pd.DataFrame, pairwise: pd.DataFrame):
    TABLES.mkdir(parents=True, exist_ok=True)
    display = summary[[
        "label", "description", "acc_clean", "acc_adv", "drop"
    ]].copy()
    display["drop_pp"] = 100.0 * display["drop"]
    display = display.drop(columns=["drop"])
    display.to_latex(
        TABLES / "component_attribution_summary.tex",
        index=False,
        float_format="%.3f",
        escape=False,
    )

    if not pairwise.empty and (pairwise["reference"] == "l2").any():
        cols = [
            "comparator", "reference_errors_adv", "comparator_errors_adv",
            "comparator_rescues_reference", "comparator_breaks_reference",
            "net_rescue_vs_reference", "error_jaccard",
        ]
        pairwise.loc[pairwise["reference"] == "l2", cols].to_latex(
            TABLES / "component_l2_overlap.tex",
            index=False,
            float_format="%.3f",
            escape=False,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--batch-size", type=int, default=BATCH)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--no-per-example", action="store_true")
    args = parser.parse_args()

    selected = [item.strip() for item in args.models.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(MODEL_SPECS))
    if unknown:
        raise ValueError(f"Unknown model keys: {unknown}")

    METRICS.mkdir(parents=True, exist_ok=True)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    sp = np.load(ADV_ROOT / "test_split.npz", allow_pickle=True)
    x_t, x_m_clean, x_m_adv, y, valid = _make_inputs(sp)
    context = _context_features(sp, valid)

    preds = {}
    for key in selected:
        spec = MODEL_SPECS[key]
        print(f"\n=== {key}: {spec.description} ===")
        try:
            tf.keras.backend.clear_session()
            encoder, classifier = _load_checkpoint(spec)
        except FileNotFoundError:
            if args.skip_missing:
                print(f"  missing checkpoint; skipping {key}")
                continue
            raise
        pred_clean, conf_clean, margin_clean = _predict(
            encoder, classifier, x_t, x_m_clean, args.batch_size
        )
        pred_adv, conf_adv, margin_adv = _predict(
            encoder, classifier, x_t, x_m_adv, args.batch_size
        )
        acc_clean = accuracy_score(y[valid], pred_clean[valid])
        acc_adv = accuracy_score(y[valid], pred_adv[valid])
        print(
            f"  clean={acc_clean:.4f} adv={acc_adv:.4f} "
            f"drop={100.0 * (acc_clean - acc_adv):+.2f} pp"
        )
        preds[key] = {
            "spec": spec,
            "pred_clean": pred_clean,
            "pred_adv": pred_adv,
            "conf_clean": conf_clean,
            "conf_adv": conf_adv,
            "margin_clean": margin_clean,
            "margin_adv": margin_adv,
        }

    summary, by_class = _summaries(preds, y, valid)
    pairwise = _pairwise_overlap(preds, y, valid)
    rescue_by_class, rescue_context = _l2_rescue_tables(preds, y, valid, context)

    summary.to_csv(METRICS / "component_attribution_summary.csv", index=False)
    by_class.to_csv(METRICS / "component_attribution_by_class.csv", index=False)
    pairwise.to_csv(METRICS / "component_pairwise_overlap.csv", index=False)
    rescue_by_class.to_csv(
        METRICS / "component_l2_rescue_by_class.csv", index=False
    )
    rescue_context.to_csv(
        METRICS / "component_l2_rescue_context.csv", index=False
    )
    if not args.no_per_example:
        _prediction_frame(preds, y, valid, context).to_csv(
            PREDICTIONS / "component_attribution_predictions.csv", index=False
        )
    _write_latex(summary, pairwise)

    print("\nWrote:")
    for path in [
        METRICS / "component_attribution_summary.csv",
        METRICS / "component_attribution_by_class.csv",
        METRICS / "component_pairwise_overlap.csv",
        METRICS / "component_l2_rescue_by_class.csv",
        METRICS / "component_l2_rescue_context.csv",
        TABLES / "component_attribution_summary.tex",
        TABLES / "component_l2_overlap.tex",
    ]:
        print(f"  {path}")
    if not args.no_per_example:
        print(f"  {PREDICTIONS / 'component_attribution_predictions.csv'}")


if __name__ == "__main__":
    main()
