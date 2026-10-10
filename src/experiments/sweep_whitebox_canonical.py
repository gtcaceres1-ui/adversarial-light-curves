"""White-box PGD sweep on the baseline checkpoints.

Re-runs the white-box attacks on the trained CNN and RNN checkpoints, using
exactly the protocol that generated the adversarial test set
(adversarial_dataset_builder._attack_test: pgd_magnitude_tf, T=20, step
0.2*eps, 3 restarts, smoothness term, seeds 42+restart+start) so that the
k=0.5 CNN run reproduces the headline 0.964 -> 0.658 cell, and the k=1.0 and
RNN rows extend it under the same attack.

Targets:
  --model cnn   runs/cnn_ogle/cnn_two_tower.keras   (also evals L5 transfer)
  --model rnn   runs/rnn_ogle/model.keras

Outputs (outputs/metrics/ogle/):
  whitebox_canonical_sweep.csv          tidy, appended per run
  rf_attack_sweep.csv / rnn_attack_sweep.csv   schema consumed by
      plot_robustness_vs_k.py

Usage (needs TensorFlow + Keras 3):
  python -m src.experiments.sweep_whitebox_canonical --model cnn --k 0.5
  python -m src.experiments.sweep_whitebox_canonical --model cnn --k 1.0
  python -m src.experiments.sweep_whitebox_canonical --model rnn --k 0.5
  python -m src.experiments.sweep_whitebox_canonical --model rnn --k 1.0
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score

from src.attacks.photometric_pgd import encode_magnitude_budget, pgd_magnitude_tf
from src.experiments.adversarial_dataset_builder import (
    BATCH_SIZE, PGD_RESTARTS, PGD_STEP_FRAC, PGD_STEPS, SMOOTH_LAMBDA,
)
from src.models.robust_raw_encoder import _Sampling

REPO = Path(__file__).resolve().parents[2]
M = REPO / "outputs/metrics/ogle"
ADV = REPO / "runs/adversarial_dataset"
MODELS = {
    "cnn": REPO / "runs/cnn_ogle/cnn_two_tower.keras",
    "rnn": REPO / "runs/rnn_ogle/model.keras",
}
L5_DIR = REPO / "runs/robust_raw_l5_ogle"
LABELS = ["ecl", "rrlyr", "cep", "lpv"]


def attack(model, x1, x2, y_oh, eps_all):
    """Verbatim _attack_test protocol, parametrized target model."""
    x2_adv = x2.copy()

    def forward(dt, dm):
        return model([dt[..., None], dm[..., None]], training=False)

    for start in range(0, len(x2), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(x2))
        dt = tf.constant(x1[start:end, :, 0], tf.float32)
        dm = tf.constant(x2[start:end, :, 0], tf.float32)
        labels = tf.constant(y_oh[start:end], tf.float32)
        eps = tf.constant(eps_all[start:end], tf.float32)
        best_loss = np.full(end - start, -np.inf, np.float32)
        best_dm = x2[start:end, :, 0].copy()
        for restart in range(PGD_RESTARTS):
            tf.random.set_seed(42 + restart + start)
            dm_adv, _ = pgd_magnitude_tf(
                forward, dt, dm, labels, eps,
                tf.constant(PGD_STEPS, tf.int32),
                tf.constant(PGD_STEP_FRAC, tf.float32),
                tf.constant(True),
                tf.constant(SMOOTH_LAMBDA, tf.float32),
            )
            loss = tf.keras.losses.categorical_crossentropy(
                labels, forward(dt, dm_adv)).numpy()
            take = loss > best_loss
            best_loss[take] = loss[take]
            best_dm[take] = dm_adv.numpy()[take]
        x2_adv[start:end, :, 0] = best_dm
        if (start // BATCH_SIZE) % 10 == 0:
            print(f"  attacked {end}/{len(x2)}")
    return x2_adv


def predict_two_tower(model, x1, x2, bs=512):
    out = np.zeros(len(x1), np.int64)
    for s in range(0, len(x1), bs):
        e = min(s + bs, len(x1))
        out[s:e] = model([x1[s:e], x2[s:e]], training=False).numpy().argmax(1)
    return out


def predict_l5(enc, clf, x1, xm, bs=512):
    out = np.zeros(len(x1), np.int64)
    for s in range(0, len(x1), bs):
        e = min(s + bs, len(x1))
        mu, _, _ = enc([x1[s:e], xm[s:e]], training=False)
        out[s:e] = clf(mu, training=False).numpy().argmax(1)
    return out


def legacy_write(path: Path, rows: list[dict]) -> None:
    """Merge rows into a legacy sweep CSV (schema: attack,model,k,acc)."""
    if path.exists() and not path.with_suffix(".csv.pre_canonical.bak").exists():
        shutil.copy(path, path.with_suffix(".csv.pre_canonical.bak"))
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        keys = ["attack", "model", "k"]
        old["k"] = old.get("k", pd.Series([np.nan] * len(old))).fillna(0.0)
        merged = pd.concat([old, new]).drop_duplicates(
            subset=keys, keep="last").reset_index(drop=True)
    else:
        merged = new
    merged.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "rnn"], required=True)
    ap.add_argument("--k", type=float, required=True)
    args = ap.parse_args()

    sp = np.load(ADV / "test_split.npz", allow_pickle=True)
    x1 = sp["X1_te"].astype(np.float32)
    x2 = sp["X2_te"].astype(np.float32)
    de = sp["DE_te"].astype(np.float32)
    e_obs = sp["E_te"].squeeze(-1).astype(np.float32)
    n_sel = sp["n_sel_te"].astype(np.int32)
    y = sp["y_te"].astype(np.int64)
    valid = sp["valid_te"].astype(bool)
    sd_m = float(sp["sd_m"])

    eps_all = encode_magnitude_budget(e_obs, n_sel, sd_m, args.k)
    y_oh = tf.keras.utils.to_categorical(y, len(LABELS)).astype(np.float32)

    target = tf.keras.models.load_model(
        MODELS[args.model], compile=False, safe_mode=False)
    print(f"== white-box PGD vs {args.model.upper()}  k={args.k} "
          f"(T={PGD_STEPS}, step={PGD_STEP_FRAC}, restarts={PGD_RESTARTS})")
    x2_adv = attack(target, x1, x2, y_oh, eps_all)

    pred_cl = predict_two_tower(target, x1, x2)
    pred_ad = predict_two_tower(target, x1, x2_adv)
    acc_cl = accuracy_score(y[valid], pred_cl[valid])
    acc_ad = accuracy_score(y[valid], pred_ad[valid])
    print(f"  {args.model.upper()}: clean={acc_cl:.4f} adv={acc_ad:.4f} "
          f"drop={100*(acc_ad-acc_cl):+.1f} pp")

    rows = [dict(attack=f"{args.model}_pgd_canonical", model=args.model.upper(),
                 k=args.k, acc_clean=acc_cl, acc_adv=acc_ad,
                 drop=acc_ad - acc_cl, steps=PGD_STEPS,
                 step_frac=PGD_STEP_FRAC, restarts=PGD_RESTARTS,
                 n_valid=int(valid.sum()))]

    legacy = [dict(attack="clean", model=args.model.upper(), k=0.0, acc=acc_cl),
              dict(attack=f"{args.model}_pgd", model=args.model.upper(),
                   k=args.k, acc=acc_ad)]

    if args.model == "cnn":
        # transferred L5 on this attack (dashed curve of fig:robustness_vs_k)
        enc = tf.keras.models.load_model(
            L5_DIR / "encoder_backbone.keras",
            custom_objects={"_Sampling": _Sampling}, compile=False,
            safe_mode=False)
        clf = tf.keras.models.load_model(
            L5_DIR / "classifier.keras", compile=False, safe_mode=False)
        sigma = (de.squeeze(-1) / max(abs(sd_m), 1e-8)).astype(np.float32)
        xm_cl = np.concatenate([x2, sigma[..., None]], axis=-1)
        xm_ad = np.concatenate([x2_adv, sigma[..., None]], axis=-1)
        l5_cl = accuracy_score(y[valid], predict_l5(enc, clf, x1, xm_cl)[valid])
        l5_ad = accuracy_score(y[valid], predict_l5(enc, clf, x1, xm_ad)[valid])
        print(f"  L5 transferred: clean={l5_cl:.4f} adv={l5_ad:.4f}")
        rows.append(dict(attack="cnn_pgd_canonical_transfer", model="L5",
                         k=args.k, acc_clean=l5_cl, acc_adv=l5_ad,
                         drop=l5_ad - l5_cl, steps=PGD_STEPS,
                         step_frac=PGD_STEP_FRAC, restarts=PGD_RESTARTS,
                         n_valid=int(valid.sum())))
        legacy += [dict(attack="clean", model="L5", k=0.0, acc=l5_cl),
                   dict(attack="cnn_pgd", model="L5", k=args.k, acc=l5_ad)]
        if abs(args.k - 0.5) < 1e-9:
            ref = float(pd.read_csv(M / "baselines_clean_vs_adv.csv")
                        .query("model=='CNN'").acc_adv.iloc[0])
            print(f"  sanity vs canonical npz: {acc_ad:.4f} vs {ref:.4f} "
                  f"({'OK' if abs(acc_ad-ref) < 0.005 else 'REVISAR'})")

    out = M / "whitebox_canonical_sweep.csv"
    df = pd.DataFrame(rows)
    if out.exists():
        prev = pd.read_csv(out)
        df = pd.concat([prev, df]).drop_duplicates(
            subset=["attack", "model", "k"], keep="last")
    df.to_csv(out, index=False)

    legacy_file = "rf_attack_sweep.csv" if args.model == "cnn" \
        else "rnn_attack_sweep.csv"
    legacy_write(M / legacy_file, legacy)
    print(f"-> {out.name} y {legacy_file} actualizados.\n"
          "   Destinos .tex: fila correspondiente de tab:targeted_attacks; "
          "luego python3 src/experiments/plot_robustness_vs_k.py")


if __name__ == "__main__":
    main()
