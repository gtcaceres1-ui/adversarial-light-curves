"""Square Attack against the 16-feature FATS-style Random Forest baseline.

Unlike attack_rf_square_ogle.py (which attacks a simplified 7-feature RF on
truncated Delta-m sequences), this attacks runs/rf_ogle/model.joblib, the
random forest reported in the paper, by perturbing the full raw light curve
within the photometric threat model

    |m_adv[i] - m_clean[i]| <= k * e[i]   for every observation i,

and recomputing the 16 features (src.metrics.lightcurve_features) at every
query. Score-based black-box random search (Andriushchenko et al. 2020 style):
each iteration proposes, for every still-active curve, new boundary values
(+/- k e) on a shrinking random subset of points; the proposal is kept when
the true-class probability decreases (or the prediction flips).

State is persisted between invocations so the attack can be resumed.

Stages:
  --build-cache N   stratified subsample of the valid test rows, load full
                    curves once, store ragged arrays in runs/rf_fats_attack/
  --attack          run iterations for ~--budget-s seconds, then checkpoint
  --report          write outputs/metrics/ogle/rf_fats_square_attack_k{K}.csv

Determinism: iteration t uses default_rng(seed*100003 + t), so resuming
reproduces the exact same proposal stream regardless of chunking.
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.metrics.lightcurve_features import features_from_arrays  # noqa: E402

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
CATALOG = REPO / "Data/OGLE-III/OGLE_dataset.dat"
LCS_DIR = REPO / "Data/OGLE-III/LCs"
SPLIT = REPO / "runs/adversarial_dataset/test_split.npz"
RF_PATH = REPO / "runs/rf_ogle/model.joblib"
WORK = REPO / "runs/rf_fats_attack"
METRICS = REPO / "outputs/metrics/ogle"

LABELS = ["ecl", "rrlyr", "cep", "lpv"]
SEED = 42
P_INIT = 0.05
N_ITER = 1500
T_PROCESS_START = time.time()


def fast_features(m: np.ndarray, e: np.ndarray, keys) -> np.ndarray:
    """Exact clone of features_from_arrays() with one shared sort.

    Verified bit-compatible (atol 1e-10) against the reference implementation
    at cache-build time; falls back to it for n < 2.
    """
    n = m.size
    if n < 2:
        ref = features_from_arrays(m, e)
        return np.array([ref[k] for k in keys])
    w = 1.0 / np.maximum(e * e, 1e-12)
    sw = w.sum()
    wm = float((w * m).sum() / sw)
    wvar = float((w * (m - wm) ** 2).sum() / sw)
    wstd = float(np.sqrt(max(wvar, 0.0)))
    delta = (m - wm) / np.maximum(e, 1e-12)
    num = np.abs(delta).mean()
    den = np.sqrt((delta * delta).mean() + 1e-12)
    stetson_k = float(num / den) if den > 0 else 0.0
    diffs = np.diff(m)
    var_total = float(np.var(m))
    eta = float((diffs * diffs).mean() / var_total) if var_total > 0 else 0.0
    mu = float(m.mean())
    sd = float(m.std()) + 1e-12
    z = (m - mu) / sd
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean() - 3.0)
    ms = np.sort(m)
    amp = float(0.5 * (ms[-1] - ms[0]))

    def q(p):
        pos = p / 100.0 * (n - 1)
        f = int(pos)
        frac = pos - f
        return float(ms[f] + frac * (ms[min(f + 1, n - 1)] - ms[f])) \
            if frac > 0 else float(ms[f])

    p10, p50, p90 = q(10), q(50), q(90)
    iqr = q(75) - q(25)
    mad = float(np.median(np.abs(m - p50)))
    n10 = max(int(0.1 * n), 1)
    slope_p10 = float(ms[n10] - ms[0])
    slope_p90 = float(ms[-1] - ms[-1 - n10])
    beyond1std = float(np.mean(np.abs(m - mu) > sd))
    vals = {
        "n_obs": float(n), "amplitude": amp, "weighted_mean": wm,
        "weighted_std": wstd, "stetson_k": stetson_k, "eta": eta,
        "kurtosis": kurt, "skew": skew, "slope_p10": slope_p10,
        "slope_p90": slope_p90, "beyond1std": beyond1std, "mad": mad,
        "median": p50, "iqr": iqr, "mean_err": float(e.mean()),
        "median_err": float(np.median(e)),
    }
    out = np.array([vals[k] for k in keys])
    return np.where(np.isfinite(out), out, 0.0)


def load_lightcurve(path: Path) -> pd.DataFrame:
    curve = pd.read_csv(path, sep=r"\s+", comment="#", header=None,
                        names=["time", "magnitude", "error"])
    return (curve.dropna().sort_values("time")
            .drop_duplicates(subset="time", keep="first"))


def canonical_test_df():
    """Rebuild the catalog/test split (verified identical to ids_te)."""
    cat = pd.read_csv(CATALOG)
    cat.columns = [c.strip() for c in cat.columns]
    cat["label4"] = cat["Path"].apply(
        lambda p: next((l for l in LABELS
                        if f"/{l}/" in str(p).lower().replace("\\", "/")),
                       None))
    cat["lc_file"] = cat["ID"].apply(lambda v: LCS_DIR / f"{v}.dat")
    cat = cat.dropna(subset=["label4"]).reset_index(drop=True)
    parts = []
    for l in LABELS:
        part = cat[cat["label4"] == l]
        if len(part) > 20000:
            part = part.sample(20000, random_state=SEED)
        parts.append(part)
    df = (pd.concat(parts).sample(frac=1.0, random_state=SEED)
          .reset_index(drop=True))
    sp = np.load(SPLIT, allow_pickle=True)
    idx_te = sp["idx_test"].astype(int)
    test_df = df.iloc[idx_te].reset_index(drop=True)
    ids_te = np.asarray(sp["ids_te"]).astype(str)
    assert (test_df["ID"].astype(str).to_numpy() == ids_te).all(), \
        "catalog/test split mismatch"
    return test_df, sp


def rf_and_keys():
    d = joblib.load(RF_PATH)
    return d["clf"], d["feature_keys"]


def featurize_rows(ms: list[np.ndarray], es: list[np.ndarray], keys) -> np.ndarray:
    return np.array([fast_features(m, e, keys) for m, e in zip(ms, es)],
                    dtype=np.float64)


def build_cache(n_sub: int) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    test_df, sp = canonical_test_df()
    y_te = sp["y_te"].astype(int)
    valid = sp["valid"].astype(bool)
    rng = np.random.default_rng(SEED)
    per_class = n_sub // len(LABELS)
    chosen = []
    for c in range(len(LABELS)):
        cand = np.where(valid & (y_te == c))[0]
        chosen.append(rng.choice(cand, min(per_class, len(cand)),
                                 replace=False))
    rows = np.sort(np.concatenate(chosen))

    ms, es, keep = [], [], []
    for r in rows:
        try:
            lc = load_lightcurve(Path(test_df.iloc[r]["lc_file"]))
            if len(lc) < 3:
                continue
            ms.append(lc["magnitude"].to_numpy(np.float64))
            es.append(lc["error"].to_numpy(np.float64))
            keep.append(r)
        except Exception:
            continue
    keep = np.array(keep)
    # parity check: fast featurizer must match the reference exactly
    _, fkeys = (lambda d: (d, d["feature_keys"]))(joblib.load(RF_PATH))
    for m, e in zip(ms[:50], es[:50]):
        ref = features_from_arrays(m, e)
        ref_v = np.array([ref[k] for k in fkeys])
        fast_v = fast_features(m, e, fkeys)
        assert np.allclose(ref_v, fast_v, atol=1e-10), "featurizer mismatch"
    print("parity fast_features vs reference: OK (50 curvas)")
    lens = np.array([len(m) for m in ms])
    off = np.concatenate([[0], np.cumsum(lens)])
    np.savez(WORK / "cache.npz",
             m=np.concatenate(ms), e=np.concatenate(es), off=off,
             rows=keep, y=y_te[keep],
             ids=test_df.iloc[keep]["ID"].astype(str).to_numpy())
    print(f"cache: {len(keep)} curvas, {lens.sum()} obs, "
          f"lens p50/p90/max = {np.percentile(lens, 50):.0f}/"
          f"{np.percentile(lens, 90):.0f}/{lens.max()}")


class State:
    def __init__(self, k: float):
        self.path = WORK / f"state_k{k:.2f}.npz"
        self.k = k

    def exists(self) -> bool:
        return self.path.exists()

    def save(self, m_adv, true_p, preds, t, n_query, t_success):
        np.savez(self.path, m_adv=m_adv, true_p=true_p, preds=preds,
                 t=t, n_query=n_query, t_success=t_success)

    def load(self):
        z = np.load(self.path)
        return (z["m_adv"], z["true_p"], z["preds"], int(z["t"]),
                z["n_query"], z["t_success"])


def split_rows(cache):
    off = cache["off"]
    m, e = cache["m"], cache["e"]
    return ([m[off[i]:off[i + 1]] for i in range(len(off) - 1)],
            [e[off[i]:off[i + 1]] for i in range(len(off) - 1)])


def attack(k: float, budget_s: float, n_iter: int) -> None:
    cache = np.load(WORK / "cache.npz", allow_pickle=True)
    y = cache["y"].astype(int)
    ms, es = split_rows(cache)
    B = len(y)
    clf, keys = rf_and_keys()
    st = State(k)

    if st.exists():
        m_flat, true_p, preds, t0_iter, n_query, t_success = st.load()
        off = cache["off"]
        m_adv = [m_flat[off[i]:off[i + 1]].copy() for i in range(B)]
    else:
        # iteration 0: random boundary init on every point
        rng = np.random.default_rng(SEED * 100003)
        m_adv = [m + rng.choice([-1.0, 1.0], size=len(m)) * k * np.abs(e)
                 for m, e in zip(ms, es)]
        F = featurize_rows(m_adv, es, keys)
        probs = clf.predict_proba(F)
        true_p = probs[np.arange(B), y]
        preds = probs.argmax(1)
        # never accept an init worse than clean: where clean already correct
        # keep init only as a starting point (standard square-attack practice)
        t0_iter = 0
        n_query = np.ones(B, dtype=np.int64)
        t_success = np.where(preds != y, 0, -1).astype(np.int64)
        st.save(np.concatenate(m_adv), true_p, preds, 0, n_query, t_success)
        print(f"init k={k}: success {np.mean(preds != y):.4f}")

    t = t0_iter
    while t < n_iter and (time.time() - T_PROCESS_START) < budget_s:
        t += 1
        rng = np.random.default_rng(SEED * 100003 + t)
        active = np.where(preds == y)[0]
        if len(active) == 0:
            break
        p_t = P_INIT * (1.0 - t / n_iter)
        prop_ms, prop_es = [], []
        positions = []
        for i in active:
            n = len(ms[i])
            npert = max(1, int(round(p_t * n)))
            pos = rng.choice(n, npert, replace=False)
            cand = m_adv[i].copy()
            cand[pos] = (ms[i][pos]
                         + rng.choice([-1.0, 1.0], size=npert)
                         * k * np.abs(es[i][pos]))
            prop_ms.append(cand)
            prop_es.append(es[i])
            positions.append(pos)
        F = featurize_rows(prop_ms, prop_es, keys)
        probs = clf.predict_proba(F)
        tp = probs[np.arange(len(active)), y[active]]
        pr = probs.argmax(1)
        improved = (tp < true_p[active]) | (pr != y[active])
        for j, i in enumerate(active):
            n_query[i] += 1
            if improved[j]:
                m_adv[i] = prop_ms[j]
                true_p[i] = tp[j]
                preds[i] = pr[j]
                if pr[j] != y[i] and t_success[i] < 0:
                    t_success[i] = t

    st.save(np.concatenate(m_adv), true_p, preds, t, n_query, t_success)
    acc = float(np.mean(preds == y))
    print(f"k={k} iter={t}/{n_iter} acc_adv={acc:.4f} "
          f"active={int(np.sum(preds == y))} "
          f"elapsed={time.time() - T_PROCESS_START:.0f}s")


def report(k: float) -> None:
    cache = np.load(WORK / "cache.npz", allow_pickle=True)
    y = cache["y"].astype(int)
    ms, es = split_rows(cache)
    clf, keys = rf_and_keys()
    st = State(k)
    m_flat, true_p, preds, t, n_query, t_success = st.load()

    F_clean = featurize_rows(ms, es, keys)
    pred_clean = clf.predict(F_clean)
    acc_clean = float(np.mean(pred_clean == y))
    acc_adv = float(np.mean(preds == y))

    METRICS.mkdir(parents=True, exist_ok=True)
    rows = [{"attack": "clean", "k": 0.0, "model": "RF-FATS16",
             "acc": acc_clean, "n_eval": len(y), "n_iter": 0}]
    rows.append({"attack": "square_fats", "k": k, "model": "RF-FATS16",
                 "acc": acc_adv, "n_eval": len(y), "n_iter": t})
    per_class = []
    for c, lab in enumerate(LABELS):
        m = y == c
        per_class.append({
            "class": lab, "n": int(m.sum()),
            "acc_clean": float(np.mean(pred_clean[m] == y[m])),
            "acc_adv": float(np.mean(preds[m] == y[m])),
            "median_queries_success": float(
                np.median(n_query[m & (t_success >= 0)]))
            if (m & (t_success >= 0)).any() else np.nan,
        })
    pd.DataFrame(rows).to_csv(
        METRICS / f"rf_fats_square_attack_k{k:.2f}.csv", index=False)
    pd.DataFrame(per_class).to_csv(
        METRICS / f"rf_fats_square_attack_k{k:.2f}_per_class.csv", index=False)
    print(f"clean={acc_clean:.4f}  adv(k={k})={acc_adv:.4f}  "
          f"drop={(acc_clean - acc_adv) * 100:+.1f} pp  iters={t}")
    print(pd.DataFrame(per_class).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-cache", type=int, default=None)
    ap.add_argument("--attack", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--k", type=float, default=0.5)
    ap.add_argument("--budget-s", type=float, default=30.0)
    ap.add_argument("--n-iter", type=int, default=N_ITER)
    args = ap.parse_args()
    if args.build_cache:
        build_cache(args.build_cache)
    elif args.attack:
        attack(args.k, args.budget_s, args.n_iter)
    elif args.report:
        report(args.k)
