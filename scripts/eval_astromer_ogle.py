#!/usr/bin/env python3
"""
eval_astromer_ogle.py
=====================
Evaluate ASTROMER 2 (pre-trained on MACHO) under a CNN-targeted transferred
adversarial attack, with TWO-PHASE FULL FINE-TUNING on OGLE-III:
  Phase 1 — train the MLP head with the encoder frozen (LR 1e-3);
  Phase 2 — unfreeze the encoder, fine-tune end-to-end at low LR (1e-5).
Use --skip-ft to reproduce the earlier frozen-encoder probe (69.1 → 24.9).

Run from the project root:

    pip install "tensorflow==2.14" numpy pandas scikit-learn toml tqdm requests
    python scripts/eval_astromer_ogle.py [--no-gpu] [--epochs 25] [--ft-epochs 15]
                                 [--ft-lr 1e-5] [--reuse-head] [--skip-ft]
                                 [--reuse-ft] [--adv-npz PATH] [--pt-dir PATH]

What this script does
---------------------
1. Clones the ASTROMER repo (cridonoso/astromer) into ./external/astromer/
2. Downloads pre-trained MACHO weights (pt_macho_v2_2025.zip) from Zenodo
3. Loads the OGLE-III catalog with the same seed-42 stratified 70/10/20
   protocol on a balanced subsample (max_per_class=7836). Do NOT change
   MAX_PER_CLASS: the catalog positions stored in the adversarial .npz were
   generated against this exact subsample; alignment breaks otherwise.
4. Phase 1: fine-tunes a masked-average-pool MLP head (frozen encoder).
   Phase 2 (default): unfreezes the encoder and fine-tunes everything at a
   low learning rate, early-stopped on val_loss.
5. Evaluates on the clean OGLE-III test set  → acc_clean
6. Evaluates on the CNN-targeted adversarial test set  → acc_adv
   The default npz (adv_frac_1.00.npz) was generated at k = 1.0, the
   maximal photometric budget; point --adv-npz to a k = 0.5 set to compare
   against the default budget used in the main results.
7. Reports the drop (pp) and 95% bootstrap CI.

Output
------
Printed summary + outputs/astromer_ogle_results.json

Citation for ASTROMER 2
-----------------------
Donoso-Oliva et al. (2026). "Generalizing across astronomical surveys:
Few-shot light curve classification with Astromer 2." A&A (in press).
DOI: 10.5281/zenodo.18207945
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess

# ASTROMER targets Keras 2; TensorFlow >= 2.16 ships Keras 3 by default.
# This must be set before tensorflow is imported.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# Paths (relative to project root)
# ─────────────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parent
ASTROMER_DIR = REPO_ROOT / "external" / "astromer"
PT_DIR       = REPO_ROOT / "external" / "pt_macho_v2"
OUTPUT_DIR   = REPO_ROOT / "outputs"
ADV_NPZ      = OUTPUT_DIR / "adversarial" / "ogle" / "cnn" / "adv_frac_1.00.npz"
OGLE_CATALOG = REPO_ROOT / "Data" / "OGLE-III" / "OGLE_dataset.dat"
OGLE_LCS_DIR = REPO_ROOT / "Data" / "OGLE-III" / "LCs"

ZENODO_V2_URL = (
    "https://zenodo.org/records/18207945/files/pt_macho_v2_2025.zip?download=1"
)

# ── Dataset constants (must match the paper's other models) ──────────────────
CLASSES       = ["ecl", "rrlyr", "cep", "lpv"]  # OGLE 4-class
CLASS_TO_INT  = {c: i for i, c in enumerate(CLASSES)}
MAX_PER_CLASS = 7836
SEQ_LEN       = 200
SEED          = 42
TEST_FRAC     = 0.20
VAL_FRAC      = 0.15   # fraction of *total* data

# ── Training hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE = 256
EPOCHS     = 25     # phase 1: head only, frozen encoder
PATIENCE   = 5
LR         = 1e-3   # phase 1 learning rate
FT_EPOCHS   = 15    # phase 2: encoder unfrozen
FT_LR       = 1e-5  # phase 2 learning rate (low, protects pre-trained weights)
FT_PATIENCE = 4


# ─────────────────────────────────────────────────────────────────────────────
# Step 0: clone ASTROMER repo
# ─────────────────────────────────────────────────────────────────────────────
def setup_astromer() -> None:
    if not ASTROMER_DIR.exists():
        print("Cloning ASTROMER …")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/cridonoso/astromer.git",
             str(ASTROMER_DIR)],
            check=True,
        )
    if str(ASTROMER_DIR) not in sys.path:
        sys.path.insert(0, str(ASTROMER_DIR))


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: download pre-trained weights
# ─────────────────────────────────────────────────────────────────────────────
def download_weights() -> Path:
    if (PT_DIR / "config.toml").exists():
        print(f"Pre-trained weights found at {PT_DIR}")
        return PT_DIR

    zip_path = REPO_ROOT / "external" / "pt_macho_v2_2025.zip"
    REPO_ROOT.joinpath("external").mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        print("Downloading ASTROMER 2 pre-trained weights (~20 MB) …")
        try:
            import requests
            resp = requests.get(ZENODO_V2_URL, stream=True, verify=False, timeout=120)
            resp.raise_for_status()
            with open(zip_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        except Exception:
            # Fallback: urllib with SSL verification disabled
            import ssl
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(ZENODO_V2_URL, context=ctx) as r, \
                 open(zip_path, "wb") as fh:
                fh.write(r.read())
        print("  Download complete.")

    extract_dir = REPO_ROOT / "external" / "_pt_extract_tmp"
    extract_dir.mkdir(parents=True, exist_ok=True)
    print("Extracting weights …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    # Locate config.toml anywhere in the extracted tree
    found = list(extract_dir.rglob("config.toml"))
    if not found:
        raise RuntimeError(
            f"config.toml not found after extraction of {zip_path}.\n"
            f"Please manually extract and pass --pt-dir <path_to_weights_dir>."
        )
    # The weights dir is wherever config.toml lives
    weights_dir = found[0].parent
    PT_DIR.mkdir(parents=True, exist_ok=True)
    for f in weights_dir.iterdir():
        f.rename(PT_DIR / f.name)
    # Cleanup temp dir
    import shutil
    shutil.rmtree(extract_dir, ignore_errors=True)

    print(f"Weights ready at {PT_DIR}")
    return PT_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: OGLE-III catalog + data loading
# ─────────────────────────────────────────────────────────────────────────────
def _infer_ogle_class(path: str) -> str | None:
    p = path.lower().replace("\\", "/")
    for cls in ("rrlyr", "cep", "lpv", "ecl"):
        if f"/{cls}/" in p:
            return cls
    return None


def load_ogle_catalog() -> pd.DataFrame:
    """Return the subsampled, shuffled catalog (one row per object)."""
    df = pd.read_csv(OGLE_CATALOG)
    df["label"] = df["Path"].apply(_infer_ogle_class)
    df = df[df["label"].isin(CLASSES)].copy()
    df["label_int"] = df["label"].map(CLASS_TO_INT).astype(int)

    rng = np.random.default_rng(SEED)
    parts = []
    for cls in CLASSES:
        sub = df[df["label"] == cls].copy()
        if len(sub) > MAX_PER_CLASS:
            chosen = rng.choice(len(sub), MAX_PER_CLASS, replace=False)
            sub = sub.iloc[chosen]
        parts.append(sub)

    df_out = (pd.concat(parts)
                .sample(frac=1.0, random_state=SEED)
                .reset_index(drop=True))
    print(f"Catalog: {len(df_out)} objects — {df_out['label'].value_counts().to_dict()}")
    return df_out


def _uniform_indices(n: int, k: int) -> np.ndarray:
    """Deterministic uniform subsampling (mirrors the paper's pipeline)."""
    if n <= k:
        return np.arange(n, dtype=int)
    idx = np.unique(np.round(np.linspace(0, n - 1, k)).astype(int))
    if len(idx) < k:
        missing = np.setdiff1d(np.arange(n, dtype=int), idx)
        add = missing[np.round(np.linspace(0, len(missing) - 1, k - len(idx))).astype(int)]
        idx = np.sort(np.concatenate([idx, add]).astype(int))
    return idx[:k]


def _load_raw_curve(lc_id: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (time, mag) for one OGLE light curve, or None on failure."""
    path = OGLE_LCS_DIR / f"{lc_id}.dat"
    if not path.exists():
        return None
    try:
        arr = np.loadtxt(path, comments="#")
        if arr.ndim != 2 or arr.shape[0] < 10:
            return None
        t, m, e = arr[:, 0], arr[:, 1], arr[:, 2]
        good = (e > 0) & (e < 1.0) & np.isfinite(t) & np.isfinite(m)
        t, m = t[good], m[good]
        order = np.argsort(t)
        return t[order], m[order]
    except Exception:
        return None


def _normalise(t_sel: np.ndarray, m_sel: np.ndarray,
               seq_len: int = SEQ_LEN) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Pad to seq_len, normalise, return:
        t_norm (seq_len,), m_norm (seq_len,), mask (seq_len,),
        m_mean, m_std
    Times → min-max to [0,1].  Magnitudes → zero-mean, unit-std.
    """
    n   = len(t_sel)
    t_p = np.zeros(seq_len, np.float32)
    m_p = np.zeros(seq_len, np.float32)
    msk = np.zeros(seq_len, np.float32)
    t_p[:n] = t_sel.astype(np.float32)
    m_p[:n] = m_sel.astype(np.float32)
    msk[:n] = 1.0

    t_rng = t_p[:n].max() - t_p[:n].min()
    if t_rng > 0:
        t_p[:n] = (t_p[:n] - t_p[:n].min()) / t_rng

    m_mean = float(m_p[:n].mean())
    m_std  = float(m_p[:n].std()) + 1e-8
    m_p[:n] = (m_p[:n] - m_mean) / m_std

    return t_p, m_p, msk, m_mean, m_std


def build_arrays(df: pd.DataFrame) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, list[tuple[np.ndarray, np.ndarray, float, float]],
]:
    """
    Load all light curves and return ASTROMER-format arrays.

    Returns
    -------
    times   : (N, SEQ_LEN, 1) float32  — normalised times
    mags    : (N, SEQ_LEN, 1) float32  — normalised mags
    masks   : (N, SEQ_LEN, 1) float32  — 1=valid, 0=pad
    labels  : (N,) int
    cat_pos : (N,) int  — original catalog row index (for alignment with adv npz)
    stats   : list of (t_sel, m_sel, m_mean, m_std) per curve, for adv reconstruction
    """
    N = len(df)
    times_buf  = np.zeros((N, SEQ_LEN, 1), np.float32)
    mags_buf   = np.zeros((N, SEQ_LEN, 1), np.float32)
    masks_buf  = np.zeros((N, SEQ_LEN, 1), np.float32)
    labels_buf = np.zeros(N, dtype=np.int64)
    stats_buf  = [None] * N
    kept       = []

    for i, (_, row) in enumerate(df.iterrows()):
        if i % 3000 == 0:
            print(f"  loading {i}/{N} …", flush=True)
        res = _load_raw_curve(str(row["ID"]))
        if res is None:
            continue
        t_raw, m_raw = res
        idx   = _uniform_indices(len(t_raw), SEQ_LEN)
        t_sel = t_raw[idx]
        m_sel = m_raw[idx]
        t_norm, m_norm, mask, m_mean, m_std = _normalise(t_sel, m_sel)
        times_buf[i, :, 0]  = t_norm
        mags_buf[i, :, 0]   = m_norm
        masks_buf[i, :, 0]  = mask
        labels_buf[i]        = int(row["label_int"])
        stats_buf[i]         = (t_sel, m_sel, m_mean, m_std)
        kept.append(i)

    kept = np.array(kept, dtype=np.int64)
    return (times_buf[kept],
            mags_buf[kept],
            masks_buf[kept],
            labels_buf[kept],
            kept,                         # catalog positions
            [stats_buf[k] for k in kept]) # per-curve stats


def canonical_split(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """70/10/20 stratified split — mirrors splits.py."""
    idx = np.arange(len(y))
    idx_tv, idx_test = train_test_split(
        idx, test_size=TEST_FRAC, random_state=SEED, stratify=y)
    rel_val = VAL_FRAC / (1.0 - TEST_FRAC)
    idx_train, idx_val = train_test_split(
        idx_tv, test_size=rel_val, random_state=SEED, stratify=y[idx_tv])
    return idx_train, idx_val, idx_test


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: tf.data loaders
# ─────────────────────────────────────────────────────────────────────────────
def make_dataset(times: np.ndarray, mags: np.ndarray, masks: np.ndarray,
                 labels: np.ndarray, shuffle: bool = False) -> "tf.data.Dataset":
    import tensorflow as tf
    n_cls    = len(CLASSES)
    y_onehot = tf.keras.utils.to_categorical(labels, n_cls).astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((
        {"input": mags, "times": times, "mask_in": masks},
        y_onehot,
    ))
    if shuffle:
        ds = ds.shuffle(len(labels), seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: ASTROMER 2 classifier
# ─────────────────────────────────────────────────────────────────────────────
def build_classifier(pt_dir: Path):
    """
    Pre-trained ASTROMER 2 encoder (frozen at build time) + masked-average-pool
    MLP head. Returns (clf, encoder, astromer) so the caller can unfreeze the
    encoder for phase-2 full fine-tuning.
    Input keys match ASTROMER's `build_input` convention:
        "input"   — normalised magnitudes
        "times"   — normalised times
        "mask_in" — 1=valid observation, 0=padding
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    from presentation.pipelines.steps.model_design import load_pt_model
    from src.models.astromer_1 import build_input as astromer_build_input

    astromer, pt_config = load_pt_model(str(pt_dir), optimizer=None)
    window_size = int(pt_config.get("window_size", SEQ_LEN))
    if window_size != SEQ_LEN:
        raise RuntimeError(
            f"ASTROMER pre-trained with window_size={window_size}, "
            f"but this script uses SEQ_LEN={SEQ_LEN}. They must match."
        )

    # Freeze the entire pre-trained encoder
    astromer.trainable = False

    # Build new inputs + classification head
    inp     = astromer_build_input(window_size)   # {input, times, mask_in}
    encoder = astromer.get_layer("encoder")
    z       = encoder(inp)                        # (B, T, D)

    # Masked mean pooling: mask_in=1 → valid
    float_mask = tf.cast(inp["mask_in"], tf.float32)   # (B, T, 1)
    pooled = tf.math.divide_no_nan(
        tf.reduce_sum(z * float_mask, axis=1),
        tf.reduce_sum(float_mask, axis=1),
    )   # (B, D)

    x = layers.Dense(512, activation="relu")(pooled)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LayerNormalization()(x)
    out = layers.Dense(len(CLASSES), activation="softmax", name="clf_out")(x)

    clf = Model(inputs=inp, outputs=out, name="astromer2_ogle_clf")
    clf.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    n_trainable = sum(np.prod(v.shape) for v in clf.trainable_variables)
    n_frozen    = sum(np.prod(v.shape) for v in clf.non_trainable_variables)
    print(f"  Trainable params: {n_trainable:,} | Frozen (encoder): {n_frozen:,}")
    return clf, encoder, astromer


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: adversarial test inputs
# ─────────────────────────────────────────────────────────────────────────────
def build_adv_inputs(
    adv_npz: Path,
    cat_pos: np.ndarray,
    curve_stats: list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct ASTROMER-format inputs for the CNN-targeted adversarial test set.

    The adversarial magnitudes (m_adv) replace the clean magnitudes, but
    normalisation is performed with the *same mean/std as the clean curve*
    so that the perturbation amplitude is preserved in the normalised space.

    Returns
    -------
    times_adv, mags_adv, masks_adv : (N, SEQ_LEN, 1)
    y_adv                          : (N,) int labels
    adv_cat_positions              : (N,) int — positions in our loaded arrays
    """
    adv        = np.load(adv_npz, allow_pickle=True)
    idx_test   = adv["idx_test"].astype(np.int64)  # global catalog positions
    m_adv_mat  = adv["m_adv"].astype(np.float32)   # (N, SEQ_LEN) raw adv mags
    y_adv      = adv["y"].astype(np.int64)

    # Build reverse map: catalog_position → index in our loaded arrays
    cat_to_arr = {int(cp): arr_i for arr_i, cp in enumerate(cat_pos)}

    n_adv      = len(idx_test)
    times_adv  = np.zeros((n_adv, SEQ_LEN, 1), np.float32)
    mags_adv   = np.zeros((n_adv, SEQ_LEN, 1), np.float32)
    masks_adv  = np.zeros((n_adv, SEQ_LEN, 1), np.float32)
    adv_arr_idx = np.full(n_adv, -1, dtype=np.int64)  # index into loaded arrays

    n_found = 0
    for j, cat_i in enumerate(idx_test):
        arr_i = cat_to_arr.get(int(cat_i))
        if arr_i is None:
            print(f"  WARNING: catalog pos {cat_i} not found in loaded arrays — skipping.")
            continue

        t_sel, _, m_mean, m_std = curve_stats[arr_i]
        n = len(t_sel)

        # Times: same normalisation as clean
        t_p   = np.zeros(SEQ_LEN, np.float32)
        t_p[:n] = t_sel.astype(np.float32)
        t_rng = t_p[:n].max() - t_p[:n].min()
        if t_rng > 0:
            t_p[:n] = (t_p[:n] - t_p[:n].min()) / t_rng

        # Adversarial magnitudes: apply the SAME clean-curve mean/std
        m_p   = np.zeros(SEQ_LEN, np.float32)
        count = min(n, SEQ_LEN)
        m_p[:count] = (m_adv_mat[j, :count] - m_mean) / m_std

        msk   = np.zeros(SEQ_LEN, np.float32)
        msk[:n] = 1.0

        times_adv[j, :, 0] = t_p
        mags_adv[j, :, 0]  = m_p
        masks_adv[j, :, 0] = msk
        adv_arr_idx[j]      = arr_i
        n_found += 1

    print(f"  Adversarial inputs built: {n_found}/{n_adv} curves matched.")
    return times_adv, mags_adv, masks_adv, y_adv, adv_arr_idx


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: bootstrap CI for accuracy drop
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap_drop_ci(
    y_true: np.ndarray,
    pred_clean: np.ndarray,
    pred_adv: np.ndarray,
    n_boot: int = 4000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    rng  = np.random.default_rng(0)
    n    = len(y_true)
    drops = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx        = rng.integers(0, n, size=n)
        acc_c      = (pred_clean[idx] == y_true[idx]).mean()
        acc_a      = (pred_adv[idx]   == y_true[idx]).mean()
        drops[b]   = 100.0 * (acc_c - acc_a)
    lo = float(np.percentile(drops, 100 * alpha / 2))
    hi = float(np.percentile(drops, 100 * (1 - alpha / 2)))
    return lo, hi


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="ASTROMER 2 robustness eval")
    parser.add_argument("--pt-dir",  default=None,
                        help="Path to extracted ASTROMER 2 weights (skip download)")
    parser.add_argument("--epochs",  type=int, default=EPOCHS)
    parser.add_argument("--no-gpu",  action="store_true", help="Force CPU-only")
    parser.add_argument("--reuse-head", action="store_true",
                        help="Skip phase 1 and reuse outputs/astromer_clf_ogle/best")
    parser.add_argument("--ft-epochs", type=int, default=FT_EPOCHS,
                        help="Phase-2 (full fine-tune) max epochs")
    parser.add_argument("--ft-lr", type=float, default=FT_LR,
                        help="Phase-2 learning rate (encoder unfrozen)")
    parser.add_argument("--skip-ft", action="store_true",
                        help="Stop after phase 1 (frozen-encoder probe, legacy behavior)")
    parser.add_argument("--reuse-ft", action="store_true",
                        help="Skip all training and reuse outputs/astromer_clf_ogle/best_ft")
    parser.add_argument("--adv-npz", default=None,
                        help=f"Adversarial npz to evaluate (default: {ADV_NPZ})")
    opt = parser.parse_args()

    if opt.no_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # ── 0. Setup ──────────────────────────────────────────────────────────────
    setup_astromer()

    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    devices = [d.name for d in tf.config.list_logical_devices()]
    print(f"TensorFlow {tf.__version__}  |  devices: {devices}")

    pt_dir = Path(opt.pt_dir) if opt.pt_dir else download_weights()

    # ── 1. Load OGLE-III data ─────────────────────────────────────────────────
    print("\n── Loading OGLE-III catalog …")
    df = load_ogle_catalog()

    print("── Loading light curves (may take several minutes) …")
    times, mags, masks, labels, cat_pos, curve_stats = build_arrays(df)
    print(f"   Loaded {len(labels)} / {len(df)} curves successfully.")

    idx_train, idx_val, idx_test = canonical_split(labels)
    print(f"   Split → train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")

    ds_train = make_dataset(times[idx_train], mags[idx_train], masks[idx_train],
                            labels[idx_train], shuffle=True)
    ds_val   = make_dataset(times[idx_val],   mags[idx_val],   masks[idx_val],
                            labels[idx_val])
    ds_test  = make_dataset(times[idx_test],  mags[idx_test],  masks[idx_test],
                            labels[idx_test])

    # ── 2. Build & train classifier (two-phase fine-tuning) ──────────────────
    print("\n── Building ASTROMER 2 classifier …")
    clf, encoder, astromer = build_classifier(pt_dir)

    ckpt_dir = OUTPUT_DIR / "astromer_clf_ogle"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_head = ckpt_dir / "best"      # phase 1: head only (legacy name)
    ckpt_ft   = ckpt_dir / "best_ft"   # phase 2: full fine-tune

    def _fit(epochs: int, ckpt: Path, patience: int) -> None:
        clf.fit(
            ds_train, validation_data=ds_val,
            epochs=epochs,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    str(ckpt), save_best_only=True,
                    monitor="val_loss", save_weights_only=True, verbose=0),
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=patience,
                    restore_best_weights=True, verbose=1),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=3, verbose=1),
            ],
        )

    def _unfreeze_encoder() -> None:
        # Keras propagates `trainable` recursively on assignment; set it on
        # both handles to be robust across Keras 2 minor versions, then
        # recompile so the change takes effect in the train function.
        astromer.trainable = True
        encoder.trainable  = True
        clf.compile(
            optimizer=tf.keras.optimizers.Adam(opt.ft_lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        n_tr = sum(np.prod(v.shape) for v in clf.trainable_variables)
        print(f"   Encoder unfrozen — trainable params now: {n_tr:,} (LR={opt.ft_lr:g})")

    mode = "linear-probe" if opt.skip_ft else "full-finetune"

    if opt.reuse_ft and (ckpt_dir / "best_ft.index").exists():
        print(f"   Reusing fine-tuned weights from {ckpt_ft} — skipping training.")
        _unfreeze_encoder()
        clf.load_weights(str(ckpt_ft)).expect_partial()
        mode = "full-finetune"
    else:
        # Phase 1 — head only, encoder frozen (LR 1e-3)
        if opt.reuse_head and (ckpt_dir / "best.index").exists():
            print(f"   Phase 1: reusing saved head from {ckpt_head}.")
            clf.load_weights(str(ckpt_head)).expect_partial()
        else:
            print(f"   Phase 1: training head (frozen encoder, ≤{opt.epochs} epochs) …")
            _fit(opt.epochs, ckpt_head, PATIENCE)

        # Phase 2 — unfreeze encoder, fine-tune end-to-end at low LR
        if not opt.skip_ft:
            print(f"\n   Phase 2: full fine-tuning (≤{opt.ft_epochs} epochs) …")
            _unfreeze_encoder()
            _fit(opt.ft_epochs, ckpt_ft, FT_PATIENCE)

    # ── 3. Clean accuracy ────────────────────────────────────────────────────
    print("\n── Evaluating on clean test set …")
    proba_clean = clf.predict(ds_test, verbose=0)
    pred_clean  = np.argmax(proba_clean, axis=1)
    acc_clean   = float((pred_clean == labels[idx_test]).mean())
    print(f"   acc_clean = {100*acc_clean:.2f} %")

    # ── 4. Adversarial accuracy ───────────────────────────────────────────────
    adv_npz = Path(opt.adv_npz) if opt.adv_npz else ADV_NPZ
    print(f"\n── Building adversarial inputs ({adv_npz.name}) …")
    t_adv, m_adv_in, mk_adv, y_adv, adv_arr_idx = build_adv_inputs(
        adv_npz, cat_pos, curve_stats)

    # Clean predictions on the *same* curves as the adversarial set
    # (for a fair drop estimate; adv_arr_idx maps adv curve j → loaded array pos)
    valid = adv_arr_idx >= 0
    adv_arr_valid = adv_arr_idx[valid]

    # Ground truth MUST use the classifier's own label encoding (labels[...],
    # i.e. CLASS_TO_INT). The .npz stores y under a different class ordering;
    # using y_adv silently permutes the labels, pushing clean accuracy below
    # chance and inverting the sign of the measured robustness drop.
    y_aligned = labels[adv_arr_valid]

    ds_clean_matched = make_dataset(
        times[adv_arr_valid], mags[adv_arr_valid], masks[adv_arr_valid],
        y_aligned)
    ds_adv = make_dataset(
        t_adv[valid], m_adv_in[valid], mk_adv[valid], y_aligned)

    print("── Evaluating on adversarial test set …")
    proba_clean_m = clf.predict(ds_clean_matched, verbose=0)
    pred_clean_m  = np.argmax(proba_clean_m, axis=1)
    proba_adv_m   = clf.predict(ds_adv, verbose=0)
    pred_adv_m    = np.argmax(proba_adv_m, axis=1)

    y_valid        = y_aligned
    acc_clean_m    = float((pred_clean_m == y_valid).mean())
    acc_adv_m      = float((pred_adv_m   == y_valid).mean())
    drop_pp        = 100.0 * (acc_clean_m - acc_adv_m)

    # Guard: the adv subset is (essentially) the test set, so its clean accuracy
    # must track the full-test clean accuracy. A large gap signals a regression
    # in label/curve alignment between the npz and the classifier.
    if abs(acc_clean_m - acc_clean) > 0.10:
        print("  WARNING: clean-matched accuracy differs from full-test clean by "
              f">10pp ({100*acc_clean_m:.1f}% vs {100*acc_clean:.1f}%) — check "
              "npz↔classifier label alignment.")

    # ── 5. Bootstrap CI ───────────────────────────────────────────────────────
    ci_lo, ci_hi = bootstrap_drop_ci(y_valid, pred_clean_m, pred_adv_m)

    # ── 5b. Confusion matrices (clean vs adversarial) ─────────────────────────
    # Reveals whether the attack collapses predictions onto a single class or
    # scatters them uniformly (≈ chance). Rows = true class, cols = predicted.
    n_cls = len(CLASSES)
    cm_clean = np.zeros((n_cls, n_cls), dtype=int)
    cm_adv   = np.zeros((n_cls, n_cls), dtype=int)
    for t, pc, pa in zip(y_valid, pred_clean_m, pred_adv_m):
        cm_clean[t, pc] += 1
        cm_adv[t, pa]   += 1
    adv_pred_dist = np.bincount(pred_adv_m, minlength=n_cls)
    print("\n  Adversarial confusion matrix (rows=true, cols=pred):")
    print("         " + " ".join(f"{c:>6s}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"   {c:>6s} " + " ".join(f"{cm_adv[i, j]:6d}" for j in range(n_cls)))
    print("   Adv predicted-class distribution: "
          + ", ".join(f"{CLASSES[j]}={int(adv_pred_dist[j])}" for j in range(n_cls)))

    # ── 6. Print results ──────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print(f"  ASTROMER 2 ({mode}) — transferred-attack evaluation (OGLE-III)")
    print(f"  CNN-targeted adversarial examples: {adv_npz.name}")
    print("  (the adv_frac_*.npz sets were generated at k = 1.0 — compare")
    print("   against the k=1.0 budget, not the k=0.5 default)")
    print("=" * 62)
    print(f"  Clean accuracy  (full test):  {100*acc_clean:.2f} %")
    print(f"  Clean accuracy  (adv subset): {100*acc_clean_m:.2f} %")
    print(f"  Adv   accuracy  (adv subset): {100*acc_adv_m:.2f} %")
    print(f"  Drop                        : {drop_pp:+.2f} pp")
    print(f"  95 % bootstrap CI for drop  : [{ci_lo:.2f}, {ci_hi:.2f}] pp")
    print("=" * 62)
    print()
    ft_label = "full fine-tune" if mode == "full-finetune" else "linear probe"
    print("LaTeX snippet (k=1.0 npz; use a k=0.5 npz for the default budget):")
    print(f"  ASTROMER 2 (pre-trained MACHO, {ft_label} OGLE-III) & "
          f"{100*acc_clean:.1f} & {100*acc_adv_m:.1f} & {drop_pp:+.1f} \\\\")

    # ── 7. Save results ───────────────────────────────────────────────────────
    results = {
        "model":       "ASTROMER2",
        "mode":        mode,
        "head_epochs": int(opt.epochs),
        "ft_epochs":   None if opt.skip_ft else int(opt.ft_epochs),
        "ft_lr":       None if opt.skip_ft else float(opt.ft_lr),
        "survey":      "ogle",
        "attack":      f"CNN-targeted transferred ({adv_npz.name}; frac sets generated at k=1.0)",
        "acc_clean":   round(acc_clean, 4),
        "acc_clean_matched": round(acc_clean_m, 4),
        "acc_adv":     round(acc_adv_m, 4),
        "drop_pp":     round(drop_pp, 2),
        "ci_lo_95":    round(ci_lo, 2),
        "ci_hi_95":    round(ci_hi, 2),
        "n_test":      int(len(idx_test)),
        "n_adv_valid": int(valid.sum()),
        "classes":         CLASSES,
        "confusion_clean": cm_clean.tolist(),
        "confusion_adv":   cm_adv.tolist(),
        "adv_pred_dist":   adv_pred_dist.tolist(),
    }
    out_path = OUTPUT_DIR / f"astromer_ogle_results_{mode}.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
