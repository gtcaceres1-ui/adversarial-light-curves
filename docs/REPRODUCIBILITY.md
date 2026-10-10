# Reproduction notes

## Protocol

* Seed 42 everywhere; one model per survey, no pooling.
* Stratified 70/10/20 object split. The 20% test fold is fixed first and the
  remaining 80% is split again into train and validation, so the test fold
  is identical across every model and experiment.
* Preprocessing statistics (standardization of `Delta t`, `Delta m`,
  `sigma_Delta_m`) are computed on the training fold only.
* The transferred adversarial test set is generated once per survey from the
  surrogate CNN (`build_adversarials_*`) and reused by every evaluation, so
  all models are attacked on exactly the same perturbed curves.
* Budget `k = 0.5` by default; `k = 1.0` and larger values are used only in
  the budget sweeps.

## Checks that need no data

```bash
export PYTHONPATH="$PWD"
python -m pytest tests -q
find src tests scripts -name "*.py" -exec python -m py_compile {} +
```

## Heavy stages

Training the robust encoder takes 30 to 50 minutes per survey on a laptop
CPU; the OGLE ablation ladder (five levels) and the adaptive attacks with
five restarts are the longest stages. Do not run several TensorFlow jobs in
parallel on a CPU-only machine; `rebuild_publication_pipeline.py` runs them
sequentially for that reason.

Two scripts are written to be resumable because they were run under a
wall-clock limit: `attack_rf_fats_square.py` checkpoints its state under
`runs/rf_fats_attack/`, and `bootstrap_canonical_chunked.py` saves prediction
progress under `outputs/metrics/<survey>/`. Both produce the same numbers as
their one-shot counterparts.

## Where each number comes from

| paper | file |
|-------|------|
| baseline clean / adversarial accuracy | `outputs/metrics/<survey>/baselines_clean_vs_adv.csv` |
| RRE and ablation levels | `outputs/metrics/<survey>/robust_raw_ablation.csv` |
| adaptive white-box attack | `outputs/metrics/<survey>/adaptive_pgd_*.csv` |
| adversarially trained CNN | `outputs/metrics/<survey>/at_cnn_eval.csv` |
| bootstrap confidence intervals | `outputs/metrics/<survey>/bootstrap_metrics.csv`, `bootstrap_diffs.csv` |
| Square attack on the RF | `outputs/metrics/ogle/rf_fats_square_attack_k*.csv` |
| Square attack on RRE | `outputs/metrics/ogle/l5_raw_square_attack_k0.50.csv` |
| budget sweep | `outputs/metrics/ogle/whitebox_canonical_sweep.csv`, `adaptive_pgd_l5_raw_all_k*.csv` |
| noise-augmentation control | `outputs/metrics/ogle/noise_baseline.csv` |
| VIB-only sweep | `outputs/metrics/ogle/vib_only_beta_sweep.csv` |
| per-source robustness flag | `outputs/metrics/flip_rate_audit.csv`, `flip_rate_bins.csv` |
| perturbation structure and detectability | `outputs/perturbation_stats/`, `outputs/detectability/` |
| dataset counts | `outputs/tables/dataset_counts.csv`, `obs_count_distribution.csv` |
