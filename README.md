# Adversarial robustness of variable-star classifiers within the photometric error budget

Code, derived results and manuscript source for

> G. Cáceres, K. Pichara, M. Catelan, F. Pérez-Galarce,
> *Adversarial robustness of variable-star classifiers within the photometric error budget*,
> submitted to Astronomy & Astrophysics (2026).

The question the code answers is simple to state: if every magnitude of a
light curve is allowed to move only inside its reported photometric error
bar, can the classification change? For standard classifiers it can, and
by a lot. The repository contains the attack that finds such perturbations,
the defended classifier (an adversarially trained encoder on raw differential
sequences, called RRE in the paper), and the evaluation protocol used to
audit it on OGLE-III, *Gaia* DR2 and WISE.

## Threat model

For each observation `(t_i, m_i, e_i)` the attack may change only the
magnitude, and only within a fixed multiple `k` of its reported error:

```
|m_adv_i - m_i| <= k * e_i        (k = 0.5 by default, k = 1 at most)
```

Observation times and reported errors are never modified. The PGD attack
optimizes per-observation magnitude displacements and recomputes the model
input (`Delta t`, `Delta m`, `sigma_Delta_m`) at every iteration, so the
tensor a model sees is always the tensor of a light curve that satisfies the
bound. `src/attacks/photometric_pgd.py` implements this; the tests in
`tests/test_attack_constraints.py` check the geometry without any trained model.

## Layout

```
src/attacks/       photometrically bounded PGD and perturbation validation
src/data/          survey loaders, splits, raw differential sequences
src/models/        CNN, GRU and random-forest baselines; robust raw encoder
src/metrics/       classification metrics, light-curve features for the RF
src/experiments/   training, attack, ablation, bootstrap and plotting scripts
src/plotting/      matplotlib style and shared figure helpers
configs/           per-survey data configuration (OGLE-III, Gaia DR2, WISE)
scripts/           standalone ASTROMER 2 probe
tests/             unit tests (no data or checkpoints required)
outputs/           metric CSVs, generated tables and figures used in the paper
paper/             A&A manuscript source, bibliography and figures
docs/              data layout and reproduction notes
```

### Names used in the code versus the paper

The code predates the final naming of the paper. The mapping is:

| code            | paper |
|-----------------|-------|
| `L1` … `L5`     | V1 … V5 (ablation levels) |
| `L5-raw`, `l5_raw` | RRE, the proposed classifier |
| `AT-CNN`        | adversarially trained CNN baseline |
| `RF-FATS16`     | 16-feature random forest |

## Environment

The experiments were run with Python 3.12 and TensorFlow 2.16 on CPU.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD"
python -m pytest tests -q
```

## Data

Raw light curves are not distributed here. Place them under `Data/` following
`docs/DATA.md`; the paths are read from `src/data/survey_config.py` and
`configs/*.yaml`. Generated adversarial sets, trained checkpoints and cached
periods live under `runs/` and are not tracked either. The derived data
products (adversarial light curves, periods, checkpoints) are archived on
Zenodo; see the Data availability section of the paper.

## Reproducing the main results

The full pipeline trains the surrogate CNN, builds the bounded adversarial
test sets, trains the baselines and the robust encoder, and runs every
evaluation. It is sequential by design and takes several hours on CPU:

```bash
python -m src.experiments.rebuild_publication_pipeline --force
```

The individual stages, in order:

```bash
# 1. Transfer CNN and bounded adversarial test sets
python -m src.experiments.build_adversarials_ogle
python -m src.experiments.build_adversarials_survey --survey gaia
python -m src.experiments.build_adversarials_survey --survey wise

# 2. Baselines (RF, CNN, GRU) and adversarially trained CNN
python -m src.experiments.eval_baselines_ogle
python -m src.experiments.train_eval_baselines_survey --survey gaia
python -m src.experiments.train_eval_baselines_survey --survey wise
python -m src.experiments.train_at_cnn --survey ogle

# 3. Robust raw encoder: ablation ladder on OGLE, full model on the other surveys
python -m src.experiments.train_robust_raw_ablation --levels 1,2,3,4,5 --force
python -m src.experiments.train_robust_raw_survey --survey gaia
python -m src.experiments.train_robust_raw_survey --survey wise
python -m src.experiments.train_robust_raw_survey_ablation --survey gaia --levels 1,2
python -m src.experiments.train_robust_raw_survey_ablation --survey wise --levels 1,2
python -m src.experiments.train_vib_only_sweep --force
python -m src.experiments.train_noise_baseline --dist uniform

# 4. Evaluation: bounds, adaptive white-box, gradient-free and margin-based attacks
python -m src.experiments.validate_perturbation_bounds
python -m src.experiments.attack_l5_raw_adaptive_multi
python -m src.experiments.attack_raw_ablation_adaptive
python -m src.experiments.attack_raw_ablation_adaptive_survey --survey gaia --levels 1,2,5
python -m src.experiments.attack_raw_ablation_adaptive_survey --survey wise --levels 1,2,5
python -m src.experiments.attack_l5_raw_square --survey ogle
python -m src.experiments.attack_l5_raw_adaptive_dlr --surveys ogle
python -m src.experiments.attack_l5_raw_adaptive_free_delta1
python -m src.experiments.attack_rf_fats_square --build-cache 800
python -m src.experiments.attack_rf_fats_square --attack --budget-s 3000
python -m src.experiments.attack_rf_fats_square --report
python -m src.experiments.sweep_whitebox_canonical --model cnn --k 0.5
python -m src.experiments.sweep_whitebox_canonical --model rnn --k 0.5

# 5. Statistics, per-source flag, tables and figures
python -m src.experiments.bootstrap_metrics
python -m src.experiments.flip_rate_audit
python -m src.experiments.flag_quality_crossmatch
python -m src.experiments.dataset_counts
python -m src.experiments.export_paper_tables_multisurvey
python -m src.experiments.emit_latex_tables > outputs/tables/publication_tables.tex
python -m src.experiments.plot_fig1_adversarial_example
python -m src.experiments.plot_baseline_vulnerability
python -m src.experiments.plot_latent_comparison_l1_vs_l5_raw
python -m src.experiments.plot_robustness_vs_k
python -m src.experiments.plot_flag_quality
```

The ASTROMER 2 probe reported in the paper is a standalone script with its
own dependencies (TensorFlow 2.14, Keras 2); see the header of
`scripts/eval_astromer_ogle.py`.

## Adversarial light-curve dataset

The adversarial test sets themselves (clean and adversarial magnitude for
every epoch of every test star, for the three surveys) are released as a
separate Zenodo deposit; the DOI will be added here once it is minted. They
are built from the attack outputs with
`scripts/package_adversarial_dataset.py`, which also re-checks the
photometric bound on every observation before writing the files.

## Results shipped with the repository

`outputs/metrics/` holds the CSV files every number in the paper is read
from, per survey (`ogle/`, `gaia/`, `wise/`) plus the cross-survey bootstrap
and flip-rate summaries. `outputs/tables/` contains the LaTeX tables generated
from them, and `outputs/figures/` the figures. The manuscript in `paper/` can
be compiled with `tectonic aa_submission.tex` or `pdflatex` + `bibtex`.

## License

MIT, see `LICENSE`.
