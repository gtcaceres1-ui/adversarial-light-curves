# Data layout

The repository ships no light curves, adversarial sets or checkpoints. The
code expects the following tree at the repository root.

```
Data/
  OGLE-III/
    OGLE_dataset.dat          catalog: ID, Class, Path, N
    LCs/<class>/<ID>.dat      whitespace, no header: time, mag, err
  Gaia/
    GAIA_dataset.dat
    LCs/...                   CSV with header
  WISE/
    WISE_dataset.dat
    LCs/...                   CSV with header
runs/                         created by the pipeline
  adversarial_dataset/        OGLE-III splits, surrogate CNN and attack
  gaia_adversarial_dataset/
  wise_adversarial_dataset/
  cnn_ogle/, rnn_ogle/, rf_ogle/      baselines
  robust_raw_l*_*/            ablation levels (OGLE)
  gaia_robust_raw_l*/, wise_robust_raw_l*/
  periods_ogle.npz            Lomb-Scargle periods (quality filter, figures)
```

Catalog columns, file formats and class groupings per survey are defined in
`src/data/survey_config.py`; the OGLE-III loader used by the baselines reads
`configs/ogle.yaml`. The four working classes are RR Lyrae, Cepheids,
long-period variables and delta Scuti (OGLE-III uses eclipsing binaries in
place of delta Scuti, which it lacks).

The light curves come from the public OGLE-III, *Gaia* DR2 and WISE archives
as compiled by Becker et al. (2020). The derived products needed to reproduce
the tables without re-running the attacks (adversarial light curves, cached
periods and trained checkpoints) are archived on Zenodo, see the paper's Data
availability section.
