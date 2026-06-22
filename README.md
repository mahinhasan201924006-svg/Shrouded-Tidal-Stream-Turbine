# Power-Loading Trade-Off Optimization of a Shrouded Tidal Stream Turbine

Data and code accompanying the manuscript:

> **Power-Loading Trade-Off Optimization of a Shrouded Tidal Stream Turbine Using Parametric RANS CFD and Physics-Aware Surrogate Modelling.**
> M. H. Moon et al., Department of Naval Architecture and Marine Engineering, Military Institute of Science and Technology, Dhaka, Bangladesh.

This repository contains the Latin hypercube CFD dataset, the surrogate-modelling pipeline,
and the figure-generation scripts needed to reproduce the machine-learning results and the
data-driven figures in the paper. CFD-contour figures (domain, mesh, velocity fields) are
produced in ANSYS Fluent and are not included here.

## Overview

A reduced-order, two-dimensional axisymmetric actuator-disk RANS model is used to screen the
duct geometry of a shrouded horizontal-axis tidal turbine over four design variables
(chord-to-diameter ratio, angle of attack, radial offset, actuator-disk pressure jump).
A 64-point Latin hypercube campaign (61 converged, 3 failed) is used to train a calibrated
surrogate that blends Extra Trees regression, support-vector regression, and ridge-regularized
response surfaces; the rotor-area power coefficient is reconstructed from the actuator-disk
relation. A Pareto search over the surrogate identifies three power-loading optima, each
confirmed by direct CFD.

## Repository structure

```
duct-turbine-surrogate/
├── data/
│   ├── Turbine_LHS_Data_64.csv              # raw 64-point LHS design + CFD outputs
│   ├── Turbine_LHS_Data_64_ML_completed.csv # cleaned dataset used for training (see data dictionary)
│   └── data_dictionary.md                   # column definitions and units
├── src/
│   ├── turbine_surrogate_training_pipeline.py  # reference pipeline (scikit-learn)
│   ├── sklearn_free.py                          # pure-NumPy surrogate reimplementation
│   └── make_figures.py                          # regenerate all data-driven figures
├── figures/                                  # generated figures (PNG 600 dpi + PDF + SVG)
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Reproducing the figures

No scikit-learn required (uses the NumPy reimplementation in `sklearn_free.py`):

```bash
pip install -r requirements.txt
python src/make_figures.py
```

Figures are written to `figures/`. Generated: Fig. 4 (mesh convergence), Fig. 7 (LHS design
space), Fig. 8 (C_P,r distribution), Fig. 10 (physics-consistency check), Fig. 11 (cross-validated
parity), Fig. 12 (cross-validated R^2), Fig. 13 (power-loading trade-off), Fig. 14 (relative benefit).

## Reproducing the exact published metrics

The reported cross-validated metrics (R^2 = 0.752 for |F_d|, 0.657 for u_d, 0.776 for C_P,r)
were produced with scikit-learn. To reproduce them exactly, install scikit-learn and run the
reference pipeline:

```bash
pip install scikit-learn
python src/turbine_surrogate_training_pipeline.py
```

`sklearn_free.py` reproduces the same model specification in pure NumPy for portability; the
Extra Trees + ridge model for duct loading matches closely, while the epsilon-SVR velocity
model is marginally more conservative than the libsvm solver used by scikit-learn.

## Surrogate specification

| Target            | Model                                                              |
|-------------------|--------------------------------------------------------------------|
| Duct load `|F_d|` | 0.60 Extra Trees (500 trees, min_samples_leaf=2) + 0.40 ridge RSM (λ=1.0) |
| Disk velocity `u_d` | 0.70 SVR-RBF (C=10, ε=0.04) + 0.30 ridge RSM (λ=0.8)             |
| Power coeff. `C_P,r` | Physics reconstruction: C_P,r = ΔP·u_d / (0.5·ρ·U∞³)            |

Inputs are standardized; candidates are restricted to the interpolation region by the
nearest-neighbour distance in standardized space (d_NN ≤ 1.75). Evaluation is five-fold
cross-validation (no separate hold-out set).

## License

Code is released under the MIT License (see `LICENSE`). If you reuse the dataset or code,
please cite the paper (see `CITATION.cff`).
