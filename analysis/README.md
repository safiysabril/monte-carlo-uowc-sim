# Analysis

Downstream analysis of stored Parquet outputs
(pipeline: Simulation -> Parquet -> Analysis -> Visualization).

Reusable routines live in `src/uowc/analysis/`; this directory holds scripts and
notebooks that produce publication figures/tables from stored results **without
rerunning simulations**.

- `medium_representation.py` - convergence, scenario comparison, sensitivity.
- `notebooks/` - exploratory notebooks.

Figures are written to `figures/` (git-ignored).
