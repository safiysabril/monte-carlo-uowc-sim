# Visualization

## Purpose

Visualization communicates simulation results.

Visualization is a downstream consumer of simulation data.

---

## Design Rules

Visualization must not:

* Control transport
* Generate optical coefficients
* Modify simulation outputs

Visualization reads data and presents results.

---

## Figure Standards

Figures should be:

* Reproducible
* Publication quality
* Clearly labeled
* Unit aware

Axes must include units.

Legends should be unambiguous.

---

## Recommended Outputs

Power Analysis

* Power vs depth
* Path loss curves

Temporal Analysis

* CIR plots
* Delay spread comparisons

Frequency Analysis

* Frequency response
* Bandwidth comparisons

Comparative Analysis

* Scenario I vs II
* Scenario II vs III
* Sensitivity studies

---

## Research Comparisons

Visualizations should emphasize:

* Homogeneous assumptions
* Depth-dependent effects
* Environmental-effect contributions

Plots should support scientific interpretation rather than aesthetics alone.

---

## Reproducibility

All figures should be reproducible from stored Parquet outputs without rerunning simulations.
