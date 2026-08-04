# Visualization

## Purpose

Visualization communicates simulation results. It is a downstream consumer of stored
simulation data.

---

## Design Rules

Visualization must not:

* Control transport
* Generate optical coefficients
* Compute metrics (it plots metrics; it does not derive them)
* Modify simulation outputs

Visualization reads data and presents results.

---

## Figure Standards

Figures must be reproducible, publication quality, clearly labelled and unit aware.

* **Axes must carry units.** No exceptions.
* **dB axes must state optical or electrical** (metrics.md). An axis labelled only
  "dB" is ambiguous by a factor of two and is not publishable.
* **Bandwidth axes must state optical or electrical 3 dB.**
* Legends must be unambiguous, and must name the homogenization rule wherever a
  Scenario I curve appears.
* Every figure caption must carry the conditions needed to interpret it: wavelength,
  range, phase function, photon count, and detected photon count.

---

## Uncertainty Must Be Visible

**Plot uncertainty, not just point estimates.** A Monte Carlo result plotted as a bare
line asserts a precision the simulation does not have.

* Show confidence intervals as error bars or shaded bands.
* For scenario differences, plot the **paired** interval from the replication
  procedure (research-methodology.md), not the visual gap between two independent
  bands. Two overlapping intervals do not imply a non-significant difference.
* Mark points with zero detections explicitly as upper bounds (a downward arrow at the
  rule-of-three bound), never as a value of zero and never by dropping them — silently
  omitting non-detections biases a curve upward exactly where the channel is worst.
* Where the effective sample size is far below the detected count, say so.

---

## Scale Conventions

* **Path loss and received power span orders of magnitude — use a log scale or dB.** A
  linear power axis compresses the entire operating range into the bottom of the plot.
* **CIRs need a log intensity axis.** The multipath tail that sets the delay spread is
  typically decades below the ballistic peak and is invisible on a linear axis.
* **Plot CIRs against excess delay** `t − t₀`, stating `t₀`. Absolute-time CIRs from
  different geometries cannot be meaningfully overlaid.
* Convergence plots should show the **standard error against photon count on log–log
  axes**, where the expected `N^{-1/2}` behaviour is a straight line of slope `−½`.
  Plotting only the estimate hides whether it converged.

---

## Recommended Outputs

**Power analysis**

* Received power fraction vs range
* Received power vs deployment depth (a family of runs — label it as such)
* Path loss curves

**Temporal analysis**

* CIR, log intensity vs excess delay
* Delay spread comparisons across scenarios

**Frequency analysis**

* `|H(f)|` with the 3 dB crossing marked
* Bandwidth comparisons across scenarios

**Comparative analysis**

* Scenario I vs II, **across all homogenization rules** — the spread across rules is a
  result, not a nuisance, and hiding it behind a single favoured rule overstates the
  finding
* Scenario II vs III
* Sensitivity studies over model inputs (research-methodology.md)

**Diagnostics** (not for publication, but required before it)

* Convergence per metric
* Weight-balance closure
* Majorant-invariance check

---

## Research Comparisons

Visualizations should emphasize the homogeneous approximation, depth-dependent
structure, and environmental-effect contributions.

Plots must support scientific interpretation rather than aesthetics alone. In
particular, a comparison figure should make the *magnitude of the difference* legible
against its *uncertainty* — plotting the difference directly, with its paired interval
and a zero reference line, is usually clearer than overlaying two curves and inviting
the reader to subtract by eye.

---

## Reproducibility

All figures must be reproducible from stored Parquet outputs without re-running
simulations.

A figure script that re-runs a simulation is a simulation script, and its output is not
reproducible from the archive.
