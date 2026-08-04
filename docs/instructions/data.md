# Data Management

## Purpose

Simulation outputs are research data.

Data storage must support:

* Reproducibility
* Large-scale experiments
* Downstream analysis without re-running simulations

---

## Storage Format

Primary format: **Parquet** (via pyarrow).

Avoid embedding analysis logic inside storage layers. The store writes and reads; it
does not compute metrics, and it does not silently transform values.

---

## Numerical Precision

**Store float64 for all physical quantities.** Do not downcast on write.

* **Arrival times** are ~10⁻⁷ s while the delay spread of interest is ~10⁻⁹ s or
  smaller. A float32 column carries only about four to five significant digits on that
  spread, and per-step accumulation over thousands of collisions compounds the
  rounding. float64 leaves twelve or more.
* **Weights** must not be downcast: `Σwᵢ²` and `Σwᵢ` are both needed for the variance
  and the effective sample size, and the ratio is precision-sensitive when weights are
  skewed.
* Scatter counts are integers and must be stored as integers, not as floats.

Independently of dtype, compute weighted variance in the **two-pass form**
`Σw(t − t̄)²/Σw`, never as `E[t²] − E[t]²`. With `t̄ ≈ 2×10⁻⁷ s` and a spread of
`10⁻⁹ s` the second form subtracts two nearly-equal numbers and loses most of its
significant digits — in float64 it is merely degraded, and in float32 it can return a
negative variance.

---

## Metadata

Metadata must be sufficient to **reproduce the run from storage alone**. A run that
cannot be reconstructed from its metadata is not a research record.

Every simulation records:

**Scientific configuration**

* Scenario identifier
* Optical-property model name and its **full coefficient set** (not just the name —
  placeholder coefficients must be recoverable)
* Wavelength (nm)
* Environmental parameters / profile definition
* Medium representation
* **Homogenization rule**, where a homogeneous representation was used
* Environmental effects, their parameters, and their **composition order**
* Phase function(s) and asymmetry parameters
* Boundary configuration (surface model, bottom albedo, edge policy)
* Source configuration: position, direction, divergence, beam model
* Receiver configuration: position, normal, aperture radius, field of view

**Numerical configuration**

* Transport engine and estimator
* Photon count, chunk size
* Majorant (the value actually used)
* Russian-roulette threshold and survival probability, where applicable

**Reproducibility**

* Random seed **and** RNG implementation / bit generator
* Stream or spawn layout
* Code version (git commit, and whether the tree was dirty)
* Library versions (numpy, scipy, pyarrow)
* Timestamp

**Units**

* Units for every stored column, carried in the schema or the column name.

This list and the one in scientific-modelling.md must agree; if they diverge, this file
is authoritative.

---

## Raw Results

Store raw simulation outputs whenever practical. Derived metrics must be reproducible
from stored results without re-running transport.

Distinguish two record types; they have different sizes and different roles:

* **Per-photon detection records** — one row per *detected* photon: arrival time, path
  length, weight, scatter count, incidence angle. This is small even for huge runs,
  because detection is rare, and it is what every temporal and frequency metric is
  computed from. Store it always.
* **Aggregate tallies** — launched, detected, absorbed, escaped, killed weights. Small,
  and required for the conservation check and for every normalization. Store it always.

Storing only aggregates makes the CIR, delay spread and bandwidth unrecoverable.
Storing only detections makes normalization and conservation unrecoverable. Both are
required.

**Killed weight must be stored**, not discarded. It is the diagnostic that says whether
histories were truncated, and a result cannot be accepted without checking it.

---

## Data Pipeline

```
Simulation  →  Parquet storage  →  Analysis  →  Visualization
```

Keep these stages independent. Each stage reads the previous stage's output and does
not reach past it.

---

## Versioning

Changes affecting scientific interpretation must be traceable.

* Prefer explicit schema evolution over silent format changes.
* Carry a schema version in the metadata.
* **A change in physics is not a schema change.** Results produced before and after a
  correction to a model, a sampler or the majorant are not comparable, regardless of
  whether the columns match. Record the code version so such results can be identified
  and separated rather than silently pooled.
* Never overwrite a stored raw result in place. Runs are append-only records.
