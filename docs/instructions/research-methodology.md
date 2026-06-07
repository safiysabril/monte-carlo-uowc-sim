# Research Methodology

## Purpose

This framework investigates the scientific impact of medium representation in UOWC simulations.

The primary research question is:

> What differences arise when a depth-dependent underwater environment is approximated as a homogeneous medium?

The framework must support scientifically fair, reproducible, and statistically defensible comparisons.

---

## Scientific Hierarchy

Environmental Parameters
↓
Optical-Property Model
↓
Optical Coefficients
↓
Medium Representation
↓
Environmental Effects
↓
Photon Transport
↓
Metrics
↓
Scientific Conclusions

---

## Research Scenarios

### Scenario I

Homogeneous representation of an optical-property model.

Characteristics:

* Spatially constant optical properties
* Baseline approximation
* Reference scenario

---

### Scenario II

Depth-dependent representation of the same optical-property model.

Characteristics:

* Properties vary with depth
* Same physical model as Scenario I
* Different spatial representation

Purpose:

Measure the impact of replacing a homogeneous approximation with a depth-dependent medium.

---

### Scenario III

Scenario II with environmental-effect modules enabled.

Examples:

* Turbulence
* Bubble layers
* Sediment concentration
* Chlorophyll variation
* Thermoclines
* Salinity gradients

Purpose:

Measure the impact of environmental processes beyond depth dependence.

---

## Fair Comparison Rules

### Scenario I vs Scenario II

Only medium representation should change.

The following should remain unchanged:

* Optical-property model
* Environmental parameters
* Source configuration
* Receiver configuration
* Transport algorithm
* Simulation settings
* Random-seed strategy

---

### Scenario II vs Scenario III

Only environmental effects should change.

All other parameters should remain identical.

---

### Multi-Scenario Experiments

Scientific comparisons should isolate the variable under investigation.

Avoid changing multiple major factors simultaneously.

---

## Verification

Verification answers:

> Was the model implemented correctly?

Examples:

* Unit testing
* Conservation checks
* Sampling validation
* Boundary-condition validation
* Numerical consistency tests

Verification should occur before scientific interpretation.

---

## Validation

Validation answers:

> Does the model reproduce known physical behavior?

Examples:

* Published literature
* Analytical approximations
* Benchmark simulations
* Experimental measurements

Verification and validation are separate activities.

---

## Convergence Studies

Monte Carlo outputs are statistical estimates.

Demonstrate convergence by evaluating metric stability as photon count increases.

Typical workflow:

1. Run increasing photon counts.
2. Monitor metric convergence.
3. Evaluate confidence intervals.
4. Select justified production settings.

Avoid arbitrary photon counts.

---

## Statistical Reporting

Report uncertainty whenever practical.

Preferred reporting includes:

* Mean
* Standard deviation
* Confidence interval
* Sample count

Avoid reporting only point estimates.

---

## Randomness and Reproducibility

Record:

* Random seed
* RNG implementation
* Scenario definition
* Simulation configuration

Results should be reproducible from stored metadata.

---

## Sensitivity Analysis

Sensitivity studies should identify which parameters most influence channel behavior.

Typical parameters include:

* Absorption coefficient
* Scattering coefficient
* Receiver aperture
* Receiver field of view
* Source divergence
* Water depth
* Environmental-effect strength

Unless studying interactions, vary one parameter at a time.

---

## Uncertainty Quantification

Distinguish between:

* Monte Carlo variance
* Environmental uncertainty
* Model uncertainty
* Measurement uncertainty

Do not treat all uncertainty sources as equivalent.

---

## Publication-Quality Results

Scientific claims should be supported by:

* Reproducible simulations
* Quantitative evidence
* Statistical uncertainty
* Clearly defined experimental conditions

Figures and tables should be reproducible directly from stored simulation outputs.

---

## Reproducibility Checklist

Before accepting a result:

* Scenario is documented.
* Model is documented.
* Parameters are documented.
* Random seeds are recorded.
* Raw outputs are stored.
* Metrics are reproducible.
* Figures are reproducible.

Results that cannot be reproduced should not be considered final.

---

## Research Priorities

Prioritize:

1. Scientific correctness
2. Reproducibility
3. Statistical rigor
4. Interpretability
5. Extensibility
6. Performance

Performance improvements must preserve scientific validity.
