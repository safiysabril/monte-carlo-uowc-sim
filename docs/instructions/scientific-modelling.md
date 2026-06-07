# Scientific Modeling

## Purpose

This framework studies the impact of medium representation on UOWC channel behavior.

The primary research question is:

> How much error is introduced when a depth-dependent underwater environment is approximated as a homogeneous medium?

The framework must support scientifically fair comparisons where only the medium representation changes.

---

## Scientific Hierarchy

Environmental Parameters
↓
Optical Property Model
↓
Optical Coefficients
↓
Medium Representation
↓
Photon Transport
↓
Metrics
↓
Research Conclusions

---

## Optical Property Models

Optical-property models convert environmental parameters into optical coefficients.

Typical outputs include:

* Absorption coefficient (a)
* Scattering coefficient (b)
* Attenuation coefficient (c)
* Phase-function parameters

Examples:

* Kameda
* Haltrin
* Future ocean-optics models

Models must not depend on:

* Monte Carlo transport
* Metrics
* Plotting
* Storage

---

## Scientific Validity

All scientific assumptions should be:

* Explicit
* Traceable
* Documented near implementation

Avoid hidden assumptions.

Document references for equations whenever possible.

---

## Units

Use SI units unless otherwise specified.

Examples:

* Distance: m
* Time: s
* Absorption: m⁻¹
* Scattering: m⁻¹

Never mix unit systems.

---

## Reproducibility

Scientific results must be reproducible.

Simulation metadata should record:

* Model used
* Environmental parameters
* Random seed
* Scenario
* Simulation settings
