# CLAUDE.md

## Purpose

This repository develops a research-grade Monte Carlo Underwater Optical Wireless Communication (UOWC) simulation framework.

The primary research objective is to evaluate the impact of channel representation by comparing:

* Homogeneous underwater environments
* Depth-dependent underwater environments
* Depth-dependent environments with additional environmental effects

The framework must support scientifically rigorous, reproducible, and extensible research.

---

## Core Principles

### Separation of Concerns

Keep the following concerns independent:

* Optical-property models
* Medium representations
* Environmental effects
* Photon transport
* Metrics and analysis
* Data persistence
* Visualization
* Experiment orchestration

Avoid coupling physics, transport, analysis, storage, and plotting.

---

### Composition Over Inheritance

Prefer composition and interfaces.

Environmental effects modify behavior through composition rather than inheritance.

Avoid deep inheritance hierarchies.

---

### Physics Independence

Optical-property models compute physical coefficients.

Mediums define spatial distribution.

Transport propagates photons.

Metrics analyze outputs.

Each component should remain independently extensible.

---

## Scientific Architecture

### Optical-Property Models

Optical-property models compute optical coefficients from environmental parameters.

Examples include:

* Kameda
* Haltrin
* Future ocean-optics models

Models should not depend on transport, storage, plotting, or metrics.

---

### Medium Representations

Supported representations include:

* Homogeneous
* Inhomogeneous (depth-dependent)

Mediums consume optical-property models and expose optical properties through stable interfaces.

---

### Environmental Effects

Environmental effects are optional modifiers.

Examples:

* Turbulence
* Bubble layers
* Sediment concentration
* Chlorophyll variation
* Thermoclines
* Salinity gradients

Effects should remain modular and composable.

---

### Photon Transport

Woodcock Delta Tracking is the primary transport algorithm.

All scientific scenarios use the same transport engine.

Scenario differences should originate from medium representation and environmental effects rather than transport implementation.

---

## Data Outputs

Simulation outputs are research data.

Persist outputs in Parquet format.

Analysis and visualization should operate as downstream consumers rather than being embedded inside simulation components.

---

## Working Guidelines

When modifying the codebase:

1. Preserve scientific correctness before optimization.
2. Preserve reproducibility before convenience.
3. Extend through interfaces and composition.
4. Prefer small, verifiable changes.
5. Document scientific assumptions near the implementation that uses them.

---

## Documentation Map

Consult the documentation relevant to the task being performed.

| Topic                  | Document                                  |
| ---------------------- | ----------------------------------------- |
| Research methodology   | docs/instructions/research-methodology.md |
| Scientific modeling    | docs/instructions/scientific-modeling.md  |
| Photon transport       | docs/instructions/transport.md            |
| Medium representations | docs/instructions/mediums.md              |
| Metrics and analysis   | docs/instructions/metrics.md              |
| Data management        | docs/instructions/data.md                 |
| Visualization          | docs/instructions/visualization.md        |
| Python conventions | docs/instructions/python.md |

Load only the documentation relevant to the current task.

---

## Instruction Priority

When instructions conflict, follow:

1. CLAUDE.md
2. Domain-specific instruction file
3. Existing architecture and code
4. New proposals

Prefer extending existing patterns over introducing new ones.
