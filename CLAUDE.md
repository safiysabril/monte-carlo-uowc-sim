# CLAUDE.md

## Purpose

This repository develops a research-grade Monte Carlo Underwater Optical Wireless Communication (UOWC) simulation framework.

The primary research objective is to evaluate the impact of channel representation by comparing:

* Homogeneous underwater environments
* Depth-dependent underwater environments
* Depth-dependent environments with additional environmental effects

The framework must support scientifically rigorous, reproducible, and extensible research.

The homogeneous baseline is derived from the *same* profile as the depth-dependent case
via an explicit **homogenization rule**. That rule is a declared experimental factor: it
changes the answer, so it must be recorded and named in every result. See
docs/instructions/research-methodology.md.

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

* Haltrin
* Future ocean-optics models

Kameda & Matsumura (1998) is the vertical **chlorophyll-profile** model that supplies
`C(z)` to a depth-dependent medium (`uowc.media.profiles.KamedaModel`) — it does not
publish its own absorption/scattering coefficients, so it is not itself an
optical-property model. Do not attribute a fabricated a(λ)/b(λ) formula to "Kameda";
Haltrin (or another cited IOP model) converts whatever profile is used into IOPs.

Models should not depend on transport, storage, plotting, or metrics.

---

### Medium Representations

Supported representations include:

* Homogeneous
* Inhomogeneous (depth-dependent)

Mediums consume optical-property models and expose optical properties through stable interfaces.

---

### Environmental Effects

Environmental effects are optional modifiers, acting at one of three layers. The layer
is a physical statement, not an implementation detail:

* **Parameter effects** modify environmental constituents *before* the optical model,
  so bio-optical coupling stays consistent (sediment, chlorophyll variation).
* **Optical effects** modify local optical state *after* the model, typically by
  adding a scattering population with its own phase function (bubble layers).
* **Refractive effects** contribute to the refractive-index field and its gradient,
  changing optical path and ray geometry but not `a` or `b` (turbulence, thermoclines,
  salinity gradients).

Effects should remain modular and composable. An effect that raises extinction must
also contribute to the medium's majorant bound.

---

### Photon Transport

Woodcock Delta Tracking is the primary transport algorithm.

Its correctness condition is `c_max >= c(x)` everywhere in the tracked region, after
all effects are applied. Violating it biases results with no runtime symptom. Results
must be invariant to the value of `c_max`.

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
| Scientific modeling    | docs/instructions/scientific-modelling.md |
| Photon transport       | docs/instructions/transport.md            |
| Medium representations | docs/instructions/mediums.md              |
| Metrics and analysis   | docs/instructions/metrics.md              |
| Data management        | docs/instructions/data.md                 |
| Visualization          | docs/instructions/visualization.md        |
| Python conventions     | docs/instructions/python.md               |

Load only the documentation relevant to the current task.

---

## Instruction Priority

When instructions conflict, follow:

1. CLAUDE.md
2. Domain-specific instruction file
3. Existing architecture and code
4. New proposals

Prefer extending existing patterns over introducing new ones.
