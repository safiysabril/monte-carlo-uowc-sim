# Medium Representations

## Purpose

Mediums describe how optical properties are distributed spatially.

Mediums do not generate optical coefficients.

Optical-property models generate coefficients.
Mediums organize them in space.

---

## Homogeneous Medium

Properties are spatially constant.

A single set of optical coefficients applies throughout the simulation domain.

Used for:

* Scenario I
* Baseline comparisons

---

## Inhomogeneous Medium

Properties vary spatially.

Initial implementation uses depth-dependent variation.

Future implementations may support:

* 2D variation
* 3D variation
* Measured oceanographic fields

---

## Environmental Effects

Environmental effects modify channel behavior.

Examples:

* Turbulence
* Bubble layers
* Sediment concentration
* Chlorophyll variation
* Thermoclines
* Salinity gradients

---

## Design Rules

Prefer composition.

Example:

InhomogeneousMedium(
profile=...,
effects=[...]
)

Avoid deep inheritance hierarchies.

Environmental effects should be attachable and removable without changing transport code.

---

## Scenario Mapping

Scenario I:
Homogeneous Medium

Scenario II:
Inhomogeneous Medium

Scenario III:
Inhomogeneous Medium + Environmental Effects
