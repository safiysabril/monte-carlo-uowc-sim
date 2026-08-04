# Medium Representations

## Purpose

A medium describes how optical properties are **distributed in space** and exposes
that distribution to transport.

Optical-property models generate coefficients from environmental parameters. Mediums
organize them spatially. Environmental effects modify either the parameters, the local
optical state, or the refractive index — see *Environmental Effects* below, which is
where the "mediums do not generate coefficients" rule needs qualifying.

---

## Coordinate and Depth Convention

**This convention is normative (see `uowc.core.units`). Every profile, effect,
boundary and scenario in the framework assumes it. There are two related but
distinct scalars — do not conflate them.**

* **Cartesian position** `(x, y, z)`, metres, right-handed. `z` points **up**; the
  air–water surface is at `z = 0`; underwater, `z < 0`. This is the frame transport,
  geometry, sources and receivers operate in.
* **Depth** `d`, metres, **positive downward**, with `d = 0` at the surface:
  `d = -z`. This is the frame chlorophyll profiles and homogenization operate in,
  because "depth of the deep chlorophyll maximum" is naturally a positive number in
  the oceanographic literature these profiles are drawn from.
* The conversion is exactly `d = -z` (equivalently `z = -d`), applied at the one
  seam where a spatial field queries a profile — nowhere else. A field implementation
  that queries a `ChlorophyllProfile` must convert; the profile itself is never handed
  a raw Cartesian `z`.

An unstated or silently-doubled sign flip is how a profile inverts — a deep
chlorophyll maximum becomes a surface bloom, every metric shifts plausibly, and
nothing fails loudly. Any code consuming a depth or a Cartesian `z` must state which
one it takes in its docstring, and any profile must be verifiable by a monotonicity
or known-value test at a named depth.

---

## The Medium Interface

A medium is a composition of three decoupled capabilities, not a monolith:

| Capability | Responsibility |
| ---------- | -------------- |
| **Optical field** | `c(x)` and `n(x)` batched for the transport hot path; full local optical state (including the scattering mixture) at real interaction points |
| **Domain** | Spatial extent; containment test and bounds. Geometry only — no interaction physics |
| **Acceleration** | The Woodcock majorant over a region |

Separating acceleration from the field keeps the majorant out of transport and lets a
non-Woodcock engine omit it entirely.

### The majorant is a medium responsibility

The medium — not transport — must supply `c_max ≥ c(x)` over a region, and the bound
must hold **after every effect has been applied**. This is a correctness obligation,
not an optimization; see transport.md for what a violated majorant does (silent bias,
no runtime symptom).

Two properties to preserve:

* **Regional, not global.** A thin high-extinction layer (a bubble plume, a nepheloid
  layer) must not inflate the bound over the entire domain, or delta tracking degrades
  to near-pure null collisions everywhere.
* **Composable.** An effect that adds extinction supplies its own additive bound over
  a region; the medium sums them. An effect must never be able to raise `c(x)` without
  a corresponding contribution to the bound.

A loose majorant costs only speed. An invalid one costs the result.

---

## Homogeneous Medium

Spatially constant optical properties; a single IOP set applies throughout the domain.

Used for Scenario I and baseline comparisons.

**The IOPs must be derived from the same profile that drives Scenario II, via a
declared homogenization rule.** A homogeneous medium configured from independently
chosen coefficients is not a comparable baseline — it is a different experiment. See
research-methodology.md, *Homogenization*, for the rules and for the Jensen bias that
makes the choice of averaging space consequential.

Homogenization is a property of the **experiment**, not of the medium. A homogeneous
medium consumes a scalar concentration or an IOP set; it does not decide how that
value was obtained.

---

## Inhomogeneous Medium

Properties vary spatially. The initial implementation is depth-dependent: a profile
`C(z)` feeds the optical model at each queried position.

Requirements:

* **A constant profile must reproduce the homogeneous medium exactly** (to Monte Carlo
  error). This is the profile-collapse verification test and the cheapest guard
  against a depth-indexing error.
* Profiles must be defined over the full domain extent, including behavior outside the
  sampled range. Extrapolation policy (clamp, constant, error) must be explicit —
  silent extrapolation of a fitted profile is a common source of unphysical negative
  or runaway concentrations.
* Profiles must be evaluable in batch, since transport queries them per photon per
  step.

Future work: 2D and 3D variation, measured oceanographic fields, time-varying fields.

---

## Environmental Effects

Effects are optional, composable modifiers. They act at **three distinct layers**, and
the layer determines both the physics and where the effect may intervene.

### Parameter effects — before the optical model

Modify the environmental state (chlorophyll, CDOM, NAP, temperature, salinity) at a
position; the optical model then converts the modified state to IOPs as usual.

This is the correct layer for anything that changes the **constituents** of the water:
suspended sediment adding mineral particles, chlorophyll variation, riverine CDOM.
Working at this layer keeps the bio-optical coupling internally consistent — a
sediment load that raises scattering also raises absorption, in the proportion the
model dictates, rather than in a proportion chosen by hand.

Applying such an effect at the IOP layer instead breaks that coupling and produces
water with no consistent composition.

### Optical effects — after the optical model

Modify the local optical state directly, typically by **adding a scattering
population** with its own coefficient and phase function.

The correct layer for constituents the bio-optical model does not represent at all — a
bubble layer being the canonical case, since bubbles are a distinct scatterer with a
distinct phase function rather than a change in water composition.

An optical effect must supply:

* a batched **extinction contribution** for the transport hot path, and
* an additive, **regional extinction bound** so the medium's majorant stays valid and
  tight.

An effect that adds a scattering population must add it **as a population**, with its
own phase function. Folding it into an existing population's coefficient discards its
directionality, which is usually the physically interesting part.

### Refractive effects — the index field

Contribute an additive perturbation to the refractive index, plus that perturbation's
**gradient**.

The correct layer for turbulence, thermoclines and haloclines. These effects change
optical path length and bend rays; they do **not** change `a` or `b`, and must not be
modelled as if they did.

Requirements:

* The gradient must be the analytic derivative of the same expression that produces
  the index — not an independently specified field.
* The field must be **realized and spatially correlated**: fixed by a seed, so
  querying the same position twice returns the same value. Independent per-point noise
  has no correlation length and no meaningful gradient, and reproduces none of the
  physics of turbulence.
* The medium composes `n(x) = n₀ + Σᵢ n'ᵢ(x)`; effects return their contribution, not
  the absolute index.

### Effects and the fair-comparison rule

Enabling any effect that raises extinction **requires the majorant to be recomputed**.
This is a correctness consequence of the effect, not an additional experimental
variable, and does not violate the Scenario II vs III comparison rule.

---

## Boundaries

Boundary physics belongs to a boundary, not to the domain. The domain answers "is this
position inside"; the boundary answers "what happens on contact".

A domain that spans `z = 0` **has an air–water interface**, and ignoring it is a
physical error, not a simplification — it silently converts a reflecting surface into a
perfect absorber and biases both received power and delay spread.

### Air–water surface

* **Fresnel reflection and refraction** at the interface, using the polarization-
  averaged coefficients (this framework does not track polarization).
* At normal incidence, reflectance `R = ((n_w − n_a)/(n_w + n_a))² ≈ 2 %` for
  `n_w ≈ 1.34`.
* **Total internal reflection** for upward rays beyond the critical angle
  `θ_c = arcsin(n_a/n_w) ≈ 48–49°`. Beyond `θ_c` the surface is a perfect mirror; this
  is what confines the upwelling field and it materially affects long-path photons.
* A flat surface is the default. A wind-roughened surface (a slope distribution) is a
  distinct model and must be declared when used.

### Bottom

Reflectance and angular response (Lambertian albedo as the default, BRDF where
justified). Bottom type and depth must be recorded — a bright sand bottom and a dark
mud bottom are different channels.

### Domain edges

Lateral and lower exits must have a stated policy — escape (absorb and tally) is the
default. Whatever the policy, escaped weight must be **tallied**, because the
conservation check `detected + absorbed + escaped + killed = launched` is the primary
guard against a leaking transport loop.

Periodic boundaries are not appropriate here and must not be introduced silently: they
would recirculate photons and inflate received power.

---

## Design Rules

Prefer composition:

```
InhomogeneousMedium(
    model=...,
    profile=...,
    effects=[...],
)
```

* Avoid deep inheritance hierarchies.
* Effects must be attachable and removable without changing transport code.
* A medium must be constructible without effects, and behave identically to one with
  an empty effect list.
* Effects must compose in a defined, recorded order; where effects do not commute,
  the order is a scientific parameter and belongs in metadata.

---

## Scenario Mapping

| Scenario | Medium | Notes |
| -------- | ------ | ----- |
| I | Homogeneous | Requires a declared homogenization rule |
| II | Inhomogeneous (depth-dependent) | Same model, same profile as I |
| III | Inhomogeneous + effects | Majorant must be recomputed |
