# Photon Transport

## Purpose

The transport engine propagates photons through an underwater medium and records
detections.

It should remain independent of:

* Specific optical-property models
* Specific medium implementations
* Specific environmental effects
* Metrics
* Persistence

It reaches the medium only through the `Medium` port (`field`, `domain`,
`acceleration`). Adding a model, a profile, or an effect must never require a change
here.

---

## Primary Algorithm

Woodcock delta tracking (null-collision tracking) is the default transport algorithm.

All scenarios use the same transport implementation. Scientific comparisons must
isolate medium and environmental effects, never transport differences.

### Why delta tracking

In a spatially varying medium the free-path PDF is
`c(x(s)) * exp(-∫₀ˢ c(x(s')) ds')`, which cannot be inverted analytically. Delta
tracking avoids the integral by adding a fictitious ("null") absorber that brings the
total extinction up to a spatially constant majorant `c_max`. Free paths in the
homogenized medium are analytically samplable; the fictitious component is then
removed by rejecting a fraction of the collisions.

### The majorant condition

> **`c_max ≥ c(x)` for every `x` in the region being tracked, at every time `t`
> sampled.**

This is the correctness condition of the method. If it is violated anywhere, the
acceptance probability `c(x)/c_max` exceeds 1, the null-collision density becomes
negative, and the estimator is biased with **no runtime symptom**.

Consequences that must be respected:

* The majorant is a **medium** responsibility (`medium.acceleration.majorant(region)`),
  not a transport constant.
* It must bound the extinction **after all effects are applied**. An `OpticalEffect`
  that adds a scattering population (bubbles, sediment) raises `c(x)`; its
  `extinction_bound(region)` must be included in the bound. A majorant computed from
  the base medium alone is invalid for Scenario III.
* It is **regional**, so a thin high-extinction layer does not inflate the bound over
  the whole domain. A tighter bound is purely an efficiency gain.
* `c_max` affects **efficiency only, never the answer**. Doubling it doubles the null
  collisions and leaves every expectation unchanged. This is the algorithm's sharpest
  verification test — see *Verification hooks* below.

---

## Photon Lifecycle

```
Launch (position, direction, weight = 1, t = 0)
  │
  ├─► sample free-flight distance   s = -ln(ξ) / c_max
  │
  ├─► RECEIVER TEST over the segment (x, x + s·d]        ◄── before the collision
  │        ray/aperture intersection at t_hit ∈ (0, s]
  │        if hit and within FOV: record, terminate
  │
  ├─► advance   x ← x + s·d,   t ← t + s·n(x)/c₀
  │
  ├─► domain test: outside → escape, terminate
  │
  ├─► collision type: draw ξ' ~ U[0,1)
  │        ξ' <  c(x)/c_max   →  REAL collision
  │        ξ' ≥  c(x)/c_max   →  NULL collision:
  │                               direction, weight and photon state UNCHANGED
  │                               ── continue free flight ──┐
  │                                                          │
  ├─► real collision → absorption or scattering              │
  │        analog:   absorb with prob a(x)/c(x), else scatter│
  │        survival: w ← w · ω₀(x),  always scatter          │
  │        scatter:  pick population by bᵢ/b, sample the     │
  │                  population's phase function, rotate d   │
  │                                                          │
  ├─► Russian roulette if w < w_min                          │
  │                                                          │
  └──────────────────── loop ◄───────────────────────────────┘
```

### Null collisions are not optional

The null-collision branch **is** the algorithm. A lifecycle that treats every
candidate collision as physical propagates photons through a medium of extinction
`c_max` rather than `c(x)`, and is biased by that ratio. A null collision changes
nothing about the photon — not its direction, not its weight, not its scatter count.
It only ends one sampled free flight and starts another.

### Detection is tested along the segment, not at collision points

The receiver is a small aperture. Photons that cross it do so almost always *between*
collisions, not at one. Detection must therefore be a ray/aperture intersection
evaluated over the free-flight segment, accepted when:

* the ray approaches the front face (`d · n̂ < 0`),
* the intersection parameter satisfies `0 < t_hit ≤ s`,
* the hit lies within the aperture radius, and
* the incidence angle is within the field of view (`cos θᵢ ≥ cos(FOV)`).

Testing only at collision points discards nearly every real detection and understates
received power by orders of magnitude. Because a null collision does not change
direction, testing per-segment and testing over the concatenated straight path are
equivalent — but the test must not be deferred to *after* the collision is processed.

---

## Arrival Time

Arrival time is accumulated per segment using the **local** index:

```
t = ∫ n(x) / c₀  ds        c₀ = 299 792 458 m/s
```

Not `path_length / c₀`, and not a domain-constant `n`. This is the only route by which
a refractive effect (turbulence, thermocline) reaches a measurable quantity.

Two documented approximations:

* **Phase index used in place of group index.** Pulse energy travels at the group
  velocity `c₀/n_g`, where `n_g = n − λ (dn/dλ)`. For seawater near 450–550 nm
  `n ≈ 1.34` while `n_g ≈ 1.36`, so absolute delays carry a ≈2 % bias. RMS *delay
  spread* is a difference of times and is affected only through the spread of the
  index, so it is second-order — but absolute time-of-flight comparisons against
  literature must state which index was used.
* **Straight-line free flight.** Ray bending along `∇n` is not yet applied, so the
  index enters timing but not geometry. This is accurate while
  `|∇n| · L ≪ 1` over a correlation length; it is the assumption to revisit before
  claiming quantitative turbulence results.

---

## Weights and Termination

The analog estimator (absorb-or-scatter by `a/c` vs `b/c`, unit weights) is the
unbiased reference and the basis of the verification suite.

Weighted estimators are permitted where they reduce variance, and must remain
unbiased:

* **Survival biasing / implicit capture** — never absorb; multiply the weight by the
  single-scattering albedo `ω₀ = b/c` at each real collision. Unbiased, since
  `E[w] = ω₀ w`.
* **Russian roulette** — the *only* sanctioned termination for low-weight photons.
  Below a threshold `w_min`, kill with probability `q` and scale survivors by
  `1/(1−q)`. Unbiased: `E[w'] = (1−q) · w/(1−q) = w`.

### Do not truncate histories

Capping the number of collisions and discarding the survivor is a **biased**
termination. It preferentially removes long, heavily scattered paths — exactly the
paths that carry the delay-spread tail and set the 3 dB bandwidth. The effect is a
systematic *narrowing* of the measured impulse response, which looks entirely
plausible in a plot.

An iteration cap may exist as a **safety guard against non-termination only**. If it
ever fires it is a defect, not a modelling choice: the killed weight must be tallied
separately and reported, and a run in which it is non-negligible must be rejected
rather than interpreted.

Also forbidden: negative weights, non-finite positions/directions/times, and
un-normalized direction vectors after a scattering rotation.

---

## Sampling Conventions

Record these with the run; they are physical choices, not implementation details.

* **Free path** — `s = -ln(ξ)/c_max` with `ξ ~ U[0,1)`. Use `-log1p(-ξ)` so `ξ = 0`
  is safe and precision is preserved for small `ξ`.
* **Source cone** — sampling `cos θ ~ U[cos θ_div, 1]` gives a beam of **uniform
  radiance** within the cone (a flat-top beam). A Gaussian beam is a different
  distribution and must be implemented as such rather than approximated by a cone.
* **Scattering** — direction cosines are rotated by the sampled `(θ, φ)`. With more
  than one scattering population, first select population `i` with probability
  `bᵢ/b`, then sample that population's phase function. Averaging asymmetry
  parameters across populations is not equivalent and must not be used.
* **Streams** — free-path, collision-type, absorption, phase and azimuth draws should
  come from a reproducible stream layout that does not depend on worker count or
  execution order.

---

## Verification Hooks

Transport-specific checks that must pass before any result is interpreted:

* **Majorant invariance** — the same scenario run with `c_max`, `2·c_max` and
  `10·c_max` must agree within Monte Carlo error on every metric. This is the
  definitive test that the null-collision branch is correct.
* **Beer–Lambert** — in a homogeneous medium with an on-axis receiver, the unscattered
  (zero-scatter) detected fraction must reproduce `exp(-c·d)`.
* **Profile collapse** — an inhomogeneous medium built from a constant profile must
  reproduce the homogeneous medium result to within Monte Carlo error.
* **Conservation** — `detected + absorbed + escaped + killed` must equal the launched
  weight, to floating-point tolerance.
* **Sampling validation** — sampled free paths must match the exponential
  distribution; sampled `cos θ` must reproduce the phase function's mean
  (`⟨cos θ⟩ = g`) and its CDF.

---

## Transport Responsibilities

Responsible for:

* Photon launch and propagation
* Free-path and collision sampling, including the null-collision branch
* Weight updates and unbiased termination
* Optical-path time accumulation
* Domain-exit handling and delegation to a boundary
* Receiver intersection and acceptance testing
* Tallies sufficient to close the weight balance

Not responsible for:

* Optical-property generation
* Majorant computation
* Plotting, analysis, or file export

---

## Future Extensions

Future transport algorithms implement the same `TransportEngine` port. The rest of
the framework should not require modification.

Planned, in order of scientific impact:

1. **Next-event estimation / detector-directed scoring** — the analog estimator wastes
   nearly all photons on a small aperture. A deterministic
   scatter-toward-the-receiver contribution at each collision, weighted by the phase
   function and the transmittance along the connecting ray, reduces variance by orders
   of magnitude. Must be validated against analog before replacing it.
2. **Ray bending** along `∇n` during free flight, at which point delta tracking must
   step along a curved path rather than a straight one.
3. **Boundary interaction** — Fresnel reflection/refraction at the air–water surface
   and bottom reflectance (see mediums.md).
