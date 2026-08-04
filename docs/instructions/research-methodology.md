# Research Methodology

## Purpose

This framework investigates the scientific impact of medium representation in UOWC
simulations.

The primary research question is:

> What differences arise when a depth-dependent underwater environment is
> approximated as a homogeneous medium?

The framework must support scientifically fair, reproducible, and statistically
defensible comparisons.

**Framing note.** "Difference" is the claim this framework can support. "Error"
presupposes that the depth-dependent representation is ground truth — which is a
statement about the *real ocean*, not about two simulations. Scenario II is a
better-resolved model, not a measurement. Wording that implies otherwise ("the error
introduced by the homogeneous approximation") is only admissible once validation
against measurement supports it, and must be stated as an assumption until then.

---

## Scientific Hierarchy

```
Environmental Parameters
        ↓
Optical-Property Model      (at a stated wavelength)
        ↓
Inherent Optical Properties
        ↓
Environmental Effects
        ↓
Medium Representation
        ↓
Photon Transport
        ↓
Metrics
        ↓
Scientific Conclusions
```

Identical to the hierarchy in scientific-modelling.md. Sensitivity studies must
respect this direction of causality — see *Sensitivity Analysis*.

---

## Research Scenarios

### Scenario I — Homogeneous

Spatially constant IOPs, derived from the same optical-property model and the same
environmental profile as Scenario II via an explicit **homogenization rule**.

Baseline / reference scenario.

### Scenario II — Depth-dependent

Depth-varying IOPs from the same model and the same profile.

Purpose: measure the effect of resolving vertical structure that Scenario I averages
away.

### Scenario III — Depth-dependent + environmental effects

Scenario II with effect modules enabled: turbulence, bubble layers, sediment,
chlorophyll variation, thermoclines, salinity gradients.

Purpose: measure the impact of environmental processes beyond depth dependence.

---

## Homogenization

**This is the central methodological issue of the study and must never be left
implicit.**

Scenario I requires collapsing a depth-varying profile to a single value. That
collapse is a *free choice*, not a neutral operation. Different rules give different
answers, and the headline quantity — "the difference between homogeneous and
depth-dependent" — **can be made arbitrarily large or small by choosing the rule.**

Therefore:

* The homogenization rule is a **declared experimental factor**, recorded in metadata
  and named in every reported result, figure caption and table.
* A Scenario I vs II number quoted without its rule is uninterpretable and must not be
  published.
* Results should be reported across **all** implemented rules, not a single favoured
  one. The spread across rules is itself a headline result: it bounds how much of the
  reported "difference" is physics and how much is bookkeeping.

### Averaging space matters: the Jensen bias

IOPs are **concave** functions of concentration (`a ∝ C^0.602`, `b_p ∝ C^0.62`). By
Jensen's inequality, for concave `φ`, `⟨φ(C)⟩ ≤ φ(⟨C⟩)`. Hence:

```
⟨ a(C(z)) ⟩_z   ≤   a( ⟨C(z)⟩_z )
```

**Averaging chlorophyll and then computing IOPs systematically overestimates
attenuation** relative to computing IOPs and then averaging them. Equality holds only
for a constant profile; the bias grows with the variance of `C` along the path.

This bias is confounded with the effect under study. Left undocumented, a portion of
the reported "homogeneous vs depth-dependent difference" is an artifact of
homogenizing in the wrong space, not a property of the channel.

### Rules and what each preserves

| Rule | Space | Preserves | Status |
| ---- | ----- | --------- | ------ |
| Surface value `C(z_min)` | chlorophyll | nothing optical | Naive-practitioner baseline. Include it because it is what is often done, not because it is defensible. |
| Depth-average `⟨C⟩` | chlorophyll | mean concentration | Carries the Jensen bias above. Must be reported with that caveat. |
| **Optical-depth-preserving** | IOP | `∫a ds`, `∫b ds`, hence `τ` and `ω̄₀` | The physically motivated rule. Preferred reference. |

The optical-depth-preserving rule, along the nominal source→receiver line of sight
parameterized by arc length `s ∈ [0, D]`:

```
ā = (1/D) ∫₀ᴰ a(s) ds        b̄ = (1/D) ∫₀ᴰ b(s) ds
```

This preserves the optical depth `τ = ∫c ds` **exactly**, and therefore reproduces the
unscattered (ballistic) transmittance `e^{-τ}` exactly. It also preserves the
path-averaged single-scattering albedo, so the scattering regime is not distorted.

Its limits must be stated: it does **not** preserve the multiply-scattered field,
because the scattered contribution depends on *where along the path* scattering
occurs, not only on the integral. It is defined along a nominal straight path, so for
a scattering-dominated link where photons wander far from the line of sight it is an
approximation whose quality degrades as `ω₀ → 1`.

Because this rule operates in IOP space, it returns IOPs directly and does not pass
through a scalar concentration. It is a distinct interface from a chlorophyll-space
rule, not a variant of one.

---

## Fair Comparison Rules

### Scenario I vs Scenario II

Only the medium representation, and its associated homogenization rule, may change.

Held fixed:

* Optical-property model and all its coefficients
* Wavelength
* Environmental profile (the *same* profile drives both scenarios)
* Source and receiver configuration
* Transport algorithm and estimator
* Simulation settings
* Seeding strategy

Note the asymmetry that cannot be removed: Scenario I consumes one derived scalar
where Scenario II consumes the whole profile. "Environmental parameters unchanged"
means *the same profile is the input to both*, not that no derived quantity differs —
by construction one must. The homogenization rule is the name of that difference and
is why it must be declared rather than buried.

### Scenario II vs Scenario III

Only the set of enabled environmental effects may change. All else identical,
including the profile and the seeding strategy.

Note that enabling an effect that raises extinction **requires the majorant to be
recomputed** (see transport.md). This is a correctness requirement, not a
configuration change, and does not count as an unfair difference.

### Multi-scenario experiments

Isolate the variable under investigation. Do not change multiple major factors
simultaneously.

---

## Statistical Comparison

### The relevant sample size is the number of detected photons

`N_launched` sets the cost; `N_detected` sets the precision of every metric computed
from the detected population. A run with `10⁹` launched and `40` detected photons has
40 samples of arrival time, not `10⁹`. Report both.

### Proportions: do not use the Wald interval

Capture probability in a UOWC link is routinely `10⁻⁶` or smaller. In this regime the
normal-approximation (Wald) interval `p̂ ± z√(p̂(1−p̂)/n)` is **invalid**: its coverage
collapses, and at `p̂ = 0` it returns the interval `[0, 0]` — a claim of perfect
certainty derived from no observations.

Use instead:

* **Wilson score interval** — the default. Well-behaved for small `p` and small
  counts, and non-degenerate at zero:

  ```
  centre     = (p̂ + z²/2n) / (1 + z²/n)
  half-width = z/(1 + z²/n) · √( p̂(1−p̂)/n + z²/(4n²) )
  ```

* **Clopper–Pearson** — where guaranteed (conservative) coverage is wanted.

* **Zero detections**: report the one-sided bound, not a point estimate of zero. The
  rule of three gives `p < 3/n` at 95 % confidence. "No photons detected" is an upper
  bound on the channel, and must be reported as one.

### Weighted estimators: variance over weights, not counts

For a weighted estimator with per-photon scores `wᵢ` (zero for undetected photons):

```
X̄  = (1/N) Σ wᵢ
SE = √( ( Σwᵢ² − (Σwᵢ)²/N ) / ( N (N−1) ) )
```

Effective sample size `N_eff = (Σwᵢ)² / Σwᵢ²`. When `N_eff` is far below `N_detected`,
the result is dominated by a few heavy photons and the confidence interval is
optimistic; report `N_eff` in that case.

### Comparing scenarios: use independent replications

Reusing a seed across scenarios (common random numbers) is intended to correlate them
and reduce the variance of their *difference*. Two facts must be accounted for:

1. **CRN does not fully work under delta tracking.** Scenarios I and II have different
   majorants, so they consume different numbers of variates per photon. The streams
   desynchronize at the first null collision and the induced correlation is partial
   and uncontrolled. "Same seed" does **not** mean paired samples here.
2. **If correlation exists at all, `Var(A−B) ≠ Var(A) + Var(B)`.** Differencing two
   independently-computed confidence intervals misstates the uncertainty on the
   study's central quantity — conservatively if the correlation is positive,
   anti-conservatively if negative.

The robust procedure, valid whether or not CRN works:

1. Run each scenario `R` independent replicates (`R ≥ 10`), each with its own
   independent seed set, all scenarios sharing the seed set within a replicate.
2. Form the per-replicate difference `Δᵣ = X_{A,r} − X_{B,r}`.
3. Report `Δ̄ ± t_{R−1, 0.975} · s_Δ/√R`, where `s_Δ` is the sample standard deviation
   of the `Δᵣ`.

This estimates the variance of the difference *directly* and needs no assumption about
the correlation structure. A difference whose interval spans zero is not a result.

---

## Convergence Studies

Monte Carlo outputs are statistical estimates.

### Stability is not convergence

An estimate that stops moving as `N` grows is **not** evidence of convergence. A
biased estimator is perfectly stable. In the rare-event regime a metric sits flat
because nothing has been detected yet, not because it has converged.

The criterion is the **error**, not the estimate:

* The standard error must fall as `N^{-1/2}`. Plot `SE · √N` against `N`: it must be
  flat. A rising trend indicates a heavy-tailed estimator; a falling trend indicates
  a bug.
* Successive estimates must lie within each other's confidence intervals.
* The killed-weight tally (transport.md) must be negligible; if histories are being
  truncated, more photons will converge to the wrong answer.

### Convergence is per-metric

Different metrics converge at very different rates, and a photon count justified by
one will be badly under-converged for another:

| Metric | Nature | Relative convergence |
| ------ | ------ | -------------------- |
| Received power fraction | proportion | fastest; relative SE ≈ `1/√N_det` |
| Mean arrival time | first moment | fast |
| RMS delay spread | second moment | slower; higher variance |
| 3 dB bandwidth | tail-sensitive | slowest; set by the late-arrival tail |

Demonstrate and report convergence **for every metric that appears in a conclusion**,
not once for the cheapest one.

### Workflow

1. Run increasing photon counts.
2. Verify `SE ∝ N^{-1/2}` per metric.
3. Evaluate confidence intervals.
4. Select and justify production settings.

Avoid arbitrary photon counts.

---

## Sensitivity Analysis

Sensitivity studies identify which parameters most influence channel behavior.

### Vary model inputs, not model outputs

`a` and `b` are **not** independent parameters — both are derived from the same
environmental state by the same model. Under a chlorophyll-parameterized model,
varying `a` while holding `b` fixed is **physically unrealizable**: it corresponds to
no water the model can describe, and it breaks the causal chain declared in the
scientific hierarchy above.

Sensitivity must therefore be run over **model inputs**:

* Chlorophyll concentration and profile shape (surface value, deep maximum depth and
  magnitude, gradient)
* CDOM and NAP loading
* Wavelength
* Link range and geometry
* Receiver aperture and field of view
* Source divergence
* Deployment depth
* Environmental-effect strength (turbulence RMS and correlation length, bubble void
  fraction, sediment load)
* Homogenization rule (a discrete factor — see *Homogenization*)

Direct sweeps of `a`, `b` or `ω₀` are permitted **only** when explicitly labelled as
model-free "what-if" probes, presented separately, and not attributed to any water
type.

### One-at-a-time has known limits

OFAT explores a cross-shaped slice through parameter space, detects no interactions,
and understates sensitivity when responses are non-monotone. It is an acceptable
screening tool. Where interactions are suspected — and depth structure × scattering
albedo is a prime candidate — use a factorial or variance-based (Sobol) design and say
which was used.

---

## Verification

Verification answers:

> Was the model implemented correctly?

Required checks, to run before any scientific interpretation:

* Unit tests of every sampler and coefficient path
* **Majorant invariance** — metrics must be independent of `c_max`
* **Beer–Lambert** — unscattered transmittance reproduces `e^{-cd}`
* **Profile collapse** — a constant profile in the inhomogeneous medium reproduces the
  homogeneous medium
* **Conservation** — `detected + absorbed + escaped + killed = launched`
* **Sampling validation** — free paths are exponential; `⟨cos θ⟩ = g`; sampled CDFs
  match analytic ones
* **Reproducibility** — identical seed and configuration reproduce identical output
  bit-for-bit
* **Round-trip** — results survive persistence and reload unchanged

---

## Validation

Validation answers:

> Does the model reproduce known physical behavior?

Examples: published literature, analytical approximations, benchmark simulations,
experimental measurements.

When comparing against literature, confirm before concluding anything:

* Which attenuation coefficient was reported (`c` or `K_d` — see
  scientific-modelling.md)
* Whether dB figures are optical or electrical (see metrics.md)
* Whether bandwidth is optical or electrical 3 dB
* The phase function used
* The receiver aperture and FOV, which dominate absolute received power

A discrepancy traceable to any of these is a units mismatch, not a physics finding.

Verification and validation are separate activities. Verification cannot substitute
for validation, and agreement with another simulation is not validation.

---

## Uncertainty Quantification

Distinguish, and never sum as though equivalent:

* **Monte Carlo variance** — reducible by more photons; quantified by the intervals
  above.
* **Model uncertainty** — phase function, Case-1 applicability, placeholder
  coefficients. Typically *larger* than Monte Carlo variance and not reducible by more
  photons. Bound it by re-running under alternative model choices.
* **Environmental uncertainty** — the profile itself is uncertain.
* **Measurement uncertainty** — in any reference data used for validation.

A tight Monte Carlo interval around a result computed from placeholder coefficients is
precision without accuracy. Report both, and do not let the former imply the latter.

---

## Randomness and Reproducibility

Record: random seed, RNG implementation and bit generator, stream/spawn layout,
scenario definition, full simulation configuration, code version, and library
versions.

Stream assignment must be independent of worker count and execution order, so that a
parallel run reproduces a serial one exactly.

---

## Publication-Quality Results

Scientific claims must be supported by:

* Reproducible simulations
* Quantitative evidence with stated uncertainty
* Clearly defined experimental conditions, including wavelength, phase function,
  homogenization rule and receiver geometry
* Demonstrated convergence for the specific metric claimed

Figures and tables must be reproducible directly from stored simulation outputs.

---

## Reproducibility Checklist

Before accepting a result:

* Scenario, model, wavelength and parameters are documented.
* Homogenization rule is documented (Scenario I).
* Phase function is documented.
* Random seeds and RNG implementation are recorded.
* Raw outputs are stored.
* Verification suite passes, including majorant invariance.
* Convergence is demonstrated for this metric.
* Uncertainty is reported with a valid interval estimator.
* Killed weight is negligible.
* Metrics and figures regenerate from stored outputs.

Results that cannot be reproduced should not be considered final.

---

## Research Priorities

1. Scientific correctness
2. Reproducibility
3. Statistical rigor
4. Interpretability
5. Extensibility
6. Performance

Performance improvements must preserve scientific validity. A variance-reduction
technique is a performance improvement **only** after it has been shown unbiased
against the analog reference.
