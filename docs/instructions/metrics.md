# Metrics and Analysis

## Purpose

Metrics convert raw photon results into research outputs.

Metrics must be independent of transport implementation. They consume a stored result
and produce a value with its uncertainty.

---

## Conventions That Must Be Fixed Before Anything Is Reported

Two conventions in optical wireless are routinely left implicit and each is worth a
factor of two. Both must be recorded with every result.

### Optical vs electrical decibels

In an intensity-modulated / direct-detection link the photocurrent is proportional to
**optical power**, so electrical power is proportional to *optical power squared*:

```
optical dB     PL_opt = −10 · log₁₀( P_rx / P_tx )
electrical dB  PL_el  = −20 · log₁₀( P_rx / P_tx )  =  2 · PL_opt
```

**This framework reports optical dB (`10 log₁₀`) by default.** Any electrical figure
must be labelled as such. A path-loss comparison against literature that ignores this
is wrong by exactly a factor of two in dB.

Path loss also requires a stated **reference**: `P_tx` is the launched optical power,
so `P_rx/P_tx` is the received power fraction — the same quantity as the capture
probability under an analog estimator. Geometric-spreading-only baselines, if used,
are a separate reported quantity, not a redefinition of path loss.

### Optical vs electrical 3 dB bandwidth

With `H(f)` the **optical-power** frequency response normalized to `H(0) = 1`:

```
optical    3 dB bandwidth  B_opt :  |H(B_opt)| = 0.5
electrical 3 dB bandwidth  B_el  :  |H(B_el)|² = 0.5   ⟺  |H(B_el)| = 1/√2 ≈ 0.707
```

Since `|H|` decreases with frequency, `B_el < B_opt`. **This framework reports the
electrical 3 dB bandwidth by default**, as it is the quantity that bounds achievable
data rate; the optical value must be labelled if reported.

---

## Power Metrics

### Received power fraction

```
P_rx / P_tx = Σᵢ wᵢ / N_launched
```

Under an analog estimator this is simultaneously the received power fraction, the
photon capture probability, and the detection efficiency. These are **one metric under
three names**, not three metrics — do not report them as independent results.

Uncertainty: a proportion interval (see *Uncertainty* below), never a bare point
estimate.

### Path loss

`−10 log₁₀(P_rx/P_tx)`, optical dB, per the convention above.

Undefined when no photons are detected. Report the one-sided bound derived from the
zero-detection rule instead of `∞` or a silently substituted floor.

### Power versus depth

Received power as the **link is placed at successive depths**, holding range and
geometry fixed — a family of independent runs, not a spatial field within one run.
State which is meant in any axis label; the two are different experiments.

---

## Temporal Metrics

### Channel impulse response

```
h(t) = (1/N_launched) · Σᵢ wᵢ · δ(t − tᵢ)
```

Normalized so that `∫h(t) dt` equals the received power fraction. This normalization
is what makes CIRs from runs with different photon counts comparable; an unnormalized
histogram is not a CIR.

`tᵢ` is the optical-path arrival time `∫n/c₀ ds` (transport.md), not `path/c₀`.

Report the CIR against **excess delay** `t − t₀`, where `t₀ = n̄·d/c₀` is the ballistic
arrival time, and state `t₀`. Absolute-time CIRs from different geometries cannot be
overlaid.

### Mean arrival time and RMS delay spread

```
τ̄     = Σwᵢtᵢ / Σwᵢ
τ_rms  = √( Σwᵢ(tᵢ − τ̄)² / Σwᵢ )
```

`τ_rms` is invariant to a constant time offset, so it is the same whether computed
about the absolute mean or the mean excess delay. `τ̄` is not, and must state its
reference.

Delay spread and bandwidth are inversely related, but the proportionality constant
depends on the shape of the impulse response. **Compute both from the CIR; do not
infer one from the other** via a rule of thumb.

`τ_rms` is a second moment and converges more slowly than mean power — its convergence
must be demonstrated separately (research-methodology.md).

---

## Frequency Metrics

### Compute `H(f)` directly from arrival times — do not bin first

The frequency response follows from the photon list without histogramming:

```
H(f) = Σᵢ wᵢ · exp(−j2πf tᵢ)  /  Σᵢ wᵢ
```

with `H(0) = 1` by construction.

**Binning the CIR before transforming introduces a systematic bandwidth bias.**
Binning at width `Δ` convolves `h(t)` with a rectangle, which multiplies the response
by `sinc(fΔ) = sin(πfΔ)/(πfΔ)`. At `f = 1/(2Δ)` this factor is `2/π ≈ 0.637` — an
attenuation of the same order as the 3 dB point being measured. The measured bandwidth
would then be a property of the bin width, not the channel.

If a binned CIR must be used, either divide `H(f)` by `sinc(fΔ)` to deconvolve the bin
shape, or choose `Δ` small enough that `sinc(fΔ) ≈ 1` across the entire band of
interest — and state which was done.

The direct sum has no bin width, no aliasing and no windowing artifact, and is the
required implementation.

### 3 dB bandwidth

Located by interpolation on `|H(f)|`, per the electrical/optical convention above.

`|H(f)|` from a finite photon sample is noisy and **not** guaranteed monotone. Locate
the crossing on a smoothed or fitted response, and report the frequency grid used. A
bandwidth read off a single noisy crossing is not a measurement — its uncertainty
should come from the replication procedure (research-methodology.md), not from a
single run.

Bandwidth is the most tail-sensitive metric in this framework and therefore the
slowest to converge.

---

## Statistical Metrics

* Detected photon count `N_det`
* Effective sample size `N_eff = (Σwᵢ)² / Σwᵢ²`
* Mean number of scattering events among detected photons
* Angular distribution of arrivals at the aperture

---

## Uncertainty

Every metric returns a value **and** its uncertainty. Point estimates alone are not
acceptable output.

### The sample size is `N_det`, not `N_launched`

`N_launched` sets the cost; `N_det` sets the precision of every metric derived from
the detected population. Report both, and `N_eff` where weights vary.

### Proportions

Use the **Wilson score interval** by default; Clopper–Pearson where conservative
coverage is required.

**The Wald interval `p̂ ± z√(p̂(1−p̂)/n)` must not be used here.** Capture probabilities
of `10⁻⁶` and below are normal for this channel, and in that regime Wald's coverage
collapses. At `p̂ = 0` it returns `[0, 0]` — certainty from no data.

Wilson:

```
centre     = (p̂ + z²/2n) / (1 + z²/n)
half-width = z/(1 + z²/n) · √( p̂(1−p̂)/n + z²/(4n²) )
```

**Zero detections**: report a one-sided upper bound (`p < 3/n` at 95 %, the rule of
three), not a point estimate of zero. No detections is an upper bound on the channel
and must be reported as one — never as a received power of exactly zero, and never
silently dropped from a plot.

### Weighted means

```
SE = √( ( Σwᵢ² − (Σwᵢ)²/N ) / ( N(N−1) ) )
```

Variance over **weights**, not over counts. Where `N_eff ≪ N_det`, the estimate is
dominated by a few photons and the interval is optimistic; flag it.

### Derived and compared quantities

Uncertainty must propagate through derived metrics — a delay spread with no interval
cannot produce a bandwidth with one.

Uncertainty on a **difference between scenarios** must come from the independent
replication procedure in research-methodology.md, not from differencing two
single-run intervals.

---

## Design Rules

Metrics operate on stored simulation results.

Metrics must not:

* Launch photons
* Modify transport behavior
* Depend on plotting libraries
* Depend on a specific transport engine, medium, or optical model
* Silently substitute a value for an undefined result (`0` for "no detections", a
  floor for `log(0)`, `nan` swallowed into a mean)

An undefined metric must be representable as undefined and carried through as such.

---

## Extensibility

New metrics are independent modules implementing the metric port. Existing metrics
should not require modification.

Comparative metrics (scenario A vs scenario B) are a separate port from single-run
metrics, because they carry the paired-uncertainty logic that a single-run metric
cannot express.
