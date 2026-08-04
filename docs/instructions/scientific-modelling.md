# Scientific Modeling

## Purpose

This framework studies the impact of medium representation on UOWC channel behavior.

The primary research question is:

> What differences arise when a depth-dependent underwater environment is
> approximated as a homogeneous medium?

The framework must support scientifically fair comparisons where only the medium
representation changes. See research-methodology.md for what "only the medium
representation" actually requires — it is stronger than it sounds.

---

## Scientific Hierarchy

```
Environmental Parameters      (chlorophyll, CDOM, NAP, temperature, salinity, depth)
        ↓
Optical Property Model        (Haltrin, …)   — at a stated wavelength
        ↓
Inherent Optical Properties   (a, b, phase function)
        ↓
Environmental Effects         (parameter- / optical- / refractive-level)
        ↓
Medium Representation         (homogeneous | depth-dependent)
        ↓
Photon Transport
        ↓
Metrics
        ↓
Research Conclusions
```

This is the same hierarchy as research-methodology.md; the two must not drift apart.
Effects appear *before* the medium because they act at the level of parameters or
local optical state, and the medium is what exposes the result to transport.

---

## Inherent Optical Properties

Optical-property models convert environmental parameters into **inherent** optical
properties (IOPs) — properties of the water alone, independent of the ambient light
field.

| Symbol | Quantity | Unit | Definition |
| ------ | -------- | ---- | ---------- |
| `a(λ)` | Absorption coefficient | m⁻¹ | — |
| `b(λ)` | Scattering coefficient | m⁻¹ | `b = 2π ∫₀^π β(θ,λ) sin θ dθ` |
| `c(λ)` | Beam attenuation coefficient | m⁻¹ | **`c = a + b`** (definitional) |
| `β(θ,λ)` | Volume scattering function | m⁻¹ sr⁻¹ | — |
| `p(θ,λ)` | Scattering phase function | sr⁻¹ | `p = β/b`, with `∫_{4π} p dΩ = 1` |
| `g` | Asymmetry parameter | – | `g = ⟨cos θ⟩ = 2π ∫₀^π p(θ) cos θ sin θ dθ` |
| `b_b(λ)` | Backscattering coefficient | m⁻¹ | `b_b = 2π ∫_{π/2}^π β(θ,λ) sin θ dθ` |
| `ω₀(λ)` | Single-scattering albedo | – | `ω₀ = b/c` |

`ω₀` is the parameter that actually governs channel character: `ω₀ → 0` gives an
absorption-limited, near-ballistic link; `ω₀ → 1` gives a scattering-dominated link
with substantial multipath. Report it alongside `a` and `b` — a scenario is not
characterized by `c` alone.

### Beam attenuation `c` is not diffuse attenuation `K_d`

This distinction is the most common source of quantitative error in the UOWC
literature and must be stated in any comparison against published numbers.

* **`c` is an inherent optical property.** It removes *all* light from a collimated
  beam, including light merely scattered out of the acceptance angle. It is what a
  narrow-FOV transmissometer measures, and it is the coefficient Monte Carlo
  transport samples free paths from.
* **`K_d` is an apparent optical property.** It describes the exponential decay of
  downwelling irradiance in a broad, diffuse light field, and depends on the
  illumination geometry (sun angle, sky state) as well as the water.

Because forward-scattered light is retained in an irradiance measurement but lost in
a beam measurement, `a ≤ K_d ≪ c` in strongly forward-scattering natural waters,
often by a factor of 3–5. Substituting a published `K_d` for `c` (or vice versa)
misstates a link budget by several dB per metre.

Values taken from literature must record which quantity was reported and under what
geometry.

---

## Wavelength

**All IOPs are functions of wavelength.** There is no wavelength-free `a`, `b` or `c`
in this framework.

* Every IOP must carry the wavelength it was evaluated at, and IOPs from different
  wavelengths must never be combined.
* Model reference data (pure-water absorption and scattering, chlorophyll-specific
  absorption, CDOM slope) are valid only at the wavelength they were tabulated for.
  Reusing them at another wavelength is a silent physical error, so models should
  reject a mismatched wavelength rather than extrapolate implicitly.
* A scenario definition is incomplete without its wavelength. Blue-green
  (≈450–550 nm) is the low-attenuation window in clear ocean water; the optimum
  shifts toward green/yellow as chlorophyll and CDOM increase, so the "best"
  wavelength is itself scenario-dependent and should not be hard-coded as a constant.
* This framework is monochromatic. Source linewidth, dispersion of `n(λ)`, and
  wavelength-dependent detector response are outside the current model and must be
  declared as such.

---

## Scattering Directionality

Directionality lives in the phase function, never in a bare coefficient.

### Multiple populations

Scattering is represented as a **mixture** of populations (molecular water,
phytoplankton, mineral particles, bubbles), each with its own partial coefficient
`bᵢ` and its own phase function. Total `b = Σ bᵢ`.

A mixture must be sampled by first choosing population `i` with probability `bᵢ/b`
and then sampling that population's phase function. **Collapsing a mixture to a
single phase function with an averaged asymmetry parameter is not equivalent** — the
`b`-weighted mean of `g` reproduces neither the shape of the combined phase function
nor its backscatter fraction, both of which control delay spread.

### Henyey–Greenstein is a documented approximation, not the truth

The current implementation uses Henyey–Greenstein throughout. Its limitations are
material to this study and must appear in any reported assumptions:

* HG **underestimates the near-forward lobe** of natural seawater by orders of
  magnitude at `θ ≲ 1°`, and misrepresents the backscatter tail. Real oceanic VSFs
  (Petzold; Fournier–Forand) are far more sharply peaked.
* The near-forward lobe is precisely what sets multipath delay spread and therefore
  the 3 dB bandwidth. The choice of phase function is plausibly a **larger** error
  source than the homogeneous-vs-depth-dependent effect this study is designed to
  measure.
* `g ≈ 0.924` (Petzold average, Mobley 1994) is the conventional open-ocean
  particulate value and reproduces the correct *mean* cosine — but a single `g` does
  not pin down the shape, and two phase functions with identical `g` can give
  materially different impulse responses.
* Molecular scattering modeled as isotropic (`g = 0`) is a simplification; the
  Rayleigh phase function `p(θ) ∝ (1 + 0.835 cos²θ)` is near-symmetric but not flat.

Any bandwidth or delay-spread claim must state the phase function used. Where
feasible, report the metric under both HG and a Fournier–Forand/Petzold form to bound
the sensitivity.

---

## Optical Property Models

Models convert environmental parameters into IOPs at a stated wavelength.

Examples: Haltrin, future ocean-optics models.

**Kameda & Matsumura (1998) is not an optical-property model** despite being
frequently named alongside Haltrin in the UOWC literature. Its published contribution
is the vertical chlorophyll-profile parameterization `C(z)` (a shifted-Gaussian deep
chlorophyll maximum), implemented as `uowc.media.profiles.KamedaModel` — a *medium*
concern, not an optics one. It does not define its own `a(λ)` or `b(λ)`. A Kameda
profile is combined with an actual optical-property model (Haltrin here) to get IOPs;
inventing absorption/scattering coefficients and attributing them to "Kameda" would be
exactly the kind of unverifiable, unreferenced physics this document exists to
prevent.

Models must not depend on:

* Monte Carlo transport
* Metrics
* Plotting
* Storage

### Model scope must be explicit

Chlorophyll-parameterized (Case-1) models assume that all optically active
constituents co-vary with phytoplankton. They are **not valid** in Case-2 water where
CDOM and mineral particles vary independently — coastal, riverine, harbour and
resuspension-affected water. A scenario that exercises sediment or CDOM loading
independently of chlorophyll has left the Case-1 regime, and results from a Case-1
model there are extrapolation, not prediction.

### Established form vs. supplied data

Distinguish two kinds of number in a model, and document which is which:

* **Established model form and universal exponents** — e.g. chlorophyll absorption
  scaling `C^0.602` (Morel 1988), particle scattering `b_p(550) = 0.30 · C^0.62` with
  `b_p(λ) ∝ (550/λ)` (Gordon & Morel 1983).
* **Wavelength-specific reference data** — pure-water absorption and scattering,
  chlorophyll-specific absorption. These are *inputs*. Placeholder or illustrative
  values must be labelled as such and replaced with tabulated sources (e.g. Pope &
  Fry 1997; Smith & Baker 1981) before any quantitative claim.

A result computed from placeholder coefficients is a code test, not a measurement.

### Non-linearity is a first-class property

IOPs are **concave, sub-linear** functions of concentration (`C^0.602`, `C^0.62`).
This has a direct methodological consequence — averaging a concentration profile and
averaging the resulting IOPs are not the same operation. See research-methodology.md,
*Homogenization*, before using any depth-averaged quantity.

---

## Refractive Index

`n(x) = n₀ + Σᵢ n'ᵢ(x)`, with effect contributions added around a base index.

* Seawater `n ≈ 1.33–1.35` in the blue-green, increasing with salinity and pressure
  and decreasing with temperature.
* Turbulent fluctuations are small (`n' ~ 10⁻⁸–10⁻⁵`) but spatially **correlated**.
  Independent per-point noise is not turbulence: it has no correlation length, no
  well-defined gradient, and produces none of the physical effects of interest. A
  refractive effect must be a *realized* field — fixed by a seed, so the same position
  always returns the same value.
* A refractive effect contributes both an index perturbation and its **gradient**;
  the gradient is what bends rays and must be analytically consistent with the index
  (the derivative of the same expression, not a separate model).

---

## Scientific Validity

All scientific assumptions should be:

* Explicit
* Traceable
* Documented near the implementation that relies on them

Avoid hidden assumptions. Document a reference for every equation. Where a reference
could not be verified, say so in the docstring rather than implying authority.

Assumptions currently in force, which must be restated in any write-up:

* Monochromatic, unpolarized light
* Elastic scattering only (no Raman scattering, no fluorescence, no
  chlorophyll/CDOM re-emission)
* Steady-state medium over the flight time of a photon
* Scalar radiative transfer (no polarization tracking)
* Straight-line free flight (index affects timing, not geometry)
* Henyey–Greenstein phase functions
* No boundary interaction unless a boundary is explicitly configured

---

## Units

Use SI units for all transport, geometry and derived quantities.

* Distance: m
* Time: s
* Absorption, scattering, attenuation: m⁻¹
* Volume scattering function: m⁻¹ sr⁻¹
* Phase function: sr⁻¹
* Refractive-index gradient: m⁻¹

### Conventional non-SI units, permitted and required

Ocean optics has entrenched conventional units. Using SI here would break every
published coefficient and is **not** wanted. These are the sanctioned exceptions:

* Wavelength: **nm**
* Chlorophyll concentration: **mg m⁻³**
* Non-algal / mineral particles: **g m⁻³**
* Temperature: **°C**
* Salinity: **PSU** (dimensionless practical salinity)
* CDOM: absorption at 440 nm, m⁻¹

The rule is therefore: **never mix unit systems silently.** Every quantity carries its
unit in the field name, the docstring, or the type. Conversions happen at one
documented boundary, not opportunistically at call sites.

---

## Reproducibility

Scientific results must be reproducible. Simulation metadata must record:

* Optical-property model and its full coefficient set
* Wavelength
* Environmental parameters
* Medium representation, and the homogenization rule if one was applied
* Environmental effects and their parameters
* Phase function(s)
* Random seed and RNG implementation
* Scenario and simulation settings

See data.md for the authoritative metadata schema; that list and this one must agree.
