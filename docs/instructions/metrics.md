# Metrics and Analysis

## Purpose

Metrics convert raw photon results into research outputs.

Metrics must be independent of transport implementation.

---

## Typical Outputs

Power Metrics

* Received power
* Power versus depth
* Path loss

Temporal Metrics

* Channel impulse response (CIR)
* Delay spread

Frequency Metrics

* Frequency response
* 3 dB bandwidth

Statistical Metrics

* Photon capture probability
* Capture counts
* Detection efficiency

---

## Design Rules

Metrics operate on simulation results.

Metrics should not:

* Launch photons
* Modify transport behavior
* Depend on plotting libraries

---

## Statistical Reporting

Whenever possible report:

* Mean
* Variance
* Confidence interval
* Sample count

Avoid reporting only point estimates.

---

## Extensibility

New metrics should be implemented as independent modules.

Existing metrics should not require modification.
