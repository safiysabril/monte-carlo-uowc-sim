# Photon Transport

## Purpose

The transport engine propagates photons through an underwater medium.

It should remain independent of:

* Specific optical-property models
* Environmental effects
* Metrics
* Persistence

---

## Primary Algorithm

Woodcock Delta Tracking is the default transport algorithm.

All scenarios use the same transport implementation.

Scientific comparisons must isolate:

* Medium effects
* Environmental effects

and not transport differences.

---

## Transport Responsibilities

The transport engine is responsible for:

* Photon propagation
* Interaction sampling
* Weight updates
* Boundary handling
* Receiver detection

The transport engine is not responsible for:

* Optical-property generation
* Plotting
* Data analysis
* File export

---

## Photon Lifecycle

Launch
↓
Free-path sampling
↓
Position update
↓
Interaction determination
↓
Scattering or absorption
↓
Receiver test
↓
Termination

---

## Numerical Stability

Avoid:

* Negative weights
* Infinite loops
* Unbounded step counts

Validate all sampled values.

---

## Future Extensions

Future transport algorithms should implement the same interfaces as Woodcock Delta Tracking.

The rest of the framework should not require modification.
