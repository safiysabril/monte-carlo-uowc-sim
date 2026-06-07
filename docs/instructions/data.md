# Data Management

## Purpose

Simulation outputs are research data.

Data storage must support:

* Reproducibility
* Large-scale experiments
* Downstream analysis

---

## Storage Format

Primary storage format:

* Parquet

Avoid embedding analysis logic inside storage layers.

---

## Metadata

Every simulation should record:

* Scenario
* Medium type
* Optical-property model
* Environmental effects
* Random seed
* Simulation parameters
* Timestamp

---

## Raw Results

Store raw simulation outputs whenever practical.

Derived metrics should be reproducible from stored results.

---

## Data Pipeline

Simulation
↓
Parquet Storage
↓
Analysis
↓
Visualization

Keep these stages independent.

---

## Versioning

Changes affecting scientific interpretation should be traceable.

Prefer explicit schema evolution over silent format changes.
