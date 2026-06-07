# Python Development

## Language

Primary language: Python 3.12+

## Style

Prefer:

- dataclasses
- type hints
- pathlib
- Protocols
- composition

Avoid:

- global state
- circular imports
- deep inheritance

## Scientific Computing

Preferred libraries:

- numpy
- scipy
- pandas
- pyarrow

Visualization:

- matplotlib

## Data Storage

Use:

- parquet
- pyarrow

## Testing

Use:

- pytest

All scientific components should be unit-testable.

## Numerical Design

Favor:

- vectorization where practical
- deterministic behavior
- explicit units
- reproducibility

Scientific correctness takes priority over micro-optimizations.