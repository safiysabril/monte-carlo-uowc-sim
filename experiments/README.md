# Experiments

Concrete study definitions: declarative config plus a thin entry script.

The reusable orchestration **engine** lives in `src/uowc/experiments/`
(composition root, scenario/campaign builders, runner). This directory holds
study **instances** that use that engine.

- `medium_representation/` - flagship study: homogeneous vs depth-dependent vs
  depth-dependent + environmental effects.

Raw outputs are written to `data/` and figures to `figures/` (both git-ignored).
