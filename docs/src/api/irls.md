# IRLS

```@meta
CurrentModule = RobustVisualGeometry
```

## Estimator

```@docs
MEstimator
```

## Problem

```@docs
LinearRobustProblem
```

## Solver

The `fit` function is the unified entry point for robust estimation. For IRLS:

```julia
result = fit(problem, MEstimator(TukeyLoss()))
result = fit(problem, GNCEstimator())
```

See the [RANSAC API](@ref) for the RANSAC `fit(problem, mask, weights, ::LinearFit)` method.

## Loss Functions

The following loss functions are re-exported from VisualGeometryCore for convenience.
See the [VisualGeometryCore documentation](https://prittjam.github.io/VisualGeometryCore.jl) for full API details.

- `AbstractLoss` — abstract supertype
- `L2Loss` — standard least squares (no robustness)
- `HuberLoss` — linear penalty for large residuals
- `CauchyLoss` — logarithmic penalty (heavy-tailed)
- `TukeyLoss` — biweight, hard redescender (zero weight beyond threshold)
- `GemanMcClureLoss` — smooth redescender
- `WelschLoss` — exponential penalty
- `FairLoss` — log-linear penalty

Key functions: `rho(loss, r)`, `psi(loss, r)`, `weight(loss, r)`, `tuning_constant(loss)`.

## Scale Estimators

Re-exported from VisualGeometryCore:

- `AbstractScaleEstimator` — abstract supertype
- `MADScale` — Median Absolute Deviation
- `WeightedScale` — weighted variance estimator
- `FixedScale` — user-specified fixed scale
- `SpatialMADScale` — spatial MAD for structured data

Key functions: `estimate_scale(estimator, residuals)`, `chi2_threshold(loss, dof)`.
