# Interface

```@meta
CurrentModule = RobustVisualGeometry
```

## Abstract Types

```@docs
AbstractEstimator
AbstractRobustProblem
```

## Robust Problem Interface

```@docs
initial_solve
compute_residuals
compute_residuals!
weighted_solve
data_size
problem_dof
convergence_metric
```

## Result Types

```@docs
RobustEstimate
RobustAttributes
```

## Result Accessors

StatsBase-compatible accessors for `RobustEstimate`:

- `coef(result)` — estimated parameters
- `residuals(result)` — residual vector
- `weights(result)` — final IRLS weights
- `scale(result)` — estimated scale
- `converged(result)` — convergence flag
- `niter(result)` — number of iterations

## Workspace

```@docs
IRLSWorkspace
```
