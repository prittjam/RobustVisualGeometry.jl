# RANSAC

```@meta
CurrentModule = RobustVisualGeometry
```

## Main Function

```@docs
ransac
```

## Problem Interface

```@docs
AbstractRansacProblem
sample_size
model_type
codimension
solve
residuals!
test_sample
test_model
test_consensus
fit
solver_cardinality
draw_sample!
measurement_covariance
residual_jacobian
measurement_logdets!
solver_jacobian
```

## Samplers

```@docs
AbstractSampler
UniformSampler
ProsacSampler
sampler
```

## Configuration and Results

```@docs
RansacConfig
RansacEstimate
RansacAttributes
```

## Solver Traits

```@docs
SolverCardinality
SingleSolution
MultipleSolutions
ConstraintType
Constrained
Unconstrained
constraint_type
codimension
inlier_ratio
```

## Fit Strategy Trait

```@docs
FitStrategy
LinearFit
fit_strategy
```

## Workspace

```@docs
RansacWorkspace
```

## SVD Workspace

Pre-allocated workspace for zero-allocation SVD computation. Defined in VisualGeometryCore,
re-exported here for RANSAC solvers.

- `SVDWorkspace{T}(max_rows, ncols)` — pre-allocate LAPACK buffers
- `svd_nullvec!(ws, A, m, Val(N))` — compute null-space vector of first `m` rows

## Fixed Models

```@docs
FixedModels
```
