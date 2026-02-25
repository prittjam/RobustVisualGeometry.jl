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
solve
residuals!
test_sample
test_model
refine
solver_cardinality
draw_sample!
test_consensus
```

## Refinement

```@docs
AbstractRefinement
NoRefinement
DltRefinement
IrlsRefinement
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
constraint_type
weighted_system
model_from_solution
codimension
inlier_ratio
```

## SVD Workspace

Pre-allocated workspace for zero-allocation SVD computation. Defined in VisualGeometryCore,
re-exported here for RANSAC solvers.

- `SVDWorkspace{T}(max_rows, ncols)` — pre-allocate LAPACK buffers
- `svd_nullvec!(ws, A, m, Val(N))` — compute null-space vector of first `m` rows

## Fixed Models

```@docs
FixedModels
RansacRefineProblem
```
