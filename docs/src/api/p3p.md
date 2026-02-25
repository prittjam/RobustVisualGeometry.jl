# P3P

```@meta
CurrentModule = RobustVisualGeometry
```

## RANSAC Problem

```@docs
P3PProblem
```

## Result Types

**`Pose3`** — Type alias for `EuclideanMap{3,Float64,RotMatrix{3,Float64,9},SVector{3,Float64}}`.
A rigid 3D pose (rotation + translation). Defined in VisualGeometryCore, re-exported here.

`model_type(::P3PProblem) = Pose3` and `solver_cardinality(::P3PProblem) = MultipleSolutions()`,
so `solve` returns `FixedModels{4, Pose3} | nothing` (up to 4 P3P solutions).

## Correspondence Infrastructure

`AbstractCspondProblem` is the abstract supertype for RANSAC problems based on
point correspondences (homographies, fundamental matrices, P3P).
