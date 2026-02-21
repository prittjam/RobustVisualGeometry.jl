# RobustVisualGeometry.jl

Robust estimation algorithms for geometric vision.

## Features

- **M-estimation (IRLS)**: Iteratively Reweighted Least Squares with pluggable loss functions (Huber, Cauchy, Tukey, Geman-McClure, Welsch, Fair)
- **Graduated Non-Convexity (GNC)**: Handles high outlier rates by gradually tightening the loss function
- **RANSAC**: Full-featured implementation with pluggable quality functions, local optimization strategies, stopping criteria, and samplers
- **Problem implementations**: Lines, conics (ellipses), homographies, fundamental matrices, and P3P pose estimation

Depends on [VisualGeometryCore.jl](https://github.com/prittjam/VisualGeometryCore.jl) for geometry types, solvers, and loss/scale primitives.

## Contents

```@contents
Pages = [
    "guide/getting-started.md",
    "guide/architecture.md",
    "guide/ransac.md",
    "guide/fitting.md",
    "guide/extending.md",
]
Depth = 2
```
