# Getting Started

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/prittjam/RobustVisualGeometry.jl")
```

## RANSAC Homography

Estimate a homography from noisy point correspondences with outliers:

```julia
using RobustVisualGeometry
using VisualGeometryCore: csponds, SA

# Point correspondences (source → target)
src = [SA[100.0, 200.0], SA[300.0, 150.0], ...]
dst = [SA[112.0, 195.0], SA[310.0, 148.0], ...]

# RANSAC with marginal likelihood scoring (threshold-free, scale-free)
problem = HomographyProblem(csponds(src, dst))
scoring = MarginalScoring(problem, 50.0)  # a = outlier half-width
result = ransac(problem, scoring)

H = result.value                          # 3×3 homography matrix
inliers = result.inlier_mask              # BitVector
println("Inliers: $(sum(inliers)) / $(length(inliers))")
```

## M-Estimation (IRLS)

Solve a linear system robustly in the presence of outliers:

```julia
# Weighted least squares with robust loss
A = [1.0 0.0; 0.0 1.0; 1.0 1.0; 10.0 0.0]  # last row: outlier
b = [1.0, 1.0, 2.0, 100.0]

prob = LinearRobustProblem(A, b)
result = fit(prob, MEstimator(TukeyLoss()))

x = result.value          # estimated parameters
w = result.weights        # final IRLS weights (outlier → 0)
```

## Robust Conic Fitting

Fit an ellipse to noisy points with outliers:

```julia
using VisualGeometryCore: Point2

points = [Point2(x, y) for (x, y) in noisy_ellipse_data]

# Robust Taubin → FNS two-phase fitting
result = fit_conic_robust_taubin_fns(points; loss=GemanMcClureLoss())
ellipse = conic_to_ellipse(result.value, result.scale)
```

## Next Steps

- [Architecture](architecture.md) — understand the type hierarchy
- [RANSAC](ransac.md) — quality functions, local optimization, stopping criteria
- [Fitting Algorithms](fitting.md) — conic, line, homography, F-matrix fitting
- [Extending](extending.md) — implement your own RANSAC problem
