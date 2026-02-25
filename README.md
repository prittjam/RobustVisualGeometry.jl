# RobustVisualGeometry.jl

Robust estimation algorithms for geometric vision: M-estimation (IRLS) with pluggable loss functions, Graduated Non-Convexity (GNC) for high outlier rates, and RANSAC with pluggable quality functions and LO-RANSAC local optimization. Includes problem implementations for lines, conics, homographies, fundamental matrices, and P3P pose estimation.

Depends on [VisualGeometryCore.jl](https://github.com/prittjam/VisualGeometryCore.jl) for geometry types, solvers, and loss/scale primitives.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/prittjam/RobustVisualGeometry.jl")
```

## Quick Examples

### RANSAC Homography

```julia
using RobustVisualGeometry
using VisualGeometryCore: csponds, SA

# Point correspondences (source → target)
src = [SA[100.0, 200.0], SA[300.0, 150.0], ...]
dst = [SA[112.0, 195.0], SA[310.0, 148.0], ...]

# RANSAC with marginal likelihood scoring (threshold-free, scale-free)
problem = HomographyProblem(csponds(src, dst))
scoring = MarginalQuality(problem, 50.0)  # a = outlier half-width
result = ransac(problem, scoring)

H = result.value                          # 3×3 homography matrix
inliers = result.inlier_mask              # BitVector
println("Inliers: $(sum(inliers)) / $(length(inliers))")
```

### RANSAC Fundamental Matrix

```julia
problem = FundamentalMatrixProblem(csponds(src, dst))
scoring = MarginalQuality(problem, 50.0)  # a = outlier half-width
result = ransac(problem, scoring)

F = result.value                          # 3×3 fundamental matrix
```

### LO-RANSAC (Local Optimization)

```julia
# ConvergeThenRescore: WLS to convergence, then re-sweep
result = ransac(problem, scoring; local_optimization=ConvergeThenRescore())

# StepAndRescore: single WLS step, then re-sweep
result = ransac(problem, scoring; local_optimization=StepAndRescore())
```

### Robust Line Fitting

```julia
using VisualGeometryCore: Point2

points = [Point2(1.0, 2.1), Point2(2.0, 4.0), ...]  # with outliers

result = fit_line_ransac(points; σ=0.5, confidence=0.999)
line = result.value       # Uncertain{Line2D} with covariance
inliers = result.inlier_mask
```

### M-Estimation (IRLS)

```julia
# Weighted least squares with robust loss
A = [1.0 0.0; 0.0 1.0; 1.0 1.0; 10.0 0.0]  # last row: outlier
b = [1.0, 1.0, 2.0, 100.0]

prob = LinearRobustProblem(A, b)
result = robust_solve(prob, MEstimator(TukeyLoss()))

x = result.value          # estimated parameters
w = result.weights        # final IRLS weights (outlier → 0)
```

### Conic (Ellipse) Fitting

```julia
using VisualGeometryCore: Point2

points = [Point2(x, y) for (x, y) in noisy_ellipse_data]

# Robust Taubin → FNS two-phase fitting
result = fit_conic_robust_taubin_fns(points; loss=GemanMcClureLoss())
ellipse = conic_to_ellipse(result.value, result.scale)
```

## Architecture

```
AbstractEstimator
├── MEstimator{L<:AbstractLoss}              IRLS solver
└── GNCEstimator{G<:GNCLoss}                 Graduated Non-Convexity

AbstractRobustProblem                        Generic robust optimization
├── LinearRobustProblem                      Ax ≈ b with outliers
├── ConicTaubinProblem / ConicFNSProblem     Conic fitting
└── FMatTaubinProblem / FMatFNSProblem       F-matrix fitting

AbstractRansacProblem                        RANSAC problem interface
├── LineFittingProblem
├── HomographyProblem
├── FundamentalMatrixProblem
└── P3PProblem

AbstractQualityFunction                      RANSAC quality scoring
├── MarginalQuality                          Threshold-free marginal likelihood
└── PredictiveMarginalQuality                Prediction-corrected variant

AbstractLocalOptimization                    LO-RANSAC strategies
├── NoLocalOptimization                      No local optimization (default)
├── ConvergeThenRescore                      WLS to convergence, then re-sweep
└── StepAndRescore                           Single WLS step, then re-sweep
```

### Key design decisions

- **Pluggable quality functions**: RANSAC loop is quality-agnostic via trait dispatch
- **LO-RANSAC via `fit`**: Problems implement `fit(problem, mask, weights, strategy)` for WLS refit; LO strategies (`ConvergeThenRescore`, `StepAndRescore`) alternate refit + re-sweep
- **Holy trait dispatch**: `SolverCardinality`, `ConstraintType`, `CovarianceStructure`, `FitStrategy` — compile-time dispatch on solver cardinality, gauge constraints, noise structure, and refit method
- **Uncertainty quantification**: Full covariance propagation through Hartley normalization

## Documentation

Full documentation available at `docs/build/index.html` after building:

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

## Testing

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```
