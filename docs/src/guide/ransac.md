# RANSAC

The RANSAC implementation is fully modular: quality functions, local optimization strategies,
and samplers can all be swapped independently.

## Basic Usage

```julia
using RobustVisualGeometry
using VisualGeometryCore: csponds, SA

src = [SA[100.0, 200.0], SA[300.0, 150.0], ...]
dst = [SA[112.0, 195.0], SA[310.0, 148.0], ...]

problem = HomographyProblem(csponds(src, dst))
quality = MarginalScoring(problem, 50.0)  # a = outlier half-width
result = ransac(problem, quality)
```

## Quality Functions

Quality functions score how well a model hypothesis fits the data.
RobustVisualGeometry provides two quality functions based on Bayesian marginal
likelihoods: `MarginalScoring` and `PredictiveMarginalScoring`. Both are
threshold-free and scale-free.

For detailed descriptions and mathematical formulas, see the
[Scoring Functions guide](scoring.md).

```julia
# Marginal likelihood (threshold-free, scale-free)
# Problem-aware constructor derives n, p, codimension automatically
scoring = MarginalScoring(problem, 50.0)    # a = outlier half-width

# Marginal + prediction correction (accounts for minimal-sample uncertainty)
scoring = PredictiveMarginalScoring(problem, 50.0)
```

## Local Optimization (LO-RANSAC)

Local optimization refines promising hypotheses during the RANSAC loop via
alternating refit-resweep cycles. Problems that support LO-RANSAC implement
`fit(problem, mask, weights, ::LinearFit)` for weighted least-squares fitting.

| Type | Description |
|------|-------------|
| `NoLocalOptimization()` | No local optimization (default) |
| `PosteriorIrls()` | Posterior-weight IRLS refinement |

```julia
# LO-RANSAC with PosteriorIrls
problem = HomographyProblem(csponds(src, dst))
scoring = MarginalScoring(problem, 50.0)
result = ransac(problem, scoring; local_optimization=PosteriorIrls())
```

Both strategies use monotonicity guards: the outer loop terminates when the
score stops improving.

## Samplers

| Sampler | Description |
|---------|-------------|
| `UniformSampler()` | Uniform random sampling |
| `ProsacSampler()` | Progressive sampling using match quality ordering |

## Configuration

`RansacConfig` controls the RANSAC loop:

```julia
config = RansacConfig(;
    max_trials=1000,
    confidence=0.999,
)
result = ransac(problem, quality; config=config)
```

## Result Type

`ransac` returns a `RansacEstimate`:

```julia
result.value        # estimated model (e.g., 3×3 matrix)
result.inlier_mask  # BitVector of inliers
result.quality      # quality score
result.trials       # number of RANSAC iterations used
```

Additional fields available from `RansacAttributes`:

```julia
result.scale        # s = √(RSS/ν)
result.dof          # ν = n_inliers - p
```
