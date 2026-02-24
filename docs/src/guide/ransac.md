# RANSAC

The RANSAC implementation is fully modular: quality functions, local optimization strategies,
stopping criteria, and samplers can all be swapped independently.

## Basic Usage

```julia
using RobustVisualGeometry
using VisualGeometryCore: csponds, SA

src = [SA[100.0, 200.0], SA[300.0, 150.0], ...]
dst = [SA[112.0, 195.0], SA[310.0, 148.0], ...]

problem = HomographyProblem(csponds(src, dst))
quality = ThresholdQuality(CauchyLoss(), 3.0, FixedScale())
result = ransac(problem, quality)
```

## Quality Functions

Quality functions score how well a model hypothesis fits the data.
RobustVisualGeometry provides four quality functions: `ThresholdQuality`
(MSAC), `ChiSquareQuality`, `MarginalQuality`, and `PredictiveMarginalQuality`.

For detailed descriptions, mathematical formulas, a comparison table, and
the statistical framework for post-selection inference, see the
[Scoring Functions guide](scoring.md).

```julia
# MSAC with fixed threshold
scoring = ThresholdQuality(L2Loss(), 3.0, FixedScale(σ=1.0))

# Chi-square with significance level
scoring = ChiSquareQuality(FixedScale(σ=2.0), 0.01)

# Marginal likelihood (threshold-free, scale-free)
scoring = MarginalQuality(n, p, 50.0)

# Marginal + prediction correction
scoring = PredictiveMarginalQuality(n, p, 50.0)
```

## Refinement

### Global Refinement

Applied to the final best model:

| Type | Description |
|------|-------------|
| `NoRefinement()` | Use minimal sample solution |
| `DltRefinement()` | Refit with DLT on inliers |
| `IrlsRefinement()` | Refit with IRLS on inliers |

### Local Optimization (LO-RANSAC)

Applied during the RANSAC loop to promising hypotheses:

| Type | Description |
|------|-------------|
| `NoLocalOptimization()` | No local optimization |

The `Lo` prefix variants (e.g., `LoHomographyProblem`) enable local optimization.

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
