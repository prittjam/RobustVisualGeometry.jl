# Architecture

## Type Hierarchy

RobustVisualGeometry is organized around three abstract type hierarchies:

### Estimators

```
AbstractEstimator
├── MEstimator{L<:AbstractLoss}              IRLS solver
└── GNCEstimator{G<:GNCLoss}                 Graduated Non-Convexity
```

`MEstimator` wraps any loss function (Huber, Cauchy, Tukey, etc.) and solves via
Iteratively Reweighted Least Squares. `GNCEstimator` wraps a GNC-compatible loss
(TruncatedLS or Geman-McClure) and graduates from convex to non-convex.

### Robust Problems

```
AbstractRobustProblem                        Generic robust optimization
├── LinearRobustProblem                      Ax ≈ b with outliers
├── ConicTaubinProblem / ConicFNSProblem     Conic fitting
├── FMatTaubinProblem / FMatFNSProblem       F-matrix fitting
└── RansacRefineProblem                      Wraps RANSAC problem for IRLS
```

Each problem type implements a standard interface: `initial_solve`, `compute_residuals!`,
`weighted_solve`, `data_size`, `problem_dof`, and `convergence_metric`.

### RANSAC Problems

```
AbstractRansacProblem                        RANSAC problem interface
├── LineFittingProblem / LoLineFittingProblem
├── HomographyProblem / LoHomographyProblem
├── FundamentalMatrixProblem / LoFundamentalMatrixProblem
└── P3PProblem
```

RANSAC problems implement: `sample_size`, `data_size`, `model_type`, `solve`,
`residuals!`, and optionally `test_sample`, `test_model`, `refine`.

### Quality Functions

```
AbstractQualityFunction                      RANSAC quality scoring
├── ThresholdQuality (MSAC)
├── ChiSquareQuality
├── TruncatedQuality
├── MarginalQuality
└── PredictiveMarginalQuality
```

Quality functions score model hypotheses. `ThresholdQuality` is the standard MSAC
approach; `MarginalQuality` and `PredictiveMarginalQuality` use Bayesian marginal
likelihoods for automatic threshold selection.

## Composable Refinement

RANSAC refinement is composable via the `AbstractRefinement` trait:

- `NoRefinement` — use the minimal sample solution as-is
- `DltRefinement` — refit with DLT on inliers
- `IrlsRefinement` — refit with IRLS on inliers (most robust)

Local optimization (LO-RANSAC) is controlled via `AbstractLocalOptimization`:

- `NoLocalOptimization` — no local optimization

## Holy Trait Dispatch

Key traits that control RANSAC behavior at compile time:

| Trait | Values | Purpose |
|-------|--------|---------|
| `SolverCardinality` | `SingleSolution`, `MultipleSolutions` | Number of models per sample |
| `ConstraintType` | `Constrained`, `Unconstrained` | Whether the system needs SVD null-space |

## Dependency on VisualGeometryCore

RobustVisualGeometry imports geometry types, solvers, loss functions, and scale
estimators from VGC. It extends VGC's `rho`, `psi`, `weight`, `tuning_constant`,
`sampson_distance`, and `scale` with new methods.

```
VisualGeometryCore (types, solvers, losses, scale)
        ↓
RobustVisualGeometry (IRLS, GNC, RANSAC, problem implementations)
```
