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
├── AbstractTaubinProblem                    Taubin GEP fitting
│   ├── ConicTaubinProblem                   Conic fitting
│   └── FMatTaubinProblem                    F-matrix fitting
└── AbstractFNSProblem                       FNS iterative fitting
    ├── ConicFNSProblem                      Conic fitting
    └── FMatFNSProblem                       F-matrix fitting
```

Each problem type implements a standard interface: `initial_solve`, `compute_residuals!`,
`weighted_solve`, `data_size`, `problem_dof`, and `convergence_metric`.

### RANSAC Problems

```
AbstractRansacProblem                        RANSAC problem interface
├── LineFittingProblem
├── InhomLineFittingProblem
├── EivLineFittingProblem
└── AbstractCspondProblem                    Correspondence-based problems
    ├── AbstractDltProblem{T}                2D-2D with DLT buffer
    │   ├── HomographyProblem{T,S}
    │   └── FundMatProblem{T,S}
    └── P3PProblem{S,F}                     3D-2D (bearing rays + projection)
```

RANSAC problems implement: `sample_size`, `data_size`, `model_type`, `codimension`,
`solve`, `residuals!`, and optionally `test_sample`, `test_model`, `fit`.

### Quality Functions

```
AbstractScoring                      RANSAC quality scoring
└── MarginalScoring{P}                       Scale-free marginal likelihood
    P=Nothing  → model-certain (default)
    P=Predictive → prediction-corrected variant
```

Quality functions score model hypotheses. `MarginalScoring{Nothing}` and
`MarginalScoring{Predictive}` (aliased as `PredictiveMarginalScoring`) use
Bayesian marginal likelihoods for automatic threshold and scale selection.

## Local Optimization (LO-RANSAC)

Local optimization refines promising hypotheses during the RANSAC loop via
alternating refit-resweep cycles. Controlled by `AbstractLocalOptimization`:

- `NoLocalOptimization` — no local optimization (default)
- `PosteriorIrls` — posterior-weight IRLS refinement

Problems that support LO-RANSAC implement `fit(problem, mask, weights, ::LinearFit)`
for weighted least-squares fitting on the inlier subset.

## Holy Trait Dispatch

Key traits that control RANSAC behavior at compile time:

| Trait | Values | Purpose |
|-------|--------|---------|
| `SolverCardinality` | `SingleSolution`, `MultipleSolutions` | Number of models per minimal sample |

## Dependency on VisualGeometryCore

RobustVisualGeometry imports geometry types, solvers, loss functions, and scale
estimators from VGC. It extends VGC's `rho`, `psi`, `weight`, `tuning_constant`,
`sampson_distance`, `scale`, and `test_model` with new methods.

Geometry-level model feasibility checks (`test_model` for `HomographyMat`,
`FundMat`) live in VGC and dispatch on model type. RVG problem wrappers
extract sample points and delegate to these VGC methods.

```
VisualGeometryCore (types, solvers, losses, scale, test_model)
        ↓
RobustVisualGeometry (IRLS, GNC, RANSAC, problem implementations)
```
