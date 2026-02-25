# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Run all tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Run a single test file
julia --project=. -e 'include("test/test_ransac_homography.jl")'

# Run tests interactively with Revise
julia --project=.
julia> using Revise, Test
julia> include("test/test_ransac.jl")
```

Revise.jl caveats:
- Compact one-liner redefinitions sometimes not picked up; use `RobustVisualGeometry.eval(quote ... end)` as workaround
- Struct changes always require REPL restart
- Full test suite via `Pkg.test()` in fresh process avoids Revise issues

## Architecture

RobustVisualGeometry provides robust estimation algorithms: M-estimation (IRLS), Graduated Non-Convexity (GNC), and RANSAC with scale-free marginal scoring. Depends on VisualGeometryCore for geometry types, solvers, losses, and scale estimators.

### Module Structure

```
src/
├── RobustVisualGeometry.jl    # Entry point, exports, include ordering
├── interface.jl               # Shared interface (AbstractRobustProblem)
├── gep.jl                     # Generalized Eigenvalue Problem solver
├── estimators/
│   ├── irls.jl                # M-estimation (IRLS)
│   ├── gnc.jl                 # Graduated Non-Convexity
│   └── ransac/
│       ├── types.jl           # RansacConfig, RansacWorkspace
│       ├── traits.jl          # Holy traits: SolverCardinality, ConstraintType
│       ├── samplers.jl        # UniformSampler, ProsacSampler
│       ├── interface.jl       # AbstractRansacProblem API
│       ├── scoring.jl         # MarginalQuality, PredictiveMarginalQuality, sweep!
│       ├── loop.jl            # Main RANSAC loop (Algorithm 2)
│       └── problems/          # Problem implementations
│           ├── line.jl
│           ├── cspond.jl      # AbstractCspondProblem, AbstractDltProblem
│           ├── p3p.jl
│           ├── homography.jl
│           └── fundmat.jl
└── fitting/
    ├── conic/                 # Conic fitting (ALS, Taubin, FNS, GNC, geometric)
    ├── line.jl
    ├── line_ransac.jl
    ├── homography.jl
    └── fundmat.jl
```

### Include Order

Include order in `src/RobustVisualGeometry.jl` is dependency-aware:

1. **interface.jl, gep.jl** — shared types loaded first
2. **estimators/irls.jl, gnc.jl** — M-estimation framework
3. **estimators/ransac/** — framework files (types → traits → samplers → interface → scoring → loop)
4. **fitting/conic/** — conic fitting pipelines
5. **fitting/line.jl** — line fitting
6. **estimators/ransac/problems/** — RANSAC problem implementations (depend on scoring + fitting)
7. **fitting/\*.jl** — high-level fitting pipelines (depend on RANSAC problems)

Scoring files load before problem files. Problem-specific specializations go in `problems/`.

### Holy Traits System

Compile-time dispatch via traits defined in `estimators/ransac/traits.jl` and `estimators/ransac/scoring.jl`:

| Trait | Values | Purpose |
|-------|--------|---------|
| `SolverCardinality` | `SingleSolution`, `MultipleSolutions` | Number of models per minimal sample |
| `ConstraintType` | `Constrained`, `Unconstrained` | Gauge constraint (`Ah=0` via SVD) vs free coordinates (`Ax≈b` via WLS) |
| `CovarianceStructure` | `Homoscedastic`, `Heteroscedastic`, `Predictive` | Constraint covariance shape Σ̃\_{gᵢ} for scoring (Section 3.3) |

## Code Conventions

- **Functions**: `snake_case` (`residuals!`, `sample_size`, `homography_jac_det`)
- **Types**: `CamelCase` (`HomographyProblem`, `RansacWorkspace`, `MarginalQuality`)
- **Constants**: `UPPERCASE_SNAKE_CASE`
- **Section dividers**: `# =============================================================================`
- **Private helpers**: Prefix with `_` (`_try_model!`, `_adaptive_trials`, `_eiv_covariance`)

## Function Signature Conventions

Functions follow consistent argument ordering based on semantic role (adopted from VisualGeometryCore):

1. **ACTION** (operator first): `action(operator, target)`
   - The operator/transform acts ON the target
   - Example: `project(camera, point3d)` — camera projects the point
   - Example: `homography_jac_det(H, u)` — Jacobian of H at point u

2. **QUERY** (subject first): `query(subject, reference)`
   - Query about the subject relative to a reference
   - Example: `sampson_distance(u₁, u₂, H)` — error of points under H
   - Example: `distance(point, line)` — distance from point to line

3. **ACCESSOR** (single argument): `property(object)`
   - Get a property of an object
   - Example: `sample_size(problem)`, `model_type(problem)`

4. **MUTATING** (output first): `compute!(output, problem, model, ...)`
   - Julia convention: mutated argument first, then inputs
   - Example: `residuals!(r, problem, model)` — fill r with residuals
   - Example: `score!(ws, problem, scoring, model, cov)` — fill workspace scores

5. **RANSAC interface** (problem first): `method(problem, model, ...)`
   - Problem carries data, model is the hypothesis
   - Example: `test_model(problem, model, sample_indices)`
   - Example: `solve(problem, sample_indices)`
   - Example: `fit(problem, inlier_mask, weights)`

## Git Commit Conventions

- **NEVER** add "Co-Authored-By: Claude" or similar AI attribution to commit messages
- **NEVER** add "Generated with Claude Code" or similar footers
- Write clean, concise commit messages focused on the change itself

## Export Organization

Exports follow numbered category subsections with `-` dividers:

```julia
# -----------------------------------------------------------------------------
# 1. ABSTRACT TYPES - Interface definitions
# -----------------------------------------------------------------------------
export AbstractEstimator

# -----------------------------------------------------------------------------
# 2. ROBUST PROBLEM INTERFACE - Generic problem API
# -----------------------------------------------------------------------------
export initial_solve, compute_residuals, ...
```

Group related exports together (types, then functions). Add brief comments for exports whose purpose isn't obvious.

## Benchmarking

Use `@btime` or `@benchmark` from BenchmarkTools.jl. Never use `@time` for benchmarking (includes JIT compilation on first run).

```julia
using BenchmarkTools
@btime my_function($input)  # $ interpolates to avoid benchmarking global access
```

BenchmarkTools and Cthulhu are in the global environment — always available regardless of project.
