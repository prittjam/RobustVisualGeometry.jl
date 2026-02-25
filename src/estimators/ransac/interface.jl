# =============================================================================
# RANSAC Interface — AbstractRansacProblem API, method stubs, RansacRefineProblem
# =============================================================================

# -----------------------------------------------------------------------------
# Abstract Problem Type
# -----------------------------------------------------------------------------

"""
    AbstractRansacProblem

Abstract type for RANSAC problem definitions.

A concrete subtype encapsulates the data, sampling strategy, minimal solver,
and residual computation for a specific estimation task.

# Required Methods
- `sample_size(p)::Int` — minimum data points for a minimal solve
- `data_size(p)::Int` — total number of data points
- `model_type(p)::Type` — concrete model type `M` (for workspace pre-allocation)
- `solve(p, indices)` — fit model(s) from sample indices
- `residuals!(r, p, model)` — compute residuals in-place

# Optional Methods
- `solver_cardinality(p)` — `SingleSolution()` or `MultipleSolutions()` (default)
- `draw_sample!(indices, p)` — custom sampling (default: uniform without replacement)
- `test_sample(p, indices)` — degeneracy check (default: `true`)
- `test_model(p, model)` — feasibility check (default: `true`)
- `refine(p, model, mask)` — local optimization on inliers (default: `nothing`)

# Example
```julia
struct LineFittingProblem <: AbstractRansacProblem
    points::Vector{Point2{Float64}}
end

sample_size(::LineFittingProblem) = 2
data_size(p::LineFittingProblem) = length(p.points)
model_type(::LineFittingProblem) = SVector{3, Float64}  # [a, b, c] for ax+by+c=0

function solve(p::LineFittingProblem, idx)
    p1, p2 = p.points[idx[1]], p.points[idx[2]]
    # ... return line model
end

function residuals!(r, p::LineFittingProblem, model)
    for i in eachindex(r)
        r[i] = dot(model, SA[p.points[i]..., 1.0]) / norm(model[SA[1,2]])
    end
end
```
"""
abstract type AbstractRansacProblem end

# -----------------------------------------------------------------------------
# Refinement Strategies (LO-RANSAC inner-loop refinement)
# -----------------------------------------------------------------------------

"""
    AbstractRefinement

Abstract type for RANSAC local optimization (LO-RANSAC) refinement strategies.

Concrete subtypes control what happens after a candidate model is scored:
- `NoRefinement()`: Plain RANSAC, no local optimization
- `DltRefinement()`: Re-estimate on inliers via DLT (fast, single solve)
- `IrlsRefinement(max_iter=5)`: IRLS with Sampson-corrected weights (slower, more accurate)

The refinement strategy is a type parameter on the problem, enabling
zero-cost compile-time dispatch.

# Example
```julia
prob = HomographyProblem(cs; refinement=NoRefinement())        # plain RANSAC
prob = HomographyProblem(cs; refinement=DltRefinement())       # DLT refit
prob = HomographyProblem(cs; refinement=IrlsRefinement())      # LO-RANSAC (default)
prob = HomographyProblem(cs; refinement=IrlsRefinement(max_iter=10))
```
"""
abstract type AbstractRefinement end

"""
    NoRefinement <: AbstractRefinement

No local optimization. The minimal solver output is scored directly.
Fastest option; use when speed matters more than accuracy.
"""
struct NoRefinement <: AbstractRefinement end

"""
    DltRefinement <: AbstractRefinement

Re-estimate the model on the inlier set using DLT (Direct Linear Transform).
Single linear solve, no iteration. Good balance of speed and accuracy.
"""
struct DltRefinement <: AbstractRefinement end

"""
    IrlsRefinement <: AbstractRefinement

Iteratively Reweighted Least Squares with Sampson-corrected weights.
Most accurate inner-loop refinement; default for both homography and F-matrix.

# Constructor
```julia
IrlsRefinement()             # 5 iterations (default)
IrlsRefinement(max_iter=10)  # custom iteration count
```
"""
Base.@kwdef struct IrlsRefinement <: AbstractRefinement
    max_iter::Int = 5
end

# -----------------------------------------------------------------------------
# Required Method Stubs
# -----------------------------------------------------------------------------

"""
    sample_size(problem::AbstractRansacProblem) -> Int

Return the minimum number of data points needed for a minimal solve.

E.g., 2 for line fitting, 4 for homography, 7 for fundamental matrix.
"""
function sample_size end

# data_size(problem) is declared in interface.jl alongside the AbstractRobustProblem
# interface. RANSAC problems also implement it.

"""
    model_type(problem::AbstractRansacProblem) -> Type

Return the concrete type of the model being estimated.

Used to parameterize `RansacWorkspace{M,T}` for type-stable storage.
Must be a concrete type (e.g., `SMatrix{3,3,Float64,9}`, `Vector{Float64}`).
"""
function model_type end

"""
    solve(problem::AbstractRansacProblem, indices::Vector{Int})

Fit model(s) from a minimal sample specified by `indices`.

Return type depends on `solver_cardinality(problem)`:
- `SingleSolution`: return `M` or `nothing`
- `MultipleSolutions`: return an iterable of `M` (e.g., `FixedModels{N,M}`) or `nothing`

Returns `nothing` if the solve fails (e.g., degenerate configuration).
"""
function solve end

"""
    residuals!(r::Vector, problem::AbstractRansacProblem, model)

Compute residuals for all data points given a model, writing into `r` in-place.

`r` is pre-allocated with length `data_size(problem)`. The function must fill
all elements. Residuals should be signed (or unsigned) scalar distances.
"""
function residuals! end

# -----------------------------------------------------------------------------
# Optional Method Stubs with Defaults
# -----------------------------------------------------------------------------

"""
    solver_cardinality(problem::AbstractRansacProblem) -> SolverCardinality

Return the solver cardinality trait for compile-time dispatch.

Default: `MultipleSolutions()` (safe, handles both cases).
Override to `SingleSolution()` for solvers that always return exactly one model.
"""
solver_cardinality(::AbstractRansacProblem) = MultipleSolutions()

"""
    codimension(problem::AbstractRansacProblem) -> Int

Co-dimension `d_g` of the model manifold: the number of independent constraint
equations `g(x̄, θ) = 0` that a true correspondence must satisfy.

Under H₀ with Gaussian observation noise, the residual `r` (distance from
observation to manifold) satisfies `(r/σ)² ~ χ²(d_g)`. This determines the
chi-square cutoff for inlier classification.

No default — each problem type must implement this method.

Common values:
- F-matrix / essential matrix (`d_g = 1`): one scalar epipolar constraint
- 2D line (`d_g = 1`): signed distance is scalar
- Homography (`d_g = 2`): two constraint equations from `v̄ = λHū`
- Absolute pose / PnP (`d_g = 2`): two reprojection equations
"""
function codimension end

"""
    draw_sample!(indices::Vector{Int}, problem::AbstractRansacProblem)

Fill `indices` with a random sample of data point indices (without replacement).

Default: delegates to `draw!(indices, sampler(problem))`.
Problems with a stored sampler just need to define `sampler(p::MyProblem) = p._sampler`.
"""
function draw_sample!(indices::Vector{Int}, problem::AbstractRansacProblem)
    draw!(indices, sampler(problem))
    nothing
end

"""
    sampler(problem::AbstractRansacProblem) -> AbstractSampler

Return the sampler used by this problem.

Default: `UniformSampler(data_size(p))`.
Override for problems with a stored sampler field:

    sampler(p::MyProblem) = p._sampler
"""
sampler(p::AbstractRansacProblem) = UniformSampler(data_size(p))

"""
    test_sample(problem::AbstractRansacProblem, indices::Vector{Int}) -> Bool

Check whether a sample is non-degenerate (e.g., points not collinear).

Default: `true` (no degeneracy check). Override for problem-specific checks.
"""
test_sample(::AbstractRansacProblem, ::Vector{Int}) = true

"""
    test_model(problem::AbstractRansacProblem, model) -> Bool

Check whether a fitted model is feasible (e.g., positive focal length).

Default: `true` (no feasibility check). Override for problem-specific checks.
"""
test_model(::AbstractRansacProblem, _model) = true

"""
    test_consensus(problem::AbstractRansacProblem, model, mask::BitVector) -> Bool

Validate a model against its consensus (inlier) set. Called after scoring and
mask computation, before accepting a candidate as the new best model.

Use this for checks that require the inlier set, e.g., the oriented epipolar
constraint for fundamental matrices. The `mask` is a `BitVector` where `true`
entries are classified inliers.

Default: `true` (no consensus check). Override for problem-specific checks.
"""
test_consensus(::AbstractRansacProblem, _model, _mask::BitVector) = true

"""
    refine(problem::AbstractRansacProblem, model, mask::BitVector)

Perform local optimization (e.g., IRLS) on inlier subset.

Return `(refined_model, scale)` or `nothing` to skip refinement.

Default: `nothing` (no refinement). Override to add LO-RANSAC behavior.
The `mask` is a BitVector where `true` indicates an inlier.
"""
refine(::AbstractRansacProblem, _model, ::BitVector) = nothing

"""
    solver_jacobian(problem, sample_indices, model) -> NamedTuple or nothing

Compute the Jacobian of the minimal solver evaluated at the given sample.

Returns a problem-specific NamedTuple containing the Jacobian(s), or `nothing`
if the problem does not support solver Jacobians.

Default: `nothing` (problem does not provide solver Jacobian).
"""
solver_jacobian(::AbstractRansacProblem, ::Vector{Int}, _model) = nothing

"""
    measurement_logdets!(out, problem, model)

Compute per-point covariance penalty ℓᵢ = log|Σ̃_{gᵢ}|_{Σ_θ=0}| (Eq. 12).

For each data point i, sets `out[i] = ℓᵢ` where Σ̃_{gᵢ} = ∂ₓgᵢ Σ̃_{xᵢ} (∂ₓgᵢ)ᵀ
is the model-certain constraint covariance shape (Section 3.3).

This penalty enters the covariance penalty term of Algorithm 1:
    −½ Σᵢ∈I log|Σ̃_{gᵢ}| = −½ L   where L = Σᵢ∈I ℓᵢ

Used in Phase 3 of `_try_model!` (re-scoring after local optimization).
For Phase 1 scoring, the heteroscedastic path computes ℓᵢ directly via
`residual_jacobian` to avoid duplicate computation.

Default: fills `out` with zeros (correct for isotropic problems where
Σ̃_{gᵢ} is constant and cancels in the sweep).
"""
function measurement_logdets!(out::AbstractVector, problem::AbstractRansacProblem, model)
    fill!(out, zero(eltype(out)))
    return out
end

"""
    measurement_covariance(problem::AbstractRansacProblem) -> CovarianceStructure

Return the measurement covariance structure trait for Σ̃_{xᵢ} (Section 3.3).

This trait determines which `_fill_scores!` method is dispatched for
model-certain scoring (`MarginalQuality`):

- `Homoscedastic()`:   Σ̃_{xᵢ} = I for all i (isotropic, dg=1). ℓᵢ = 0.
- `Heteroscedastic()`: Σ̃_{xᵢ} varies per point. ℓᵢ = log|Σ̃_{gᵢ}| via
  `residual_jacobian`.

The actual per-point covariance values are NOT returned by this function —
they are computed inside `residual_jacobian(problem, model, i)` which
returns the whitened quantities and ℓᵢ = log|Σ̃_{gᵢ}|.

Default: `Homoscedastic()`.
"""
measurement_covariance(::AbstractRansacProblem) = Homoscedastic()

"""
    residual_jacobian(problem, model, i) -> (rᵢ, ∂θgᵢ_w, ℓᵢ)

Whitened constraint, whitened model Jacobian, and measurement covariance
log-determinant at data point `i` (Section 3, Eq. 5-7, 12).

Returns a 3-tuple where:

- `rᵢ = L⁻¹gᵢ` — whitened constraint (Cholesky: LLᵀ = Σ̃_{gᵢ}|_{Σ_θ=0}).
  Satisfies `rᵢᵀrᵢ = gᵢᵀ Σ̃_{gᵢ}⁻¹ gᵢ = qᵢ` (weighted squared residual).
  For dg=1: this equals the signed Sampson residual rᵢ = gᵢ/√cᵢ (Eq. 21-22).
- `∂θgᵢ_w = L⁻¹ ∂θgᵢ` — whitened constraint Jacobian w.r.t. model θ.
  Satisfies `(∂θgᵢ_w)ᵀ(∂θgᵢ_w) = (∂θgᵢ)ᵀ Σ̃_{gᵢ}⁻¹ (∂θgᵢ)` (Fisher information).
- `ℓᵢ = log|Σ̃_{gᵢ}|_{Σ_θ=0}|` — covariance penalty for Algorithm 1 (Eq. 12).
  This is the measurement-only part of log|Σ̃_{gᵢ}|.

Return shapes depend on the codimension dg = codimension(problem):
- **dg=1** (line, F-matrix): `(T, SVector{n_θ,T}, T)`
- **dg≥2** (homography):     `(SVector{dg,T}, SMatrix{dg,n_θ,T}, T)`

For isotropic problems (Σ̃_{xᵢ} = I, dg=1), ℓᵢ = 0 (the constant cᵢ cancels
in Algorithm 1). For heteroscedastic problems (varying Σ̃_{xᵢ}), ℓᵢ varies
per point and must be included in the covariance penalty sum L = Σᵢ∈I ℓᵢ.

Callers needing only `(rᵢ, ∂θgᵢ_w)` can destructure as `r, G = residual_jacobian(...)`.

Default: not implemented (will error if called without a problem-specific method).
"""
function residual_jacobian end

"""
    constraint_type(problem::AbstractRansacProblem) -> ConstraintType

Return the constraint type trait for the problem's parameterization.
Default: `Constrained()`.
"""
constraint_type(::AbstractRansacProblem) = Constrained()

# =============================================================================
# IRLS Extension Points
# =============================================================================

"""
    weighted_system(problem, model, mask, w)

Build a weighted constraint system for IRLS refinement.

Returns a NamedTuple with at minimum `:A` (and `:b` for `Unconstrained`),
plus any context needed by `model_from_solution` (e.g., normalization transforms).
Returns `nothing` if the system cannot be built (too few inliers, etc.).
"""
function weighted_system end

"""
    model_from_solution(problem, x, sys)

Convert the raw solution vector `x` and system context `sys` back to a model.
Returns `nothing` if the solution is invalid.
"""
function model_from_solution end

# NOTE: SVDWorkspace and svd_nullvec! are defined in VisualGeometryCore
# (imported at module top level) since VGC's own solvers also need them.

# =============================================================================
# Trait-Dispatched Linear Solve
# =============================================================================

_ls_solve(::Constrained, A) = svd(A).Vt[end, :]
_ls_solve(::Constrained, A, ws::SVDWorkspace) = svd_nullvec!(ws, A, size(A, 1), Val(9))
_ls_solve(::Unconstrained, A, b) = (A' * A) \ (A' * b)

_dispatch_solve(::Constrained, sys) = _ls_solve(Constrained(), sys.A)
_dispatch_solve(::Constrained, sys, ws::SVDWorkspace) = _ls_solve(Constrained(), sys.A, ws)
_dispatch_solve(::Unconstrained, sys) = _ls_solve(Unconstrained(), sys.A, sys.b)
# Unconstrained + SVDWorkspace: some problems (e.g., HomographyProblem) use
# Unconstrained for covariance estimation but their weighted_system returns
# only (A, T₁, T₂) without a `b` field — fall back to SVD null-space.
function _dispatch_solve(::Unconstrained, sys, ws::SVDWorkspace)
    hasproperty(sys, :b) ? _ls_solve(Unconstrained(), sys.A, sys.b) :
                           _ls_solve(Constrained(), sys.A, ws)
end

# =============================================================================
# Shared Utility: Masked Gather (inlier compaction)
# =============================================================================

"""
    _gather_masked!(dst, src, mask::BitVector) -> Int

Compact elements of `src` where `mask` is `true` into `dst`, returning the count.

Composable per-array primitive for inlier gathering:
```julia
k = _gather_masked!(u₁_buf, cs.first, mask)
_gather_masked!(u₂_buf, cs.second, mask)
_gather_masked!(w_buf, w, mask)
```
"""
function _gather_masked!(dst, src, mask)
    k = 0
    @inbounds for i in eachindex(mask)
        if mask[i]
            k += 1
            dst[k] = src[i]
        end
    end
    return k
end

# =============================================================================
# Problem Lifecycle: prepare! (called before RANSAC main loop)
# =============================================================================

"""
    _prepare!(problem::AbstractRansacProblem)

Called once before the RANSAC main loop starts.

Default: resets the problem's sampler via `reset!(sampler(problem))`.
"""
_prepare!(p::AbstractRansacProblem) = (reset!(sampler(p)); nothing)

# =============================================================================
# Shared Utilities: Sampler Builder
# =============================================================================

"""
    _build_sampler(correspondences, m) -> AbstractSampler

Build the appropriate sampler from correspondences and minimal sample size `m`.

Returns `ProsacSampler` for scored correspondences (progressive sampling from
high-quality matches), `UniformSampler` otherwise.
"""
function _build_sampler(correspondences, m)
    C = eltype(correspondences)
    if scoring(C) isa HasScore
        scores = [Float64(c.attributes) for c in correspondences]
        order = sortperm(scores; rev=true)
        ProsacSampler(order, m)
    else
        UniformSampler(length(correspondences))
    end
end

# =============================================================================
# RansacRefineProblem — Adapter for IRLS refinement of RANSAC models
# =============================================================================

"""
    RansacRefineProblem{P,W} <: AbstractRobustProblem

Adapter wrapping an `AbstractRansacProblem` as an `AbstractRobustProblem`,
allowing `robust_solve(adapter, MEstimator(...))` to serve as IRLS refinement.

The three-step RANSAC solve (build system → SVD → reconstruct model) is
encapsulated in `weighted_solve`.

# Constructor
```julia
adapter = RansacRefineProblem(ransac_problem, mask, svd_workspace)
result = robust_solve(adapter, MEstimator(loss); init=model, scale=FixedScale(σ=σ), max_iter=5)
```

Always used with `init` kwarg — `initial_solve` is not defined.
"""
struct RansacRefineProblem{P<:AbstractRansacProblem, W} <: AbstractRobustProblem
    problem::P
    mask::BitVector
    svd_ws::W  # Union{Nothing, SVDWorkspace}
end

data_size(a::RansacRefineProblem) = data_size(a.problem)
problem_dof(a::RansacRefineProblem) = sample_size(a.problem)

function compute_residuals!(r::AbstractVector, a::RansacRefineProblem, model)
    residuals!(r, a.problem, model)
end

function compute_residuals(a::RansacRefineProblem, model)
    r = Vector{Float64}(undef, data_size(a.problem))
    residuals!(r, a.problem, model)
    r
end

function weighted_solve(a::RansacRefineProblem, model, ω)
    # Zero out non-inlier weights
    @inbounds for i in eachindex(ω, a.mask)
        if !a.mask[i]
            ω[i] = zero(eltype(ω))
        end
    end

    sys = weighted_system(a.problem, model, a.mask, ω)
    isnothing(sys) && return model

    x = if !isnothing(a.svd_ws)
        _dispatch_solve(constraint_type(a.problem), sys, a.svd_ws)
    else
        _dispatch_solve(constraint_type(a.problem), sys)
    end

    model_new = model_from_solution(a.problem, x, sys)
    isnothing(model_new) && return model
    model_new
end

convergence_metric(::RansacRefineProblem, θ_new, θ_old) =
    norm(θ_new - θ_old) / (norm(θ_new) + eps())
