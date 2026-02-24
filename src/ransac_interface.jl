# =============================================================================
# RANSAC Interface — Abstract types, traits, and problem interface
# =============================================================================
#
# Defines the problem interface, configuration, workspace, and result types
# for the RANSAC estimator. Users implement concrete subtypes of
# AbstractRansacProblem for their specific estimation task.
#
# DEPENDENCY: Uses Attributed{V,A} from base/types.jl (loaded before Estimators)
#
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
# NOTE: Quality functions, F-test types, local optimization strategies, and
# stopping strategies have been moved to scoring.jl (included before this file).
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Holy Trait: Solver Cardinality
# -----------------------------------------------------------------------------

"""
    SolverCardinality

Holy Trait indicating whether a minimal solver returns one or multiple candidate models.

Used for compile-time dispatch to eliminate iteration overhead for single-model solvers.

- `SingleSolution()`: `solve` returns `M` or `nothing`
- `MultipleSolutions()`: `solve` returns an iterable of `M` or `nothing`
"""
abstract type SolverCardinality end

"""
    SingleSolution <: SolverCardinality

Trait indicating the solver returns exactly one model (or nothing).
Enables a faster code path without iteration over candidates.
"""
struct SingleSolution <: SolverCardinality end

"""
    MultipleSolutions <: SolverCardinality

Trait indicating the solver may return multiple candidate models.
The RANSAC loop iterates over all candidates and keeps the best.
"""
struct MultipleSolutions <: SolverCardinality end

# -----------------------------------------------------------------------------
# FixedModels: Zero-allocation iterable for multiple solver solutions
# -----------------------------------------------------------------------------

"""
    FixedModels{N, M}

Stack-allocated container for up to `N` candidate models of type `M`.

Supports `for model in solutions` iteration and `length(solutions)`.
Used by `MultipleSolutions` solvers to avoid heap-allocating a `Vector`
in the RANSAC hot loop.

`isbitstype(FixedModels{N,M}) == true` when `isbitstype(M) == true`.
"""
struct FixedModels{N, M}
    count::Int
    data::NTuple{N, M}
end

@inline function Base.iterate(fm::FixedModels, i::Int=1)
    i > fm.count && return nothing
    return (@inbounds fm.data[i], i + 1)
end
Base.length(fm::FixedModels) = fm.count
Base.eltype(::Type{FixedModels{N,M}}) where {N,M} = M

# -----------------------------------------------------------------------------
# Abstract Sampler Type
# -----------------------------------------------------------------------------

"""
    AbstractSampler

Abstract type for RANSAC sampling strategies.

Subtypes implement `draw!` and `reset!` to provide different sampling behaviors
(uniform, PROSAC, NAPSAC, etc.). Problems store a sampler as a type-parameterized
field for compile-time dispatch.

# Required Methods
- `draw!(indices::Vector{Int}, sampler)` — fill indices with a random sample
- `reset!(sampler)` — reset mutable state for reuse across `ransac()` calls

# Built-in Samplers
- `UniformSampler`: Stateless uniform random sampling
- `ProsacSampler`: Progressive sampling from high-quality matches
"""
abstract type AbstractSampler end

"""
    UniformSampler <: AbstractSampler

Stateless uniform random sampling without replacement.

Stores the data size `n` for sampling from `1:n`.
"""
struct UniformSampler <: AbstractSampler
    n::Int
end

draw!(indices::Vector{Int}, s::UniformSampler) = (_draw_uniform!(indices, s.n); nothing)
reset!(::UniformSampler) = nothing

"""
    sampler(problem::AbstractRansacProblem) -> AbstractSampler

Return the sampler used by this problem.

Default: `UniformSampler(data_size(p))`.
Override for problems with a stored sampler field:

    sampler(p::MyProblem) = p._sampler
"""
sampler(p::AbstractRansacProblem) = UniformSampler(data_size(p))

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
    refine(problem, model, mask, loss, σ)

5-argument refine called from RANSAC inner loop with loss function and scale.
Default: delegates to 3-argument `refine(problem, model, mask)`.
Override for IRLS refinement (see `irls_refine`).
"""
refine(p::AbstractRansacProblem, model, mask::BitVector,
       ::AbstractLoss, ::Real) = refine(p, model, mask)

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

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

"""
    RansacConfig

Configuration parameters for the RANSAC algorithm.

# Fields
- `confidence::Float64=0.99`: Desired probability of finding an all-inlier sample
- `max_trials::Int=10_000`: Maximum number of random trials
- `min_trials::Int=100`: Minimum number of trials (overrides adaptive stopping)

# Example
```julia
config = RansacConfig(confidence=0.999, max_trials=50_000)
result = ransac(problem, CauchyLoss(), 3.0; config)
```
"""
Base.@kwdef struct RansacConfig
    confidence::Float64 = 0.99
    max_trials::Int = 10_000
    min_trials::Int = 100
end

# -----------------------------------------------------------------------------
# Result Type — RansacAttributes + Attributed{V,A}
# -----------------------------------------------------------------------------

"""
    RansacAttributes{T} <: AbstractAttributes

RANSAC-specific attributes stored in `Attributed{M, RansacAttributes{T}}`.

# Constructor
    RansacAttributes(stop_reason; inlier_mask, residuals, weights, quality, scale, trials,
                     sample_acceptance_rate, dof=0)

`converged` is derived automatically from `stop_reason`.

# Fields
- `stop_reason::Symbol`: `:converged` (found model) or `:no_model`
- `converged::Bool`: Whether a valid model was found (derived)
- `inlier_mask::BitVector`: Boolean mask over data (true = inlier)
- `residuals::Vector{T}`: Final residuals for all data points
- `weights::Vector{T}`: Robust weights from loss function
- `quality::T`: Total model quality (higher = better)
- `scale::T`: Estimated residual scale (s = √(RSS/ν) when dof > 0)
- `dof::Int`: Residual degrees of freedom ν = n_inliers - p. When `dof > 0`,
  the noise posterior is `σ² | data ~ InvGamma(dof/2, dof·scale²/2)` and
  residual predictions follow `r/scale ~ t_dof`. Set to 0 for threshold-based
  scoring where σ is either known or estimated externally.
- `trials::Int`: Number of RANSAC trials executed
- `sample_acceptance_rate::Float64`: Fraction of trials that passed `test_sample`
  (non-degenerate). Equal to `effective_trials / trials`. A value of 1.0 means no
  samples were rejected; lower values indicate frequent degeneracy (e.g., collinear
  points for homography).
"""
struct RansacAttributes{T} <: AbstractAttributes
    stop_reason::Symbol
    converged::Bool
    inlier_mask::BitVector
    residuals::Vector{T}
    weights::Vector{T}
    quality::T
    scale::T
    dof::Int
    trials::Int
    sample_acceptance_rate::Float64
    function RansacAttributes(stop_reason::Symbol; inlier_mask::BitVector, residuals::Vector{T}, weights::Vector{T}, quality::T, scale::T, trials::Int, sample_acceptance_rate::Float64=NaN, dof::Int=0) where T
        converged = stop_reason === :converged
        new{T}(stop_reason, converged, inlier_mask, residuals, weights, quality, scale, dof, trials, sample_acceptance_rate)
    end
end

"""
    RansacEstimate{M, T}

Type alias for `Attributed{M, RansacAttributes{T}}` — the return type of `ransac()`.

Access model via `result.value`, attributes via property forwarding:
- `result.converged` — true if a model was found
- `result.inlier_mask` — BitVector of inliers
- `result.quality` — total model quality (higher = better)
- `result.trials` — number of RANSAC trials

See also: [`Attributed`](@ref), [`RansacAttributes`](@ref)
"""
const RansacEstimate{M, T} = Attributed{M, RansacAttributes{T}}

"""
    inlier_ratio(result::RansacEstimate) -> Float64

Return the fraction of data points classified as inliers.
"""
inlier_ratio(r::RansacEstimate) = sum(r.inlier_mask) / length(r.inlier_mask)

# -----------------------------------------------------------------------------
# Workspace (Pre-allocated Buffers)
# -----------------------------------------------------------------------------

"""
    RansacWorkspace{M, T}

Pre-allocated workspace for zero-allocation RANSAC main loop.

Stores all scoring buffers, the best model found so far, and sampling indices.
For `isbitstype` models (e.g., `SMatrix{3,3,Float64,9}`), `best_model` is
stored inline in the mutable struct — no heap allocation on update.

# Constructor
```julia
ws = RansacWorkspace(n_data, sample_size, ModelType)
ws = RansacWorkspace(n_data, sample_size, ModelType, Float64)
```

# Reuse
Pass a workspace to `ransac(; workspace=ws)` to avoid allocation across calls.
"""
mutable struct RansacWorkspace{M, T<:AbstractFloat}
    # Scoring buffers (length = data_size)
    residuals::Vector{T}
    scores::Vector{T}
    penalties::Vector{T}
    mask::BitVector
    # Best-so-far (mutated in-place)
    best_model::M
    best_residuals::Vector{T}
    best_scores::Vector{T}
    best_mask::BitVector
    # Sampling buffer (length = sample_size)
    sample_indices::Vector{Int}
    # State
    has_best::Bool
end

function RansacWorkspace(n::Int, k::Int, ::Type{M}, ::Type{T}=Float64) where {M, T<:AbstractFloat}
    # Uninitialized model placeholder — has_best=false guards against reading
    RansacWorkspace{M,T}(
        Vector{T}(undef, n),
        Vector{T}(undef, n),
        zeros(T, n),
        falses(n),
        Ref{M}()[],
        Vector{T}(undef, n),
        Vector{T}(undef, n),
        falses(n),
        Vector{Int}(undef, k),
        false,
    )
end

# =============================================================================
# PROSAC Sampler (Chum & Matas, CVPR 2005)
# =============================================================================
#
# Progressive Sample Consensus: samples high-quality correspondences first,
# then gradually expands the pool to include all data. Degenerates to standard
# RANSAC after T_N iterations.
#
# Reference: Chum & Matas, "Matching with PROSAC — Progressive Sample Consensus"
#

"""
    ProsacSampler

Mutable state for PROSAC progressive sampling.

Maintains a pool of correspondences sorted by quality score. Sampling starts
from the top-m highest-scored matches and progressively expands the pool.

# Fields
- `sorted_indices::Vector{Int}` — data indices sorted by score (best first)
- `m::Int` — sample size (constant, e.g. 4 for homography)
- `N::Int` — total data size (constant)
- `T_N::Int` — total iteration budget (constant)
- `n::Int` — current pool size (grows from m to N)
- `t::Int` — current iteration count
- `T_n::Float64` — current T_n value (recurrence relation)
- `T_n_prime::Int` — discretized cumulative threshold

See also: [`reset!`](@ref), [`_draw_prosac!`](@ref)
"""
mutable struct ProsacSampler <: AbstractSampler
    sorted_indices::Vector{Int}
    m::Int
    N::Int
    T_N::Int
    n::Int
    t::Int
    T_n::Float64
    T_n_prime::Int
end

"""
    ProsacSampler(sorted_indices, m; T_N=200_000)

Create a PROSAC sampler from pre-sorted indices.

`sorted_indices` must be ordered by decreasing quality score.
`m` is the minimal sample size.
`T_N` is the maximum iteration budget (default: 200,000).
"""
function ProsacSampler(sorted_indices::Vector{Int}, m::Int; T_N::Int=200_000)
    N = length(sorted_indices)
    N >= m || throw(ArgumentError("Need at least $m data points for PROSAC, got $N"))
    # Initial T_n for n=m: T_N * C(m,m)/C(N,m) = T_N / C(N,m)
    T_n = one(Float64)
    for i in 0:(m-1)
        T_n *= (m - i) / (N - i)
    end
    T_n *= T_N
    ProsacSampler(sorted_indices, m, N, T_N, m, 0, T_n, 1)
end

"""
    reset!(ps::ProsacSampler)

Reset PROSAC sampler state for reuse (e.g., across multiple `ransac()` calls).
"""
function reset!(ps::ProsacSampler)
    m, N, T_N = ps.m, ps.N, ps.T_N
    T_n = one(Float64)
    for i in 0:(m-1)
        T_n *= (m - i) / (N - i)
    end
    T_n *= T_N
    ps.n = m
    ps.t = 0
    ps.T_n = T_n
    ps.T_n_prime = 1
    nothing
end

"""
    _draw_prosac!(indices, ps::ProsacSampler)

Draw a PROSAC sample into `indices`.

Implements the Chum & Matas growth function:
1. Increment t, expand pool n while t > T'_n
2. If in PROSAC phase: sample m-1 from U_{n-1}, force index n
3. If in RANSAC phase: sample m from U_n (uniform)
"""
function _draw_prosac!(indices::Vector{Int}, ps::ProsacSampler)
    ps.t += 1
    m = ps.m

    # Expand pool: grow n while t exceeds T'_n threshold
    while ps.n < ps.N && ps.t > ps.T_n_prime
        # Recurrence: T_{n+1} = T_n * (n+1) / (n+1-m)
        ps.T_n *= (ps.n + 1) / (ps.n + 1 - m)
        ps.n += 1
        ps.T_n_prime = floor(Int, ps.T_n)
    end

    n = ps.n

    if ps.t > ps.T_n_prime
        # RANSAC mode: uniform sample of m from top-n
        _draw_uniform!(indices, n)
        # Map to actual data indices
        @inbounds for i in eachindex(indices)
            indices[i] = ps.sorted_indices[indices[i]]
        end
    else
        # PROSAC mode: sample m-1 from U_{n-1}, force u_n
        if m > 1
            _draw_uniform!(indices, n - 1, m - 1)
            @inbounds for i in 1:(m-1)
                indices[i] = ps.sorted_indices[indices[i]]
            end
        end
        @inbounds indices[m] = ps.sorted_indices[n]
    end
    nothing
end

draw!(indices::Vector{Int}, s::ProsacSampler) = (_draw_prosac!(indices, s); nothing)

"""
    _draw_uniform!(indices, n, k=length(indices))

Fill `indices[1:k]` with a uniform random sample without replacement from 1:n.
"""
function _draw_uniform!(indices::Vector{Int}, n::Int, k::Int=length(indices))
    @inbounds for i in 1:k
        while true
            j = rand(1:n)
            duplicate = false
            for q in 1:(i-1)
                indices[q] == j && (duplicate = true; break)
            end
            if !duplicate
                indices[i] = j
                break
            end
        end
    end
    nothing
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
# Holy Trait: Linear System Type
# =============================================================================
#
# Distinguishes null-space problems (Ah = 0, SVD) from overdetermined
# linear systems (Ax ≈ b, weighted LS). Enables trait-dispatched solve
# in the generic IRLS loop.
#

"""
    ConstraintType

Holy Trait for the constraint structure of a RANSAC problem's parameterization.

- `Constrained()`: Model has a gauge constraint h(θ)=0 (e.g., ‖h‖=1).
  Produces a null-space problem `Ah = 0`, solved via SVD.
  Covariance requires the bordered Hessian / implicit function theorem.
- `Unconstrained()`: Model uses minimal (free) coordinates.
  Produces an overdetermined system `Ax ≈ b`, solved via weighted LS.
  Covariance is σ²(X'X)⁻¹.
"""
abstract type ConstraintType end

"""
    Constrained <: ConstraintType

Model parameterization has a gauge constraint h(θ)=0 (e.g., ‖h‖=1 for
homography, fundamental matrix). The IRLS refinement solves the null-space
problem `Ah = 0` via SVD, and the covariance is computed by projecting
I⁻¹ onto the tangent plane of the constraint manifold via the bordered
Hessian from the Lagrangian formulation.
"""
struct Constrained <: ConstraintType end

"""
    Unconstrained <: ConstraintType

Model parameterization uses minimal (free) coordinates with no constraint.
The IRLS refinement solves `Ax ≈ b` via normal equations, and the
covariance is σ²·inv(I) where I is the Fisher information.
"""
struct Unconstrained <: ConstraintType end

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
