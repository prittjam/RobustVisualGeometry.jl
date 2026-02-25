# =============================================================================
# RANSAC Types — Configuration, workspace, result types, FixedModels
# =============================================================================

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
