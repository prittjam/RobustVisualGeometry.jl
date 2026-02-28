# =============================================================================
# Scoring — Scoring functions for scale-free marginal RANSAC
# =============================================================================
#
# Defines how RANSAC candidate models are evaluated.
#
# Contents:
#   1. AbstractScoring
#   2. Uncertain{M} type dispatch for model-uncertain scoring
#   3. Local optimization strategies (NoLocalOptimization)
#   4. MarginalScoring{P} (P=Nothing for model-certain, P=Predictive for model-uncertain)
#   5. init_score, default_local_optimization, _score_improved
#
# DEPENDENCY: Requires AbstractRansacProblem from ransac_interface.jl
#
# =============================================================================

# -----------------------------------------------------------------------------
# Scoring Strategies
# -----------------------------------------------------------------------------

"""
    AbstractScoring

Abstract type for RANSAC model scoring functions.

Scoring functions determine how candidate models are scored. The scale-free
marginal score (Eq. 12) eliminates σ by marginalizing under the Jeffreys prior
π(σ²) ∝ 1/σ², then optimizes over partitions I. Higher = better.

Built-in strategies:
- `MarginalScoring`: Model-certain score S(θ,I) (Section 3, Eq. 12)
- `PredictiveMarginalScoring`: Predictive score S_pred(θ̂,I) incorporating
  per-point leverages (Section 4, Eq. 20)

See also: [`MarginalScoring`](@ref), [`PredictiveMarginalScoring`](@ref)
"""
abstract type AbstractScoring end

# -----------------------------------------------------------------------------
# Local Refinement Strategies
# -----------------------------------------------------------------------------

"""
    AbstractLocalOptimization

Abstract type for local optimization strategies passed to `ransac()`.

Concrete subtypes control how the inlier mask is refined after initial scoring:
- `NoLocalOptimization()`: No local optimization

Passed as a keyword argument to `ransac()`, separate from the scoring strategy.
Use `default_local_optimization(scoring)` to get the default for a given scoring strategy.
"""
abstract type AbstractLocalOptimization end

"""
    NoLocalOptimization <: AbstractLocalOptimization

No local optimization. The candidate model is scored directly without any
refit. Default local optimization for all scoring functions
(via `default_local_optimization`).
"""
struct NoLocalOptimization <: AbstractLocalOptimization end

"""
    PosteriorIrls <: AbstractLocalOptimization

Strategy C: Posterior-weight IRLS refinement. Computes posterior
inlier probabilities from the marginal score partition, uses them as
soft weights for WLS refit, re-scores, and iterates.

# Fields
- `max_outer_iter::Int=5`: Max posterior-IRLS outer iterations
"""
Base.@kwdef struct PosteriorIrls <: AbstractLocalOptimization
    max_outer_iter::Int = 5
end


# =============================================================================
# Posterior Weights — Soft inlier/outlier probabilities
# =============================================================================

"""
    _posterior_weights!(w, scores, perm, k, dg, m, log2a)

Posterior inlier probabilities (Eq. w-post-refit).

Given the sweep partition (perm[1:k] = inliers), estimates the scale
ŝ² = RSS_I / ν from inlier RSS with ν = (k−m)·dg effective DOF,
then for each point computes the posterior inlier probability under
the Gaussian-vs-Uniform mixture:

    wᵢ = p_in(rᵢ; ŝ) / (p_in(rᵢ; ŝ) + p_out)

where p_in(rᵢ; ŝ) = (2π ŝ²)^{-dg/2} exp(−‖rᵢ‖²/(2ŝ²)) and
p_out = (2a)^{-dg}.
"""
function _posterior_weights!(w::AbstractVector{T}, scores::AbstractVector{T},
                              perm::Vector{Int}, k::Int,
                              dg::Int, m::Int, log2a::Float64) where T
    # ŝ² = RSS_I / ν, where ν = (k−m)·dg  (Eq. 12 DOF)
    RSS = zero(T)
    @inbounds for j in 1:k
        RSS += scores[perm[j]]
    end
    ν = (k - m) * dg
    ŝ² = RSS / T(ν)
    ŝ² = max(ŝ², eps(T))

    log_pin_norm = -T(dg) / 2 * log(2 * T(π) * ŝ²)
    log_pout = -T(dg) * T(log2a)

    @inbounds for i in eachindex(w)
        log_pin = log_pin_norm - scores[i] / (2 * ŝ²)
        w[i] = one(T) / (one(T) + exp(log_pout - log_pin))
    end
    nothing
end

# -----------------------------------------------------------------------------
# Uncertain{M} Dispatch (Section 3.3: Specializations)
# -----------------------------------------------------------------------------
#
# Model uncertainty dispatch (type dispatch on model argument):
#   model::M             → model-certain (Σ_θ = 0, treat θ as exact)
#   model::Uncertain{M}  → model-uncertain (Σ_θ = model.param_cov)
#
# The model carries its own covariance via VGC's Uncertain{V,T,N} wrapper.
# _with_model_cov() wraps after minimal solve, _with_fit_cov() wraps after LO.
# MarginalScoring never wraps → zero overhead.
#
# All residuals!() implementations return whitened residuals such that
# ‖rᵢ‖² = Mahalanobis distance. The two score! methods:
#
#   model::M (certain)     → scores[i] = rᵢ²
#   model::Uncertain{M}   → scores[i] = rᵢᵀ Ṽᵢ⁻¹ rᵢ  (predictive variance)
#
# ℓᵢ = 0 for both: the outlier density is defined in the whitened constraint
# space rᵢ = Lᵢ⁻¹gᵢ (where LᵢLᵢᵀ = Σ̃_{gᵢ}), giving
# p_out(gᵢ) = (2a)^{-dg} |Σ̃_{gᵢ}|^{-1/2}. The Jacobian
# |Σ̃_{gᵢ}|^{-1/2} cancels with the identical factor from the inlier
# Gaussian normalization, so ℓᵢ = 0.
# -----------------------------------------------------------------------------
#
# (Removed: MeasurementCovariance trait — all residuals! return whitened r, so
#  the Isotropic/Heteroscedastic axis was redundant. See git log for history.)

# -----------------------------------------------------------------------------
# Concrete Scoring Strategies
# -----------------------------------------------------------------------------

"""
    Predictive

Type tag for the predictive variant of `MarginalScoring`.

`MarginalScoring{Predictive}` incorporates model estimation uncertainty
(Σ̃_θ from the minimal solver or LO fit) into per-point scores.
`MarginalScoring{Nothing}` treats the model as exact (Σ_θ = 0).

See also: [`MarginalScoring`](@ref), [`PredictiveMarginalScoring`](@ref)
"""
struct Predictive end

"""
    MarginalScoring{P} <: AbstractScoring

Scale-free marginal score (Section 3, Eq. 12).

Parameterized by `P`:
- `MarginalScoring{Nothing}` — Model-certain: treats θ as exact (Σ_θ = 0).
- `MarginalScoring{Predictive}` — Model-uncertain: incorporates estimation
  uncertainty Σ̃_θ via per-point leverages (Section 4, Eq. 20).

The score marginalizes σ² under the Jeffreys prior π(σ²) ∝ 1/σ²:

    S(θ, I) = log Γ((nᵢ·dg − n_θ)/2) − ((nᵢ·dg − n_θ)/2) log(π·RSS_I) − nₒ·dg·log(2a)

where nᵢ = |I|, RSS_I = Σᵢ∈I rᵢ², nₒ = N − nᵢ, n_θ = m·dg (model parameters),
a is the outlier domain half-width, and dg the codimension.

Per-point scores ‖rᵢ‖² = gᵢᵀ Σ̃_{gᵢ}⁻¹ gᵢ, where rᵢ = Lᵢ⁻¹gᵢ is the
shape-whitened residual. For the predictive variant, Σ̃_{gᵢ} includes
model estimation uncertainty.

# Fields
- `log2a::Float64`: log(2a), precomputed outlier penalty per point
- `model_dof::Int`: Minimal sample size m = ⌈n_θ/dg⌉
- `codimension::Int`: Codimension dg of the model manifold
- `perm::Vector{Int}`: Pre-allocated sortperm buffer (mutated by `sweep!`)
- `lg_table::Vector{Float64}`: Precomputed log Γ(k·dg/2) for k = 1..N; indexed as lg_table[k−m] in sweep!
- `scratch::Vector{Int}`: Pre-allocated scratch buffer for `sortperm!` (eliminates per-call allocation)
"""
mutable struct MarginalScoring{P} <: AbstractScoring
    log2a::Float64
    model_dof::Int
    codimension::Int
    perm::Vector{Int}
    lg_table::Vector{Float64}
    scratch::Vector{Int}       # pre-allocated scratch for sortperm!
end

"""
    PredictiveMarginalScoring

Alias for `MarginalScoring{Predictive}`. Model-uncertain scoring that
incorporates estimation uncertainty Σ̃_θ (Section 4, Eq. 20).
"""
const PredictiveMarginalScoring = MarginalScoring{Predictive}

"""
    _build_lg_table(n, d_g=1)

Precompute lg[k] = log Γ(k·dg/2) for k = 1..N, used by Algorithm 1.

For dg=1: log Γ(k/2) via step-2 recurrence.
For dg=2: log Γ(k) via step-1 recurrence.
"""
function _build_lg_table(n::Int, d_g::Int=1)
    lg = Vector{Float64}(undef, n)
    if d_g == 1
        # loggamma(k/2) via step-2 recurrence: Γ(x+1) = xΓ(x)
        if n >= 1
            lg[1] = 0.5 * log(π)  # loggamma(1/2) = log(√π)
        end
        if n >= 2
            lg[2] = 0.0  # loggamma(1) = 0
        end
        @inbounds for k in 3:n
            lg[k] = log((k - 2) / 2) + lg[k - 2]
        end
    elseif d_g == 2
        # loggamma(k) via step-1 recurrence: Γ(k) = (k-1)·Γ(k-1)
        if n >= 1
            lg[1] = 0.0  # loggamma(1) = 0
        end
        @inbounds for k in 2:n
            lg[k] = log(k - 1) + lg[k - 1]
        end
    else
        error("_build_lg_table: unsupported codimension d_g=$d_g (only 1 and 2 supported)")
    end
    return lg
end

function MarginalScoring{P}(n::Int, p::Int, a::Float64; codimension::Int=1) where P
    a > 0 || throw(ArgumentError("outlier_halfwidth must be positive, got $a"))
    lg = _build_lg_table(n, codimension)
    MarginalScoring{P}(log(2a), p, codimension, Vector{Int}(undef, n), lg, Vector{Int}(undef, n))
end

# Bare MarginalScoring(n, p, a) defaults to model-certain
MarginalScoring(n::Int, p::Int, a::Float64; codimension::Int=1) =
    MarginalScoring{Nothing}(n, p, a; codimension=codimension)

"""
    MarginalScoring(problem::AbstractRansacProblem, a::Float64)
    MarginalScoring{P}(problem::AbstractRansacProblem, a::Float64)

Problem-aware constructor. Derives N, m, and dg from the problem.
"""
MarginalScoring(problem::AbstractRansacProblem, a::Float64) =
    MarginalScoring{Nothing}(data_size(problem), sample_size(problem), a;
                              codimension=codimension(problem))

function MarginalScoring{P}(problem::AbstractRansacProblem, a::Float64) where P
    MarginalScoring{P}(data_size(problem), sample_size(problem), a;
                        codimension=codimension(problem))
end

"""
    init_score(scoring::AbstractScoring)

Return the initial "best" score value for the scoring strategy.
`MarginalScoring`: `(-Inf, -Inf)` (global, local score tuple to maximize)
"""
init_score(::MarginalScoring) = (-Inf, -Inf)

"""
    default_local_optimization(scoring::AbstractScoring) -> AbstractLocalOptimization

Return the default local optimization strategy. Returns `NoLocalOptimization()` for
all scoring functions.

# Example
```julia
scoring = MarginalScoring(100, 2, 50.0)
default_local_optimization(scoring)  # NoLocalOptimization()
```
"""
default_local_optimization(::AbstractScoring) = NoLocalOptimization()

# -----------------------------------------------------------------------------
# Score Improvement Check (unified: higher = better)
# -----------------------------------------------------------------------------

"""
    _score_improved(scoring, new, old) -> Bool

Check whether `new` score is strictly better than `old`.
All strategies use higher = better convention.
"""
_score_improved(::MarginalScoring, new, old) = new[2] > old[2]

# =============================================================================
# Sort-and-Sweep (Algorithm 1, Section 5.1)
# =============================================================================

"""
    sweep!(perm, lg_table, model_dof, log2a, dg, scores, n) -> (S*, k*)

Sort-and-sweep computation of the scale-free marginal score (Algorithm 1, Eq. 12).

Sorts per-point squared whitened residuals rᵢ², then sweeps prefixes
k = m+1,...,N evaluating (Eq. 12):

    S = log Γ(ν/2) − (ν/2) log(π·RSS_I) − (N−k)·dg·log(2a)

where ν = (k−m)·dg is the effective DOF (k·dg constraints minus n_θ = m·dg
model parameters), RSS_I = Σ_{j=1}^k r²_{(j)}, and lg_table[k−m] = log Γ((k−m)·dg/2).

Returns the best score S* and optimal inlier count k*. O(N log N) time.

The three terms of S (Section 3.2):
  1. Inlier reward:   log Γ(ν/2)
  2. Fit quality:     −(ν/2) log(π·RSS_I)
  3. Outlier penalty: −(N−k)·dg·log(2a)

Mutates `perm` via `sortperm!`.
"""
function sweep!(perm::Vector{Int}, lg_table::Vector{Float64},
                          model_dof::Int, log2a::Float64, dg::Int,
                          scores::Vector{T},
                          n::Int;
                          scratch::Union{Nothing,Vector{Int}}=nothing) where T
    if isnothing(scratch)
        sortperm!(perm, scores)
    else
        sortperm!(perm, scores; scratch)
    end
    m = model_dof
    dg_T = T(dg)
    RSS = zero(T)
    best_S = typemin(T)
    best_k = 0
    @inbounds for k in 1:n
        idx = perm[k]
        RSS += scores[idx]
        k <= m && continue                          # need k > m for positive DOF
        RSS = max(RSS, eps(T))
        ν = dg_T * (k - m)                           # effective DOF = k·dg − n_θ
        ν_half = ν / 2
        S = lg_table[k - m] - ν_half * log(T(π) * RSS) - (n - k) * dg_T * T(log2a)
        if S > best_S
            best_S = S
            best_k = k
        end
    end
    return best_S, best_k
end

# Dispatch wrapper for all MarginalScoring{P} variants
sweep!(s::MarginalScoring, scores::Vector{T}, n::Int) where T =
    sweep!(s.perm, s.lg_table, s.model_dof, s.log2a, s.codimension, scores, n;
           scratch=s.scratch)

# =============================================================================
# Model Covariance Wrapping — Uncertain{M} construction
# =============================================================================

"""
    _with_model_cov(scoring, problem, model, sample_indices)

Wrap `model` with estimation covariance Σ̃_θ = JJᵀ from the minimal solver.

- `MarginalScoring{Nothing}`: returns plain model (model-certain, no wrapping)
- `MarginalScoring{Predictive}`: returns `Uncertain(model, JJᵀ)` when
  `solver_jacobian` is available, otherwise plain model (fallback)
"""
_with_model_cov(::MarginalScoring{Nothing}, _problem, model, _idx) = model

function _with_model_cov(::MarginalScoring{Predictive}, problem, model, idx)
    jac_info = solver_jacobian(problem, idx, model)
    isnothing(jac_info) && return model
    Uncertain(model, jac_info.J * jac_info.J')
end

"""
    _with_fit_cov(scoring, problem, model)

Wrap `model` with estimation covariance from the least-squares fit.

- `MarginalScoring{Nothing}`: returns plain model (no wrapping)
- `MarginalScoring{Predictive}`: returns `Uncertain(model, Σ̃)` when
  `fit_param_covariance(problem)` is available, otherwise plain model
"""
_with_fit_cov(::MarginalScoring{Nothing}, _problem, model) = model

function _with_fit_cov(::MarginalScoring{Predictive}, problem, model)
    cov = fit_param_covariance(problem)
    isnothing(cov) && return model
    Uncertain(model, cov)
end

"""
    _plain_model(model)

Extract the plain model from an `Uncertain` wrapper, or return as-is.
"""
_plain_model(m::Uncertain) = m.value
_plain_model(m) = m

# =============================================================================
# _predictive_score_penalty — Per-point (‖rᵢ‖², ℓᵢ) with model covariance
# =============================================================================

"""
    _predictive_score_penalty(rᵢ::T, ∂θgᵢ_w::SVector{n_θ,T}, Σ̃_θ, s2) -> (rᵢ², ℓᵢ)

Model-uncertain score and penalty for scalar constraint (dg=1, Appendix D).

In whitened coordinates (rᵢ = gᵢ/√cᵢ, ∂θgᵢ_w = ∂θgᵢ/√cᵢ):
  Ṽᵢ = s² + (∂θgᵢ_w)ᵀ Σ̃_θ (∂θgᵢ_w)    (scalar prediction variance shape)
  rᵢ² / Ṽᵢ                                (predictive squared residual)
  ℓᵢ = log(Ṽᵢ)                            (model uncertainty contribution)
"""
@inline function _predictive_score_penalty(rᵢ::T, ∂θgᵢ_w::SVector{n_θ,T},
                                            Σ̃_θ, s2) where {n_θ,T}
    Ṽᵢ = s2 + dot(∂θgᵢ_w, Σ̃_θ * ∂θgᵢ_w)
    Ṽᵢ > eps(T) || return (typemax(T), zero(T))
    r²ᵢ = rᵢ^2 / Ṽᵢ
    return (r²ᵢ, log(Ṽᵢ))
end

"""
    _predictive_score_penalty(rᵢ::SVector{dg,T}, ∂θgᵢ_w::SMatrix{dg,n_θ,T}, Σ̃_θ, s2)
        -> (‖rᵢ‖², ℓᵢ)

Model-uncertain Mahalanobis score and penalty for vector constraint
(dg≥2, Eq. 7, Appendix D).

In whitened coordinates (rᵢ = Lᵢ⁻¹gᵢ, ∂θgᵢ_w = Lᵢ⁻¹∂θgᵢ):
  Ṽᵢ = s²I + (∂θgᵢ_w) Σ̃_θ (∂θgᵢ_w)ᵀ    (dg×dg prediction covariance shape)
  rᵢᵀ Ṽᵢ⁻¹ rᵢ                             (predictive Mahalanobis distance)
  ℓᵢ = log|Ṽᵢ|                             (model uncertainty contribution)
"""
@inline function _predictive_score_penalty(rᵢ::SVector{dg,T}, ∂θgᵢ_w::SMatrix{dg,n_θ,T},
                                            Σ̃_θ, s2) where {dg,n_θ,T}
    Ṽᵢ = s2 * SMatrix{dg,dg,T}(I) + ∂θgᵢ_w * Σ̃_θ * ∂θgᵢ_w'
    det_Ṽᵢ = det(Ṽᵢ)
    det_Ṽᵢ > eps(T) || return (typemax(T), zero(T))
    r²ᵢ = dot(rᵢ, Ṽᵢ \ rᵢ)
    return (r²ᵢ, log(det_Ṽᵢ))
end

# =============================================================================
# mask! — Build inlier set from sweep result
# =============================================================================

"""
    mask!(ws, perm, k_star)

Build the inlier set I* = {(1),...,(k*)} from the sorted order (Section 3.5).
"""
function mask!(ws::RansacWorkspace, perm::Vector{Int}, k_star::Int)
    n = length(ws.mask)
    @inbounds for i in 1:n
        ws.mask[i] = false
    end
    @inbounds for j in 1:k_star
        ws.mask[perm[j]] = true
    end
    nothing
end

# =============================================================================
# score! — Type-dispatched per-point scoring (Section 3.3)
# =============================================================================
#
# All residuals!() implementations return whitened residuals rᵢ such that
# ‖rᵢ‖² = Mahalanobis distance. Two methods dispatch on model type:
#
#   model::M (certain)     → scores[i] = rᵢ²
#   model::Uncertain{M}   → scores[i] = rᵢᵀ Ṽᵢ⁻¹ rᵢ  (predictive)
#
# ℓᵢ = 0 for both: the outlier density is defined in whitened space,
# so covariance factors cancel.
# =============================================================================

"""
    _sq_norm(r) -> T

Squared norm: `rᵢ²` for scalar, `rᵢᵀrᵢ` for vector. Computes ‖rᵢ‖²
from the shape-whitened residual (Table 2).
"""
@inline _sq_norm(r::Real) = r * r
@inline _sq_norm(r::SVector) = dot(r, r)

# --- Model-certain: scores[i] = rᵢ², ℓᵢ = 0 ---
# Fast path: whitened residuals from residuals!, no Jacobians needed.
function score!(ws, problem, ::MarginalScoring, model)
    n = data_size(problem)
    residuals!(ws.residuals, problem, model)
    @inbounds for i in 1:n
        ws.scores[i] = ws.residuals[i]^2
    end
end

# --- Model-uncertain: scores[i] = rᵢᵀ Ṽᵢ⁻¹ rᵢ, ℓᵢ = 0 ---
# Model uncertainty inflates prediction variance via Σ̃_θ from
# the Uncertain{M} wrapper (not from ws.sample_indices).
function score!(ws::RansacWorkspace{<:Any,M,T}, problem,
                scoring::MarginalScoring,
                model::Uncertain{M}) where {M,T}
    n = data_size(problem)
    θ = model.value
    Σ̃_θ = model.param_cov
    @inbounds for i in 1:n
        rᵢ, ∂θgᵢ_w, _ = residual_jacobian(problem, θ, i)
        ws.residuals[i] = sqrt(_sq_norm(rᵢ))
        r²ᵢ, _ = _predictive_score_penalty(rᵢ, ∂θgᵢ_w, Σ̃_θ, one(T))
        ws.scores[i] = r²ᵢ
    end
end

