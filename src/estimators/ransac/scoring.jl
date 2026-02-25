# =============================================================================
# Scoring — Quality functions for scale-free marginal RANSAC
# =============================================================================
#
# Defines how RANSAC candidate models are evaluated.
#
# Contents:
#   1. AbstractQualityFunction
#   2. CovarianceStructure trait (Homoscedastic, Heteroscedastic, Predictive)
#   3. Local optimization strategies (NoLocalOptimization)
#   4. Concrete quality types (MarginalQuality, PredictiveMarginalQuality)
#   5. init_quality, default_local_optimization, _quality_improved
#
# DEPENDENCY: Requires AbstractRansacProblem from ransac_interface.jl
#
# =============================================================================

# -----------------------------------------------------------------------------
# Quality Functions (Scoring Strategies)
# -----------------------------------------------------------------------------

"""
    AbstractQualityFunction

Abstract type for RANSAC model quality functions.

Quality functions determine how candidate models are scored. The scale-free
marginal score (Eq. 12) eliminates σ by marginalizing under the Jeffreys prior
π(σ²) ∝ 1/σ², then optimizes over partitions I. Higher = better.

Built-in strategies:
- `MarginalQuality`: Model-certain score S(θ,I) (Section 3, Eq. 12)
- `PredictiveMarginalQuality`: Predictive score S_pred(θ̂,I) incorporating
  per-point leverages (Section 4, Eq. 20)

See also: [`MarginalQuality`](@ref), [`PredictiveMarginalQuality`](@ref)
"""
abstract type AbstractQualityFunction end

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
refit. Default local optimization for all quality functions
(via `default_local_optimization`).
"""
struct NoLocalOptimization <: AbstractLocalOptimization end

"""
    ConvergeThenRescore <: AbstractLocalOptimization

Strategy A (Algorithm 3): Run WLS to convergence at fixed inlier mask,
then re-sweep to find the new optimal partition. Repeat until score
does not improve.

# Fields
- `max_fit_iter::Int=5`: Max WLS iterations per outer loop
- `max_outer_iter::Int=3`: Max refit-resweep outer iterations
"""
Base.@kwdef struct ConvergeThenRescore <: AbstractLocalOptimization
    max_fit_iter::Int = 5
    max_outer_iter::Int = 3
end

"""
    StepAndRescore <: AbstractLocalOptimization

Strategy B (Algorithm 4): Single WLS step, then re-sweep. Repeat until
score does not improve. Keeps partition maximally current.

# Fields
- `max_outer_iter::Int=5`: Max step-resweep outer iterations
"""
Base.@kwdef struct StepAndRescore <: AbstractLocalOptimization
    max_outer_iter::Int = 5
end

"""
    fit_strategy(lo::AbstractLocalOptimization) -> FitStrategy

Return the fit strategy trait for the given local optimization type.
Default: `LinearFit()`.
"""
fit_strategy(::AbstractLocalOptimization) = LinearFit()

# -----------------------------------------------------------------------------
# Covariance Structure Trait (Section 3.3: Specializations)
# -----------------------------------------------------------------------------

"""
    CovarianceStructure

Holy Trait for the constraint covariance shape Σ̃_{gᵢ} in the scale-free
marginal score (Section 3.3, Eq. 12).

The three specializations (Section 3.3, listed most to least general)
differ only in the covariance penalty ℓᵢ = log|Σ̃_{gᵢ}| that enters
Algorithm 1:

- `Predictive()`:      Σ̃_{gᵢ} = ∂ₓgᵢ Σ̃_{xᵢ} (∂ₓgᵢ)ᵀ + ∂θgᵢ Σ̃_θ (∂θgᵢ)ᵀ  (Eq. 7)
- `Heteroscedastic()`: Σ̃_{gᵢ} = ∂ₓgᵢ Σ̃_{xᵢ} (∂ₓgᵢ)ᵀ                        (Σ_θ = 0)
- `Homoscedastic()`:   Σ̃_{gᵢ} = σ² ‖∂ₓgᵢ‖²  (scalar, dg=1, Σ̃_{xᵢ} = I)     (Eq. 21)

For the homoscedastic case, ℓᵢ cancels in the sweep (constant factor), so
ℓᵢ = 0 in Algorithm 1.
"""
abstract type CovarianceStructure end

"""
    Homoscedastic <: CovarianceStructure

Section 3.3, specialization "Isotropic, dg=1": Σ̃_{xᵢ} = I and Σ_θ = 0.
The constraint covariance shape reduces to the scalar cᵢ = ‖∂ₓgᵢ‖² (Eq. 21)
which is constant across points for many problems. The covariance penalty
ℓᵢ = 0 because the constant factor cancels between RSS_I and L in Eq. 12.
"""
struct Homoscedastic <: CovarianceStructure end

"""
    Heteroscedastic <: CovarianceStructure

Section 3.3, specialization "Model certain": Σ_θ = 0, only measurement noise
contributes to Σ̃_{gᵢ} = ∂ₓgᵢ Σ̃_{xᵢ} (∂ₓgᵢ)ᵀ. The covariance penalty
ℓᵢ = log|Σ̃_{gᵢ}| varies per point and must be included in Algorithm 1.
"""
struct Heteroscedastic <: CovarianceStructure end

"""
    Predictive <: CovarianceStructure

Section 3.3, specialization "Model uncertain": Σ_θ ≠ 0, both measurement noise
and parameter estimation error contribute to Σ̃_{gᵢ} (Eq. 7). The covariance
penalty ℓᵢ = log|Σ̃_{gᵢ}| includes both the measurement term log|∂ₓgᵢ Σ̃_{xᵢ} (∂ₓgᵢ)ᵀ|
and the model uncertainty term log|I + (L⁻¹∂θgᵢ) Σ̃_θ (L⁻¹∂θgᵢ)ᵀ|.
"""
struct Predictive <: CovarianceStructure end

# -----------------------------------------------------------------------------
# Concrete Scoring Strategies
# -----------------------------------------------------------------------------

"""
    AbstractMarginalQuality <: AbstractQualityFunction

Abstract supertype for scale-free marginal likelihood scoring (Section 3).

All variants share `sweep!` (Algorithm 1) for finding the optimal partition
I* = {(1),...,(k*)} and the two-level gating loop of Algorithm 2.
"""
abstract type AbstractMarginalQuality <: AbstractQualityFunction end

"""
    MarginalQuality <: AbstractMarginalQuality

Model-certain scale-free marginal score (Section 3, Eq. 12).

Treats the model θ as exact (Σ_θ = 0, Section 3.3 "model certain"):
the only randomness is measurement noise. The score marginalizes σ²
under the Jeffreys prior π(σ²) ∝ 1/σ² (Section 3.1):

    S(θ, I) = log Γ(nᵢ dg/2) − (nᵢ dg/2) log(π RSS_I)
              − ½ Σᵢ∈I log|Σ̃_{gᵢ}| − nₒ dg log(2a)              (12)

where nᵢ = |I|, RSS_I = Σᵢ∈I qᵢ (weighted residual sum of squares),
nₒ = N − nᵢ, a is the outlier domain half-width, and dg the codimension.

The four terms of the score (Section 3.2):
  1. "Inlier reward":      log Γ(nᵢ dg/2)
  2. "Fit quality":        −(nᵢ dg/2) log(π RSS_I)
  3. "Covariance penalty": −½ Σᵢ∈I log|Σ̃_{gᵢ}|
  4. "Outlier penalty":    −nₒ dg log(2a)

Per-point scores: qᵢ = gᵢᵀ Σ̃_{gᵢ}⁻¹ gᵢ.
Per-point penalties: ℓᵢ = log|Σ̃_{gᵢ}|.

# Fields
- `log2a::Float64`: log(2a), precomputed outlier penalty per point
- `model_dof::Int`: Minimal sample size m = ⌈n_θ/dg⌉
- `codimension::Int`: Codimension dg of the model manifold
- `perm::Vector{Int}`: Pre-allocated sortperm buffer (mutated by `sweep!`)
- `lg_table::Vector{Float64}`: Precomputed log Γ(k·dg/2) for k = 1..N
"""
mutable struct MarginalQuality <: AbstractMarginalQuality
    log2a::Float64
    model_dof::Int
    codimension::Int
    perm::Vector{Int}
    lg_table::Vector{Float64}
end

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

function MarginalQuality(n::Int, p::Int, a::Float64; codimension::Int=1)
    a > 0 || throw(ArgumentError("outlier_halfwidth must be positive, got $a"))
    lg = _build_lg_table(n, codimension)
    MarginalQuality(log(2a), p, codimension, Vector{Int}(undef, n), lg)
end

"""
    MarginalQuality(problem::AbstractRansacProblem, a::Float64)

Problem-aware constructor. Derives N, m, and dg from the problem.
"""
function MarginalQuality(problem::AbstractRansacProblem, a::Float64)
    MarginalQuality(data_size(problem), sample_size(problem), a;
                     codimension=codimension(problem))
end

"""
    PredictiveMarginalQuality <: AbstractMarginalQuality

Model-uncertain scale-free marginal score (Section 3.3 "model uncertain",
Section 4, Appendix D).

Because θ̂ is estimated from a minimal sample, each constraint gᵢ carries
additional variance from the estimation error θ̂ − θ. The full constraint
covariance shape (Eq. 7, 28) is:

    Σ̃_{gᵢ} = ∂ₓgᵢ Σ̃_{xᵢ} (∂ₓgᵢ)ᵀ + ∂θgᵢ Σ̃_θ (∂θgᵢ)ᵀ

where Σ̃_θ = ((∂θg_min)ᵀ W (∂θg_min))⁻¹ is the estimation covariance
shape from the minimal sample (Eq. 27, Appendix D).

The score (Eq. 12) uses qᵢ = gᵢᵀ Σ̃_{gᵢ}⁻¹ gᵢ and ℓᵢ = log|Σ̃_{gᵢ}|.

Falls back to `MarginalQuality` (Σ_θ = 0) when `solver_jacobian` is
not available for the problem.

# Fields
Same as [`MarginalQuality`](@ref).
"""
mutable struct PredictiveMarginalQuality <: AbstractMarginalQuality
    log2a::Float64
    model_dof::Int
    codimension::Int
    perm::Vector{Int}
    lg_table::Vector{Float64}
end

function PredictiveMarginalQuality(n::Int, p::Int, a::Float64; codimension::Int=1)
    a > 0 || throw(ArgumentError("outlier_halfwidth must be positive, got $a"))
    lg = _build_lg_table(n, codimension)
    PredictiveMarginalQuality(log(2a), p, codimension, Vector{Int}(undef, n), lg)
end

"""
    PredictiveMarginalQuality(problem::AbstractRansacProblem, a::Float64)

Problem-aware constructor. Derives N, m, and dg from the problem.
"""
function PredictiveMarginalQuality(problem::AbstractRansacProblem, a::Float64)
    PredictiveMarginalQuality(data_size(problem), sample_size(problem), a;
                               codimension=codimension(problem))
end

"""
    init_quality(scoring::AbstractQualityFunction)

Return the initial "best" quality value for the scoring strategy.
`AbstractMarginalQuality`: `(-Inf, -Inf)` (global, local score tuple to maximize)
"""
init_quality(::AbstractMarginalQuality) = (-Inf, -Inf)

"""
    default_local_optimization(scoring::AbstractQualityFunction) -> AbstractLocalOptimization

Return the default local optimization strategy. Returns `NoLocalOptimization()` for
all quality functions.

# Example
```julia
scoring = MarginalQuality(100, 2, 50.0)
default_local_optimization(scoring)  # NoLocalOptimization()
```
"""
default_local_optimization(::AbstractQualityFunction) = NoLocalOptimization()

"""
    covariance_structure(problem, scoring) -> CovarianceStructure

Determine the effective covariance structure for the constraint covariance
shape Σ̃_{gᵢ} (Section 3.3).

- `MarginalQuality`:            delegates to `measurement_covariance(problem)`
                                 (model certain: Σ_θ = 0)
- `PredictiveMarginalQuality`:  always `Predictive()` (model uncertain: Σ_θ ≠ 0)
"""
covariance_structure(problem, ::MarginalQuality) = measurement_covariance(problem)
covariance_structure(problem, ::PredictiveMarginalQuality) = Predictive()

# -----------------------------------------------------------------------------
# Quality Improvement Check (unified: higher = better)
# -----------------------------------------------------------------------------

"""
    _quality_improved(scoring, new, old) -> Bool

Check whether `new` quality is strictly better than `old`.
All strategies use higher = better convention.
"""
_quality_improved(::AbstractMarginalQuality, new, old) = new[2] > old[2]

# =============================================================================
# Sort-and-Sweep (Algorithm 1, Section 5.1)
# =============================================================================

"""
    sweep!(perm, lg_table, model_dof, log2a, dg, scores, penalties, n) -> (S*, k*)

Sort-and-sweep computation of the scale-free marginal score (Algorithm 1, Eq. 12).

Sorts per-point weighted squared residuals qᵢ = gᵢᵀ Σ̃_{gᵢ}⁻¹ gᵢ, then sweeps
prefixes k = m+1,...,N evaluating (Eq. 12):

    S = log Γ(k·dg/2) − (k·dg/2) log(π·RSS_I) − ½L − (N−k)·dg·log(2a)

where RSS_I = Σ_{j=1}^k q_{(j)} (weighted residual sum of squares),
L = Σ_{j=1}^k ℓ_{(j)} (accumulated covariance penalty, ℓᵢ = log|Σ̃_{gᵢ}|),
and lg_table[k] = log Γ(k·dg/2) (precomputed inlier reward).

Returns the best score S* and optimal inlier count k*. O(N log N) time.

The four terms of S (Section 3.2):
  1. Inlier reward:      log Γ(k·dg/2)
  2. Fit quality:        −(k·dg/2) log(π·RSS_I)
  3. Covariance penalty: −½L
  4. Outlier penalty:    −(N−k)·dg·log(2a)

Mutates `perm` via `sortperm!`.
"""
function sweep!(perm::Vector{Int}, lg_table::Vector{Float64},
                          model_dof::Int, log2a::Float64, d_g::Int,
                          scores::Vector{T}, penalties::Vector{T},
                          n::Int) where T
    sortperm!(perm, scores)
    p = model_dof
    d2 = T(d_g) / 2
    d_g_T = T(d_g)
    log_pi = T(log(π))
    RSS = zero(T)
    L = zero(T)
    best_S = typemin(T)
    best_k = 0
    @inbounds for k in 1:n
        idx = perm[k]
        RSS += scores[idx]
        L += penalties[idx]
        k <= p && continue
        RSS = max(RSS, eps(T))
        S = lg_table[k] - d2 * k * (log_pi + log(RSS)) - T(0.5) * L - (n - k) * d_g_T * log2a
        if S > best_S
            best_S = S
            best_k = k
        end
    end
    return best_S, best_k
end

# Dispatch wrapper for all AbstractMarginalQuality subtypes
sweep!(s::AbstractMarginalQuality, scores::Vector{T}, penalties::Vector{T}, n::Int) where T =
    sweep!(s.perm, s.lg_table, s.model_dof, s.log2a, s.codimension, scores, penalties, n)

# =============================================================================
# model_covariance — Dispatch point for prediction variance (Section 4)
# =============================================================================

"""
    model_covariance(scoring, problem, model, sample_indices) -> Σ̃_θ or nothing

Compute the estimation covariance shape Σ̃_θ (Eq. 27, Appendix D).

- `MarginalQuality`: returns `nothing` (model certain: Σ_θ = 0)
- `PredictiveMarginalQuality`: returns `Σ̃_θ = J Jᵀ` where J is the solver
  Jacobian ∂θ/∂x_min. The product J·Jᵀ gives the estimation covariance shape
  (Eq. 27): Σ_θ = σ² Σ̃_θ.

Returns `nothing` if solver_jacobian is unavailable.
"""
model_covariance(::MarginalQuality, problem, model, sample_indices) = nothing

function model_covariance(::PredictiveMarginalQuality, problem, model, sample_indices)
    jac_info = solver_jacobian(problem, sample_indices, model)
    isnothing(jac_info) && return nothing
    return jac_info.J * jac_info.J'
end

# =============================================================================
# _predictive_score_penalty — Per-point (q_i, log_v_i) with model covariance
# =============================================================================

"""
    _predictive_score_penalty(rᵢ::T, ∂θgᵢ_w::SVector{n_θ,T}, Σ̃_θ, s2) -> (qᵢ, ℓᵢ_model)

Model-uncertain score and penalty for scalar constraint (dg=1, Appendix D).

In whitened coordinates (rᵢ = gᵢ/√cᵢ, ∂θgᵢ_w = ∂θgᵢ/√cᵢ):
  Σ̃_w = s² + (∂θgᵢ_w)ᵀ Σ̃_θ (∂θgᵢ_w)    (scalar prediction variance shape)
  qᵢ = rᵢ² / Σ̃_w                          (weighted squared residual)
  ℓᵢ_model = log(Σ̃_w)                      (model uncertainty contribution)
"""
@inline function _predictive_score_penalty(rᵢ::T, ∂θgᵢ_w::SVector{n_θ,T},
                                            Σ̃_θ, s2) where {n_θ,T}
    Σ̃_w = s2 + dot(∂θgᵢ_w, Σ̃_θ * ∂θgᵢ_w)
    Σ̃_w > eps(T) || return (typemax(T), zero(T))
    return (rᵢ^2 / Σ̃_w, log(Σ̃_w))
end

"""
    _predictive_score_penalty(rᵢ::SVector{dg,T}, ∂θgᵢ_w::SMatrix{dg,n_θ,T}, Σ̃_θ, s2)
        -> (qᵢ, ℓᵢ_model)

Model-uncertain Mahalanobis score and penalty for vector constraint
(dg≥2, Eq. 7, Appendix D).

In whitened coordinates (rᵢ = L⁻¹gᵢ, ∂θgᵢ_w = L⁻¹∂θgᵢ):
  Σ̃_w = s²I + (∂θgᵢ_w) Σ̃_θ (∂θgᵢ_w)ᵀ    (dg×dg prediction covariance shape)
  qᵢ = rᵢᵀ Σ̃_w⁻¹ rᵢ                       (weighted squared residual)
  ℓᵢ_model = log|Σ̃_w|                       (model uncertainty contribution)
"""
@inline function _predictive_score_penalty(rᵢ::SVector{dg,T}, ∂θgᵢ_w::SMatrix{dg,n_θ,T},
                                            Σ̃_θ, s2) where {dg,n_θ,T}
    Σ̃_w = s2 * SMatrix{dg,dg,T}(I) + ∂θgᵢ_w * Σ̃_θ * ∂θgᵢ_w'
    det_Σ̃_w = det(Σ̃_w)
    det_Σ̃_w > eps(T) || return (typemax(T), zero(T))
    qᵢ = dot(rᵢ, Σ̃_w \ rᵢ)
    return (qᵢ, log(det_Σ̃_w))
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
# score! — Trait-dispatched per-point scoring (Section 3.3)
# =============================================================================
#
# Three specializations of the constraint covariance shape Σ̃_{gᵢ} (Section 3.3):
#
#   Homoscedastic   (isotropic, dg=1):  qᵢ = rᵢ²,               ℓᵢ = 0
#   Heteroscedastic (model certain):    qᵢ = gᵢᵀΣ̃_{gᵢ}⁻¹gᵢ,    ℓᵢ = log|Σ̃_{gᵢ}|
#   Predictive      (model uncertain):  qᵢ = gᵢᵀΣ̃_{gᵢ}⁻¹gᵢ,    ℓᵢ = log|Σ̃_{gᵢ}|
#
# For Cases 2–3, residual_jacobian(problem, model, i) → (rᵢ, ∂θgᵢ_w, ℓᵢ_meas)
# computes the whitened quantities and measurement log-determinant in one pass,
# avoiding duplicate computation of sampson_jacobians.
# =============================================================================

"""
    _sq_norm(r) -> T

Squared norm: `r²` for scalar, `rᵀr` for vector. Computes the weighted
squared residual qᵢ = gᵢᵀ Σ̃_{gᵢ}⁻¹ gᵢ from whitened quantities (Eq. 12).
"""
@inline _sq_norm(r::Real) = r * r
@inline _sq_norm(r::SVector) = dot(r, r)

# Isotropic, dg=1 (Section 3.3): qᵢ = rᵢ², ℓᵢ = 0 (cᵢ cancels in Eq. 12)
function score!(ws, problem, ::AbstractMarginalQuality, model, ::Homoscedastic)
    n = data_size(problem)
    residuals!(ws.residuals, problem, model)
    @inbounds for i in 1:n
        ws.scores[i] = ws.residuals[i]^2
    end
    fill!(ws.penalties, zero(eltype(ws.penalties)))
end

# Model certain (Section 3.3): qᵢ = gᵢᵀΣ̃_{gᵢ}⁻¹gᵢ, ℓᵢ = log|Σ̃_{gᵢ}| (Eq. 12)
# Uses residual_jacobian to compute whitened rᵢ = L⁻¹gᵢ and ℓᵢ in one pass.
function score!(ws, problem, ::AbstractMarginalQuality, model, ::Heteroscedastic)
    n = data_size(problem)
    @inbounds for i in 1:n
        rᵢ, _, ℓᵢ = residual_jacobian(problem, model, i)
        ws.scores[i] = _sq_norm(rᵢ)         # qᵢ = rᵢᵀrᵢ = gᵢᵀΣ̃_{gᵢ}⁻¹gᵢ
        ws.residuals[i] = sqrt(ws.scores[i]) # |rᵢ| for adaptive stopping
        ws.penalties[i] = ℓᵢ                 # log|Σ̃_{gᵢ}|
    end
end

# Model uncertain (Section 3.3, Eq. 7, 28): Σ̃_{gᵢ} includes estimation error.
# Decomposition in whitened coordinates (LLᵀ = ∂ₓgᵢ Σ̃_{xᵢ} (∂ₓgᵢ)ᵀ):
#   Σ̃_{gᵢ} = LLᵀ + ∂θgᵢ Σ̃_θ (∂θgᵢ)ᵀ
#   log|Σ̃_{gᵢ}| = log|LLᵀ| + log|I + (L⁻¹∂θgᵢ) Σ̃_θ (L⁻¹∂θgᵢ)ᵀ|
#                = ℓᵢ_meas + ℓᵢ_model
function score!(ws::RansacWorkspace{M,T}, problem,
                       scoring::PredictiveMarginalQuality,
                       model::M, ::Predictive) where {M,T}
    n = data_size(problem)
    Σ̃_θ = model_covariance(scoring, problem, model, ws.sample_indices)

    if !isnothing(Σ̃_θ)
        @inbounds for i in 1:n
            rᵢ, ∂θgᵢ_w, ℓᵢ_meas = residual_jacobian(problem, model, i)
            ws.residuals[i] = sqrt(_sq_norm(rᵢ))
            qᵢ, ℓᵢ_model = _predictive_score_penalty(rᵢ, ∂θgᵢ_w, Σ̃_θ, one(T))
            ws.scores[i] = qᵢ
            ws.penalties[i] = ℓᵢ_meas + ℓᵢ_model  # log|Σ̃_{gᵢ}|
        end
    else
        # Fallback to model-certain (Σ_θ = 0) when solver_jacobian unavailable
        score!(ws, problem, scoring, model, measurement_covariance(problem))
    end
end

