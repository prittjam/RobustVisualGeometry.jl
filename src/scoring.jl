# =============================================================================
# Scoring — Quality functions and stopping strategies
# =============================================================================
#
# Defines how RANSAC candidate models are evaluated and when the loop stops.
#
# Contents:
#   1. AbstractQualityFunction
#   2. Local optimization strategies (NoLocalOptimization)
#   3. Concrete quality types (MarginalQuality, PredictiveMarginalQuality)
#   4. init_quality, default_local_optimization, _quality_improved
#   5. Stopping strategies (HypergeometricStopping, ScoreGapStopping)
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

# -----------------------------------------------------------------------------
# Stopping Strategies
# -----------------------------------------------------------------------------

"""
    AbstractStoppingStrategy

Abstract type for RANSAC early stopping strategies.

Stopping strategies are queried each iteration via `_check_early_stop` to
determine whether to terminate the main loop early. The default
`HypergeometricStopping` never stops early (adaptive trial count suffices).

See also: [`stopping_strategy`](@ref), [`_check_early_stop`](@ref)
"""
abstract type AbstractStoppingStrategy end

"""
    HypergeometricStopping <: AbstractStoppingStrategy

Default stopping strategy: rely on the hypergeometric adaptive trial count.
`_check_early_stop` always returns `false`.
"""
struct HypergeometricStopping <: AbstractStoppingStrategy end

"""
    ScoreGapStopping <: AbstractStoppingStrategy

Score-gap early stopping for marginal scoring strategies.

Terminates when `T_rem / (gap + 2) < eps`, where `T_rem` is the remaining
trial budget and `gap` is the number of trials since the last score improvement.
"""
struct ScoreGapStopping <: AbstractStoppingStrategy
    eps::Float64
end

"""
    stopping_strategy(scoring::AbstractQualityFunction) -> AbstractStoppingStrategy

Return the stopping strategy for the given quality function.

Default: `HypergeometricStopping()` for all strategies.
"""
stopping_strategy(::AbstractQualityFunction) = HypergeometricStopping()

"""
    _check_early_stop(strategy, trial, needed, best, old_best, gap) -> Bool

Check whether the main loop should terminate early.
"""
_check_early_stop(::HypergeometricStopping, trial, needed, best, old_best, gap) = false

function _check_early_stop(s::ScoreGapStopping, trial, needed, best, old_best, gap)
    T_rem = needed - trial
    T_rem > 0 && gap > 0 && Float64(T_rem) / Float64(gap + 2) < s.eps
end
