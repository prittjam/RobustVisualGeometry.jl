# =============================================================================
# Scoring — Quality functions, F-tests, and stopping strategies
# =============================================================================
#
# Defines how RANSAC candidate models are evaluated and when the loop stops.
# Extracted from ransac_interface.jl for co-location of scoring types with
# their evaluation logic.
#
# Contents:
#   1. AbstractQualityFunction
#   2. Holy Trait: F-test type (BasicFTest, PredictiveFTest)
#   3. Local optimization strategies (NoLocalOptimization, SimpleRefit, FTestLocalOptimization)
#   4. Concrete quality types (ThresholdQuality, ChiSquareQuality, MarginalQuality, ...)
#   5. init_quality, default_local_optimization, _quality_improved
#   6. Stopping strategies (HypergeometricStopping, ScoreGapStopping)
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

Quality functions determine how candidate models are evaluated in the RANSAC
inner loop. Following Shekhovtsov's "RANSAC Scoring Functions" (arXiv:2512.19850),
all quality functions compute Q(θ) = Σ ρ(rᵢ) where **higher = better**.

The main loop is strategy-agnostic; quality-specific behavior is dispatched
through `init_quality`, `_try_model!`, and `_finalize`.

Local optimization (LO-RANSAC) is orthogonal to quality and passed as the
`local_optimization` kwarg to `ransac()`. Use `default_local_optimization(quality)` to get
the default local optimization for a given strategy.

Built-in strategies:
- `ThresholdQuality`: Truncated quality with fixed inlier threshold (MSAC)
- `ChiSquareQuality`: Truncated quality with chi-square cutoff
- `MarginalQuality`: Threshold-free marginal likelihood scoring (Bayesian)

See also: [`ThresholdQuality`](@ref), [`MarginalQuality`](@ref),
[`default_local_optimization`](@ref)
"""
abstract type AbstractQualityFunction end

# -----------------------------------------------------------------------------
# Holy Trait: F-test Type (Basic vs Predictive)
# -----------------------------------------------------------------------------
#
# NOTE: Defined early because FTestLocalOptimization{T} depends on AbstractTestType.
#

"""
    AbstractTestType

Holy Trait for the statistical test variant used in F-test inlier classification.

- `BasicFTest()`: Standard F-test using `r²/s²` (default)
- `PredictiveFTest()`: Prediction-corrected F-test using `r²/vᵢ` where
  `vᵢ = s²(1 + gᵢ' Σ_θ gᵢ)` accounts for parameter uncertainty (leverage)

Problems that implement [`prediction_variances!`](@ref) should override
`test_type` to return `PredictiveFTest()`.
"""
abstract type AbstractTestType end

"""
    BasicFTest <: AbstractTestType

Standard F-test: classifies point `i` as inlier when `rᵢ²/s² < F_{1,ν,α}`.

Assumes all points have equal prediction variance `s²`. This is exact when
the design matrix has uniform leverage (e.g., well-distributed data), but
underestimates variance for high-leverage points.
"""
struct BasicFTest <: AbstractTestType end

"""
    PredictiveFTest <: AbstractTestType

Prediction-corrected F-test: classifies point `i` as inlier when `rᵢ²/vᵢ < F_{1,ν,α}`
where `vᵢ = s²(1 + gᵢ' Σ_θ gᵢ)`.

The prediction variance `vᵢ` accounts for both measurement noise (`s²`) and
parameter uncertainty propagated through the Jacobian at point `i`. High-leverage
points get a wider acceptance band.

Requires the problem to implement [`prediction_variances!`](@ref).
"""
struct PredictiveFTest <: AbstractTestType end

"""
    test_type(problem::AbstractRansacProblem) -> AbstractTestType

Return the statistical test variant for F-test inlier classification.

Default: `BasicFTest()`. Override to `PredictiveFTest()` for problems that
implement [`prediction_variances!`](@ref).
"""
test_type(::AbstractRansacProblem) = BasicFTest()

# -----------------------------------------------------------------------------
# Local Refinement Strategies
# -----------------------------------------------------------------------------

"""
    AbstractLocalOptimization

Abstract type for local optimization strategies passed to `ransac()`.

Concrete subtypes control how the inlier mask is refined after initial scoring:
- `NoLocalOptimization()`: No local optimization (default for ThresholdQuality)
- `SimpleRefit()`: LS refit on inlier set, re-sweep; no iterative classification
- `FTestLocalOptimization(test, alpha, max_iter)`: Iterative F-test classification + refit

Passed as a keyword argument to `ransac()`, separate from the scoring strategy.
Use `default_local_optimization(scoring)` to get the default for a given scoring strategy.
"""
abstract type AbstractLocalOptimization end

"""
    SimpleRefit <: AbstractLocalOptimization

Least-squares refit on the k*-mask inlier set, then re-sweep for score.
No iterative inlier reclassification. Default local optimization for `MarginalQuality`
(via `default_local_optimization`).
"""
struct SimpleRefit <: AbstractLocalOptimization end

"""
    NoLocalOptimization <: AbstractLocalOptimization

No local optimization. The candidate model is scored directly without any
refit or F-test iteration. Default local optimization for `ThresholdQuality`
(via `default_local_optimization`).
"""
struct NoLocalOptimization <: AbstractLocalOptimization end

"""
    FTestLocalOptimization{T<:AbstractTestType} <: AbstractLocalOptimization

Iterative F-test local optimization: reclassify inliers via F(d, ν) test, refit model
on the new inlier set, and iterate until convergence or `max_iter`.

Each iteration is guarded by a **monotonicity check**: a scoring-consistent
quality (`_lo_quality`) is evaluated before and after the iteration, and the
loop stops if the quality does not increase. This guarantees the local search
improves (or at worst preserves) the real scoring objective.

Used both as LO-RANSAC (per trial in the main loop) and as the final
local optimization after the main loop. Pass as the `local_optimization` kwarg to `ransac()`.

# Fields
- `test::T`: Statistical test variant (`BasicFTest()` or `PredictiveFTest()`)
- `alpha::Float64`: Significance level for inlier classification
- `max_iter::Int`: Maximum local optimization iterations

# Constructor
```julia
FTestLocalOptimization()                                 # BasicFTest, α=0.01, 5 iters
FTestLocalOptimization(; test=PredictiveFTest(), alpha=0.005, max_iter=10)
```

# Example
```julia
result = ransac(problem, scoring;
                local_optimization=FTestLocalOptimization(test=PredictiveFTest(), alpha=0.01))
```
"""
struct FTestLocalOptimization{T<:AbstractTestType} <: AbstractLocalOptimization
    test::T
    alpha::Float64
    max_iter::Int
end
FTestLocalOptimization(; test::AbstractTestType=BasicFTest(), alpha::Float64=0.01,
                  max_iter::Int=5) =
    FTestLocalOptimization(test, alpha, max_iter)

# -----------------------------------------------------------------------------
# Concrete Scoring Strategies
# -----------------------------------------------------------------------------

"""
    ThresholdQuality{L, S} <: AbstractQualityFunction

MSAC-style truncated quality with a fixed inlier threshold.

Per-point quality is `max(threshold - ρ(loss, r/σ), 0)`, and the total quality
Q = Σ max(threshold - ρ(rᵢ/σ), 0) is **maximized** (higher = better).

Local optimization is controlled separately via the `local_optimization` kwarg to `ransac()`.

# Fields
- `loss::L`: Loss function (e.g., `CauchyLoss()`, `L2Loss()`)
- `threshold::Float64`: MSAC truncation threshold
- `scale_estimator::S`: Scale estimator (e.g., `FixedScale()`)

# Default Refinement
`NoLocalOptimization()` (plain MSAC). Pass `local_optimization` kwarg to enable LO.

# Examples
```julia
# Plain MSAC
scoring = ThresholdQuality(L2Loss(), 0.05, FixedScale())
result = ransac(problem, scoring)

# MSAC + LO-RANSAC via F-test local optimization
result = ransac(problem, scoring;
                local_optimization=FTestLocalOptimization(test=PredictiveFTest()))
```

See also: [`ransac`](@ref), [`default_local_optimization`](@ref)
"""
struct ThresholdQuality{L<:AbstractLoss, S<:AbstractScaleEstimator} <: AbstractQualityFunction
    loss::L
    threshold::Float64
    scale_estimator::S
end

"""
    ChiSquareQuality{S} <: AbstractQualityFunction

Chi-square hypothesis test with truncated quality.

The chi-square test is inherently L2 — `(rᵢ/σ)²` follows `χ²(d_g)` under
the null hypothesis, where `d_g = codimension(problem)` is the co-dimension
of the model manifold.

- **Inlier classification**: `(rᵢ/σ)² < χ²(d_g, 1-α)`
- **Per-point quality**: `q(rᵢ) = max(cutoff - (rᵢ/σ)², 0)`
- **Model quality**: `Q(θ) = Σᵢ q(rᵢ)` — **higher = better**

The cutoff is computed internally from `α` and `codimension(problem)`.

# Fields
- `scale_estimator::S`: Scale estimator (e.g., `FixedScale(σ=1.0)`, `MADScale()`)
- `α::Float64`: Significance level for chi-square inlier test (e.g., 0.01)

# Examples
```julia
# Known noise σ=2.0, 1% significance
scoring = ChiSquareQuality(FixedScale(σ=2.0), 0.01)
result = ransac(problem, scoring)

# With F-test local optimization
result = ransac(problem, scoring;
                local_optimization=FTestLocalOptimization(test=PredictiveFTest(), alpha=0.01))
```

See also: [`ThresholdQuality`](@ref), [`ransac`](@ref)
"""
struct ChiSquareQuality{S<:AbstractScaleEstimator} <: AbstractQualityFunction
    scale_estimator::S
    α::Float64
end

"""
    TruncatedQuality

Union of truncated quality strategies (ThresholdQuality and ChiSquareQuality).
Used for shared dispatch of `_score_candidates!` and `_finalize`.
"""
const TruncatedQuality = Union{ThresholdQuality, ChiSquareQuality}

"""
    AbstractMarginalQuality <: AbstractQualityFunction

Abstract supertype for marginal likelihood quality functions.

All marginal quality variants share `init_quality`, `_marginal_sweep!`, and
the k*-mask `_try_model!` inner loop. Local optimization is passed separately
to `ransac()` via the `local_optimization` keyword argument.
"""
abstract type AbstractMarginalQuality <: AbstractQualityFunction end

"""
    MarginalQuality <: AbstractMarginalQuality

Threshold-free marginal likelihood quality function.

Scores models by marginalizing the noise variance σ² against the Jeffreys
prior `π(σ²) ∝ 1/σ²`, producing a σ-free score:

    S(θ, I) = logΓ(k/2) - (k/2)·log(RSS) - (n-k)·log(2a)

where k is the inlier count, RSS the inlier residual sum of squares,
and a the outlier half-width. Local optimization is controlled separately
via the `local_optimization` kwarg to `ransac()`.

# Fields
- `log2a::Float64`: log(2a), precomputed outlier penalty per point
- `model_dof::Int`: Model degrees of freedom (minimum inlier count)
- `codimension::Int`: Co-dimension d_g of the model manifold
- `perm::Vector{Int}`: Pre-allocated sortperm buffer (mutated in-place)
- `lg_table::Vector{Float64}`: Precomputed loggamma(k/2) for k=1..n

# Default Refinement
`NoLocalOptimization()`. Pass `local_optimization` kwarg to enable LO (e.g., `SimpleRefit()`).

# Examples
```julia
# Marginal quality with no local optimization (default)
scoring = MarginalQuality(n, p, 50.0)
result = ransac(problem, scoring)

# With F-test local optimization → returns UncertainRansacEstimate
result = ransac(problem, scoring;
                local_optimization=FTestLocalOptimization(test=PredictiveFTest()))
```

See also: [`ransac`](@ref), [`default_local_optimization`](@ref)
"""
mutable struct MarginalQuality <: AbstractMarginalQuality
    log2a::Float64
    model_dof::Int
    codimension::Int
    perm::Vector{Int}
    lg_table::Vector{Float64}
end

function _build_lg_table(n::Int)
    lg = Vector{Float64}(undef, n)
    # loggamma(k/2) via recurrence: Γ(x+1) = xΓ(x)
    if n >= 1
        lg[1] = 0.5 * log(π)  # loggamma(1/2) = log(√π)
    end
    if n >= 2
        lg[2] = 0.0  # loggamma(1) = 0
    end
    @inbounds for k in 3:n
        lg[k] = log((k - 2) / 2) + lg[k - 2]
    end
    return lg
end

function MarginalQuality(n::Int, p::Int, a::Float64; codimension::Int=1)
    a > 0 || throw(ArgumentError("outlier_halfwidth must be positive, got $a"))
    lg = _build_lg_table(n)
    MarginalQuality(log(2a), p, codimension, Vector{Int}(undef, n), lg)
end

"""
    PredictiveMarginalQuality <: AbstractMarginalQuality

Threshold-free marginal likelihood quality on prediction-corrected F-statistics.

Like [`MarginalQuality`](@ref), but the marginal sweep operates on
`F_i = r_i² / V_i` (where `V_i = s²(1 + leverage_i)`) instead of raw `r_i²`.
This accounts for parameter uncertainty from the minimal sample, giving
high-leverage points a wider acceptance band.

When `solver_jacobian(problem, ...)` returns `nothing` (problem does not
support it), falls back to raw `r²` (identical to `MarginalQuality`).

# Fields
Same as [`MarginalQuality`](@ref).

# Constructor
```julia
scoring = PredictiveMarginalQuality(n, p, a)
```

See also: [`MarginalQuality`](@ref), [`solver_jacobian`](@ref),
[`prediction_fstats_from_cov!`](@ref)
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
    lg = _build_lg_table(n)
    PredictiveMarginalQuality(log(2a), p, codimension, Vector{Int}(undef, n), lg)
end

"""
    init_quality(scoring::AbstractQualityFunction)

Return the initial "best" quality value for the scoring strategy.
All strategies use -Inf variants (higher = better).

- `TruncatedQuality`: `-Inf` (scalar quality to maximize)
- `AbstractMarginalQuality`: `(-Inf, -Inf)` (global, local score tuple to maximize)
"""
init_quality(::TruncatedQuality) = -Inf
init_quality(::AbstractMarginalQuality) = (-Inf, -Inf)

"""
    default_local_optimization(scoring::AbstractQualityFunction) -> AbstractLocalOptimization

Return the default local optimization strategy. Returns `NoLocalOptimization()` for
all quality functions. Pass an explicit `local_optimization` kwarg to `ransac()` to
enable local optimization (e.g., `SimpleRefit()` or `FTestLocalOptimization()`).

# Example
```julia
scoring = ThresholdQuality(L2Loss(), 0.05, FixedScale())
default_local_optimization(scoring)  # NoLocalOptimization()

scoring = MarginalQuality(100, 2, 50.0)
default_local_optimization(scoring)  # NoLocalOptimization()
```
"""
default_local_optimization(::AbstractQualityFunction) = NoLocalOptimization()

# -----------------------------------------------------------------------------
# Quality Improvement Check (unified: higher = better)
# -----------------------------------------------------------------------------

"""
    _quality_improved(scoring, new, old) -> Bool

Check whether `new` quality is strictly better than `old`.
All strategies use higher = better convention.
"""
_quality_improved(::TruncatedQuality, new, old) = new > old
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
