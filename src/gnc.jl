# =============================================================================
# Graduated Non-Convexity (GNC)
# =============================================================================
#
# GNC solver for robust estimation with high outlier rates.
# Based on Black-Rangarajan duality between robust estimation and outlier processes.
#
# Reference:
# Yang et al. "Graduated Non-Convexity for Robust Spatial Perception"
# IEEE RA-L, 2020. https://arxiv.org/abs/1909.08605

# -----------------------------------------------------------------------------
# GNC Loss Functions
# -----------------------------------------------------------------------------

"""
    GNCTruncatedLS <: GNCLoss

GNC-parameterized Truncated Least Squares loss.

As μ increases, transitions from convex (L2-like) to hard threshold:
- μ → 0: All weights ≈ 1 (convex)
- μ → ∞: Hard threshold at c (non-convex)

Weight function:
- w = 0 if |r| ≥ c√((μ+1)/μ)  (outlier)
- w = 1 if |r| ≤ c√(μ/(μ+1))  (inlier)
- w = c/|r|·√(μ(μ+1)) - μ otherwise (transition)

# Fields
- `c::Float64`: Inlier/outlier threshold
- `μ::Float64`: Convexity parameter (increases during optimization)
"""
struct GNCTruncatedLS <: GNCLoss
    c::Float64
    μ::Float64
end

function weight(loss::GNCTruncatedLS, r::Real)
    c, μ = loss.c, loss.μ
    r_abs = abs(r)

    # Thresholds from Black-Rangarajan duality
    thresh_hi = c * sqrt((μ + 1) / μ)
    thresh_lo = c * sqrt(μ / (μ + 1))

    if r_abs >= thresh_hi
        return zero(r)  # outlier
    elseif r_abs <= thresh_lo
        return one(r)   # inlier
    else
        # Smooth transition
        return c / r_abs * sqrt(μ * (μ + 1)) - μ
    end
end

# GNC losses don't have traditional rho/psi (weights encode everything)
rho(loss::GNCTruncatedLS, r::Real) = weight(loss, r) * r^2 / 2
psi(loss::GNCTruncatedLS, r::Real) = weight(loss, r) * r
tuning_constant(loss::GNCTruncatedLS) = loss.c

"""
    GNCGemanMcClure <: GNCLoss

GNC-parameterized Geman-McClure loss.

Smoother transition than TruncatedLS. As μ increases,
the effective threshold narrows.

Weight: w = (c²/μ / (r² + c²/μ))²

# Fields
- `c::Float64`: Base threshold parameter
- `μ::Float64`: Convexity parameter
"""
struct GNCGemanMcClure <: GNCLoss
    c::Float64
    μ::Float64
end

function weight(loss::GNCGemanMcClure, r::Real)
    c_eff = loss.c / sqrt(loss.μ)  # effective threshold shrinks as μ grows
    c_eff_sq = c_eff^2
    return (c_eff_sq / (r^2 + c_eff_sq))^2
end

rho(loss::GNCGemanMcClure, r::Real) = weight(loss, r) * r^2 / 2
psi(loss::GNCGemanMcClure, r::Real) = weight(loss, r) * r
tuning_constant(loss::GNCGemanMcClure) = loss.c

# -----------------------------------------------------------------------------
# Convergence Helpers (from MATLAB implementation)
# -----------------------------------------------------------------------------

"""
    are_binary_weights(w; tol=1e-12) -> Bool

Check if all weights are approximately binary (0 or 1).
Matches MATLAB areBinaryWeights.m convergence criterion.
"""
function are_binary_weights(w::AbstractVector; tol::Real=1e-12)
    for wi in w
        if wi > tol && abs(1 - wi) > tol
            return false
        end
    end
    return true
end

"""
    uses_binary_convergence(::Type{G}) -> Bool

Dispatch function for convergence criterion selection.
- TLS: weights become exactly binary (0 or 1) → use binary check
- GM: weights are smooth → use weight change check
"""
uses_binary_convergence(::Type{GNCTruncatedLS}) = true
uses_binary_convergence(::Type{GNCGemanMcClure}) = false

# -----------------------------------------------------------------------------
# GNC Estimator
# -----------------------------------------------------------------------------

"""
    GNCEstimator{G<:GNCLoss} <: AbstractEstimator

Graduated Non-Convexity estimator.

Solves robust estimation by gradually transitioning from convex
to non-convex optimization, avoiding local minima.

# Algorithm
1. Initialize with unweighted solve (or provided init)
2. Set μ from residuals (start convex)
3. Iterate: update GNC weights → weighted solve → anneal μ
4. Stop when weights become binary, cost stabilizes, or max iterations

# Constructor
```julia
GNCEstimator()                           # defaults: TLS, c=1.0
GNCEstimator(GNCTruncatedLS)             # TLS with defaults
GNCEstimator(GNCGemanMcClure; c=2.0)     # GM with custom threshold
```

# Fields
- `c::Float64`: Inlier threshold (in units of residual scale)
- `μ_factor::Float64`: Annealing rate (μ *= μ_factor each iteration, default 1.4)

Solver parameters (`max_iter`, `weight_tol`, `cost_tol`) are kwargs to `robust_solve`.

# Example
```julia
result = robust_solve(prob, GNCEstimator(GNCTruncatedLS; c=1.0); max_iter=100)
```

# Reference
Yang et al. "Graduated Non-Convexity for Robust Spatial Perception"
IEEE RA-L, 2020. https://arxiv.org/abs/1909.08605
"""
struct GNCEstimator{G<:GNCLoss} <: AbstractEstimator
    c::Float64
    μ_factor::Float64
end

function GNCEstimator(::Type{G}=GNCTruncatedLS;
                      c::Real=1.0,
                      μ_factor::Real=1.4) where {G<:GNCLoss}
    GNCEstimator{G}(Float64(c), Float64(μ_factor))
end

# -----------------------------------------------------------------------------
# Generic GNC Solver
# -----------------------------------------------------------------------------

"""
    _gnc_result(prob, θ, ω, r, scale_est, n, dof, stop_reason, iter)

Package GNC solver state into an Attributed result. Avoids 4× code duplication.
Uses workspace buffer `r` for final residuals, then copies for result.
"""
function _gnc_result(prob, θ, ω, r, scale_est, n, dof, stop_reason, iter)
    compute_residuals!(r, prob, θ)
    σ = _corrected_scale(scale_est, r, n, dof)
    Attributed(θ, RobustAttributes(stop_reason, copy(r), copy(ω), σ, iter))
end

"""
    robust_solve(prob::AbstractRobustProblem, estimator::GNCEstimator; kwargs...)

Solve a robust estimation problem using Graduated Non-Convexity.

# Arguments
- `prob::AbstractRobustProblem`: Problem defining solver mechanics
- `estimator::GNCEstimator`: GNC estimator configuration

# Keyword Arguments
- `scale::AbstractScaleEstimator=MADScale()`: Scale estimation method
- `init=nothing`: Initial θ (default: `initial_solve(prob)`)
- `max_iter::Int=100`: Maximum GNC iterations
- `weight_tol::Float64=1e-12`: Binary weight tolerance (matches MATLAB)
- `cost_tol::Float64=0.0`: Cost convergence tolerance (0 = disabled)
- `workspace::Union{Nothing,IRLSWorkspace}=nothing`: Pre-allocated buffers

# Returns
`Attributed{V, RobustAttributes{T}}`.
Access via property forwarding: `result.value`, `result.residuals`,
`result.weights`, `result.scale`, `result.converged`, `result.stop_reason`.

# Example
```julia
result = robust_solve(prob, GNCEstimator(GNCTruncatedLS; c=1.0); max_iter=100)
inliers = result.weights .> 0.5
```
"""

function robust_solve(prob::AbstractRobustProblem, estimator::GNCEstimator{G};
                      scale::AbstractScaleEstimator=MADScale(),
                      init=nothing,
                      max_iter::Int=100,
                      weight_tol::Float64=1e-12,
                      cost_tol::Float64=0.0,
                      workspace::Union{Nothing, IRLSWorkspace}=nothing) where {G}
    n = data_size(prob)
    dof = problem_dof(prob)
    c = estimator.c

    ws = something(workspace, IRLSWorkspace(n))
    r = ws.residuals
    ω = ws.weights
    fill!(ω, one(eltype(ω)))

    # Initial solve
    θ = init === nothing ? initial_solve(prob) : init
    compute_residuals!(r, prob, θ)
    σ = _corrected_scale(scale, r, n, dof)

    # Normalize threshold by scale
    c_scaled = c * σ

    # Initialize μ using Black-Rangarajan formula
    r_max_sq = maximum(ri -> ri^2, r)
    c_sq = c_scaled^2
    μ = if 2 * r_max_sq > c_sq
        c_sq / (2 * r_max_sq - c_sq)
    else
        1e6
    end
    μ = clamp(μ, 1e-6, 1e8)

    ω_prev = copy(ω)
    use_binary = uses_binary_convergence(G)
    cost_prev = Inf

    for iter in 1:max_iter
        copyto!(ω_prev, ω)

        # Create GNC loss at current μ
        loss = G(c_scaled, μ)

        # Compute GNC weights from residuals
        compute_residuals!(r, prob, θ)
        @inbounds for i in 1:n
            ω[i] = weight(loss, r[i])
        end

        # Weighted solve
        θ = weighted_solve(prob, θ, ω)

        # Convergence check
        if use_binary
            are_binary_weights(ω; tol=weight_tol) &&
                return _gnc_result(prob, θ, ω, r, scale, n, dof, :converged, iter)
        else
            max_wc = maximum(i -> abs(ω[i] - ω_prev[i]), 1:n)
            max_wc < sqrt(weight_tol) &&
                return _gnc_result(prob, θ, ω, r, scale, n, dof, :converged, iter)
        end

        # Cost convergence
        cost = sum(i -> ω[i] * r[i]^2, 1:n)
        if cost_tol > 0 && abs(cost - cost_prev) < cost_tol
            return _gnc_result(prob, θ, ω, r, scale, n, dof, :converged, iter)
        end
        cost_prev = cost

        # Anneal: harden the threshold
        μ *= estimator.μ_factor
    end

    _gnc_result(prob, θ, ω, r, scale, n, dof, :max_iterations, max_iter)
end

# Convenience wrapper for (A, b)
robust_solve(A::AbstractMatrix, b::AbstractVector, est::GNCEstimator; kwargs...) =
    robust_solve(LinearRobustProblem(A, b), est; kwargs...)
