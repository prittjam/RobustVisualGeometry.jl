# =============================================================================
# Iteratively Reweighted Least Squares (IRLS)
# =============================================================================
#
# Generic IRLS solver for M-estimation problems via AbstractRobustProblem.
# Uses the generic irls! loop from irls_framework.jl.

# -----------------------------------------------------------------------------
# M-Estimator Type
# -----------------------------------------------------------------------------

"""
    MEstimator{L<:AbstractLoss} <: AbstractEstimator

M-estimator using Iteratively Reweighted Least Squares (IRLS).

Solves: min_θ Σ ρ(rᵢ/σ) by iterating:
1. Compute residuals: r = compute_residuals(prob, θ)
2. Estimate scale: σ = _corrected_scale(scale, r, n, dof)
3. Compute weights: wᵢ = weight(loss, rᵢ/σ)
4. Solve weighted sub-problem: θ = weighted_solve(prob, θ, ω)

# Constructor
```julia
MEstimator(CauchyLoss())           # with loss instance
MEstimator(CauchyLoss)             # with loss type (uses defaults)
MEstimator(CauchyLoss; c=3.0)      # with loss type and kwargs
```

# Example
```julia
result = fit(problem, MEstimator(TukeyLoss()))
θ = result.value
```
"""
struct MEstimator{L<:AbstractLoss} <: AbstractEstimator
    loss::L
end

# Convenience constructor from type
MEstimator(::Type{L}; kwargs...) where {L<:AbstractLoss} = MEstimator(L(; kwargs...))

# -----------------------------------------------------------------------------
# LinearRobustProblem — wraps (A, b) for the generic API
# -----------------------------------------------------------------------------

"""
    LinearRobustProblem{M,V} <: AbstractRobustProblem

Wraps a design matrix `A` and response vector `b` as a robust problem.

# Example
```julia
prob = LinearRobustProblem(A, b)
result = fit(prob, MEstimator(TukeyLoss()))
```
"""
struct LinearRobustProblem{M<:AbstractMatrix, V<:AbstractVector} <: AbstractRobustProblem
    A::M
    b::V
end

initial_solve(prob::LinearRobustProblem) = prob.A \ prob.b
compute_residuals(prob::LinearRobustProblem, θ) = prob.b - prob.A * θ

function weighted_solve(prob::LinearRobustProblem, θ, ω)
    W = Diagonal(ω)
    AtWA = prob.A' * W * prob.A
    if cond(AtWA) > 1e12
        AtWA += 1e-10 * I
    end
    AtWA \ (prob.A' * W * prob.b)
end

data_size(prob::LinearRobustProblem) = size(prob.A, 1)
problem_dof(prob::LinearRobustProblem) = size(prob.A, 2)

# -----------------------------------------------------------------------------
# MEstimatorMethod — IRLS framework specialization
# -----------------------------------------------------------------------------

struct MEstimatorMethod{L,S} <: AbstractIRLSMethod
    loss::L
    scale_est::S
    rtol::Float64
    n::Int
    dof::Int
end

struct MEstimatorState{T}
    r::Vector{T}
    w::Vector{T}
    σ::Base.RefValue{T}
end

function init_state(method::MEstimatorMethod, prob, θ₀)
    r = method.n > 0 ? Vector{Float64}(undef, method.n) : Float64[]
    w = ones(Float64, method.n)
    σ = Ref(0.0)
    compute_residuals!(r, prob, θ₀)
    σ[] = _corrected_scale(method.scale_est, r, method.n, method.dof)
    @inbounds for i in 1:method.n
        w[i] = weight(method.loss, r[i] / σ[])
    end
    MEstimatorState(r, w, σ)
end

function solve_step(state::MEstimatorState, method::MEstimatorMethod, prob, θ)
    weighted_solve(prob, θ, state.w)
end

function update_residuals!(state::MEstimatorState, method::MEstimatorMethod, prob, θ)
    compute_residuals!(state.r, prob, θ)
end

function update_scale!(state::MEstimatorState, method::MEstimatorMethod, prob)
    state.σ[] = _corrected_scale(method.scale_est, state.r, method.n, method.dof)
end

function update_weights!(state::MEstimatorState, method::MEstimatorMethod, prob)
    σ = state.σ[]
    @inbounds for i in 1:method.n
        state.w[i] = weight(method.loss, state.r[i] / σ)
    end
end

function is_converged(state::MEstimatorState, method::MEstimatorMethod, prob, θ, θ_old, iter)
    convergence_metric(prob, θ, θ_old) < method.rtol
end

function irls_result(state::MEstimatorState, method::MEstimatorMethod, prob, θ, converged, final_iter)
    # Compute final residuals and scale
    compute_residuals!(state.r, prob, θ)
    σ = _corrected_scale(method.scale_est, state.r, method.n, method.dof)
    stop = converged ? :converged : :max_iterations
    Attributed(θ, RobustAttributes(stop, copy(state.r), copy(state.w), σ, final_iter))
end

# -----------------------------------------------------------------------------
# Generic IRLS Solver
# -----------------------------------------------------------------------------

"""
    fit(prob::AbstractRobustProblem, estimator::MEstimator; kwargs...)

Solve a robust estimation problem using IRLS.

# Arguments
- `prob::AbstractRobustProblem`: Problem defining solver mechanics
- `estimator::MEstimator`: M-estimator with loss function

# Keyword Arguments
- `scale::AbstractScaleEstimator=MADScale()`: Scale estimation method
- `init=nothing`: Initial θ (default: `initial_solve(prob)`)
- `max_iter::Int=50`: Maximum IRLS iterations
- `rtol::Float64=1e-6`: Convergence tolerance
- `refine::Union{Nothing,AbstractRobustProblem}=nothing`: Optional second problem
  for two-phase estimation (e.g., Taubin → FNS). After IRLS converges on `prob`,
  automatically chains into `fit(refine, ...; init=θ)`.
- `refine_max_iter::Int=30`: Maximum iterations for the refinement phase

# Returns
`Attributed{V, RobustAttributes{T}}` where `V` is the parameter type.
Access via property forwarding: `result.value`, `result.residuals`,
`result.weights`, `result.scale`, `result.converged`, `result.stop_reason`.

# Example
```julia
# Single-phase
result = fit(prob, MEstimator(GemanMcClureLoss()))

# Two-phase: robust Taubin → robust FNS
result = fit(taubin_prob, MEstimator(GemanMcClureLoss());
                      refine=fns_prob, max_iter=20, refine_max_iter=30)
```
"""
function fit(prob::AbstractRobustProblem, estimator::MEstimator;
                      scale::AbstractScaleEstimator=MADScale(),
                      init=nothing,
                      max_iter::Int=50,
                      rtol::Float64=1e-6,
                      refine::Union{Nothing, AbstractRobustProblem}=nothing,
                      refine_max_iter::Int=30,
                      workspace::Union{Nothing, IRLSWorkspace}=nothing)
    n = data_size(prob)
    dof = problem_dof(prob)
    θ₀ = init === nothing ? initial_solve(prob) : init

    method = MEstimatorMethod(estimator.loss, scale, rtol, n, dof)
    result = irls!(method, prob, θ₀; max_iter)

    # Optional refinement phase (e.g., Taubin → FNS)
    if refine !== nothing
        r2 = fit(refine, estimator; scale, init=result.value,
                          max_iter=refine_max_iter, rtol, workspace)
        total = result.iterations + r2.iterations
        return Attributed(r2.value, RobustAttributes(r2.stop_reason, r2.residuals,
                                                     copy(r2.weights), r2.scale, total))
    end
    result
end

# Convenience wrapper for (A, b)
fit(A::AbstractMatrix, b::AbstractVector, est::MEstimator; kwargs...) =
    fit(LinearRobustProblem(A, b), est; kwargs...)
