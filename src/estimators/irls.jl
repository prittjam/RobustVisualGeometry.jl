# =============================================================================
# Iteratively Reweighted Least Squares (IRLS)
# =============================================================================
#
# Generic IRLS solver for M-estimation problems via AbstractRobustProblem.

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
result = robust_solve(problem, MEstimator(TukeyLoss()))
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
result = robust_solve(prob, MEstimator(TukeyLoss()))
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
# Generic IRLS Solver
# -----------------------------------------------------------------------------

"""
    robust_solve(prob::AbstractRobustProblem, estimator::MEstimator; kwargs...)

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
  automatically chains into `robust_solve(refine, ...; init=θ)`.
- `refine_max_iter::Int=30`: Maximum iterations for the refinement phase

# Returns
`Attributed{V, RobustAttributes{T}}` where `V` is the parameter type.
Access via property forwarding: `result.value`, `result.residuals`,
`result.weights`, `result.scale`, `result.converged`, `result.stop_reason`.

# Example
```julia
# Single-phase
result = robust_solve(prob, MEstimator(GemanMcClureLoss()))

# Two-phase: robust Taubin → robust FNS
result = robust_solve(taubin_prob, MEstimator(GemanMcClureLoss());
                      refine=fns_prob, max_iter=20, refine_max_iter=30)
```
"""
function robust_solve(prob::AbstractRobustProblem, estimator::MEstimator;
                      scale::AbstractScaleEstimator=MADScale(),
                      init=nothing,
                      max_iter::Int=50,
                      rtol::Float64=1e-6,
                      refine::Union{Nothing, AbstractRobustProblem}=nothing,
                      refine_max_iter::Int=30,
                      workspace::Union{Nothing, IRLSWorkspace}=nothing)
    n = data_size(prob)
    dof = problem_dof(prob)
    loss = estimator.loss

    ws = something(workspace, IRLSWorkspace(n))
    r = ws.residuals
    ω = ws.weights
    fill!(ω, one(eltype(ω)))

    θ = init === nothing ? initial_solve(prob) : init
    local σ::Float64
    stop = :max_iterations
    final_iter = max_iter

    for iter in 1:max_iter
        θ_old = θ

        compute_residuals!(r, prob, θ)
        σ = _corrected_scale(scale, r, n, dof)

        @inbounds for i in 1:n
            ω[i] = weight(loss, r[i] / σ)
        end

        θ = weighted_solve(prob, θ, ω)

        if convergence_metric(prob, θ, θ_old) < rtol
            stop = :converged
            final_iter = iter
            break
        end
    end

    compute_residuals!(r, prob, θ)
    σ = _corrected_scale(scale, r, n, dof)
    result = Attributed(θ, RobustAttributes(stop, copy(r), copy(ω), σ, final_iter))

    # Optional refinement phase (e.g., Taubin → FNS)
    if refine !== nothing
        r2 = robust_solve(refine, estimator; scale, init=result.value,
                          max_iter=refine_max_iter, rtol, workspace)
        total = result.iterations + r2.iterations
        return Attributed(r2.value, RobustAttributes(r2.stop_reason, r2.residuals,
                                                     copy(r2.weights), r2.scale, total))
    end
    result
end

# Convenience wrapper for (A, b)
robust_solve(A::AbstractMatrix, b::AbstractVector, est::MEstimator; kwargs...) =
    robust_solve(LinearRobustProblem(A, b), est; kwargs...)
