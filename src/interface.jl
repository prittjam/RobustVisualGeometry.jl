# =============================================================================
# Robust Estimation Interface
# =============================================================================
#
# Abstract types and interfaces for robust estimation algorithms.
# Loss functions and scale estimators are defined in VisualGeometryCore
# and imported by the parent module.
#
# =============================================================================

# -----------------------------------------------------------------------------
# Abstract Types
# -----------------------------------------------------------------------------

"""
    AbstractEstimator

Abstract type for estimation strategies.

Implementations define how to solve robust fitting problems:
- `MEstimator{L}`: IRLS with loss L
- `GNCEstimator{G}`: Graduated non-convexity with GNC loss G
"""
abstract type AbstractEstimator end

# -----------------------------------------------------------------------------
# Robust Problem Interface
# -----------------------------------------------------------------------------

"""
    AbstractRobustProblem

Abstract type for robust estimation problems.

A concrete subtype encapsulates the data, solver mechanics, and residual
computation for a specific estimation task. The generic `robust_solve`
dispatches on both the problem type and estimator strategy.

# Required Methods
- `initial_solve(prob)` — compute unweighted initial estimate θ
- `compute_residuals(prob, θ)` — signed residuals from current θ
- `weighted_solve(prob, θ, ω)` — solve weighted sub-problem given IRLS weights ω
- `data_size(prob)` — number of data points
- `problem_dof(prob)` — degrees of freedom consumed by the model

# Optional Methods
- `convergence_metric(prob, θ_new, θ_old)` — default: relative parameter change

# Example
```julia
struct MyProblem <: AbstractRobustProblem
    A::Matrix{Float64}
    b::Vector{Float64}
end

initial_solve(p::MyProblem) = p.A \\ p.b
compute_residuals(p::MyProblem, θ) = p.b - p.A * θ
weighted_solve(p::MyProblem, θ, ω) = (W = Diagonal(ω); (p.A'*W*p.A) \\ (p.A'*W*p.b))
data_size(p::MyProblem) = length(p.b)
problem_dof(p::MyProblem) = size(p.A, 2)

result = robust_solve(MyProblem(A, b), MEstimator(TukeyLoss()))
```
"""
abstract type AbstractRobustProblem end

"""
    initial_solve(prob::AbstractRobustProblem) -> θ

Compute the initial unweighted estimate.
"""
function initial_solve end

"""
    compute_residuals(prob::AbstractRobustProblem, θ) -> Vector{Float64}

Compute signed residuals from current parameter estimate θ.
"""
function compute_residuals end

"""
    compute_residuals!(r::AbstractVector, prob::AbstractRobustProblem, θ)

Compute signed residuals in-place into `r`. Default copies from `compute_residuals`.
"""
function compute_residuals!(r::AbstractVector, prob::AbstractRobustProblem, θ)
    copyto!(r, compute_residuals(prob, θ))
end

"""
    weighted_solve(prob::AbstractRobustProblem, θ, ω) -> θ_new

Solve the weighted sub-problem given IRLS weights ω.
"""
function weighted_solve end

"""
    data_size(prob::Union{AbstractRobustProblem, AbstractRansacProblem}) -> Int

Return the number of data points in the problem.

Implemented by both `AbstractRobustProblem` and `AbstractRansacProblem` subtypes.
"""
function data_size end

"""
    problem_dof(prob::AbstractRobustProblem) -> Int

Return the number of degrees of freedom consumed by the model.
Used for finite-sample DOF correction in MAD scale estimation.
"""
function problem_dof end

"""
    convergence_metric(prob::AbstractRobustProblem, θ_new, θ_old) -> Float64

Compute convergence criterion between consecutive iterates.

Default: relative parameter change `norm(θ_new - θ_old) / (norm(θ_new) + eps())`.
Override for homogeneous parameters: `1 - abs(dot(θ_new, θ_old))`.
"""
convergence_metric(::AbstractRobustProblem, θ_new, θ_old) =
    norm(θ_new - θ_old) / (norm(θ_new) + eps())

# -----------------------------------------------------------------------------
# Result Type
# -----------------------------------------------------------------------------

"""
    RobustAttributes{T} <: AbstractAttributes

Attributes for robust fitting results (IRLS, GNC, conic fitting).

# Constructor
    RobustAttributes(stop_reason, residuals, weights, scale, iterations)

`converged` is derived: `true` unless `stop_reason === :max_iterations`.

# Fields
- `stop_reason::Symbol`: Why the estimator terminated
  (`:converged`, `:max_iterations`, `:closed_form`)
- `converged::Bool`: Whether the estimator converged (derived)
- `residuals::Vector{T}`: Final residuals
- `weights::Vector{T}`: Final robust weights (0 = outlier, 1 = inlier)
- `scale::T`: Estimated residual scale
- `iterations::Int`: Number of iterations performed
"""
struct RobustAttributes{T} <: AbstractAttributes
    stop_reason::Symbol
    converged::Bool
    residuals::Vector{T}
    weights::Vector{T}
    scale::T
    iterations::Int
    function RobustAttributes(stop_reason::Symbol, residuals::Vector{T}, weights::Vector{T}, scale::T, iterations::Int) where T
        converged = stop_reason !== :max_iterations
        new{T}(stop_reason, converged, residuals, weights, scale, iterations)
    end
end

"""
    RobustEstimate{T}

Type alias for `Attributed{Vector{T}, RobustAttributes{T}}`.

Result of robust linear fitting (IRLS, GNC).

# Property Access (via Attributed forwarding)
- `r.value` → coefficient vector
- `r.residuals`, `.weights`, `.scale`, `.iterations` → forwarded from attributes
- `r.converged` → stored in attributes
- `r.stop_reason` → `:converged`, `:max_iterations`, `:closed_form`

# Example
```julia
result = robust_solve(A, b, GNCEstimator())
θ = result.value
inliers = result.weights .> 0.5
```
"""
const RobustEstimate{T} = Attributed{Vector{T}, RobustAttributes{T}}

# -----------------------------------------------------------------------------
# IRLS Workspace
# -----------------------------------------------------------------------------

"""
    IRLSWorkspace{T<:AbstractFloat}

Pre-allocated buffers for IRLS iterations. Avoids per-iteration allocation
of residual and weight vectors.

# Constructor
```julia
IRLSWorkspace{Float64}(n)   # n = data_size(prob)
IRLSWorkspace(n)             # defaults to Float64
```

Pass to `robust_solve` via `workspace` kwarg. The workspace is reusable —
results copy `r` and `ω` before returning.
"""
struct IRLSWorkspace{T<:AbstractFloat}
    residuals::Vector{T}
    weights::Vector{T}
end
IRLSWorkspace{T}(n::Int) where T = IRLSWorkspace{T}(Vector{T}(undef, n), ones(T, n))
IRLSWorkspace(n::Int) = IRLSWorkspace{Float64}(n)

# StatsBase-compatible accessors (work for any Attributed with RobustAttributes)
coef(r::Attributed{<:Any, <:RobustAttributes}) = r.value
residuals(r::Attributed{<:Any, <:RobustAttributes}) = r.residuals
weights(r::Attributed{<:Any, <:RobustAttributes}) = r.weights
scale(r::Attributed{<:Any, <:RobustAttributes}) = r.scale
converged(r::Attributed{<:Any, <:RobustAttributes}) = r.converged
niter(r::Attributed{<:Any, <:RobustAttributes}) = r.iterations

# Pretty printing
function Base.show(io::IO, r::Attributed{<:AbstractVector, RobustAttributes{T}}) where {T}
    n_inliers = count(>(0.5), r.weights)
    n_total = length(r.weights)
    status = r.converged ? "converged" : string(r.stop_reason)
    print(io, "RobustResult{$T}($n_inliers/$n_total inliers, $(r.iterations) iter, $status)")
end
