# =============================================================================
# Shared GEP Solver and Fitting Helpers
# =============================================================================
#
# Generic infrastructure used by conic_fitting.jl and fundmat_fitting.jl:
#
#   _solve_smallest_gep    - Generalized eigenvalue problem (dimension-generic)
#   _convergence_angle     - Angle-based convergence metric
#   _scale_estimator       - Convert user scale arg to framework estimator
#   _compute_sampson_residuals - Generic Sampson residual loop
#
# PLACEMENT: Included from VisualGeometryCore.jl BEFORE conic_fitting.jl and
# fundmat_fitting.jl — both depend on these shared functions.
#
# =============================================================================

# Dependencies: LinearAlgebra (eigen, Symmetric, norm, dot, Diagonal)
#               VGC: MADScale, FixedScale (via parent module)

# =============================================================================
# GEP Solver (dimension-generic via SMatrix{N,N})
# =============================================================================

"""
    _solve_smallest_gep(A, B) → SVector{N}

Solve the generalized eigenvalue problem Ax = λBx for the eigenvector
corresponding to the smallest eigenvalue. Regularizes B if near-singular.
Returns a unit-norm SVector whose dimension matches the input matrices.
"""
function _solve_smallest_gep(A::SMatrix{N,N,Float64}, B::SMatrix{N,N,Float64}) where N
    B_mat = Matrix(B)
    A_mat = Matrix(A)
    F = eigen(Symmetric(B_mat))
    lambda_max = maximum(abs, F.values)
    tol = 1e-12 * lambda_max

    # Regularize B if rank-deficient (e.g., Taubin's N matrix is rank 5)
    if any(v -> abs(v) <= tol, F.values)
        eps_reg = 1e-10 * lambda_max
        B_mat = B_mat + eps_reg * LinearAlgebra.I
        F = eigen(Symmetric(B_mat))
    end

    D_inv_half = Diagonal(1.0 ./ sqrt.(F.values))
    V = F.vectors
    C = D_inv_half * V' * A_mat * V * D_inv_half
    eig = eigen(Symmetric(C))
    theta = V * D_inv_half * eig.vectors[:, 1]
    SVector{N,Float64}(theta / norm(theta))
end

# =============================================================================
# Unweighted Taubin GEP (dimension-generic)
# =============================================================================

"""
    _taubin_seed_gep(xis, Js) -> SVector{N}

Unweighted Taubin solve: M = Σ ξᵢξᵢᵀ, N = Σ JᵢJᵢᵀ, solve smallest GEP(M, N).
Dimension-generic via SVector{N,T}; used by both conic and F-matrix Taubin seeds.
"""
function _taubin_seed_gep(xis::Vector{SVector{N,T}}, Js) where {N,T}
    M = zero(SMatrix{N,N,T,N*N})
    N_mat = zero(SMatrix{N,N,T,N*N})
    @inbounds for i in 1:length(xis)
        M += xis[i] * xis[i]'
        N_mat += Js[i] * Js[i]'
    end
    _solve_smallest_gep(M, N_mat)
end

# =============================================================================
# Convergence Metric
# =============================================================================

@inline function _convergence_angle(v_new::AbstractVector, v_old::AbstractVector)
    1.0 - abs(dot(v_new, v_old))
end

# =============================================================================
# Scale Estimator Conversion
# =============================================================================

"""
    _scale_estimator(s) -> AbstractScaleEstimator

Convert user-facing scale argument to framework scale estimator.
"""
_scale_estimator(::Nothing) = MADScale()
_scale_estimator(s::Real) = FixedScale(σ=Float64(s))

# =============================================================================
# Generic Sampson Residual Loop
# =============================================================================

"""
    _compute_sampson_residuals(sampson_fn, theta, xis, Lambdas) -> Vector{Float64}

Compute signed Sampson residuals using the provided distance function.
`sampson_fn(theta, xi, Lambda) -> Float64` is the domain-specific distance.
"""
function _compute_sampson_residuals(sampson_fn, theta, xis, Lambdas)
    n = length(xis)
    r = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        r[i] = sampson_fn(theta, xis[i], Lambdas[i])
    end
    r
end

# =============================================================================
# GEP Problem Type Hierarchy (Taubin / FNS)
# =============================================================================
#
# Abstract types for Taubin and FNS problems. Concrete subtypes must have
# fields `xis` and `Lambdas`; Taubin subtypes also need `Js`.
#
# Dispatch points (each concrete subtype implements):
#   _project(prob, θ)   — post-solve projection (default: identity)
#   _seed(prob)         — initial Taubin seed (domain-specific)
#   _sampson_fn(prob)   — signed Sampson distance function
#   problem_dof(prob)   — degrees of freedom
# =============================================================================

abstract type AbstractTaubinProblem <: AbstractRobustProblem end
abstract type AbstractFNSProblem <: AbstractRobustProblem end

const AbstractGEPProblem = Union{AbstractTaubinProblem, AbstractFNSProblem}

# --- Dispatch points (stubs) ---
function _project end
function _seed end
function _sampson_fn end

# --- Default: identity projection ---
_project(::AbstractGEPProblem, θ) = θ

# --- Shared interface on AbstractGEPProblem ---
data_size(prob::AbstractGEPProblem) = length(prob.xis)
convergence_metric(::AbstractGEPProblem, θ_new, θ_old) = _convergence_angle(θ_new, θ_old)
compute_residuals(prob::AbstractGEPProblem, θ) =
    _compute_sampson_residuals(_sampson_fn(prob), θ, prob.xis, prob.Lambdas)

# --- Taubin: weighted_solve and initial_solve ---
weighted_solve(prob::AbstractTaubinProblem, θ, ω) =
    _project(prob, _taubin_weighted_gep(prob.xis, prob.Js, ω))

initial_solve(prob::AbstractTaubinProblem) = _seed(prob)

# --- FNS: weighted_solve and initial_solve ---
weighted_solve(prob::AbstractFNSProblem, θ, ω) =
    _project(prob, _fns_weighted_gep(prob.xis, prob.Lambdas, θ, ω))

initial_solve(prob::AbstractFNSProblem) =
    _fns_initial_iterate(prob.xis, prob.Lambdas, _seed(prob);
                         project=θ -> _project(prob, θ))

# =============================================================================
# Generic Weighted GEP Solves (dimension-generic via SVector{N,T})
# =============================================================================

"""
    _taubin_weighted_gep(xis, Js, ω) -> SVector{N}

Weighted Taubin GEP: M = Σ ωᵢ(ξᵢξᵢᵀ), N = Σ ωᵢ(JᵢJᵢᵀ), solve GEP(M, N).
"""
function _taubin_weighted_gep(xis::Vector{SVector{N,T}}, Js, ω) where {N,T}
    M = zero(SMatrix{N,N,T,N*N})
    N_mat = zero(SMatrix{N,N,T,N*N})
    @inbounds for i in 1:length(xis)
        M += ω[i] * (xis[i] * xis[i]')
        N_mat += ω[i] * (Js[i] * Js[i]')
    end
    _solve_smallest_gep(M, N_mat)
end

"""
    _fns_weighted_gep(xis, Lambdas, θ, ω) -> SVector{N}

Weighted FNS GEP with bias correction vᵢ = 1/(θᵀΛᵢθ):
M = Σ wᵢvᵢ(ξᵢξᵢᵀ), N = Σ wᵢvᵢΛᵢ, solve GEP(M-N, M).
"""
function _fns_weighted_gep(xis::Vector{SVector{N,T}}, Lambdas, θ, ω) where {N,T}
    M = zero(SMatrix{N,N,T,N*N})
    N_mat = zero(SMatrix{N,N,T,N*N})
    @inbounds for i in 1:length(xis)
        s2 = dot(θ, Lambdas[i] * θ)
        v = one(T) / max(s2, eps(T))
        w = ω[i] * v
        M += w * (xis[i] * xis[i]')
        N_mat += w * Lambdas[i]
    end
    _solve_smallest_gep(M - N_mat, M)
end

"""
    _fns_initial_iterate(xis, Lambdas, seed; project=identity, n_iter=5, tol=1e-10)

FNS warmup: starting from `seed`, run `n_iter` unweighted FNS iterations with
optional post-projection (e.g., rank-2 enforcement for fundamental matrices).
"""
function _fns_initial_iterate(xis::Vector{SVector{N,T}}, Lambdas, seed::SVector{N,T};
                               project=identity, n_iter::Int=5,
                               tol::Float64=1e-10) where {N,T}
    θ = seed
    for _ in 1:n_iter
        θ_old = θ
        M = zero(SMatrix{N,N,T,N*N})
        N_mat = zero(SMatrix{N,N,T,N*N})
        @inbounds for i in 1:length(xis)
            s2 = dot(θ, Lambdas[i] * θ)
            v = one(T) / max(s2, eps(T))
            M += v * (xis[i] * xis[i]')
            N_mat += v * Lambdas[i]
        end
        θ = project(_solve_smallest_gep(M - N_mat, M))
        _convergence_angle(θ, θ_old) < tol && break
    end
    θ
end
