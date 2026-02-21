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
