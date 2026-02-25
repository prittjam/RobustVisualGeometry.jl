# =============================================================================
# Conic Fitting — Algebraic Least Squares + Finalization Helpers
# =============================================================================

# =============================================================================
# Finalization Helpers
# =============================================================================

"""
    _finalize_conic_result(result, T, pts, sigma) -> ConicFitResult

Denormalize θ from normalized coordinates, recompute residuals in original
coordinates, and package into a ConicFitResult.
"""
function _finalize_conic_result(result, T::SMatrix{3,3,Float64,9},
                                pts::Vector{SVector{2,Float64}}, sigma::Real)
    theta_orig = _denormalize_conic(result.value, T)
    xis_orig = _build_carriers(pts)
    Lambdas_orig = _build_covariances(pts, sigma^2)
    residuals = _compute_sampson_residuals(sampson_distance, theta_orig, xis_orig, Lambdas_orig)
    Attributed(theta_orig, RobustAttributes(result.stop_reason, residuals,
                                            copy(result.weights), result.scale,
                                            result.iterations))
end

"""
    _finalize_closed_form(theta_norm, T, pts, sigma) -> ConicFitResult

Denormalize θ, compute residuals, return result with :closed_form stop reason.
Used by ALS and Taubin.
"""
function _finalize_closed_form(theta_norm::SVector{6,Float64}, T::SMatrix{3,3,Float64,9},
                               pts::Vector{SVector{2,Float64}}, sigma::Real)
    theta = _denormalize_conic(theta_norm, T)
    xis_orig = _build_carriers(pts)
    Lambdas_orig = _build_covariances(pts, sigma^2)
    residuals = _compute_sampson_residuals(sampson_distance, theta, xis_orig, Lambdas_orig)
    n = length(pts)
    Attributed(theta, RobustAttributes(:closed_form, residuals, ones(n), sigma, 0))
end

# =============================================================================
# 1. Algebraic Least Squares
# =============================================================================

"""
    fit_conic_als(points; sigma=1.0) -> ConicFitResult

Algebraic least squares: minimize theta^T M theta subject to ||theta|| = 1,
where M = sum(xi_i * xi_i^T).

# Arguments
- `points`: 2×N matrix, N×2 matrix, or Vector of 2D points
- `sigma::Real=1.0`: Noise standard deviation (for residual computation only)
"""
function fit_conic_als(points; sigma::Real=1.0)
    pts, norm_pts, T, _ = _prepare_and_normalize(points, sigma)
    xis = _build_carriers(norm_pts)

    M = zeros(SMatrix{6,6,Float64,36})
    @inbounds for i in 1:length(xis)
        M += xis[i] * xis[i]'
    end

    _finalize_closed_form(_solve_smallest_eigvec(M), T, pts, sigma)
end
