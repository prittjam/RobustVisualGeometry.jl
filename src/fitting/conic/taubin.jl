# =============================================================================
# Conic Fitting — Taubin and Robust Taubin
# =============================================================================

"""
    fit_conic_taubin(points; sigma=1.0) -> ConicFitResult

Taubin's gradient-weighted method: minimize theta^T M theta / (theta^T N theta),
where M = sum(xi_i * xi_i^T) and N = sum(J_i * J_i^T).

# Arguments
- `points`: 2×N matrix, N×2 matrix, or Vector of 2D points
- `sigma::Real=1.0`: Noise standard deviation
"""
function fit_conic_taubin(points; sigma::Real=1.0)
    pts, norm_pts, T, _ = _prepare_and_normalize(points, sigma)
    xis = _build_carriers(norm_pts)
    _finalize_closed_form(_taubin_seed(xis), T, pts, sigma)
end

"""
    fit_conic_robust_taubin(points; sigma=1.0, loss=TukeyLoss(), scale=nothing,
                            max_iter=50, rtol=1e-5) -> ConicFitResult

Robust Taubin: IRLS-weighted generalized eigenvalue problem using Taubin's
gradient-weighted scatter matrices.

Uses the generic `robust_solve` with `ConicTaubinProblem`.

# Arguments
- `points`: 2×N matrix, N×2 matrix, or Vector of 2D points
- `sigma::Real=1.0`: Noise standard deviation (for carrier covariances)
- `loss::AbstractLoss=TukeyLoss()`: Robust loss function
- `scale::Union{Nothing,Real}=nothing`: Residual scale. `nothing` → MAD estimation,
  a number → fixed scale.
- `max_iter::Int=50`: Maximum IRLS iterations
- `rtol::Float64=1e-5`: Convergence tolerance
"""
function fit_conic_robust_taubin(points; sigma::Real=1.0, loss::AbstractLoss=TukeyLoss(),
                                 scale::Union{Nothing,Real}=nothing,
                                 max_iter::Int=50, rtol::Float64=1e-5)
    pts, norm_pts, T, sigma_norm = _prepare_and_normalize(points, sigma)
    prob = ConicTaubinProblem(norm_pts, sigma_norm)
    result = robust_solve(prob, MEstimator(loss); scale=_scale_estimator(scale), max_iter, rtol)
    _finalize_conic_result(result, T, pts, sigma)
end
