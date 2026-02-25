# =============================================================================
# Conic Fitting — GNC-FNS
# =============================================================================

"""
    fit_conic_gnc_fns(points; sigma=1.0, gnc_type=GNCTruncatedLS, c=1.0,
                      scale=nothing, mu_factor=1.4, max_iter=100) -> ConicFitResult

GNC-FNS: Graduated Non-Convexity wrapping FNS, following Yang et al. (2020).

Uses the generic `robust_solve` with `ConicFNSProblem` and `GNCEstimator`.

# Arguments
- `points`: 2×N matrix, N×2 matrix, or Vector of 2D points
- `sigma::Real=1.0`: Noise standard deviation
- `gnc_type::Type{<:GNCLoss}=GNCTruncatedLS`: GNC loss type
- `c::Real=1.0`: Inlier threshold (in units of scale)
- `scale::Union{Nothing,Real}=nothing`: Residual scale. `nothing` → MAD estimation,
  a number → fixed scale.
- `mu_factor::Real=1.4`: Annealing rate (μ *= mu_factor each iteration)
- `max_iter::Int=100`: Maximum GNC iterations
"""
function fit_conic_gnc_fns(points; sigma::Real=1.0,
                           gnc_type::Type{G}=GNCTruncatedLS,
                           c::Real=1.0, scale::Union{Nothing,Real}=nothing,
                           mu_factor::Real=1.4,
                           max_iter::Int=100) where {G<:GNCLoss}
    pts, norm_pts, T, sigma_norm = _prepare_and_normalize(points, sigma)
    prob = ConicFNSProblem(norm_pts, sigma_norm)
    est = GNCEstimator(G; c=Float64(c), μ_factor=Float64(mu_factor))
    result = robust_solve(prob, est; scale=_scale_estimator(scale), max_iter)
    _finalize_conic_result(result, T, pts, sigma)
end
