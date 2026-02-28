# =============================================================================
# Conic Fitting — FNS, Robust FNS, Robust Taubin→FNS, Lifted FNS
# =============================================================================

"""
    fit_conic_fns(points; sigma=1.0, max_iter=20, rtol=1e-10) -> ConicFitResult

Fundamental Numerical Scheme: iteratively bias-corrected Sampson distance
minimization. Typically converges in 3-5 iterations.

Uses the framework: `fit` with `L2Loss` (unit weights) reduces the
IRLS loop to pure FNS iteration.

# Arguments
- `points`: 2×N matrix, N×2 matrix, or Vector of 2D points
- `sigma::Real=1.0`: Noise standard deviation
- `max_iter::Int=20`: Maximum iterations
- `rtol::Float64=1e-10`: Convergence tolerance (angle between iterates)
"""
function fit_conic_fns(points; sigma::Real=1.0, max_iter::Int=20, rtol::Float64=1e-10)
    pts, norm_pts, T, sigma_norm = _prepare_and_normalize(points, sigma)
    prob = ConicFNSProblem(norm_pts, sigma_norm)
    # L2Loss gives unit weights → IRLS loop = pure FNS iteration.
    # init=_taubin_seed bypasses the 5-iteration FNS warmup in initial_solve.
    result = fit(prob, MEstimator(L2Loss());
                          init=_taubin_seed(prob.xis),
                          scale=FixedScale(σ=1.0), max_iter, rtol)
    _finalize_conic_result(result, T, pts, sigma)
end

"""
    fit_conic_robust_fns(points; sigma=1.0, loss=TukeyLoss(), scale=nothing,
                         max_iter=50, rtol=1e-5) -> ConicFitResult

Robust FNS with IRLS: combines FNS bias correction with robust weighting
via a redescending M-estimator.

Uses the generic `fit` with `ConicFNSProblem`.

# Arguments
- `points`: 2×N matrix, N×2 matrix, or Vector of 2D points
- `sigma::Real=1.0`: Noise standard deviation (for carrier covariances)
- `loss::AbstractLoss=TukeyLoss()`: Robust loss function
- `scale::Union{Nothing,Real}=nothing`: Residual scale. `nothing` → MAD estimation,
  a number → fixed scale.
- `max_iter::Int=50`: Maximum IRLS iterations
- `rtol::Float64=1e-5`: Convergence tolerance
"""
function fit_conic_robust_fns(points; sigma::Real=1.0, loss::AbstractLoss=TukeyLoss(),
                              scale::Union{Nothing,Real}=nothing,
                              max_iter::Int=50, rtol::Float64=1e-5)
    pts, norm_pts, T, sigma_norm = _prepare_and_normalize(points, sigma)
    prob = ConicFNSProblem(norm_pts, sigma_norm)
    result = fit(prob, MEstimator(loss); scale=_scale_estimator(scale), max_iter, rtol)
    _finalize_conic_result(result, T, pts, sigma)
end

"""
    fit_conic_robust_taubin_fns(points; sigma=1.0, loss=GemanMcClureLoss(),
                                scale=nothing, max_iter_taubin=20,
                                max_iter_fns=30, rtol=1e-5) -> ConicFitResult

Two-phase robust conic fitting: Robust Taubin initialization followed by
Robust FNS refinement.

Phase 1 (Robust Taubin) provides a stable initialization — Taubin's gradient-
weighted eigenproblem always produces a valid conic, and IRLS weighting rejects
outliers. Phase 2 (Robust FNS) adds bias correction via `v_i = 1/(θᵀΛ_iθ)`,
improving accuracy while inheriting the good starting point.

This combination achieves **zero failures** at all outlier rates up to 70%,
unlike Robust FNS alone which fails catastrophically when its FNS initialization
is corrupted. Geman-McClure loss is the default because its smooth `1/r⁴` decay
provides better weight discrimination than Tukey's hard cutoff when MAD scale
is inflated at high outlier rates.

# Arguments
- `points`: 2×N matrix, N×2 matrix, or Vector of 2D points
- `sigma::Real=1.0`: Noise standard deviation (for carrier covariances)
- `loss::AbstractLoss=GemanMcClureLoss()`: Robust loss function
- `scale::Union{Nothing,Real}=nothing`: Residual scale. `nothing` → MAD estimation,
  a number → fixed scale.
- `max_iter_taubin::Int=20`: Maximum Robust Taubin iterations (phase 1)
- `max_iter_fns::Int=30`: Maximum Robust FNS iterations (phase 2)
- `rtol::Float64=1e-5`: Convergence tolerance (angle between iterates)
"""
function fit_conic_robust_taubin_fns(points; sigma::Real=1.0,
                                     loss::AbstractLoss=GemanMcClureLoss(),
                                     scale::Union{Nothing,Real}=nothing,
                                     max_iter_taubin::Int=20,
                                     max_iter_fns::Int=30,
                                     rtol::Float64=1e-5)
    pts, norm_pts, T, sigma_norm = _prepare_and_normalize(points, sigma)
    prob_taubin = ConicTaubinProblem(norm_pts, sigma_norm)
    prob_fns = ConicFNSProblem(norm_pts, sigma_norm)
    result = fit(prob_taubin, MEstimator(loss);
                          scale=_scale_estimator(scale), max_iter=max_iter_taubin, rtol,
                          refine=prob_fns, refine_max_iter=max_iter_fns)
    _finalize_conic_result(result, T, pts, sigma)
end

"""
    fit_conic_lifted_fns(points; sigma=1.0, c=4.685, scale=nothing,
                         max_iter=50, rtol=1e-5) -> ConicFitResult

Lifted FNS via half-quadratic splitting with coupling correction.

Unlike standard IRLS, the lifted formulation has no flat plateau: points
can be reclassified from outlier to inlier across iterations. The coupling
correction delta_i accounts for dw_i/dtheta, giving faster convergence.

# Arguments
- `points`: 2×N matrix, N×2 matrix, or Vector of 2D points
- `sigma::Real=1.0`: Noise standard deviation
- `c::Real=4.685`: Tuning constant for scale (tau = c * s)
- `scale::Union{Nothing,Real}=nothing`: Residual scale. `nothing` → MAD estimation,
  a number → fixed scale. When sigma is known, pass `scale=1.0`.
- `max_iter::Int=50`: Maximum iterations
- `rtol::Float64=1e-5`: Convergence tolerance
"""
function fit_conic_lifted_fns(points; sigma::Real=1.0, c::Real=4.685,
                              scale::Union{Nothing,Real}=nothing,
                              max_iter::Int=50, rtol::Float64=1e-5)
    pts, norm_pts, T, sigma_norm = _prepare_and_normalize(points, sigma)
    n = length(pts)

    xis = _build_carriers(norm_pts)
    Lambdas = _build_covariances(norm_pts, sigma_norm^2)

    # Initialize with FNS (Taubin seed + FNS bias correction)
    theta = initial_solve(ConicFNSProblem(xis, Lambdas))

    w = ones(n)
    local s_scale::Float64

    for iter in 1:max_iter
        theta_old = theta

        # (a) Signed Sampson distances and squared distances
        d = _compute_sampson_residuals(sampson_distance, theta, xis, Lambdas)
        d2 = d .* d

        # (b) Scale and threshold (use signed residuals for MAD)
        s_scale = if scale === nothing
            _corrected_scale(MADScale(), d, n, _CONIC_DOF)
        else
            Float64(scale)
        end
        tau = c * s_scale
        tau2 = tau^2

        # (c) Auxiliary weights: w_i = sqrt(max(0, 1 - d_i^2/tau^2))
        @inbounds for i in 1:n
            w[i] = sqrt(max(0.0, 1.0 - d2[i] / tau2))
        end

        # (d-h) Form modified scatter matrices with coupling correction
        M_tilde = zeros(SMatrix{6,6,Float64,36})
        N_tilde = zeros(SMatrix{6,6,Float64,36})
        @inbounds for i in 1:n
            s2 = dot(theta, Lambdas[i] * theta)
            v_hat = w[i]^2 / max(s2, eps())

            denom = d2[i] + tau2 * (3.0 * w[i]^2 - 1.0)
            if abs(denom) > eps() && d2[i] > eps()
                delta = 4.0 * d2[i] / denom
                if (1.0 - delta) <= eps()
                    delta = 0.0
                end
            else
                delta = 0.0
            end

            factor = v_hat * (1.0 - delta)
            M_tilde += factor * (xis[i] * xis[i]')
            N_tilde += factor * Lambdas[i]
        end

        # (i) Solve (M_tilde - N_tilde) theta = lambda * M_tilde * theta
        theta = _solve_smallest_gep(M_tilde - N_tilde, M_tilde)

        if _convergence_angle(theta, theta_old) < rtol
            theta_orig = _denormalize_conic(theta, T)
            xis_orig = _build_carriers(pts)
            Lambdas_orig = _build_covariances(pts, sigma^2)
            residuals = _compute_sampson_residuals(sampson_distance, theta_orig, xis_orig, Lambdas_orig)
            return Attributed(theta_orig, RobustAttributes(:converged, residuals, copy(w), s_scale, iter))
        end
    end

    theta_orig = _denormalize_conic(theta, T)
    xis_orig = _build_carriers(pts)
    Lambdas_orig = _build_covariances(pts, sigma^2)
    residuals = _compute_sampson_residuals(sampson_distance, theta_orig, xis_orig, Lambdas_orig)
    Attributed(theta_orig, RobustAttributes(:max_iterations, residuals, copy(w), s_scale, max_iter))
end
