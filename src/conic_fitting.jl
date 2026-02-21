# =============================================================================
# Conic Fitting Estimators
# =============================================================================
#
# Conic fitting methods from algebraic LS through Robust Taubin→FNS:
#
#   fit_conic_als              - Algebraic Least Squares (smallest eigenvector of M)
#   fit_conic_taubin           - Gradient-weighted LS (Taubin's method)
#   fit_conic_fns              - Fundamental Numerical Scheme (bias-corrected)
#   fit_conic_robust_taubin    - Taubin + IRLS via robust_solve
#   fit_conic_robust_fns       - FNS + IRLS via robust_solve
#   fit_conic_robust_taubin_fns- Two-phase: Robust Taubin → Robust FNS
#   fit_conic_gnc_fns          - GNC + FNS via robust_solve
#   fit_conic_lifted_fns       - Lifted FNS via half-quadratic splitting
#   fit_conic_geometric        - Geometric distance minimization (Levenberg-Marquardt)
#
# Data carrier: xi(x,y) = (x^2, xy, y^2, x, y, 1)^T
# Conic equation: theta^T * xi = 0
#
# PLACEMENT: Included from main VisualGeometryCore.jl (NOT from Estimators
# submodule) because it depends on Ellipse and HomEllipseMat from Primitives
# (loaded after Estimators).
#
# =============================================================================

# Dependencies: LinearAlgebra, StaticArrays, VGC (Ellipse, HomEllipseMat, Point2)
#               All estimation types available from parent module

# =============================================================================
# Data Carrier Functions
# =============================================================================

"""
    conic_carrier(x, y) -> SVector{6}

Compute the data carrier (design vector) for a conic at point (x, y):
    xi = (x^2, xy, y^2, x, y, 1)^T
"""
@inline function conic_carrier(x::Real, y::Real)
    T = promote_type(typeof(x), typeof(y))
    SVector{6,T}(x*x, x*y, y*y, x, y, one(T))
end

@inline conic_carrier(p::AbstractVector) = conic_carrier(p[1], p[2])

"""
    conic_carrier_jacobian(x, y) -> SMatrix{6,2}

Jacobian of the data carrier with respect to (x, y):
    J = d(xi)/d(x,y)
"""
@inline function conic_carrier_jacobian(x::Real, y::Real)
    T = float(promote_type(typeof(x), typeof(y)))
    z = zero(T)
    SMatrix{6,2,T}(
        2*T(x), T(y), z,      one(T), z,      z,   # column 1: d/dx
        z,      T(x), 2*T(y), z,      one(T), z    # column 2: d/dy
    )
end

@inline conic_carrier_jacobian(p::AbstractVector) = conic_carrier_jacobian(p[1], p[2])

"""
    conic_carrier_covariance(x, y, sigma_sq) -> SMatrix{6,6}

Covariance of the data carrier under isotropic noise with variance sigma^2:
    Lambda = sigma^2 * J * J^T
"""
@inline function conic_carrier_covariance(x::Real, y::Real, sigma_sq::Real)
    J = conic_carrier_jacobian(x, y)
    sigma_sq * (J * J')
end

@inline conic_carrier_covariance(p::AbstractVector, sigma_sq::Real) =
    conic_carrier_covariance(p[1], p[2], sigma_sq)

# =============================================================================
# Distance Functions
# =============================================================================

"""
    sampson_distance_sq(theta, xi, Lambda) -> Real

Squared Sampson distance: (theta^T xi)^2 / (theta^T Lambda theta).
"""
@inline function sampson_distance_sq(theta::AbstractVector, xi::AbstractVector,
                                     Lambda::AbstractMatrix)
    e = dot(theta, xi)
    s2 = dot(theta, Lambda * theta)
    e^2 / max(s2, eps(float(eltype(theta))))
end

"""
    sampson_distance(theta, xi, Lambda) -> Real

Signed Sampson distance: (theta^T xi) / sqrt(theta^T Lambda theta).
"""
@inline function sampson_distance(theta::AbstractVector, xi::AbstractVector,
                                  Lambda::AbstractMatrix)
    e = dot(theta, xi)
    s2 = dot(theta, Lambda * theta)
    e / sqrt(max(s2, eps(float(eltype(theta)))))
end

# =============================================================================
# Result Types
# =============================================================================

"""
    ConicFitResult{T}

Type alias for `Attributed{SVector{6,T}, RobustAttributes{T}}`.

Conic fitting results use `RobustAttributes` from the generic framework.
Access via property forwarding: `r.value` (SVector{6}), `r.residuals`,
`r.weights`, `r.scale`, `r.iterations`, `r.stop_reason`, `r.converged`.
"""
const ConicFitResult{T} = Attributed{SVector{6,T}, RobustAttributes{T}}

function Base.show(io::IO, r::ConicFitResult{T}) where {T}
    n_inliers = count(>(0.5), r.weights)
    n_total = length(r.weights)
    status = r.converged ? "converged" : string(r.stop_reason)
    print(io, "ConicFitResult{$T}($n_inliers/$n_total inliers, $(r.iterations) iter, $status)")
end

# =============================================================================
# Conic ↔ Ellipse Conversion
# =============================================================================

"""
    conic_to_ellipse(theta::AbstractVector) -> Ellipse

Convert a 6-vector conic representation to an Ellipse.

The conic equation is: theta[1]*x^2 + theta[2]*xy + theta[3]*y^2 + theta[4]*x + theta[5]*y + theta[6] = 0
"""
function conic_to_ellipse(theta::AbstractVector{T}) where {T<:Real}
    A, B, C, D, E, F = theta
    Q = HomEllipseMat{T}((A, B/2, D/2, B/2, C, E/2, D/2, E/2, F))
    Ellipse(Q)
end

# =============================================================================
# Input & Normalization Helpers
# =============================================================================

function _prepare_points(points::AbstractMatrix{T}) where {T<:Real}
    if size(points, 1) == 2
        return [SVector{2,Float64}(points[1,i], points[2,i]) for i in 1:size(points, 2)]
    elseif size(points, 2) == 2
        return [SVector{2,Float64}(points[i,1], points[i,2]) for i in 1:size(points, 1)]
    else
        error("Points matrix must be 2×N or N×2, got $(size(points))")
    end
end

function _prepare_points(points::AbstractVector{<:AbstractVector})
    [SVector{2,Float64}(p[1], p[2]) for p in points]
end

"""
    _normalize_points(pts) -> (normalized_pts, T)

Hartley normalization: translate centroid to origin, scale so average distance
from origin is sqrt(2). Returns normalized points and 3×3 homogeneous transform T.
"""
function _normalize_points(pts::AbstractVector{<:SVector{2}})
    n = length(pts)
    mx = sum(p -> p[1], pts) / n
    my = sum(p -> p[2], pts) / n
    avg_dist = sum(p -> sqrt((p[1] - mx)^2 + (p[2] - my)^2), pts) / n
    s = sqrt(2.0) / max(avg_dist, eps())
    T = @SMatrix [s 0.0 -s*mx; 0.0 s -s*my; 0.0 0.0 1.0]
    norm_pts = [SVector{2,Float64}(s*(p[1]-mx), s*(p[2]-my)) for p in pts]
    norm_pts, T
end

"""
    _prepare_and_normalize(points, sigma) -> (pts, norm_pts, T, sigma_norm)

Combined preparation and normalization. Used by all conic fit functions.
"""
function _prepare_and_normalize(points, sigma::Real)
    pts = _prepare_points(points)
    norm_pts, T = _normalize_points(pts)
    sigma_norm = sigma * T[1,1]
    (pts, norm_pts, T, sigma_norm)
end

"""
    _denormalize_conic(theta_norm, T) -> SVector{6}

Transform conic parameters from normalized to original coordinates.
If T maps original→normalized, the conic matrix transforms as:
    Q_orig = T' * Q_norm * T
"""
function _denormalize_conic(theta_norm::SVector{6,Float64}, T::SMatrix{3,3,Float64,9})
    A, B, C, D, E, F = theta_norm
    Q_norm = @SMatrix [A B/2 D/2; B/2 C E/2; D/2 E/2 F]
    Q_orig = T' * Q_norm * T
    # Extract 6-vector: (A, B, C, D, E, F) with B=2*Q[1,2], D=2*Q[1,3], E=2*Q[2,3]
    theta_orig = SVector{6,Float64}(Q_orig[1,1], 2*Q_orig[1,2], Q_orig[2,2],
                                    2*Q_orig[1,3], 2*Q_orig[2,3], Q_orig[3,3])
    theta_orig / norm(theta_orig)
end

function _build_carriers(pts)
    n = length(pts)
    xis = Vector{SVector{6,Float64}}(undef, n)
    @inbounds for i in 1:n
        xis[i] = conic_carrier(pts[i])
    end
    xis
end

function _build_covariances(pts, sigma_sq::Real)
    n = length(pts)
    Lambdas = Vector{SMatrix{6,6,Float64,36}}(undef, n)
    @inbounds for i in 1:n
        Lambdas[i] = conic_carrier_covariance(pts[i], sigma_sq)
    end
    Lambdas
end

function _solve_smallest_eigvec(M::AbstractMatrix)
    eig = eigen(Symmetric(Matrix(M)))
    v = eig.vectors[:, 1]
    SVector{6,Float64}(v / norm(v))
end

const _CONIC_DOF = 5  # 6 params - 1 for homogeneous normalization

"""
    _taubin_seed(xis) -> SVector{6}

Unweighted Taubin solve: build M = Σ ξᵢξᵢᵀ, N = Σ JᵢJᵢᵀ, solve smallest GEP(M, N).
Shared by `ConicTaubinProblem.initial_solve` and `ConicFNSProblem.initial_solve`.
"""
function _taubin_seed(xis::Vector{SVector{6,Float64}})
    M = zeros(SMatrix{6,6,Float64,36})
    N = zeros(SMatrix{6,6,Float64,36})
    @inbounds for i in 1:length(xis)
        M += xis[i] * xis[i]'
        J = conic_carrier_jacobian(xis[i][4], xis[i][5])
        N += J * J'
    end
    _solve_smallest_gep(M, N)
end

# =============================================================================
# Conic Problem Types (AbstractRobustProblem implementations)
# =============================================================================

"""
    ConicTaubinProblem <: AbstractRobustProblem

Robust Taubin problem: IRLS-weighted generalized eigenvalue problem
using Taubin's gradient-weighted scatter matrices.

Taubin's eigenproblem `M*θ = λ*N*θ` always produces a structurally
valid conic, making it robust as an initialization strategy.
"""
struct ConicTaubinProblem <: AbstractRobustProblem
    xis::Vector{SVector{6,Float64}}
    Lambdas::Vector{SMatrix{6,6,Float64,36}}
    Js::Vector{SMatrix{6,2,Float64,12}}
end

function ConicTaubinProblem(pts::AbstractVector{<:SVector{2}}, sigma::Real)
    xis = _build_carriers(pts)
    Lambdas = _build_covariances(pts, sigma^2)
    Js = [conic_carrier_jacobian(p) for p in pts]
    ConicTaubinProblem(xis, Lambdas, Js)
end

initial_solve(prob::ConicTaubinProblem) = _taubin_seed(prob.xis)

compute_residuals(prob::ConicTaubinProblem, θ) =
    _compute_sampson_residuals(sampson_distance, θ, prob.xis, prob.Lambdas)

function weighted_solve(prob::ConicTaubinProblem, θ, ω)
    M = zeros(SMatrix{6,6,Float64,36})
    N = zeros(SMatrix{6,6,Float64,36})
    @inbounds for i in 1:length(prob.xis)
        M += ω[i] * (prob.xis[i] * prob.xis[i]')
        N += ω[i] * (prob.Js[i] * prob.Js[i]')
    end
    _solve_smallest_gep(M, N)
end

data_size(prob::ConicTaubinProblem) = length(prob.xis)
problem_dof(::ConicTaubinProblem) = _CONIC_DOF
convergence_metric(::ConicTaubinProblem, θ_new, θ_old) =
    _convergence_angle(θ_new, θ_old)

"""
    ConicFNSProblem <: AbstractRobustProblem

Robust FNS problem: IRLS-weighted FNS with bias correction.

The FNS bias correction `v_i = 1/(θᵀΛ_iθ)` gives an asymptotically
optimal estimate by accounting for carrier covariance.
"""
struct ConicFNSProblem <: AbstractRobustProblem
    xis::Vector{SVector{6,Float64}}
    Lambdas::Vector{SMatrix{6,6,Float64,36}}
end

function ConicFNSProblem(pts::AbstractVector{<:SVector{2}}, sigma::Real)
    ConicFNSProblem(_build_carriers(pts), _build_covariances(pts, sigma^2))
end

function initial_solve(prob::ConicFNSProblem)
    theta = _taubin_seed(prob.xis)

    # A few FNS iterations for bias correction
    for _ in 1:5
        theta_old = theta
        M = zeros(SMatrix{6,6,Float64,36})
        N = zeros(SMatrix{6,6,Float64,36})
        @inbounds for i in 1:length(prob.xis)
            s2 = dot(theta, prob.Lambdas[i] * theta)
            v = 1.0 / max(s2, eps())
            M += v * (prob.xis[i] * prob.xis[i]')
            N += v * prob.Lambdas[i]
        end
        theta = _solve_smallest_gep(M - N, M)
        _convergence_angle(theta, theta_old) < 1e-10 && break
    end
    theta
end

compute_residuals(prob::ConicFNSProblem, θ) =
    _compute_sampson_residuals(sampson_distance, θ, prob.xis, prob.Lambdas)

function weighted_solve(prob::ConicFNSProblem, θ, ω)
    M = zeros(SMatrix{6,6,Float64,36})
    N = zeros(SMatrix{6,6,Float64,36})
    @inbounds for i in 1:length(prob.xis)
        s2 = dot(θ, prob.Lambdas[i] * θ)
        v = 1.0 / max(s2, eps())
        w = ω[i] * v
        M += w * (prob.xis[i] * prob.xis[i]')
        N += w * prob.Lambdas[i]
    end
    _solve_smallest_gep(M - N, M)
end

data_size(prob::ConicFNSProblem) = length(prob.xis)
problem_dof(::ConicFNSProblem) = _CONIC_DOF
convergence_metric(::ConicFNSProblem, θ_new, θ_old) =
    _convergence_angle(θ_new, θ_old)

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

"""
    _finalize_geometric(p, pts, sigma, stop_reason, iter) -> ConicFitResult

Convert 5-param ellipse vector to conic θ, compute Sampson residuals, package result.
Used by geometric distance fitting exit paths.
"""
function _finalize_geometric(p, pts::Vector{SVector{2,Float64}}, sigma::Real,
                             stop_reason::Symbol, iter::Int)
    cx, cy, a, b, phi = _ellipse_params_from_5(p)
    ell = Ellipse(Point2(cx, cy), a, b, phi)
    Q = HomEllipseMat(ell)
    theta = SVector{6,Float64}(Q[1,1], 2*Q[1,2], Q[2,2], 2*Q[1,3], 2*Q[2,3], Q[3,3])
    theta = theta / norm(theta)
    xis = _build_carriers(pts)
    Lambdas = _build_covariances(pts, sigma^2)
    residuals = _compute_sampson_residuals(sampson_distance, theta, xis, Lambdas)
    n = length(pts)
    Attributed(theta, RobustAttributes(stop_reason, residuals, ones(n), Float64(sigma), iter))
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

# =============================================================================
# 2. Taubin's Method
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

# =============================================================================
# 2b. Robust Taubin (via generic robust_solve)
# =============================================================================

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

# =============================================================================
# 3. FNS (Fundamental Numerical Scheme)
# =============================================================================

"""
    fit_conic_fns(points; sigma=1.0, max_iter=20, rtol=1e-10) -> ConicFitResult

Fundamental Numerical Scheme: iteratively bias-corrected Sampson distance
minimization. Typically converges in 3-5 iterations.

Uses the framework: `robust_solve` with `L2Loss` (unit weights) reduces the
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
    result = robust_solve(prob, MEstimator(L2Loss());
                          init=_taubin_seed(prob.xis),
                          scale=FixedScale(σ=1.0), max_iter, rtol)
    _finalize_conic_result(result, T, pts, sigma)
end

# =============================================================================
# 4. Robust FNS (via generic robust_solve)
# =============================================================================

"""
    fit_conic_robust_fns(points; sigma=1.0, loss=TukeyLoss(), scale=nothing,
                         max_iter=50, rtol=1e-5) -> ConicFitResult

Robust FNS with IRLS: combines FNS bias correction with robust weighting
via a redescending M-estimator.

Uses the generic `robust_solve` with `ConicFNSProblem`.

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
    result = robust_solve(prob, MEstimator(loss); scale=_scale_estimator(scale), max_iter, rtol)
    _finalize_conic_result(result, T, pts, sigma)
end

# =============================================================================
# 4a. Robust Taubin → FNS (Two-Phase, via refine kwarg)
# =============================================================================

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
    result = robust_solve(prob_taubin, MEstimator(loss);
                          scale=_scale_estimator(scale), max_iter=max_iter_taubin, rtol,
                          refine=prob_fns, refine_max_iter=max_iter_fns)
    _finalize_conic_result(result, T, pts, sigma)
end

# =============================================================================
# 4b. GNC-FNS (via generic robust_solve with GNCEstimator)
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

# =============================================================================
# 5. Lifted FNS (Half-Quadratic Splitting)
# =============================================================================

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

# =============================================================================
# 6. Geometric Distance Fitting (Levenberg-Marquardt)
# =============================================================================

"""
    _closest_point_on_ellipse(cx, cy, a, b, phi, px, py) -> (cpx, cpy)

Find the closest point on an ellipse to (px, py) using the Eberly bisection
algorithm (robust, guaranteed convergence).

Reference: D. Eberly, "Distance from a Point to an Ellipse, an Ellipsoid, or a
Hyperellipsoid", Geometric Tools.
"""
function _closest_point_on_ellipse(cx, cy, a, b, phi, px, py)
    cos_phi, sin_phi = cos(phi), sin(phi)
    dx = px - cx
    dy = py - cy
    u = cos_phi * dx + sin_phi * dy
    v = -sin_phi * dx + cos_phi * dy

    # Ensure e0 >= e1 (largest semi-axis first)
    if a >= b
        e0, e1, y0, y1 = a, b, abs(u), abs(v)
    else
        e0, e1, y0, y1 = b, a, abs(v), abs(u)
    end

    x0, x1 = _eberly_bisect_2d(e0, e1, y0, y1)

    # Restore signs
    if a >= b
        x_local = copysign(x0, u)
        y_local = copysign(x1, v)
    else
        x_local = copysign(x1, u)
        y_local = copysign(x0, v)
    end

    cpx = cx + cos_phi * x_local - sin_phi * y_local
    cpy = cy + sin_phi * x_local + cos_phi * y_local
    cpx, cpy
end

"""
    _eberly_bisect_2d(e0, e1, y0, y1) -> (x0, x1)

Eberly's bisection for point-to-ellipse distance in axis-aligned first-quadrant
coordinates. e0 >= e1 > 0, y0 >= 0, y1 >= 0.
"""
function _eberly_bisect_2d(e0, e1, y0, y1)
    if y1 > 0
        if y0 > 0
            # General case: both coordinates nonzero
            z0 = y0 / e0
            z1 = y1 / e1
            g = z0^2 + z1^2 - 1.0
            if g != 0.0
                r0 = (e0 / e1)^2
                sbar = _eberly_find_root(r0, z0, z1, g)
                x0 = r0 * y0 / (sbar + r0)
                x1 = y1 / (sbar + 1.0)
            else
                x0 = y0
                x1 = y1
            end
        else
            # y0 == 0: closest point is on the y-axis portion of the ellipse
            x0 = 0.0
            x1 = e1
        end
    else
        # y1 == 0: closest point is on the x-axis
        numer0 = e0 * y0
        denom0 = e0^2 - e1^2
        if numer0 < denom0
            xde0 = numer0 / denom0
            x0 = e0 * xde0
            x1 = e1 * sqrt(max(0.0, 1.0 - xde0^2))
        else
            x0 = e0
            x1 = 0.0
        end
    end
    x0, x1
end

"""
    _eberly_find_root(r0, z0, z1, g) -> s

Find the unique root s > -1 of
    g(s) = (r0*z0/(s+r0))^2 + (z1/(s+1))^2 - 1 = 0
via bisection, where r0 = (e0/e1)^2 >= 1.
"""
function _eberly_find_root(r0, z0, z1, g)
    n0 = r0 * z0
    s0 = z1 - 1.0
    if g < 0.0
        s1 = 0.0
    else
        s1 = sqrt(n0^2 + z1^2) - 1.0
    end
    s = 0.0
    for _ in 1:100
        s = 0.5 * (s0 + s1)
        (s == s0 || s == s1) && break
        g = (n0 / (s + r0))^2 + (z1 / (s + 1.0))^2 - 1.0
        if g > 0.0
            s0 = s
        elseif g < 0.0
            s1 = s
        else
            break
        end
    end
    s
end

@inline _ellipse_params_from_5(p) = (p[1], p[2], abs(p[3]), abs(p[4]), p[5])

"""
    fit_conic_geometric(points; sigma=1.0, max_iter=100, rtol=1e-10) -> ConicFitResult

Levenberg-Marquardt minimization of geometric (orthogonal) distance.
Gives the ML estimate under isotropic Gaussian noise but is expensive
and not robust to outliers.

# Arguments
- `points`: 2×N matrix, N×2 matrix, or Vector of 2D points
- `sigma::Real=1.0`: Noise standard deviation
- `max_iter::Int=100`: Maximum LM iterations
- `rtol::Float64=1e-10`: Convergence tolerance
"""
function fit_conic_geometric(points; sigma::Real=1.0, max_iter::Int=100, rtol::Float64=1e-10)
    pts = _prepare_points(points)
    n = length(pts)

    # Initialize from FNS
    result0 = fit_conic_fns(points; sigma=sigma)
    theta0 = result0.value

    local e0
    try
        e0 = conic_to_ellipse(theta0)
    catch
        cx = sum(p -> p[1], pts) / n
        cy = sum(p -> p[2], pts) / n
        r = sqrt(sum(p -> (p[1]-cx)^2 + (p[2]-cy)^2, pts) / n)
        e0 = Ellipse(Point2(cx, cy), r, r, 0.0)
    end
    p = [Float64(e0.center[1]), Float64(e0.center[2]),
         Float64(e0.a), Float64(e0.b), Float64(e0.θ)]

    lambda = 1e-3
    nu = 2.0

    function compute_geo_residuals(p)
        cx, cy, a, b, phi = _ellipse_params_from_5(p)
        r = Vector{Float64}(undef, n)
        @inbounds for i in 1:n
            cpx, cpy = _closest_point_on_ellipse(cx, cy, a, b, phi, pts[i][1], pts[i][2])
            r[i] = sqrt((pts[i][1] - cpx)^2 + (pts[i][2] - cpy)^2)
            dx_c = pts[i][1] - cx
            dy_c = pts[i][2] - cy
            cos_phi, sin_phi = cos(phi), sin(phi)
            u = cos_phi * dx_c + sin_phi * dy_c
            v = -sin_phi * dx_c + cos_phi * dy_c
            if (u/max(a,eps()))^2 + (v/max(b,eps()))^2 < 1.0
                r[i] = -r[i]
            end
        end
        r
    end

    function compute_jacobian(p, r)
        J = Matrix{Float64}(undef, n, 5)
        dp = 1e-7
        for j in 1:5
            p_plus = copy(p)
            p_plus[j] += dp
            r_plus = compute_geo_residuals(p_plus)
            J[:, j] = (r_plus - r) / dp
        end
        J
    end

    r = compute_geo_residuals(p)
    cost = sum(abs2, r)

    for iter in 1:max_iter
        J = compute_jacobian(p, r)
        JtJ = J' * J
        Jtr = J' * r

        H = JtJ + lambda * Diagonal(max.(diag(JtJ), 1e-10))
        dp = -(H \ Jtr)

        p_new = p + dp
        r_new = compute_geo_residuals(p_new)
        cost_new = sum(abs2, r_new)

        if cost_new < cost
            p = p_new
            r = r_new
            cost = cost_new
            lambda = max(lambda / 3.0, 1e-10)
            nu = 2.0
        else
            lambda = min(lambda * nu, 1e10)
            nu *= 2.0
        end

        rel_change = norm(dp) / (norm(p) + 1e-10)
        if rel_change < rtol
            return _finalize_geometric(p, pts, sigma, :converged, iter)
        end
    end

    _finalize_geometric(p, pts, sigma, :max_iterations, max_iter)
end
