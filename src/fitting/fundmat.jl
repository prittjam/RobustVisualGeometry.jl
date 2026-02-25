# =============================================================================
# Fundamental Matrix Fitting Estimators
# =============================================================================
#
# Robust fundamental matrix estimation via Taubin→FNS, mirroring the conic
# fitting pipeline in conic_fitting.jl.
#
#   fit_fundmat                   - RANSAC (composable quality + LO)
#   fit_fundmat_robust_taubin     - Taubin + IRLS via robust_solve
#   fit_fundmat_robust_fns        - FNS + IRLS via robust_solve
#   fit_fundmat_robust_taubin_fns - Two-phase: Robust Taubin → Robust FNS
#
# Epipolar constraint: dᵀ F s = 0  where s = T₁[u₁; 1], d = T₂[u₂; 1]
# Carrier: ξᵢ = kron(dᵢ, sᵢ)  (9-vec from Hartley-normalized points)
# So fᵀξᵢ = 0 where f = vec_rowmajor(F_norm).
#
# Carrier Jacobian is w.r.t. ORIGINAL (pixel) coordinates via the chain rule
# through Hartley normalization transforms T₁, T₂. This means the covariance
# Λᵢ = σ² JᵢJᵢᵀ uses σ in original pixel units (not σ_norm).
#
# PLACEMENT: Included from main VisualGeometryCore.jl (NOT from Estimators
# submodule) because it depends on enforce_rank_two, sampson_distance, etc.
#
# =============================================================================

# Dependencies: LinearAlgebra (norm, dot), StaticArrays
#               VGC (hartley_normalization, enforce_rank_two, sampson_distance, etc.)
#               All estimation types available from parent module

# =============================================================================
# Constants
# =============================================================================

const _FMAT_DOF = 7  # 9 params - 1 scale - 1 rank constraint

# =============================================================================
# Carrier and Jacobian
# =============================================================================

"""
    _fmat_carrier(s, d) -> SVector{9}

Row-major Kronecker product kron(d, s) for the epipolar constraint.
s = [u₁; 1], d = [u₂; 1] are homogeneous source/destination points.
"""
@inline function _fmat_carrier(s::SVector{3}, d::SVector{3})
    SVector{9,Float64}(
        d[1]*s[1], d[1]*s[2], d[1]*s[3],
        d[2]*s[1], d[2]*s[2], d[2]*s[3],
        d[3]*s[1], d[3]*s[2], d[3]*s[3]
    )
end

"""
    _fmat_carrier_jacobian(s, d, T1, T2) -> SMatrix{9,4}

Jacobian of ξ = kron(d, s) w.r.t. original coordinates (u₁₁, u₁₂, u₂₁, u₂₂),
where s = T₁[u₁;1] and d = T₂[u₂;1].

Chain rule: ∂s/∂u₁ⱼ = T₁[:,j], ∂d/∂u₂ⱼ = T₂[:,j].

Column 1: ∂ξ/∂u₁₁ = kron(d, T₁[:,1])
Column 2: ∂ξ/∂u₁₂ = kron(d, T₁[:,2])
Column 3: ∂ξ/∂u₂₁ = kron(T₂[:,1], s)
Column 4: ∂ξ/∂u₂₂ = kron(T₂[:,2], s)
"""
@inline function _fmat_carrier_jacobian(s::SVector{3}, d::SVector{3},
                                         T1::SMatrix{3,3,Float64,9},
                                         T2::SMatrix{3,3,Float64,9})
    t1_1 = T1[:,1]  # ∂s/∂u₁₁
    t1_2 = T1[:,2]  # ∂s/∂u₁₂
    t2_1 = T2[:,1]  # ∂d/∂u₂₁
    t2_2 = T2[:,2]  # ∂d/∂u₂₂
    SMatrix{9,4,Float64,36}(
        # col1: kron(d, T₁[:,1])
        d[1]*t1_1[1], d[1]*t1_1[2], d[1]*t1_1[3],
        d[2]*t1_1[1], d[2]*t1_1[2], d[2]*t1_1[3],
        d[3]*t1_1[1], d[3]*t1_1[2], d[3]*t1_1[3],
        # col2: kron(d, T₁[:,2])
        d[1]*t1_2[1], d[1]*t1_2[2], d[1]*t1_2[3],
        d[2]*t1_2[1], d[2]*t1_2[2], d[2]*t1_2[3],
        d[3]*t1_2[1], d[3]*t1_2[2], d[3]*t1_2[3],
        # col3: kron(T₂[:,1], s)
        t2_1[1]*s[1], t2_1[1]*s[2], t2_1[1]*s[3],
        t2_1[2]*s[1], t2_1[2]*s[2], t2_1[2]*s[3],
        t2_1[3]*s[1], t2_1[3]*s[2], t2_1[3]*s[3],
        # col4: kron(T₂[:,2], s)
        t2_2[1]*s[1], t2_2[1]*s[2], t2_2[1]*s[3],
        t2_2[2]*s[1], t2_2[2]*s[2], t2_2[2]*s[3],
        t2_2[3]*s[1], t2_2[3]*s[2], t2_2[3]*s[3]
    )
end

# =============================================================================
# Data Building
# =============================================================================

"""
    _fmat_build_data(u1, u2, T1, T2, sigma) -> (xis, Lambdas, Js)

Build carriers, covariances, and Jacobians for the epipolar constraint.

Points are Hartley-normalized internally (s = T₁[u₁;1], d = T₂[u₂;1]).
Carriers ξ = kron(d, s) live in normalized coordinates.
Jacobians J = ∂ξ/∂u_original include the chain rule through T₁, T₂,
so the covariance Λ = σ² JJᵀ uses σ in original pixel units.
"""
function _fmat_build_data(u1::AbstractVector{SVector{2,Float64}},
                          u2::AbstractVector{SVector{2,Float64}},
                          T1::SMatrix{3,3,Float64,9},
                          T2::SMatrix{3,3,Float64,9},
                          sigma::Float64)
    n = length(u1)
    xis = Vector{SVector{9,Float64}}(undef, n)
    Lambdas = Vector{SMatrix{9,9,Float64,81}}(undef, n)
    Js = Vector{SMatrix{9,4,Float64,36}}(undef, n)
    sigma_sq = sigma^2

    @inbounds for i in 1:n
        s = T1 * SA[u1[i][1], u1[i][2], 1.0]
        d = T2 * SA[u2[i][1], u2[i][2], 1.0]
        xis[i] = _fmat_carrier(s, d)
        J = _fmat_carrier_jacobian(s, d, T1, T2)
        Js[i] = J
        Lambdas[i] = sigma_sq * (J * J')
    end
    (xis, Lambdas, Js)
end

# =============================================================================
# Rank-2 Projection
# =============================================================================

"""
    _fmat_project_rank2(f::SVector{9}) -> SVector{9}

Reshape 9-vec → 3×3, enforce rank-2 via SVD, sign-normalize, flatten back.
"""
function _fmat_project_rank2(f::SVector{9,Float64})
    F = _vec9_to_mat33(f, Float64)
    F2 = enforce_rank_two(F)
    Fn = sign_normalize(F2)
    Fn === nothing && return f
    # Flatten back to row-major 9-vec
    SVector{9,Float64}(
        Fn[1,1], Fn[1,2], Fn[1,3],
        Fn[2,1], Fn[2,2], Fn[2,3],
        Fn[3,1], Fn[3,2], Fn[3,3]
    )
end

# Signed Sampson distance for the fundmat GEP uses the generic
# sampson_distance(theta, xi, Lambda) from VGC — same formula applies
# to any linear constraint: dot(θ, ξ) / √(θᵀΛθ).

# =============================================================================
# Taubin Seed
# =============================================================================

"""
    _fmat_taubin_seed(xis, Lambdas, Js) -> SVector{9}

Unweighted Taubin: M = Σ ξᵢξᵢᵀ, N = Σ JᵢJᵢᵀ, smallest GEP + rank-2.
"""
function _fmat_taubin_seed(xis::Vector{SVector{9,Float64}},
                            Lambdas::Vector{SMatrix{9,9,Float64,81}},
                            Js::Vector{SMatrix{9,4,Float64,36}})
    M = zeros(SMatrix{9,9,Float64,81})
    N = zeros(SMatrix{9,9,Float64,81})
    @inbounds for i in 1:length(xis)
        M += xis[i] * xis[i]'
        N += Js[i] * Js[i]'
    end
    _fmat_project_rank2(_solve_smallest_gep(M, N))
end

# =============================================================================
# FMatTaubinProblem <: AbstractTaubinProblem
# =============================================================================

"""
    FMatTaubinProblem <: AbstractTaubinProblem

Robust Taubin problem for fundamental matrix estimation.
IRLS-weighted GEP using Taubin's gradient-weighted scatter matrices
with rank-2 projection after each solve.
"""
struct FMatTaubinProblem <: AbstractTaubinProblem
    xis::Vector{SVector{9,Float64}}
    Lambdas::Vector{SMatrix{9,9,Float64,81}}
    Js::Vector{SMatrix{9,4,Float64,36}}
end

# --- Dispatch points for AbstractTaubinProblem ---
_project(::FMatTaubinProblem, θ) = _fmat_project_rank2(θ)
_seed(prob::FMatTaubinProblem) = _fmat_taubin_seed(prob.xis, prob.Lambdas, prob.Js)
_sampson_fn(::FMatTaubinProblem) = sampson_distance
problem_dof(::FMatTaubinProblem) = _FMAT_DOF

# =============================================================================
# FMatFNSProblem <: AbstractFNSProblem
# =============================================================================

"""
    FMatFNSProblem <: AbstractFNSProblem

Robust FNS problem for fundamental matrix estimation.
Combines FNS bias correction (vᵢ = 1/(fᵀΛᵢf)) with IRLS weighting
and rank-2 projection.
"""
struct FMatFNSProblem <: AbstractFNSProblem
    xis::Vector{SVector{9,Float64}}
    Lambdas::Vector{SMatrix{9,9,Float64,81}}
    Js::Vector{SMatrix{9,4,Float64,36}}
end

# --- Dispatch points for AbstractFNSProblem ---
_project(::FMatFNSProblem, θ) = _fmat_project_rank2(θ)
_seed(prob::FMatFNSProblem) = _fmat_taubin_seed(prob.xis, prob.Lambdas, prob.Js)
_sampson_fn(::FMatFNSProblem) = sampson_distance
problem_dof(::FMatFNSProblem) = _FMAT_DOF

# =============================================================================
# Result Type
# =============================================================================

"""
    FMatFitResult{T}

Type alias for `Attributed{FundMat{T}, RobustAttributes{T}}`.
"""
const FMatFitResult{T} = Attributed{FundMat{T}, RobustAttributes{T}}

function Base.show(io::IO, r::FMatFitResult{T}) where {T}
    n_inliers = count(>(0.5), r.weights)
    n_total = length(r.weights)
    status = r.converged ? "converged" : string(r.stop_reason)
    print(io, "FMatFitResult{$T}($n_inliers/$n_total inliers, $(r.iterations) iter, $status)")
end

# =============================================================================
# Preparation and Finalization
# =============================================================================

"""
    _fmat_prepare(u1, u2, sigma) -> (T1, T2, xis, Lambdas, Js)

Hartley-normalize both point sets, build carriers/covariances/Jacobians.

The carrier Jacobian is w.r.t. original pixel coordinates (includes chain rule
through T₁, T₂), so `sigma` is in original pixel units — no normalization needed.
"""
function _fmat_prepare(u1::AbstractVector{SVector{2,Float64}},
                       u2::AbstractVector{SVector{2,Float64}},
                       sigma::Float64)
    T1 = hartley_normalization(u1)
    T2 = hartley_normalization(u2)
    xis, Lambdas, Js = _fmat_build_data(u1, u2, T1, T2, sigma)
    (T1, T2, xis, Lambdas, Js)
end

"""
    _finalize_fmat_result(result, T1, T2, u1, u2, sigma) -> FMatFitResult

Unnormalize F, recompute Sampson residuals in original coordinates.
"""
function _finalize_fmat_result(result, T1, T2, u1, u2, sigma)
    F_norm = FundMat{Float64}(Tuple(_vec9_to_mat33(result.value, Float64)))
    F_rank2 = enforce_rank_two(F_norm)
    F = hartley_unnormalize(F_rank2, T1, T2)
    if F === nothing
        F = FundMat{Float64}(Tuple(SMatrix{3,3,Float64,9}(F_rank2) / norm(F_rank2)))
    end
    n = length(u1)
    residuals = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        residuals[i] = sampson_distance(F, u1[i], u2[i])
    end
    Attributed(F, RobustAttributes(result.stop_reason, residuals,
                                   copy(result.weights), result.scale,
                                   result.iterations))
end

# =============================================================================
# Public API
# =============================================================================

"""
    fit_fundmat_robust_taubin(correspondences; sigma=1.0, loss=TukeyLoss(),
        scale=nothing, max_iter=50, rtol=1e-5) -> FMatFitResult

Robust Taubin: IRLS-weighted GEP with Taubin's gradient-weighted scatter
matrices and rank-2 projection.

# Arguments
- `correspondences`: Vector of `Pair{SVector{2,Float64}, SVector{2,Float64}}`
  or any iterable of (source, destination) point pairs
- `sigma::Real=1.0`: Noise standard deviation (pixels)
- `loss::AbstractLoss=TukeyLoss()`: Robust loss function
- `scale::Union{Nothing,Real}=nothing`: Residual scale (`nothing` → MAD)
- `max_iter::Int=50`: Maximum IRLS iterations
- `rtol::Float64=1e-5`: Convergence tolerance
"""
function fit_fundmat_robust_taubin(correspondences;
                                    sigma::Real=1.0,
                                    loss::AbstractLoss=TukeyLoss(),
                                    scale::Union{Nothing,Real}=nothing,
                                    max_iter::Int=50,
                                    rtol::Float64=1e-5)
    u1, u2 = _extract_correspondences(correspondences)
    T1, T2, xis, Lambdas, Js = _fmat_prepare(u1, u2, Float64(sigma))
    prob = FMatTaubinProblem(xis, Lambdas, Js)
    result = robust_solve(prob, MEstimator(loss);
                          scale=_scale_estimator(scale), max_iter, rtol)
    _finalize_fmat_result(result, T1, T2, u1, u2, Float64(sigma))
end

"""
    fit_fundmat_robust_fns(correspondences; sigma=1.0, loss=GemanMcClureLoss(),
        scale=nothing, max_iter=30, rtol=1e-5) -> FMatFitResult

Robust FNS: IRLS with FNS bias correction and rank-2 projection.

# Arguments
- `correspondences`: Vector of point pair correspondences
- `sigma::Real=1.0`: Noise standard deviation (pixels)
- `loss::AbstractLoss=GemanMcClureLoss()`: Robust loss function
- `scale::Union{Nothing,Real}=nothing`: Residual scale (`nothing` → MAD)
- `max_iter::Int=30`: Maximum IRLS iterations
- `rtol::Float64=1e-5`: Convergence tolerance
"""
function fit_fundmat_robust_fns(correspondences;
                                 sigma::Real=1.0,
                                 loss::AbstractLoss=GemanMcClureLoss(),
                                 scale::Union{Nothing,Real}=nothing,
                                 max_iter::Int=30,
                                 rtol::Float64=1e-5)
    u1, u2 = _extract_correspondences(correspondences)
    T1, T2, xis, Lambdas, Js = _fmat_prepare(u1, u2, Float64(sigma))
    prob = FMatFNSProblem(xis, Lambdas, Js)
    result = robust_solve(prob, MEstimator(loss);
                          scale=_scale_estimator(scale), max_iter, rtol)
    _finalize_fmat_result(result, T1, T2, u1, u2, Float64(sigma))
end

"""
    fit_fundmat_robust_taubin_fns(correspondences; sigma=1.0,
        loss=GemanMcClureLoss(), scale=nothing, max_iter_taubin=20,
        max_iter_fns=30, rtol=1e-5) -> FMatFitResult

Two-phase robust fundamental matrix fitting: Robust Taubin initialization
followed by Robust FNS refinement.

Phase 1 (Robust Taubin) provides stable initialization — the gradient-weighted
eigenproblem with rank-2 projection always produces a valid F, and IRLS
weighting rejects outliers. Phase 2 (Robust FNS) adds bias correction via
`vᵢ = 1/(fᵀΛᵢf)`, improving accuracy while inheriting the good starting point.

# Arguments
- `correspondences`: Vector of point pair correspondences
- `sigma::Real=1.0`: Noise standard deviation (pixels)
- `loss::AbstractLoss=GemanMcClureLoss()`: Robust loss function
- `scale::Union{Nothing,Real}=nothing`: Residual scale (`nothing` → MAD)
- `max_iter_taubin::Int=20`: Maximum Robust Taubin iterations (phase 1)
- `max_iter_fns::Int=30`: Maximum Robust FNS iterations (phase 2)
- `rtol::Float64=1e-5`: Convergence tolerance
"""
function fit_fundmat_robust_taubin_fns(correspondences;
                                        sigma::Real=1.0,
                                        loss::AbstractLoss=GemanMcClureLoss(),
                                        scale::Union{Nothing,Real}=nothing,
                                        max_iter_taubin::Int=20,
                                        max_iter_fns::Int=30,
                                        rtol::Float64=1e-5)
    u1, u2 = _extract_correspondences(correspondences)
    T1, T2, xis, Lambdas, Js = _fmat_prepare(u1, u2, Float64(sigma))
    prob_taubin = FMatTaubinProblem(xis, Lambdas, Js)
    prob_fns = FMatFNSProblem(xis, Lambdas, Js)
    result = robust_solve(prob_taubin, MEstimator(loss);
                          scale=_scale_estimator(scale), max_iter=max_iter_taubin, rtol,
                          refine=prob_fns, refine_max_iter=max_iter_fns)
    _finalize_fmat_result(result, T1, T2, u1, u2, Float64(sigma))
end

"""
    fit_fundmat(correspondences; quality=nothing,
        config=RansacConfig(), outlier_halfwidth=50.0)

Robust fundamental matrix estimation via 7-point RANSAC with scale-free
marginal scoring.

Returns a `RansacEstimate`.

For post-RANSAC polishing, use the standalone functions:
- `fit_fundmat_robust_fns` — FNS with bias correction
- `fit_fundmat_robust_taubin` — Taubin with gradient-weighted GEP
- `fit_fundmat_robust_taubin_fns` — Two-phase Taubin → FNS

# Arguments
- `quality`: Quality function for model selection.
  Default: `MarginalQuality(problem, outlier_halfwidth)`.
- `config`: RANSAC loop configuration.
- `outlier_halfwidth`: Half-width of the outlier domain (parameter `a` in Eq. 12).
  Only used when `quality` is not provided.

# Examples
```julia
result = fit_fundmat(correspondences)

# Custom quality
quality = PredictiveMarginalQuality(length(correspondences), 7, 30.0; codimension=2)
result = fit_fundmat(correspondences; quality)
```
"""
function fit_fundmat(correspondences;
    quality::Union{AbstractQualityFunction, Nothing} = nothing,
    config::RansacConfig = RansacConfig(),
    outlier_halfwidth::Real = 50.0)

    prob = FundMatProblem(correspondences)
    scoring = something(quality, MarginalQuality(prob, Float64(outlier_halfwidth)))
    ransac(prob, scoring; config)
end

# =============================================================================
# Correspondence Extraction Helper
# =============================================================================

"""
    _extract_correspondences(cs) -> (u1, u2)

Extract source and destination point vectors from correspondences.
Accepts Vector{Pair{SVector{2}, SVector{2}}} or similar iterables.
"""
function _extract_correspondences(cs)
    n = length(cs)
    u1 = Vector{SVector{2,Float64}}(undef, n)
    u2 = Vector{SVector{2,Float64}}(undef, n)
    @inbounds for (i, c) in enumerate(cs)
        u1[i] = SVector{2,Float64}(first(c)[1], first(c)[2])
        u2[i] = SVector{2,Float64}(last(c)[1], last(c)[2])
    end
    (u1, u2)
end
