# =============================================================================
# Conic Fitting Utilities — Normalization, carrier builders, helpers
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
