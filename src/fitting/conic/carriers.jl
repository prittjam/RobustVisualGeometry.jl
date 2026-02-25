# =============================================================================
# Conic Carrier Functions and Distance Functions
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
