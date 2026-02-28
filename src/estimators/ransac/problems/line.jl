# =============================================================================
# RANSAC Line Fitting — LineFittingProblem
# =============================================================================
#
# Implements AbstractRansacProblem for 2D line estimation from points with
# optional per-point covariances. Uses standardized residuals for comparable
# scoring across heterogeneous noise levels.
#
# =============================================================================

# =============================================================================
# LineFittingProblem — Plain RANSAC (no local optimization)
# =============================================================================

"""
    LineFittingProblem{T} <: AbstractRansacProblem

Plain RANSAC problem for estimating a 2D line from points with per-point covariances.

Residuals are **standardized**: `r_std[i] = (n⊤pᵢ + d) / √(n⊤Σᵢn)`, so all
points contribute on a comparable scale regardless of their covariance shape.

Supports LO-RANSAC via `fit(problem, mask, weights)` which performs weighted
`fit_line` on the inlier subset.

# Constructors
    LineFittingProblem(pts::AbstractVector{<:Uncertain{<:Point2}})
    LineFittingProblem(pts::AbstractVector{<:Point2})  # identity covariances

# Solver Details
- Minimal sample: 2 points (`SingleSolution`)
- Model type: `Line2D{T}` (Hesse normal form)
- Residual: Standardized orthogonal distance
- LO-RANSAC: `fit(p, mask, weights)` for weighted `fit_line` on inliers
"""
struct LineFittingProblem{T} <: AbstractRansacProblem
    points::Vector{Point2{T}}
    cov_shapes::Vector{SMatrix{2,2,T,4}}
    _inlier_buf::Vector{Uncertain{Point2{T}, T, 2}}
end

function LineFittingProblem(pts::AbstractVector{<:Uncertain{<:Point2}})
    n = length(pts)
    T = float(eltype(param_cov(pts[1])))
    points = Vector{Point2{T}}(undef, n)
    cov_shapes = Vector{SMatrix{2,2,T,4}}(undef, n)
    @inbounds for i in eachindex(pts)
        points[i] = Point2{T}(pts[i].value...)
        cov_shapes[i] = SMatrix{2,2,T,4}(param_cov(pts[i]))
    end
    buf = Vector{Uncertain{Point2{T}, T, 2}}(undef, n)
    LineFittingProblem{T}(points, cov_shapes, buf)
end

function LineFittingProblem(pts::AbstractVector{<:Point2})
    n = length(pts)
    T = float(eltype(eltype(pts)))
    points = Vector{Point2{T}}(undef, n)
    I2 = SMatrix{2,2,T,4}(one(T), zero(T), zero(T), one(T))
    cov_shapes = fill(I2, n)
    @inbounds for i in eachindex(pts)
        points[i] = Point2{T}(pts[i]...)
    end
    buf = Vector{Uncertain{Point2{T}, T, 2}}(undef, n)
    LineFittingProblem{T}(points, cov_shapes, buf)
end

# =============================================================================
# AbstractRansacProblem Interface — LineFittingProblem
# =============================================================================

sample_size(::LineFittingProblem) = 2
codimension(::LineFittingProblem) = 1  # d_g = 1: signed distance is scalar
data_size(p::LineFittingProblem) = length(p.points)
model_type(::LineFittingProblem{T}) where T = Line2D{T}
solver_cardinality(::LineFittingProblem) = SingleSolution()

function solve(p::LineFittingProblem, idx::AbstractVector{Int})
    @inbounds Line2D(p.points[idx[1]], p.points[idx[2]])
end

function residuals!(r::Vector, p::LineFittingProblem{T}, line::Line2D{T}) where T
    nv = normal(line)
    d = line.coeffs[3]
    @inbounds for i in eachindex(r, p.points)
        pt = p.points[i]
        Σ = p.cov_shapes[i]
        ri = nv[1] * T(pt[1]) + nv[2] * T(pt[2]) + d
        σ_i = sqrt(nv[1]^2 * Σ[1,1] + 2*nv[1]*nv[2]*Σ[1,2] + nv[2]^2 * Σ[2,2])
        r[i] = ri / max(σ_i, eps(T))
    end
    return r
end

function test_sample(p::LineFittingProblem{T}, idx::AbstractVector{Int}) where T
    @inbounds begin
        p1 = p.points[idx[1]]
        p2 = p.points[idx[2]]
        dx = T(p1[1]) - T(p2[1])
        dy = T(p1[2]) - T(p2[2])
        return dx*dx + dy*dy > eps(T)
    end
end

# =============================================================================
# fit — Weighted fit_line for LO-RANSAC
# =============================================================================

function fit(p::LineFittingProblem{T}, mask::BitVector, ::AbstractVector, ::LinearFit) where T
    n_inliers = sum(mask)
    n_inliers < 3 && return nothing
    k = 0
    @inbounds for i in eachindex(mask)
        if mask[i]
            k += 1
            p._inlier_buf[k] = Uncertain(p.points[i], p.cov_shapes[i])
        end
    end
    result = fit_line(@view p._inlier_buf[1:k])
    return result.line.value
end

# --- residual_jacobian for LineFittingProblem ---
#
# Returns whitened residual, Jacobian, and log-determinant (Section 3, Eq. 5-7):
#   rᵢ = r_raw / σᵢ, Gᵢ = [t⊤pᵢ/σᵢ, 1/σᵢ], ℓᵢ = log(σ²ᵢ) = log(n⊤Σᵢn)
# where σᵢ = √(n⊤Σᵢn) is the projected per-point noise and Cᵢ = σ²ᵢ (scalar, dg=1).

function residual_jacobian(p::LineFittingProblem{T},
                                       model::Line2D{T}, i::Int) where T
    nv = normal(model)
    d_val = model.coeffs[3]
    tv = SVector{2,T}(-nv[2], nv[1])  # tangent vector

    @inbounds begin
        pt = p.points[i]
        Σ = p.cov_shapes[i]

        r_raw = nv[1] * T(pt[1]) + nv[2] * T(pt[2]) + d_val
        σ²ᵢ = nv[1]^2 * Σ[1,1] + 2*nv[1]*nv[2]*Σ[1,2] + nv[2]^2 * Σ[2,2]
        σᵢ = sqrt(max(σ²ᵢ, eps(T)))
        rᵢ = r_raw / σᵢ

        tᵢ = tv[1] * T(pt[1]) + tv[2] * T(pt[2])
        Gᵢ = SVector{2,T}(tᵢ / σᵢ, one(T) / σᵢ)

        # ℓᵢ = log|Cᵢ| = log(σ²ᵢ) since dg=1 and Cᵢ = n⊤Σᵢn (scalar)
        ℓᵢ = σ²ᵢ > eps(T) ? log(σ²ᵢ) : zero(T)
    end
    return (rᵢ, Gᵢ, ℓᵢ)
end

# =============================================================================
# InhomLineFittingProblem — Simplest possible RANSAC problem (y = a + bx)
# =============================================================================

"""
    InhomLineFittingProblem{T} <: AbstractRansacProblem

Inhomogeneous line fitting: `y = a + b·x`, 2 parameters, scalar residual.

The simplest possible RANSAC problem for debugging and testing the
covariance-aware pipeline. No standardization, no per-point covariances.

# Model
- `SVector{2,T}`: `(a, b)` where `y = a + b·x`
- Residual: `r_i = y_i - a - b·x_i`  (scalar, raw)
- Residual DOF: 1
- Model DOF: 2
"""
struct InhomLineFittingProblem{T} <: AbstractRansacProblem
    points::Vector{SVector{2,T}}
end

function InhomLineFittingProblem(pts::AbstractVector)
    T = float(eltype(eltype(pts)))
    InhomLineFittingProblem{T}([SVector{2,T}(p...) for p in pts])
end

sample_size(::InhomLineFittingProblem) = 2
codimension(::InhomLineFittingProblem) = 1  # d_g = 1: signed distance is scalar
data_size(p::InhomLineFittingProblem) = length(p.points)
model_type(::InhomLineFittingProblem{T}) where T = SVector{2,T}
solver_cardinality(::InhomLineFittingProblem) = SingleSolution()

function solve(p::InhomLineFittingProblem{T}, idx::AbstractVector{Int}) where T
    @inbounds begin
        p1 = p.points[idx[1]]
        p2 = p.points[idx[2]]
        dx = p2[1] - p1[1]
        abs(dx) < eps(T) && return nothing
        b = (p2[2] - p1[2]) / dx
        a = p1[2] - b * p1[1]
    end
    return SVector{2,T}(a, b)
end

function residuals!(r::Vector, p::InhomLineFittingProblem{T},
                                model::SVector{2,T}) where T
    a, b = model
    @inbounds for i in eachindex(r, p.points)
        pt = p.points[i]
        r[i] = pt[2] - a - b * pt[1]
    end
    return r
end

function test_sample(p::InhomLineFittingProblem{T}, idx::AbstractVector{Int}) where T
    @inbounds begin
        dx = p.points[idx[2]][1] - p.points[idx[1]][1]
        return abs(dx) > eps(T)
    end
end

# --- Solver Jacobian: ∂(a,b)/∂(y₁,y₂) ---
#
# For inhomogeneous regression y = a + bx, x is exact (independent variable).
# Only y₁, y₂ have measurement noise, so we return the 2×2 Jacobian of the
# solver output w.r.t. the noisy inputs only:
#
#   J_y = [x₂/dx, -x₁/dx;  -1/dx, 1/dx]
#
# This gives Σ_θ = s²·J_y·J_y' = s²·(X'X)⁻¹ (exact OLS covariance for 2 pts).

function solver_jacobian(p::InhomLineFittingProblem{T}, idx::AbstractVector{Int},
                                     model::SVector{2,T}) where T
    @inbounds begin
        p1 = p.points[idx[1]]
        p2 = p.points[idx[2]]
    end

    dx = p2[1] - p1[1]
    abs(dx) < eps(T) && return nothing
    inv_dx = one(T) / dx

    # J_y[i,j] = ∂θ_i/∂y_j  (2×2, y-only columns of full 2×4 Jacobian)
    J = SMatrix{2,2,T}(
         p2[1] * inv_dx,  -inv_dx,           # column 1: ∂(a,b)/∂y₁
        -p1[1] * inv_dx,   inv_dx            # column 2: ∂(a,b)/∂y₂
    )

    return (J=J, model=model)
end

# --- residual_jacobian: rᵢ = yᵢ - a - bxᵢ, Gᵢ = [-1, -xᵢ], ℓᵢ = 0 (homoscedastic) ---

function residual_jacobian(p::InhomLineFittingProblem{T},
                                       model::SVector{2,T}, i::Int) where T
    @inbounds begin
        pt = p.points[i]
        a, b = model
        rᵢ = pt[2] - a - b * pt[1]
        Gᵢ = SA[-one(T), -pt[1]]
    end
    return (rᵢ, Gᵢ, zero(T))
end

# =============================================================================
# EivLineFittingProblem — Errors-in-Variables Line (noise in both x and y)
# =============================================================================

"""
    EivLineFittingProblem{T} <: AbstractRansacProblem

Errors-in-variables line fitting: `cos(φ)·x + sin(φ)·y + d = 0`, 2 parameters, scalar residual.

Both x and y have measurement noise (errors-in-variables). The model is
parameterized as `(φ, d)` where the normal is `n = (cos φ, sin φ)`.

# Model
- `SVector{2,T}`: `(φ, d)` where `cos(φ)·x + sin(φ)·y + d = 0`
- Residual: `r_i = cos(φ)·x_i + sin(φ)·y_i + d`  (scalar, signed distance)
- Residual DOF: 1
- Model DOF: 2
"""
struct EivLineFittingProblem{T} <: AbstractRansacProblem
    points::Vector{SVector{2,T}}
end

function EivLineFittingProblem(pts::AbstractVector)
    T = float(eltype(eltype(pts)))
    EivLineFittingProblem{T}([SVector{2,T}(p...) for p in pts])
end

sample_size(::EivLineFittingProblem) = 2
codimension(::EivLineFittingProblem) = 1
data_size(p::EivLineFittingProblem) = length(p.points)
model_type(::EivLineFittingProblem{T}) where T = SVector{2,T}
solver_cardinality(::EivLineFittingProblem) = SingleSolution()

function solve(p::EivLineFittingProblem{T}, idx::AbstractVector{Int}) where T
    @inbounds begin
        p1 = p.points[idx[1]]
        p2 = p.points[idx[2]]
    end
    v = p2 - p1
    L2 = v[1]^2 + v[2]^2
    L2 < eps(T) && return nothing
    inv_L = one(T) / sqrt(L2)
    # Normal: 90° CCW rotation of direction, normalized
    n1 = -v[2] * inv_L
    n2 =  v[1] * inv_L
    φ = atan(n2, n1)
    d = -(n1 * p1[1] + n2 * p1[2])
    return SVector{2,T}(φ, d)
end

function residuals!(r::Vector, p::EivLineFittingProblem{T},
                                model::SVector{2,T}) where T
    φ, d = model
    n1, n2 = cos(φ), sin(φ)
    @inbounds for i in eachindex(r, p.points)
        pt = p.points[i]
        r[i] = n1 * pt[1] + n2 * pt[2] + d
    end
    return r
end

function test_sample(p::EivLineFittingProblem{T}, idx::AbstractVector{Int}) where T
    @inbounds begin
        v = p.points[idx[2]] - p.points[idx[1]]
        return v[1]^2 + v[2]^2 > eps(T)
    end
end

# --- Solver Jacobian: ∂(φ,d)/∂(x₁,y₁,x₂,y₂) ---
#
# For homogeneous line fitting n·p + d = 0, both x and y have noise.
# J is 2×4: ∂(φ,d)/∂(x₁,y₁,x₂,y₂).
# Σ_θ = σ²·J·J' gives the 2×2 parameter covariance.
#
# Derivation: with v = p₂ - p₁, L = ‖v‖, n = (-v₂, v₁)/L, t = (-n₂, n₁):
#   ∂φ/∂z = (1/L)[-n; n]                         (4-vector)
#   ∂d/∂z = [n(τ₁/L - 1); -n·τ₁/L]              (4-vector, τ₁ = t·p₁)

function solver_jacobian(p::EivLineFittingProblem{T}, idx::AbstractVector{Int},
                                     model::SVector{2,T}) where T
    @inbounds begin
        p1 = p.points[idx[1]]
        p2 = p.points[idx[2]]
    end

    v = p2 - p1
    L2 = v[1]^2 + v[2]^2
    L2 < eps(T) && return nothing
    inv_L = one(T) / sqrt(L2)

    φ, d = model
    n1, n2 = cos(φ), sin(φ)
    t1, t2 = -n2, n1           # tangent = (-sinφ, cosφ)
    τ1 = t1 * p1[1] + t2 * p1[2]   # t · p₁

    a = τ1 * inv_L - one(T)    # coefficient for ∂d/∂(p₁)
    b = -τ1 * inv_L            # coefficient for ∂d/∂(p₂)

    # Column-major: (row1, row2) for each column
    J = SMatrix{2,4,T}(
        -n1 * inv_L,    # ∂φ/∂x₁
         n1 * a,         # ∂d/∂x₁
        -n2 * inv_L,    # ∂φ/∂y₁
         n2 * a,         # ∂d/∂y₁
         n1 * inv_L,    # ∂φ/∂x₂
         n1 * b,         # ∂d/∂x₂
         n2 * inv_L,    # ∂φ/∂y₂
         n2 * b          # ∂d/∂y₂
    )

    return (J=J, model=model)
end

# --- residual_jacobian: rᵢ = n·pᵢ + d, Gᵢ = [t·pᵢ, 1], ℓᵢ = 0 (homoscedastic) ---

function residual_jacobian(p::EivLineFittingProblem{T},
                                       model::SVector{2,T}, i::Int) where T
    @inbounds begin
        pt = p.points[i]
        φ, d = model
        n₁, n₂ = cos(φ), sin(φ)
        t₁, t₂ = -n₂, n₁
        rᵢ = n₁ * pt[1] + n₂ * pt[2] + d
        Gᵢ = SA[t₁ * pt[1] + t₂ * pt[2], one(T)]
    end
    return (rᵢ, Gᵢ, zero(T))
end
