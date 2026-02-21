# =============================================================================
# Line Fitting — Orthogonal distance with optional per-point covariances
# =============================================================================
#
# Implements Case B from student_t_ransac.tex (Section 9):
#   - fit_line(points)                  — unweighted total least squares
#   - fit_line(uncertain_points)        — weighted by projected covariances
#
# Line parametrization: n⊤p + d = 0, n = (cos φ, sin φ)
# Parameter vector θ = (φ, d), covariance Cov(θ̂) = s² · (J⊤WJ)⁻¹
#
# =============================================================================

# Dependencies: LinearAlgebra (eigen, Symmetric, dot)
#               StaticArrays (SVector, SMatrix)
#               VGC: Line2D, Point2, Uncertain, normal

# =============================================================================
# Result Type
# =============================================================================

"""
    LineFitResult{T}

Result of orthogonal distance line fitting.

# Fields
- `line::Uncertain{Line2D{T}, T, 2}`: fitted line with (φ, d) parameter covariance
- `s²::T`: estimated residual scale (S/(n-2))
- `ν::Int`: degrees of freedom (n - 2)

# Property Forwarding
- `result.line` → the Uncertain{Line2D} (access `.value` for bare Line2D)
- `result.s²`, `result.ν` → scale and degrees of freedom
"""
struct LineFitResult{T}
    line::Uncertain{Line2D{T}, T, 2}
    s²::T
    ν::Int
end

# =============================================================================
# Unweighted fit_line
# =============================================================================

"""
    fit_line(points::AbstractVector{<:Point2}) -> LineFitResult

Fit a 2D line by total least squares (orthogonal distance regression).

Returns the line in Hesse normal form n⊤p + d = 0 with parameter covariance
for θ = (φ, d) where n = (cos φ, sin φ).

The residual scale estimate s² = Σrᵢ²/(n-2) and its degrees of freedom ν = n-2
are included for subsequent F-test inlier classification.
"""
function fit_line(points::AbstractVector{<:Point2})
    n = length(points)
    n >= 3 || throw(ArgumentError("fit_line requires at least 3 points, got $n"))

    T = float(eltype(eltype(points)))

    # Centroid
    mx = zero(T)
    my = zero(T)
    @inbounds for p in points
        mx += T(p[1])
        my += T(p[2])
    end
    mx /= n
    my /= n

    # Scatter matrix M = Σ(pᵢ - p̄)(pᵢ - p̄)⊤
    m11 = zero(T)
    m12 = zero(T)
    m22 = zero(T)
    @inbounds for p in points
        dx = T(p[1]) - mx
        dy = T(p[2]) - my
        m11 += dx * dx
        m12 += dx * dy
        m22 += dy * dy
    end
    M = Symmetric(SMatrix{2,2,T}(m11, m12, m12, m22))

    # Eigenvector for smallest eigenvalue → normal direction
    E = eigen(M)
    idx = argmin(E.values)
    nv = SVector{2,T}(E.vectors[:, idx])

    # d = -n⊤p̄
    d = -(nv[1] * mx + nv[2] * my)
    line = Line2D{T}(SVector{3,T}(nv[1], nv[2], d))

    # Actual normal after Line2D normalization (should be same, but be safe)
    nv = normal(line)
    d = line.coeffs[3]
    tv = SVector{2,T}(-nv[2], nv[1])  # tangent

    # Residuals, cost, and information matrix
    S = zero(T)
    j11 = zero(T)  # Σ tᵢ²
    j12 = zero(T)  # Σ tᵢ
    @inbounds for p in points
        ri = nv[1] * T(p[1]) + nv[2] * T(p[2]) + d
        S += ri * ri
        ti = tv[1] * T(p[1]) + tv[2] * T(p[2])
        j11 += ti * ti
        j12 += ti
    end

    ν = n - 2
    s² = S / ν

    # J⊤J and parameter covariance
    JtJ = SMatrix{2,2,T}(j11, j12, j12, T(n))
    param_cov = s² * inv(JtJ)

    LineFitResult(Uncertain(line, param_cov), s², ν)
end

# =============================================================================
# Weighted fit_line (per-point covariances)
# =============================================================================

"""
    fit_line(points::AbstractVector{<:Uncertain{<:Point2}}) -> LineFitResult

Fit a 2D line with per-point covariance weighting.

Each point carries a 2×2 covariance matrix Σᵢ. The projected variance along the
line normal is σᵢ² = n⊤Σᵢn, giving weight wᵢ = 1/σᵢ².

Uses one refinement iteration: initial TLS direction → weighted scatter → refined
normal → recompute weights → weighted fit.
"""
function fit_line(points::AbstractVector{<:Uncertain{<:Point2}})
    n = length(points)
    n >= 3 || throw(ArgumentError("fit_line requires at least 3 points, got $n"))

    T = float(eltype(param_cov(points[1])))

    # --- Step 1: Initial unweighted TLS for direction ---
    mx = zero(T)
    my = zero(T)
    @inbounds for u in points
        p = u.value
        mx += T(p[1])
        my += T(p[2])
    end
    mx /= n
    my /= n

    m11 = zero(T)
    m12 = zero(T)
    m22 = zero(T)
    @inbounds for u in points
        p = u.value
        dx = T(p[1]) - mx
        dy = T(p[2]) - my
        m11 += dx * dx
        m12 += dx * dy
        m22 += dy * dy
    end
    M = Symmetric(SMatrix{2,2,T}(m11, m12, m12, m22))
    E = eigen(M)
    idx = argmin(E.values)
    nv = SVector{2,T}(E.vectors[:, idx])

    # --- Step 2: Weighted scatter with initial direction ---
    # Compute weights from projected variances σᵢ² = n⊤Σᵢn
    Σs = [SMatrix{2,2,T}(param_cov(u)) for u in points]
    ws = Vector{T}(undef, n)
    @inbounds for i in 1:n
        σ² = nv[1]^2 * Σs[i][1,1] + 2 * nv[1] * nv[2] * Σs[i][1,2] + nv[2]^2 * Σs[i][2,2]
        ws[i] = inv(max(σ², eps(T)))
    end

    # Weighted centroid
    W = zero(T)
    wmx = zero(T)
    wmy = zero(T)
    @inbounds for i in 1:n
        p = points[i].value
        w = ws[i]
        W += w
        wmx += w * T(p[1])
        wmy += w * T(p[2])
    end
    wmx /= W
    wmy /= W

    # Weighted scatter → refined normal
    m11 = zero(T)
    m12 = zero(T)
    m22 = zero(T)
    @inbounds for i in 1:n
        p = points[i].value
        w = ws[i]
        dx = T(p[1]) - wmx
        dy = T(p[2]) - wmy
        m11 += w * dx * dx
        m12 += w * dx * dy
        m22 += w * dy * dy
    end
    M = Symmetric(SMatrix{2,2,T}(m11, m12, m12, m22))
    E = eigen(M)
    idx = argmin(E.values)
    nv = SVector{2,T}(E.vectors[:, idx])

    # --- Step 3: Recompute weights with refined normal ---
    @inbounds for i in 1:n
        σ² = nv[1]^2 * Σs[i][1,1] + 2 * nv[1] * nv[2] * Σs[i][1,2] + nv[2]^2 * Σs[i][2,2]
        ws[i] = inv(max(σ², eps(T)))
    end

    # Weighted offset: d = -Σwᵢ(n⊤pᵢ) / Σwᵢ
    W = zero(T)
    wd = zero(T)
    @inbounds for i in 1:n
        p = points[i].value
        w = ws[i]
        W += w
        wd += w * (nv[1] * T(p[1]) + nv[2] * T(p[2]))
    end
    d = -wd / W

    line = Line2D{T}(SVector{3,T}(nv[1], nv[2], d))

    # Actual normal after normalization
    nv = normal(line)
    d = line.coeffs[3]
    tv = SVector{2,T}(-nv[2], nv[1])

    # --- Step 4: Weighted residuals, cost, information matrix ---
    # Recompute weights with final normal
    @inbounds for i in 1:n
        σ² = nv[1]^2 * Σs[i][1,1] + 2 * nv[1] * nv[2] * Σs[i][1,2] + nv[2]^2 * Σs[i][2,2]
        ws[i] = inv(max(σ², eps(T)))
    end

    S = zero(T)
    j11 = zero(T)
    j12 = zero(T)
    j22 = zero(T)
    @inbounds for i in 1:n
        p = points[i].value
        w = ws[i]
        ri = nv[1] * T(p[1]) + nv[2] * T(p[2]) + d
        S += w * ri * ri
        ti = tv[1] * T(p[1]) + tv[2] * T(p[2])
        j11 += w * ti * ti
        j12 += w * ti
        j22 += w
    end

    ν = n - 2
    s² = S / ν

    JtWJ = SMatrix{2,2,T}(j11, j12, j12, j22)
    param_cov_mat = s² * inv(JtWJ)

    LineFitResult(Uncertain(line, param_cov_mat), s², ν)
end
