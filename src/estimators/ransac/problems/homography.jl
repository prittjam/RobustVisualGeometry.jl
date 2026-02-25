# =============================================================================
# RANSAC Homography Problem
# =============================================================================
#
# Implements AbstractCspondProblem for 2D projective homography estimation
# from point correspondences.
#
# PLACEMENT: Included from main VisualGeometryCore.jl (NOT from Estimators
# submodule) because it depends on public solver functions from
# geometry/homography_solvers.jl (loaded after Estimators).
#
# =============================================================================

# Dependencies: VGC (homography_4pt, _transfer_error_jacobian_wrt_h, etc.)
#               All estimation types available from parent module

using LinearAlgebra: det

# =============================================================================
# HomographyProblem Type
# =============================================================================

"""
    HomographyProblem{T,S,R} <: AbstractCspondProblem{T}

RANSAC problem for estimating a 2D projective homography from point correspondences.

Estimates H such that `u₂ ~ H * [u₁; 1]` (in homogeneous coordinates).

Type parameters:
- `T`: Element type (Float64, etc.)
- `S <: AbstractSampler`: Sampling strategy (uniform, PROSAC)
- `R <: AbstractRefinement`: LO-RANSAC refinement strategy

# Constructor
```julia
cs = [SA[1.0,2.0] => SA[3.0,4.0], ...]
HomographyProblem(cs)                                  # plain RANSAC (default)
HomographyProblem(cs; refinement=DltRefinement())      # DLT refit only
HomographyProblem(cs; refinement=IrlsRefinement())     # LO-RANSAC (IRLS)
```

# Solver Details
- Minimal sample: 4 point pairs (`SingleSolution`)
- Model type: `SMatrix{3,3,T,9}` (Frobenius-normalized, H[3,3] >= 0)
- Residual: Sampson distance (EIV-corrected)
- Degeneracy: Collinearity + convexity check
- Refinement: Controlled by `R` type parameter
"""
struct HomographyProblem{T<:AbstractFloat, S<:AbstractSampler, R<:AbstractRefinement} <: AbstractCspondProblem{T}
    cs::StructArrays.StructVector{Pair{SVector{2,T},SVector{2,T}}, @NamedTuple{first::Vector{SVector{2,T}}, second::Vector{SVector{2,T}}}}
    _sampler::S
    _refinement::R
    _dlt_buf::FixedSizeArray{T,2,Memory{T}}
    _u₁_buf::FixedSizeArray{SVector{2,T},1,Memory{SVector{2,T}}}
    _u₂_buf::FixedSizeArray{SVector{2,T},1,Memory{SVector{2,T}}}
    _w_buf::FixedSizeArray{T,1,Memory{T}}
    _sampson_buf::FixedSizeArray{SMatrix{2,2,T,4},1,Memory{SMatrix{2,2,T,4}}}
    _svd_ws::SVDWorkspace{T}
end

function HomographyProblem(correspondences::AbstractVector;
                           refinement::AbstractRefinement=NoRefinement())
    n = length(correspondences)
    n >= 4 || throw(ArgumentError("Need at least 4 correspondences, got $n"))

    cs, smplr, u₁_buf, u₂_buf, w_buf = _build_cspond(correspondences, 4)
    T = eltype(first(correspondences).first)

    HomographyProblem{T, typeof(smplr), typeof(refinement)}(
        cs, smplr, refinement,
        FixedSizeArray{T}(undef, 2n, 9),
        u₁_buf, u₂_buf, w_buf,
        FixedSizeArray{SMatrix{2,2,T,4}}(undef, n),
        SVDWorkspace{T}(2n, 9))
end

"""
    LoHomographyProblem(correspondences; refinement=IrlsRefinement())

Convenience constructor for LO-RANSAC homography estimation.

Equivalent to `HomographyProblem(cs; refinement=IrlsRefinement())`.
Returns a `HomographyProblem` with IRLS refinement enabled by default.
"""
LoHomographyProblem(correspondences::AbstractVector;
                    refinement::AbstractRefinement=IrlsRefinement()) =
    HomographyProblem(correspondences; refinement)

# =============================================================================
# AbstractRansacProblem Interface
# =============================================================================

sample_size(::HomographyProblem) = 4
model_type(::HomographyProblem{T}) where T = SMatrix{3,3,T,9}
solver_cardinality(::HomographyProblem) = SingleSolution()
codimension(::HomographyProblem) = 2  # d_g = 2: two constraint equations from v̄ = λHū
measurement_covariance(::HomographyProblem) = Heteroscedastic()

function solve(p::HomographyProblem, idx::Vector{Int})
    u₁ = p.cs.first; u₂ = p.cs.second
    @inbounds homography_4pt(
        u₁[idx[1]], u₁[idx[2]], u₁[idx[3]], u₁[idx[4]],
        u₂[idx[1]], u₂[idx[2]], u₂[idx[3]], u₂[idx[4]])
end

function residuals!(r::Vector, p::HomographyProblem{T}, H::SMatrix{3,3,T,9}) where T
    u₁ = p.cs.first; u₂ = p.cs.second
    @inbounds for i in eachindex(r, u₁, u₂)
        r[i] = _sampson_distance_homography(u₁[i], u₂[i], H)
    end
    return r
end

function test_sample(p::HomographyProblem{T}, idx::Vector{Int}) where T
    u₁ = p.cs.first; u₂ = p.cs.second
    @inbounds _homography_sample_nondegenerate(
        u₁[idx[1]], u₁[idx[2]], u₁[idx[3]], u₁[idx[4]],
        u₂[idx[1]], u₂[idx[2]], u₂[idx[3]], u₂[idx[4]])
end

function test_model(::HomographyProblem{T}, H::SMatrix{3,3,T,9}) where T
    return abs(det(H)) > eps(T)
end

# =============================================================================
# AbstractCspondProblem Dispatch Points
# =============================================================================

min_dlt_inliers(::HomographyProblem) = 5

function _dlt_refit(p::HomographyProblem, k::Int)
    homography_dlt!(@view(p._dlt_buf[1:2k, :]),
                     @view(p._u₁_buf[1:k]), @view(p._u₂_buf[1:k]);
                     svd_ws=p._svd_ws)
end

# =============================================================================
# IRLS Refinement: Weighted DLT with Sampson Correction
# =============================================================================

constraint_type(::HomographyProblem) = Unconstrained()

function weighted_system(p::HomographyProblem{T}, H, mask, w) where T
    k = _gather_cspond_inliers!(p, mask, w)
    k < min_dlt_inliers(p) && return nothing

    T₁ = hartley_normalization(@view p._u₁_buf[1:k])
    T₂ = hartley_normalization(@view p._u₂_buf[1:k])

    # Compute 2×2 Sampson correction factors per correspondence
    Hn = T₂ * H * inv(T₁)

    @inbounds for i in 1:k
        s = T₁ * SA[p._u₁_buf[i][1], p._u₁_buf[i][2], one(T)]
        d = T₂ * SA[p._u₂_buf[i][1], p._u₂_buf[i][2], one(T)]
        wi = p._w_buf[i]

        # Sampson Jacobian
        w3 = Hn[3,1]*s[1] + Hn[3,2]*s[2] + Hn[3,3]*s[3]

        j11 = Hn[1,1] - d[1]*Hn[3,1]
        j12 = Hn[1,2] - d[1]*Hn[3,2]
        j13 = -w3

        j21 = Hn[2,1] - d[2]*Hn[3,1]
        j22 = Hn[2,2] - d[2]*Hn[3,2]
        j24 = -w3

        # S = JJᵀ (2×2)
        s11 = j11*j11 + j12*j12 + j13*j13
        s12 = j11*j21 + j12*j22
        s22 = j21*j21 + j22*j22 + j24*j24

        # Cholesky of S → L⁻¹ * sqrt(w_robust)
        det_s = s11 * s22 - s12 * s12
        if det_s > eps(T) * max(s11, s22)^2 && s11 > eps(T)
            l11 = sqrt(s11)
            l21 = s12 / l11
            l22 = sqrt(max(s22 - l21*l21, zero(T)))

            sqrt_w = sqrt(wi)
            inv_l11 = one(T) / l11
            inv_l22 = one(T) / l22

            c11 = sqrt_w * inv_l11
            c21 = -sqrt_w * l21 * inv_l11 * inv_l22
            c22 = sqrt_w * inv_l22
        else
            sqrt_w = sqrt(wi)
            c11 = sqrt_w
            c21 = zero(T)
            c22 = sqrt_w
        end

        # Lower-triangular weight matrix (applied to row pair)
        p._sampson_buf[i] = SMatrix{2,2,T,4}(c11, c21, zero(T), c22)
    end

    A = @view p._dlt_buf[1:2k, :]
    _fill_homography_dlt!(A, @view(p._u₁_buf[1:k]), @view(p._u₂_buf[1:k]),
                           T₁, T₂; weights=(@view p._sampson_buf[1:k]))

    return (A = A, T₁ = T₁, T₂ = T₂)
end

function model_from_solution(::HomographyProblem{T}, h, sys) where T
    H_norm = _vec9_to_mat33(h, T)
    H = inv(sys.T₂) * H_norm * sys.T₁
    return sign_normalize(H)
end

# =============================================================================
# Solver Jacobian — Forward Homography Only
# =============================================================================

"""
    solver_jacobian(p::HomographyProblem, idx, H) -> NamedTuple or nothing

Compute the solver Jacobian for the forward homography (src → dst).

We only propagate uncertainty through the forward direction because the inverse
homography `H⁻¹` is a highly nonlinear function of the H parameters, making the
first-order covariance approximation unreliable.

Returns `(J=J, H=H_fwd)` where:
- `J::SMatrix{9,16}`: ∂vec(H)/∂[s₁;s₂;s₃;s₄;d₁;d₂;d₃;d₄]
- `H`: the forward homography (Frobenius-normalized, H[3,3] ≥ 0)
"""
function solver_jacobian(p::HomographyProblem{T}, idx::Vector{Int},
                                     H::SMatrix{3,3,T,9}) where T
    u₁ = p.cs.first; u₂ = p.cs.second

    @inbounds begin
        s1 = u₁[idx[1]]; s2 = u₁[idx[2]]; s3 = u₁[idx[3]]; s4 = u₁[idx[4]]
        d1 = u₂[idx[1]]; d2 = u₂[idx[2]]; d3 = u₂[idx[3]]; d4 = u₂[idx[4]]
    end

    result = homography_4pt_with_jacobian(s1, s2, s3, s4, d1, d2, d3, d4)
    isnothing(result) && return nothing
    H_fwd, J = result

    return (J=J, H=H_fwd)
end

# =============================================================================
# Sampson Error — Algebraic Constraint Formulation (Chum et al.)
# =============================================================================
#
# For correspondence (s, d) = ((x,y), (x',y')) and homography H, the algebraic
# constraint g = 0 encodes d̃ ∝ H·s̃:
#
#   g₁ = h₁ᵀs̃ − x'·h₃ᵀs̃
#   g₂ = h₂ᵀs̃ − y'·h₃ᵀs̃
#
# The measurement Jacobian G_x (2×4) w.r.t. (x, y, x', y'):
#   G_x = [ H[1,1]−x'H[3,1]   H[1,2]−x'H[3,2]   −c    0  ]
#         [ H[2,1]−y'H[3,1]   H[2,2]−y'H[3,2]    0   −c  ]
# where c = h₃ᵀs̃.
#
# The projected covariance (isotropic Σ_i = I₄):
#   C_i = G_x G_xᵀ   (2×2 Sampson matrix)
#
# Sampson distance: d_S = sqrt(g' C⁻¹ g)
# Whitened residual: r_w = L⁻¹g where LL' = C
#
# The model Jacobian G_θ (2×9) w.r.t. vec(H) (column-major):
#   ∂g₁/∂vec(H) = [x, 0, −x'x,  y, 0, −x'y,  1, 0, −x']
#   ∂g₂/∂vec(H) = [0, x, −y'x,  0, y, −y'y,  0, 1, −y']
#
# =============================================================================

# --- 2×2 Cholesky and forward-substitution helpers ---

"""
    _cholesky2x2(a11, a21, a22) -> (l11, l21, l22)

Cholesky factor of a 2×2 SPD matrix [[a11, a21]; [a21, a22]].
Falls back to identity if degenerate.
"""
@inline function _cholesky2x2(a11::T, a21::T, a22::T) where T
    if a11 > eps(T)
        l11 = sqrt(a11)
        l21 = a21 / l11
        diag2 = a22 - l21 * l21
        l22 = diag2 > zero(T) ? sqrt(diag2) : zero(T)
    else
        l11 = one(T); l21 = zero(T); l22 = one(T)
    end
    return (l11, l21, l22)
end

"""
    _solve_lower2x2(l11, l21, l22, b1, b2) -> (x1, x2)

Forward-substitution: solve L x = b for 2×2 lower-triangular L.
"""
@inline function _solve_lower2x2(l11::T, l21::T, l22::T, b1::T, b2::T) where T
    x1 = b1 / l11
    x2 = (b2 - l21 * x1) / l22
    return (x1, x2)
end

"""
    _solve_lower2x2_mat(l11, l21, l22, G::SMatrix{2,N,T}) -> SMatrix{2,N,T}

Forward-substitution: solve L X = G for 2×2 lower-triangular L, applied to
each column of the 2×N matrix G.
"""
@inline function _solve_lower2x2_mat(l11::T, l21::T, l22::T,
                                      G::SMatrix{2,N,T}) where {N,T}
    inv_l11 = one(T) / l11
    inv_l22 = one(T) / l22
    # Build column-major tuple: for each column j, row1 then row2
    vals = ntuple(Val(2N)) do k
        j = (k - 1) >> 1 + 1  # column index (1-based)
        row = ((k - 1) & 1) + 1
        if row == 1
            G[1, j] * inv_l11
        else
            (G[2, j] - l21 * G[1, j] * inv_l11) * inv_l22
        end
    end
    return SMatrix{2,N,T}(vals)
end

# --- Core Sampson quantities ---

"""
    _sampson_quantities(H, s, d) -> (gᵢ, ∂ₓgᵢ, ∂θgᵢ)

Compute the algebraic constraint gᵢ and its Jacobians (Section 3, Eq. 5-7).

The constraint gᵢ = 0 encodes d̃ ∝ H·s̃ (dg = 2 equations):
  g₁ = h₁ᵀs̃ − x'·h₃ᵀs̃
  g₂ = h₂ᵀs̃ − y'·h₃ᵀs̃

Returns:
- gᵢ::SVector{2,T}     — constraint vector
- ∂ₓgᵢ::SMatrix{2,4,T} — ∂gᵢ/∂xᵢ (measurement Jacobian, for Σ̃_{gᵢ} = ∂ₓgᵢ Σ̃_{xᵢ} (∂ₓgᵢ)ᵀ)
- ∂θgᵢ::SMatrix{2,9,T} — ∂gᵢ/∂θ (model Jacobian, for estimation covariance, Eq. 7)
"""
@inline function _sampson_quantities(H::SMatrix{3,3,T,9},
                                      s::SVector{2,T},
                                      d::SVector{2,T}) where T
    x, y = s[1], s[2]
    xp, yp = d[1], d[2]

    # h₃ᵀs̃ = H[3,1]*x + H[3,2]*y + H[3,3]
    c = H[3,1] * x + H[3,2] * y + H[3,3]

    # Algebraic constraint
    g1 = (H[1,1] * x + H[1,2] * y + H[1,3]) - xp * c
    g2 = (H[2,1] * x + H[2,2] * y + H[2,3]) - yp * c
    g = SVector{2,T}(g1, g2)

    # Measurement Jacobian G_x (2×4) w.r.t. (x, y, x', y')
    gx_11 = H[1,1] - xp * H[3,1]
    gx_12 = H[1,2] - xp * H[3,2]
    gx_13 = -c
    gx_21 = H[2,1] - yp * H[3,1]
    gx_22 = H[2,2] - yp * H[3,2]
    gx_24 = -c
    G_x = SMatrix{2,4,T}(gx_11, gx_21,   # col 1: ∂g/∂x
                          gx_12, gx_22,   # col 2: ∂g/∂y
                          gx_13, zero(T), # col 3: ∂g/∂x'
                          zero(T), gx_24) # col 4: ∂g/∂y'

    # Model Jacobian G_θ (2×9) w.r.t. vec(H) column-major:
    # vec(H) = [H[1,1],H[2,1],H[3,1], H[1,2],H[2,2],H[3,2], H[1,3],H[2,3],H[3,3]]
    G_θ = SMatrix{2,9,T}(
        x,       zero(T),  # col 1: ∂g/∂H[1,1], ∂g/∂H[2,1]
        zero(T), x,        # col 2: ∂g/∂H[2,1]... wait, column-major
        -xp*x,   -yp*x,    # col 3: ∂g/∂H[3,1]
        y,       zero(T),  # col 4: ∂g/∂H[1,2]
        zero(T), y,        # col 5: ∂g/∂H[2,2]
        -xp*y,   -yp*y,    # col 6: ∂g/∂H[3,2]
        one(T),  zero(T),  # col 7: ∂g/∂H[1,3]
        zero(T), one(T),   # col 8: ∂g/∂H[2,3]
        -xp,     -yp,      # col 9: ∂g/∂H[3,3]
    )

    return (g, G_x, G_θ)
end

"""
    _sampson_distance_homography(s, d, H) -> T

Scalar Sampson distance for a single homography correspondence.

    d_S = sqrt(g' C⁻¹ g)

where g is the algebraic constraint and C = G_x G_xᵀ (2×2).
Uses inline 2×2 inverse for efficiency.
"""
@inline function _sampson_distance_homography(s::SVector{2,T}, d::SVector{2,T},
                                               H::SMatrix{3,3,T,9}) where T
    g, G_x, _ = _sampson_quantities(H, s, d)

    # C = G_x G_xᵀ (2×2)
    c11 = G_x[1,1]^2 + G_x[1,2]^2 + G_x[1,3]^2 + G_x[1,4]^2
    c21 = G_x[2,1]*G_x[1,1] + G_x[2,2]*G_x[1,2] + G_x[2,3]*G_x[1,3] + G_x[2,4]*G_x[1,4]
    c22 = G_x[2,1]^2 + G_x[2,2]^2 + G_x[2,3]^2 + G_x[2,4]^2

    # Inline 2×2 inverse: C⁻¹ = (1/det) * [c22, -c21; -c21, c11]
    det_c = c11 * c22 - c21 * c21
    abs(det_c) < eps(T) && return typemax(T)
    inv_det = one(T) / det_c

    # g' C⁻¹ g
    q = inv_det * (c22 * g[1]^2 - 2 * c21 * g[1] * g[2] + c11 * g[2]^2)
    return sqrt(max(q, zero(T)))
end

# =============================================================================
# residual_jacobian — Sampson-whitened (2D residual, 2×9 Jacobian)
# =============================================================================

"""
    residual_jacobian(p::HomographyProblem, H, i) -> (rᵢ, ∂θgᵢ_w, ℓᵢ)

Whitened constraint, whitened model Jacobian, and covariance penalty
for correspondence `i` (Section 3.3, Eq. 7, 12, 21).

For homography with isotropic noise (Σ̃_{xᵢ} = I):
  Σ̃_{gᵢ}|_{Σ_θ=0} = (∂ₓgᵢ)(∂ₓgᵢ)ᵀ    (2×2, Sampson matrix)

Returns `(rᵢ, ∂θgᵢ_w, ℓᵢ)` where:
- `rᵢ = L⁻¹gᵢ` — whitened constraint (SVector{2}),
  satisfies qᵢ = rᵢᵀrᵢ = gᵢᵀΣ̃_{gᵢ}⁻¹gᵢ (weighted squared residual)
- `∂θgᵢ_w = L⁻¹∂θgᵢ` — whitened constraint Jacobian w.r.t. model (SMatrix{2,9}),
  satisfies (∂θgᵢ_w)ᵀ(∂θgᵢ_w) = (∂θgᵢ)ᵀΣ̃_{gᵢ}⁻¹(∂θgᵢ) (Fisher information)
- `ℓᵢ = log|Σ̃_{gᵢ}|` — covariance penalty for Algorithm 1 (Eq. 12)

Here LLᵀ = Σ̃_{gᵢ}|_{Σ_θ=0} is the Cholesky factorization.
"""
function residual_jacobian(p::HomographyProblem{T},
                           H::SMatrix{3,3,T,9}, i::Int) where T
    @inbounds begin
        sᵢ = p.cs.first[i]
        dᵢ = p.cs.second[i]
    end

    gᵢ, ∂ₓgᵢ, ∂θgᵢ = _sampson_quantities(H, sᵢ, dᵢ)

    # Σ̃_{gᵢ} = (∂ₓgᵢ)(∂ₓgᵢ)ᵀ  (2×2 Sampson matrix, isotropic noise)
    c₁₁ = ∂ₓgᵢ[1,1]^2 + ∂ₓgᵢ[1,2]^2 + ∂ₓgᵢ[1,3]^2 + ∂ₓgᵢ[1,4]^2
    c₂₁ = ∂ₓgᵢ[2,1]*∂ₓgᵢ[1,1] + ∂ₓgᵢ[2,2]*∂ₓgᵢ[1,2] + ∂ₓgᵢ[2,3]*∂ₓgᵢ[1,3] + ∂ₓgᵢ[2,4]*∂ₓgᵢ[1,4]
    c₂₂ = ∂ₓgᵢ[2,1]^2 + ∂ₓgᵢ[2,2]^2 + ∂ₓgᵢ[2,3]^2 + ∂ₓgᵢ[2,4]^2

    # Cholesky: LLᵀ = Σ̃_{gᵢ}
    l₁₁, l₂₁, l₂₂ = _cholesky2x2(c₁₁, c₂₁, c₂₂)

    # rᵢ = L⁻¹gᵢ  (whitened constraint)
    rw₁, rw₂ = _solve_lower2x2(l₁₁, l₂₁, l₂₂, gᵢ[1], gᵢ[2])
    rᵢ = SVector{2,T}(rw₁, rw₂)

    # ∂θgᵢ_w = L⁻¹∂θgᵢ  (whitened model Jacobian)
    ∂θgᵢ_w = _solve_lower2x2_mat(l₁₁, l₂₁, l₂₂, ∂θgᵢ)

    # ℓᵢ = log|Σ̃_{gᵢ}| = log(det(L)²) = log(l₁₁² l₂₂²)
    det_Σ̃ = c₁₁ * c₂₂ - c₂₁ * c₂₁
    ℓᵢ = det_Σ̃ > zero(T) ? log(det_Σ̃) : zero(T)

    return (rᵢ, ∂θgᵢ_w, ℓᵢ)
end

# =============================================================================
# measurement_logdets! — Per-point log|C_i| for Algorithm 1 penalty
# =============================================================================

"""
    measurement_logdets!(out, p::HomographyProblem, H)

Per-point covariance penalty ℓᵢ = log|Σ̃_{gᵢ}| (Algorithm 1, Eq. 12).

For homography with isotropic noise: Σ̃_{gᵢ} = (∂ₓgᵢ)(∂ₓgᵢ)ᵀ (2×2 Sampson matrix).

Note: used only in Phase 3 of `_try_model!` (re-scoring after LO refit).
Phase 1 computes ℓᵢ via `residual_jacobian` to avoid duplicate computation.
"""
function measurement_logdets!(out::AbstractVector, p::HomographyProblem{T},
                              H::SMatrix{3,3,T,9}) where T
    u₁ = p.cs.first; u₂ = p.cs.second
    @inbounds for i in eachindex(out, u₁, u₂)
        _, G_x, _ = _sampson_quantities(H, u₁[i], u₂[i])

        # C = G_x G_xᵀ (2×2), det(C) = c11*c22 - c21²
        c11 = G_x[1,1]^2 + G_x[1,2]^2 + G_x[1,3]^2 + G_x[1,4]^2
        c21 = G_x[2,1]*G_x[1,1] + G_x[2,2]*G_x[1,2] + G_x[2,3]*G_x[1,3] + G_x[2,4]*G_x[1,4]
        c22 = G_x[2,1]^2 + G_x[2,2]^2 + G_x[2,3]^2 + G_x[2,4]^2

        det_c = c11 * c22 - c21 * c21
        out[i] = det_c > zero(T) ? log(det_c) : zero(T)
    end
    return out
end
