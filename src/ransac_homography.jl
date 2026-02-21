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

# Dependencies: VGC (homography_4pt, symmetric_transfer_error, etc.)
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
- Residual: Symmetric transfer error
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

function solve(p::HomographyProblem, idx::Vector{Int})
    u₁ = p.cs.first; u₂ = p.cs.second
    @inbounds homography_4pt(
        u₁[idx[1]], u₁[idx[2]], u₁[idx[3]], u₁[idx[4]],
        u₂[idx[1]], u₂[idx[2]], u₂[idx[3]], u₂[idx[4]])
end

function residuals!(r::Vector, p::HomographyProblem{T}, H::SMatrix{3,3,T,9}) where T
    u₁ = p.cs.first; u₂ = p.cs.second
    @inbounds for i in eachindex(r, u₁, u₂)
        r[i] = symmetric_transfer_error(u₁[i], u₂[i], H)
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

# Scale-fixing augmentation: add the λ₄=1 constraint row to make the
# information matrix full rank (9×9 instead of rank-8).
#
# The constraint h₃ᵀũ = 1 (projective depth = 1 at the reference point)
# adds the missing rank direction. The reference point is the first inlier.
# Choice of reference doesn't affect prediction variances (scale-invariant).
function _augment_info(p::HomographyProblem{T}, info, mask) where T
    ref_idx = findfirst(mask)
    isnothing(ref_idx) && return info
    s_ref = p.cs.first[ref_idx]
    # Scale-fixing row: ∂(h₃ᵀũ)/∂vec(H) in column-major order
    # vec(H) = [H[1,1],H[2,1],H[3,1], H[1,2],H[2,2],H[3,2], H[1,3],H[2,3],H[3,3]]
    # h₃ᵀũ = H[3,1]·s₁ + H[3,2]·s₂ + H[3,3]·1 → positions 3, 6, 9
    c = SVector{9,T}(0, 0, s_ref[1], 0, 0, s_ref[2], 0, 0, one(T))
    return info + c * c'
end

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
# residual_jacobian — Forward Transfer Error (2D residual, 2×9 Jacobian)
# =============================================================================

"""
    residual_jacobian(p::HomographyProblem, H, i) -> (r::SVector{2}, G::SMatrix{2,9})

Forward transfer residual and its Jacobian w.r.t. vec(H).

Returns:
- `r = proj(H·[sᵢ;1]) - dᵢ` (2D forward residual vector)
- `G = ∂r/∂vec(H)` (2×9 transfer error Jacobian)

Note: `residuals!` returns scalar symmetric_transfer_error, but
`residual_jacobian` returns the 2D forward-only vector — these are
intentionally different (scoring uses symmetric, prediction uses forward).
"""
function residual_jacobian(p::HomographyProblem{T},
                                       H::SMatrix{3,3,T,9}, i::Int) where T
    @inbounds begin
        si = p.cs.first[i]
        di = p.cs.second[i]
    end

    # Forward residual vector (2D)
    h = H * SA[si[1], si[2], one(T)]
    w = h[3]
    inv_w = one(T) / w
    r = SVector{2,T}(h[1] * inv_w - di[1], h[2] * inv_w - di[2])

    # Transfer error Jacobian w.r.t. vec(H)  (2×9)
    G = _transfer_error_jacobian_wrt_h(si, di, H)

    return (r, G)
end

# =============================================================================
# Transfer Jacobian w.r.t. Source Point (2×2)
# =============================================================================

"""
    _transfer_jacobian_wrt_source(H, s) -> SMatrix{2,2}

Jacobian of `proj(H · [s; 1])` w.r.t. the 2D source point `s`.

Used to account for source-point noise in the information matrix (EIV model).
"""
@inline function _transfer_jacobian_wrt_source(H::SMatrix{3,3,T,9},
                                                s::SVector{2,T}) where T
    s̃ = SA[s[1], s[2], one(T)]
    Hs = H * s̃
    w = Hs[3]
    inv_w = one(T) / w
    px = Hs[1] * inv_w
    py = Hs[2] * inv_w
    SMatrix{2,2,T}(
        (H[1,1] - px * H[3,1]) * inv_w,
        (H[2,1] - py * H[3,1]) * inv_w,
        (H[1,2] - px * H[3,2]) * inv_w,
        (H[2,2] - py * H[3,2]) * inv_w)
end

# =============================================================================
# EIV-corrected Information Matrix Accumulation
# =============================================================================

"""
    _accumulate_info(p::HomographyProblem, H, mask)

EIV-corrected Fisher information for homography estimation.

The forward transfer residual `r = proj(H·[s;1]) - d` has covariance
`C_i = σ²(I₂ + J_s J_s')` when both source and destination have iid noise σ²I₂.
The correct information contribution per point is `G'C̃⁻¹G` where `C̃ = C_i/σ²`.

Without this correction, the information matrix overestimates precision by ~2×
(it assumes only destination noise contributes to the residual variance).
"""
function _accumulate_info(p::HomographyProblem{T},
                                      H::SMatrix{3,3,T,9}, mask) where T
    info = nothing
    @inbounds for i in eachindex(mask)
        if mask[i]
            _, G = residual_jacobian(p, H, i)
            si = p.cs.first[i]
            J_s = _transfer_jacobian_wrt_source(H, si)
            # Normalized residual covariance: C̃ = I₂ + J_s J_s'
            C̃ = SMatrix{2,2,T}(I) + J_s * J_s'
            contrib = G' * inv(C̃) * G
            info = isnothing(info) ? contrib : info + contrib
        end
    end
    isnothing(info) && return nothing
    return _augment_info(p, info, mask)
end
