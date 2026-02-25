# =============================================================================
# RANSAC Fundamental Matrix Problem
# =============================================================================
#
# Implements AbstractCspondProblem for fundamental matrix estimation
# from point correspondences.
#
# PLACEMENT: Included from main VisualGeometryCore.jl (NOT from Estimators
# submodule) because it depends on public solver functions from
# geometry/fundamental_matrix.jl (loaded after Estimators).
#
# =============================================================================

# Dependencies: VGC (fundamental_matrix_7pt, sampson_distance, enforce_rank_two, etc.)
#               All estimation types available from parent module

# =============================================================================
# FundamentalMatrixProblem Type
# =============================================================================

"""
    FundamentalMatrixProblem{T,S,R} <: AbstractCspondProblem{T}

RANSAC problem for estimating a fundamental matrix from point correspondences.

Estimates F such that `u₂ᵀ F u₁ = 0` (in homogeneous coordinates).

Type parameters:
- `T`: Element type (Float64, etc.)
- `S <: AbstractSampler`: Sampling strategy (uniform, PROSAC)
- `R <: AbstractRefinement`: LO-RANSAC refinement strategy

# Constructor
```julia
cs = [SA[1.0,2.0] => SA[3.0,4.0], ...]
FundamentalMatrixProblem(cs)                                  # plain RANSAC (default)
FundamentalMatrixProblem(cs; refinement=DltRefinement())      # DLT refit only
FundamentalMatrixProblem(cs; refinement=IrlsRefinement())     # LO-RANSAC (IRLS)
```

# Solver Details
- Minimal sample: 7 point pairs (`MultipleSolutions`, returns 1-3 F matrices)
- Model type: `SMatrix{3,3,T,9}` (Frobenius-normalized, F[3,3] >= 0)
- Residual: Sampson distance
- Degeneracy: Oriented epipolar constraint
- Refinement: Controlled by `R` type parameter
"""
struct FundamentalMatrixProblem{T<:AbstractFloat, S<:AbstractSampler, R<:AbstractRefinement} <: AbstractCspondProblem{T}
    cs::StructArrays.StructVector{Pair{SVector{2,T},SVector{2,T}}, @NamedTuple{first::Vector{SVector{2,T}}, second::Vector{SVector{2,T}}}}
    _sampler::S
    _refinement::R
    _dlt_buf::FixedSizeArray{T,2,Memory{T}}            # N×9 (one row per correspondence)
    _u₁_buf::FixedSizeArray{SVector{2,T},1,Memory{SVector{2,T}}}
    _u₂_buf::FixedSizeArray{SVector{2,T},1,Memory{SVector{2,T}}}
    _w_buf::FixedSizeArray{T,1,Memory{T}}
    _svd_ws::SVDWorkspace{T}
end

"""
    FundamentalMatrixProblem(correspondences; refinement=NoRefinement())

Construct a `FundamentalMatrixProblem` from a vector of correspondences.

Accepts any correspondence type with `.first` and `.second` properties
returning 2D points (works with `Pair`, `Attributed`, `ScoredCspond`).

Requires at least 7 correspondences.
"""
function FundamentalMatrixProblem(correspondences::AbstractVector;
                                  refinement::AbstractRefinement=NoRefinement())
    n = length(correspondences)
    n >= 7 || throw(ArgumentError("Need at least 7 correspondences, got $n"))

    cs, smplr, u₁_buf, u₂_buf, w_buf = _build_cspond(correspondences, 7)
    T = eltype(first(correspondences).first)

    FundamentalMatrixProblem{T, typeof(smplr), typeof(refinement)}(
        cs, smplr, refinement,
        FixedSizeArray{T}(undef, n, 9),
        u₁_buf, u₂_buf, w_buf,
        SVDWorkspace{T}(n, 9))
end

"""
    LoFundamentalMatrixProblem(correspondences; refinement=IrlsRefinement())

Convenience constructor for LO-RANSAC fundamental matrix estimation.

Equivalent to `FundamentalMatrixProblem(cs; refinement=IrlsRefinement())`.
Returns a `FundamentalMatrixProblem` with IRLS refinement enabled by default.
"""
LoFundamentalMatrixProblem(correspondences::AbstractVector;
                           refinement::AbstractRefinement=IrlsRefinement()) =
    FundamentalMatrixProblem(correspondences; refinement)

# =============================================================================
# AbstractRansacProblem Interface
# =============================================================================

sample_size(::FundamentalMatrixProblem) = 7
codimension(::FundamentalMatrixProblem) = 1  # d_g = 1: one scalar epipolar constraint
model_type(::FundamentalMatrixProblem{T}) where T = SMatrix{3,3,T,9}
solver_cardinality(::FundamentalMatrixProblem) = MultipleSolutions()

function solve(p::FundamentalMatrixProblem{T}, idx::Vector{Int}) where T
    u₁ = p.cs.first; u₂ = p.cs.second
    result = @inbounds fundamental_matrix_7pt(
        u₁[idx[1]], u₁[idx[2]], u₁[idx[3]], u₁[idx[4]],
        u₁[idx[5]], u₁[idx[6]], u₁[idx[7]],
        u₂[idx[1]], u₂[idx[2]], u₂[idx[3]], u₂[idx[4]],
        u₂[idx[5]], u₂[idx[6]], u₂[idx[7]], T)
    isnothing(result) && return nothing

    # Oriented epipolar check on SAMPLE points only (not all data).
    # Stack-allocated sample SVectors avoid Vector allocation from u₁[idx].
    @inbounds u₁s = SVector(u₁[idx[1]], u₁[idx[2]], u₁[idx[3]], u₁[idx[4]],
                             u₁[idx[5]], u₁[idx[6]], u₁[idx[7]])
    @inbounds u₂s = SVector(u₂[idx[1]], u₂[idx[2]], u₂[idx[3]], u₂[idx[4]],
                             u₂[idx[5]], u₂[idx[6]], u₂[idx[7]])

    # Union-split: 1 solution (SMatrix) or 3 solutions (SVector{3})
    if result isa SMatrix{3,3,T,9}
        _oriented_epipolar_check(u₁s, u₂s, result) || return nothing
        return FixedModels{1, SMatrix{3,3,T,9}}(1, (result,))
    else
        # SVector{3, SMatrix{3,3,T,9}} — filter with oriented epipolar check
        n_out = 0
        G₁ = G₂ = G₃ = zero(SMatrix{3,3,T,9})
        for i in 1:3
            F = result[i]
            _oriented_epipolar_check(u₁s, u₂s, F) || continue
            n_out += 1
            if n_out == 1; G₁ = F
            elseif n_out == 2; G₂ = F
            else; G₃ = F
            end
        end
        n_out == 0 && return nothing
        return FixedModels{3, SMatrix{3,3,T,9}}(n_out, (G₁, G₂, G₃))
    end
end

function residuals!(r::Vector, p::FundamentalMatrixProblem{T}, F::SMatrix{3,3,T,9}) where T
    u₁ = p.cs.first; u₂ = p.cs.second
    @inbounds for i in eachindex(r, u₁, u₂)
        r[i] = sampson_distance(u₁[i], u₂[i], F)
    end
    return r
end

# No pre-solve degeneracy check: the 7pt solver detects true degeneracy via
# SVD kernel dimension. Collinearity checks on C(7,3)=35 triplets per image
# reject ~75% of valid all-inlier samples (matching SuperANSAC's approach).
test_sample(::FundamentalMatrixProblem, ::Vector{Int}) = true

# NOTE: test_consensus defaults to `true`. The oriented epipolar check cannot
# be used on the consensus set because inliers spread across the image may
# lie on opposite sides of the epipole (valid geometry, not degenerate).
# The check is applied in solve() on the 7 sample points only.

# =============================================================================
# AbstractCspondProblem Dispatch Points
# =============================================================================

min_dlt_inliers(::FundamentalMatrixProblem) = 8

function _dlt_refit(p::FundamentalMatrixProblem, k::Int)
    fundamental_matrix_dlt!(@view(p._dlt_buf[1:k, :]),
                             @view(p._u₁_buf[1:k]), @view(p._u₂_buf[1:k]);
                             svd_ws=p._svd_ws)
end

# =============================================================================
# IRLS Refinement: Weighted DLT with Sampson Correction
# =============================================================================

constraint_type(::FundamentalMatrixProblem) = Constrained()

function weighted_system(p::FundamentalMatrixProblem{T}, F, mask, w) where T
    k = _gather_cspond_inliers!(p, mask, w)
    k < min_dlt_inliers(p) && return nothing

    T₁ = hartley_normalization(@view p._u₁_buf[1:k])
    T₂ = hartley_normalization(@view p._u₂_buf[1:k])

    # Compute scalar Sampson weights in-place into _w_buf (overwriting robust weights).
    # For F-matrix: JJᵀ is scalar, so correction is just w_robust / JJᵀ.
    # Safe because we read _w_buf[i] before writing back to _w_buf[i].
    Fn = inv(T₂)' * F * inv(T₁)

    @inbounds for i in 1:k
        s = T₁ * SA[p._u₁_buf[i][1], p._u₁_buf[i][2], one(T)]
        d = T₂ * SA[p._u₂_buf[i][1], p._u₂_buf[i][2], one(T)]
        wi = p._w_buf[i]

        # Sampson Jacobian: J = [∂e/∂s₁, ∂e/∂s₂, ∂e/∂d₁, ∂e/∂d₂]
        Fns = Fn * s
        Fntd = Fn' * d

        jjt = Fntd[1]^2 + Fntd[2]^2 + Fns[1]^2 + Fns[2]^2

        if jjt > eps(T)
            p._w_buf[i] = wi / jjt
        end
    end

    A = @view p._dlt_buf[1:k, :]
    _fill_fundamental_dlt!(A, @view(p._u₁_buf[1:k]), @view(p._u₂_buf[1:k]),
                            T₁, T₂; weights=(@view p._w_buf[1:k]))

    return (A = A, T₁ = T₁, T₂ = T₂)
end

function model_from_solution(::FundamentalMatrixProblem{T}, f, sys) where T
    F_norm = _vec9_to_mat33(f, T)
    F_raw = sys.T₂' * F_norm * sys.T₁
    F_rank2 = enforce_rank_two(F_raw)
    return sign_normalize(F_rank2)
end
