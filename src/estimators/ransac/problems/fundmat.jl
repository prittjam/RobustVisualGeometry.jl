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
    FundamentalMatrixProblem{T,S} <: AbstractCspondProblem{T}

RANSAC problem for estimating a fundamental matrix from point correspondences.

Estimates F such that `u₂ᵀ F u₁ = 0` (in homogeneous coordinates).

Type parameters:
- `T`: Element type (Float64, etc.)
- `S <: AbstractSampler`: Sampling strategy (uniform, PROSAC)

# Constructor
```julia
cs = [SA[1.0,2.0] => SA[3.0,4.0], ...]
FundamentalMatrixProblem(cs)
```

# Solver Details
- Minimal sample: 7 point pairs (`MultipleSolutions`, returns 1-3 F matrices)
- Model type: `FundamentalMat{T}` (Frobenius-normalized, F[3,3] >= 0)
- Residual: Sampson distance
- Degeneracy: Oriented epipolar constraint
"""
struct FundamentalMatrixProblem{T<:AbstractFloat, S<:AbstractSampler} <: AbstractCspondProblem{T}
    cs::StructArrays.StructVector{Pair{SVector{2,T},SVector{2,T}}, @NamedTuple{first::Vector{SVector{2,T}}, second::Vector{SVector{2,T}}}}
    _sampler::S
    _dlt_buf::FixedSizeArray{T,2,Memory{T}}
    _svd_ws::SVDWorkspace{T}
end

FundamentalMatrixProblem(correspondences::AbstractVector) =
    _make_cspond_problem(FundamentalMatrixProblem, correspondences, 7, identity)

# =============================================================================
# AbstractRansacProblem Interface
# =============================================================================

sample_size(::FundamentalMatrixProblem) = 7
codimension(::FundamentalMatrixProblem) = 1  # d_g = 1: one scalar epipolar constraint
model_type(::FundamentalMatrixProblem{T}) where T = FundamentalMat{T}
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

    # Union-split: 1 solution (FundamentalMat) or 3 solutions (SVector{3})
    if result isa FundamentalMat{T}
        _oriented_epipolar_check(u₁s, u₂s, result) || return nothing
        return FixedModels{1, FundamentalMat{T}}(1, (result,))
    else
        # SVector{3, FundamentalMat{T}} — filter with oriented epipolar check
        n_out = 0
        G₁ = G₂ = G₃ = FundamentalMat{T}(Tuple(zero(SMatrix{3,3,T,9})))
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
        return FixedModels{3, FundamentalMat{T}}(n_out, (G₁, G₂, G₃))
    end
end

residuals!(r::Vector, p::FundamentalMatrixProblem{T}, F::FundamentalMat{T}) where T =
    sampson_distances!(r, F, p.cs)

# No pre-solve degeneracy check: the 7pt solver detects true degeneracy via
# SVD kernel dimension. Collinearity checks on C(7,3)=35 triplets per image
# reject ~75% of valid all-inlier samples (matching SuperANSAC's approach).
test_sample(::FundamentalMatrixProblem, ::Vector{Int}) = true

# NOTE: test_consensus defaults to `true`. The oriented epipolar check cannot
# be used on the consensus set because inliers spread across the image may
# lie on opposite sides of the epipole (valid geometry, not degenerate).
# The check is applied in solve() on the 7 sample points only.

# =============================================================================
# fit — Weighted DLT for LO-RANSAC
# =============================================================================

function fit(p::FundamentalMatrixProblem, mask::BitVector, weights::AbstractVector, ::LinearFit)
    sum(mask) < 8 && return nothing
    fundamental_matrix_dlt!(p._dlt_buf, p.cs.first, p.cs.second;
                             mask, weights, svd_ws=p._svd_ws)
end
