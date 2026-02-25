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

function solve(p::FundamentalMatrixProblem{T}, idx::AbstractVector{Int}) where T
    u₁ = p.cs.first; u₂ = p.cs.second
    result = @inbounds fundamental_matrix_7pt(
        u₁[idx[1]], u₁[idx[2]], u₁[idx[3]], u₁[idx[4]],
        u₁[idx[5]], u₁[idx[6]], u₁[idx[7]],
        u₂[idx[1]], u₂[idx[2]], u₂[idx[3]], u₂[idx[4]],
        u₂[idx[5]], u₂[idx[6]], u₂[idx[7]], T)
    isnothing(result) && return nothing

    # Union-split: 1 solution (FundamentalMat) or 3 solutions (SVector{3})
    if result isa FundamentalMat{T}
        return FixedModels{1, FundamentalMat{T}}(1, (result,))
    else
        F₁, F₂, F₃ = result[1], result[2], result[3]
        return FixedModels{3, FundamentalMat{T}}(3, (F₁, F₂, F₃))
    end
end

residuals!(r::Vector, p::FundamentalMatrixProblem{T}, F::FundamentalMat{T}) where T =
    sampson_distances!(r, F, p.cs)

# No pre-solve degeneracy check: the 7pt solver detects true degeneracy via
# SVD kernel dimension. Collinearity checks on C(7,3)=35 triplets per image
# reject ~75% of valid all-inlier samples (matching SuperANSAC's approach).
test_sample(::FundamentalMatrixProblem, ::AbstractVector{Int}) = true

test_model(p::FundamentalMatrixProblem{T}, F::FundamentalMat{T}, idx::AbstractVector{Int}) where T =
    @inbounds test_model(F, SVector(ntuple(i -> p.cs.first[idx[i]], Val(7))),
                            SVector(ntuple(i -> p.cs.second[idx[i]], Val(7))))

# =============================================================================
# fit — Weighted DLT for LO-RANSAC
# =============================================================================

function fit(p::FundamentalMatrixProblem, mask::BitVector, weights::AbstractVector, ::LinearFit)
    sum(mask) < 8 && return nothing
    fundamental_matrix_dlt!(p._dlt_buf, p.cs.first, p.cs.second;
                             mask, weights, svd_ws=p._svd_ws)
end
