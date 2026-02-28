# =============================================================================
# RANSAC Fundamental Matrix Problem
# =============================================================================
#
# Implements AbstractDltProblem for fundamental matrix estimation
# from point correspondences.
#
# PLACEMENT: Included from main VisualGeometryCore.jl (NOT from Estimators
# submodule) because it depends on public solver functions from
# geometry/fundamental_matrix.jl (loaded after Estimators).
#
# =============================================================================

# Dependencies: VGC (fundmat_7pt, sampson_distance, enforce_rank_two, etc.)
#               All estimation types available from parent module

# =============================================================================
# FundMatProblem Type
# =============================================================================

"""
    FundMatProblem{T,S} <: AbstractDltProblem{T}

RANSAC problem for estimating a fundamental matrix from point correspondences.

Estimates F such that `u₂ᵀ F u₁ = 0` (in homogeneous coordinates).

Type parameters:
- `T`: Element type (Float64, etc.)
- `S <: AbstractSampler`: Sampling strategy (uniform, PROSAC)

# Constructor
```julia
cs = [SA[1.0,2.0] => SA[3.0,4.0], ...]
FundMatProblem(cs)
```

# Solver Details
- Minimal sample: 7 point pairs (`MultipleSolutions`, returns 1-3 F matrices)
- Model type: `FundMat{T}` (Frobenius-normalized, F[3,3] >= 0)
- Residual: Sampson distance
- Degeneracy: Oriented epipolar constraint
"""
struct FundMatProblem{T<:AbstractFloat, S<:AbstractSampler} <: AbstractDltProblem{T}
    cs::StructArrays.StructVector{Pair{SVector{2,T},SVector{2,T}}, @NamedTuple{first::Vector{SVector{2,T}}, second::Vector{SVector{2,T}}}}
    _sampler::S
    _dlt_buf::FixedSizeArray{T,2,Memory{T}}
    _svd_ws::SVDWorkspace{T}
end

FundMatProblem(correspondences::AbstractVector) =
    _make_cspond_problem(FundMatProblem, correspondences, 7, identity)

# =============================================================================
# AbstractRansacProblem Interface
# =============================================================================

sample_size(::FundMatProblem) = 7
codimension(::FundMatProblem) = 1  # d_g = 1: one scalar epipolar constraint
model_type(::FundMatProblem{T}) where T = FundMat{T}
solver_cardinality(::FundMatProblem) = MultipleSolutions()

# Inline indexing to call the 14-arg solver directly — avoids heap-boxing
# the Union{Nothing, FixedModels} return across the non-inlined VGC wrapper.
function solve(p::FundMatProblem{T}, idx::AbstractVector{Int}) where T
    u1 = p.cs.first; u2 = p.cs.second
    @inbounds fundmat_7pt(
        u1[idx[1]], u1[idx[2]], u1[idx[3]], u1[idx[4]],
        u1[idx[5]], u1[idx[6]], u1[idx[7]],
        u2[idx[1]], u2[idx[2]], u2[idx[3]], u2[idx[4]],
        u2[idx[5]], u2[idx[6]], u2[idx[7]], T)
end

# No pre-solve degeneracy check: the 7pt solver detects true degeneracy via
# SVD kernel dimension. Collinearity checks on C(7,3)=35 triplets per image
# reject ~75% of valid all-inlier samples (matching SuperANSAC's approach).
test_sample(::FundMatProblem, ::AbstractVector{Int}) = true

# DLT solver dispatch for AbstractDltProblem shared fit()
_dlt_solver!(::FundMatProblem, args...; kw...) = fundmat_dlt!(args...; kw...)
