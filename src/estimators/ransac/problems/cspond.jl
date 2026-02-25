# =============================================================================
# Shared Infrastructure for Correspondence-based RANSAC Problems
# =============================================================================
#
# AbstractCspondProblem{T} factors out the common structure shared by
# HomographyProblem and FundamentalMatrixProblem: constructor helpers,
# gather routines, and strategy-dispatched refinement.
#
# Each concrete problem implements:
#   min_dlt_inliers(p) — minimum inlier count for DLT refit
#   _dlt_refit(p, k)   — problem-specific DLT on first k gathered points
#
# =============================================================================

# Dependencies: StructArrays, FixedSizeArrays
#               All estimation types available from parent module

# =============================================================================
# Abstract Type
# =============================================================================

abstract type AbstractCspondProblem{T} <: AbstractRansacProblem end

# =============================================================================
# Shared Interface Methods
# =============================================================================

data_size(p::AbstractCspondProblem) = length(p.cs)
sampler(p::AbstractCspondProblem) = p._sampler

# =============================================================================
# Dispatch Points (required per problem)
# =============================================================================

function min_dlt_inliers end
function _dlt_refit end

# =============================================================================
# Shared Constructor Helper
# =============================================================================

function _build_cspond(correspondences::AbstractVector, sample_sz::Int)
    n = length(correspondences)
    c₁ = first(correspondences)
    T = eltype(c₁.first)
    u₁ = Vector{SVector{2,T}}(undef, n)
    u₂ = Vector{SVector{2,T}}(undef, n)
    @inbounds for i in eachindex(correspondences)
        c = correspondences[i]
        u₁[i] = SVector{2,T}(c.first[1], c.first[2])
        u₂[i] = SVector{2,T}(c.second[1], c.second[2])
    end
    cs = StructArrays.StructArray{Pair{SVector{2,T},SVector{2,T}}}((u₁, u₂))
    smplr = _build_sampler(correspondences, sample_sz)
    u₁_buf = FixedSizeArray{SVector{2,T}}(undef, n)
    u₂_buf = FixedSizeArray{SVector{2,T}}(undef, n)
    w_buf = FixedSizeArray{T}(undef, n)
    return cs, smplr, u₁_buf, u₂_buf, w_buf
end

# =============================================================================
# Gather Helpers
# =============================================================================

function _gather_cspond_inliers!(p::AbstractCspondProblem, mask::BitVector)
    k = _gather_masked!(p._u₁_buf, p.cs.first, mask)
    _gather_masked!(p._u₂_buf, p.cs.second, mask)
    return k
end

function _gather_cspond_inliers!(p::AbstractCspondProblem, mask::BitVector, w)
    k = _gather_masked!(p._u₁_buf, p.cs.first, mask)
    _gather_masked!(p._u₂_buf, p.cs.second, mask)
    _gather_masked!(p._w_buf, w, mask)
    return k
end

# =============================================================================
# Generic Refinement Bridge
# =============================================================================

refine(p::AbstractCspondProblem, model, mask::BitVector) =
    _ransac_refine(p._refinement, p, model, mask)
refine(p::AbstractCspondProblem, model, mask::BitVector,
                  loss::AbstractLoss, σ::Real) =
    _ransac_refine(p._refinement, p, model, mask, loss, σ)

# =============================================================================
# Generic Refinement Strategies
# =============================================================================

_ransac_refine(::NoRefinement, ::AbstractCspondProblem, args...) = nothing

function _ransac_refine(::DltRefinement, p::AbstractCspondProblem{T},
                        _model, mask::BitVector) where T
    n_inliers = sum(mask)
    n_inliers < min_dlt_inliers(p) && return nothing
    _gather_cspond_inliers!(p, mask)
    model_ref = _dlt_refit(p, n_inliers)
    isnothing(model_ref) && return nothing
    return (model_ref, one(T))
end

_ransac_refine(::DltRefinement, p::AbstractCspondProblem, model, mask,
               _loss, _σ) =
    _ransac_refine(DltRefinement(), p, model, mask)

_ransac_refine(::IrlsRefinement, p::AbstractCspondProblem, model, mask) =
    _ransac_refine(DltRefinement(), p, model, mask)

function _ransac_refine(r::IrlsRefinement, p::AbstractCspondProblem{T},
                        model, mask::BitVector, loss::AbstractLoss,
                        σ::Real) where T
    adapter = RansacRefineProblem(p, mask, p._svd_ws)
    result = robust_solve(adapter, MEstimator(loss);
                          init=model, scale=FixedScale(σ=Float64(σ)),
                          max_iter=r.max_iter)
    (result.value, σ)
end
