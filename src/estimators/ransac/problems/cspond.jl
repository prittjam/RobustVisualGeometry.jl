# =============================================================================
# Shared Infrastructure for Correspondence-based RANSAC Problems
# =============================================================================
#
# AbstractCspondProblem{T} factors out the common structure shared by
# HomographyProblem and FundamentalMatrixProblem: constructor helpers
# and strategy-dispatched refinement.
#
# Each concrete problem implements:
#   min_dlt_inliers(p) — minimum inlier count for DLT refit
#   _dlt_refit(p; mask, weights) — problem-specific DLT with mask/weights
#
# =============================================================================

# Dependencies: StructArrays
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
# Shared Constructor Helpers
# =============================================================================

"""
    _make_cspond_problem(P, correspondences, sample_sz, dlt_rows_fn; refinement)

Shared constructor for correspondence-based RANSAC problems.
`dlt_rows_fn(n)` returns the number of rows for the DLT buffer.
"""
function _make_cspond_problem(::Type{P}, correspondences::AbstractVector,
                               sample_sz::Int, dlt_rows_fn;
                               refinement::AbstractRefinement=NoRefinement()) where P
    n = length(correspondences)
    n >= sample_sz || throw(ArgumentError(
        "Need at least $sample_sz correspondences, got $n"))
    cs, smplr = _build_cspond(correspondences, sample_sz)
    T = eltype(first(correspondences).first)
    n_rows = dlt_rows_fn(n)
    P{T, typeof(smplr), typeof(refinement)}(
        cs, smplr, refinement,
        FixedSizeArray{T}(undef, n_rows, 9),
        SVDWorkspace{T}(n_rows, 9))
end

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
    return cs, smplr
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
    sum(mask) < min_dlt_inliers(p) && return nothing
    model_ref = _dlt_refit(p; mask)
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

# =============================================================================
# Weighted Solve — passes mask + weights to DLT solver
# =============================================================================

function weighted_solve(a::RansacRefineProblem{<:AbstractCspondProblem}, model, ω)
    p = a.problem
    @inbounds for i in eachindex(ω, a.mask)
        a.mask[i] || (ω[i] = zero(eltype(ω)))
    end
    sum(a.mask) < min_dlt_inliers(p) && return model
    result = _dlt_refit(p; mask=a.mask, weights=ω)
    isnothing(result) && return model
    result
end
