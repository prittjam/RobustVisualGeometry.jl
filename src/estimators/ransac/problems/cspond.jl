# =============================================================================
# Shared Infrastructure for Correspondence-based RANSAC Problems
# =============================================================================
#
# AbstractCspondProblem{T} factors out the common structure shared by
# HomographyProblem and FundMatProblem: constructor helpers
# and shared interface methods.
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
# Shared Constructor Helpers
# =============================================================================

"""
    _make_cspond_problem(P, correspondences, sample_sz, dlt_rows_fn)

Shared constructor for correspondence-based RANSAC problems.
`dlt_rows_fn(n)` returns the number of rows for the DLT buffer.
"""
function _make_cspond_problem(::Type{P}, correspondences::AbstractVector,
                               sample_sz::Int, dlt_rows_fn) where P
    n = length(correspondences)
    n >= sample_sz || throw(ArgumentError(
        "Need at least $sample_sz correspondences, got $n"))
    cs, smplr = _build_cspond(correspondences, sample_sz)
    T = eltype(first(correspondences).first)
    n_rows = dlt_rows_fn(n)
    P{T, typeof(smplr)}(
        cs, smplr,
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
