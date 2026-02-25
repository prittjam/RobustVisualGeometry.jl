# =============================================================================
# Shared Infrastructure for Correspondence-based RANSAC Problems
# =============================================================================
#
# Two-level type hierarchy:
#
#   AbstractCspondProblem <: AbstractRansacProblem
#     Thin base for any correspondence problem (cs + sampler).
#     Covers 3D-2D (P3P) and 2D-2D (homography, F-matrix) problems.
#
#   AbstractDltProblem{T} <: AbstractCspondProblem
#     Adds DLT buffer + SVDWorkspace for 2D-2D linear algebraic solvers.
#     HomographyProblem and FundMatProblem inherit from this.
#
# =============================================================================

# Dependencies: StructArrays
#               All estimation types available from parent module

# =============================================================================
# Abstract Types
# =============================================================================

"""
    AbstractCspondProblem <: AbstractRansacProblem

Base type for RANSAC problems defined by point correspondences.

Subtypes must have fields `cs` (a `StructVector` of `Pair`s) and
`_sampler` (an `AbstractSampler`). Provides default implementations of
`data_size(p) = length(p.cs)` and `sampler(p) = p._sampler`.

Concrete subtypes:
- [`AbstractDltProblem{T}`](@ref) — 2D-2D problems with DLT buffer (homography, F-matrix)
- [`P3PProblem`](@ref) — 3D-2D pose estimation from bearing rays
"""
abstract type AbstractCspondProblem <: AbstractRansacProblem end

"""
    AbstractDltProblem{T} <: AbstractCspondProblem

Base type for 2D-2D correspondence problems that use a DLT linear solver.

Adds pre-allocated `_dlt_buf::FixedSizeArray{T}` and `_svd_ws::SVDWorkspace{T}`
for zero-allocation weighted DLT fitting in LO-RANSAC.

Concrete subtypes: [`HomographyProblem`](@ref), [`FundMatProblem`](@ref).
"""
abstract type AbstractDltProblem{T} <: AbstractCspondProblem end

# =============================================================================
# Shared Interface Methods
# =============================================================================

data_size(p::AbstractCspondProblem) = length(p.cs)
sampler(p::AbstractCspondProblem) = p._sampler

# =============================================================================
# Shared Constructor Helpers (DLT problems)
# =============================================================================

"""
    _make_cspond_problem(P, correspondences, sample_sz, dlt_rows_fn)

Shared constructor for DLT correspondence problems.
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
