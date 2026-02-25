# =============================================================================
# RANSAC Samplers — Uniform and PROSAC sampling strategies
# =============================================================================

# -----------------------------------------------------------------------------
# Abstract Sampler Type
# -----------------------------------------------------------------------------

"""
    AbstractSampler

Abstract type for RANSAC sampling strategies.

Subtypes implement `draw!` and `reset!` to provide different sampling behaviors
(uniform, PROSAC, NAPSAC, etc.). Problems store a sampler as a type-parameterized
field for compile-time dispatch.

# Required Methods
- `draw!(indices::Vector{Int}, sampler)` — fill indices with a random sample
- `reset!(sampler)` — reset mutable state for reuse across `ransac()` calls

# Built-in Samplers
- `UniformSampler`: Stateless uniform random sampling
- `ProsacSampler`: Progressive sampling from high-quality matches
"""
abstract type AbstractSampler end

"""
    UniformSampler <: AbstractSampler

Stateless uniform random sampling without replacement.

Stores the data size `n` for sampling from `1:n`.
"""
struct UniformSampler <: AbstractSampler
    n::Int
end

draw!(indices::Vector{Int}, s::UniformSampler) = (_draw_uniform!(indices, s.n); nothing)
reset!(::UniformSampler) = nothing

"""
    _draw_uniform!(indices, n, k=length(indices))

Fill `indices[1:k]` with a uniform random sample without replacement from 1:n.
"""
function _draw_uniform!(indices::Vector{Int}, n::Int, k::Int=length(indices))
    @inbounds for i in 1:k
        while true
            j = rand(1:n)
            duplicate = false
            for q in 1:(i-1)
                indices[q] == j && (duplicate = true; break)
            end
            if !duplicate
                indices[i] = j
                break
            end
        end
    end
    nothing
end

# =============================================================================
# PROSAC Sampler (Chum & Matas, CVPR 2005)
# =============================================================================
#
# Progressive Sample Consensus: samples high-quality correspondences first,
# then gradually expands the pool to include all data. Degenerates to standard
# RANSAC after T_N iterations.
#
# Reference: Chum & Matas, "Matching with PROSAC — Progressive Sample Consensus"
#

"""
    ProsacSampler

Mutable state for PROSAC progressive sampling.

Maintains a pool of correspondences sorted by quality score. Sampling starts
from the top-m highest-scored matches and progressively expands the pool.

# Fields
- `sorted_indices::Vector{Int}` — data indices sorted by score (best first)
- `m::Int` — sample size (constant, e.g. 4 for homography)
- `N::Int` — total data size (constant)
- `T_N::Int` — total iteration budget (constant)
- `n::Int` — current pool size (grows from m to N)
- `t::Int` — current iteration count
- `T_n::Float64` — current T_n value (recurrence relation)
- `T_n_prime::Int` — discretized cumulative threshold

See also: `reset!`, `_draw_prosac!`
"""
mutable struct ProsacSampler <: AbstractSampler
    sorted_indices::Vector{Int}
    m::Int
    N::Int
    T_N::Int
    n::Int
    t::Int
    T_n::Float64
    T_n_prime::Int
end

"""
    _prosac_initial_Tn(m, N, T_N) -> Float64

Compute the initial T_n value for PROSAC: T_N * C(m,m)/C(N,m).
"""
function _prosac_initial_Tn(m::Int, N::Int, T_N::Int)
    T_n = one(Float64)
    for i in 0:(m-1)
        T_n *= (m - i) / (N - i)
    end
    T_n * T_N
end

"""
    ProsacSampler(sorted_indices, m; T_N=200_000)

Create a PROSAC sampler from pre-sorted indices.

`sorted_indices` must be ordered by decreasing quality score.
`m` is the minimal sample size.
`T_N` is the maximum iteration budget (default: 200,000).
"""
function ProsacSampler(sorted_indices::Vector{Int}, m::Int; T_N::Int=200_000)
    N = length(sorted_indices)
    N >= m || throw(ArgumentError("Need at least $m data points for PROSAC, got $N"))
    ProsacSampler(sorted_indices, m, N, T_N, m, 0, _prosac_initial_Tn(m, N, T_N), 1)
end

"""
    reset!(ps::ProsacSampler)

Reset PROSAC sampler state for reuse (e.g., across multiple `ransac()` calls).
"""
function reset!(ps::ProsacSampler)
    ps.n = ps.m
    ps.t = 0
    ps.T_n = _prosac_initial_Tn(ps.m, ps.N, ps.T_N)
    ps.T_n_prime = 1
    nothing
end

"""
    _draw_prosac!(indices, ps::ProsacSampler)

Draw a PROSAC sample into `indices`.

Implements the Chum & Matas growth function:
1. Increment t, expand pool n while t > T'_n
2. If in PROSAC phase: sample m-1 from U_{n-1}, force index n
3. If in RANSAC phase: sample m from U_n (uniform)
"""
function _draw_prosac!(indices::Vector{Int}, ps::ProsacSampler)
    ps.t += 1
    m = ps.m

    # Expand pool: grow n while t exceeds T'_n threshold
    while ps.n < ps.N && ps.t > ps.T_n_prime
        # Recurrence: T_{n+1} = T_n * (n+1) / (n+1-m)
        ps.T_n *= (ps.n + 1) / (ps.n + 1 - m)
        ps.n += 1
        ps.T_n_prime = floor(Int, ps.T_n)
    end

    n = ps.n

    if ps.t > ps.T_n_prime
        # RANSAC mode: uniform sample of m from top-n
        _draw_uniform!(indices, n)
        # Map to actual data indices
        @inbounds for i in eachindex(indices)
            indices[i] = ps.sorted_indices[indices[i]]
        end
    else
        # PROSAC mode: sample m-1 from U_{n-1}, force u_n
        if m > 1
            _draw_uniform!(indices, n - 1, m - 1)
            @inbounds for i in 1:(m-1)
                indices[i] = ps.sorted_indices[indices[i]]
            end
        end
        @inbounds indices[m] = ps.sorted_indices[n]
    end
    nothing
end

draw!(indices::Vector{Int}, s::ProsacSampler) = (_draw_prosac!(indices, s); nothing)
