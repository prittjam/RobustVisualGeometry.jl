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

# Dependencies: VGC (homography_4pt, _transfer_error_jacobian_wrt_h, etc.)
#               All estimation types available from parent module

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
- Model type: `HomographyMat{T}` (Frobenius-normalized, H[3,3] >= 0)
- Residual: Sampson distance (EIV-corrected)
- Degeneracy: Collinearity + convexity check
- Refinement: Controlled by `R` type parameter
"""
struct HomographyProblem{T<:AbstractFloat, S<:AbstractSampler, R<:AbstractRefinement} <: AbstractCspondProblem{T}
    cs::StructArrays.StructVector{Pair{SVector{2,T},SVector{2,T}}, @NamedTuple{first::Vector{SVector{2,T}}, second::Vector{SVector{2,T}}}}
    _sampler::S
    _refinement::R
    _dlt_buf::FixedSizeArray{T,2,Memory{T}}
    _svd_ws::SVDWorkspace{T}
end

HomographyProblem(correspondences::AbstractVector;
                  refinement::AbstractRefinement=NoRefinement()) =
    _make_cspond_problem(HomographyProblem, correspondences, 4, n -> 2n; refinement)

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
model_type(::HomographyProblem{T}) where T = HomographyMat{T}
solver_cardinality(::HomographyProblem) = SingleSolution()
codimension(::HomographyProblem) = 2  # d_g = 2: two constraint equations from v̄ = λHū
measurement_covariance(::HomographyProblem) = Homoscedastic()

function solve(p::HomographyProblem, idx::Vector{Int})
    u₁ = p.cs.first; u₂ = p.cs.second
    @inbounds homography_4pt(
        u₁[idx[1]], u₁[idx[2]], u₁[idx[3]], u₁[idx[4]],
        u₂[idx[1]], u₂[idx[2]], u₂[idx[3]], u₂[idx[4]])
end

residuals!(r::Vector, p::HomographyProblem{T}, H::HomographyMat{T}) where T =
    sampson_distances!(r, H, p.cs)

function test_sample(p::HomographyProblem{T}, idx::Vector{Int}) where T
    u₁ = p.cs.first; u₂ = p.cs.second
    @inbounds _homography_sample_nondegenerate(
        u₁[idx[1]], u₁[idx[2]], u₁[idx[3]], u₁[idx[4]],
        u₂[idx[1]], u₂[idx[2]], u₂[idx[3]], u₂[idx[4]])
end

function test_model(p::HomographyProblem{T}, H::HomographyMat{T},
                    idx::Vector{Int}) where T
    u₁ = p.cs.first
    @inbounds for j in 1:4
        dj = jac_det(H, u₁[idx[j]])
        (dj ≤ T(1e-4) || dj ≥ T(1e4)) && return false
    end
    return true
end

# =============================================================================
# AbstractCspondProblem Dispatch Points
# =============================================================================

min_dlt_inliers(::HomographyProblem) = 5

function _dlt_refit(p::HomographyProblem; mask=nothing, weights=nothing)
    homography_dlt!(p._dlt_buf, p.cs.first, p.cs.second;
                     mask, weights, svd_ws=p._svd_ws)
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
                                     H::HomographyMat{T}) where T
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
# Sampson Error — Algebraic Constraint Formulation (Chum et al.)
# =============================================================================
#
# For correspondence (s, d) = ((x,y), (x',y')) and homography H, the algebraic
# constraint g = 0 encodes d̃ ∝ H·s̃:
#
#   g₁ = h₁ᵀs̃ − x'·h₃ᵀs̃
#   g₂ = h₂ᵀs̃ − y'·h₃ᵀs̃
#
# The measurement Jacobian G_x (2×4) w.r.t. (x, y, x', y'):
#   G_x = [ H[1,1]−x'H[3,1]   H[1,2]−x'H[3,2]   −c    0  ]
#         [ H[2,1]−y'H[3,1]   H[2,2]−y'H[3,2]    0   −c  ]
# where c = h₃ᵀs̃.
#
# The projected covariance (isotropic Σ_i = I₄):
#   C_i = G_x G_xᵀ   (2×2 Sampson matrix)
#
# Sampson distance: d_S = sqrt(g' C⁻¹ g)
# Whitened residual: r_w = L⁻¹g where LL' = C
#
# The model Jacobian G_θ (2×9) w.r.t. vec(H) (column-major):
#   ∂g₁/∂vec(H) = [x, 0, −x'x,  y, 0, −x'y,  1, 0, −x']
#   ∂g₂/∂vec(H) = [0, x, −y'x,  0, y, −y'y,  0, 1, −y']
#
# =============================================================================

# =============================================================================
# residual_jacobian — delegates to VGC's sampson_whitened
# =============================================================================

residual_jacobian(p::HomographyProblem{T}, H::HomographyMat{T}, i::Int) where T =
    @inbounds sampson_whitened(H, p.cs[i])

# =============================================================================
# measurement_logdets! — delegates to VGC's sampson_logdets!
# =============================================================================

measurement_logdets!(out::AbstractVector, p::HomographyProblem{T},
                     H::HomographyMat{T}) where T =
    sampson_logdets!(out, H, p.cs)
