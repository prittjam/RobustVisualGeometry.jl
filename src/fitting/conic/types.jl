# =============================================================================
# Conic Fitting Types — Result type, problem types, conversion
# =============================================================================

# =============================================================================
# Result Types
# =============================================================================

"""
    ConicFitResult{T}

Type alias for `Attributed{SVector{6,T}, RobustAttributes{T}}`.

Conic fitting results use `RobustAttributes` from the generic framework.
Access via property forwarding: `r.value` (SVector{6}), `r.residuals`,
`r.weights`, `r.scale`, `r.iterations`, `r.stop_reason`, `r.converged`.
"""
const ConicFitResult{T} = Attributed{SVector{6,T}, RobustAttributes{T}}

function Base.show(io::IO, r::ConicFitResult{T}) where {T}
    n_inliers = count(>(0.5), r.weights)
    n_total = length(r.weights)
    status = r.converged ? "converged" : string(r.stop_reason)
    print(io, "ConicFitResult{$T}($n_inliers/$n_total inliers, $(r.iterations) iter, $status)")
end

# =============================================================================
# Conic ↔ Ellipse Conversion
# =============================================================================

"""
    conic_to_ellipse(theta::AbstractVector) -> Ellipse

Convert a 6-vector conic representation to an Ellipse.

The conic equation is: theta[1]*x^2 + theta[2]*xy + theta[3]*y^2 + theta[4]*x + theta[5]*y + theta[6] = 0
"""
function conic_to_ellipse(theta::AbstractVector{T}) where {T<:Real}
    A, B, C, D, E, F = theta
    Q = HomEllipseMat{T}((A, B/2, D/2, B/2, C, E/2, D/2, E/2, F))
    Ellipse(Q)
end

# =============================================================================
# Conic Problem Types (AbstractRobustProblem implementations)
# =============================================================================

"""
    _taubin_seed(xis) -> SVector{6}

Unweighted Taubin solve: build M = Σ ξᵢξᵢᵀ, N = Σ JᵢJᵢᵀ, solve smallest GEP(M, N).
Shared by `ConicTaubinProblem.initial_solve` and `ConicFNSProblem.initial_solve`.
"""
function _taubin_seed(xis::Vector{SVector{6,Float64}})
    M = zeros(SMatrix{6,6,Float64,36})
    N = zeros(SMatrix{6,6,Float64,36})
    @inbounds for i in 1:length(xis)
        M += xis[i] * xis[i]'
        J = conic_carrier_jacobian(xis[i][4], xis[i][5])
        N += J * J'
    end
    _solve_smallest_gep(M, N)
end

"""
    ConicTaubinProblem <: AbstractTaubinProblem

Robust Taubin problem: IRLS-weighted generalized eigenvalue problem
using Taubin's gradient-weighted scatter matrices.

Taubin's eigenproblem `M*θ = λ*N*θ` always produces a structurally
valid conic, making it robust as an initialization strategy.
"""
struct ConicTaubinProblem <: AbstractTaubinProblem
    xis::Vector{SVector{6,Float64}}
    Lambdas::Vector{SMatrix{6,6,Float64,36}}
    Js::Vector{SMatrix{6,2,Float64,12}}
end

function ConicTaubinProblem(pts::AbstractVector{<:SVector{2}}, sigma::Real)
    xis = _build_carriers(pts)
    Lambdas = _build_covariances(pts, sigma^2)
    Js = [conic_carrier_jacobian(p) for p in pts]
    ConicTaubinProblem(xis, Lambdas, Js)
end

# --- Dispatch points for AbstractTaubinProblem ---
_seed(prob::ConicTaubinProblem) = _taubin_seed(prob.xis)
_sampson_fn(::ConicTaubinProblem) = sampson_distance
problem_dof(::ConicTaubinProblem) = _CONIC_DOF

"""
    ConicFNSProblem <: AbstractFNSProblem

Robust FNS problem: IRLS-weighted FNS with bias correction.

The FNS bias correction `v_i = 1/(θᵀΛ_iθ)` gives an asymptotically
optimal estimate by accounting for carrier covariance.
"""
struct ConicFNSProblem <: AbstractFNSProblem
    xis::Vector{SVector{6,Float64}}
    Lambdas::Vector{SMatrix{6,6,Float64,36}}
end

function ConicFNSProblem(pts::AbstractVector{<:SVector{2}}, sigma::Real)
    ConicFNSProblem(_build_carriers(pts), _build_covariances(pts, sigma^2))
end

# --- Dispatch points for AbstractFNSProblem ---
_seed(prob::ConicFNSProblem) = _taubin_seed(prob.xis)
_sampson_fn(::ConicFNSProblem) = sampson_distance
problem_dof(::ConicFNSProblem) = _CONIC_DOF
