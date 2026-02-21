# =============================================================================
# RANSAC — Random Sample Consensus Algorithm
# =============================================================================
#
# Core RANSAC implementation with pluggable quality functions:
# - ThresholdQuality: MSAC quality with fixed inlier threshold
# - MarginalQuality: threshold-free marginal quality
#
# Architecture:
# - Unified quality-agnostic main loop via _quality_improved/init_quality traits
# - Type-stable inner loop via pre-allocated RansacWorkspace
# - Holy Trait dispatch on SolverCardinality (single vs. multiple solutions)
# - Strategy dispatch on AbstractQualityFunction
# - Two-level gating for MarginalQuality: global sweep S_g, local refined S_l
# - Composable local local_optimization via `local_optimization` kwarg: NoLocalOptimization, SimpleRefit, FTestLocalOptimization
# - Quality-based monotonicity check in FTestLocalOptimization via `_lo_quality`
# - Exact Hypergeometric adaptive trial count (no Distributions.jl dependency)
# - Integration with VGC's AbstractLoss and AbstractScaleEstimator
#
# =============================================================================

# =============================================================================
# Unified RANSAC Main Loop (quality-agnostic via _quality_improved trait)
# =============================================================================

"""
    ransac(problem, scoring; local_optimization, config, workspace) -> Attributed{M, RansacAttributes}

Run RANSAC with the given quality function and optional local local_optimization.

The main loop is quality-agnostic: `init_quality(scoring)` provides the initial
best value and `_quality_improved(scoring, new, old)` checks for improvement.
Local local_optimization (LO-RANSAC) is orthogonal to quality and passed separately.

# Arguments
- `problem::AbstractRansacProblem`: Problem definition (data, solver, residuals)
- `scoring::AbstractQualityFunction`: Quality function

# Keyword Arguments
- `local_optimization::AbstractLocalOptimization=default_local_optimization(scoring)`: Local
  local_optimization strategy. Defaults to `NoLocalOptimization()` for `ThresholdQuality`
  and `SimpleRefit()` for `MarginalQuality`. Use `FTestLocalOptimization(...)` for
  iterative F-test inlier reclassification (LO-RANSAC).
- `config::RansacConfig=RansacConfig()`: Algorithm parameters
- `workspace::Union{Nothing,RansacWorkspace}=nothing`: Pre-allocated workspace

# Examples
```julia
# Plain MSAC (no local_optimization)
result = ransac(problem, ThresholdQuality(L2Loss(), 0.05, FixedScale()))

# MSAC + F-test local_optimization (LO-RANSAC)
result = ransac(problem, ThresholdQuality(L2Loss(), threshold, FixedScale());
                local_optimization=FTestLocalOptimization(test=PredictiveFTest(), alpha=0.01))

# Marginal quality (default: SimpleRefit)
result = ransac(problem, MarginalQuality(n, p, 50.0))

# Marginal quality + F-test local_optimization → UncertainRansacEstimate
result = ransac(problem, MarginalQuality(n, p, 50.0);
                local_optimization=FTestLocalOptimization(test=PredictiveFTest()))
```

See also: [`ThresholdQuality`](@ref), [`MarginalQuality`](@ref),
[`default_local_optimization`](@ref), [`FTestLocalOptimization`](@ref)
"""
function ransac(problem::AbstractRansacProblem,
                scoring::AbstractQualityFunction;
                local_optimization::AbstractLocalOptimization = default_local_optimization(scoring),
                config::RansacConfig = RansacConfig(),
                workspace::Union{Nothing, RansacWorkspace} = nothing)

    n = data_size(problem)
    k = sample_size(problem)
    M = model_type(problem)
    T = Float64

    ws = something(workspace, RansacWorkspace(n, k, M, T))

    best = init_quality(scoring)
    needed = config.max_trials
    trial = 0
    effective = 0

    _prepare!(problem)

    while trial < needed && trial < config.max_trials
        trial += 1

        draw_sample!(ws.sample_indices, problem)
        test_sample(problem, ws.sample_indices) || continue
        effective += 1

        old_best = best
        best = _score_candidates!(ws, problem, scoring, local_optimization, best,
                                   solver_cardinality(problem))

        if _quality_improved(scoring, best, old_best) && ws.has_best
            needed = _adaptive_trials(ws.best_mask, k, config)
        end
    end

    sar = trial > 0 ? Float64(effective) / Float64(trial) : 1.0
    return _finalize(scoring, local_optimization, ws, problem, best, sar, trial)
end

# =============================================================================
# Trait-dispatched Quality — TruncatedQuality (scalar best)
# =============================================================================

# MultipleSolutions path — iterates over candidate models
function _score_candidates!(ws, problem, scoring::TruncatedQuality, local_optimization,
                            best, ::MultipleSolutions)
    solutions = solve(problem, ws.sample_indices)
    isnothing(solutions) && return best
    for model in solutions
        test_model(problem, model) || continue
        best = _try_model!(ws, problem, scoring, local_optimization, model, best)
    end
    return best
end

# SingleSolution path — no iteration, no isempty check
function _score_candidates!(ws, problem, scoring::TruncatedQuality, local_optimization,
                            best, ::SingleSolution)
    model = solve(problem, ws.sample_indices)
    isnothing(model) && return best
    test_model(problem, model) || return best
    return _try_model!(ws, problem, scoring, local_optimization, model, best)
end

# =============================================================================
# Trait-dispatched Quality — MarginalQuality (tuple best: (best_g, best_l))
# =============================================================================

function _score_candidates!(ws, problem, scoring::AbstractMarginalQuality,
                            local_optimization, best::Tuple{T,T},
                            ::MultipleSolutions) where T
    solutions = solve(problem, ws.sample_indices)
    isnothing(solutions) && return best
    best_g, best_l = best
    for model in solutions
        test_model(problem, model) || continue
        best_g, best_l = _try_model!(ws, problem, scoring, local_optimization,
                                      model, best_g, best_l)
    end
    return (best_g, best_l)
end

function _score_candidates!(ws, problem, scoring::AbstractMarginalQuality,
                            local_optimization, best::Tuple{T,T},
                            ::SingleSolution) where T
    model = solve(problem, ws.sample_indices)
    isnothing(model) && return best
    test_model(problem, model) || return best
    best_g, best_l = _try_model!(ws, problem, scoring, local_optimization,
                                  model, best[1], best[2])
    return (best_g, best_l)
end

# =============================================================================
# _try_model! — ThresholdQuality (MSAC quality + composable local_optimization)
# =============================================================================
#
# Performance-critical inner loop. Zero-allocation after warmup:
# - All buffers pre-allocated in RansacWorkspace
# - Model stored inline in mutable struct (no Ref boxing)
# - copyto! for buffer updates (no allocation)

function _try_model!(ws::RansacWorkspace{M,T}, problem, scoring::ThresholdQuality,
                     local_optimization, model::M, best_quality::T) where {M,T}
    n = data_size(problem)
    loss = scoring.loss
    threshold = scoring.threshold
    scale_est = scoring.scale_estimator

    # Compute residuals in-place
    residuals!(ws.residuals, problem, model)
    σ = estimate_scale(scale_est, ws.residuals)

    # Compute scores and MSAC-truncated quality in a single pass
    quality = zero(T)
    inv_σ = one(T) / max(σ, eps(T))
    @inbounds for i in 1:n
        ws.scores[i] = max(threshold - rho(loss, ws.residuals[i] * inv_σ), zero(T))
        quality += ws.scores[i]
    end
    quality <= best_quality && return best_quality

    # Compute inlier mask and count
    n_inliers = 0
    @inbounds for i in 1:n
        ws.mask[i] = ws.scores[i] > zero(T)
        n_inliers += ws.mask[i]
    end
    n_inliers <= sample_size(problem) && return best_quality

    # Validate model against consensus set (e.g., oriented epipolar constraint)
    test_consensus(problem, model, ws.mask) || return best_quality

    # Composable local local_optimization
    lo = _lo_refine!(ws, problem, model, local_optimization, scoring)

    # Re-score if model changed
    if lo.model !== model
        σ_r = estimate_scale(scale_est, ws.residuals)
        inv_σ_r = one(T) / max(σ_r, eps(T))
        quality = zero(T)
        @inbounds for i in 1:n
            ws.scores[i] = max(threshold - rho(loss, ws.residuals[i] * inv_σ_r), zero(T))
            quality += ws.scores[i]
        end
        quality <= best_quality && return best_quality
        @inbounds for i in 1:n
            ws.mask[i] = ws.scores[i] > zero(T)
        end
    end

    _update_best!(ws, lo.model)
    return quality
end

# =============================================================================
# _try_model! — ChiSquareQuality (truncated χ² quality)
# =============================================================================
#
# Chi-square is inherently L2: quality = Σ max(χ²(d,1-α) - (r/σ)², 0).
# No loss parameter needed.

"""
    _chi2_cutoff(scoring::ChiSquareQuality, problem) -> Float64

Compute the chi-square cutoff from `α` and `codimension(problem)`.
"""
function _chi2_cutoff(scoring::ChiSquareQuality, problem)
    d = codimension(problem)
    quantile(Chisq(d), 1.0 - scoring.α)
end

function _try_model!(ws::RansacWorkspace{M,T}, problem, scoring::ChiSquareQuality,
                     local_optimization, model::M, best_quality::T) where {M,T}
    n = data_size(problem)
    scale_est = scoring.scale_estimator

    cutoff = _chi2_cutoff(scoring, problem)

    # Compute residuals in-place
    residuals!(ws.residuals, problem, model)
    σ = estimate_scale(scale_est, ws.residuals)

    # Truncated chi-square quality + inlier classification in one pass
    quality = zero(T)
    inv_σ = one(T) / max(σ, eps(T))
    n_inliers = 0
    @inbounds for i in 1:n
        x2 = (ws.residuals[i] * inv_σ)^2
        ws.scores[i] = max(cutoff - x2, zero(T))
        quality += ws.scores[i]
        ws.mask[i] = x2 < cutoff
        n_inliers += ws.mask[i]
    end
    quality <= best_quality && return best_quality
    n_inliers <= sample_size(problem) && return best_quality

    # Validate model against consensus set
    test_consensus(problem, model, ws.mask) || return best_quality

    # Composable local local_optimization
    lo = _lo_refine!(ws, problem, model, local_optimization, scoring)

    # Re-score if model changed
    if lo.model !== model
        σ_r = estimate_scale(scale_est, ws.residuals)
        inv_σ_r = one(T) / max(σ_r, eps(T))
        quality = zero(T)
        n_inliers = 0
        @inbounds for i in 1:n
            x2 = (ws.residuals[i] * inv_σ_r)^2
            ws.scores[i] = max(cutoff - x2, zero(T))
            quality += ws.scores[i]
            ws.mask[i] = x2 < cutoff
            n_inliers += ws.mask[i]
        end
        quality <= best_quality && return best_quality
    end

    _update_best!(ws, lo.model)
    return quality
end

# =============================================================================
# Marginal Sweep (shared by all MarginalQuality variants)
# =============================================================================

"""
    _marginal_sweep!(perm, lg_table, model_dof, log2a, scores, n) -> (best_score, best_k)

Sweep sorted squared residuals to find the optimal inlier count k*
that maximizes the marginal likelihood score:

    S = logΓ(k/2) - (k/2)·log(RSS) - (n-k)·log(2a)

Mutates `perm` via `sortperm!`.
"""
function _marginal_sweep!(perm::Vector{Int}, lg_table::Vector{Float64},
                          model_dof::Int, log2a::Float64,
                          scores::Vector{T}, n::Int) where T
    sortperm!(perm, scores)
    p = model_dof
    RSS = zero(T)
    best_S = typemin(T)
    best_k = 0
    @inbounds for k in 1:n
        RSS += scores[perm[k]]
        k <= p && continue
        RSS <= zero(T) && continue
        S = lg_table[k] - T(0.5) * k * log(RSS) - (n - k) * log2a
        if S > best_S
            best_S = S
            best_k = k
        end
    end
    return best_S, best_k
end

# Dispatch wrapper for all AbstractMarginalQuality subtypes
_marginal_sweep!(s::AbstractMarginalQuality, scores::Vector{T}, n::Int) where T =
    _marginal_sweep!(s.perm, s.lg_table, s.model_dof, s.log2a, scores, n)

# =============================================================================
# _fill_scores! — Dispatch point for score computation (r² vs F-stats)
# =============================================================================

"""
    _fill_scores!(ws, problem, ::AbstractMarginalQuality, model)

Fill `ws.scores` with squared residuals (default for `MarginalQuality`).
Requires `ws.residuals` to be already computed.
"""
function _fill_scores!(ws::RansacWorkspace{M,T}, problem,
                       ::AbstractMarginalQuality, model::M) where {M,T}
    n = data_size(problem)
    @inbounds for i in 1:n
        ws.scores[i] = ws.residuals[i]^2
    end
end

"""
    _fill_scores!(ws, problem, ::PredictiveMarginalQuality, model)

Fill `ws.scores` with prediction-corrected F-statistics using the solver
Jacobian. Falls back to raw r² if `solver_jacobian` returns `nothing`.

Uses `s²=1` because the common s² factor cancels in the argmax of the
marginal score (it shifts S(k) by a term monotone in k).
"""
function _fill_scores!(ws::RansacWorkspace{M,T}, problem,
                       ::PredictiveMarginalQuality, model::M) where {M,T}
    jac_info = solver_jacobian(problem, ws.sample_indices, model)
    if !isnothing(jac_info)
        prediction_fstats_from_cov!(ws.scores, problem, model, jac_info, 1.0)
    else
        n = data_size(problem)
        @inbounds for i in 1:n
            ws.scores[i] = ws.residuals[i]^2
        end
    end
end

# =============================================================================
# Generic Prediction F-statistics (via residual_jacobian dispatch)
# =============================================================================
#
# These generic defaults replace per-problem implementations of
# prediction_fstats_from_cov!, prediction_fstats_from_inliers!, and
# prediction_variances_from_cov!. Each problem only needs to implement
# residual_jacobian(problem, model, i) → (r, G).
#

# Dependencies: LinearAlgebra (det, dot, tr, pinv), Distributions (Chisq, FDist, quantile)

"""
    _info_contrib(G::SVector{p,T}) -> SMatrix{p,p,T}

Outer product G * G' for a gradient vector (d=1 case).
"""
@inline _info_contrib(G::SVector{p,T}) where {p,T} = G * G'

"""
    _info_contrib(G::SMatrix{d,p,T}) -> SMatrix{p,p,T}

Information contribution G' * G for a Jacobian matrix (d≥2 case).
"""
@inline _info_contrib(G::SMatrix{d,p,T}) where {d,p,T} = G' * G

"""
    _prediction_fstat(r::T, G::SVector{p,T}, Σ_θ, s2) -> T

Prediction F-statistic for scalar residual (d=1).
F_i = r² / V_i where V_i = s² + G' Σ_θ G.
"""
@inline function _prediction_fstat(r::T, G::SVector{p,T}, Σ_θ, s2) where {p,T}
    V_i = s2 + dot(G, Σ_θ * G)
    V_i > eps(T) ? r^2 / V_i : typemax(T)
end

"""
    _prediction_fstat(r::SVector{d,T}, G::SMatrix{d,p,T}, Σ_θ, s2) -> T

Prediction F-statistic for vector residual (d≥2).
F_i = r' V_i⁻¹ r / d where V_i = s² I_d + G Σ_θ G'.
StaticArrays optimizes inv/\\ for small matrices at compile time.
"""
@inline function _prediction_fstat(r::SVector{d,T}, G::SMatrix{d,p,T}, Σ_θ, s2) where {d,p,T}
    V_i = s2 * SMatrix{d,d,T}(I) + G * Σ_θ * G'
    det(V_i) > eps(T) || return typemax(T)
    dot(r, V_i \ r) / d
end

"""
    _accumulate_info(problem, model, mask) -> SMatrix or nothing

Accumulate Fisher information I = Σ_{inliers} G_i' G_i from residual Jacobians.
Returns `nothing` if no inliers are found.
"""
function _accumulate_info(problem, model, mask)
    info = nothing
    @inbounds for i in eachindex(mask)
        if mask[i]
            _, G = residual_jacobian(problem, model, i)
            contrib = _info_contrib(G)
            info = isnothing(info) ? contrib : info + contrib
        end
    end
    isnothing(info) && return nothing
    return _augment_info(problem, info, mask)
end

"""
    _augment_info(problem, info, mask) -> SMatrix

Hook for adding problem-specific contributions to the information matrix.
Default: identity (no augmentation).

Override for problems with scale ambiguity resolved by a normalization
constraint (e.g., HomographyProblem with λ₄=1) to make the information
matrix full rank.
"""
_augment_info(problem, info, mask) = info

"""
    _invert_info(::Unconstrained, info, s2) -> SMatrix

Full-rank inverse: Σ_θ = s² · inv(I).
"""
function _invert_info(::Unconstrained, info, s2)
    d = det(info)
    abs(d) < eps(eltype(info)) && return nothing
    s2 * inv(info)
end

"""
    _invert_info(::Constrained, info, s2) -> SMatrix

Bordered-Hessian pseudo-inverse: Σ_θ = s² · pinv(I).

NOTE: This is an approximation. The exact constrained covariance from
the Lagrangian / implicit function theorem is:
  Σ_θ = s² · [I⁻¹ - I⁻¹c(c'I⁻¹c)⁻¹c'I⁻¹]
where c = ∇h(θ) is the constraint gradient. This projection of I⁻¹ onto
the tangent plane of the constraint manifold equals pinv(I) only when c
is in the null space of I (noiseless limit).
"""
_invert_info(::Constrained, info, s2) = s2 * pinv(info)

# --- Generic prediction_fstats_from_cov! default ---

function prediction_fstats_from_cov!(fstats, problem, model, jac_info, s2)
    Σ_θ = s2 * (jac_info.J * jac_info.J')
    n = data_size(problem)
    @inbounds for i in 1:n
        r, G = residual_jacobian(problem, model, i)
        fstats[i] = _prediction_fstat(r, G, Σ_θ, s2)
    end
    return fstats
end

# --- Generic prediction_fstats_from_inliers! default ---

function prediction_fstats_from_inliers!(fstats, problem, model, mask, s2)
    applicable(residual_jacobian, problem, model, 1) || return nothing

    info = _accumulate_info(problem, model, mask)
    isnothing(info) && return nothing

    Σ_θ = _invert_info(constraint_type(problem), info, s2)
    isnothing(Σ_θ) && return nothing

    @inbounds for i in eachindex(fstats)
        r, G = residual_jacobian(problem, model, i)
        fstats[i] = _prediction_fstat(r, G, Σ_θ, s2)
    end
    return (fstats, Σ_θ)
end

# --- Prediction variance contribution: tr(G Σ_θ G') ---

@inline _pred_var_contrib(G::SVector{p,T}, Σ_θ) where {p,T} = dot(G, Σ_θ * G)
@inline _pred_var_contrib(G::SMatrix{d,p,T}, Σ_θ) where {d,p,T} = tr(G * Σ_θ * G')

# --- Generic prediction_variances_from_cov! default ---

function prediction_variances_from_cov!(pred_var, problem, model, jac_info, s2)
    Σ_θ = s2 * (jac_info.J * jac_info.J')
    d = codimension(problem)
    @inbounds for i in eachindex(pred_var)
        _, G = residual_jacobian(problem, model, i)
        pred_var[i] = d * s2 + _pred_var_contrib(G, Σ_θ)
    end
    return pred_var
end

# =============================================================================
# _try_model! — AbstractMarginalQuality (two-level gating)
# =============================================================================

"""
    _sweep_gate!(ws, problem, scoring, model, best_g) -> (score_g, k_star) or nothing

Phase 1: Compute residuals, marginal sweep, check if S_g > best_g.
Returns `(score_g, k_star)` if the model passes the gate, `nothing` to reject.
"""
function _sweep_gate!(ws::RansacWorkspace{M,T}, problem,
                      scoring::AbstractMarginalQuality,
                      model::M, best_g::T) where {M,T}
    residuals!(ws.residuals, problem, model)
    _fill_scores!(ws, problem, scoring, model)

    n = data_size(problem)
    score_g, k_star = _marginal_sweep!(scoring, ws.scores, n)

    (k_star <= scoring.model_dof || score_g <= best_g) && return nothing

    return (score_g, k_star)
end

"""
    _build_mask!(ws, perm, k_star)

Fill `ws.mask` from the k* smallest entries in `perm` (the sortperm order).
"""
function _build_mask!(ws::RansacWorkspace, perm::Vector{Int}, k_star::Int)
    n = length(ws.mask)
    @inbounds for i in 1:n
        ws.mask[i] = false
    end
    @inbounds for j in 1:k_star
        ws.mask[perm[j]] = true
    end
    nothing
end

"""
    _update_best!(ws, model)

Copy current ws buffers (residuals, scores, mask) into best_* fields.
"""
function _update_best!(ws::RansacWorkspace, model)
    ws.best_model = model
    ws.has_best = true
    copyto!(ws.best_residuals, ws.residuals)
    copyto!(ws.best_scores, ws.scores)
    copyto!(ws.best_mask, ws.mask)
    nothing
end

function _try_model!(ws::RansacWorkspace{M,T}, problem,
                     scoring::AbstractMarginalQuality, local_optimization,
                     model::M, best_g::T, best_l::T) where {M,T}

    # Phase 1: Sweep gate
    result = _sweep_gate!(ws, problem, scoring, model, best_g)
    isnothing(result) && return (best_g, best_l)
    score_g, k_star = result

    # Build k* mask
    _build_mask!(ws, scoring.perm, k_star)

    # Consensus check
    test_consensus(problem, model, ws.mask) || return (best_g, best_l)

    # Phase 2: Local local_optimization (refine only, no scoring)
    lo = _lo_refine!(ws, problem, model, local_optimization, scoring)

    # Phase 3: Re-score with marginal sweep
    n = data_size(problem)
    _fill_scores!(ws, problem, scoring, lo.model)
    score_l, k_l = _marginal_sweep!(scoring, ws.scores, n)
    _build_mask!(ws, scoring.perm, k_l)
    score_l <= best_l && return (best_g, best_l)

    # Accept: update best
    _update_best!(ws, lo.model)
    return (score_g, score_l)
end

# =============================================================================
# _lo_quality — Quality function for local search monotonicity check
# =============================================================================
#
# The local search (FTestLocalOptimization) maximizes `_lo_quality`, which is a scalar
# quality consistent with the scoring objective (higher = better):
#
# - ThresholdQuality → MSAC quality (sum of truncated per-point quality)
# - MarginalQuality  → marginal score S (higher = better)
#
# Both dispatch on the F-test type (BasicFTest vs PredictiveFTest) to use
# the same residual standardization as the local_optimization loop.
#
# For non-iterative local_optimization (NoLocalOptimization, SimpleRefit), _lo_quality
# returns Inf (unused; monotonicity is trivial or guaranteed).
#
# =============================================================================

"""
    _inlier_s2(ws, problem) -> Float64

Compute inlier variance s² = RSS / (n_in - p) from the current mask and residuals.
Returns `Inf` if ν ≤ 0 (too few inliers).
"""
function _inlier_s2(ws, problem)
    n = data_size(problem); p = sample_size(problem)
    RSS = zero(Float64)
    n_in = 0
    @inbounds for i in 1:n
        if ws.mask[i]; RSS += ws.residuals[i]^2; n_in += 1; end
    end
    nu = n_in - p
    nu > 0 ? RSS / nu : Inf
end

"""
    _lo_quality(scoring, ws, problem, model, local_optimization) -> Float64

Evaluate the current model state as a scalar quality for monotonicity checking
in the local search. Always maximized (higher = better). Dispatches on both
`scoring` and `local_optimization.test` (the statistical test type).

- `TruncatedQuality` + `BasicFTest`: MSAC quality (sum of truncated scores)
- `TruncatedQuality` + `PredictiveFTest`: sum of truncated prediction scores
- `MarginalQuality` + `BasicFTest`: marginal score on r²
- `MarginalQuality` + `PredictiveFTest`: marginal score on F-stats
- `NoLocalOptimization` / `SimpleRefit`: returns Inf (no-op, never triggers break)
"""
_lo_quality(scoring::ThresholdQuality, ws, problem, model, ref) =
    _lo_quality(scoring, ws, problem, model, ref.test)

function _lo_quality(scoring::ThresholdQuality, ws, problem, model, ::BasicFTest)
    n = data_size(problem)
    σ = estimate_scale(scoring.scale_estimator, ws.residuals)
    inv_σ = 1.0 / max(σ, eps())
    quality = 0.0
    @inbounds for i in 1:n
        quality += max(scoring.threshold - rho(scoring.loss, ws.residuals[i] * inv_σ), 0.0)
    end
    quality
end

function _lo_quality(scoring::ThresholdQuality, ws, problem, model, ::PredictiveFTest)
    n = data_size(problem)
    s2 = _inlier_s2(ws, problem)
    fstats = Vector{Float64}(undef, n)
    result = prediction_fstats_from_inliers!(fstats, problem, model, ws.mask, s2)
    isnothing(result) && return _lo_quality(scoring, ws, problem, model, BasicFTest())
    quality = 0.0
    @inbounds for i in 1:n
        quality += max(scoring.threshold - fstats[i], 0.0)
    end
    quality
end

# --- ChiSquareQuality: truncated χ² quality ---
_lo_quality(scoring::ChiSquareQuality, ws, problem, model, ref) =
    _lo_quality(scoring, ws, problem, model, ref.test)

function _lo_quality(scoring::ChiSquareQuality, ws, problem, model, ::BasicFTest)
    n = data_size(problem)
    cutoff = _chi2_cutoff(scoring, problem)
    σ = estimate_scale(scoring.scale_estimator, ws.residuals)
    inv_σ = 1.0 / max(σ, eps())
    quality = 0.0
    @inbounds for i in 1:n
        quality += max(cutoff - (ws.residuals[i] * inv_σ)^2, 0.0)
    end
    quality
end

function _lo_quality(scoring::ChiSquareQuality, ws, problem, model, ::PredictiveFTest)
    n = data_size(problem)
    cutoff = _chi2_cutoff(scoring, problem)
    s2 = _inlier_s2(ws, problem)
    fstats = Vector{Float64}(undef, n)
    result = prediction_fstats_from_inliers!(fstats, problem, model, ws.mask, s2)
    isnothing(result) && return _lo_quality(scoring, ws, problem, model, BasicFTest())
    quality = 0.0
    @inbounds for i in 1:n
        quality += max(cutoff - fstats[i], 0.0)
    end
    quality
end

# --- MarginalQuality: marginal score (higher = better) ---
_lo_quality(scoring::AbstractMarginalQuality, ws, problem, model, ref) =
    _lo_quality(scoring, ws, problem, model, ref.test)

function _lo_quality(scoring::AbstractMarginalQuality, ws, problem, model, ::BasicFTest)
    n = data_size(problem)
    @inbounds for i in 1:n
        ws.scores[i] = ws.residuals[i]^2
    end
    score, _ = _marginal_sweep!(scoring, ws.scores, n)
    score
end

function _lo_quality(scoring::AbstractMarginalQuality, ws, problem, model, ::PredictiveFTest)
    n = data_size(problem)
    s2 = _inlier_s2(ws, problem)
    fstats = Vector{Float64}(undef, n)
    result = prediction_fstats_from_inliers!(fstats, problem, model, ws.mask, s2)
    if isnothing(result)
        return _lo_quality(scoring, ws, problem, model, BasicFTest())
    end
    @inbounds for i in 1:n
        ws.scores[i] = fstats[i]
    end
    score, _ = _marginal_sweep!(scoring, ws.scores, n)
    score
end

# --- No-ops for non-iterative local_optimization ---
_lo_quality(scoring, ws, problem, model, ::NoLocalOptimization) = Inf
_lo_quality(scoring, ws, problem, model, ::SimpleRefit) = Inf

# =============================================================================
# _lo_refine! — Local Refinement Dispatch (refine only, no scoring)
# =============================================================================

"""
    _lo_refine!(ws, problem, model, ::NoLocalOptimization, scoring) -> (; model, param_cov)

No-op: returns model unchanged.
"""
function _lo_refine!(ws::RansacWorkspace{M,T}, problem, model::M,
                     ::NoLocalOptimization, scoring) where {M,T}
    return (; model, param_cov=nothing)
end

"""
    _lo_refine!(ws, problem, model, ::SimpleRefit, scoring) -> (; model, param_cov)

Simple refit: call `refine(problem, model, mask)` once, update residuals.
Provably monotone (fixed-mask LS), no loss check needed.
"""
function _lo_refine!(ws::RansacWorkspace{M,T}, problem, model::M,
                     ::SimpleRefit, scoring) where {M,T}
    refined = refine(problem, model, ws.mask)
    if !isnothing(refined)
        model_r, _ = refined
        residuals!(ws.residuals, problem, model_r)
        return (; model=model_r, param_cov=nothing)
    end
    return (; model, param_cov=nothing)
end

"""
    _lo_refine!(ws, problem, model, ref::FTestLocalOptimization, scoring) -> (; model, param_cov)

Iterative F-test local_optimization with quality-based monotonicity check.
Reclassifies inliers via F(d,ν) test, refits model on the new inlier set,
and iterates until convergence or quality stops improving.
"""
function _lo_refine!(ws::RansacWorkspace{M,T}, problem, model::M,
                     ref::FTestLocalOptimization, scoring) where {M,T}
    n = data_size(problem)
    p = sample_size(problem)
    d = codimension(problem)
    fstats = Vector{Float64}(undef, n)
    param_cov = nothing
    entry_quality = _lo_quality(scoring, ws, problem, model, ref)

    for _iter in 1:ref.max_iter
        n_in = sum(ws.mask)
        nu = n_in - p
        nu <= 0 && break

        RSS = zero(T)
        @inbounds for i in 1:n
            ws.mask[i] && (RSS += ws.residuals[i]^2)
        end
        s2 = RSS / nu
        s2 <= zero(T) && break

        changed, pc = _classify!(ref.test, ws, problem, model, fstats,
                                  s2, d, nu, ref.alpha)
        !isnothing(pc) && (param_cov = pc)
        !changed && break

        refined = refine(problem, model, ws.mask)
        if !isnothing(refined)
            model, _ = refined
            residuals!(ws.residuals, problem, model)
        end

        # Monotonicity: stop if quality didn't improve
        new_quality = _lo_quality(scoring, ws, problem, model, ref)
        new_quality <= entry_quality && break
        entry_quality = new_quality  # ratchet up
    end

    return (; model, param_cov)
end

# =============================================================================
# _classify! — F-test Classification Dispatch
# =============================================================================

"""
    _classify!(::BasicFTest, ws, problem, model, fstats, s2, d, nu, alpha)
        -> (changed::Bool, param_cov)

Basic F-test: classify point i as inlier when rᵢ²/(d·s²) < F_crit.
Returns `(changed, nothing)`.
"""
function _classify!(::BasicFTest, ws::RansacWorkspace{M,T}, problem, model,
                    fstats::Vector{Float64}, s2::T, d::Int, nu::Int,
                    alpha::Float64) where {M,T}
    n = data_size(problem)
    F_crit = T(quantile(FDist(d, nu), 1.0 - alpha))

    changed = false
    @inbounds for i in 1:n
        new_in = ws.residuals[i]^2 / (d * s2) < F_crit
        if ws.mask[i] != new_in
            changed = true
            ws.mask[i] = new_in
        end
    end
    return changed, nothing
end

"""
    _classify!(::PredictiveFTest, ws, problem, model, fstats, s2, d, nu, alpha)
        -> (changed::Bool, param_cov)

Prediction-corrected F-test using inlier-based OLS covariance.
Falls back to BasicFTest if `prediction_fstats_from_inliers!` is not available.
"""
function _classify!(::PredictiveFTest, ws::RansacWorkspace{M,T}, problem, model,
                    fstats::Vector{Float64}, s2::T, d::Int, nu::Int,
                    alpha::Float64) where {M,T}
    n = data_size(problem)

    result = prediction_fstats_from_inliers!(fstats, problem, model, ws.mask, s2)

    if isnothing(result)
        return _classify!(BasicFTest(), ws, problem, model, fstats,
                           s2, d, nu, alpha)
    end

    _, param_cov = result
    F_crit = T(quantile(FDist(d, nu), 1.0 - alpha))

    changed = false
    @inbounds for i in 1:n
        new_in = fstats[i] < F_crit
        if ws.mask[i] != new_in
            changed = true
            ws.mask[i] = new_in
        end
    end
    return changed, param_cov
end

# =============================================================================
# _finalize — Post-loop finalization (dispatched on quality × local_optimization)
# =============================================================================
#
# Dispatches on (quality, local_optimization) to produce the appropriate result type:
#
# - (ThresholdQuality, NoLocalOptimization|SimpleRefit) → RansacAttributes
# - (ThresholdQuality, FTestLocalOptimization)               → RansacAttributes or
#                                                       UncertainRansacAttributes
# - (MarginalQuality,  NoLocalOptimization|SimpleRefit) → RansacAttributes
# - (MarginalQuality,  FTestLocalOptimization)               → RansacAttributes or
#                                                       UncertainRansacAttributes
#
# FTestLocalOptimization runs a final _lo_refine! on the best model and may produce
# a parameter covariance matrix (via PredictiveFTest), yielding an
# UncertainRansacEstimate.
#
# =============================================================================

function _finalize(scoring::TruncatedQuality,
                   local_optimization::Union{NoLocalOptimization,SimpleRefit},
                   ws, problem, best_quality, sar, trial)
    n = data_size(problem)
    M = model_type(problem)
    T = Float64

    if !ws.has_best
        attrs = RansacAttributes(:no_model;
            inlier_mask = falses(n),
            residuals = zeros(T, n),
            weights = zeros(T, n),
            quality = T(-Inf),
            scale = T(NaN),
            trials = trial,
            sample_acceptance_rate = sar)
        return Attributed(zero(M), attrs)
    end

    # Final weights from loss function
    σ = estimate_scale(scoring.scale_estimator, ws.best_residuals)
    w = Vector{T}(undef, n)
    @inbounds for i in 1:n
        w[i] = weight(scoring.loss, ws.best_residuals[i] / σ)
    end

    attrs = RansacAttributes(:converged;
        inlier_mask = copy(ws.best_mask),
        residuals = copy(ws.best_residuals),
        weights = w,
        quality = best_quality,
        scale = σ,
        trials = trial,
        sample_acceptance_rate = sar)
    return Attributed(ws.best_model, attrs)
end

# =============================================================================
# _finalize — TruncatedQuality with FTestLocalOptimization
# =============================================================================

function _finalize(scoring::TruncatedQuality, local_optimization::FTestLocalOptimization,
                   ws, problem, best_quality, sar, trial)
    n = data_size(problem)
    M = model_type(problem)
    T = Float64

    if !ws.has_best
        attrs = RansacAttributes(:no_model;
            inlier_mask = falses(n),
            residuals = zeros(T, n),
            weights = zeros(T, n),
            quality = T(-Inf),
            scale = T(NaN),
            trials = trial,
            sample_acceptance_rate = sar)
        return Attributed(zero(M), attrs)
    end

    # Restore best into ws, run final _lo_refine!
    copyto!(ws.residuals, ws.best_residuals)
    copyto!(ws.mask, ws.best_mask)
    model = ws.best_model

    lo = _lo_refine!(ws, problem, model, local_optimization, scoring)
    model = lo.model

    # Scale from final mask
    p = sample_size(problem)
    n_in = sum(ws.mask)
    nu = n_in - p
    RSS = zero(T)
    @inbounds for i in 1:n
        if ws.mask[i]
            RSS += ws.residuals[i]^2
        end
    end
    s = nu > 0 ? sqrt(RSS / nu) : T(NaN)

    w = Vector{T}(undef, n)
    @inbounds for i in 1:n
        w[i] = ws.mask[i] ? one(T) : zero(T)
    end

    base_attrs = RansacAttributes(:converged;
        inlier_mask = copy(ws.mask),
        residuals = copy(ws.residuals),
        weights = w,
        quality = best_quality,
        scale = s,
        dof = max(nu, 0),
        trials = trial,
        sample_acceptance_rate = sar)

    if !isnothing(lo.param_cov)
        return Attributed(model, UncertainRansacAttributes(base_attrs, lo.param_cov))
    end
    return Attributed(model, base_attrs)
end

# =============================================================================
# _finalize — AbstractMarginalQuality (no local_optimization / simple refit)
# =============================================================================

function _finalize(scoring::AbstractMarginalQuality,
                   local_optimization::Union{NoLocalOptimization,SimpleRefit},
                   ws, problem, best::Tuple, sar, trial)
    n = data_size(problem)
    M = model_type(problem)
    T = Float64
    p = scoring.model_dof
    best_score = best[2]

    if !ws.has_best
        attrs = RansacAttributes(:no_model;
            inlier_mask = falses(n),
            residuals = zeros(T, n),
            weights = zeros(T, n),
            quality = T(-Inf),
            scale = T(NaN),
            dof = 0,
            trials = trial,
            sample_acceptance_rate = sar)
        return Attributed(zero(M), attrs)
    end

    model = ws.best_model

    # Final LO-refine on best model
    copyto!(ws.residuals, ws.best_residuals)
    copyto!(ws.mask, ws.best_mask)

    lo = _lo_refine!(ws, problem, model, local_optimization, scoring)

    # Re-score to check if local_optimization improved
    @inbounds for i in 1:n
        ws.scores[i] = ws.residuals[i]^2
    end
    score_final, k_final = _marginal_sweep!(scoring, ws.scores, n)
    _build_mask!(ws, scoring.perm, k_final)

    if score_final >= best_score
        model = lo.model
    end

    # Compute scale and dof from final mask
    n_in = sum(ws.mask)
    nu = n_in - p
    RSS = zero(T)
    @inbounds for i in 1:n
        if ws.mask[i]
            RSS += ws.residuals[i]^2
        end
    end
    s2 = nu > 0 ? RSS / nu : T(NaN)
    s = s2 > zero(T) ? sqrt(s2) : T(NaN)

    # Binary weights from final mask
    w = Vector{T}(undef, n)
    @inbounds for i in 1:n
        w[i] = ws.mask[i] ? one(T) : zero(T)
    end

    base_attrs = RansacAttributes(:converged;
        inlier_mask = copy(ws.mask),
        residuals = copy(ws.residuals),
        weights = w,
        quality = best_score,
        scale = s,
        dof = max(nu, 0),
        trials = trial,
        sample_acceptance_rate = sar)

    return Attributed(model, base_attrs)
end

# =============================================================================
# _finalize — AbstractMarginalQuality with FTestLocalOptimization
# =============================================================================

function _finalize(scoring::AbstractMarginalQuality, local_optimization::FTestLocalOptimization,
                   ws, problem, best::Tuple, sar, trial)
    n = data_size(problem)
    M = model_type(problem)
    T = Float64
    p = scoring.model_dof
    best_score = best[2]

    if !ws.has_best
        attrs = RansacAttributes(:no_model;
            inlier_mask = falses(n),
            residuals = zeros(T, n),
            weights = zeros(T, n),
            quality = T(-Inf),
            scale = T(NaN),
            dof = 0,
            trials = trial,
            sample_acceptance_rate = sar)
        return Attributed(zero(M), attrs)
    end

    model = ws.best_model

    # Final LO-refine on best model
    copyto!(ws.residuals, ws.best_residuals)
    copyto!(ws.mask, ws.best_mask)

    lo = _lo_refine!(ws, problem, model, local_optimization, scoring)

    # Re-score to check if local_optimization improved
    @inbounds for i in 1:n
        ws.scores[i] = ws.residuals[i]^2
    end
    score_final, k_final = _marginal_sweep!(scoring, ws.scores, n)
    _build_mask!(ws, scoring.perm, k_final)

    if score_final >= best_score
        model = lo.model
    end

    # Compute scale and dof from final mask
    n_in = sum(ws.mask)
    nu = n_in - p
    RSS = zero(T)
    @inbounds for i in 1:n
        if ws.mask[i]
            RSS += ws.residuals[i]^2
        end
    end
    s2 = nu > 0 ? RSS / nu : T(NaN)
    s = s2 > zero(T) ? sqrt(s2) : T(NaN)

    # Binary weights from final mask
    w = Vector{T}(undef, n)
    @inbounds for i in 1:n
        w[i] = ws.mask[i] ? one(T) : zero(T)
    end

    base_attrs = RansacAttributes(:converged;
        inlier_mask = copy(ws.mask),
        residuals = copy(ws.residuals),
        weights = w,
        quality = best_score,
        scale = s,
        dof = max(nu, 0),
        trials = trial,
        sample_acceptance_rate = sar)

    # Attach param_cov if available (from FTestLocalOptimization with PredictiveFTest)
    param_cov = lo.param_cov
    if !isnothing(param_cov)
        attrs = UncertainRansacAttributes(base_attrs, param_cov)
        return Attributed(model, attrs)
    end

    return Attributed(model, base_attrs)
end

# =============================================================================
# Standalone F-test Iterative Refinement (for external use)
# =============================================================================

"""
    _ftest_refine!(mask, r, problem, model, p, alpha, max_iter)

Iterative F-test local_optimization of the inlier set (standalone version).

Dispatches on `test_type(problem)`:
- `BasicFTest`: classifies using `rᵢ²/s² < F_crit` (uniform variance)
- `PredictiveFTest`: classifies using F-statistics from
  `prediction_fstats_from_inliers!`

Returns `(mask, s², ν, model)` for `BasicFTest`, or
`(mask, s², ν, model, param_cov)` for `PredictiveFTest`.
"""
function _ftest_refine!(mask::BitVector, r::Vector{Float64},
                        problem::AbstractRansacProblem, model,
                        p::Int, alpha::Float64, max_iter::Int)
    _ftest_refine!(test_type(problem), mask, r, problem, model, p, alpha, max_iter)
end

# --- BasicFTest: rᵢ²/s² < F_crit ---

function _ftest_refine!(::BasicFTest, mask::BitVector, r::Vector{Float64},
                        problem::AbstractRansacProblem, model,
                        p::Int, alpha::Float64, max_iter::Int)
    n = data_size(problem)
    d = codimension(problem)

    for _iter in 1:max_iter
        n_in = sum(mask)
        nu = n_in - p
        nu <= 0 && break

        RSS = 0.0
        @inbounds for i in 1:n
            if mask[i]
                RSS += r[i]^2
            end
        end
        s2 = RSS / nu
        s2 <= 0.0 && break

        F_crit = quantile(FDist(d, nu), 1.0 - alpha)

        changed = false
        @inbounds for i in 1:n
            new_in = r[i]^2 / (d * s2) < F_crit
            if mask[i] != new_in
                changed = true
                mask[i] = new_in
            end
        end

        !changed && break

        refined = refine(problem, model, mask)
        if !isnothing(refined)
            model, _ = refined
            residuals!(r, problem, model)
        end
    end

    # Final scale estimate
    n_in = sum(mask)
    nu = n_in - p
    RSS = 0.0
    @inbounds for i in 1:n
        if mask[i]
            RSS += r[i]^2
        end
    end
    s2 = nu > 0 ? RSS / nu : NaN

    return mask, s2, nu, model
end

# --- PredictiveFTest: F-stats from prediction_fstats_from_inliers! ---

function _ftest_refine!(::PredictiveFTest, mask::BitVector, r::Vector{Float64},
                        problem::AbstractRansacProblem, model,
                        p::Int, alpha::Float64, max_iter::Int)
    n = data_size(problem)
    d = codimension(problem)
    fstats = Vector{Float64}(undef, n)
    last_param_cov = nothing

    for _iter in 1:max_iter
        n_in = sum(mask)
        nu = n_in - p
        nu <= 0 && break

        RSS = 0.0
        @inbounds for i in 1:n
            if mask[i]
                RSS += r[i]^2
            end
        end
        s2 = RSS / nu
        s2 <= 0.0 && break

        result = prediction_fstats_from_inliers!(fstats, problem, model, mask, s2)
        if isnothing(result)
            # Fall back to basic F-test for this iteration
            F_crit = quantile(FDist(d, nu), 1.0 - alpha)
            changed = false
            @inbounds for i in 1:n
                new_in = r[i]^2 / (d * s2) < F_crit
                if mask[i] != new_in
                    changed = true
                    mask[i] = new_in
                end
            end
            !changed && break
        else
            _, last_param_cov = result

            F_crit = quantile(FDist(d, nu), 1.0 - alpha)

            changed = false
            @inbounds for i in 1:n
                new_in = fstats[i] < F_crit
                if mask[i] != new_in
                    changed = true
                    mask[i] = new_in
                end
            end
            !changed && break
        end

        refined = refine(problem, model, mask)
        if !isnothing(refined)
            model, _ = refined
            residuals!(r, problem, model)
        end
    end

    # Final scale estimate
    n_in = sum(mask)
    nu = n_in - p
    RSS = 0.0
    @inbounds for i in 1:n
        if mask[i]
            RSS += r[i]^2
        end
    end
    s2 = nu > 0 ? RSS / nu : NaN

    # Final param_cov at converged scale
    if !isnothing(last_param_cov) && s2 > 0.0
        result = prediction_fstats_from_inliers!(fstats, problem, model, mask, s2)
        if !isnothing(result)
            _, last_param_cov = result
        end
    end

    return mask, s2, nu, model, last_param_cov
end

# =============================================================================
# Adaptive Trial Count (Hypergeometric, exact)
# =============================================================================

"""
    _p_all_inliers(n_inliers, n_total, k) -> Float64

Probability that all `k` draws are inliers when sampling without replacement
from `n_total` items with `n_inliers` successes.
"""
function _p_all_inliers(n_inliers::Int, n_total::Int, k::Int)
    n_inliers < k && return 0.0
    p = 1.0
    @inbounds for i in 0:(k-1)
        p *= (n_inliers - i) / (n_total - i)
    end
    return p
end

function _adaptive_trials(mask::BitVector, k::Int, config::RansacConfig)
    p_success = _p_all_inliers(sum(mask), length(mask), k)
    p_success ≈ 0 && return config.max_trials
    p_failure = 1 - p_success
    p_failure ≈ 0 && return config.min_trials
    trials = ceil(Int, log(1 - config.confidence) / log(p_failure))
    return clamp(trials, config.min_trials, config.max_trials)
end
