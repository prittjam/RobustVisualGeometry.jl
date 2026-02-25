# =============================================================================
# RANSAC with Predictive Marginal Scoring (Algorithm 2)
# =============================================================================
#
# Implements the RANSAC pipeline from "RANSAC Done Right" (ECCV 2026):
# - Algorithm 1: Sort-and-sweep (Section 5.1) — finds optimal partition I*
# - Algorithm 2: RANSAC loop (Appendix A) — two-level gating with S_gl, S_lo
# - Algorithm 3/4: Local optimization refit strategies (Section 5.2)
#
# The sort-and-sweep (Algorithm 1) evaluates the scale-free marginal score
# S(θ,I) (Eq. 12) or S_pred(θ̂,I) (Eq. 20) by sorting per-point scores qᵢ
# and sweeping prefixes in O(N log N) time.
#
# =============================================================================

# =============================================================================
# Unified RANSAC Main Loop (quality-agnostic via _quality_improved trait)
# =============================================================================

"""
    ransac(problem, scoring; config, workspace) -> Attributed{M, RansacAttributes}

RANSAC with predictive marginal scoring (Algorithm 2, Appendix A).

For each minimal sample, fits θₜ, computes per-point scores qᵢ and penalties
ℓᵢ, then calls the sort-and-sweep (Algorithm 1) to find the optimal partition.
Two independent best-so-far scores are maintained: S_gl gates entry into local
optimization; S_lo tracks the best locally optimized score. The adaptive trial
count uses the hypergeometric distribution (Appendix A, stopping criterion).

# Arguments
- `problem::AbstractRansacProblem`: Problem definition (data, solver, residuals)
- `scoring::AbstractQualityFunction`: `MarginalQuality` (Eq. 12) or
  `PredictiveMarginalQuality` (Eq. 20)

# Keyword Arguments
- `config::RansacConfig=RansacConfig()`: max_trials, confidence η, min_trials
- `workspace::Union{Nothing,RansacWorkspace}=nothing`: Pre-allocated workspace

# Examples
```julia
# Model-certain score (Section 3, Eq. 12)
result = ransac(problem, MarginalQuality(N, m, a))

# Predictive score with leverages (Section 4, Eq. 20)
result = ransac(problem, PredictiveMarginalQuality(N, m, a))
```
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
# Solve dispatch: wrap SingleSolution output in FixedModels{1,M} so the
# scoring loop always iterates over a FixedModels container.
# =============================================================================

_solve_models(problem, idx, ::MultipleSolutions) = solve(problem, idx)

function _solve_models(problem, idx, ::SingleSolution)
    model = solve(problem, idx)
    isnothing(model) && return nothing
    M = typeof(model)
    FixedModels{1, M}(1, (model,))
end

# =============================================================================
# _score_candidates! — single method for all solver cardinalities
# =============================================================================

function _score_candidates!(ws, problem, scoring::AbstractMarginalQuality,
                            local_optimization, best::Tuple{T,T},
                            cardinality::SolverCardinality) where T
    solutions = _solve_models(problem, ws.sample_indices, cardinality)
    isnothing(solutions) && return best
    best_g, best_l = best
    for model in solutions
        test_model(problem, model, ws.sample_indices) || continue
        best_g, best_l = _try_model!(ws, problem, scoring, local_optimization,
                                      model, best_g, best_l)
    end
    return (best_g, best_l)
end

# =============================================================================
# Two-level gating (Algorithm 2, lines 7-9)
# =============================================================================

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

"""
    _try_model!(ws, problem, scoring, local_optimization, model, best_g, best_l)

Algorithm 2, lines 6-14: two-level gating with S_gl and S_lo.

Phase 1 (lines 6-8): compute per-point scores qᵢ and penalties ℓᵢ via
trait-dispatched `score!`, call `sweep!` (Algorithm 1) to find
optimal partition I*, gate on S* > S_gl.

Phase 2 (line 10): local optimization refit (Algorithms 3/4).

Phase 3 (lines 11-13): re-score refined model (always model-certain:
qᵢ = rᵢ², ℓᵢ = log|Cᵢ|), gate on S_lo.

The three covariance cases (Section 9) are handled by `score!`:
- Homoscedastic:   qᵢ = rᵢ²,       ℓᵢ = 0
- Heteroscedastic: qᵢ = rᵢᵀCᵢ⁻¹rᵢ, ℓᵢ = log|Cᵢ|
- Predictive:      qᵢ = rᵢᵀVᵢ⁻¹rᵢ, ℓᵢ = log|Cᵢ| + log|V_wᵢ|
"""
function _try_model!(ws::RansacWorkspace{M,T}, problem,
                     scoring::AbstractMarginalQuality, local_optimization,
                     model::M, best_g::T, best_l::T) where {M,T}
    n = data_size(problem)
    cov = covariance_structure(problem, scoring)

    # Phase 1: per-point scores + penalties (trait-dispatched)
    score!(ws, problem, scoring, model, cov)

    score_g, k_star = sweep!(scoring, ws.scores, ws.penalties, n)
    (k_star <= scoring.model_dof || score_g <= best_g) && return (best_g, best_l)

    mask!(ws, scoring.perm, k_star)
    test_consensus(problem, model, ws.mask) || return (best_g, best_l)

    # Phase 2: Local optimization
    lo = _locally_optimize!(ws, problem, model, local_optimization, scoring)

    # Phase 3: Re-score refined model (always model-certain)
    score_l, _ = _rescore_model_certain!(ws, problem, scoring, lo.model)
    score_l <= best_l && return (best_g, best_l)

    _update_best!(ws, lo.model)
    return (score_g, score_l)
end

# =============================================================================
# _rescore_model_certain! — Re-score with model-certain (Σ_θ = 0) scoring
# =============================================================================
#
# Phase 3 (re-scoring after LO) is always "model certain" (Σ_θ = 0).
# Dispatches on measurement_covariance(problem) to compute qᵢ and ℓᵢ:
#   Homoscedastic   → residuals! + rᵢ², ℓᵢ = 0
#   Heteroscedastic → residual_jacobian + _sq_norm(rᵢ), ℓᵢ = log|Σ̃_{gᵢ}|
#
# Mirrors score!(... ::Homoscedastic) and score!(... ::Heteroscedastic)
# from scoring.jl, but without the Predictive path.
# =============================================================================

"""
    _rescore_model_certain!(ws, problem, scoring, model) -> (score, k)

Re-score a model with model-certain (Σ_θ = 0) scoring (Section 3.3).
Dispatches on `measurement_covariance(problem)` for both qᵢ and ℓᵢ.

Used by `_try_model!` (Phase 3), `_locally_optimize!`, and `_finalize`.
"""
_rescore_model_certain!(ws, problem, scoring, model) =
    _rescore_model_certain!(ws, problem, scoring, model, measurement_covariance(problem))

# Homoscedastic: qᵢ = rᵢ², ℓᵢ = 0 (fast path)
function _rescore_model_certain!(ws, problem, scoring, model, ::Homoscedastic)
    residuals!(ws.residuals, problem, model)
    n = length(ws.residuals)
    @inbounds for i in 1:n
        ws.scores[i] = ws.residuals[i]^2
    end
    fill!(ws.penalties, zero(eltype(ws.penalties)))
    score, k = sweep!(scoring, ws.scores, ws.penalties, n)
    mask!(ws, scoring.perm, k)
    (score, k)
end

# Heteroscedastic: qᵢ = rᵢᵀrᵢ via residual_jacobian, ℓᵢ = log|Σ̃_{gᵢ}|
function _rescore_model_certain!(ws, problem, scoring, model, ::Heteroscedastic)
    n = data_size(problem)
    @inbounds for i in 1:n
        rᵢ, _, ℓᵢ = residual_jacobian(problem, model, i)
        ws.scores[i] = _sq_norm(rᵢ)
        ws.residuals[i] = sqrt(ws.scores[i])
        ws.penalties[i] = ℓᵢ
    end
    score, k = sweep!(scoring, ws.scores, ws.penalties, n)
    mask!(ws, scoring.perm, k)
    (score, k)
end

# =============================================================================
# _locally_optimize! — Local Refinement Dispatch (refine only, no scoring)
# =============================================================================

"""
    _locally_optimize!(ws, problem, model, ::NoLocalOptimization, scoring) -> (; model, param_cov)

No-op: returns model unchanged.
"""
function _locally_optimize!(ws::RansacWorkspace{M,T}, problem, model::M,
                     ::NoLocalOptimization, scoring) where {M,T}
    return (; model, param_cov=nothing)
end

"""
    _locally_optimize!(ws, problem, model, lo::ConvergeThenRescore, scoring)

Strategy A (Algorithm 3): WLS to convergence at fixed mask, then re-sweep.
Repeat until score does not improve.
"""
function _locally_optimize!(ws::RansacWorkspace{M,T}, problem, model::M,
                     lo::ConvergeThenRescore, scoring) where {M,T}
    n = data_size(problem)
    best_model = model
    best_score = typemin(T)
    strat = fit_strategy(lo)

    w = ws.penalties  # reuse penalties buffer as weight scratch
    @inbounds for i in 1:n
        w[i] = ws.mask[i] ? one(T) : zero(T)
    end

    θ = model
    for outer in 1:lo.max_outer_iter
        # Step 1: Refit — WLS to convergence at fixed mask
        for _ in 1:lo.max_fit_iter
            θ_new = fit(problem, ws.mask, w, strat)
            isnothing(θ_new) && break
            θ_new == θ && break
            θ = θ_new
        end

        # Step 2: Re-score and re-sweep
        score_new, k = _rescore_model_certain!(ws, problem, scoring, θ)

        score_new <= best_score && break
        best_score = score_new
        best_model = θ
        mask!(ws, scoring.perm, k)

        @inbounds for i in 1:n
            w[i] = ws.mask[i] ? one(T) : zero(T)
        end
    end

    return (; model=best_model, param_cov=nothing)
end

"""
    _locally_optimize!(ws, problem, model, lo::StepAndRescore, scoring)

Strategy B (Algorithm 4): Single WLS step, then re-sweep. Repeat until
score does not improve.
"""
function _locally_optimize!(ws::RansacWorkspace{M,T}, problem, model::M,
                     lo::StepAndRescore, scoring) where {M,T}
    n = data_size(problem)
    best_model = model
    best_score = typemin(T)
    strat = fit_strategy(lo)

    w = ws.penalties
    @inbounds for i in 1:n
        w[i] = ws.mask[i] ? one(T) : zero(T)
    end

    θ = model
    for outer in 1:lo.max_outer_iter
        θ_new = fit(problem, ws.mask, w, strat)
        isnothing(θ_new) && break
        θ = θ_new

        score_new, k = _rescore_model_certain!(ws, problem, scoring, θ)

        score_new <= best_score && break
        best_score = score_new
        best_model = θ
        mask!(ws, scoring.perm, k)

        @inbounds for i in 1:n
            w[i] = ws.mask[i] ? one(T) : zero(T)
        end
    end

    return (; model=best_model, param_cov=nothing)
end

# =============================================================================
# _inlier_scale_and_weights — Compute scale, DOF, and binary weights from mask
# =============================================================================

"""
    _inlier_scale_and_weights(residuals, mask, model_dof) -> (scale, dof, weights)

Compute inlier scale (sqrt(RSS/ν)), degrees of freedom, and binary weight
vector from residuals and inlier mask.
"""
function _inlier_scale_and_weights(residuals::AbstractVector{T},
                                    mask::BitVector, model_dof::Int) where T
    n = length(residuals)
    n_in = sum(mask)
    nu = n_in - model_dof
    RSS = zero(T)
    @inbounds for i in 1:n
        mask[i] && (RSS += residuals[i]^2)
    end
    s2 = nu > 0 ? RSS / nu : T(NaN)
    s = s2 > zero(T) ? sqrt(s2) : T(NaN)

    w = Vector{T}(undef, n)
    @inbounds for i in 1:n
        w[i] = mask[i] ? one(T) : zero(T)
    end
    (s, nu, w)
end

# =============================================================================
# _finalize — AbstractMarginalQuality
# =============================================================================

function _finalize(scoring::AbstractMarginalQuality,
                   local_optimization::AbstractLocalOptimization,
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

    lo = _locally_optimize!(ws, problem, model, local_optimization, scoring)

    # Re-score to check if local optimization improved
    score_final, _ = _rescore_model_certain!(ws, problem, scoring, lo.model)

    if score_final >= best_score
        model = lo.model
    else
        # Restore residuals/mask for the original best model
        _rescore_model_certain!(ws, problem, scoring, model)
    end

    s, nu, w = _inlier_scale_and_weights(ws.residuals, ws.mask, p)

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
