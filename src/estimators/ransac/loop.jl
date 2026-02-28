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
# Unified RANSAC Main Loop (scoring-agnostic via _score_improved trait)
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
- `scoring::AbstractScoring`: `MarginalScoring` (Eq. 12) or
  `PredictiveMarginalScoring` (Eq. 20)

# Keyword Arguments
- `config::RansacConfig=RansacConfig()`: max_trials, confidence η, min_trials
- `workspace::Union{Nothing,RansacWorkspace}=nothing`: Pre-allocated workspace

# Examples
```julia
# Model-certain score (Section 3, Eq. 12)
result = ransac(problem, MarginalScoring(N, m, a))

# Predictive score with leverages (Section 4, Eq. 20)
result = ransac(problem, PredictiveMarginalScoring(N, m, a))
```
"""
function ransac(problem::AbstractRansacProblem,
                scoring::AbstractScoring;
                local_optimization::AbstractLocalOptimization = default_local_optimization(scoring),
                config::RansacConfig = RansacConfig(),
                workspace::Union{Nothing, RansacWorkspace} = nothing)

    n = data_size(problem)
    k = sample_size(problem)
    M = model_type(problem)
    T = Float64

    ws = something(workspace, RansacWorkspace(n, k, M, T))

    best = init_score(scoring)
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

        if _score_improved(scoring, best, old_best) && ws.has_best
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

function _score_candidates!(ws, problem, scoring::MarginalScoring,
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

Phase 1 (lines 6-8): wrap model with covariance (Uncertain{M} for PredMQ),
compute per-point scores qᵢ via type-dispatched `score!`, call `sweep!`
(Algorithm 1) to find optimal partition I*, gate on S* > S_gl.

Phase 2 (line 10): local optimization refit (Algorithms 3/4).

Phase 3 (lines 11-13): wrap refined model with fit covariance, re-score,
gate on S_lo. Model covariance dispatch:
- Plain model M → model-certain scoring
- Uncertain{M} → model-uncertain scoring (reads model.param_cov)
"""
function _try_model!(ws::RansacWorkspace{<:Any,M,T}, problem,
                     scoring::MarginalScoring, local_optimization,
                     model::M, best_g::T, best_l::T) where {M,T}
    # Phase 1: wrap with model covariance, score + sweep + mask
    model_scored = _with_model_cov(scoring, problem, model, ws.sample_indices)
    score_g, k_star = _score_and_sweep!(ws, problem, scoring, model_scored)
    (k_star <= scoring.model_dof || score_g <= best_g) && return (best_g, best_l)

    test_consensus(problem, model, ws.mask) || return (best_g, best_l)

    # Phase 2: Local optimization (returns plain M)
    lo_model = _locally_optimize!(ws, problem, model, local_optimization, scoring)

    # Phase 3: wrap with fit covariance, re-score refined model
    lo_scored = _with_fit_cov(scoring, problem, lo_model)
    score_l, _ = _score_and_sweep!(ws, problem, scoring, lo_scored)
    score_l <= best_l && return (best_g, best_l)

    _update_best!(ws, lo_model)
    return (score_g, score_l)
end

# =============================================================================
# _score_and_sweep! — Single scoring path for all phases
# =============================================================================
#
# Every score computation in the RANSAC loop goes through this function:
# Phase 1 (initial), Phase 2 (LO iterations), Phase 3 (final re-score),
# and _finalize. Using a single path preserves monotonicity — scores
# from different phases are always comparable.
# =============================================================================

"""
    _score_and_sweep!(ws, problem, scoring, model) -> (score, k)

Score a model: compute per-point scores via `score!` (type-dispatched
on plain M vs Uncertain{M}), find the optimal partition via `sweep!`,
and set the inlier mask via `mask!`.  Returns the score and inlier count.
"""
function _score_and_sweep!(ws, problem, scoring, model)
    n = data_size(problem)
    score!(ws, problem, scoring, model)
    s, k = sweep!(scoring, ws.scores, ws.penalties, n)
    mask!(ws, scoring.perm, k)
    (s, k)
end

# =============================================================================
# _locally_optimize! — Local Refinement Dispatch (refine only, no scoring)
# =============================================================================

"""
    _locally_optimize!(ws, problem, model, ::NoLocalOptimization, scoring) -> M

No-op: returns model unchanged.
"""
function _locally_optimize!(ws::RansacWorkspace{<:Any,M,T}, problem, model::M,
                     ::NoLocalOptimization, scoring) where {M,T}
    return model
end

# =============================================================================
# PosteriorIrls — Posterior-weight IRLS via generic irls! loop
# =============================================================================

struct PosteriorIrlsMethod{S<:MarginalScoring,K,M,T} <: AbstractIRLSMethod
    scoring::S
    ws::RansacWorkspace{K,M,T}
    strat::LinearFit
end

mutable struct PosteriorIrlsState{M,T}
    w::Vector{T}
    best_model::M
    best_score::T
    current_score::T
end

function init_state(method::PosteriorIrlsMethod{S,K,M,T}, problem, θ₀) where {S,K,M,T}
    ws = method.ws
    scoring = method.scoring
    n = data_size(problem)

    # Score + sweep the initial model
    θ_scored = _with_fit_cov(scoring, problem, θ₀)
    score, k = _score_and_sweep!(ws, problem, scoring, θ_scored)

    # Compute posterior weights from the sweep partition
    w = Vector{T}(undef, n)
    _posterior_weights!(w, ws.scores, scoring.perm, k, scoring.codimension, scoring.log2a)

    PosteriorIrlsState{M,T}(w, θ₀, score, score)
end

function solve_step(state::PosteriorIrlsState, method::PosteriorIrlsMethod, problem, θ)
    ws = method.ws
    # Build mask from weights (w > 0.5 → inlier)
    n = length(state.w)
    @inbounds for i in 1:n
        ws.mask[i] = state.w[i] > 0.5
    end
    fit(problem, ws.mask, state.w, method.strat)
end

function update_residuals!(state::PosteriorIrlsState, method::PosteriorIrlsMethod, problem, θ)
    ws = method.ws
    scoring = method.scoring
    θ_scored = _with_fit_cov(scoring, problem, θ)
    score, k = _score_and_sweep!(ws, problem, scoring, θ_scored)
    state.current_score = score
    nothing
end

function update_scale!(state::PosteriorIrlsState, method::PosteriorIrlsMethod, prob)
    nothing  # scale is implicit in sweep
end

function update_weights!(state::PosteriorIrlsState{M,T}, method::PosteriorIrlsMethod, prob) where {M,T}
    ws = method.ws
    scoring = method.scoring
    k = sum(ws.mask)
    _posterior_weights!(state.w, ws.scores, scoring.perm, k, scoring.codimension, scoring.log2a)
end

function post_step!(state::PosteriorIrlsState, method::PosteriorIrlsMethod, prob, θ, iter)
    if state.current_score > state.best_score
        state.best_score = state.current_score
        state.best_model = θ
    end
end

function is_converged(state::PosteriorIrlsState, method::PosteriorIrlsMethod, prob, θ, θ_old, iter)
    state.current_score <= state.best_score && iter > 1
end

function irls_result(state::PosteriorIrlsState{M,T}, method::PosteriorIrlsMethod, prob, θ, converged, final_iter) where {M,T}
    state.best_model
end

"""
    _locally_optimize!(ws, problem, model, lo::PosteriorIrls, scoring) -> M

Strategy C: Posterior-weight IRLS refinement. Computes posterior inlier
probabilities, uses them as soft weights for WLS, re-scores, and iterates
via the generic `irls!` loop.
"""
function _locally_optimize!(ws::RansacWorkspace{K,M,T}, problem, model::M,
                     lo::PosteriorIrls, scoring::MarginalScoring) where {K,M,T}
    method = PosteriorIrlsMethod(scoring, ws, LinearFit())
    irls!(method, problem, model; max_iter=lo.max_outer_iter)
end

# =============================================================================
# _inlier_scale_and_weights — Compute scale, DOF, and binary weights from mask
# =============================================================================

"""
    _inlier_scale_and_weights(residuals, mask, model_dof, d_g) -> (scale, dof, weights)

Compute inlier scale (sqrt(RSS/ν)), degrees of freedom, and binary weight
vector from residuals and inlier mask.

Each inlier contributes `d_g` scalar constraint equations, so RSS is a sum of
`n_in * d_g` squared terms. The degrees of freedom are `ν = n_in * d_g - n_θ`
where `n_θ = model_dof * d_g`.

Note: this uses the *unbiased* estimator RSS/ν (dividing by n-p) for the
user-facing scale output, unlike `_posterior_weights!` which uses the MLE
RSS/(n·d_g) for internal posterior computation (the mode of the inverse-gamma
posterior under the Jeffreys prior).
"""
function _inlier_scale_and_weights(residuals::AbstractVector{T},
                                    mask::BitVector, model_dof::Int,
                                    d_g::Int) where T
    n = length(residuals)
    n_in = sum(mask)
    nu = d_g * (n_in - model_dof)
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
# _finalize — MarginalScoring
# =============================================================================

function _finalize(scoring::MarginalScoring,
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

    lo_model = _locally_optimize!(ws, problem, model, local_optimization, scoring)

    # Re-score to check if local optimization improved
    lo_scored = _with_fit_cov(scoring, problem, lo_model)
    score_final, _ = _score_and_sweep!(ws, problem, scoring, lo_scored)

    if score_final >= best_score
        model = lo_model
    else
        # Restore residuals/mask for the original best model
        _score_and_sweep!(ws, problem, scoring, model)
    end

    d_g = scoring.codimension
    s, nu, w = _inlier_scale_and_weights(ws.residuals, ws.mask, p, d_g)

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
