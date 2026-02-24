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
# Sort-and-Sweep (Algorithm 1, Section 5.1)
# =============================================================================

"""
    sweep!(perm, lg_table, model_dof, log2a, dg, scores, penalties, n) -> (S*, k*)

Sort-and-sweep computation of the scale-free marginal score (Algorithm 1, Eq. 12).

Sorts per-point weighted squared residuals qᵢ = gᵢᵀ Σ̃_{gᵢ}⁻¹ gᵢ, then sweeps
prefixes k = m+1,...,N evaluating (Eq. 12):

    S = log Γ(k·dg/2) − (k·dg/2) log(π·RSS_I) − ½L − (N−k)·dg·log(2a)

where RSS_I = Σ_{j=1}^k q_{(j)} (weighted residual sum of squares),
L = Σ_{j=1}^k ℓ_{(j)} (accumulated covariance penalty, ℓᵢ = log|Σ̃_{gᵢ}|),
and lg_table[k] = log Γ(k·dg/2) (precomputed inlier reward).

Returns the best score S* and optimal inlier count k*. O(N log N) time.

The four terms of S (Section 3.2):
  1. Inlier reward:      log Γ(k·dg/2)
  2. Fit quality:        −(k·dg/2) log(π·RSS_I)
  3. Covariance penalty: −½L
  4. Outlier penalty:    −(N−k)·dg·log(2a)

Mutates `perm` via `sortperm!`.
"""
function sweep!(perm::Vector{Int}, lg_table::Vector{Float64},
                          model_dof::Int, log2a::Float64, d_g::Int,
                          scores::Vector{T}, penalties::Vector{T},
                          n::Int) where T
    sortperm!(perm, scores)
    p = model_dof
    d2 = T(d_g) / 2
    d_g_T = T(d_g)
    log_pi = T(log(π))
    RSS = zero(T)
    L = zero(T)
    best_S = typemin(T)
    best_k = 0
    @inbounds for k in 1:n
        idx = perm[k]
        RSS += scores[idx]
        L += penalties[idx]
        k <= p && continue
        RSS <= zero(T) && continue
        S = lg_table[k] - d2 * k * (log_pi + log(RSS)) - T(0.5) * L - (n - k) * d_g_T * log2a
        if S > best_S
            best_S = S
            best_k = k
        end
    end
    return best_S, best_k
end

# Dispatch wrapper for all AbstractMarginalQuality subtypes
sweep!(s::AbstractMarginalQuality, scores::Vector{T}, penalties::Vector{T}, n::Int) where T =
    sweep!(s.perm, s.lg_table, s.model_dof, s.log2a, s.codimension, scores, penalties, n)

# =============================================================================
# model_covariance — Dispatch point for prediction variance (Section 4)
# =============================================================================

"""
    model_covariance(scoring, problem, model, sample_indices) -> Σ̃_θ or nothing

Compute the estimation covariance shape Σ̃_θ (Eq. 27, Appendix D).

- `MarginalQuality`: returns `nothing` (model certain: Σ_θ = 0)
- `PredictiveMarginalQuality`: returns `Σ̃_θ = J Jᵀ` where J is the solver
  Jacobian ∂θ/∂x_min. The product J·Jᵀ gives the estimation covariance shape
  (Eq. 27): Σ_θ = σ² Σ̃_θ.

Returns `nothing` if solver_jacobian is unavailable.
"""
model_covariance(::MarginalQuality, problem, model, sample_indices) = nothing

function model_covariance(::PredictiveMarginalQuality, problem, model, sample_indices)
    jac_info = solver_jacobian(problem, sample_indices, model)
    isnothing(jac_info) && return nothing
    return jac_info.J * jac_info.J'
end

# =============================================================================
# _predictive_score_penalty — Per-point (q_i, log_v_i) with model covariance
# =============================================================================

"""
    _predictive_score_penalty(rᵢ::T, ∂θgᵢ_w::SVector{n_θ,T}, Σ̃_θ, s2) -> (qᵢ, ℓᵢ_model)

Model-uncertain score and penalty for scalar constraint (dg=1, Appendix D).

In whitened coordinates (rᵢ = gᵢ/√cᵢ, ∂θgᵢ_w = ∂θgᵢ/√cᵢ):
  Σ̃_w = s² + (∂θgᵢ_w)ᵀ Σ̃_θ (∂θgᵢ_w)    (scalar prediction variance shape)
  qᵢ = rᵢ² / Σ̃_w                          (weighted squared residual)
  ℓᵢ_model = log(Σ̃_w)                      (model uncertainty contribution)
"""
@inline function _predictive_score_penalty(rᵢ::T, ∂θgᵢ_w::SVector{n_θ,T},
                                            Σ̃_θ, s2) where {n_θ,T}
    Σ̃_w = s2 + dot(∂θgᵢ_w, Σ̃_θ * ∂θgᵢ_w)
    Σ̃_w > eps(T) || return (typemax(T), zero(T))
    return (rᵢ^2 / Σ̃_w, log(Σ̃_w))
end

"""
    _predictive_score_penalty(rᵢ::SVector{dg,T}, ∂θgᵢ_w::SMatrix{dg,n_θ,T}, Σ̃_θ, s2)
        -> (qᵢ, ℓᵢ_model)

Model-uncertain Mahalanobis score and penalty for vector constraint
(dg≥2, Eq. 7, Appendix D).

In whitened coordinates (rᵢ = L⁻¹gᵢ, ∂θgᵢ_w = L⁻¹∂θgᵢ):
  Σ̃_w = s²I + (∂θgᵢ_w) Σ̃_θ (∂θgᵢ_w)ᵀ    (dg×dg prediction covariance shape)
  qᵢ = rᵢᵀ Σ̃_w⁻¹ rᵢ                       (weighted squared residual)
  ℓᵢ_model = log|Σ̃_w|                       (model uncertainty contribution)
"""
@inline function _predictive_score_penalty(rᵢ::SVector{dg,T}, ∂θgᵢ_w::SMatrix{dg,n_θ,T},
                                            Σ̃_θ, s2) where {dg,n_θ,T}
    Σ̃_w = s2 * SMatrix{dg,dg,T}(I) + ∂θgᵢ_w * Σ̃_θ * ∂θgᵢ_w'
    det_Σ̃_w = det(Σ̃_w)
    det_Σ̃_w > eps(T) || return (typemax(T), zero(T))
    qᵢ = dot(rᵢ, Σ̃_w \ rᵢ)
    return (qᵢ, log(det_Σ̃_w))
end

# =============================================================================
# Two-level gating (Algorithm 2, lines 7-9)
# =============================================================================

"""
    mask!(ws, perm, k_star)

Build the inlier set I* = {(1),...,(k*)} from the sorted order (Section 3.5).
"""
function mask!(ws::RansacWorkspace, perm::Vector{Int}, k_star::Int)
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

"""
    _try_model!(ws, problem, scoring, local_optimization, model, best_g, best_l)

Algorithm 2, lines 6-14: two-level gating with S_gl and S_lo.

Phase 1 (lines 6-8): compute per-point scores qᵢ and penalties ℓᵢ via
trait-dispatched `_fill_scores!`, call `sweep!` (Algorithm 1) to find
optimal partition I*, gate on S* > S_gl.

Phase 2 (line 10): local optimization refit (Algorithms 3/4).

Phase 3 (lines 11-13): re-score refined model (always model-certain:
qᵢ = rᵢ², ℓᵢ = log|Cᵢ|), gate on S_lo.

The three covariance cases (Section 9) are handled by `_fill_scores!`:
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
    _fill_scores!(ws, problem, scoring, model, cov)

    score_g, k_star = sweep!(scoring, ws.scores, ws.penalties, n)
    (k_star <= scoring.model_dof || score_g <= best_g) && return (best_g, best_l)

    mask!(ws, scoring.perm, k_star)
    test_consensus(problem, model, ws.mask) || return (best_g, best_l)

    # Phase 2: Local optimization
    lo = _lo_refine!(ws, problem, model, local_optimization, scoring)

    # Phase 3: Re-score refined model (always model-certain)
    @inbounds for i in 1:n
        ws.scores[i] = ws.residuals[i]^2
    end
    measurement_logdets!(ws.penalties, problem, lo.model)
    score_l, k_l = sweep!(scoring, ws.scores, ws.penalties, n)
    mask!(ws, scoring.perm, k_l)
    score_l <= best_l && return (best_g, best_l)

    _update_best!(ws, lo.model)
    return (score_g, score_l)
end

# =============================================================================
# _fill_scores! — Trait-dispatched per-point scoring (Section 3.3)
# =============================================================================
#
# Three specializations of the constraint covariance shape Σ̃_{gᵢ} (Section 3.3):
#
#   Homoscedastic   (isotropic, dg=1):  qᵢ = rᵢ²,               ℓᵢ = 0
#   Heteroscedastic (model certain):    qᵢ = gᵢᵀΣ̃_{gᵢ}⁻¹gᵢ,    ℓᵢ = log|Σ̃_{gᵢ}|
#   Predictive      (model uncertain):  qᵢ = gᵢᵀΣ̃_{gᵢ}⁻¹gᵢ,    ℓᵢ = log|Σ̃_{gᵢ}|
#
# For Cases 2–3, residual_jacobian(problem, model, i) → (rᵢ, ∂θgᵢ_w, ℓᵢ_meas)
# computes the whitened quantities and measurement log-determinant in one pass,
# avoiding duplicate computation of _sampson_quantities.
# =============================================================================

"""
    _sq_norm(r) -> T

Squared norm: `r²` for scalar, `rᵀr` for vector. Computes the weighted
squared residual qᵢ = gᵢᵀ Σ̃_{gᵢ}⁻¹ gᵢ from whitened quantities (Eq. 12).
"""
@inline _sq_norm(r::Real) = r * r
@inline _sq_norm(r::SVector) = dot(r, r)

# Isotropic, dg=1 (Section 3.3): qᵢ = rᵢ², ℓᵢ = 0 (cᵢ cancels in Eq. 12)
function _fill_scores!(ws, problem, ::AbstractMarginalQuality, model, ::Homoscedastic)
    n = data_size(problem)
    residuals!(ws.residuals, problem, model)
    @inbounds for i in 1:n
        ws.scores[i] = ws.residuals[i]^2
    end
    fill!(ws.penalties, zero(eltype(ws.penalties)))
end

# Model certain (Section 3.3): qᵢ = gᵢᵀΣ̃_{gᵢ}⁻¹gᵢ, ℓᵢ = log|Σ̃_{gᵢ}| (Eq. 12)
# Uses residual_jacobian to compute whitened rᵢ = L⁻¹gᵢ and ℓᵢ in one pass.
function _fill_scores!(ws, problem, ::AbstractMarginalQuality, model, ::Heteroscedastic)
    n = data_size(problem)
    @inbounds for i in 1:n
        rᵢ, _, ℓᵢ = residual_jacobian(problem, model, i)
        ws.scores[i] = _sq_norm(rᵢ)         # qᵢ = rᵢᵀrᵢ = gᵢᵀΣ̃_{gᵢ}⁻¹gᵢ
        ws.residuals[i] = sqrt(ws.scores[i]) # |rᵢ| for adaptive stopping
        ws.penalties[i] = ℓᵢ                 # log|Σ̃_{gᵢ}|
    end
end

# Model uncertain (Section 3.3, Eq. 7, 28): Σ̃_{gᵢ} includes estimation error.
# Decomposition in whitened coordinates (LLᵀ = ∂ₓgᵢ Σ̃_{xᵢ} (∂ₓgᵢ)ᵀ):
#   Σ̃_{gᵢ} = LLᵀ + ∂θgᵢ Σ̃_θ (∂θgᵢ)ᵀ
#   log|Σ̃_{gᵢ}| = log|LLᵀ| + log|I + (L⁻¹∂θgᵢ) Σ̃_θ (L⁻¹∂θgᵢ)ᵀ|
#                = ℓᵢ_meas + ℓᵢ_model
function _fill_scores!(ws::RansacWorkspace{M,T}, problem,
                       scoring::PredictiveMarginalQuality,
                       model::M, ::Predictive) where {M,T}
    n = data_size(problem)
    Σ̃_θ = model_covariance(scoring, problem, model, ws.sample_indices)

    if !isnothing(Σ̃_θ)
        @inbounds for i in 1:n
            rᵢ, ∂θgᵢ_w, ℓᵢ_meas = residual_jacobian(problem, model, i)
            ws.residuals[i] = sqrt(_sq_norm(rᵢ))
            qᵢ, ℓᵢ_model = _predictive_score_penalty(rᵢ, ∂θgᵢ_w, Σ̃_θ, one(T))
            ws.scores[i] = qᵢ
            ws.penalties[i] = ℓᵢ_meas + ℓᵢ_model  # log|Σ̃_{gᵢ}|
        end
    else
        # Fallback to model-certain (Σ_θ = 0) when solver_jacobian unavailable
        _fill_scores!(ws, problem, scoring, model, measurement_covariance(problem))
    end
end

# =============================================================================
# _lo_quality — No-ops for non-iterative local optimization
# =============================================================================

_lo_quality(scoring, ws, problem, model, ::NoLocalOptimization) = Inf

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

# =============================================================================
# _finalize — AbstractMarginalQuality
# =============================================================================

function _finalize(scoring::AbstractMarginalQuality,
                   local_optimization::NoLocalOptimization,
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

    # Re-score to check if local optimization improved
    @inbounds for i in 1:n
        ws.scores[i] = ws.residuals[i]^2
    end
    measurement_logdets!(ws.penalties, problem, lo.model)
    score_final, k_final = sweep!(scoring, ws.scores, ws.penalties, n)
    mask!(ws, scoring.perm, k_final)

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
