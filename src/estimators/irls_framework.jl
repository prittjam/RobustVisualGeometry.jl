# =============================================================================
# Generic IRLS Framework
# =============================================================================
#
# Provides a generic iterate-reweight loop with extension points.
# Concrete methods: MEstimatorMethod (irls.jl), GNCMethod (gnc.jl),
# PosteriorIrlsMethod (ransac/loop.jl).
#
# Loop order: refit → residuals → scale → weights → check.
# init_state computes initial residuals/scale/weights from θ₀ so the first
# refit has valid weights.
# =============================================================================

"""
    AbstractIRLSMethod

Abstract type for IRLS method specializations.

Each concrete subtype defines seven extension points:
- `init_state(method, prob, θ₀)` — allocate state, compute initial r/σ/w
- `solve_step(state, method, prob, θ)` — weighted solve step
- `update_residuals!(state, method, prob, θ)` — compute residuals from θ
- `update_scale!(state, method, prob)` — estimate scale from residuals
- `update_weights!(state, method, prob)` — compute IRLS weights
- `post_step!(state, method, prob, θ, iter)` — per-iteration bookkeeping
- `is_converged(state, method, prob, θ, θ_old, iter)` — convergence check
- `irls_result(state, method, prob, θ, converged, final_iter)` — package result
"""
abstract type AbstractIRLSMethod end

# Default no-op for post_step!
post_step!(state, ::AbstractIRLSMethod, prob, θ, iter) = nothing

"""
    irls!(method::AbstractIRLSMethod, prob, θ₀; max_iter::Int) -> result

Generic IRLS loop with extension points.

Loop order: refit → residuals → scale → weights → check.
`init_state` computes initial residuals/scale/weights from θ₀
so the first `refit` has valid weights.
"""
function irls!(method::AbstractIRLSMethod, prob, θ₀; max_iter::Int)
    state = init_state(method, prob, θ₀)
    θ = θ₀
    converged = false
    final_iter = 0
    for iter in 1:max_iter
        final_iter = iter
        θ_old = θ
        θ_new = solve_step(state, method, prob, θ)
        isnothing(θ_new) && break
        θ = θ_new
        update_residuals!(state, method, prob, θ)
        update_scale!(state, method, prob)
        update_weights!(state, method, prob)
        post_step!(state, method, prob, θ, iter)
        if is_converged(state, method, prob, θ, θ_old, iter)
            converged = true
            break
        end
    end
    return irls_result(state, method, prob, θ, converged, final_iter)
end
