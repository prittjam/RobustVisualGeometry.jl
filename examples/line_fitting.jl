# =============================================================================
# Line Fitting Example — All RANSAC Scoring Variants
# =============================================================================
#
# Demonstrates robust line fitting (y = a + bx) using InhomLineFittingProblem
# with every scoring variant available in RobustVisualGeometry:
#
#   1. MarginalQuality — threshold-free marginal likelihood (Eq. 17)
#   2. MarginalQuality + LO — with iterative refit
#   3. PredictiveMarginalQuality + LO — leverage-corrected (Eq. 30)
#   4. ThresholdQuality (MSAC) — classical fixed-threshold baseline
#   5. ChiSquareQuality — chi-square inlier test baseline
#
# Run:  julia --project examples/line_fitting.jl
#
# =============================================================================

using LinearAlgebra
using StaticArrays
using Random
using RobustVisualGeometry

# =============================================================================
# Data Generation
# =============================================================================

"""
    generate_line_data(; n_inliers, n_outliers, noise, outlier_range, seed)

Generate 2D points on `y = 0.5 + 2x` with Gaussian noise, plus uniform outliers.
Returns `(points, true_params, n_inliers)` where `true_params = [a, b]`.
"""
function generate_line_data(; n_inliers::Int=150, n_outliers::Int=50,
                              noise::Float64=0.3, outlier_range::Float64=50.0,
                              seed::Int=42)
    rng = MersenneTwister(seed)
    a_true, b_true = 0.5, 2.0

    points = SVector{2,Float64}[]

    # Inliers: x uniform in [-5, 5], y = a + b*x + noise
    for _ in 1:n_inliers
        x = 10.0 * rand(rng) - 5.0
        y = a_true + b_true * x + noise * randn(rng)
        push!(points, SA[x, y])
    end

    # Outliers: uniform in [-outlier_range/2, outlier_range/2]^2
    for _ in 1:n_outliers
        x = outlier_range * rand(rng) - outlier_range / 2
        y = outlier_range * rand(rng) - outlier_range / 2
        push!(points, SA[x, y])
    end

    return points, SA[a_true, b_true], n_inliers
end

# =============================================================================
# Helper: Print Result Summary
# =============================================================================

function print_result(label::String, result, true_params)
    model = result.value
    n_inliers = sum(result.inlier_mask)
    n_total = length(result.inlier_mask)
    println("\n--- $label ---")
    println("  Model:     y = $(round(model[1]; digits=4)) + $(round(model[2]; digits=4)) x")
    println("  True:      y = $(true_params[1]) + $(true_params[2]) x")
    println("  Param err: |da| = $(round(abs(model[1] - true_params[1]); digits=6)), " *
            "|db| = $(round(abs(model[2] - true_params[2]); digits=6))")
    println("  Inliers:   $n_inliers / $n_total ($(round(100*n_inliers/n_total; digits=1))%)")
    println("  Scale:     $(round(result.scale; digits=6))")
    println("  DOF:       $(result.dof)")
    println("  Quality:   $(round(result.quality; digits=2))")
    println("  Trials:    $(result.trials)")

    if hasproperty(result, :param_cov) && !isnothing(result.param_cov)
        pc = result.param_cov
        println("  Param cov: diag = [$(round(pc[1,1]; digits=8)), $(round(pc[2,2]; digits=8))]")
    end
end

# =============================================================================
# Standard Scenario: 75% inliers, moderate noise
# =============================================================================

println("=" ^ 72)
println("  Line Fitting Example: y = a + bx")
println("=" ^ 72)

points, true_params, n_inliers = generate_line_data(
    n_inliers=150, n_outliers=50, noise=0.3, seed=42)
println("\nData: $(length(points)) points ($(n_inliers) inliers, " *
        "$(length(points) - n_inliers) outliers, noise sigma=0.3)")

prob = InhomLineFittingProblem(points)
config = RansacConfig(max_trials=5000, min_trials=200)

# ---- 1. MarginalQuality (no LO) ----
scoring1 = MarginalQuality(prob, 50.0)
result1 = ransac(prob, scoring1; config)
print_result("MarginalQuality (no LO)", result1, true_params)

# ---- 2. MarginalQuality + LO ----
scoring2 = MarginalQuality(prob, 50.0; max_lo_iter=5)
result2 = ransac(prob, scoring2; config)
print_result("MarginalQuality + LO (5 iter)", result2, true_params)

# ---- 3. PredictiveMarginalQuality + LO ----
scoring3 = PredictiveMarginalQuality(prob, 50.0; max_lo_iter=5)
result3 = ransac(prob, scoring3; config)
print_result("PredictiveMarginalQuality + LO (5 iter)", result3, true_params)

# ---- 4. ThresholdQuality (MSAC baseline) ----
# Threshold chosen as ~3*noise_sigma for the truncation
scoring4 = ThresholdQuality(L2Loss(), 1.0, FixedScale())
result4 = ransac(prob, scoring4; config)
print_result("ThresholdQuality / MSAC (threshold=1.0)", result4, true_params)

# ---- 5. ChiSquareQuality baseline ----
# noise sigma ~0.3 for this problem, alpha=0.01
scoring5 = ChiSquareQuality(FixedScale(σ=0.3), 0.01)
result5 = ransac(prob, scoring5; config)
print_result("ChiSquareQuality (sigma=0.3, alpha=0.01)", result5, true_params)

# =============================================================================
# High-Outlier Scenario: 25% inliers
# =============================================================================

println("\n\n" * "=" ^ 72)
println("  High-Outlier Scenario: 25% inliers")
println("=" ^ 72)

points_hard, true_params_hard, n_inliers_hard = generate_line_data(
    n_inliers=50, n_outliers=150, noise=0.3, seed=99)
println("\nData: $(length(points_hard)) points ($(n_inliers_hard) inliers, " *
        "$(length(points_hard) - n_inliers_hard) outliers, noise sigma=0.3)")

prob_hard = InhomLineFittingProblem(points_hard)
config_hard = RansacConfig(max_trials=10_000, min_trials=500)

# Marginal scoring adapts threshold automatically — no sigma needed
scoring_m = MarginalQuality(prob_hard, 50.0; max_lo_iter=5)
result_m = ransac(prob_hard, scoring_m; config=config_hard)
print_result("MarginalQuality + LO (high outliers)", result_m, true_params_hard)

# MSAC needs a good threshold choice — harder with unknown noise
scoring_t = ThresholdQuality(L2Loss(), 1.0, FixedScale())
result_t = ransac(prob_hard, scoring_t; config=config_hard)
print_result("ThresholdQuality / MSAC (high outliers)", result_t, true_params_hard)

println("\n" * "=" ^ 72)
println("  Done.")
println("=" ^ 72)
