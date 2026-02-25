# =============================================================================
# Homography Estimation Example — All RANSAC Scoring Variants
# =============================================================================
#
# Demonstrates robust homography estimation from 2D point correspondences
# using HomographyProblem with every scoring variant:
#
#   1. MarginalQuality — threshold-free (d_g=2, Mahalanobis + log-det)
#   2. MarginalQuality + LO — with iterative refit
#   3. PredictiveMarginalQuality + LO — full predictive pipeline with EIV cov
#   4. ThresholdQuality (MSAC) — classical fixed-threshold baseline
#   5. ChiSquareQuality — chi-square inlier test baseline
#
# Run:  julia --project examples/homography_estimation.jl
#
# =============================================================================

using LinearAlgebra
using StaticArrays
using Random
using RobustVisualGeometry
using VisualGeometryCore.Matching: csponds

# =============================================================================
# Data Generation
# =============================================================================

"""Apply homography H to 2D point (homogeneous multiply + perspective divide)."""
function apply_H(H::SMatrix{3,3,T,9}, p::SVector{2,T}) where T
    h = H * SA[p[1], p[2], one(T)]
    SA[h[1]/h[3], h[2]/h[3]]
end

"""Max reprojection error between two homographies on test points."""
function max_reproj_error(H1, H2, test_pts)
    maximum(norm(apply_H(H1, p) - apply_H(H2, p)) for p in test_pts)
end

"""
    generate_homography_data(; n_inliers, n_outliers, noise, seed)

Generate point correspondences under a known homography with noise and outliers.
Returns `(correspondences, H_true, n_inliers)`.
"""
function generate_homography_data(; n_inliers::Int=120, n_outliers::Int=40,
                                    noise::Float64=0.5, seed::Int=42)
    rng = MersenneTwister(seed)

    # Ground truth: rotation + translation + mild perspective
    H_true = @SMatrix [
         0.95  -0.10  15.0;
         0.12   0.93  -8.0;
         1e-4   2e-4   1.0
    ]
    H_true = H_true / norm(H_true)
    if H_true[3,3] < 0
        H_true = -H_true
    end

    src = SVector{2,Float64}[]
    dst = SVector{2,Float64}[]

    # Inliers: source in [100,900] x [100,700], target = H*source + noise
    for _ in 1:n_inliers
        s = SA[100.0 + 800.0*rand(rng), 100.0 + 600.0*rand(rng)]
        d = apply_H(H_true, s) + SA[noise*randn(rng), noise*randn(rng)]
        push!(src, s)
        push!(dst, d)
    end

    # Outliers: random pairs
    for _ in 1:n_outliers
        s = SA[100.0 + 800.0*rand(rng), 100.0 + 600.0*rand(rng)]
        d = SA[100.0 + 800.0*rand(rng), 100.0 + 600.0*rand(rng)]
        push!(src, s)
        push!(dst, d)
    end

    return csponds(src, dst), H_true, n_inliers
end

# =============================================================================
# Helper: Print Result Summary
# =============================================================================

function print_result(label::String, result, H_true)
    H = result.value
    n_in = sum(result.inlier_mask)
    n_total = length(result.inlier_mask)

    # Reprojection error on a grid of test points
    test_pts = [SA[200.0, 300.0], SA[500.0, 200.0], SA[800.0, 500.0],
                SA[300.0, 600.0], SA[600.0, 400.0]]
    reproj = max_reproj_error(H, H_true, test_pts)

    println("\n--- $label ---")
    println("  Inliers:       $n_in / $n_total ($(round(100*n_in/n_total; digits=1))%)")
    println("  Max reproj:    $(round(reproj; digits=4)) px")
    println("  Scale:         $(round(result.scale; digits=6))")
    println("  DOF:           $(result.dof)")
    println("  Quality:       $(round(result.quality; digits=2))")
    println("  Trials:        $(result.trials)")

    if hasproperty(result, :param_cov) && !isnothing(result.param_cov)
        pc = result.param_cov
        println("  Param cov:     $(size(pc)) matrix, tr = $(round(tr(pc); digits=8))")
    end
end

# =============================================================================
# Standard Scenario: 75% inliers, moderate noise
# =============================================================================

println("=" ^ 72)
println("  Homography Estimation Example")
println("=" ^ 72)

cs, H_true, n_inliers = generate_homography_data(
    n_inliers=120, n_outliers=40, noise=0.5, seed=42)
println("\nData: $(length(cs)) correspondences ($(n_inliers) inliers, " *
        "$(length(cs) - n_inliers) outliers, noise sigma=0.5 px)")

prob = HomographyProblem(cs)
config = RansacConfig(max_trials=5000, min_trials=200)

# ---- 1. MarginalQuality (no LO) ----
scoring1 = MarginalQuality(prob, 50.0)
result1 = ransac(prob, scoring1; config)
print_result("MarginalQuality (no LO)", result1, H_true)

# ---- 2. MarginalQuality + LO ----
scoring2 = MarginalQuality(prob, 50.0; max_lo_iter=5)
result2 = ransac(prob, scoring2; config)
print_result("MarginalQuality + LO (5 iter)", result2, H_true)

# ---- 3. PredictiveMarginalQuality + LO ----
scoring3 = PredictiveMarginalQuality(prob, 50.0; max_lo_iter=5)
result3 = ransac(prob, scoring3; config)
print_result("PredictiveMarginalQuality + LO (5 iter)", result3, H_true)

# ---- 4. ThresholdQuality (MSAC baseline) ----
scoring4 = ThresholdQuality(CauchyLoss(), 3.0, FixedScale())
result4 = ransac(prob, scoring4; config)
print_result("ThresholdQuality / MSAC (Cauchy, threshold=3.0)", result4, H_true)

# ---- 5. ChiSquareQuality baseline ----
scoring5 = ChiSquareQuality(FixedScale(σ=0.5), 0.01)
result5 = ransac(prob, scoring5; config)
print_result("ChiSquareQuality (sigma=0.5, alpha=0.01)", result5, H_true)

# =============================================================================
# High-Outlier Scenario: ~33% inliers
# =============================================================================

println("\n\n" * "=" ^ 72)
println("  High-Outlier Scenario: ~33% inliers")
println("=" ^ 72)

cs_hard, H_true_hard, n_inliers_hard = generate_homography_data(
    n_inliers=50, n_outliers=100, noise=1.0, seed=99)
println("\nData: $(length(cs_hard)) correspondences ($(n_inliers_hard) inliers, " *
        "$(length(cs_hard) - n_inliers_hard) outliers, noise sigma=1.0 px)")

prob_hard = HomographyProblem(cs_hard)
config_hard = RansacConfig(max_trials=10_000, min_trials=500)

# Marginal scoring — no threshold needed
scoring_m = MarginalQuality(prob_hard, 50.0; max_lo_iter=5)
result_m = ransac(prob_hard, scoring_m; config=config_hard)
print_result("MarginalQuality + LO (high outliers)", result_m, H_true_hard)

# PredictiveMarginalQuality
scoring_pm = PredictiveMarginalQuality(prob_hard, 50.0; max_lo_iter=5)
result_pm = ransac(prob_hard, scoring_pm; config=config_hard)
print_result("PredictiveMarginalQuality + LO (high outliers)", result_pm, H_true_hard)

# MSAC baseline
scoring_t = ThresholdQuality(CauchyLoss(), 3.0, FixedScale())
result_t = ransac(prob_hard, scoring_t; config=config_hard)
print_result("ThresholdQuality / MSAC (high outliers)", result_t, H_true_hard)

# =============================================================================
# Result Inspection: Homography Matrix
# =============================================================================

println("\n\n" * "=" ^ 72)
println("  Best Model Inspection (PredictiveMarginalQuality)")
println("=" ^ 72)

H_best = result3.value
println("\nEstimated H (Frobenius-normalized):")
for row in 1:3
    vals = [round(H_best[row, col]; digits=8) for col in 1:3]
    println("  ", vals)
end
println("\nTrue H:")
for row in 1:3
    vals = [round(H_true[row, col]; digits=8) for col in 1:3]
    println("  ", vals)
end

println("\n" * "=" ^ 72)
println("  Done.")
println("=" ^ 72)
