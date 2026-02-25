# =============================================================================
# Line Fitting Example — "RANSAC Done Right" Paper
# =============================================================================
#
# Demonstrates scale-free marginal likelihood scoring on a simple
# inhomogeneous line fitting problem (y = a + bx).
#
# Scoring variants:
#   1. MarginalQuality — scale-free (Algorithm 1), no sigma needed
#   2. PredictiveMarginalQuality — leverage-corrected (Algorithm 2)
#
# Usage:
#   julia --project examples/line_fitting.jl
#
# =============================================================================

using RobustVisualGeometry
using Random
using StaticArrays
using Printf

# =============================================================================
# Data Generation
# =============================================================================

function generate_line_data(; rng=MersenneTwister(42),
                             a_true=2.0, b_true=3.0,
                             n_inliers=80, n_outliers=40,
                             noise_std=0.5, outlier_range=50.0,
                             x_range=(-10.0, 10.0))
    # Inliers: y = a + b*x + noise
    x_in = range(x_range[1], x_range[2], length=n_inliers)
    pts_in = [SA[x, a_true + b_true * x + noise_std * randn(rng)] for x in x_in]

    # Outliers: uniformly distributed
    pts_out = [SA[x_range[1] + (x_range[2] - x_range[1]) * rand(rng),
                  outlier_range * (rand(rng) - 0.5)] for _ in 1:n_outliers]

    pts = vcat(pts_in, pts_out)
    return pts, n_inliers, (a=a_true, b=b_true)
end

# =============================================================================
# Helper: Print Results
# =============================================================================

function print_result(name, result, truth, n_inliers)
    model = result.value
    attrs = result.attributes
    n = length(attrs.inlier_mask)
    n_in = sum(attrs.inlier_mask)
    true_inliers_found = sum(attrs.inlier_mask[1:n_inliers])
    false_inliers = sum(attrs.inlier_mask[n_inliers+1:end])

    @printf("  %-45s  a=%.3f  b=%.3f  |  inliers=%d/%d  true_in=%d  false_in=%d",
            name, model[1], model[2], n_in, n, true_inliers_found, false_inliers)

    if hasproperty(attrs, :scale) && !isnan(attrs.scale)
        @printf("  s=%.3f", attrs.scale)
    end
    if hasproperty(attrs, :dof) && attrs.dof > 0
        @printf("  dof=%d", attrs.dof)
    end
    println()
end

# =============================================================================
# Main
# =============================================================================

function main()
    println("=" ^ 80)
    println("Line Fitting Example — RANSAC Scoring Comparison")
    println("=" ^ 80)

    # --- Standard scenario ---
    pts, n_inliers, truth = generate_line_data()
    n = length(pts)
    inlier_frac = n_inliers / n
    println("\nScenario 1: $(n_inliers) inliers + $(n - n_inliers) outliers " *
            "($(round(100*inlier_frac, digits=1))% inlier rate)")
    println("  Ground truth: a=$(truth.a), b=$(truth.b), noise_std=0.5")
    println()

    prob = InhomLineFittingProblem(pts)
    config = RansacConfig(; max_trials=2000, confidence=0.999)

    # 1. MarginalQuality — no sigma needed (Algorithm 1)
    scoring1 = MarginalQuality(prob, 50.0)
    result1 = ransac(prob, scoring1; config)
    print_result("MarginalQuality", result1, truth, n_inliers)

    # 2. PredictiveMarginalQuality — leverage-corrected (Algorithm 2)
    scoring2 = PredictiveMarginalQuality(prob, 50.0)
    result2 = ransac(prob, scoring2; config)
    print_result("PredictiveMarginalQuality", result2, truth, n_inliers)

    # --- High-outlier scenario ---
    println("\n" * "-" ^ 80)
    pts_hard, n_in_hard, truth_hard = generate_line_data(;
        rng=MersenneTwister(123),
        a_true=1.0, b_true=-2.0,
        n_inliers=30, n_outliers=70,
        noise_std=1.0, outlier_range=100.0)
    n_hard = length(pts_hard)
    println("\nScenario 2: $(n_in_hard) inliers + $(n_hard - n_in_hard) outliers " *
            "($(round(100*n_in_hard/n_hard, digits=1))% inlier rate — challenging)")
    println("  Ground truth: a=$(truth_hard.a), b=$(truth_hard.b), noise_std=1.0")
    println()

    prob_hard = InhomLineFittingProblem(pts_hard)
    config_hard = RansacConfig(; max_trials=5000, confidence=0.999)

    # Marginal — adapts automatically
    s_hard1 = MarginalQuality(prob_hard, 100.0)
    r_hard1 = ransac(prob_hard, s_hard1; config=config_hard)
    print_result("MarginalQuality", r_hard1, truth_hard, n_in_hard)

    # Predictive — leverage-corrected
    s_hard2 = PredictiveMarginalQuality(prob_hard, 100.0)
    r_hard2 = ransac(prob_hard, s_hard2; config=config_hard)
    print_result("PredictiveMarginalQuality", r_hard2, truth_hard, n_in_hard)

    println("\n" * "=" ^ 80)
    println("Key takeaway: MarginalQuality adapts to the noise level automatically,")
    println("without requiring a sigma estimate. PredictiveMarginalQuality additionally")
    println("accounts for the geometric conditioning of the minimal sample (leverage).")
    println("=" ^ 80)
end

main()
