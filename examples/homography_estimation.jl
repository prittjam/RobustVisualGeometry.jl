# =============================================================================
# Homography Estimation Example — "RANSAC Done Right" Paper
# =============================================================================
#
# Demonstrates scale-free marginal likelihood scoring on homography estimation
# from point correspondences.
#
# For homography, the co-dimension d_g = 2 (two constraint equations from
# the projective transfer v ~ H u), so the marginal scoring uses 2D
# Sampson-whitened Mahalanobis scores.
#
# Scoring variants:
#   1. MarginalQuality — scale-free, no sigma needed (Algorithm 1)
#   2. PredictiveMarginalQuality — leverage-corrected (Algorithm 2)
#
# Usage:
#   julia --project examples/homography_estimation.jl
#
# =============================================================================

using RobustVisualGeometry
using VisualGeometryCore: csponds
using Random
using StaticArrays
using LinearAlgebra
using Printf

# =============================================================================
# Helpers
# =============================================================================

"""Apply homography H to 2D point (homogeneous multiply + perspective divide)."""
function apply_homography(H::SMatrix{3,3,T,9}, p::SVector{2,T}) where T
    h = H * SA[p[1], p[2], one(T)]
    return SA[h[1]/h[3], h[2]/h[3]]
end

"""Max reprojection error of estimated H vs ground truth on test points."""
function max_reprojection_error(H_est, H_true, test_points)
    max_err = 0.0
    for p in test_points
        p1 = apply_homography(H_est, p)
        p2 = apply_homography(H_true, p)
        max_err = max(max_err, norm(p1 - p2))
    end
    return max_err
end

"""Mean reprojection error of estimated H vs ground truth on test points."""
function mean_reprojection_error(H_est, H_true, test_points)
    total = 0.0
    for p in test_points
        p1 = apply_homography(H_est, p)
        p2 = apply_homography(H_true, p)
        total += norm(p1 - p2)
    end
    return total / length(test_points)
end

# =============================================================================
# Data Generation
# =============================================================================

function generate_homography_data(; rng=MersenneTwister(42),
                                    n_inliers=100, n_outliers=40,
                                    noise_px=1.0)
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

    source_pts = SVector{2, Float64}[]
    target_pts = SVector{2, Float64}[]

    # Inliers: source in [100, 900] x [100, 700], target = H * source + noise
    for _ in 1:n_inliers
        s = SA[100.0 + 800.0 * rand(rng), 100.0 + 600.0 * rand(rng)]
        d = apply_homography(H_true, s) + SA[noise_px * randn(rng), noise_px * randn(rng)]
        push!(source_pts, s)
        push!(target_pts, d)
    end

    # Outliers: random pairs
    for _ in 1:n_outliers
        s = SA[100.0 + 800.0 * rand(rng), 100.0 + 600.0 * rand(rng)]
        d = SA[100.0 + 800.0 * rand(rng), 100.0 + 600.0 * rand(rng)]
        push!(source_pts, s)
        push!(target_pts, d)
    end

    # Test points for reprojection error evaluation (on a grid, no noise)
    test_pts = [SA[Float64(x), Float64(y)]
                for x in 200:200:800 for y in 200:200:600]

    return source_pts, target_pts, H_true, n_inliers, test_pts
end

# =============================================================================
# Print Results
# =============================================================================

function print_result(name, result, H_true, test_pts, n_inliers)
    H_est = result.value
    attrs = result.attributes
    n = length(attrs.inlier_mask)
    n_in = sum(attrs.inlier_mask)
    true_in = sum(attrs.inlier_mask[1:n_inliers])
    false_in = sum(attrs.inlier_mask[n_inliers+1:end])

    # Reprojection error
    mean_err = mean_reprojection_error(H_est, H_true, test_pts)
    max_err = max_reprojection_error(H_est, H_true, test_pts)

    @printf("  %-45s  inliers=%d/%d  true_in=%d  false_in=%d",
            name, n_in, n, true_in, false_in)
    @printf("  reproj: mean=%.2f max=%.2f", mean_err, max_err)

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
    println("=" ^ 90)
    println("Homography Estimation Example — RANSAC Scoring Comparison")
    println("=" ^ 90)

    # --- Scenario 1: Moderate noise, moderate outliers ---
    src, dst, H_true, n_inliers, test_pts = generate_homography_data(;
        rng=MersenneTwister(42), n_inliers=100, n_outliers=40, noise_px=1.0)
    n = length(src)
    println("\nScenario 1: $(n_inliers) inliers + $(n - n_inliers) outliers " *
            "($(round(100*n_inliers/n, digits=1))% inlier rate, noise=1.0 px)")
    println()

    correspondences = csponds(src, dst)
    prob = HomographyProblem(correspondences)
    config = RansacConfig(; max_trials=2000, confidence=0.999)

    # 1. MarginalQuality — no sigma needed (Algorithm 1)
    s1 = MarginalQuality(prob, 50.0)
    r1 = ransac(prob, s1; config)
    print_result("MarginalQuality", r1, H_true, test_pts, n_inliers)

    # 2. PredictiveMarginalQuality — leverage-corrected (Algorithm 2)
    s2 = PredictiveMarginalQuality(prob, 50.0)
    r2 = ransac(prob, s2; config)
    print_result("PredictiveMarginalQuality", r2, H_true, test_pts, n_inliers)

    # --- Scenario 2: High noise + high outlier rate ---
    println("\n" * "-" ^ 90)
    src2, dst2, H_true2, n_in2, test_pts2 = generate_homography_data(;
        rng=MersenneTwister(99), n_inliers=50, n_outliers=100, noise_px=3.0)
    n2 = length(src2)
    println("\nScenario 2: $(n_in2) inliers + $(n2 - n_in2) outliers " *
            "($(round(100*n_in2/n2, digits=1))% inlier rate, noise=3.0 px — challenging)")
    println()

    cs2 = csponds(src2, dst2)
    prob2 = HomographyProblem(cs2)
    config2 = RansacConfig(; max_trials=5000, confidence=0.999)

    # Marginal — adapts automatically
    sh1 = MarginalQuality(prob2, 100.0)
    rh1 = ransac(prob2, sh1; config=config2)
    print_result("MarginalQuality", rh1, H_true2, test_pts2, n_in2)

    # Predictive — leverage-corrected
    sh2 = PredictiveMarginalQuality(prob2, 100.0)
    rh2 = ransac(prob2, sh2; config=config2)
    print_result("PredictiveMarginalQuality", rh2, H_true2, test_pts2, n_in2)

    println("\n" * "=" ^ 90)
    println("Key takeaway: MarginalQuality scores models without needing a noise estimate,")
    println("adapting to the true inlier noise level. PredictiveMarginalQuality additionally")
    println("accounts for the geometric conditioning of the minimal sample (leverage).")
    println("=" ^ 90)
end

main()
