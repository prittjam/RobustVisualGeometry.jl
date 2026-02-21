using Test
using StaticArrays
using Random
using LinearAlgebra

using VisualGeometryCore
using RobustVisualGeometry

# =============================================================================
# Helper: generate noiseless fundamental matrix correspondences + outliers
# =============================================================================

function make_fundmat_data(rng, n_inliers, n_outliers)
    # Two cameras with intrinsics K = diag(f, f, 1), f = 500
    f = 500.0
    K = SA[f 0 320.0; 0 f 240.0; 0 0 1.0]

    # Rotation ~17 degrees around Y axis + baseline
    θ = 0.3
    R = SA[cos(θ) 0 sin(θ); 0 1 0; -sin(θ) 0 cos(θ)]
    t = SA[1.0, 0.2, 0.1]

    # P1 = K[I|0], P2 = K[R|t]
    P1 = K * SA[1.0 0 0 0; 0 1 0 0; 0 0 1 0]
    P2 = K * hcat(R, t)

    # F = K⁻ᵀ [t]× R K⁻¹
    tx = SA[0 -t[3] t[2]; t[3] 0 -t[1]; -t[2] t[1] 0]
    F_true = inv(K)' * tx * R * inv(K)
    F_true = F_true / norm(F_true)
    if F_true[3,3] < 0
        F_true = -F_true
    end

    cs = Vector{Pair{SVector{2,Float64}, SVector{2,Float64}}}()

    for _ in 1:n_inliers
        X = SA[randn(rng)*2, randn(rng)*2, randn(rng)*0.5 + 5.0]
        p1h = P1 * SA[X[1], X[2], X[3], 1.0]
        p2h = P2 * SA[X[1], X[2], X[3], 1.0]
        s = SVector{2,Float64}(p1h[1]/p1h[3], p1h[2]/p1h[3])
        d = SVector{2,Float64}(p2h[1]/p2h[3], p2h[2]/p2h[3])
        push!(cs, s => d)
    end

    for _ in 1:n_outliers
        s = SVector{2,Float64}(randn(rng, 2) * 200 .+ 320)
        d = SVector{2,Float64}(randn(rng, 2) * 200 .+ 320)
        push!(cs, s => d)
    end

    return cs, F_true
end

function make_homography_data(rng, H_true, n_inliers, n_outliers)
    cs = Vector{Pair{SVector{2,Float64}, SVector{2,Float64}}}()
    for _ in 1:n_inliers
        s = SVector{2,Float64}(randn(rng, 2) * 200 .+ 300)
        sh = SA[s[1], s[2], 1.0]
        dh = H_true * sh
        d = SVector{2,Float64}(dh[1]/dh[3], dh[2]/dh[3])
        push!(cs, s => d)
    end
    for _ in 1:n_outliers
        s = SVector{2,Float64}(randn(rng, 2) * 200 .+ 300)
        d = SVector{2,Float64}(randn(rng, 2) * 200 .+ 300)
        push!(cs, s => d)
    end
    return cs
end

# =============================================================================
# Diagnostic 1: DLT precision on pure inlier data (no RANSAC)
# =============================================================================

@testset "F-matrix DLT — noiseless precision (no RANSAC)" begin
    rng = MersenneTwister(42)
    cs, F_true = make_fundmat_data(rng, 100, 0)

    u₁ = [c.first for c in cs]
    u₂ = [c.second for c in cs]

    F_est = fundamental_matrix_dlt(u₁, u₂)

    # Verify epipolar constraint
    max_epi = 0.0
    max_sampson = 0.0
    for i in eachindex(cs)
        s = cs[i].first; d = cs[i].second
        sh = SA[s[1], s[2], 1.0]; dh = SA[d[1], d[2], 1.0]
        max_epi = max(max_epi, abs(dot(dh, F_est * sh)))
        max_sampson = max(max_sampson, sampson_distance(s, d, F_est))
    end
    println("  DLT (100 noiseless): max_epipolar=$max_epi, max_sampson=$max_sampson")
    @test max_sampson < 1e-10
end

# =============================================================================
# Diagnostic 2: Check outlier Sampson distances with the true F
# =============================================================================

@testset "F-matrix — outlier Sampson distance distribution" begin
    rng = MersenneTwister(42)
    cs, F_true = make_fundmat_data(rng, 100, 30)

    println("  Sampson distances under TRUE F:")
    for i in 1:min(10, length(cs))
        s = cs[i].first; d = cs[i].second
        sd = sampson_distance(s, d, F_true)
        label = i <= 100 ? "inlier" : "outlier"
        println("    [$label] $i: sampson=$sd")
    end
    # Check the outliers specifically
    min_outlier_sampson = Inf
    for i in 101:130
        sd = sampson_distance(cs[i].first, cs[i].second, F_true)
        min_outlier_sampson = min(min_outlier_sampson, sd)
    end
    println("  Min outlier Sampson distance: $min_outlier_sampson")
    @test min_outlier_sampson > 0.1  # outliers should have large Sampson distance
end

# =============================================================================
# Homography noiseless tests
# =============================================================================

@testset "Homography RANSAC — noiseless theoretical guarantees" begin
    rng = MersenneTwister(42)
    H_true = SA[1.2 -0.1 30.0; 0.05 1.1 -20.0; 1e-4 2e-4 1.0]

    scenarios = [
        (100, 0,   "0% outliers"),
        (100, 30,  "23% outliers"),
        (100, 67,  "40% outliers"),
        (100, 100, "50% outliers"),
        (50,  100, "67% outliers"),
        (50,  150, "75% outliers"),
    ]

    for (n_in, n_out, label) in scenarios
        @testset "$label (n_in=$n_in, n_out=$n_out)" begin
            cs = make_homography_data(rng, H_true, n_in, n_out)
            shuffle!(rng, cs)

            problem = HomographyProblem(cs)
            config = RansacConfig(; max_trials=10_000, confidence=0.9999, min_trials=100)
            result = ransac(problem, ThresholdQuality(L2Loss(), 1e-4, FixedScale()); config)

            @test result.converged
            H_est = result.value

            H_est_n = H_est / H_est[3,3]
            H_ref = H_true / H_true[3,3]
            rel_err = norm(H_est_n - H_ref) / norm(H_ref)
            @test rel_err < 1e-8

            n_found = sum(result.inlier_mask)
            @test n_found >= n_in

            max_inlier_residual = 0.0
            for i in eachindex(cs)
                if result.inlier_mask[i]
                    err = symmetric_transfer_error(cs[i].first, cs[i].second, H_est)
                    max_inlier_residual = max(max_inlier_residual, err)
                end
            end
            @test max_inlier_residual < 1e-8

            println("  H $label: rel_err=$(round(rel_err, sigdigits=3)), " *
                    "inliers=$n_found/$n_in, max_res=$(round(max_inlier_residual, sigdigits=3))")
        end
    end
end

# =============================================================================
# F-matrix noiseless tests
# =============================================================================

@testset "F-matrix RANSAC — noiseless theoretical guarantees" begin
    rng = MersenneTwister(123)

    scenarios = [
        (100, 0,   "0% outliers"),
        (100, 30,  "23% outliers"),
        (100, 67,  "40% outliers"),
        (100, 100, "50% outliers"),
        (50,  100, "67% outliers"),
        (50,  150, "75% outliers"),
    ]

    for (n_in, n_out, label) in scenarios
        @testset "$label (n_in=$n_in, n_out=$n_out)" begin
            cs, F_true = make_fundmat_data(rng, n_in, n_out)
            shuffle!(rng, cs)

            problem = FundamentalMatrixProblem(cs)
            # P(clean 7-pt sample) = ((1-ε)^7). At 75% outliers: (0.25)^7 ≈ 6e-5
            # For 99.99% confidence: ceil(log(1e-4)/log(1-6e-5)) ≈ 150K trials
            config = RansacConfig(; max_trials=200_000, confidence=0.9999, min_trials=500)

            # L2Loss with very tight threshold for noiseless data.
            # True inliers have Sampson distance ~1e-15, outliers have ~O(1-100).
            # Threshold in L2 loss units: inlier if r²/2 < thresh, i.e. r < sqrt(2*thresh).
            # thresh=1e-8 → r < 1.4e-4 pixels — wide enough for machine precision,
            # tight enough to exclude all outliers.
            result = ransac(problem, ThresholdQuality(L2Loss(), 1e-8, FixedScale()); config)

            @test result.converged
            F_est = result.value

            n_found = sum(result.inlier_mask)
            @test n_found >= n_in

            max_inlier_sampson = 0.0
            for i in eachindex(cs)
                if result.inlier_mask[i]
                    err = sampson_distance(cs[i].first, cs[i].second, F_est)
                    max_inlier_sampson = max(max_inlier_sampson, err)
                end
            end
            @test max_inlier_sampson < 1e-6

            max_epipolar = 0.0
            for i in eachindex(cs)
                if result.inlier_mask[i]
                    sh = SA[cs[i].first[1], cs[i].first[2], 1.0]
                    dh = SA[cs[i].second[1], cs[i].second[2], 1.0]
                    max_epipolar = max(max_epipolar, abs(dot(dh, F_est * sh)))
                end
            end
            @test max_epipolar < 1e-4

            println("  F $label: inliers=$n_found/$n_in, " *
                    "max_sampson=$(round(max_inlier_sampson, sigdigits=3)), " *
                    "max_epipolar=$(round(max_epipolar, sigdigits=3))")
        end
    end
end
