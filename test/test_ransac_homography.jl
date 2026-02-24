using Test
using LinearAlgebra
using StaticArrays
using Random
using VisualGeometryCore
using VisualGeometryCore: ScoringTrait, HasScore, NoScore, scoring
using RobustVisualGeometry
using RobustVisualGeometry: sample_size, data_size, model_type,
    solver_cardinality, SingleSolution, ProsacSampler, reset!,
    RansacRefineProblem, MEstimator, FixedScale, robust_solve
using VisualGeometryCore.Matching: Attributed, ScoredCspond, csponds

# =============================================================================
# Test Utilities
# =============================================================================

"""Apply homography H to 2D point (homogeneous multiply + perspective divide)."""
function apply_homography(H::SMatrix{3,3,T,9}, p::SVector{2,T}) where T
    h = H * SA[p[1], p[2], one(T)]
    return SA[h[1]/h[3], h[2]/h[3]]
end

"""Max reprojection error between two homographies on a set of test points."""
function max_reprojection_error(H1, H2, test_points)
    max_err = 0.0
    for p in test_points
        p1 = apply_homography(H1, p)
        p2 = apply_homography(H2, p)
        max_err = max(max_err, norm(p1 - p2))
    end
    return max_err
end

# =============================================================================
# Test Data Generation
# =============================================================================

function make_homography_data(; n_inliers=100, n_outliers=30, noise=0.5, seed=42)
    rng = MersenneTwister(seed)

    # Ground truth homography: mild projective transform
    # (rotation + translation + slight perspective)
    H_true = @SMatrix [
         0.95  -0.10  15.0;
         0.12   0.93  -8.0;
         1e-4   2e-4   1.0
    ]
    # Normalize
    H_true = H_true / norm(H_true)
    if H_true[3,3] < 0
        H_true = -H_true
    end

    # Generate source points in image coordinate range [100, 900] x [100, 700]
    source_pts = SVector{2, Float64}[]
    target_pts = SVector{2, Float64}[]
    for _ in 1:n_inliers
        s = SA[100.0 + 800.0 * rand(rng), 100.0 + 600.0 * rand(rng)]
        d = apply_homography(H_true, s) + SA[noise * randn(rng), noise * randn(rng)]
        push!(source_pts, s)
        push!(target_pts, d)
    end

    # Outliers: random point pairs
    for _ in 1:n_outliers
        s = SA[100.0 + 800.0 * rand(rng), 100.0 + 600.0 * rand(rng)]
        d = SA[100.0 + 800.0 * rand(rng), 100.0 + 600.0 * rand(rng)]
        push!(source_pts, s)
        push!(target_pts, d)
    end

    return source_pts, target_pts, H_true, n_inliers
end

# =============================================================================
# Tests
# =============================================================================

@testset "RANSAC Homography" begin

    @testset "HomographyProblem construction" begin
        src = [SA[1.0, 2.0], SA[3.0, 4.0], SA[5.0, 6.0], SA[7.0, 8.0]]
        dst = [SA[2.0, 3.0], SA[4.0, 5.0], SA[6.0, 7.0], SA[8.0, 9.0]]
        p = HomographyProblem(csponds(src, dst))
        @test sample_size(p) == 4
        @test data_size(p) == 4
        @test model_type(p) == SMatrix{3,3,Float64,9}
        @test solver_cardinality(p) isa SingleSolution

        # Too few correspondences
        @test_throws ArgumentError HomographyProblem(csponds(src[1:3], dst[1:3]))

        # UniformSampler for Pair correspondences
        @test p isa HomographyProblem{Float64, UniformSampler}
    end

    @testset "ScoringTrait dispatch" begin
        # Pair correspondences → NoScore
        @test scoring(Pair{SVector{2,Float64}, SVector{2,Float64}}) === NoScore()

        # ScoredCspond → HasScore
        @test scoring(ScoredCspond{SVector{2,Float64}, SVector{2,Float64}}) === HasScore()

        # HomographyProblem from scored correspondences → HasScore
        src = [SA[1.0, 2.0], SA[3.0, 4.0], SA[5.0, 6.0], SA[7.0, 8.0]]
        dst = [SA[2.0, 3.0], SA[4.0, 5.0], SA[6.0, 7.0], SA[8.0, 9.0]]
        scored = [ScoredCspond(s, d, Float64(i)) for (i, (s, d)) in enumerate(zip(src, dst))]
        p = HomographyProblem(scored)
        @test p isa HomographyProblem{Float64, ProsacSampler}
        @test sampler(p) isa ProsacSampler
    end

    @testset "4-point DLT solver — identity" begin
        # 4 points mapped by identity should recover identity
        src = [SA[100.0, 200.0], SA[500.0, 100.0], SA[800.0, 600.0], SA[200.0, 700.0]]
        dst = copy(src)
        p = HomographyProblem(csponds(src, dst))

        H = solve(p, [1, 2, 3, 4])
        @test !isnothing(H)

        # Should be close to identity (up to scale)
        H_id = SMatrix{3,3,Float64,9}(1,0,0, 0,1,0, 0,0,1) / norm(SMatrix{3,3,Float64,9}(1,0,0, 0,1,0, 0,0,1))
        # Compare via reprojection
        test_pts = [SA[300.0, 400.0], SA[600.0, 300.0]]
        @test max_reprojection_error(H, H_id, test_pts) < 1e-10
    end

    @testset "4-point DLT solver — known H" begin
        H_true = @SMatrix [
            0.95 -0.10  15.0;
            0.12  0.93  -8.0;
            1e-4  2e-4   1.0
        ]
        H_true = H_true / norm(H_true)

        src = [SA[100.0, 200.0], SA[500.0, 100.0], SA[800.0, 600.0], SA[200.0, 700.0]]
        dst = [apply_homography(H_true, s) for s in src]

        p = HomographyProblem(csponds(src, dst))
        H = solve(p, [1, 2, 3, 4])
        @test !isnothing(H)

        test_pts = [SA[300.0, 400.0], SA[600.0, 300.0], SA[450.0, 550.0]]
        @test max_reprojection_error(H, H_true, test_pts) < 1e-8
    end

    @testset "Inhomogeneous solver — small h33 case" begin
        # Homography with very small h33 (near-degeneracy for DLT with h33=1)
        H_true = @SMatrix [
            1.0  0.0    0.0;
            0.0  1.0    0.0;
            0.002 0.001  1e-4
        ]
        H_true = H_true / norm(H_true)
        if H_true[3,3] < 0; H_true = -H_true; end

        src = [SA[100.0, 200.0], SA[500.0, 100.0], SA[800.0, 600.0], SA[200.0, 700.0]]
        dst = [apply_homography(H_true, s) for s in src]

        H = homography_4pt(src[1], src[2], src[3], src[4],
                                                dst[1], dst[2], dst[3], dst[4])
        @test !isnothing(H)

        test_pts = [SA[300.0, 400.0], SA[600.0, 300.0], SA[450.0, 550.0]]
        @test max_reprojection_error(H, H_true, test_pts) < 1e-6
    end

    @testset "Inhomogeneous solver — pure translation" begin
        H_true = @SMatrix [1.0 0.0 50.0; 0.0 1.0 -30.0; 0.0 0.0 1.0]
        H_true = H_true / norm(H_true)

        src = [SA[100.0, 200.0], SA[500.0, 100.0], SA[800.0, 600.0], SA[200.0, 700.0]]
        dst = [apply_homography(H_true, s) for s in src]

        H = homography_4pt(src[1], src[2], src[3], src[4],
                                                dst[1], dst[2], dst[3], dst[4])
        @test !isnothing(H)

        test_pts = [SA[300.0, 400.0], SA[600.0, 300.0]]
        @test max_reprojection_error(H, H_true, test_pts) < 1e-10
    end

    @testset "Inhomogeneous solver — strong perspective" begin
        H_true = @SMatrix [
            2.0   0.5   100.0;
           -0.3   1.5   -50.0;
            0.002 0.001   1.0
        ]
        H_true = H_true / norm(H_true)
        if H_true[3,3] < 0; H_true = -H_true; end

        src = [SA[100.0, 200.0], SA[500.0, 100.0], SA[800.0, 600.0], SA[200.0, 700.0]]
        dst = [apply_homography(H_true, s) for s in src]

        H = homography_4pt(src[1], src[2], src[3], src[4],
                                                dst[1], dst[2], dst[3], dst[4])
        @test !isnothing(H)

        test_pts = [SA[300.0, 400.0], SA[600.0, 300.0], SA[450.0, 550.0]]
        @test max_reprojection_error(H, H_true, test_pts) < 1e-8
    end

    @testset "Inhomogeneous solver — collinear returns nothing" begin
        # Collinear source points → should return nothing
        src_c = [SA[0.0, 0.0], SA[1.0, 1.0], SA[2.0, 2.0], SA[3.0, 3.0]]
        dst_ok = [SA[10.0, 20.0], SA[30.0, 10.0], SA[50.0, 40.0], SA[20.0, 60.0]]

        H = homography_4pt(src_c[1], src_c[2], src_c[3], src_c[4],
                                                dst_ok[1], dst_ok[2], dst_ok[3], dst_ok[4])
        @test isnothing(H)
    end

    @testset "Symmetric transfer error" begin
        H_true = @SMatrix [1.0 0.0 10.0; 0.0 1.0 -5.0; 0.0 0.0 1.0]
        H_true = H_true / norm(H_true)

        src = [SA[100.0, 200.0], SA[0.0,0.0], SA[1.0,0.0], SA[0.0,1.0]]
        dst = [apply_homography(H_true, src[1]), SA[10.0,-5.0], SA[11.0,-5.0], SA[10.0,-4.0]]
        p = HomographyProblem(csponds(src, dst))

        r = Vector{Float64}(undef, 4)
        residuals!(r, p, H_true)
        # Inlier residuals should be near zero
        @test r[1] < 1e-10
    end

    @testset "Degeneracy check" begin
        # Exactly collinear points should be rejected
        src_collinear = [SA[0.0, 0.0], SA[1.0, 1.0], SA[2.0, 2.0], SA[3.0, 3.0], SA[100.0, 100.0]]
        dst_ok = [SA[10.0, 20.0], SA[30.0, 10.0], SA[50.0, 40.0], SA[20.0, 60.0], SA[200.0, 200.0]]
        p = HomographyProblem(csponds(src_collinear, dst_ok))
        @test !test_sample(p, [1, 2, 3, 4])

        # Nearly collinear at image scale (< 2° angle) should be rejected
        # Points along y=500 with a tiny 0.1px offset over 1000px span → angle ≈ 0.006°
        src_near = [SA[100.0, 500.0], SA[600.0, 500.0], SA[1100.0, 500.1], SA[500.0, 200.0]]
        dst_near = [SA[10.0, 20.0], SA[30.0, 10.0], SA[50.0, 40.0], SA[20.0, 60.0]]
        p_near = HomographyProblem(csponds(src_near, dst_near))
        @test !test_sample(p_near, [1, 2, 3, 4])

        # Coincident points should be rejected
        src_coinc = [SA[500.0, 500.0], SA[500.0, 500.0], SA[800.0, 600.0], SA[200.0, 700.0]]
        dst_coinc = [SA[150.0, 250.0], SA[550.0, 150.0], SA[850.0, 650.0], SA[250.0, 750.0]]
        p_coinc = HomographyProblem(csponds(src_coinc, dst_coinc))
        @test !test_sample(p_coinc, [1, 2, 3, 4])

        # Non-degenerate points should pass
        src_ok = [SA[100.0, 200.0], SA[500.0, 100.0], SA[800.0, 600.0], SA[200.0, 700.0]]
        dst_ok2 = [SA[150.0, 250.0], SA[550.0, 150.0], SA[850.0, 650.0], SA[250.0, 750.0]]
        p2 = HomographyProblem(csponds(src_ok, dst_ok2))
        @test test_sample(p2, [1, 2, 3, 4])
    end

    @testset "RANSAC — Cauchy + FixedScale" begin
        source_pts, target_pts, H_true, n_inliers = make_homography_data(
            n_inliers=100, n_outliers=30, noise=0.5, seed=42)
        problem = HomographyProblem(csponds(source_pts, target_pts))

        result = ransac(problem, MarginalQuality(problem, 50.0);
                        config=RansacConfig(max_trials=3000, min_trials=200))

        @test result.converged
        @test result.converged

        # Verify model recovery via reprojection
        H = result.value
        test_pts = [SA[300.0, 400.0], SA[600.0, 300.0], SA[450.0, 550.0],
                    SA[200.0, 200.0], SA[700.0, 500.0]]
        @test max_reprojection_error(H, H_true, test_pts) < 2.0

        # Check inlier ratio
        @test inlier_ratio(result) > 0.5
        @test sum(result.inlier_mask) >= n_inliers * 0.7
    end

    @testset "RANSAC — L2 + FixedScale" begin
        source_pts, target_pts, H_true, _ = make_homography_data(
            n_inliers=100, n_outliers=30, noise=0.3, seed=123)
        problem = HomographyProblem(csponds(source_pts, target_pts))

        # L2 with tight threshold (MSAC truncation)
        result = ransac(problem, MarginalQuality(problem, 50.0);
                        config=RansacConfig(max_trials=3000))

        @test result.converged
        H = result.value
        test_pts = [SA[400.0, 300.0], SA[600.0, 500.0]]
        @test max_reprojection_error(H, H_true, test_pts) < 2.0
    end

    @testset "RANSAC — high outlier rate" begin
        source_pts, target_pts, H_true, _ = make_homography_data(
            n_inliers=50, n_outliers=100, noise=0.5, seed=456)
        problem = HomographyProblem(csponds(source_pts, target_pts))

        result = ransac(problem, MarginalQuality(problem, 50.0);
                        config=RansacConfig(max_trials=5000, min_trials=500))

        @test result.converged
        H = result.value
        test_pts = [SA[400.0, 300.0], SA[600.0, 500.0], SA[250.0, 450.0]]
        @test max_reprojection_error(H, H_true, test_pts) < 3.0
    end

    @testset "IRLS — weighted_system + model_from_solution roundtrip" begin
        H_true = @SMatrix [
             0.95  -0.10  15.0;
             0.12   0.93  -8.0;
             1e-4   2e-4   1.0
        ]
        H_true = H_true / norm(H_true)
        if H_true[3,3] < 0; H_true = -H_true; end

        rng = MersenneTwister(42)
        n = 20
        src = [SA[100.0 + 800.0*rand(rng), 100.0 + 600.0*rand(rng)] for _ in 1:n]
        dst = [apply_homography(H_true, s) for s in src]

        p = HomographyProblem(csponds(src, dst))
        mask = trues(n)
        w = ones(n)

        sys = weighted_system(p, H_true, mask, w)
        @test !isnothing(sys)
        @test :A in propertynames(sys)
        @test :T₁ in propertynames(sys)
        @test :T₂ in propertynames(sys)

        # Solve via SVD and reconstruct
        h = svd(sys.A).Vt[end, :]
        H_rec = model_from_solution(p, h, sys)
        @test !isnothing(H_rec)

        test_pts = [SA[300.0, 400.0], SA[600.0, 300.0], SA[450.0, 550.0]]
        @test max_reprojection_error(H_rec, H_true, test_pts) < 1e-8
    end

    @testset "IRLS — L2Loss recovers clean homography" begin
        H_true = @SMatrix [
             0.95  -0.10  15.0;
             0.12   0.93  -8.0;
             1e-4   2e-4   1.0
        ]
        H_true = H_true / norm(H_true)
        if H_true[3,3] < 0; H_true = -H_true; end

        rng = MersenneTwister(42)
        n = 30
        noise = 0.3
        src = [SA[100.0 + 800.0*rand(rng), 100.0 + 600.0*rand(rng)] for _ in 1:n]
        dst = [apply_homography(H_true, s) + SA[noise*randn(rng), noise*randn(rng)] for s in src]

        p = HomographyProblem(csponds(src, dst))
        mask = trues(n)

        # L2 IRLS (all weights = 1.0) should recover H well
        adapter = RansacRefineProblem(p, mask, p._svd_ws)
        result = robust_solve(adapter, MEstimator(L2Loss());
                              init=H_true, scale=FixedScale(σ=1.0), max_iter=5)
        H_irls = result.value

        test_pts = [SA[300.0, 400.0], SA[600.0, 300.0], SA[450.0, 550.0]]
        @test max_reprojection_error(H_irls, H_true, test_pts) < 1.0
    end

    @testset "IRLS — Cauchy refine on noisy inliers" begin
        source_pts, target_pts, H_true, n_inliers = make_homography_data(
            n_inliers=100, n_outliers=30, noise=1.0, seed=42)

        problem = HomographyProblem(csponds(source_pts, target_pts))
        # Mask with known inliers only
        mask = falses(length(source_pts))
        mask[1:n_inliers] .= true

        # IRLS with Cauchy should recover H well despite noisy inliers
        adapter = RansacRefineProblem(problem, mask, problem._svd_ws)
        result = robust_solve(adapter, MEstimator(CauchyLoss());
                              init=H_true, scale=FixedScale(σ=1.0), max_iter=5)
        H_irls = result.value

        test_pts = [SA[300.0, 400.0], SA[600.0, 300.0], SA[450.0, 550.0]]
        @test max_reprojection_error(H_irls, H_true, test_pts) < 3.0
    end

    @testset "IRLS — full RANSAC integration" begin
        source_pts, target_pts, H_true, n_inliers = make_homography_data(
            n_inliers=100, n_outliers=30, noise=0.5, seed=42)
        problem = HomographyProblem(csponds(source_pts, target_pts))

        result = ransac(problem, MarginalQuality(problem, 50.0);
                        config=RansacConfig(max_trials=3000, min_trials=200))

        @test result.converged
        H = result.value
        test_pts = [SA[300.0, 400.0], SA[600.0, 300.0], SA[450.0, 550.0],
                    SA[200.0, 200.0], SA[700.0, 500.0]]
        @test max_reprojection_error(H, H_true, test_pts) < 2.0
        @test sum(result.inlier_mask) >= n_inliers * 0.7
    end

    @testset "Estimate integration" begin
        source_pts, target_pts, _, _ = make_homography_data()
        problem = HomographyProblem(csponds(source_pts, target_pts))

        result = ransac(problem, MarginalQuality(problem, 50.0);
                        config=RansacConfig(max_trials=2000))

        @test result isa RansacEstimate
        @test result.value isa SMatrix{3,3,Float64,9}
        @test result.stop_reason === :converged
        @test result.inlier_mask isa BitVector
        @test result.residuals isa Vector{Float64}
        @test result.weights isa Vector{Float64}
        @test result.quality isa Float64
        @test result.scale isa Float64
        @test result.trials > 0
    end

    # =================================================================
    # PROSAC Tests
    # =================================================================

    @testset "ProsacSampler — pool expansion" begin
        # 20 data points, sample size 4
        sorted = collect(1:20)
        ps = ProsacSampler(sorted, 4; T_N=1000)
        @test ps.n == 4   # starts at m
        @test ps.t == 0

        # After many iterations, pool should expand toward N
        idx = Vector{Int}(undef, 4)
        for _ in 1:500
            RobustVisualGeometry._draw_prosac!(idx, ps)
        end
        @test ps.n > 4    # pool has expanded
        @test ps.t == 500

        # After T_N iterations, should have fully expanded
        reset!(ps)
        for _ in 1:1000
            RobustVisualGeometry._draw_prosac!(idx, ps)
        end
        @test ps.n == 20  # fully expanded to N
    end

    @testset "ProsacSampler — early samples from top" begin
        # Verify that early PROSAC samples come from high-scored indices
        sorted = collect(1:100)  # index 1 = best, 100 = worst
        ps = ProsacSampler(sorted, 4; T_N=200_000)
        idx = Vector{Int}(undef, 4)

        # First 50 samples should all use indices from the top of the pool
        max_idx_seen = 0
        for _ in 1:50
            RobustVisualGeometry._draw_prosac!(idx, ps)
            max_idx_seen = max(max_idx_seen, maximum(idx))
        end
        # Pool should still be small (well below N=100)
        @test ps.n < 20
        # All sampled indices should be from the top of the sorted list
        @test max_idx_seen <= ps.n
    end

    @testset "ProsacSampler — reset!" begin
        sorted = collect(1:50)
        ps = ProsacSampler(sorted, 4; T_N=500)
        idx = Vector{Int}(undef, 4)

        for _ in 1:200
            RobustVisualGeometry._draw_prosac!(idx, ps)
        end
        @test ps.t == 200
        @test ps.n > 4

        reset!(ps)
        @test ps.t == 0
        @test ps.n == 4
    end

    @testset "RANSAC — PROSAC with scored correspondences" begin
        source_pts, target_pts, H_true, n_inliers = make_homography_data(
            n_inliers=100, n_outliers=30, noise=0.5, seed=42)

        # Assign scores: inliers get high scores, outliers get low scores
        rng = MersenneTwister(99)
        n_total = length(source_pts)
        scored = Vector{ScoredCspond{SVector{2,Float64}, SVector{2,Float64}}}(undef, n_total)
        for i in 1:n_total
            score = i <= n_inliers ? 0.8 + 0.2*rand(rng) : 0.1*rand(rng)
            scored[i] = ScoredCspond(source_pts[i], target_pts[i], score)
        end

        problem = HomographyProblem(scored)
        @test problem isa HomographyProblem{Float64, ProsacSampler}

        result = ransac(problem, MarginalQuality(problem, 50.0);
                        config=RansacConfig(max_trials=3000, min_trials=200))

        @test result.converged
        H = result.value
        test_pts = [SA[300.0, 400.0], SA[600.0, 300.0], SA[450.0, 550.0],
                    SA[200.0, 200.0], SA[700.0, 500.0]]
        @test max_reprojection_error(H, H_true, test_pts) < 2.0
        @test sum(result.inlier_mask) >= n_inliers * 0.7
    end

end
