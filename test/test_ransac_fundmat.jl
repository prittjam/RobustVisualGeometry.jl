using Test
using LinearAlgebra
using StaticArrays
using Random
using Statistics: median
using VisualGeometryCore
using VisualGeometryCore: ScoringTrait, HasScore, NoScore, scoring
using RobustVisualGeometry
using RobustVisualGeometry: sample_size, data_size, model_type,
    solver_cardinality, MultipleSolutions,
    RansacRefineProblem, MEstimator, FixedScale, robust_solve
using VisualGeometryCore.Matching: Attributed, ScoredCspond, csponds

# =============================================================================
# Test Utilities
# =============================================================================

"""Skew-symmetric matrix from 3-vector."""
function skew(t::SVector{3,T}) where T
    @SMatrix [zero(T) -t[3] t[2];
              t[3] zero(T) -t[1];
             -t[2] t[1] zero(T)]
end

"""Compute fundamental matrix from camera matrices P₁, P₂."""
function cameras_to_fundmat(K₁, R₁, t₁, K₂, R₂, t₂)
    # F = K₂⁻ᵀ [t]× R K₁⁻¹, where R,t is relative pose
    R_rel = R₂ * R₁'
    t_rel = t₂ - R_rel * t₁
    E = skew(t_rel) * R_rel
    F = inv(K₂)' * E * inv(K₁)
    F = F / norm(F)
    if F[3,3] < 0; F = -F; end
    return SMatrix{3,3,Float64,9}(F)
end

"""Project 3D point to 2D using camera (K, R, t)."""
function project_point(K, R, t, X::SVector{3,Float64})
    p = K * (R * X + t)
    return SA[p[1]/p[3], p[2]/p[3]]
end

"""Check epipolar constraint: u₂ᵀ F u₁ ≈ 0."""
function epipolar_error(u₁, u₂, F)
    s = SA[u₁[1], u₁[2], 1.0]
    d = SA[u₂[1], u₂[2], 1.0]
    return abs(d' * F * s)
end

"""Generate synthetic stereo scene data."""
function make_fundmat_data(; n_inliers=100, n_outliers=30, noise=0.5, seed=42)
    rng = MersenneTwister(seed)

    # Camera intrinsics
    K₁ = @SMatrix [500.0 0.0 320.0; 0.0 500.0 240.0; 0.0 0.0 1.0]
    K₂ = @SMatrix [600.0 0.0 300.0; 0.0 600.0 250.0; 0.0 0.0 1.0]

    # Camera 1: at origin
    R₁ = SMatrix{3,3,Float64,9}(I)
    t₁ = SA[0.0, 0.0, 0.0]

    # Camera 2: rotated + translated
    θ = 0.15  # ~8.6°
    R₂ = @SMatrix [cos(θ) 0.0 sin(θ); 0.0 1.0 0.0; -sin(θ) 0.0 cos(θ)]
    t₂ = SA[1.0, 0.2, 0.1]

    F_true = cameras_to_fundmat(K₁, R₁, t₁, K₂, R₂, t₂)

    # Generate 3D points in front of both cameras
    source_pts = SVector{2, Float64}[]
    target_pts = SVector{2, Float64}[]
    for _ in 1:n_inliers
        X = SA[4.0*(rand(rng)-0.5), 3.0*(rand(rng)-0.5), 5.0 + 3.0*rand(rng)]
        s = project_point(K₁, R₁, t₁, X)
        d = project_point(K₂, R₂, t₂, X)
        s = s + SA[noise * randn(rng), noise * randn(rng)]
        d = d + SA[noise * randn(rng), noise * randn(rng)]
        push!(source_pts, s)
        push!(target_pts, d)
    end

    # Outliers: random point pairs
    for _ in 1:n_outliers
        s = SA[100.0 + 440.0 * rand(rng), 50.0 + 380.0 * rand(rng)]
        d = SA[80.0 + 440.0 * rand(rng), 60.0 + 380.0 * rand(rng)]
        push!(source_pts, s)
        push!(target_pts, d)
    end

    return source_pts, target_pts, F_true, n_inliers, K₁, K₂, R₁, t₁, R₂, t₂
end

"""Generate exact (noiseless) correspondences from a known F."""
function make_exact_fundmat_data(; n=7, seed=42)
    rng = MersenneTwister(seed)

    K₁ = @SMatrix [500.0 0.0 320.0; 0.0 500.0 240.0; 0.0 0.0 1.0]
    K₂ = @SMatrix [600.0 0.0 300.0; 0.0 600.0 250.0; 0.0 0.0 1.0]

    R₁ = SMatrix{3,3,Float64,9}(I)
    t₁ = SA[0.0, 0.0, 0.0]

    θ = 0.15
    R₂ = @SMatrix [cos(θ) 0.0 sin(θ); 0.0 1.0 0.0; -sin(θ) 0.0 cos(θ)]
    t₂ = SA[1.0, 0.2, 0.1]

    F_true = cameras_to_fundmat(K₁, R₁, t₁, K₂, R₂, t₂)

    source_pts = SVector{2, Float64}[]
    target_pts = SVector{2, Float64}[]
    for _ in 1:n
        X = SA[4.0*(rand(rng)-0.5), 3.0*(rand(rng)-0.5), 5.0 + 3.0*rand(rng)]
        s = project_point(K₁, R₁, t₁, X)
        d = project_point(K₂, R₂, t₂, X)
        push!(source_pts, s)
        push!(target_pts, d)
    end

    return source_pts, target_pts, F_true
end

# =============================================================================
# Tests
# =============================================================================

@testset "RANSAC Fundamental Matrix" begin

    @testset "FundamentalMatrixProblem construction" begin
        src, dst, _ = make_exact_fundmat_data(n=10)
        p = FundamentalMatrixProblem(csponds(src, dst))
        @test sample_size(p) == 7
        @test data_size(p) == 10
        @test model_type(p) == SMatrix{3,3,Float64,9}
        @test solver_cardinality(p) isa MultipleSolutions

        # Too few correspondences
        @test_throws ArgumentError FundamentalMatrixProblem(csponds(src[1:6], dst[1:6]))

        # UniformSampler for Pair correspondences
        @test p isa FundamentalMatrixProblem{Float64, UniformSampler}
    end

    @testset "Cubic solver" begin
        # Known roots: (x-1)(x-2)(x-3) = x³ - 6x² + 11x - 6
        roots = VisualGeometryCore._real_cubic_roots(1.0, -6.0, 11.0, -6.0)
        sorted = sort(filter(!isnan, collect(roots)))
        @test length(sorted) == 3
        @test sorted ≈ [1.0, 2.0, 3.0] atol=1e-10

        # Single real root: (x-1)(x²+1) = x³ - x² + x - 1
        roots = VisualGeometryCore._real_cubic_roots(1.0, -1.0, 1.0, -1.0)
        real_roots = filter(!isnan, collect(roots))
        @test length(real_roots) >= 1
        @test any(r -> abs(r - 1.0) < 1e-10, real_roots)

        # Triple root: (x-2)³ = x³ - 6x² + 12x - 8
        roots = VisualGeometryCore._real_cubic_roots(1.0, -6.0, 12.0, -8.0)
        real_roots = filter(!isnan, collect(roots))
        @test all(r -> abs(r - 2.0) < 1e-6, real_roots)
    end

    @testset "Rank-2 enforcement" begin
        # Start with a full-rank 3×3 matrix
        A = @SMatrix [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 10.0]
        @test rank(A) == 3

        A_rank2 = enforce_rank_two(A)
        @test rank(A_rank2) == 2

        # Singular values: first two preserved, third zeroed
        s_orig = svdvals(A)
        s_new = svdvals(A_rank2)
        @test s_new[1] ≈ s_orig[1] atol=1e-10
        @test s_new[2] ≈ s_orig[2] atol=1e-10
        @test abs(s_new[3]) < 1e-10
    end

    @testset "7-point solver — exact data" begin
        src, dst, F_true = make_exact_fundmat_data(n=7)
        p = FundamentalMatrixProblem(csponds(src, dst))

        solutions = solve(p, collect(1:7))
        @test !isnothing(solutions)
        @test length(solutions) >= 1

        # At least one solution should satisfy epipolar constraint
        best_err = Inf
        for F in solutions
            max_err = maximum(epipolar_error(s, d, F) for (s, d) in zip(src, dst))
            best_err = min(best_err, max_err)
        end
        @test best_err < 1e-8
    end

    @testset "Sampson distance" begin
        src, dst, F_true = make_exact_fundmat_data(n=10)
        p = FundamentalMatrixProblem(csponds(src, dst))

        r = Vector{Float64}(undef, 10)
        residuals!(r, p, F_true)

        # Exact correspondences should have near-zero Sampson distance
        @test all(r .< 1e-6)
    end

    @testset "Sampson distance — with noise" begin
        src, dst, F_true, _, _, _, _, _, _, _ = make_fundmat_data(
            n_inliers=50, n_outliers=0, noise=1.0, seed=42)
        p = FundamentalMatrixProblem(csponds(src, dst))

        r = Vector{Float64}(undef, 50)
        residuals!(r, p, F_true)

        # With noise ~1px, Sampson distance should be moderate
        @test median(r) < 5.0
        @test maximum(r) < 20.0
    end

    @testset "N-point DLT solver" begin
        src, dst, F_true = make_exact_fundmat_data(n=20)
        p = FundamentalMatrixProblem(csponds(src, dst); refinement=DltRefinement())
        mask = trues(20)

        result = refine(p, F_true, mask)
        @test !isnothing(result)
        F_ref, σ = result

        # Should satisfy epipolar constraint on exact data
        max_err = maximum(epipolar_error(s, d, F_ref) for (s, d) in zip(src, dst))
        @test max_err < 1e-8

        # F should be rank 2
        @test svdvals(F_ref)[3] < 1e-10 * svdvals(F_ref)[1]
    end

    @testset "Degeneracy handling — collinear data" begin
        # test_sample is intentionally always-true for F-matrix (collinearity
        # checks on C(7,3)=35 triplets reject ~75% of valid samples).
        # Collinear data is a soft degeneracy — the solver may still return
        # solutions, but RANSAC scoring will discard them.
        src_c = [SA[0.0+i*10.0, 100.0] for i in 0:6]
        dst_ok = [SA[10.0+i*30.0, 20.0+i*15.0] for i in 0:6]
        p = FundamentalMatrixProblem(csponds(src_c, dst_ok))
        @test test_sample(p, collect(1:7))  # always true by design

        # Non-degenerate points should produce a valid solution
        src_ok, dst_ok2, _ = make_exact_fundmat_data(n=7)
        p2 = FundamentalMatrixProblem(csponds(src_ok, dst_ok2))
        @test test_sample(p2, collect(1:7))
        @test !isnothing(solve(p2, collect(1:7)))
    end

    @testset "RANSAC — Cauchy + FixedScale" begin
        source_pts, target_pts, F_true, n_inliers, = make_fundmat_data(
            n_inliers=100, n_outliers=30, noise=0.5, seed=42)
        problem = FundamentalMatrixProblem(csponds(source_pts, target_pts))

        result = ransac(problem, MarginalQuality(data_size(problem), sample_size(problem), 50.0);
                        config=RansacConfig(max_trials=5000, min_trials=500))

        @test result.converged

        F = result.value
        # F should be rank 2
        sv = svdvals(F)
        @test sv[3] / sv[1] < 0.01

        # Check inlier ratio
        @test inlier_ratio(result) > 0.5
        @test sum(result.inlier_mask) >= n_inliers * 0.6

        # Sampson distance on inlier points should be small (pixels)
        r_inliers = Vector{Float64}(undef, n_inliers)
        for i in 1:n_inliers
            r_inliers[i] = sampson_distance(source_pts[i], target_pts[i], F)
        end
        @test median(r_inliers) < 3.0
    end

    @testset "RANSAC — L2 + FixedScale" begin
        source_pts, target_pts, F_true, _ = make_fundmat_data(
            n_inliers=100, n_outliers=30, noise=0.3, seed=123)
        problem = FundamentalMatrixProblem(csponds(source_pts, target_pts))

        result = ransac(problem, MarginalQuality(data_size(problem), sample_size(problem), 50.0);
                        config=RansacConfig(max_trials=5000))

        @test result.converged
        F = result.value
        sv = svdvals(F)
        @test sv[3] / sv[1] < 0.01
    end

    @testset "RANSAC — high outlier rate" begin
        source_pts, target_pts, F_true, _ = make_fundmat_data(
            n_inliers=50, n_outliers=100, noise=0.5, seed=456)
        problem = FundamentalMatrixProblem(csponds(source_pts, target_pts))

        result = ransac(problem, MarginalQuality(data_size(problem), sample_size(problem), 50.0);
                        config=RansacConfig(max_trials=10000, min_trials=1000))

        @test result.converged
        F = result.value
        sv = svdvals(F)
        @test sv[3] / sv[1] < 0.01
    end

    @testset "IRLS — weighted_system + model_from_solution roundtrip" begin
        src, dst, F_true = make_exact_fundmat_data(n=20)
        p = FundamentalMatrixProblem(csponds(src, dst))
        mask = trues(20)
        w = ones(20)

        sys = weighted_system(p, F_true, mask, w)
        @test !isnothing(sys)
        @test :A in propertynames(sys)
        @test :T₁ in propertynames(sys)
        @test :T₂ in propertynames(sys)

        # Solve via SVD and reconstruct
        f = svd(sys.A).Vt[end, :]
        F_rec = model_from_solution(p, f, sys)
        @test !isnothing(F_rec)

        # Should satisfy epipolar constraint
        max_err = maximum(epipolar_error(s, d, F_rec) for (s, d) in zip(src, dst))
        @test max_err < 1e-6
    end

    @testset "IRLS — L2Loss recovers clean F" begin
        src, dst, F_true = make_exact_fundmat_data(n=30)
        # Add small noise
        rng = MersenneTwister(42)
        noise = 0.3
        src_n = [s + SA[noise*randn(rng), noise*randn(rng)] for s in src]
        dst_n = [d + SA[noise*randn(rng), noise*randn(rng)] for d in dst]

        p = FundamentalMatrixProblem(csponds(src_n, dst_n))
        mask = trues(30)

        adapter = RansacRefineProblem(p, mask, p._svd_ws)
        result = robust_solve(adapter, MEstimator(L2Loss());
                              init=F_true, scale=FixedScale(σ=1.0), max_iter=5)
        F_irls = result.value

        # Sampson distance on clean (non-noisy) data should be reasonable
        # (IRLS fits to noisy data, so some deviation from clean ground truth is expected)
        errs = [sampson_distance(s, d, F_irls) for (s, d) in zip(src, dst)]
        @test median(errs) < 3.0
    end

    @testset "IRLS — full RANSAC integration" begin
        source_pts, target_pts, F_true, n_inliers = make_fundmat_data(
            n_inliers=100, n_outliers=30, noise=0.5, seed=42)
        problem = FundamentalMatrixProblem(csponds(source_pts, target_pts))

        result = ransac(problem, MarginalQuality(data_size(problem), sample_size(problem), 50.0);
                        config=RansacConfig(max_trials=5000, min_trials=500))

        @test result.converged
        F = result.value
        @test sum(result.inlier_mask) >= n_inliers * 0.6
    end

    @testset "Estimate integration" begin
        source_pts, target_pts, _, _ = make_fundmat_data()
        problem = FundamentalMatrixProblem(csponds(source_pts, target_pts))

        result = ransac(problem, MarginalQuality(data_size(problem), sample_size(problem), 50.0);
                        config=RansacConfig(max_trials=5000))

        @test result isa RansacEstimate
        @test result.value isa SMatrix{3,3,Float64,9}
        @test result.inlier_mask isa BitVector
        @test result.residuals isa Vector{Float64}
        @test result.weights isa Vector{Float64}
        @test result.quality isa Float64
        @test result.scale isa Float64
        @test result.trials > 0
    end

end
