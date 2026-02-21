using Test
using LinearAlgebra
using StaticArrays
using Random
using ForwardDiff
using Distributions: FDist, quantile
using VisualGeometryCore
using RobustVisualGeometry
using RobustVisualGeometry: MarginalQuality, FTestLocalOptimization,
    PredictiveFTest, RansacConfig, UncertainRansacAttributes, ransac

# =============================================================================
# ForwardDiff reference: inhomogeneous line solver
# =============================================================================

"""Autodiff reference: solver (x₁,y₁,x₂,y₂) → (a,b) where y = a + bx."""
function _inhom_line_solver_ad(x::AbstractVector{T}) where T
    x1, y1, x2, y2 = x[1], x[2], x[3], x[4]
    dx = x2 - x1
    b = (y2 - y1) / dx
    a = y1 - b * x1
    SVector{2,T}(a, b)
end

# =============================================================================
# Tests
# =============================================================================

@testset "Inhomogeneous Line Fitting" begin

    @testset "solver_jacobian — ForwardDiff verification (y-only)" begin
        rng = MersenneTwister(42)

        for trial in 1:10
            # Random two points
            p1 = SA[randn(rng), randn(rng)]
            p2 = SA[randn(rng), randn(rng)]
            while abs(p2[1] - p1[1]) < 0.1
                p2 = SA[randn(rng), randn(rng)]
            end

            pts = [p1, p2, SA[0.0, 0.0]]  # 3rd point unused
            prob = InhomLineFittingProblem(pts)
            idx = [1, 2]

            model = solve(prob, idx)
            @test !isnothing(model)

            result = solver_jacobian(prob, idx, model)
            @test !isnothing(result)
            J_y = result.J
            @test J_y isa SMatrix{2,2,Float64}

            # ForwardDiff reference: full 2×4, then extract y-columns [2, 4]
            x0 = [p1[1], p1[2], p2[1], p2[2]]
            J_ad_full = ForwardDiff.jacobian(_inhom_line_solver_ad, x0)
            J_ad_y = J_ad_full[:, [2, 4]]  # y-only columns

            @test maximum(abs.(Matrix(J_y) - J_ad_y)) < 1e-10
        end
    end

    @testset "prediction_fstats — manual verification" begin
        # y = 2 + 3x, points on the line + noise
        a_true, b_true = 2.0, 3.0
        rng = MersenneTwister(123)

        # Two sample points (exact)
        p1 = SA[1.0, a_true + b_true * 1.0]
        p2 = SA[4.0, a_true + b_true * 4.0]

        # Test point: on line (should have small F-stat)
        p_on = SA[2.5, a_true + b_true * 2.5]
        # Test point: off line (should have large F-stat)
        p_off = SA[2.5, a_true + b_true * 2.5 + 10.0]

        pts = [p1, p2, p_on, p_off]
        prob = InhomLineFittingProblem(pts)
        idx = [1, 2]

        model = solve(prob, idx)
        jac_info = solver_jacobian(prob, idx, model)

        s2 = 1.0  # noise variance
        fstats = zeros(4)
        prediction_fstats_from_cov!(fstats, prob, model, jac_info, s2)

        # Points on line should have F-stat ≈ 0
        @test fstats[1] < 1e-10  # sample point 1
        @test fstats[2] < 1e-10  # sample point 2
        @test fstats[3] < 1e-10  # on-line test point

        # Point far off line should have large F-stat
        @test fstats[4] > 10.0
    end

    @testset "prediction_fstats — leverage effect" begin
        # y = 0 + 1·x, sample at x=0, x=1
        p1 = SA[0.0, 0.0]
        p2 = SA[1.0, 1.0]

        # Test at x=0.5 (low leverage) and x=100 (high leverage)
        p_low = SA[0.5, 0.5 + 0.1]   # small residual, low leverage
        p_high = SA[100.0, 100.0 + 0.1]  # same residual, high leverage

        pts = [p1, p2, p_low, p_high]
        prob = InhomLineFittingProblem(pts)
        idx = [1, 2]
        model = solve(prob, idx)
        jac_info = solver_jacobian(prob, idx, model)

        s2 = 0.01  # small noise
        fstats = zeros(4)
        prediction_fstats_from_cov!(fstats, prob, model, jac_info, s2)

        # High-leverage point should have SMALLER F-stat than low-leverage
        # because V_i is larger (more uncertainty from the model at x=100)
        @test fstats[4] < fstats[3]  # high leverage → larger V → smaller F
    end

    @testset "MarginalQuality + FTestLocalOptimization — end-to-end" begin
        rng = MersenneTwister(999)

        # True line: y = 5 + 2x
        a_true, b_true = 5.0, 2.0
        n_inliers = 50
        n_outliers = 20
        noise = 0.5

        # Inliers
        x_in = range(-10, 10, length=n_inliers)
        pts_in = [SA[x, a_true + b_true * x + noise * randn(rng)] for x in x_in]

        # Outliers
        pts_out = [SA[20.0 * rand(rng) - 10.0, 50.0 * rand(rng) - 25.0] for _ in 1:n_outliers]

        pts = vcat(pts_in, pts_out)
        n = length(pts)

        prob = InhomLineFittingProblem(pts)

        scoring = MarginalQuality(n, 2, 50.0; codimension=1)
        local_optimization = FTestLocalOptimization(PredictiveFTest(), 0.01, 5)
        config = RansacConfig(; max_trials=500, confidence=0.999)

        result = ransac(prob, scoring; local_optimization, config)

        # Check model accuracy
        model = result.value
        @test abs(model[1] - a_true) < 1.0  # intercept
        @test abs(model[2] - b_true) < 0.5  # slope

        # Check we got an uncertain result with param_cov
        attrs = result.attributes
        @test attrs isa UncertainRansacAttributes

        # Inlier ratio should be reasonable
        inlier_ratio = sum(attrs.inlier_mask) / n
        @test inlier_ratio > 0.5
        @test inlier_ratio < 0.95  # shouldn't include all outliers

        # Most inliers should be in the first n_inliers points
        n_correct_inliers = sum(attrs.inlier_mask[1:n_inliers])
        @test n_correct_inliers >= 0.8 * n_inliers

        # Few outliers should be classified as inliers
        n_false_inliers = sum(attrs.inlier_mask[n_inliers+1:end])
        @test n_false_inliers <= 10

        # param_cov should be PSD
        Σ = attrs.param_cov
        @test size(Σ) == (2, 2)
        @test all(eigvals(Symmetric(Matrix(Σ))) .>= -1e-10)

        # Scale estimate should be reasonable
        @test attrs.scale > 0
        @test attrs.scale < 2.0 * noise
    end

    @testset "F-test mask vs raw k* mask — different inlier counts" begin
        rng = MersenneTwister(777)

        # Line y = 0 + 1x, sample at x=0 and x=0.1 (close together → ill-conditioned)
        # This should inflate prediction variances for points far from sample
        a_true, b_true = 0.0, 1.0
        noise = 0.3

        pts = SVector{2,Float64}[]
        # Dense cluster near origin (well-supported)
        for _ in 1:30
            x = 0.5 * randn(rng)
            push!(pts, SA[x, a_true + b_true * x + noise * randn(rng)])
        end
        # Sparse points far away (high leverage)
        for _ in 1:10
            x = 20.0 + 5.0 * randn(rng)
            push!(pts, SA[x, a_true + b_true * x + noise * randn(rng)])
        end
        # Outliers
        for _ in 1:10
            push!(pts, SA[10.0 * randn(rng), 50.0 * randn(rng)])
        end

        n = length(pts)
        prob = InhomLineFittingProblem(pts)
        scoring = MarginalQuality(n, 2, 30.0; codimension=1)
        local_optimization = FTestLocalOptimization(PredictiveFTest(), 0.01, 5)
        config = RansacConfig(; max_trials=500, confidence=0.999)

        result = ransac(prob, scoring; local_optimization, config)
        @test result.attributes isa UncertainRansacAttributes

        # Verify the F-test mask is meaningful
        mask = result.attributes.inlier_mask
        @test sum(mask) >= 20  # at least the dense cluster
    end

end
