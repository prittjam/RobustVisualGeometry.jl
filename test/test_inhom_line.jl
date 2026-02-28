using Test
using LinearAlgebra
using StaticArrays
using Random
using ForwardDiff
using VisualGeometryCore
using RobustVisualGeometry
using RobustVisualGeometry: MarginalScoring, PredictiveMarginalScoring, RansacConfig, ransac,
    score!, _sq_norm, RansacWorkspace,
    data_size, sample_size, model_type, residual_jacobian

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

    @testset "MarginalScoring — end-to-end" begin
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

        scoring = MarginalScoring(n, 2, 50.0; codimension=1)
        config = RansacConfig(; max_trials=500, confidence=0.999)

        result = ransac(prob, scoring; config)

        # Check model accuracy
        model = result.value
        @test abs(model[1] - a_true) < 1.0  # intercept
        @test abs(model[2] - b_true) < 0.5  # slope

        attrs = result.attributes
        @test attrs isa RansacAttributes

        # Inlier ratio should be reasonable
        ir = sum(attrs.inlier_mask) / n
        @test ir > 0.5
        @test ir < 0.95  # shouldn't include all outliers

        # Most inliers should be in the first n_inliers points
        n_correct_inliers = sum(attrs.inlier_mask[1:n_inliers])
        @test n_correct_inliers >= 0.8 * n_inliers

        # Few outliers should be classified as inliers
        n_false_inliers = sum(attrs.inlier_mask[n_inliers+1:end])
        @test n_false_inliers <= 10

        # Scale estimate should be reasonable
        @test attrs.scale > 0
        @test attrs.scale < 2.0 * noise
    end

    @testset "Problem-aware constructors" begin
        pts = [SA[Float64(i), 2.0 + 3.0 * i] for i in 1:50]
        prob = InhomLineFittingProblem(pts)

        # MarginalScoring(problem, a)
        mq = MarginalScoring(prob, 50.0)
        mq_manual = MarginalScoring(50, 2, 50.0; codimension=1)
        @test mq.log2a == mq_manual.log2a
        @test mq.model_dof == mq_manual.model_dof
        @test mq.codimension == mq_manual.codimension
        @test length(mq.perm) == length(mq_manual.perm)
        @test length(mq.lg_table) == length(mq_manual.lg_table)

        # PredictiveMarginalScoring(problem, a)
        pmq = PredictiveMarginalScoring(prob, 30.0)
        pmq_manual = PredictiveMarginalScoring(50, 2, 30.0; codimension=1)
        @test pmq.log2a == pmq_manual.log2a
        @test pmq.model_dof == pmq_manual.model_dof
        @test pmq.codimension == pmq_manual.codimension
        @test length(pmq.perm) == length(pmq_manual.perm)
        @test length(pmq.lg_table) == length(pmq_manual.lg_table)

        # Homography problem has different codimension
        cs = [SA[1.0, 2.0] => SA[3.0, 4.0],
              SA[5.0, 6.0] => SA[7.0, 8.0],
              SA[9.0, 10.0] => SA[11.0, 12.0],
              SA[13.0, 14.0] => SA[15.0, 16.0],
              SA[17.0, 18.0] => SA[19.0, 20.0]]
        hprob = HomographyProblem(cs)
        hmq = MarginalScoring(hprob, 50.0)
        @test hmq.model_dof == 4  # sample_size
        @test hmq.codimension == 2
        @test length(hmq.perm) == 5

        hpmq = PredictiveMarginalScoring(hprob, 50.0)
        @test hpmq.model_dof == 4
        @test hpmq.codimension == 2
    end

    @testset "PredictiveMarginalScoring — end-to-end" begin
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
        scoring = PredictiveMarginalScoring(n, 2, 30.0; codimension=1)
        config = RansacConfig(; max_trials=500, confidence=0.999)

        result = ransac(prob, scoring; config)
        @test result.attributes isa RansacAttributes

        # Verify the mask is meaningful
        mask = result.attributes.inlier_mask
        @test sum(mask) >= 20  # at least the dense cluster
    end

    @testset "residual_jacobian — 3-tuple return verification" begin
        pts = [SA[1.0, 5.0], SA[3.0, -1.0], SA[0.5, 3.5]]
        prob = InhomLineFittingProblem(pts)
        model = SA[7.0, -2.0]  # y = 7 - 2x

        for i in 1:3
            rᵢ, ∂θgᵢ_w, ℓᵢ = residual_jacobian(prob, model, i)

            # rᵢ should be the scalar residual yᵢ - a - b·xᵢ
            expected_r = pts[i][2] - model[1] - model[2] * pts[i][1]
            @test rᵢ ≈ expected_r atol=1e-14

            # ∂θgᵢ_w should be [-1, -xᵢ]
            @test ∂θgᵢ_w ≈ SA[-1.0, -pts[i][1]] atol=1e-14

            # ℓᵢ should be 0 (homoscedastic)
            @test ℓᵢ == 0.0

            # qᵢ = rᵢ² (scalar case)
            @test _sq_norm(rᵢ) ≈ rᵢ^2 atol=1e-14
        end
    end

end
