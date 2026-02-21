using Test
using LinearAlgebra
using StaticArrays
using Random
using VisualGeometryCore: Attributed
using RobustVisualGeometry
import RobustVisualGeometry: sample_size, codimension, data_size, model_type,
    solver_cardinality, solve, residuals!, test_type,
    prediction_variances!, prediction_fstats_from_inliers!
using RobustVisualGeometry: RansacWorkspace, SingleSolution, draw_sample!, FTestLocalOptimization

# =============================================================================
# Test Problem: 2D Line Fitting
# =============================================================================
#
# Model: line ax + by + c = 0, normalized so a² + b² = 1
# Represented as SVector{3, Float64}
#
# This is a SingleSolution solver (2 points → 1 line)

struct LineFittingProblem <: AbstractRansacProblem
    points::Vector{SVector{2, Float64}}
end

sample_size(::LineFittingProblem) = 2
codimension(::LineFittingProblem) = 1
data_size(p::LineFittingProblem) = length(p.points)
model_type(::LineFittingProblem) = SVector{3, Float64}
solver_cardinality(::LineFittingProblem) = SingleSolution()

function solve(p::LineFittingProblem, idx)
    p1 = p.points[idx[1]]
    p2 = p.points[idx[2]]
    d = p2 - p1
    # Normal: (-dy, dx), offset: normal ⋅ p1
    n = SA[-d[2], d[1]]
    len = norm(n)
    len < 1e-12 && return nothing  # degenerate (identical points)
    n = n / len
    c = -dot(n, p1)
    return SA[n[1], n[2], c]
end

function residuals!(r::Vector, p::LineFittingProblem, model::SVector{3, Float64})
    a, b, c = model
    @inbounds for i in eachindex(r)
        pt = p.points[i]
        r[i] = a * pt[1] + b * pt[2] + c
    end
    return r
end

# =============================================================================
# Test Problem: 2D Line Fitting with Predictive F-test
# =============================================================================
#
# Wraps LineFittingProblem, adds prediction_variances! for PredictiveFTest.
# Jacobian of residual rᵢ = a·xᵢ + b·yᵢ + c w.r.t. [a,b,c] is gᵢ = [xᵢ, yᵢ, 1].

struct PredictiveLineFittingProblem <: AbstractRansacProblem
    points::Vector{SVector{2, Float64}}
end

sample_size(::PredictiveLineFittingProblem) = 2
codimension(::PredictiveLineFittingProblem) = 1
data_size(p::PredictiveLineFittingProblem) = length(p.points)
model_type(::PredictiveLineFittingProblem) = SVector{3, Float64}
solver_cardinality(::PredictiveLineFittingProblem) = SingleSolution()
test_type(::PredictiveLineFittingProblem) = PredictiveFTest()

function solve(p::PredictiveLineFittingProblem, idx)
    p1 = p.points[idx[1]]
    p2 = p.points[idx[2]]
    d = p2 - p1
    n = SA[-d[2], d[1]]
    len = norm(n)
    len < 1e-12 && return nothing
    n = n / len
    c = -dot(n, p1)
    return SA[n[1], n[2], c]
end

function residuals!(r::Vector, p::PredictiveLineFittingProblem, model::SVector{3, Float64})
    a, b, c = model
    @inbounds for i in eachindex(r)
        pt = p.points[i]
        r[i] = a * pt[1] + b * pt[2] + c
    end
    return r
end

function prediction_variances!(v::Vector{Float64},
        p::PredictiveLineFittingProblem, model::SVector{3,Float64},
        mask::BitVector, s2::Float64)
    n = length(p.points)
    n_params = 3  # [a, b, c]

    # Build J'J from inliers: J[i,:] = [xᵢ, yᵢ, 1]
    JtJ = zeros(n_params, n_params)
    @inbounds for i in 1:n
        if mask[i]
            pt = p.points[i]
            g = SA[pt[1], pt[2], 1.0]
            for col in 1:n_params, row in 1:n_params
                JtJ[row, col] += g[row] * g[col]
            end
        end
    end

    # Σ_θ = s² · (J'J)⁻¹
    F = lu(JtJ; check=false)
    issuccess(F) || return nothing
    Σ_θ = s2 * inv(F)

    # vᵢ = s² + gᵢ' Σ_θ gᵢ  (= s² · (1 + hᵢᵢ) where hᵢᵢ is leverage)
    @inbounds for i in 1:n
        pt = p.points[i]
        g = SA[pt[1], pt[2], 1.0]
        v[i] = s2 + dot(g, SMatrix{3,3,Float64}(Σ_θ) * g)
    end

    param_cov = SMatrix{3,3,Float64}(Σ_θ)
    return v, param_cov
end

function prediction_fstats_from_inliers!(fstats::Vector{Float64},
        p::PredictiveLineFittingProblem, model::SVector{3,Float64},
        mask::BitVector, s2::Float64)
    n = length(p.points)
    n_params = 3

    JtJ = zeros(n_params, n_params)
    @inbounds for i in 1:n
        if mask[i]
            pt = p.points[i]
            g = SA[pt[1], pt[2], 1.0]
            for col in 1:n_params, row in 1:n_params
                JtJ[row, col] += g[row] * g[col]
            end
        end
    end

    F = lu(JtJ; check=false)
    issuccess(F) || return nothing
    Σ_θ = s2 * inv(F)
    Σ_θ_s = SMatrix{3,3,Float64}(Σ_θ)

    @inbounds for i in 1:n
        pt = p.points[i]
        g = SA[pt[1], pt[2], 1.0]
        r = model[1] * pt[1] + model[2] * pt[2] + model[3]
        v_i = s2 + dot(g, Σ_θ_s * g)
        fstats[i] = v_i > 0.0 ? r^2 / v_i : Inf
    end

    return fstats, Σ_θ_s
end

# =============================================================================
# Test Data Generation
# =============================================================================

function make_line_data(; n_inliers=100, n_outliers=50, noise=0.1, seed=42)
    rng = MersenneTwister(seed)

    # True line: y = 2x + 1 → 2x - y + 1 = 0 → normalized
    a, b, c = 2.0, -1.0, 1.0
    len = sqrt(a^2 + b^2)
    a, b, c = a/len, b/len, c/len

    # Inliers: random x in [-5, 5], y = 2x + 1 + noise
    inliers = SVector{2, Float64}[]
    for _ in 1:n_inliers
        x = 10 * rand(rng) - 5
        y = 2x + 1 + noise * randn(rng)
        push!(inliers, SA[x, y])
    end

    # Outliers: random points in [-10, 20] × [-10, 20]
    outliers = SVector{2, Float64}[]
    for _ in 1:n_outliers
        x = 30 * rand(rng) - 10
        y = 30 * rand(rng) - 10
        push!(outliers, SA[x, y])
    end

    points = vcat(inliers, outliers)
    true_model = SA[a, b, c]
    return points, true_model, n_inliers
end

# =============================================================================
# Tests
# =============================================================================

@testset "RANSAC" begin

    @testset "Hypergeometric trial count" begin
        # _p_all_inliers: exact hypergeometric probability
        p = RobustVisualGeometry._p_all_inliers(50, 100, 2)
        # C(50,2)/C(100,2) = (50*49)/(100*99)
        @test p ≈ (50*49)/(100*99)

        p = RobustVisualGeometry._p_all_inliers(100, 100, 7)
        @test p ≈ 1.0

        p = RobustVisualGeometry._p_all_inliers(3, 100, 7)
        @test p == 0.0

        # Single draw: P(inlier) = K/N
        p = RobustVisualGeometry._p_all_inliers(30, 100, 1)
        @test p ≈ 0.3
    end

    @testset "RansacConfig defaults" begin
        config = RansacConfig()
        @test config.confidence == 0.99
        @test config.max_trials == 10_000
        @test config.min_trials == 100
    end

    @testset "RansacWorkspace construction" begin
        ws = RansacWorkspace(100, 7, SVector{3, Float64})
        @test length(ws.residuals) == 100
        @test length(ws.scores) == 100
        @test length(ws.mask) == 100
        @test length(ws.sample_indices) == 7
        @test ws.has_best == false
    end

    @testset "Line fitting — L2 + FixedScale" begin
        points, true_model, n_inliers = make_line_data(n_inliers=200, n_outliers=50)
        problem = LineFittingProblem(points)

        # L2Loss with FixedScale (standard MSAC): threshold on 0.5*r²
        # Inlier residuals ≈ N(0, 0.045), so 0.5*r² ≈ 0.001.
        # Threshold 0.05 corresponds to |r| < 0.316 ≈ 7σ of noise.
        # NOTE: L2Loss + MADScale is NOT a valid RANSAC scoring combination
        # because MAD normalizes each model to unit variance, erasing the signal.
        scoring = ThresholdQuality(L2Loss(), 0.05, FixedScale())
        result = ransac(problem, scoring;
                        config=RansacConfig(max_trials=2000, min_trials=100))

        @test result.converged
        @test result.converged

        # Check model recovery (sign may be flipped)
        model = result.value
        if dot(model, true_model) < 0
            model = -model
        end
        @test model[1] ≈ true_model[1] atol=0.1
        @test model[2] ≈ true_model[2] atol=0.1
        @test model[3] ≈ true_model[3] atol=0.2

        # Check inlier ratio is reasonable
        @test inlier_ratio(result) > 0.5
        @test sum(result.inlier_mask) >= n_inliers * 0.7
    end

    @testset "Line fitting — Cauchy + FixedScale" begin
        points, true_model, _ = make_line_data(n_inliers=200, n_outliers=100, seed=123)
        problem = LineFittingProblem(points)

        scoring = ThresholdQuality(CauchyLoss(), 3.0, FixedScale())
        result = ransac(problem, scoring;
                        config=RansacConfig(max_trials=2000))

        @test result.converged
        model = result.value
        if dot(model, true_model) < 0
            model = -model
        end
        @test model[1] ≈ true_model[1] atol=0.05
        @test model[2] ≈ true_model[2] atol=0.05
    end

    @testset "Line fitting — high outlier rate" begin
        points, true_model, _ = make_line_data(n_inliers=50, n_outliers=200, seed=456)
        problem = LineFittingProblem(points)

        # High outlier rate (80%): use FixedScale since MADScale breaks down
        # above 50% outliers. L2 + tight threshold gives reliable scoring.
        scoring = ThresholdQuality(L2Loss(), 0.05, FixedScale())
        result = ransac(problem, scoring;
                        config=RansacConfig(max_trials=5000, min_trials=200))

        @test result.converged
        model = result.value
        if dot(model, true_model) < 0
            model = -model
        end
        @test model[1] ≈ true_model[1] atol=0.15
        @test model[2] ≈ true_model[2] atol=0.15
    end

    @testset "Line fitting — with workspace reuse" begin
        points, _, _ = make_line_data(n_inliers=100, n_outliers=50)
        problem = LineFittingProblem(points)

        ws = RansacWorkspace(data_size(problem), sample_size(problem),
                             model_type(problem))

        # Run twice with same workspace
        scoring = ThresholdQuality(CauchyLoss(), 3.0, FixedScale())
        r1 = ransac(problem, scoring;
                     config=RansacConfig(max_trials=500), workspace=ws)
        @test r1.converged

        r2 = ransac(problem, scoring;
                     config=RansacConfig(max_trials=500), workspace=ws)
        @test r2.converged
    end

    @testset "Estimate integration" begin
        points, _, _ = make_line_data()
        problem = LineFittingProblem(points)

        scoring = ThresholdQuality(CauchyLoss(), 3.0, FixedScale())
        result = ransac(problem, scoring;
                        config=RansacConfig(max_trials=500))

        # Test Attributed property forwarding
        @test result isa Attributed
        @test result.value isa SVector{3, Float64}
        @test result.stop_reason === :converged
        @test result.converged == true
        @test result.converged == true
        @test result.inlier_mask isa BitVector
        @test result.residuals isa Vector{Float64}
        @test result.weights isa Vector{Float64}
        @test result.quality isa Float64
        @test result.scale isa Float64
        @test result.trials isa Int
        @test result.trials > 0
    end

    @testset "draw_sample! uniqueness" begin
        problem = LineFittingProblem(
            [SA[Float64(i), Float64(i)] for i in 1:20])
        indices = Vector{Int}(undef, 5)

        for _ in 1:100
            draw_sample!(indices, problem)
            @test length(unique(indices)) == 5
            @test all(1 .<= indices .<= 20)
        end
    end

    # =================================================================
    # Quality Strategy Types
    # =================================================================

    @testset "ThresholdQuality wrapper" begin
        # Verify existing behavior works through the new ThresholdQuality path
        points, true_model, n_inliers = make_line_data(n_inliers=200, n_outliers=50)
        problem = LineFittingProblem(points)

        scoring = ThresholdQuality(L2Loss(), 0.05, FixedScale())
        result = ransac(problem, scoring;
                        config=RansacConfig(max_trials=2000, min_trials=100))

        @test result.converged
        model = result.value
        if dot(model, true_model) < 0
            model = -model
        end
        @test model[1] ≈ true_model[1] atol=0.1
        @test model[2] ≈ true_model[2] atol=0.1
        @test inlier_ratio(result) > 0.5
    end

    # =================================================================
    # Marginal Quality
    # =================================================================

    @testset "MarginalQuality constructor" begin
        ms = MarginalQuality(100, 2, 50.0)
        @test ms.log2a ≈ log(100.0)
        @test ms.model_dof == 2
        @test length(ms.perm) == 100
        @test length(ms.lg_table) == 100
        # Verify loggamma table: lg[1] = log Γ(1/2) = log √π
        @test ms.lg_table[1] ≈ 0.5 * log(π) atol=1e-12
        # lg[2] = log Γ(1) = 0
        @test ms.lg_table[2] ≈ 0.0 atol=1e-12
        # lg[4] = log Γ(2) = log(1) = 0
        @test ms.lg_table[4] ≈ 0.0 atol=1e-12
        # lg[6] = log Γ(3) = log(2!) = log(2)
        @test ms.lg_table[6] ≈ log(2.0) atol=1e-12
        # Invalid halfwidth
        @test_throws ArgumentError MarginalQuality(10, 2, -1.0)
    end

    @testset "init_quality" begin
        @test init_quality(ThresholdQuality(L2Loss(), 1.0, FixedScale())) == -Inf
        @test init_quality(MarginalQuality(10, 2, 1.0)) == (-Inf, -Inf)
    end

    @testset "_marginal_sweep! known partition" begin
        # 10 inliers with small residuals, 5 outliers with large residuals
        n = 15
        p = 2
        a = 50.0
        scoring = MarginalQuality(n, p, a)

        # Squared residuals: first 10 are small (σ≈0.1), last 5 are large
        losses = Float64[0.01, 0.02, 0.005, 0.03, 0.015,
                         0.008, 0.025, 0.012, 0.018, 0.022,
                         100.0, 200.0, 150.0, 300.0, 250.0]

        best_S, best_k = RobustVisualGeometry._marginal_sweep!(scoring, losses, n)
        # Should identify k*=10 (the 10 small residuals)
        @test best_k == 10
        @test best_S > -Inf
    end

    @testset "MarginalQuality line fitting" begin
        points, true_model, n_inliers = make_line_data(n_inliers=200, n_outliers=50)
        problem = LineFittingProblem(points)

        scoring = MarginalQuality(data_size(problem), sample_size(problem), 50.0)
        result = ransac(problem, scoring;
                        config=RansacConfig(max_trials=2000, min_trials=100))

        @test result.converged
        model = result.value
        if dot(model, true_model) < 0
            model = -model
        end
        @test model[1] ≈ true_model[1] atol=0.1
        @test model[2] ≈ true_model[2] atol=0.1
        @test model[3] ≈ true_model[3] atol=0.2
        @test inlier_ratio(result) > 0.5
        @test sum(result.inlier_mask) >= n_inliers * 0.7
    end

    @testset "MarginalQuality high outlier rate" begin
        points, true_model, _ = make_line_data(n_inliers=50, n_outliers=200, seed=456)
        problem = LineFittingProblem(points)

        scoring = MarginalQuality(data_size(problem), sample_size(problem), 20.0)
        result = ransac(problem, scoring;
                        config=RansacConfig(max_trials=10_000, min_trials=500))

        @test result.converged
        model = result.value
        if dot(model, true_model) < 0
            model = -model
        end
        @test model[1] ≈ true_model[1] atol=0.2
        @test model[2] ≈ true_model[2] atol=0.2
    end

    @testset "MarginalQuality binary weights" begin
        points, _, _ = make_line_data(n_inliers=100, n_outliers=50)
        problem = LineFittingProblem(points)

        scoring = MarginalQuality(data_size(problem), sample_size(problem), 50.0)
        result = ransac(problem, scoring;
                        config=RansacConfig(max_trials=1000))

        @test result.converged
        # Weights should be exactly 0.0 or 1.0
        @test all(w -> w == 0.0 || w == 1.0, result.weights)
        # Weights should match inlier mask
        @test all(i -> result.weights[i] == (result.inlier_mask[i] ? 1.0 : 0.0),
                  eachindex(result.weights))
    end

    @testset "MarginalQuality direct API" begin
        points, true_model, n_inliers = make_line_data(n_inliers=200, n_outliers=50)
        problem = LineFittingProblem(points)

        scoring = MarginalQuality(data_size(problem), sample_size(problem), 50.0)
        result = ransac(problem, scoring;
                        config=RansacConfig(max_trials=2000, min_trials=100))

        @test result.converged
        model = result.value
        if dot(model, true_model) < 0
            model = -model
        end
        @test model[1] ≈ true_model[1] atol=0.1
        @test model[2] ≈ true_model[2] atol=0.1
        @test result.scale > 0
        @test !isnan(result.scale)
        # Binary weights from marginal scoring
        @test all(w -> w == 0.0 || w == 1.0, result.weights)
    end

    # =================================================================
    # Noise Posterior (dof field)
    # =================================================================

    @testset "dof field — ThresholdQuality" begin
        points, _, _ = make_line_data(n_inliers=100, n_outliers=50)
        problem = LineFittingProblem(points)

        scoring = ThresholdQuality(CauchyLoss(), 3.0, FixedScale())
        result = ransac(problem, scoring;
                        config=RansacConfig(max_trials=500))
        @test result.converged
        @test result.dof == 0  # threshold-based: no posterior on σ
    end

    @testset "dof field — MarginalQuality" begin
        points, _, n_inliers = make_line_data(n_inliers=200, n_outliers=50, noise=0.1)
        problem = LineFittingProblem(points)
        p = sample_size(problem)  # 2 for line fitting

        # Direct MarginalQuality: returns dof = n_inliers - p
        scoring = MarginalQuality(data_size(problem), p, 50.0)
        result = ransac(problem, scoring;
                        config=RansacConfig(max_trials=2000, min_trials=100))
        @test result.converged
        @test result.dof > 0
        @test result.dof == sum(result.inlier_mask) - p

        # scale is s = √(RSS/ν), the unbiased estimate
        n_in = sum(result.inlier_mask)
        RSS = sum(result.residuals[i]^2 for i in 1:length(result.residuals)
                  if result.inlier_mask[i])
        nu = n_in - p
        @test result.scale ≈ sqrt(RSS / nu)
    end

    @testset "dof field — MarginalQuality direct" begin
        points, _, _ = make_line_data(n_inliers=200, n_outliers=50, noise=0.1)
        problem = LineFittingProblem(points)
        p = sample_size(problem)

        scoring = MarginalQuality(data_size(problem), p, 50.0)
        result = ransac(problem, scoring;
                        config=RansacConfig(max_trials=2000, min_trials=100))
        @test result.converged
        @test result.dof > 0

        # Posterior interpretation: σ² | data ~ InvGamma(ν/2, ν·s²/2)
        ν = result.dof
        s = result.scale
        @test ν > 100  # ~200 inliers - 2 = ~198
        # Scale should be close to true noise σ=0.1 (divided by line norm ≈ √5)
        # True residual σ ≈ 0.1/√5 ≈ 0.045
        @test s > 0.02 && s < 0.2
    end

    # =================================================================
    # UncertainRansacAttributes
    # =================================================================

    @testset "UncertainRansacAttributes promotion" begin
        points, true_model, _ = make_line_data(n_inliers=200, n_outliers=50, noise=0.1)
        problem = LineFittingProblem(points)
        p = sample_size(problem)

        scoring = MarginalQuality(data_size(problem), p, 50.0)
        result = ransac(problem, scoring;
                        config=RansacConfig(max_trials=2000, min_trials=100))
        @test result.converged

        # Create a dummy 2×2 covariance (line has 2 DOF for testing)
        Σ = SMatrix{2,2,Float64}(0.01, 0.001, 0.001, 0.02)
        uattrs = UncertainRansacAttributes(result.attributes, Σ)

        # All base fields forwarded
        @test uattrs.stop_reason === :converged
        @test uattrs.converged == true
        @test uattrs.inlier_mask === result.inlier_mask
        @test uattrs.residuals === result.residuals
        @test uattrs.weights === result.weights
        @test uattrs.quality == result.quality
        @test uattrs.scale == result.scale
        @test uattrs.dof == result.dof
        @test uattrs.trials == result.trials
        @test uattrs.sample_acceptance_rate == result.sample_acceptance_rate
        # Covariance present
        @test uattrs.param_cov === Σ
    end

    @testset "UncertainRansacEstimate property forwarding" begin
        points, _, _ = make_line_data(n_inliers=200, n_outliers=50, noise=0.1)
        problem = LineFittingProblem(points)
        p = sample_size(problem)

        scoring = MarginalQuality(data_size(problem), p, 50.0)
        result = ransac(problem, scoring;
                        config=RansacConfig(max_trials=2000, min_trials=100))
        @test result.converged

        Σ = SMatrix{2,2,Float64}(0.01, 0.001, 0.001, 0.02)
        uattrs = UncertainRansacAttributes(result.attributes, Σ)
        uresult = Attributed(result.value, uattrs)

        # Type alias
        @test uresult isa UncertainRansacEstimate

        # Property forwarding through Attributed
        @test uresult.converged == true
        @test uresult.param_cov === Σ
        @test uresult.scale == result.scale
        @test uresult.dof == result.dof
        @test uresult.value === result.value

        # inlier_ratio works on uncertain variant
        @test inlier_ratio(uresult) == inlier_ratio(result)
    end

    # =================================================================
    # Predictive F-test
    # =================================================================

    @testset "test_type default is BasicFTest" begin
        points, _, _ = make_line_data()
        problem = LineFittingProblem(points)
        @test test_type(problem) isa BasicFTest
    end

    @testset "MarginalQuality + FTestLocalOptimization line fitting" begin
        # PredictiveLineFittingProblem implements prediction_fstats_from_inliers!
        # so _finalize produces UncertainRansacEstimate with param_cov.

        points, true_model, n_inliers = make_line_data(n_inliers=200, n_outliers=50, noise=0.1)
        problem = PredictiveLineFittingProblem(points)
        n = data_size(problem)
        p = sample_size(problem)

        scoring = MarginalQuality(n, p, 50.0)
        local_optimization = FTestLocalOptimization(PredictiveFTest(), 0.01, 5)
        result = ransac(problem, scoring; local_optimization,
                        config=RansacConfig(max_trials=2000, min_trials=100))

        @test result.converged
        # FTestLocalOptimization + PredictiveFTest returns UncertainRansacEstimate
        @test result isa UncertainRansacEstimate

        # param_cov is 3×3 (3 line parameters)
        @test result.param_cov isa SMatrix{3,3,Float64}
        # Covariance should be positive semidefinite
        eigvals_cov = eigvals(Symmetric(Matrix(result.param_cov)))
        @test all(λ -> λ >= -1e-12, eigvals_cov)

        # Model should still be accurate
        model = result.value
        if dot(model, true_model) < 0
            model = -model
        end
        @test model[1] ≈ true_model[1] atol=0.1
        @test model[2] ≈ true_model[2] atol=0.1

        # Scale and dof should be reasonable
        @test result.dof > 100
        @test result.scale > 0.01 && result.scale < 0.2
    end

    @testset "MarginalQuality + FTestLocalOptimization prediction variances" begin
        # Verify that prediction variances are larger than s² (leverage effect)
        points, _, _ = make_line_data(n_inliers=200, n_outliers=50, noise=0.1)
        problem = PredictiveLineFittingProblem(points)
        n = data_size(problem)
        p = sample_size(problem)

        scoring = MarginalQuality(n, p, 50.0)
        local_optimization = FTestLocalOptimization(PredictiveFTest(), 0.01, 5)
        result = ransac(problem, scoring; local_optimization,
                        config=RansacConfig(max_trials=2000, min_trials=100))
        @test result.converged

        # Compute prediction variances manually from param_cov
        s2 = result.scale^2
        Σ_θ = result.param_cov
        for i in 1:length(points)
            if result.inlier_mask[i]
                pt = points[i]
                g = SA[pt[1], pt[2], 1.0]
                v_i = s2 + dot(g, Σ_θ * g)
                # Prediction variance ≥ s² (equality only if leverage = 0)
                @test v_i >= s2 - 1e-12
            end
        end
    end

end
