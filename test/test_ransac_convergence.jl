# =============================================================================
# RANSAC Adaptive Stopping — Convergence Rate Tests
# =============================================================================
#
# Verifies that RANSAC's adaptive trial count matches the theoretical
# prediction for noiseless data at various outlier fractions.
#
# Theory (no sample rejection, q=1):
#   p = C(K,k)/C(N,k)                            hypergeometric success prob
#   N_adapt = ceil(log(1-conf)/log(1-p))          adaptive trial count
#   E[trials] = N_adapt + (1-p)^N_adapt / p      = E[max(Geom(p), N_adapt)]
#
# Theory (with sample rejection, q<1):
#   Some solvers reject valid samples (degeneracy check, oriented epipolar
#   constraint, etc.). The overall acceptance rate q decomposes as:
#
#     q = r * q_solve
#
#   where:
#     r       = P(test_sample passes)             sample acceptance rate (SAR)
#     q_solve = P(solve succeeds | test_sample passes, all-inlier sample)
#
#   Both r and q_solve are measured from 0% outlier runs:
#     r       = mean(result.sample_acceptance_rate)
#     q       = 1 / mean(result.trials)
#     q_solve = q / r
#
#   Total trials:
#     p_eff = p * q                   per-trial probability of all-inlier model
#     N_adapt = ceil(log(1-conf)/log(1-p))   RANSAC's adaptive stopping criterion
#     E[trials] = N_adapt + (1-p_eff)^N_adapt / p_eff
#
#   Effective trials (non-degenerate samples, i.e. test_sample passed):
#     E[effective] = E[trials] * SAR_ε
#     where SAR_ε = mean sample acceptance rate at outlier fraction ε
#
# Uses plain problem types (no local optimization) with chi-square inlier threshold.
# Homography and F-matrix scenes use the Synthetic module with real camera rigs.
#
# =============================================================================

using Test
using StaticArrays
using Random
using Statistics: mean, std

using VisualGeometryCore
using VisualGeometryCore: Point2
using RobustVisualGeometry
using RobustVisualGeometry: _p_all_inliers, codimension
using VisualGeometryCore.Synthetic

# =============================================================================
# Generic convergence test runner
# =============================================================================

"""
Run RANSAC convergence test for a problem type across outlier fractions.

Arguments:
- `name`: test set label
- `make_problem`: (seed, n_inliers, n_outliers) -> AbstractRansacProblem
- `k`: sample size
- `n_inliers`: number of inlier data points
- `sigma`: noise scale for chi-square threshold
- `n_runs`: Monte Carlo runs per outlier fraction
- `confidence`: RANSAC confidence
- `outlier_fractions`: vector of outlier fractions to test
"""
function test_convergence(name, make_problem;
                          k, n_inliers, sigma, n_runs=500,
                          confidence=0.99,
                          outlier_fractions=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    @testset "$name" begin
        # =====================================================================
        # Phase 1: Measure q and its decomposition from 0% outlier runs.
        # At 0% outliers, every sample is all-inlier:
        #   r       = mean(sample_acceptance_rate)     test_sample pass rate
        #   q       = 1/mean(trials)                   overall acceptance rate
        #   q_solve = q / r                            solver acceptance rate
        # =====================================================================
        q_runs = 200
        q_trials = Vector{Int}(undef, q_runs)
        q_sars = Vector{Float64}(undef, q_runs)
        for run in 1:q_runs
            problem = make_problem(run + 100_000, n_inliers, 0)
            config = RansacConfig(; confidence, max_trials=100_000, min_trials=1)
            result = let dof = codimension(problem)
                        t = chi2_threshold(Float64(confidence), dof)
                        threshold = rho(L2Loss(), t)
                        scoring = ThresholdQuality(L2Loss(), threshold, FixedScale(σ=Float64(sigma)))
                        ransac(problem, scoring; config)
                    end
            q_trials[run] = result.trials
            q_sars[run] = result.sample_acceptance_rate
        end
        q = 1.0 / mean(q_trials)
        r = mean(q_sars)
        q_solve = q / r

        println("\nRANSAC convergence: $name (k=$k, n_inliers=$n_inliers, n_runs=$n_runs, conf=$confidence)")
        println("  q = $(round(q, digits=4))  (r = $(round(r, digits=4)), q_solve = $(round(q_solve, digits=4)))")
        println("="^100)
        println(rpad("outl_%", 8), rpad("N", 6), rpad("N_hyp", 7),
                rpad("E[tot]", 9), rpad("mean_tot", 10),
                rpad("E[eff]", 9), rpad("mean_eff", 10),
                rpad("SAR", 8), rpad("min", 6), "max")
        println("-"^100)

        @testset "q decomposition" begin
            @test 0.0 < r ≤ 1.0
            @test 0.0 < q_solve ≤ 1.0
            @test q ≈ r * q_solve  rtol=1e-10
        end

        for ε in outlier_fractions
            n_total = ε > 0 ? ceil(Int, n_inliers / (1 - ε)) : n_inliers
            n_outliers = n_total - n_inliers

            # Theoretical prediction for total trials
            p_hyper = _p_all_inliers(n_inliers, n_total, k)
            p_eff = p_hyper * q

            # N_adapt from pure hypergeometric — what RANSAC computes adaptively
            # after finding the best (all-inlier) model.
            if p_hyper ≈ 1.0
                N_adapt_hyper = 1
            elseif p_hyper ≈ 0
                N_adapt_hyper = 100_000
            else
                N_adapt_hyper = ceil(Int, log(1 - confidence) / log(1 - p_hyper))
            end

            # E[trials] = E[max(Geometric(p_eff), N_adapt_hyper)]
            if p_eff ≈ 0
                E_trials = 100_000.0
            else
                E_trials = N_adapt_hyper + (1 - p_eff)^N_adapt_hyper / p_eff
            end

            trial_counts = Vector{Int}(undef, n_runs)
            sar_values = Vector{Float64}(undef, n_runs)
            for run in 1:n_runs
                problem = make_problem(run, n_inliers, n_outliers)
                config = RansacConfig(; confidence, max_trials=100_000, min_trials=1)
                result = let dof = codimension(problem)
                        t = chi2_threshold(Float64(confidence), dof)
                        threshold = rho(L2Loss(), t)
                        scoring = ThresholdQuality(L2Loss(), threshold, FixedScale(σ=Float64(sigma)))
                        ransac(problem, scoring; config)
                    end
                @test result.converged
                trial_counts[run] = result.trials
                sar_values[run] = result.sample_acceptance_rate
            end

            # Effective trials per run: trials * SAR (exact by construction)
            eff_counts = [round(Int, trial_counts[i] * sar_values[i]) for i in 1:n_runs]

            m_tot = mean(trial_counts)
            s_tot = std(trial_counts)
            sar_mean = mean(sar_values)
            m_eff = mean(eff_counts)

            # E[effective] = E[total] * SAR_ε
            E_eff = E_trials * sar_mean

            label = "$(round(Int, ε*100))%"
            println(rpad(label, 8), rpad(n_total, 6), rpad(N_adapt_hyper, 7),
                    rpad(round(E_trials, digits=1), 9),
                    rpad("$(round(m_tot, digits=1))±$(round(s_tot, digits=1))", 10),
                    rpad(round(E_eff, digits=1), 9),
                    rpad(round(m_eff, digits=1), 10),
                    rpad(round(sar_mean, digits=3), 8),
                    rpad(minimum(trial_counts), 6), maximum(trial_counts))

            @testset "ε=$label" begin
                # Total trials match theory
                @test m_tot ≈ E_trials atol=max(1.0, 0.10 * E_trials)
                @test minimum(trial_counts) >= N_adapt_hyper
                # Effective trials match E[total] * SAR
                @test m_eff ≈ E_eff atol=max(1.0, 0.15 * E_eff)
                # SAR is valid
                @test 0.0 < sar_mean ≤ 1.0
            end
        end
    end
end

# =============================================================================
# Line scene (2D — no camera rig needed)
# =============================================================================

function make_line_problem(seed, n_inliers, n_outliers)
    rng = MersenneTwister(seed)
    pts = Point2{Float64}[]
    for _ in 1:n_inliers
        x = 10 * rand(rng) - 5
        y = 2x + 1   # exact, no noise
        push!(pts, Point2(x, y))
    end
    for _ in 1:n_outliers
        x = 30 * rand(rng) - 15
        y = 30 * rand(rng) - 15
        push!(pts, Point2(x, y))
    end
    shuffle!(rng, pts)
    return LineFittingProblem(pts)
end

# =============================================================================
# Homography scene (coplanar 3D points, stereo rig)
# =============================================================================

const CONVERGENCE_RIG = synthetic_stereo_rig(;
    focal=800.0, pp=(320.0, 240.0), baseline=1.0)

function make_homography_problem(seed, n_inliers, n_outliers)
    n_total = n_inliers + n_outliers
    outlier_frac = n_outliers > 0 ? n_outliers / n_total : 0.0
    cs, _, _ = planar_correspondences(CONVERGENCE_RIG;
        n=n_inliers, outlier_frac, sigma=0.0, seed)
    return HomographyProblem(cs)
end

# =============================================================================
# Fundamental matrix scene (general 3D points, stereo rig)
# =============================================================================

function make_fundmat_problem(seed, n_inliers, n_outliers)
    n_total = n_inliers + n_outliers
    outlier_frac = n_outliers > 0 ? n_outliers / n_total : 0.0
    cs, _, _ = stereo_correspondences(CONVERGENCE_RIG;
        n=n_inliers, outlier_frac, sigma=0.0, seed)
    return FundMatProblem(cs)
end

# =============================================================================
# Tests
# =============================================================================

test_convergence("noiseless line, no local optimization", make_line_problem;
                 k=2, n_inliers=200, sigma=1e-7)

test_convergence("noiseless homography, no local optimization", make_homography_problem;
                 k=4, n_inliers=200, sigma=1e-7)

test_convergence("noiseless F-matrix, no local optimization", make_fundmat_problem;
                 k=7, n_inliers=100, sigma=1e-7, n_runs=200,
                 outlier_fractions=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
