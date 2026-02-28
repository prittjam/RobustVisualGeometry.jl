#!/usr/bin/env julia
# =============================================================================
# Benchmark: Fundamental Matrix RANSAC timing vs. noise level and outlier rate
#
# Sweeps σ × outlier_rate for three RANSAC variants.
# Reports median time, inlier count, ŝ/σ ratio, and max Sampson error.
#
# Usage:
#   julia --project=. bench/fundmat_timing.jl
# =============================================================================

using LinearAlgebra, StaticArrays, Random, Printf, Statistics
using BenchmarkTools
using RobustVisualGeometry
using VisualGeometryCore: FundMat, sampson_distance
using VisualGeometryCore.Matching: csponds

# =============================================================================
# Data Generation
# =============================================================================

function make_fundmat_data(rng, σ, n_inliers, n_outliers)
    f = 500.0
    K = SA[f 0 320.0; 0 f 240.0; 0 0 1.0]
    θ = 0.3
    R = SA[cos(θ) 0 sin(θ); 0 1 0; -sin(θ) 0 cos(θ)]
    t = SA[1.0, 0.2, 0.1]
    P1 = K * SA[1.0 0 0 0; 0 1 0 0; 0 0 1 0]
    P2 = K * hcat(R, t)

    n = n_inliers + n_outliers
    src = Vector{SVector{2,Float64}}(undef, n)
    dst = Vector{SVector{2,Float64}}(undef, n)

    @inbounds for i in 1:n_inliers
        X = SA[randn(rng)*2, randn(rng)*2, randn(rng)*0.5 + 5.0]
        p1h = P1 * SA[X[1], X[2], X[3], 1.0]
        p2h = P2 * SA[X[1], X[2], X[3], 1.0]
        src[i] = SVector{2,Float64}(p1h[1]/p1h[3] + σ*randn(rng),
                                     p1h[2]/p1h[3] + σ*randn(rng))
        dst[i] = SVector{2,Float64}(p2h[1]/p2h[3] + σ*randn(rng),
                                     p2h[2]/p2h[3] + σ*randn(rng))
    end
    @inbounds for i in 1:n_outliers
        j = n_inliers + i
        src[j] = SVector{2,Float64}(randn(rng, 2) * 200 .+ 320)
        dst[j] = SVector{2,Float64}(randn(rng, 2) * 200 .+ 320)
    end
    csponds(src, dst)
end

# =============================================================================
# Run + Measure
# =============================================================================

struct BenchResult
    time_ms::Float64
    inliers::Int
    scale::Float64
    error::Float64
    quality::Float64
    trials::Int
end

function bench_fundmat(cs, a, config, lo, scoring_ctor)
    problem = FundMatProblem(cs)
    t = @belapsed ransac($problem, scoring; local_optimization=$lo, config=$config) setup=(
        scoring = $scoring_ctor($problem, $a))
    scoring = scoring_ctor(problem, a)
    result = ransac(problem, scoring; local_optimization=lo, config)
    err = if result.converged
        F = result.value
        max_err = 0.0
        for i in eachindex(cs)
            if result.inlier_mask[i]
                max_err = max(max_err, sampson_distance(cs[i].first, cs[i].second, F))
            end
        end
        max_err
    else
        NaN
    end
    BenchResult(t * 1000, sum(result.inlier_mask), result.scale, err,
                result.quality, result.trials)
end

# =============================================================================
# Formatting
# =============================================================================

function fmt_time(t_ms)
    t_ms < 1.0    && return @sprintf("%5.2f ms", t_ms)
    t_ms < 1000.0 && return @sprintf("%5.1f ms", t_ms)
    return @sprintf("%5.2f s ", t_ms / 1000)
end

function fmt_err(e)
    isnan(e) && return "     NaN"
    e < 0.01 && return @sprintf("%.2e", e)
    return @sprintf("%8.3f", e)
end

function fmt_ratio(r)
    isnan(r) && return "  NaN"
    return @sprintf("%5.2f", r)
end

# =============================================================================
# Sweep
# =============================================================================

function main()
    methods = [
        ("A: No LO",       NoLocalOptimization(), MarginalScoring),
        ("B: LO certain",  PosteriorIrls(),       MarginalScoring),
        ("C: LO predict",  PosteriorIrls(),       PredictiveMarginalScoring),
    ]

    σ_range       = [0.0, 1.0, 3.0, 5.0, 10.0]
    outlier_fracs = [0.0, 0.3, 0.5, 0.7]
    n_inliers     = 200
    a             = 500.0
    max_trials    = 50_000
    n_reps        = 3

    config = RansacConfig(max_trials=max_trials, min_trials=100)

    W = 120
    println()
    println("=" ^ W)
    println("  Fundamental Matrix — timing + accuracy sweep (@belapsed)")
    println("  $n_inliers inliers, a=$a, max_trials=$max_trials, $n_reps reps")
    println("=" ^ W)

    for (method_label, lo, scoring_ctor) in methods
        println()
        println("  ── $method_label ──")
        @printf("  %5s  %5s  │ %9s  %7s  %5s  %8s  %6s\n",
                "σ", "out%", "time", "inliers", "ŝ/σ", "error", "trials")
        println("  " * "─" ^ (W - 4))

        for σ in σ_range
            for out_frac in outlier_fracs
                n_outliers = round(Int, n_inliers * out_frac / (1 - out_frac + eps()))
                out_pct = round(Int, 100 * n_outliers / (n_inliers + n_outliers))

                all_time = Float64[]
                all_inliers = Int[]
                all_ratio = Float64[]
                all_err = Float64[]
                all_trials = Int[]

                for rep in 1:n_reps
                    rng = MersenneTwister(1000 * rep + round(Int, 10σ))
                    cs = make_fundmat_data(rng, σ, n_inliers, n_outliers)
                    shuffle!(rng, cs)
                    r = bench_fundmat(cs, a, config, lo, scoring_ctor)
                    push!(all_time, r.time_ms)
                    push!(all_inliers, r.inliers)
                    σ > 0 && push!(all_ratio, r.scale / σ)
                    push!(all_err, r.error)
                    push!(all_trials, r.trials)
                end

                med_time = median(all_time)
                med_inliers = median(all_inliers)
                med_ratio = σ > 0 ? median(all_ratio) : NaN
                med_err = median(filter(!isnan, all_err))
                med_trials = round(Int, median(all_trials))

                ratio_str = σ > 0 ? fmt_ratio(med_ratio) : "  n/a"
                @printf("  %5.1f  %4d%%  │ %9s  %7.0f  %5s  %8s  %6d\n",
                        σ, out_pct, fmt_time(med_time), med_inliers,
                        ratio_str, fmt_err(med_err), med_trials)
            end
        end
    end
    println("=" ^ W)
end

main()
