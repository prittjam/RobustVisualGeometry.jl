#!/usr/bin/env julia
# =============================================================================
# Benchmark: Homography estimation vs. noise level
#
# Generates noisy source+target correspondences on a 1000×1000 image,
# sweeps σ = 1..10, runs three RANSAC variants:
#   1. No LO            (MarginalScoring{Nothing},    NoLocalOptimization)
#   2. LO model-certain (MarginalScoring{Nothing},    PosteriorIrls)
#   3. LO predictive    (PredictiveMarginalScoring,   PosteriorIrls)
#
# Noise: σ per coordinate on BOTH source and target (isotropic Σ_x = σ²I₄).
# Reports: inlier count, predicted scale ŝ, ŝ/σ ratio, reproj error.
# =============================================================================

using LinearAlgebra, StaticArrays, Random, Printf
using RobustVisualGeometry
using VisualGeometryCore: PerspectiveMap, ProjectiveMap, HomographyMat
using VisualGeometryCore.Matching: csponds

# ─── Ground truth homography ────────────────────────────────────────────────
H_raw = @SMatrix [
     0.98  -0.08  12.0;
     0.07   0.97  -5.0;
     5e-5   3e-5   1.0
]
H_raw = H_raw / norm(H_raw)
H_raw[3,3] < 0 && (H_raw = -H_raw)
const H_TRUE = HomographyMat{Float64}(Tuple(H_raw))
const WARP = PerspectiveMap() ∘ ProjectiveMap(H_TRUE)

# ─── Test points for reprojection error ─────────────────────────────────────
const TEST_PTS = [SA[200.0, 200.0], SA[500.0, 500.0], SA[800.0, 300.0],
                  SA[300.0, 700.0], SA[700.0, 800.0]]

function max_reproj_err(H::HomographyMat)
    m1 = PerspectiveMap() ∘ ProjectiveMap(H)
    m2 = WARP
    maximum(norm(m1(p) - m2(p)) for p in TEST_PTS)
end

# ─── Data generation ────────────────────────────────────────────────────────
function make_data(σ::Float64; n_inliers=200, n_outliers=80, seed=123)
    rng = MersenneTwister(seed)
    src = SVector{2,Float64}[]
    dst = SVector{2,Float64}[]

    for _ in 1:n_inliers
        s0 = SA[50.0 + 900.0 * rand(rng), 50.0 + 900.0 * rand(rng)]
        s = s0 + SA[σ * randn(rng), σ * randn(rng)]        # noisy source
        d = WARP(s0) + SA[σ * randn(rng), σ * randn(rng)]   # noisy target
        push!(src, s)
        push!(dst, d)
    end
    for _ in 1:n_outliers
        s = SA[50.0 + 900.0 * rand(rng), 50.0 + 900.0 * rand(rng)]
        d = SA[50.0 + 900.0 * rand(rng), 50.0 + 900.0 * rand(rng)]
        push!(src, s)
        push!(dst, d)
    end
    src, dst
end

# ─── Result type ────────────────────────────────────────────────────────────
struct RunResult
    inliers::Int
    scale::Float64
    reproj::Float64
    quality::Float64
end

function result_from(r)
    RunResult(sum(r.inlier_mask), r.scale,
              r.converged ? max_reproj_err(r.value) : NaN, r.quality)
end

# ─── Run one σ ──────────────────────────────────────────────────────────────
function run_one(σ; a=500.0, max_trials=5000, n_reps=5)
    res1 = RunResult[]   # No LO
    res2 = RunResult[]   # LO model-certain
    res3 = RunResult[]   # LO predictive

    for rep in 1:n_reps
        src, dst = make_data(σ; seed=100 + rep)
        problem = HomographyProblem(csponds(src, dst))
        config  = RansacConfig(max_trials=max_trials, min_trials=200)

        # ── 1. No LO, model-certain ──
        s1 = MarginalScoring(problem, a)
        r1 = ransac(problem, s1; config)
        push!(res1, result_from(r1))

        # ── 2. LO, model-certain ──
        s2 = MarginalScoring(problem, a)
        r2 = ransac(problem, s2; local_optimization=PosteriorIrls(), config)
        push!(res2, result_from(r2))

        # ── 3. LO, predictive ──
        s3 = PredictiveMarginalScoring(problem, a)
        r3 = ransac(problem, s3; local_optimization=PosteriorIrls(), config)
        push!(res3, result_from(r3))
    end

    return res1, res2, res3
end

# ─── Statistics helper ──────────────────────────────────────────────────────
function meanstd(f, v)
    vals = [f(x) for x in v]
    μ = sum(vals) / length(vals)
    σ = length(vals) > 1 ? sqrt(sum(x -> (x - μ)^2, vals) / (length(vals) - 1)) : 0.0
    (mean=μ, std=σ)
end

fmt(m) = @sprintf("%6.2f±%-5.2f", m.mean, m.std)

# ─── Main ────────────────────────────────────────────────────────────────────
function main()
    σ_range = 1.0:1.0:10.0
    n_reps = 5

    # ── Collect results ──
    R1 = Dict{Float64, Vector{RunResult}}()
    R2 = Dict{Float64, Vector{RunResult}}()
    R3 = Dict{Float64, Vector{RunResult}}()

    for σ in σ_range
        R1[σ], R2[σ], R3[σ] = run_one(σ; n_reps)
    end

    W = 140

    # ═══════════════════════════════════════════════════════════════════════
    # Summary table
    # ═══════════════════════════════════════════════════════════════════════
    println("=" ^ W)
    println("  Homography noise sweep — mean ± std over $n_reps reps")
    println("  200 inliers, 80 outliers, 1000×1000 image, a=500")
    println("  Noise: σ per coordinate on BOTH source and target (Σ_x = σ²I₄)")
    println("  dg=2, m=4, ν=(k−m)·dg, ŝ=√(RSS_I/ν)")
    println("=" ^ W)

    # ── Header ──
    println()
    @printf("  %5s │ %12s  %12s  %12s  %12s │", "σ", "inliers", "ŝ", "ŝ/σ", "reproj")
    @printf(" %12s  %12s  %12s  %12s\n", "inliers", "ŝ", "ŝ/σ", "reproj")

    @printf("  %5s │ %55s │ %55s\n", "",
            "   (A) No LO, model-certain", "   (B) LO, model-certain")
    println("─" ^ W)

    for σ in σ_range
        a = meanstd(r -> r.inliers, R1[σ])
        b = meanstd(r -> r.scale,   R1[σ])
        c = meanstd(r -> r.scale/σ, R1[σ])
        d = meanstd(r -> r.reproj,  R1[σ])

        e = meanstd(r -> r.inliers, R2[σ])
        f = meanstd(r -> r.scale,   R2[σ])
        g = meanstd(r -> r.scale/σ, R2[σ])
        h = meanstd(r -> r.reproj,  R2[σ])

        @printf("  %5.1f │ %12.1f  %12s  %12s  %12s │ %12.1f  %12s  %12s  %12s\n",
                σ, a.mean, fmt(b), fmt(c), fmt(d),
                   e.mean, fmt(f), fmt(g), fmt(h))
    end
    println("=" ^ W)

    # ── Second block: (B) vs (C) ──
    println()
    @printf("  %5s │ %12s  %12s  %12s  %12s │", "σ", "inliers", "ŝ", "ŝ/σ", "reproj")
    @printf(" %12s  %12s  %12s  %12s\n", "inliers", "ŝ", "ŝ/σ", "reproj")

    @printf("  %5s │ %55s │ %55s\n", "",
            "   (B) LO, model-certain", "   (C) LO, predictive")
    println("─" ^ W)

    for σ in σ_range
        e = meanstd(r -> r.inliers, R2[σ])
        f = meanstd(r -> r.scale,   R2[σ])
        g = meanstd(r -> r.scale/σ, R2[σ])
        h = meanstd(r -> r.reproj,  R2[σ])

        i = meanstd(r -> r.inliers, R3[σ])
        j = meanstd(r -> r.scale,   R3[σ])
        k = meanstd(r -> r.scale/σ, R3[σ])
        l = meanstd(r -> r.reproj,  R3[σ])

        @printf("  %5.1f │ %12.1f  %12s  %12s  %12s │ %12.1f  %12s  %12s  %12s\n",
                σ, e.mean, fmt(f), fmt(g), fmt(h),
                   i.mean, fmt(j), fmt(k), fmt(l))
    end
    println("=" ^ W)

    # ── All three side by side (compact: just ŝ/σ and reproj) ──
    println()
    println("=" ^ W)
    println("  Compact comparison: ŝ/σ and reproj (px)")
    println("=" ^ W)
    @printf("  %5s │ %28s │ %28s │ %28s\n",
            "", "(A) No LO", "(B) LO certain", "(C) LO predictive")
    @printf("  %5s │ %13s  %13s │ %13s  %13s │ %13s  %13s\n",
            "σ", "ŝ/σ", "reproj", "ŝ/σ", "reproj", "ŝ/σ", "reproj")
    println("─" ^ W)

    for σ in σ_range
        a = meanstd(r -> r.scale/σ, R1[σ])
        b = meanstd(r -> r.reproj,  R1[σ])
        c = meanstd(r -> r.scale/σ, R2[σ])
        d = meanstd(r -> r.reproj,  R2[σ])
        e = meanstd(r -> r.scale/σ, R3[σ])
        f = meanstd(r -> r.reproj,  R3[σ])

        @printf("  %5.1f │ %13s  %13s │ %13s  %13s │ %13s  %13s\n",
                σ, fmt(a), fmt(b), fmt(c), fmt(d), fmt(e), fmt(f))
    end
    println("=" ^ W)
    println("  ŝ/σ ≈ 1.0 → predicted scale matches true noise level.")
    println("  reproj = max reprojection error vs ground truth H (pixels).")
end

main()
