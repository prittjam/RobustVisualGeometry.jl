using Test
using LinearAlgebra
using StaticArrays
using Random
using ForwardDiff
using VisualGeometryCore
using RobustVisualGeometry
using RobustVisualGeometry: residual_jacobian, _sq_norm, _sampson_distance_homography

# =============================================================================
# Test Utilities
# =============================================================================

"""Apply homography H to 2D point (homogeneous multiply + perspective divide)."""
function apply_H(H::SMatrix{3,3,T,9}, p::SVector{2,T}) where T
    h = H * SA[p[1], p[2], one(T)]
    return SA[h[1]/h[3], h[2]/h[3]]
end

"""Autodiff reference: homography_4pt as a function of the 16-element input vector."""
function _h4pt_vec_ad(x::AbstractVector{T}) where T
    s1 = SVector{2,T}(x[1], x[2]);  s2 = SVector{2,T}(x[3], x[4])
    s3 = SVector{2,T}(x[5], x[6]);  s4 = SVector{2,T}(x[7], x[8])
    d1 = SVector{2,T}(x[9], x[10]); d2 = SVector{2,T}(x[11], x[12])
    d3 = SVector{2,T}(x[13], x[14]); d4 = SVector{2,T}(x[15], x[16])
    o = one(T)
    U  = SMatrix{3,3,T}(s1[1],s1[2],o, s2[1],s2[2],o, s3[1],s3[2],o)
    Up = SMatrix{3,3,T}(d1[1],d1[2],o, d2[1],d2[2],o, d3[1],d3[2],o)
    c  = U  \ SA[-s4[1], -s4[2], -o]
    cp = Up \ SA[-d4[1], -d4[2], -o]
    λ = cp ./ c
    H_raw = Up * StaticArrays.SDiagonal(λ[1], λ[2], λ[3]) * inv(U)
    n_H = norm(H_raw)
    H_norm = H_raw / n_H
    sign = H_norm[3,3] < zero(T) ? -o : o
    H_out = sign * H_norm
    SVector{9,T}(H_out[1,1], H_out[2,1], H_out[3,1],
                  H_out[1,2], H_out[2,2], H_out[3,2],
                  H_out[1,3], H_out[2,3], H_out[3,3])
end

"""Autodiff reference: transfer error as a function of vec(H)."""
function _te_vec_ad(hv::AbstractVector{T}, s, d) where T
    Hm = SMatrix{3,3,T}(hv[1], hv[2], hv[3], hv[4], hv[5], hv[6],
                          hv[7], hv[8], hv[9])
    proj = Hm * SA[s[1], s[2], one(T)]
    SA[proj[1]/proj[3] - d[1], proj[2]/proj[3] - d[2]]
end

# =============================================================================
# Tests
# =============================================================================

@testset "Homography Jacobian" begin

    @testset "homography_4pt_with_jacobian — autodiff verification" begin
        rng = MersenneTwister(42)

        H_true = @SMatrix [
             0.95  -0.10  15.0;
             0.12   0.93  -8.0;
             1e-4   2e-4   1.0
        ]
        H_true = H_true / norm(H_true)
        if H_true[3,3] < 0; H_true = -H_true; end

        src = [SA[100.0 + 700.0*rand(rng), 100.0 + 500.0*rand(rng)] for _ in 1:4]
        dst = [apply_H(H_true, s) for s in src]

        result = homography_4pt_with_jacobian(
            src[1], src[2], src[3], src[4],
            dst[1], dst[2], dst[3], dst[4])
        @test !isnothing(result)
        H, J = result
        @test H isa SMatrix{3,3,Float64,9}
        @test J isa SMatrix{9,16,Float64}

        # Verify H matches homography_4pt
        H_ref = homography_4pt(src[1], src[2], src[3], src[4],
                               dst[1], dst[2], dst[3], dst[4])
        @test !isnothing(H_ref)
        @test norm(H - H_ref) < 1e-10

        # ForwardDiff Jacobian
        x0 = vcat(src[1], src[2], src[3], src[4], dst[1], dst[2], dst[3], dst[4])
        J_ad = ForwardDiff.jacobian(_h4pt_vec_ad, collect(x0))
        @test maximum(abs.(Matrix(J) - J_ad)) < 1e-8
    end

    @testset "homography_4pt_with_jacobian — second configuration" begin
        # Different H with strong perspective
        H_true = @SMatrix [
            2.0   0.5   100.0;
           -0.3   1.5   -50.0;
            0.002 0.001   1.0
        ]
        H_true = H_true / norm(H_true)
        if H_true[3,3] < 0; H_true = -H_true; end

        src = [SA[100.0, 200.0], SA[500.0, 100.0], SA[800.0, 600.0], SA[200.0, 700.0]]
        dst = [apply_H(H_true, s) for s in src]

        result = homography_4pt_with_jacobian(
            src[1], src[2], src[3], src[4],
            dst[1], dst[2], dst[3], dst[4])
        @test !isnothing(result)
        H, J = result

        x0 = vcat(src[1], src[2], src[3], src[4], dst[1], dst[2], dst[3], dst[4])
        J_ad = ForwardDiff.jacobian(_h4pt_vec_ad, collect(x0))
        @test maximum(abs.(Matrix(J) - J_ad)) < 1e-8
    end

    @testset "homography_4pt_with_jacobian — degenerate returns nothing" begin
        src = [SA[0.0, 0.0], SA[1.0, 1.0], SA[2.0, 2.0], SA[3.0, 3.0]]
        dst = [SA[10.0, 20.0], SA[30.0, 10.0], SA[50.0, 40.0], SA[20.0, 60.0]]
        result = homography_4pt_with_jacobian(
            src[1], src[2], src[3], src[4],
            dst[1], dst[2], dst[3], dst[4])
        @test isnothing(result)
    end

    @testset "_transfer_error_jacobian_wrt_h — autodiff verification" begin
        rng = MersenneTwister(123)

        H = @SMatrix [
             0.95  -0.10  15.0;
             0.12   0.93  -8.0;
             1e-4   2e-4   1.0
        ]
        H = H / norm(H)
        if H[3,3] < 0; H = -H; end

        s = SA[300.0 + 400.0*rand(rng), 200.0 + 300.0*rand(rng)]
        d = apply_H(H, s) + SA[0.5*randn(rng), 0.5*randn(rng)]

        G = VisualGeometryCore._transfer_error_jacobian_wrt_h(s, d, H)
        @test G isa SMatrix{2,9,Float64}

        h0 = SVector{9,Float64}(H[1,1], H[2,1], H[3,1], H[1,2], H[2,2], H[3,2],
                                 H[1,3], H[2,3], H[3,3])
        G_ad = ForwardDiff.jacobian(hv -> _te_vec_ad(hv, s, d), collect(h0))
        @test maximum(abs.(Matrix(G) - G_ad)) < 1e-8
    end

    @testset "_transfer_error_jacobian_wrt_h — pure translation" begin
        H = @SMatrix [
             1.0  0.0  10.0;
             0.0  1.0  -5.0;
             0.0  0.0   1.0
        ]
        H = H / norm(H)

        s = SA[100.0, 200.0]
        d = apply_H(H, s)
        G = VisualGeometryCore._transfer_error_jacobian_wrt_h(s, d, H)

        @test all(isfinite, G)
        @test norm(G) > 0

        h0 = SVector{9,Float64}(H[1,1], H[2,1], H[3,1], H[1,2], H[2,2], H[3,2],
                                 H[1,3], H[2,3], H[3,3])
        G_ad = ForwardDiff.jacobian(hv -> _te_vec_ad(hv, s, d), collect(h0))
        @test maximum(abs.(Matrix(G) - G_ad)) < 1e-8
    end

    # =================================================================
    # residual_jacobian — ForwardDiff verification (Section 3, Eq. 7)
    # =================================================================

    @testset "residual_jacobian — ∂θgᵢ ForwardDiff verification" begin
        # Autodiff reference: constraint gᵢ(vec(H)) for fixed (s, d)
        function _constraint_ad(hv::AbstractVector{T}, s, d) where T
            H = SMatrix{3,3,T}(hv[1], hv[2], hv[3], hv[4], hv[5], hv[6],
                                hv[7], hv[8], hv[9])
            x, y = T(s[1]), T(s[2])
            xp, yp = T(d[1]), T(d[2])
            c = H[3,1]*x + H[3,2]*y + H[3,3]
            g1 = (H[1,1]*x + H[1,2]*y + H[1,3]) - xp*c
            g2 = (H[2,1]*x + H[2,2]*y + H[2,3]) - yp*c
            SVector{2,T}(g1, g2)
        end

        # Autodiff reference: constraint gᵢ(xᵢ) for fixed H
        function _constraint_x_ad(xv::AbstractVector{T}, H) where T
            x, y, xp, yp = xv[1], xv[2], xv[3], xv[4]
            c = T(H[3,1])*x + T(H[3,2])*y + T(H[3,3])
            g1 = (T(H[1,1])*x + T(H[1,2])*y + T(H[1,3])) - xp*c
            g2 = (T(H[2,1])*x + T(H[2,2])*y + T(H[2,3])) - yp*c
            SVector{2,T}(g1, g2)
        end

        rng = MersenneTwister(77)
        H_true = @SMatrix [
             0.95  -0.10  15.0;
             0.12   0.93  -8.0;
             1e-4   2e-4   1.0
        ]
        H_true = H_true / norm(H_true)
        if H_true[3,3] < 0; H_true = -H_true; end

        # Generate 10 correspondences (HomographyProblem requires ≥ 4)
        n_pts = 10
        src_pts = [SA[100.0 + 700.0*rand(rng), 100.0 + 500.0*rand(rng)] for _ in 1:n_pts]
        dst_pts = [apply_H(H_true, s) + SA[2.0*randn(rng), 2.0*randn(rng)] for s in src_pts]
        cs = [src_pts[i] => dst_pts[i] for i in 1:n_pts]
        prob = HomographyProblem(cs)

        for trial in 1:n_pts
            s = src_pts[trial]
            d = dst_pts[trial]

            # Compute via our function
            rᵢ, ∂θgᵢ_w, ℓᵢ = residual_jacobian(prob, H_true, trial)

            # --- Verify ∂θgᵢ via ForwardDiff ---
            h0 = collect(SVector{9,Float64}(
                H_true[1,1], H_true[2,1], H_true[3,1],
                H_true[1,2], H_true[2,2], H_true[3,2],
                H_true[1,3], H_true[2,3], H_true[3,3]))
            ∂θg_ad = ForwardDiff.jacobian(hv -> _constraint_ad(hv, s, d), h0)

            # --- Verify ∂ₓgᵢ via ForwardDiff ---
            x0 = collect(SVector{4,Float64}(s[1], s[2], d[1], d[2]))
            ∂ₓg_ad = ForwardDiff.jacobian(xv -> _constraint_x_ad(xv, H_true), x0)

            # Compute Σ̃_{gᵢ} = (∂ₓgᵢ)(∂ₓgᵢ)ᵀ and its Cholesky
            Σ̃_g = ∂ₓg_ad * ∂ₓg_ad'
            L = cholesky(Symmetric(Σ̃_g)).L

            # Verify whitened model Jacobian: ∂θgᵢ_w = L⁻¹∂θgᵢ
            ∂θg_w_expected = L \ ∂θg_ad
            @test maximum(abs.(Matrix(∂θgᵢ_w) - ∂θg_w_expected)) < 1e-8

            # Verify whitened constraint: rᵢ = L⁻¹gᵢ
            gᵢ = _constraint_ad(h0, s, d)
            r_expected = L \ gᵢ
            @test maximum(abs.(rᵢ - r_expected)) < 1e-8

            # Verify qᵢ = rᵢᵀrᵢ = Sampson distance²
            sampson_d = _sampson_distance_homography(s, d, H_true)
            @test abs(_sq_norm(rᵢ) - sampson_d^2) < 1e-8

            # Verify ℓᵢ = log|Σ̃_{gᵢ}|
            @test abs(ℓᵢ - log(det(Σ̃_g))) < 1e-8
        end
    end

end
