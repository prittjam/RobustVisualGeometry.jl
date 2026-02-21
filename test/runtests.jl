#!/usr/bin/env julia

using Test

@testset "RobustVisualGeometry Tests" begin
    include("test_ransac.jl")
    include("test_ransac_homography.jl")
    include("test_ransac_fundmat.jl")
    include("test_noiseless_ransac.jl")
    include("test_homography_jacobian.jl")
    include("test_inhom_line.jl")
    # NOTE: test_ransac_convergence.jl is slow (Monte Carlo) — run separately
    #   include("test_ransac_convergence.jl")
end
