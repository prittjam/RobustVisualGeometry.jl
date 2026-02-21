# =============================================================================
# Homography Fitting
# =============================================================================
#
# Robust homography estimation via RANSAC.
#
#   fit_homography — thin wrapper around ransac() with composable quality
#                    and local optimization
#
# =============================================================================

# =============================================================================
# Public API
# =============================================================================

"""
    fit_homography(correspondences; quality=nothing, local_optimization=NoLocalOptimization(),
        config=RansacConfig(), sigma=1.0, confidence=0.99)

Robust homography estimation via RANSAC.

Returns a `RansacEstimate` (or `UncertainRansacEstimate` when using
`FTestLocalOptimization` with `PredictiveFTest`).

# Arguments — RANSAC
- `quality`: Quality function for model selection.
  Default: `ChiSquareQuality(FixedScale(σ=sigma), 1-confidence)`.
  Override to use e.g. `MarginalQuality(n, p, 50.0)` or
  `ThresholdQuality(CauchyLoss(), 3.0, FixedScale())`.
- `local_optimization`: LO-RANSAC strategy. Default: `NoLocalOptimization()`.
- `config`: RANSAC loop configuration.

# Arguments — default quality construction
- `sigma`: Noise σ (pixels). Used to build the default `quality`.
  Ignored when `quality` is provided explicitly.
- `confidence`: Chi-square confidence. Used to build the default `quality`.
  Ignored when `quality` is provided explicitly.

# Examples
```julia
# Simple (default ChiSquareQuality)
result = fit_homography(csponds(src, dst))

# Custom quality + LO-RANSAC
quality = MarginalQuality(length(correspondences), 8, 50.0)
lo = FTestLocalOptimization(test=PredictiveFTest(), alpha=0.01)
result = fit_homography(correspondences; quality, local_optimization=lo)
```
"""
function fit_homography(correspondences;
    quality::Union{AbstractQualityFunction, Nothing} = nothing,
    local_optimization::AbstractLocalOptimization = NoLocalOptimization(),
    config::RansacConfig = RansacConfig(),
    sigma::Real = 1.0,
    confidence::Real = 0.99)

    prob = HomographyProblem(correspondences)
    scoring = something(quality,
        ChiSquareQuality(FixedScale(σ=Float64(sigma)), 1.0 - Float64(confidence)))
    ransac(prob, scoring; local_optimization, config)
end
