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
    fit_homography(correspondences; quality=nothing,
        config=RansacConfig(), outlier_halfwidth=50.0)

Robust homography estimation via RANSAC with scale-free marginal scoring.

Returns a `RansacEstimate`.

# Arguments
- `quality`: Quality function for model selection.
  Default: `MarginalQuality(problem, outlier_halfwidth)`.
- `config`: RANSAC loop configuration.
- `outlier_halfwidth`: Half-width of the outlier domain (parameter `a` in Eq. 12).
  Only used when `quality` is not provided.

# Examples
```julia
result = fit_homography(correspondences)

# Custom quality
quality = PredictiveMarginalQuality(length(correspondences), 4, 30.0; codimension=2)
result = fit_homography(correspondences; quality)
```
"""
function fit_homography(correspondences;
    quality::Union{AbstractQualityFunction, Nothing} = nothing,
    config::RansacConfig = RansacConfig(),
    outlier_halfwidth::Real = 50.0)

    prob = HomographyProblem(correspondences)
    scoring = something(quality, MarginalQuality(prob, Float64(outlier_halfwidth)))
    ransac(prob, scoring; config)
end
