# =============================================================================
# fit_line_ransac — Public API
# =============================================================================

"""
    fit_line_ransac(points::AbstractVector{<:Uncertain{<:Point2}};
        scoring=nothing, config=RansacConfig(), outlier_halfwidth=50.0)

Robust line fitting via RANSAC with scale-free marginal scoring.

Returns a `RansacEstimate`.

# Arguments
- `scoring`: Quality function for model selection.
  Default: `MarginalScoring(problem, outlier_halfwidth)`.
- `config`: RANSAC loop configuration.
- `outlier_halfwidth`: Half-width of the outlier domain (parameter `a` in Eq. 12).
  Only used when `scoring` is not provided.

# Examples
```julia
result = fit_line_ransac(points)

# Custom scoring
scoring = PredictiveMarginalScoring(length(points), 2, 30.0; codimension=1)
result = fit_line_ransac(points; scoring)
```
"""
function fit_line_ransac(
    points::AbstractVector{<:Uncertain{<:Point2}};
    scoring::Union{AbstractScoring, Nothing} = nothing,
    config::RansacConfig = RansacConfig(),
    outlier_halfwidth::Real = 50.0,
)
    problem = LineFittingProblem(points)
    scoring = something(scoring, MarginalScoring(problem, Float64(outlier_halfwidth)))
    ransac(problem, scoring; config)
end

"""
    fit_line_ransac(points::AbstractVector{<:Point2}; kwargs...)

Convenience overload wrapping each point with identity covariance.
"""
function fit_line_ransac(
    points::AbstractVector{<:Point2};
    kwargs...,
)
    T = float(eltype(eltype(points)))
    I2 = SMatrix{2,2,T}(one(T), zero(T), zero(T), one(T))
    uncertain_pts = [Uncertain(Point2{T}(p...), I2) for p in points]
    fit_line_ransac(uncertain_pts; kwargs...)
end
