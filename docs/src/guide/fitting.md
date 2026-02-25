# Fitting Algorithms

## Conic (Ellipse) Fitting

Multiple conic fitting methods are available, from simple algebraic to robust geometric:

### Pipeline Overview

```
ALS → Taubin → FNS → Robust Taubin → Robust FNS → GNC FNS → Lifted FNS → Geometric
```

Each successive method handles more noise and outliers at higher computational cost.

### Algebraic Least Squares

```julia
result = fit_conic_als(points)
```

Minimizes algebraic distance. Fast but biased — tends to produce hyperbolas on noisy data.

### Taubin Approximation

```julia
result = fit_conic_taubin(points)
```

Minimizes approximate geometric distance via the Taubin constraint. Better than ALS
but still not fully optimal.

### Fundamental Numerical Scheme (FNS)

```julia
result = fit_conic_fns(points)
```

Iteratively optimizes the Sampson distance (first-order approximation to geometric
distance). Produces high-quality fits with correct uncertainty estimates.

### Robust Fitting

```julia
# Two-phase: robust Taubin for initialization, then robust FNS for refinement
result = fit_conic_robust_taubin_fns(points; loss=GemanMcClureLoss())

# Individual phases
result = fit_conic_robust_taubin(points; loss=CauchyLoss())
result = fit_conic_robust_fns(points; loss=TukeyLoss())
```

### GNC Fitting

```julia
result = fit_conic_gnc_fns(points)
```

Uses Graduated Non-Convexity for very high outlier rates.

### Conversion to Ellipse

```julia
result = fit_conic_robust_taubin_fns(points)
ellipse = conic_to_ellipse(result.value, result.scale)
```

## Line Fitting

### Total Least Squares

```julia
result = fit_line(points)                    # basic TLS
result = fit_line(points, covariances)       # weighted TLS with per-point covariances
```

### RANSAC Line Fitting

```julia
result = fit_line_ransac(points; σ=0.5, confidence=0.999)
line = result.value       # Uncertain{Line2D} with covariance
inliers = result.inlier_mask
```

## Homography Fitting

### RANSAC + IRLS Pipeline

```julia
using VisualGeometryCore: csponds, SA

src = [SA[100.0, 200.0], ...]
dst = [SA[112.0, 195.0], ...]

result = fit_homography(csponds(src, dst))
H = result.value          # 3×3 homography matrix
```

Or build the pipeline manually:

```julia
problem = HomographyProblem(csponds(src, dst))
quality = MarginalQuality(problem, 50.0)
result = ransac(problem, quality)
```

## Fundamental Matrix Fitting

### Robust Fitting (No RANSAC)

```julia
# Taubin → FNS pipeline with robust weights
result = fit_fundmat_robust_taubin_fns(csponds)
F = result.value          # 3×3 fundamental matrix
```

### RANSAC + FNS Pipeline

```julia
result = fit_fundmat(csponds)
```
