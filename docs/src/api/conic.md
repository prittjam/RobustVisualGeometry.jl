# Conic Fitting

```@meta
CurrentModule = RobustVisualGeometry
```

## Fitting Functions

```@docs
fit_conic_als
fit_conic_taubin
fit_conic_robust_taubin
fit_conic_fns
fit_conic_robust_fns
fit_conic_robust_taubin_fns
fit_conic_gnc_fns
fit_conic_lifted_fns
fit_conic_geometric
```

## Problem Types

```@docs
ConicTaubinProblem
ConicFNSProblem
```

## Results and Utilities

```@docs
ConicFitResult
conic_to_ellipse
sampson_distance_sq
conic_carrier
conic_carrier_jacobian
conic_carrier_covariance
```
