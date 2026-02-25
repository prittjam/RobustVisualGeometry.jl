# Scoring API

## Quality Functions

```@docs
AbstractQualityFunction
AbstractMarginalQuality
MarginalQuality
PredictiveMarginalQuality
init_quality
default_local_optimization
```

## Local Optimization

```@docs
AbstractLocalOptimization
NoLocalOptimization
```

## Covariance Structure

The `CovarianceStructure` trait controls how per-point scores and penalties
are computed in the marginal sweep (Section 3.3). Determined automatically
from the problem and scoring type.

- `Homoscedastic` — isotropic noise, ``d_g = 1``: ``q_i = r_i^2``, ``\ell_i = 0``
- `Heteroscedastic` — anisotropic noise, model certain: ``q_i = g_i^\top \tilde\Sigma_{g_i}^{-1} g_i``, ``\ell_i = \log|\tilde\Sigma_{g_i}|``
- `Predictive` — model uncertain: includes parameter estimation error in ``\tilde\Sigma_{g_i}``
