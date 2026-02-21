# Scoring Functions

This page describes the scoring functions used by RANSAC for model selection
and the statistical framework for post-selection inference. The theory is
based on "Inlier Tests and Covariance Propagation for RANSAC with Known and
Unknown Scale" (Pritts, 2025).

## Overview

RANSAC has two phases:

1. **Model selection** — score candidate models, keep the best.
2. **Post-selection inference** — classify inliers with calibrated F-tests,
   estimate parameter covariance, propagate uncertainty.

All scoring functions are equivalent for model selection when their
hyperparameters are properly tuned (Section 7.9 of the paper). The
discriminating factor is *post-selection inference*: calibrated inlier
tests, parameter covariance ``\Sigma_\theta``, and uncertainty propagation
to derived quantities.

## The Statistical Framework

### Known scale (``\sigma`` given)

When ``\sigma`` is known, residuals follow a classical Gaussian model.
The inlier test is a ``\chi^2`` test:

```math
\frac{r_*^2}{\sigma^2} \sim \chi^2_1
```

The parameter estimate is Gaussian with covariance
``\sigma^2 (X^\top X)^{-1}`` (Eq 8). Covariance propagation to derived
quantities ``\mathbf{q} = g(\hat\theta)`` uses the Jacobian ``G``:

```math
\text{Cov}(\mathbf{q}) = \sigma^2 \, G \, (X^\top X)^{-1} G^\top
```

### Unknown scale (``\sigma`` estimated)

When ``\sigma`` is unknown, placing the Jeffreys prior
``\pi(\sigma^2) \propto 1/\sigma^2`` and marginalizing produces Student's
``t`` distributions and ``F`` test statistics. The scale estimate is
``s^2 = \text{RSS}/\nu`` where ``\nu = n - p``.

**Inlier test** (Eq 18): the squared standardized residual follows an ``F``
distribution:

```math
\frac{r_*^2}{s^2} \sim F(1, \nu)
```

**Parameter uncertainty** (Eq 22): the parameter posterior is a multivariate
Student's ``t``:

```math
\theta \mid \text{data} \sim t_\nu\!\left(\hat\theta,\; s^2(X^\top X)^{-1}\right)
```

**Prediction-corrected inlier test** (Eq 25): accounts for parameter
uncertainty via the prediction variance ``v_* = s^2(1 + G_* (X^\top X)^{-1} G_*^\top)``:

```math
\frac{r_*^2}{v_*} \sim F(1, \nu)
```

High-leverage points get a wider acceptance band, producing correctly
calibrated Type I error rates even at small sample sizes.

## Scoring Functions

### ThresholdQuality (MSAC)

MSAC-style truncated quality with a fixed inlier threshold. Requires a
loss function, threshold, and scale estimator.

**Formula**: ``Q(\theta) = \sum_i \max(\tau - \rho(r_i/\sigma), 0)``

**When to use**: You know ``\sigma`` (or have a good estimate), and want
a simple, fast scoring function with a tunable threshold.

**RVG type**: [`ThresholdQuality`](@ref)

```julia
scoring = ThresholdQuality(L2Loss(), 3.0, FixedScale(σ=1.0))
result = ransac(problem, scoring)
```

### ChiSquareQuality

Chi-square hypothesis test with truncated quality. Inherently L2 — the
cutoff is derived from the significance level ``\alpha`` and the model
codimension ``d_g``.

**Formula**: ``Q(\theta) = \sum_i \max(\chi^2_{d_g, 1-\alpha} - (r_i/\sigma)^2, 0)``

**When to use**: You know ``\sigma`` and want the threshold determined
automatically from a significance level.

**RVG type**: [`ChiSquareQuality`](@ref)

```julia
scoring = ChiSquareQuality(FixedScale(σ=2.0), 0.01)
result = ransac(problem, scoring)
```

### MarginalQuality

Threshold-free marginal likelihood scoring. Marginalizes ``\sigma^2`` against
the Jeffreys prior, producing a scale-free score (Eq 45, Algorithm 1):

```math
S(\theta, \mathcal{I}) = \log\Gamma\!\left(\frac{k}{2}\right)
    - \frac{k}{2}\log\text{RSS}_\mathcal{I}
    - (N-k)\log(2a)
```

The three terms have clear roles:
1. ``\log\Gamma(k/2)`` — **inlier reward**: grows super-linearly with inlier count
2. ``-\frac{k}{2}\log\text{RSS}_\mathcal{I}`` — **fit quality**: penalizes large residuals
3. ``-(N-k)\log(2a)`` — **outlier penalty**: each outlier costs ``\log(2a)``

The score is completely ``\sigma``-free. The only user parameter is the
outlier half-width ``a`` (maximum plausible residual).

**When to use**: You do not know ``\sigma`` and want a principled scoring
function that automatically determines both the scale and the inlier set.

**RVG type**: [`MarginalQuality`](@ref)

```julia
scoring = MarginalQuality(n, p, 50.0)  # n=data size, p=model DOF, a=50
result = ransac(problem, scoring)
```

### PredictiveMarginalQuality

Predictive variant of marginal scoring. The marginal sweep operates on
prediction-corrected F-statistics ``F_i = r_i^2 / V_i`` instead of raw
``r_i^2``, where ``V_i`` accounts for parameter uncertainty from the
minimal sample via leverage matrices (Eq 50).

Falls back to raw ``r^2`` (identical to `MarginalQuality`) when the
problem does not implement `solver_jacobian`.

**When to use**: Same as `MarginalQuality`, but you want the sweep to
account for the conditioning of the minimal sample.

**RVG type**: [`PredictiveMarginalQuality`](@ref)

```julia
scoring = PredictiveMarginalQuality(n, p, 50.0)
result = ransac(problem, scoring)
```

## F-Test Local Optimization

After model selection, iterative F-test local optimization (Algorithm 3) produces
a calibrated inlier mask and parameter covariance.

### Basic F-test

Classifies point ``i`` as inlier when (Eq 18/47):

```math
\frac{r_i^2}{d \cdot s^2} < F_{d,\nu,1-\alpha}
```

where ``d = d_g`` is the codimension, ``s^2 = \text{RSS}/\nu``, and ``\nu``
is the degrees of freedom from the current inlier set.

**RVG type**: [`BasicFTest`](@ref) (used inside [`FTestLocalOptimization`](@ref))

### Prediction-corrected F-test

Accounts for parameter uncertainty via the prediction covariance
``V_i = s^2 I_d + J_i \Sigma_\theta J_i^\top`` (Eq 49):

```math
\frac{\mathbf{r}_i^\top V_i^{-1} \mathbf{r}_i}{d} < F_{d,\nu,1-\alpha}
```

High-leverage points get wider acceptance bands.

**RVG type**: [`PredictiveFTest`](@ref) (used inside [`FTestLocalOptimization`](@ref))

### Two-level gating (Algorithm 2)

The RANSAC loop uses two-level gating for marginal quality:

1. **Global gate** (cheap): marginal sweep on raw ``r^2``, producing score
   ``S_g``. Reject if ``S_g \le S_g^*`` (current best global score).
2. **Local gate** (expensive): F-test local optimization + re-score with model
   covariance, producing ``S_l``. Accept only if ``S_l > S_l^*``.

Global scores compete with global scores, local with local. The global
gate filters unpromising trials cheaply; the local gate selects the best
model under the full covariance model.

### Monotonicity guard

Each F-test local optimization iteration is guarded by a monotonicity check: the
scoring-consistent quality (`_lo_quality`) must improve. If it doesn't,
the loop stops to prevent oscillation.

### Usage

```julia
# Basic F-test local optimization
result = ransac(problem, scoring;
                local_optimization=FTestLocalOptimization(alpha=0.01))

# Prediction-corrected F-test → returns UncertainRansacEstimate with Σ_θ
result = ransac(problem, scoring;
                local_optimization=FTestLocalOptimization(test=PredictiveFTest(), alpha=0.01))
```

## Stopping Strategies

### HypergeometricStopping

Default. Relies on the adaptive trial count derived from the inlier ratio:
stop when the probability of missing a better all-inlier sample is below
``1 - \eta`` (the `confidence` parameter in [`RansacConfig`](@ref)).

### ScoreGapStopping

Bayesian score-gap early stopping (Section 7.6). Models the per-trial
improvement probability as Beta(1, G+1) where G is the number of
consecutive non-improvements. Stops when (Eq 61):

```math
\frac{T_\text{rem}}{G + 2} < \varepsilon
```

Useful for marginal scoring where the score may converge before the
inlier-count formula would allow stopping.

## Comparison

| Property | ThresholdQuality | ChiSquareQuality | MarginalQuality |
|----------|-----------------|------------------|-----------------|
| Free parameter | threshold ``\tau`` | significance ``\alpha`` | outlier half-width ``a`` |
| Scale treatment | fixed ``\sigma`` | fixed ``\sigma`` | Jeffreys ``\int d\sigma^2`` |
| Calibrated test | no | no | yes (``F``-test) |
| Covariance | --- | --- | ``s^2(J^\top W J)^{-1}`` |
| Parameter distribution | --- | --- | ``t_\nu(\hat\theta, s^2(J^\top W J)^{-1})`` |

All scoring functions produce equivalent model selection when hyperparameters
are properly tuned. The value of `MarginalQuality` + `FTestLocalOptimization` lies
in post-selection inference.

## Post-Selection Inference

When `FTestLocalOptimization` with `PredictiveFTest` is used, the result is an
[`UncertainRansacEstimate`](@ref) that provides:

- **Calibrated inlier mask** via the ``F(d,\nu)`` test with prediction correction
- **Parameter covariance** ``\Sigma_\theta = s^2 (J^\top W J)^{-1}``
- **Scale estimate** ``s = \sqrt{\text{RSS}/\nu}`` with ``\nu`` degrees of freedom
- **Propagation** to derived quantities via ``\text{Cov}(\mathbf{q}) = s^2 \, G \, (J^\top W J)^{-1} G^\top``

The noise posterior is ``\sigma^2 \mid \text{data} \sim \text{InvGamma}(\nu/2, \nu s^2/2)``,
and residual predictions follow ``r/s \sim t_\nu``.

```julia
# Full pipeline: marginal scoring + predictive F-test
scoring = MarginalQuality(n, p, 50.0)
result = ransac(problem, scoring;
                local_optimization=FTestLocalOptimization(test=PredictiveFTest(), alpha=0.01))

result.value        # estimated model
result.inlier_mask  # calibrated inlier mask
result.scale        # s = √(RSS/ν)
result.dof          # ν = n_inliers - p
result.param_cov    # Σ_θ (SMatrix)
```
