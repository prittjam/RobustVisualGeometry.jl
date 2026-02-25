# Scoring Functions

This page describes the scoring functions used by RANSAC for model selection.
The theory is based on "RANSAC Done Right" (Pritts, ECCV 2026).

## Overview

RANSAC scores each candidate model hypothesis using a scale-free marginal
likelihood (Section 3, Eq. 12). The score marginalizes ``\sigma^2`` under
the Jeffreys prior ``\pi(\sigma^2) \propto 1/\sigma^2``, eliminating the
need to know or estimate the noise scale in advance.

The only user parameter is the **outlier half-width** ``a`` — the maximum
plausible residual magnitude for an outlier.

## The Marginal Score (Eq. 12)

Given a model ``\theta`` and inlier set ``\mathcal{I}``:

```math
S(\theta, \mathcal{I}) = \log\Gamma\!\left(\frac{n_i d_g}{2}\right)
    - \frac{n_i d_g}{2}\log(\pi\,\text{RSS}_\mathcal{I})
    - \frac{1}{2}\sum_{i \in \mathcal{I}} \log|\tilde\Sigma_{g_i}|
    - n_o\, d_g\, \log(2a)
```

where ``n_i = |\mathcal{I}|``, ``\text{RSS}_\mathcal{I} = \sum_{i \in \mathcal{I}} q_i``,
``n_o = N - n_i``, ``a`` is the outlier domain half-width, and ``d_g`` is the
codimension.

The four terms:
1. **Inlier reward**: ``\log\Gamma(n_i d_g/2)`` — grows super-linearly with inlier count
2. **Fit quality**: ``-\frac{n_i d_g}{2}\log(\pi\,\text{RSS}_\mathcal{I})`` — penalizes large residuals
3. **Covariance penalty**: ``-\frac{1}{2}\sum \log|\tilde\Sigma_{g_i}|`` — adjusts for per-point noise
4. **Outlier penalty**: ``-n_o\, d_g\, \log(2a)`` — each outlier costs ``d_g \log(2a)``

## Scoring Functions

### MarginalQuality

Model-certain scoring: treats ``\theta`` as exact (``\Sigma_\theta = 0``).
The per-point scores are ``q_i = g_i^\top \tilde\Sigma_{g_i}^{-1} g_i``
and penalties ``\ell_i = \log|\tilde\Sigma_{g_i}|``.

**When to use**: Default choice for most problems.

**RVG type**: [`MarginalQuality`](@ref)

```julia
# Problem-aware constructor (recommended) — derives n, p, codimension
scoring = MarginalQuality(problem, 50.0)

# Manual constructor
scoring = MarginalQuality(n, p, a; codimension=1)
```

### PredictiveMarginalQuality

Model-uncertain scoring: accounts for parameter estimation error from the
minimal sample (``\Sigma_\theta \neq 0``). The constraint covariance
includes both measurement noise and model uncertainty (Eq. 7):

```math
\tilde\Sigma_{g_i} = \partial_x g_i\, \tilde\Sigma_{x_i}\, (\partial_x g_i)^\top
    + \partial_\theta g_i\, \tilde\Sigma_\theta\, (\partial_\theta g_i)^\top
```

Falls back to `MarginalQuality` when the problem does not implement
`solver_jacobian`.

**When to use**: When you want the sweep to account for the conditioning of the
minimal sample (high-leverage points get appropriately wider acceptance bands).

**RVG type**: [`PredictiveMarginalQuality`](@ref)

```julia
scoring = PredictiveMarginalQuality(problem, 50.0)
```

## Covariance Structure Trait

The `CovarianceStructure` trait (Section 3.3) controls how per-point scores
and penalties are computed:

| Trait | ``q_i`` | ``\ell_i`` | When |
|-------|---------|------------|------|
| `Homoscedastic` | ``r_i^2`` | ``0`` | Isotropic noise, ``d_g = 1`` |
| `Heteroscedastic` | ``g_i^\top \tilde\Sigma_{g_i}^{-1} g_i`` | ``\log|\tilde\Sigma_{g_i}|`` | Anisotropic noise, ``\Sigma_\theta = 0`` |
| `Predictive` | ``g_i^\top \tilde\Sigma_{g_i}^{-1} g_i`` | ``\log|\tilde\Sigma_{g_i}|`` | ``\Sigma_\theta \neq 0`` |

The covariance structure is determined automatically from the problem and
scoring type. Most problems with scalar residuals use `Homoscedastic`.

## Local Optimization (LO-RANSAC)

Local optimization refines promising hypotheses during the RANSAC loop via
alternating refit-resweep cycles. Controlled via the `local_optimization`
kwarg to `ransac()`:

- `NoLocalOptimization()` — no local optimization (default)
- `ConvergeThenRescore()` — WLS to convergence at fixed mask, then re-sweep (Strategy A)
- `StepAndRescore()` — single WLS step, then re-sweep (Strategy B)

Problems that support LO-RANSAC implement `fit(problem, mask, weights)`.

```julia
# LO-RANSAC with ConvergeThenRescore
problem = HomographyProblem(csponds(src, dst))
scoring = MarginalQuality(problem, 50.0)
result = ransac(problem, scoring; local_optimization=ConvergeThenRescore())
```

## Algorithm: Optimal Partition via Sweep (Algorithm 1)

The sweep algorithm finds the optimal inlier set ``\mathcal{I}^*`` by:
1. Sort points by ``q_i - \ell_i`` (ascending)
2. Scan from ``k = m+1`` to ``N``, computing ``S_k`` incrementally
3. Return ``k^* = \arg\max_k S_k``

This runs in ``O(N \log N)`` due to the sort.

## Two-Level Gating (Algorithm 2)

The RANSAC loop uses two-level gating for marginal quality:

1. **Global gate** (cheap): sweep on initial scores, producing ``S_g``.
   Reject if ``S_g \le S_g^*`` (current best global score).
2. **Local gate** (expensive): local optimization refit + re-score,
   producing ``S_l``. Accept only if ``S_l > S_l^*``.

Global scores compete with global scores, local with local. The global
gate filters unpromising trials cheaply.
