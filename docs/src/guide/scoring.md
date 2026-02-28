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
    - n_o\, d_g\, \log(2a)
```

where ``n_i = |\mathcal{I}|``, ``\text{RSS}_\mathcal{I} = \sum_{i \in \mathcal{I}} q_i``,
``n_o = N - n_i``, ``a`` is the outlier domain half-width, and ``d_g`` is the
codimension.

The three terms:
1. **Inlier reward**: ``\log\Gamma(n_i d_g/2)`` — grows super-linearly with inlier count
2. **Fit quality**: ``-\frac{n_i d_g}{2}\log(\pi\,\text{RSS}_\mathcal{I})`` — penalizes large residuals
3. **Outlier penalty**: ``-n_o\, d_g\, \log(2a)`` — each outlier costs ``d_g \log(2a)``

!!! note "Covariance penalty"
    The general derivation includes a fourth term
    ``-\frac{1}{2}\sum_{i \in \mathcal{I}} \log|\tilde\Sigma_{g_i}|`` (per-point
    covariance penalty). In the whitened formulation used by RVG, this term is
    identically zero: the outlier density is defined in the whitened constraint
    space ``r_i = \tilde\Sigma_{g_i}^{-1/2} g_i``, so the ``|\tilde\Sigma_{g_i}|^{-1/2}``
    Jacobian from the outlier density cancels with the identical factor from the
    inlier Gaussian normalization. The `sweep!` function retains a `penalties`
    parameter for API stability but always receives zeros.

## Scoring Functions

### MarginalScoring

Model-certain scoring: treats ``\theta`` as exact (``\Sigma_\theta = 0``).
The per-point scores are ``q_i = g_i^\top \tilde\Sigma_{g_i}^{-1} g_i``
and penalties ``\ell_i = \log|\tilde\Sigma_{g_i}|``.

**When to use**: Default choice for most problems.

**RVG type**: [`MarginalScoring`](@ref)

```julia
# Problem-aware constructor (recommended) — derives n, p, codimension
scoring = MarginalScoring(problem, 50.0)

# Manual constructor
scoring = MarginalScoring(n, p, a; codimension=1)
```

### PredictiveMarginalScoring

Model-uncertain scoring: accounts for parameter estimation error from the
minimal sample (``\Sigma_\theta \neq 0``). The constraint covariance
includes both measurement noise and model uncertainty (Eq. 7):

```math
\tilde\Sigma_{g_i} = \partial_x g_i\, \tilde\Sigma_{x_i}\, (\partial_x g_i)^\top
    + \partial_\theta g_i\, \tilde\Sigma_\theta\, (\partial_\theta g_i)^\top
```

Falls back to `MarginalScoring` when the problem does not implement
`solver_jacobian`.

**When to use**: When you want the sweep to account for the conditioning of the
minimal sample (high-leverage points get appropriately wider acceptance bands).

**RVG type**: [`PredictiveMarginalScoring`](@ref)

```julia
scoring = PredictiveMarginalScoring(problem, 50.0)
```

## Whitening Contract and Scoring Dispatch

The scoring layer dispatches on a single axis: **model type**.

| Model type | ``q_i`` | When |
|------------|---------|------|
| `model::M` (certain) | ``r_i^2`` | `MarginalScoring` (default) |
| `model::Uncertain{M}` | ``r_i^2 / \tilde{v}_i`` | `PredictiveMarginalScoring` with ``\Sigma_\theta \neq 0`` |

The key design invariant is the **whitening contract**: every `residuals!`
implementation must return whitened residuals such that ``r_i^2`` equals the
Mahalanobis distance ``g_i^\top \tilde\Sigma_{g_i}^{-1} g_i``. Whether the
problem has isotropic or heteroscedastic measurement noise is the problem's
responsibility — it handles the whitening inside `residuals!`. For example:

- **Isotropic** (``\tilde\Sigma_{x_i} = I``): the Sampson distance
  ``r_i = g_i / \sqrt{c_i}`` is already whitened, where ``c_i = \|\partial_x g_i\|^2``.
- **Heteroscedastic** (``\tilde\Sigma_{x_i}`` varies): `residuals!` must
  apply ``r_i = L_i^{-1} g_i`` where ``L_i L_i^\top = C_i = \partial_x g_i \tilde\Sigma_{x_i} (\partial_x g_i)^\top``.

Because `residuals!` handles whitening, the scoring layer only needs to
square the residuals (model-certain path) or additionally inflate by the
predictive variance (model-uncertain path). The covariance penalty is
``\ell_i = 0`` for both paths (whitened outlier formulation — the
``|\tilde\Sigma_{g_i}|^{-1/2}`` factors cancel between the inlier Gaussian
and the outlier uniform density in the whitened space).

## Local Optimization (LO-RANSAC)

Local optimization refines promising hypotheses during the RANSAC loop via
alternating refit-resweep cycles. Controlled via the `local_optimization`
kwarg to `ransac()`:

- `NoLocalOptimization()` — no local optimization (default)
- `PosteriorIrls()` — posterior-weight IRLS refinement

Problems that support LO-RANSAC implement `fit(problem, mask, weights, ::LinearFit)`.

```julia
# LO-RANSAC with PosteriorIrls
problem = HomographyProblem(csponds(src, dst))
scoring = MarginalScoring(problem, 50.0)
result = ransac(problem, scoring; local_optimization=PosteriorIrls())
```

## Algorithm: Optimal Partition via Sweep (Algorithm 1)

The sweep algorithm finds the optimal inlier set ``\mathcal{I}^*`` by:
1. Sort points by ``q_i`` (ascending)
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
