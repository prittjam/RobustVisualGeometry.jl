# =============================================================================
# RANSAC Traits — SolverCardinality, ConstraintType
# =============================================================================

# -----------------------------------------------------------------------------
# Holy Trait: Solver Cardinality
# -----------------------------------------------------------------------------

"""
    SolverCardinality

Holy Trait indicating whether a minimal solver returns one or multiple candidate models.

Used for compile-time dispatch to eliminate iteration overhead for single-model solvers.

- `SingleSolution()`: `solve` returns `M` or `nothing`
- `MultipleSolutions()`: `solve` returns an iterable of `M` or `nothing`
"""
abstract type SolverCardinality end

"""
    SingleSolution <: SolverCardinality

Trait indicating the solver returns exactly one model (or nothing).
Enables a faster code path without iteration over candidates.
"""
struct SingleSolution <: SolverCardinality end

"""
    MultipleSolutions <: SolverCardinality

Trait indicating the solver may return multiple candidate models.
The RANSAC loop iterates over all candidates and keeps the best.
"""
struct MultipleSolutions <: SolverCardinality end

# -----------------------------------------------------------------------------
# Holy Trait: Linear System Type
# -----------------------------------------------------------------------------
#
# Distinguishes null-space problems (Ah = 0, SVD) from overdetermined
# linear systems (Ax ≈ b, weighted LS). Enables trait-dispatched solve
# in the generic IRLS loop.
#

"""
    ConstraintType

Holy Trait for the constraint structure of a RANSAC problem's parameterization.

- `Constrained()`: Model has a gauge constraint h(θ)=0 (e.g., ‖h‖=1).
  Produces a null-space problem `Ah = 0`, solved via SVD.
  Covariance requires the bordered Hessian / implicit function theorem.
- `Unconstrained()`: Model uses minimal (free) coordinates.
  Produces an overdetermined system `Ax ≈ b`, solved via weighted LS.
  Covariance is σ²(X'X)⁻¹.
"""
abstract type ConstraintType end

"""
    Constrained <: ConstraintType

Model parameterization has a gauge constraint h(θ)=0 (e.g., ‖h‖=1 for
homography, fundamental matrix). The IRLS refinement solves the null-space
problem `Ah = 0` via SVD, and the covariance is computed by projecting
I⁻¹ onto the tangent plane of the constraint manifold via the bordered
Hessian from the Lagrangian formulation.
"""
struct Constrained <: ConstraintType end

"""
    Unconstrained <: ConstraintType

Model parameterization uses minimal (free) coordinates with no constraint.
The IRLS refinement solves `Ax ≈ b` via normal equations, and the
covariance is σ²·inv(I) where I is the Fisher information.
"""
struct Unconstrained <: ConstraintType end
