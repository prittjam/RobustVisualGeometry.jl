# =============================================================================
# RANSAC Traits — SolverCardinality
# =============================================================================

# -----------------------------------------------------------------------------
# Holy Trait: Solver Cardinality
# -----------------------------------------------------------------------------

"""
    SolverCardinality

Holy Trait indicating whether a minimal solver returns one or multiple candidate models.

Used for compile-time dispatch to eliminate iteration overhead for single-model solvers.

- `SingleSolution()`: `solve` returns `M` or `nothing`
- `MultipleSolutions()`: `solve` returns `FixedModels{N,M}` or `nothing`
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
# LinearFit — Concrete type for LO-RANSAC refit solver dispatch
# -----------------------------------------------------------------------------

"""
    LinearFit

Concrete type for the solver used by `fit(problem, mask, weights, ::LinearFit)`
during local optimization. Covers null-space DLT (Ah=0 via SVD) for
correspondence problems and GEP/EIV solvers for line fitting.
"""
struct LinearFit end
