# =============================================================================
# RobustVisualGeometry.jl
# =============================================================================
#
# Robust estimation algorithms extracted from VisualGeometryCore:
# - M-estimation (IRLS) with pluggable loss functions
# - Graduated Non-Convexity (GNC) for high outlier rates
# - RANSAC with pluggable quality functions and LO-RANSAC refinement
# - Problem implementations: lines, conics, homographies, F-matrices, P3P
#
# Depends on VisualGeometryCore for geometry types, solvers, losses, and scale
# estimators. Re-exports VGC's losses and scale for convenience.
#
# =============================================================================

module RobustVisualGeometry

using LinearAlgebra: LinearAlgebra, Diagonal, I, cond, norm, svd, dot,
                     eigen, Symmetric, diag, det, tr, pinv, cross
using Statistics: median
using StaticArrays: StaticArrays, SVector, SMatrix, @SMatrix, SA
using StructArrays: StructArrays
using FixedSizeArrays: FixedSizeArrays, FixedSizeArray
using Random: Random

# =============================================================================
# Imports from VisualGeometryCore
# =============================================================================

using VisualGeometryCore

# Explicit imports of internal (non-exported) VGC symbols
# Import VGC functions we need to extend with new methods in RVG.
# Without `import`, `function foo(...)` creates a NEW function in RVG scope,
# shadowing VGC's version and breaking dispatch for VGC's method signatures.
import VisualGeometryCore: rho, psi, weight,           # GNC extends these
                           tuning_constant,             # GNC extends this
                           sampson_distance,             # conic_fitting extends this
                           scale,                        # interface extends this
                           SVDWorkspace, svd_nullvec!    # used by RANSAC + solvers

# Import non-exported VGC symbols
import VisualGeometryCore: RotMatrix,
                           _corrected_scale,
                           _homography_sample_nondegenerate,
                           _fill_homography_dlt!,
                           _fill_fundamental_dlt!,
                           _vec9_to_mat33,
                           _oriented_epipolar_check,
                           projection_transform

# =============================================================================
# EXPORTS - Organized by category
# =============================================================================

# Re-export VGC losses and scale for convenience
# (users get the full robust API from one package)
export AbstractLoss, GNCLoss
export L2Loss, HuberLoss, CauchyLoss, TukeyLoss
export GemanMcClureLoss, WelschLoss, FairLoss
export rho, psi, weight, tuning_constant
export AbstractScaleEstimator
export MADScale, WeightedScale, FixedScale, SpatialMADScale
export estimate_scale, chi2_threshold

# -----------------------------------------------------------------------------
# 1. ABSTRACT TYPES - Interface definitions
# -----------------------------------------------------------------------------
export AbstractEstimator
export AbstractRobustProblem

# -----------------------------------------------------------------------------
# 2. ROBUST PROBLEM INTERFACE - Generic problem API
# -----------------------------------------------------------------------------
export initial_solve, compute_residuals, compute_residuals!, weighted_solve
export data_size, problem_dof, convergence_metric
export IRLSWorkspace

# -----------------------------------------------------------------------------
# 3. M-ESTIMATORS - IRLS-based estimators
# -----------------------------------------------------------------------------
export MEstimator, LinearRobustProblem
export robust_solve

# -----------------------------------------------------------------------------
# 4. GNC ESTIMATORS - Graduated Non-Convexity
# -----------------------------------------------------------------------------
export GNCTruncatedLS, GNCGemanMcClure
export GNCEstimator

# -----------------------------------------------------------------------------
# 5. RESULT TYPE - Fitting results
# -----------------------------------------------------------------------------
export RobustEstimate, RobustAttributes

# StatsBase-compatible accessors
export coef, residuals, weights, scale, converged, niter

# -----------------------------------------------------------------------------
# 6. RANSAC - Random Sample Consensus
# -----------------------------------------------------------------------------
export AbstractRansacProblem, FixedModels, RansacRefineProblem
export AbstractRefinement, NoRefinement, DltRefinement, IrlsRefinement
export AbstractQualityFunction
export AbstractMarginalQuality, MarginalQuality, PredictiveMarginalQuality
export AbstractLocalOptimization, NoLocalOptimization
export default_local_optimization
export AbstractSampler, UniformSampler, ProsacSampler, sampler
export RansacConfig, RansacEstimate, RansacAttributes
export residual_jacobian, solver_jacobian, measurement_logdets!, model_covariance
export init_quality
export ransac, inlier_ratio, codimension
export SVDWorkspace, svd_nullvec!

# RANSAC problem interface methods
export sample_size, model_type, solve, residuals!, test_sample, test_model
export refine, solver_cardinality, draw_sample!, test_consensus
export constraint_type, weighted_system, model_from_solution

# -----------------------------------------------------------------------------
# 7. CONIC FITTING - Algebraic, Taubin, FNS, Robust FNS, Lifted FNS, Geometric
# -----------------------------------------------------------------------------
export fit_conic_als, fit_conic_taubin, fit_conic_robust_taubin, fit_conic_fns
export fit_conic_robust_fns, fit_conic_robust_taubin_fns
export fit_conic_gnc_fns, fit_conic_lifted_fns, fit_conic_geometric
export conic_carrier, conic_carrier_jacobian, conic_carrier_covariance
export conic_to_ellipse, ConicFitResult
export sampson_distance_sq
export ConicTaubinProblem, ConicFNSProblem

# -----------------------------------------------------------------------------
# 8. LINE FITTING - Orthogonal distance with per-point covariances
# -----------------------------------------------------------------------------
export LineFittingProblem, LoLineFittingProblem
export InhomLineFittingProblem, EivLineFittingProblem
export fit_line_ransac

# -----------------------------------------------------------------------------
# 9. RANSAC PROBLEM TYPES - Correspondence problems
# -----------------------------------------------------------------------------
export AbstractCspondProblem
export HomographyProblem, LoHomographyProblem
export FundamentalMatrixProblem, LoFundamentalMatrixProblem
export P3PProblem, Pose3

# -----------------------------------------------------------------------------
# 10. HOMOGRAPHY FITTING - RANSAC
# -----------------------------------------------------------------------------
export fit_homography

# -----------------------------------------------------------------------------
# 11. FUNDAMENTAL MATRIX FITTING - Taubin, FNS, Robust Taubin→FNS, RANSAC→FNS
# -----------------------------------------------------------------------------
export fit_fundmat, fit_fundmat_robust_taubin, fit_fundmat_robust_fns, fit_fundmat_robust_taubin_fns
export FMatTaubinProblem, FMatFNSProblem, FMatFitResult

# =============================================================================
# INCLUDE ORDER - Dependency-aware loading
# =============================================================================
#
# DEPENDENCY GRAPH (arrows show "depends on"):
#
#   interface.jl          (AbstractEstimator, AbstractRobustProblem, RobustAttributes)
#        ↓
#   irls.jl               (MEstimator, LinearRobustProblem, robust_solve)
#        ↓                 Uses: AbstractLoss/Scale from VGC, interface types
#   gnc.jl                (GNCTruncatedLS, GNCGemanMcClure, GNCEstimator)
#        ↓                 Uses: AbstractLoss/Scale from VGC, interface types
#   ransac_interface.jl   (AbstractRansacProblem, quality functions, workspace)
#        ↓                 Uses: Attributed from VGC, interface types
#   ransac.jl             (RANSAC algorithm)
#        ↓                 Uses: ransac_interface, losses, scale from VGC
#   gep_solver.jl         (GEP solver, shared fitting helpers)
#        ↓
#   conic_fitting.jl      (Robust conic fitting)
#        ↓                 Uses: Ellipse, HomEllipseMat from VGC
#   line_fitting.jl       (Line fitting)
#        ↓                 Uses: Line2D, Point2, Uncertain from VGC
#   ransac_line.jl        (Line RANSAC problems)
#        ↓                 Uses: line_fitting + VGC types
#   ransac_cspond.jl      (Shared correspondence infrastructure)
#        ↓                 Uses: ransac_interface + StructArrays + FixedSizeArrays
#   ransac_p3p.jl         (P3P RANSAC)
#        ↓                 Uses: EuclideanMap, p3p, CameraModel from VGC
#   ransac_homography.jl  (Homography RANSAC)
#        ↓                 Uses: homography solvers from VGC
#   ransac_fundmat.jl     (F-matrix RANSAC)
#        ↓                 Uses: fundamental_matrix solvers from VGC
#   homography_fitting.jl (Homography fitting pipeline)
#        ↓                 Uses: ransac_homography
#   fundmat_fitting.jl    (F-matrix fitting pipeline)
#                          Uses: ransac_fundmat
#
# =============================================================================

# Interface: Abstract types and result struct
include("interface.jl")

# IRLS solver and MEstimator
include("irls.jl")

# GNC solver and GNCEstimator
include("gnc.jl")

# RANSAC interface: abstract types, traits, workspace, config
include("ransac_interface.jl")

# Scoring: quality functions, stopping strategies
include("scoring.jl")

# RANSAC algorithm: main loop, scoring, adaptive trials
include("ransac.jl")

# GEP solver and shared fitting helpers (used by conic_fitting, fundmat_fitting)
include("gep_solver.jl")

# Conic fitting — depends on interface + VGC Primitives (Ellipse, HomEllipseMat)
include("conic_fitting.jl")

# Line fitting — depends on VGC Primitives (Line2D, Point2, Uncertain)
include("line_fitting.jl")

# RANSAC line fitting — depends on line_fitting + VGC types
include("ransac_line.jl")

# Shared correspondence RANSAC infrastructure
include("ransac_cspond.jl")

# P3P RANSAC — depends on VGC (EuclideanMap, p3p, CameraModel)
include("ransac_p3p.jl")

# Homography RANSAC — depends on VGC homography solvers
include("ransac_homography.jl")

# Fundamental matrix RANSAC — depends on VGC F-matrix solvers
include("ransac_fundmat.jl")

# Homography fitting — depends on ransac_homography
include("homography_fitting.jl")

# Fundamental matrix fitting — depends on ransac_fundmat + VGC solvers
include("fundmat_fitting.jl")

end # module RobustVisualGeometry
