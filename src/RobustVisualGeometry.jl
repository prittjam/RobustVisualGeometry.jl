# =============================================================================
# RobustVisualGeometry.jl
# =============================================================================
#
# Robust estimation algorithms extracted from VisualGeometryCore:
# - M-estimation (IRLS) with pluggable loss functions
# - Graduated Non-Convexity (GNC) for high outlier rates
# - RANSAC with pluggable scoring functions and LO-RANSAC refinement
# - Problem implementations: lines, conics, homographies, F-matrices, P3P
#
# Depends on VisualGeometryCore for geometry types, solvers, losses, and scale
# estimators. Re-exports VGC's losses and scale for convenience.
#
# =============================================================================

module RobustVisualGeometry

using LinearAlgebra: LinearAlgebra, Diagonal, I, cond, norm, dot,
                     eigen, Symmetric, diag, det
using StaticArrays: StaticArrays, SVector, SMatrix, MVector, @SMatrix, SA
using StructArrays: StructArrays
using FixedSizeArrays: FixedSizeArrays, FixedSizeArray
using Random: Random

# =============================================================================
# Imports from VisualGeometryCore
# =============================================================================

# Types
using VisualGeometryCore: Point2, Line2D, Ellipse, Uncertain,
                          Attributed, AbstractAttributes,
                          HomEllipseMat, CameraModel, Pose3,
                          HomographyMat, FundMat,
                          FixedModels

# Scoring trait
using VisualGeometryCore: HasScore, scoring

# Loss functions (re-exported by RVG)
using VisualGeometryCore: AbstractLoss, GNCLoss,
                          L2Loss, HuberLoss, CauchyLoss, TukeyLoss,
                          GemanMcClureLoss, WelschLoss, FairLoss

# Scale estimators (re-exported by RVG)
using VisualGeometryCore: AbstractScaleEstimator,
                          MADScale, WeightedScale, FixedScale, SpatialMADScale,
                          estimate_scale, chi2_threshold

# Geometry operations
using VisualGeometryCore: normal, param_cov, sign_normalize,
                          enforce_rank_two,
                          hartley_normalization, hartley_unnormalize,
                          backproject,
                          sampson_whitened, sampson_logdets!

# Solvers
using VisualGeometryCore: homography_4pt, homography_4pt_with_jacobian,
                          homography_dlt!,
                          fundmat_7pt, fundmat_dlt!,
                          p3p

# Import VGC functions we need to extend with new methods in RVG.
# Without `import`, `function foo(...)` creates a NEW function in RVG scope,
# shadowing VGC's version and breaking dispatch for VGC's method signatures.
import VisualGeometryCore: rho, psi, weight,           # GNC extends these
                           tuning_constant,             # GNC extends this
                           sampson_distance,             # used by conic + fundmat fitting
                           sampson_distances!,           # used by residuals!
                           scale,                        # interface extends this
                           SVDWorkspace, svd_nullvec!,   # used by RANSAC + solvers
                           test_model                    # RANSAC problems extend this

# Import non-exported VGC symbols
import VisualGeometryCore: _corrected_scale,
                           _homography_sample_nondegenerate,
                           _vec9_to_mat33,
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
export AbstractTaubinProblem, AbstractFNSProblem

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
# fit is exported in section 6 (RANSAC) — same function, multiple dispatch

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
export AbstractRansacProblem, FixedModels
export AbstractScoring
export Predictive, MarginalScoring, PredictiveMarginalScoring
export AbstractLocalOptimization, NoLocalOptimization, PosteriorIrls
export default_local_optimization
export LinearFit
export AbstractSampler, UniformSampler, ProsacSampler, sampler
export RansacConfig, RansacEstimate, RansacAttributes
export residual_jacobian, solver_jacobian, measurement_logdets!, fit_param_covariance
export init_score
export ransac, inlier_ratio, codimension
export SVDWorkspace, svd_nullvec!

# RANSAC problem interface methods
export sample_size, model_type, solve, residuals!, test_sample, test_model
export fit, solver_cardinality, draw_sample!, test_consensus

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
export LineFittingProblem
export InhomLineFittingProblem, EivLineFittingProblem
export fit_line_ransac

# -----------------------------------------------------------------------------
# 9. RANSAC PROBLEM TYPES - Correspondence problems
# -----------------------------------------------------------------------------
export AbstractCspondProblem, AbstractDltProblem
export HomographyProblem
export FundMatProblem
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

# Shared interface
include("interface.jl")
include("gep.jl")

# Estimators — IRLS framework and concrete methods
include("estimators/irls_framework.jl")
include("estimators/irls.jl")
include("estimators/gnc.jl")

# Estimators — RANSAC framework
include("estimators/ransac/types.jl")
include("estimators/ransac/traits.jl")
include("estimators/ransac/samplers.jl")
include("estimators/ransac/interface.jl")
include("estimators/ransac/scoring.jl")
include("estimators/ransac/loop.jl")

# Fitting pipelines — Conic
include("fitting/conic/utils.jl")
include("fitting/conic/carriers.jl")
include("fitting/conic/types.jl")
include("fitting/conic/algebraic.jl")
include("fitting/conic/taubin.jl")
include("fitting/conic/fns.jl")
include("fitting/conic/gnc.jl")
include("fitting/conic/geometric.jl")

# Fitting pipelines — Line
include("fitting/line.jl")

# RANSAC problem implementations
include("estimators/ransac/problems/line.jl")
include("estimators/ransac/problems/cspond.jl")
include("estimators/ransac/problems/p3p.jl")
include("estimators/ransac/problems/homography.jl")
include("estimators/ransac/problems/fundmat.jl")

# Fitting pipelines — depend on RANSAC problems
include("fitting/line_ransac.jl")
include("fitting/homography.jl")
include("fitting/fundmat.jl")

end # module RobustVisualGeometry
