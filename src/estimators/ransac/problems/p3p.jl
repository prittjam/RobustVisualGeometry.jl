# =============================================================================
# RANSAC P3P Problem
# =============================================================================
#
# Implements AbstractRansacProblem for camera pose estimation from 3D-2D
# point correspondences using the P3P minimal solver.
#
# Solver: P3P (0-4 solutions per sample)
# Residual: Forward reprojection error in pixels
# Refinement: None (no PnP solver yet)
# Sampling: PROSAC (scored correspondences) or uniform (unscored)
#
# PLACEMENT: Included from main VisualGeometryCore.jl (NOT from Estimators
# submodule) because it depends on EuclideanMap, RotMatrix, p3p, CameraModel,
# and backproject — all loaded after the Estimators submodule.
#
# =============================================================================

# Dependencies: VGC (EuclideanMap, RotMatrix, p3p, CameraModel, backproject)
#               All estimation types available from parent module

# =============================================================================
# Pose3 Type Alias
# =============================================================================

"""
    Pose3

Type alias for `EuclideanMap{3,Float64,RotMatrix{3,Float64,9},SVector{3,Float64}}`.

Represents a rigid 3D pose (rotation + translation). Used as the model type
for P3P RANSAC estimation.

`isbitstype(Pose3) == true` — stored inline in `RansacWorkspace`, zero heap
allocation on update.
"""
const Pose3 = EuclideanMap{3,Float64,RotMatrix{3,Float64,9},SVector{3,Float64}}

function Base.zero(::Type{Pose3})
    EuclideanMap(RotMatrix{3,Float64}(one(SMatrix{3,3,Float64,9})),
                 zero(SVector{3,Float64}))
end

# =============================================================================
# P3PProblem Type
# =============================================================================

"""
    P3PProblem{S,F} <: AbstractRansacProblem

RANSAC problem for camera pose estimation from 3D-2D point correspondences.

Estimates a rigid pose (`Pose3 = EuclideanMap{3,...}`) mapping world coordinates
to camera coordinates, such that `project(model, R*X + t) ≈ u`.

Type parameters:
- `S <: AbstractSampler`: Sampling strategy (uniform, PROSAC, etc.)
- `F`: Projection callable type (enables compiler inlining in `residuals!`)

# Constructor
```julia
# From 3D-2D correspondences and camera model
cs = csponds(world_points_3d, image_points_2d)
problem = P3PProblem(cs, camera_model)

# With scored correspondences → PROSAC
scored = [ScoredCspond(X, u, score) for ...]
problem = P3PProblem(scored, camera_model)
```

# Solver Details
- Minimal sample: 3 point correspondences (`MultipleSolutions`, 0-4 solutions)
- Model type: `Pose3` (EuclideanMap, `isbitstype`)
- Residual: Forward reprojection error `||project(model, R*X+t) - u||` in pixels
- Refinement: None (returns `nothing`)
"""
struct P3PProblem{S<:AbstractSampler, F} <: AbstractRansacProblem
    cs::StructArrays.StructVector{Pair{SVector{3,Float64},SVector{2,Float64}}, @NamedTuple{first::Vector{SVector{3,Float64}}, second::Vector{SVector{2,Float64}}}}
    rays::Vector{SVector{3,Float64}}
    _proj::F
    _sampler::S
end

"""
    P3PProblem(correspondences, model::CameraModel)

Construct a `P3PProblem` from 3D-2D correspondences and a camera model.

Accepts any correspondence type with `.first` (3D world point) and `.second`
(2D image point). Rays are precomputed via `backproject(model, u)` at
construction time for the P3P solver.
"""
function P3PProblem(correspondences::AbstractVector, model::CameraModel)
    n = length(correspondences)
    n >= 3 || throw(ArgumentError("Need at least 3 correspondences, got $n"))

    X = Vector{SVector{3,Float64}}(undef, n)
    u = Vector{SVector{2,Float64}}(undef, n)
    @inbounds for i in eachindex(correspondences)
        c = correspondences[i]
        X[i] = SVector{3,Float64}(c.first[1], c.first[2], c.first[3])
        u[i] = SVector{2,Float64}(c.second[1], c.second[2])
    end

    cs = StructArrays.StructArray{Pair{SVector{3,Float64},SVector{2,Float64}}}((X, u))
    rays = backproject.(Ref(model), u)
    proj = projection_transform(model)
    smplr = _build_sampler(correspondences, 3)

    P3PProblem{typeof(smplr), typeof(proj)}(cs, rays, proj, smplr)
end

sampler(p::P3PProblem) = p._sampler

# =============================================================================
# AbstractRansacProblem Interface
# =============================================================================

sample_size(::P3PProblem) = 3
codimension(::P3PProblem) = 2  # d_g = 2: two reprojection equations per point
data_size(p::P3PProblem) = length(p.cs)
model_type(::P3PProblem) = Pose3
solver_cardinality(::P3PProblem) = MultipleSolutions()

function solve(p::P3PProblem, idx::Vector{Int})
    @inbounds rays_sample = SVector(p.rays[idx[1]], p.rays[idx[2]], p.rays[idx[3]])
    X = p.cs.first
    @inbounds X_sample = SVector(X[idx[1]], X[idx[2]], X[idx[3]])

    # P3P solver can throw on degenerate configurations (e.g., collinear points)
    local Rs, ts
    try
        Rs, ts = p3p(rays_sample, X_sample)
    catch
        return nothing
    end
    n = length(Rs)
    n == 0 && return nothing

    # Stack-allocated container: zero heap allocation
    z = zero(Pose3)
    @inbounds p1 = n >= 1 ? EuclideanMap(RotMatrix{3,Float64}(Rs[1]), SVector{3,Float64}(ts[1])) : z
    @inbounds p2 = n >= 2 ? EuclideanMap(RotMatrix{3,Float64}(Rs[2]), SVector{3,Float64}(ts[2])) : z
    @inbounds p3 = n >= 3 ? EuclideanMap(RotMatrix{3,Float64}(Rs[3]), SVector{3,Float64}(ts[3])) : z
    @inbounds p4 = n >= 4 ? EuclideanMap(RotMatrix{3,Float64}(Rs[4]), SVector{3,Float64}(ts[4])) : z
    return FixedModels{4, Pose3}(n, (p1, p2, p3, p4))
end

function residuals!(r::Vector, p::P3PProblem, pose::Pose3)
    X = p.cs.first
    u = p.cs.second
    proj = p._proj
    R = pose.R
    t = pose.t
    @inbounds for i in eachindex(r, X, u)
        X_cam = R * X[i] + t
        if X_cam[3] <= 0.0
            r[i] = 1.0e8
            continue
        end
        u_proj = proj(X_cam)
        dx = u_proj[1] - u[i][1]
        dy = u_proj[2] - u[i][2]
        r[i] = sqrt(dx*dx + dy*dy)
    end
    return r
end

function test_model(::P3PProblem, pose::Pose3, ::Vector{Int})
    return all(isfinite, pose.t)
end

refine(::P3PProblem, ::Pose3, ::BitVector) = nothing
