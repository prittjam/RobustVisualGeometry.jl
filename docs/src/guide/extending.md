# Extending

## Implementing a Custom RANSAC Problem

To add a new geometry type to the RANSAC framework, implement a subtype of
`AbstractRansacProblem` with the required methods.

### Required Methods

```julia
struct MyProblem <: AbstractRansacProblem
    data::MyDataType
end

# Number of points needed for a minimal sample
sample_size(::MyProblem) = 3

# Total number of data points
data_size(p::MyProblem) = length(p.data)

# Type of the model being estimated
model_type(::MyProblem) = SMatrix{3,3,Float64,9}

# Codimension: number of independent constraint equations per point
codimension(::MyProblem) = 1

# Solve from a minimal sample (indices into data)
function solve(p::MyProblem, sample_indices::AbstractVector{Int})
    # Return a model or FixedModels{N,M}, or nothing on failure
end

# Compute residuals for all data points given a model
function residuals!(residuals::Vector{Float64}, p::MyProblem, model)
    for i in eachindex(residuals)
        residuals[i] = compute_distance(p.data[i], model)
    end
    return residuals
end
```

### Optional Methods

```julia
# Validate a minimal sample before solving (e.g., reject degenerate configs)
test_sample(p::MyProblem, indices::AbstractVector{Int}) = true

# Validate a model after solving (e.g., reject degenerate solutions)
# Receives the sample indices used to fit the model for point-specific checks
test_model(p::MyProblem, model, indices::AbstractVector{Int}) = true

# Validate model against its consensus (inlier) set
test_consensus(p::MyProblem, model, mask::BitVector) = true

# Weighted least-squares refit for LO-RANSAC
function fit(p::MyProblem, mask::BitVector, weights::AbstractVector, ::LinearFit)
    # Return refined model or nothing
end

# Number of solutions per sample (default: MultipleSolutions)
solver_cardinality(::MyProblem) = SingleSolution()
# Use MultipleSolutions() if solve() returns multiple models (e.g., P3P)

# Measurement covariance structure (default: Homoscedastic)
measurement_covariance(::MyProblem) = Homoscedastic()
```

### Usage

```julia
problem = MyProblem(data)
quality = MarginalQuality(problem, 50.0)
result = ransac(problem, quality)
```

## Implementing a Custom Robust Problem

For IRLS-based robust estimation, implement `AbstractRobustProblem`:

```julia
struct MyRobustProblem <: AbstractRobustProblem
    A::Matrix{Float64}
    b::Vector{Float64}
end

# Solve without weights (initial estimate)
initial_solve(p::MyRobustProblem) = p.A \ p.b

# Compute residuals given current estimate
compute_residuals!(r::Vector{Float64}, p::MyRobustProblem, x) = (r .= p.A * x .- p.b; r)

# Solve with diagonal weights
weighted_solve(p::MyRobustProblem, w::Vector{Float64}) = (W = Diagonal(w); (W * p.A) \ (W * p.b))

# Number of data points
data_size(p::MyRobustProblem) = length(p.b)

# Degrees of freedom
problem_dof(p::MyRobustProblem) = size(p.A, 2)
```

Then solve:

```julia
result = robust_solve(MyRobustProblem(A, b), MEstimator(CauchyLoss()))
```
