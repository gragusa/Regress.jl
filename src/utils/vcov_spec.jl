##############################################################################
##
## VcovSpec - Wrapper for vcov specification in model + vcov() syntax
##
##############################################################################

"""
    VcovSpec{V}

Wrapper type for variance estimator specification.
Used with `+` operator: `model + vcov(HC3())`

# Fields
- `estimator::V`: The variance estimator (HC0, CR1, etc.)

# Examples
```julia
model = ols(df, @formula(y ~ x))
model_hc3 = model + vcov(HC3())
model_cr1 = model + vcov(CR1(:cluster))
```
"""
struct VcovSpec{V <: CovarianceMatrices.AbstractAsymptoticVarianceEstimator}
    estimator::V
end

"""
    vcov(estimator::AbstractAsymptoticVarianceEstimator) -> VcovSpec

Create a VcovSpec for use with `model + vcov(...)` syntax.

This single-argument form wraps the variance estimator in a VcovSpec,
which can then be added to a fitted model using the `+` operator.

# Arguments
- `estimator`: A variance estimator from CovarianceMatrices.jl (HC0, HC1, CR1, etc.)

# Returns
- `VcovSpec{V}`: A wrapper containing the estimator

# Examples
```julia
# Create VcovSpec for heteroskedasticity-robust inference
v = vcov(HC3())

# Use with + operator
model = ols(df, @formula(y ~ x))
model_hc3 = model + vcov(HC3())

# Cluster-robust
model_cr = ols(df, @formula(y ~ x), save_cluster = :firm)
model_cr1 = model_cr + vcov(CR1(:firm))
```

See also: [`OLSEstimator`](@ref), [`IVEstimator`](@ref)
"""
StatsBase.vcov(v::CovarianceMatrices.AbstractAsymptoticVarianceEstimator) = VcovSpec(v)

# Show method for VcovSpec
function Base.show(io::IO, v::VcovSpec)
    print(io, "VcovSpec(", v.estimator, ")")
end
