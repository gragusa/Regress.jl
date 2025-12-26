##############################################################################
##
## Deep Copy Utilities for Variance-Covariance Estimators
##
## HAC estimators store bandwidth info and wlock that must not be shared.
## CR estimators store cluster vectors that must be copied.
##
##############################################################################

const CM = CovarianceMatrices

"""
    deepcopy_vcov(v::AbstractAsymptoticVarianceEstimator)

Deep copy a variance-covariance estimator to avoid aliasing issues.

HAC estimators (Bartlett, Parzen, etc.) store mutable bandwidth/weight arrays
that must not be shared between models. CR estimators store cluster vectors
that must also be independent.

# Examples
```julia
bart = Bartlett(4)
bart_copy = deepcopy_vcov(bart)
bart_copy.bw[1] = 10  # Does not affect original
```
"""
deepcopy_vcov(v::CM.AbstractAsymptoticVarianceEstimator) = deepcopy(v)

# HR estimators (HC0-HC5 are aliases for HR0-HR5, stateless singletons)
deepcopy_vcov(v::CM.HR0) = v
deepcopy_vcov(v::CM.HR1) = v
deepcopy_vcov(v::CM.HR2) = v
deepcopy_vcov(v::CM.HR3) = v
deepcopy_vcov(v::CM.HR4) = v
deepcopy_vcov(v::CM.HR4m) = v
deepcopy_vcov(v::CM.HR5) = v

# CR estimators with symbols (immutable) - can return same instance
deepcopy_vcov(v::CM.CR0{T}) where {T <: Tuple{Vararg{Symbol}}} = v
deepcopy_vcov(v::CM.CR1{T}) where {T <: Tuple{Vararg{Symbol}}} = v
deepcopy_vcov(v::CM.CR2{T}) where {T <: Tuple{Vararg{Symbol}}} = v
deepcopy_vcov(v::CM.CR3{T}) where {T <: Tuple{Vararg{Symbol}}} = v

# CR estimators with data vectors - deep copy the vectors
function deepcopy_vcov(v::CM.CR0{T}) where {T <: Tuple}
    CM.CR0(Tuple(copy(vec) for vec in v.g))
end
function deepcopy_vcov(v::CM.CR1{T}) where {T <: Tuple}
    CM.CR1(Tuple(copy(vec) for vec in v.g))
end
function deepcopy_vcov(v::CM.CR2{T}) where {T <: Tuple}
    CM.CR2(Tuple(copy(vec) for vec in v.g))
end
function deepcopy_vcov(v::CM.CR3{T}) where {T <: Tuple}
    CM.CR3(Tuple(copy(vec) for vec in v.g))
end

# HAC estimators - use deepcopy (copies bw, kw, wlock arrays)
# The fallback deepcopy handles these correctly
