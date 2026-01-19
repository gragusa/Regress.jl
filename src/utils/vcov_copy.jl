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

# CR estimators with data vectors/Clustering - deep copy the vectors
# CovarianceMatrices.jl wraps vectors in Clustering(groups, ngroups)
_copy_cluster_element(vec::AbstractVector) = copy(vec)
_copy_cluster_element(c::CM.Clustering) = CM.Clustering(copy(c.groups), c.ngroups)

function deepcopy_vcov(v::CM.CR0{T}) where {T <: Tuple}
    CM.CR0(Tuple(_copy_cluster_element(g) for g in v.g))
end
function deepcopy_vcov(v::CM.CR1{T}) where {T <: Tuple}
    CM.CR1(Tuple(_copy_cluster_element(g) for g in v.g))
end
function deepcopy_vcov(v::CM.CR2{T}) where {T <: Tuple}
    CM.CR2(Tuple(_copy_cluster_element(g) for g in v.g))
end
function deepcopy_vcov(v::CM.CR3{T}) where {T <: Tuple}
    CM.CR3(Tuple(_copy_cluster_element(g) for g in v.g))
end

# CachedCR estimators
# CachedCR has preallocated buffers that can be reused - return same instance
# The cache is designed to be reused for repeated variance calculations (e.g., wild bootstrap)
# Note: CachedCR in Regress.jl is always created with actual data vectors (symbols are resolved)
deepcopy_vcov(v::CM.CachedCR) = v

# HAC estimators - use deepcopy (copies bw, kw, wlock arrays)
# The fallback deepcopy handles these correctly
