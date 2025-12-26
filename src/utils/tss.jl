#=============================================================================
# Total Sum of Squares (TSS) Computation
=============================================================================#

"""
    tss(y, hasintercept, weights) -> T

Compute the total sum of squares.

If `hasintercept=true`, computes Σ(y - ȳ)² * w (centered TSS).
If `hasintercept=false`, computes Σy² * w (uncentered TSS).

Uses SIMD-optimized loops for performance.
"""
function tss(y::AbstractVector, hasintercept::Bool, weights::AbstractWeights)
    T = eltype(y)
    if hasintercept
        m = T(mean(y, weights))  # Ensure mean is same type as y
        return _tss_centered(y, m, weights)
    else
        return _tss_uncentered(y, weights)
    end
end

"""
Internal SIMD-optimized centered TSS computation.
"""
function _tss_centered(y::AbstractVector, m, weights::AbstractWeights)
    T = eltype(y)
    s = zero(T)
    @inbounds @simd for i in eachindex(y)
        s += (y[i] - m)^2 * weights[i]
    end
    return s
end

"""
Internal SIMD-optimized uncentered TSS computation.
"""
function _tss_uncentered(y::AbstractVector, weights::AbstractWeights)
    T = eltype(y)
    s = zero(T)
    @inbounds @simd for i in eachindex(y)
        s += y[i]^2 * weights[i]
    end
    return s
end

# Convenience wrapper that assumes intercept
function tss(y::AbstractVector, weights::AbstractWeights)
    tss(y, true, weights)
end

# Convenience wrapper for compute_tss (used in fit_common.jl)
function compute_tss(y::AbstractVector, weights::AbstractWeights, hasintercept::Bool)
    tss(y, hasintercept, weights)
end

#=============================================================================
# F-Statistic Computation
=============================================================================#

"""
    Fstat(coef, matrix_vcov, has_intercept) -> Float64

Compute the F-statistic for testing joint significance of coefficients.
"""
function Fstat(coef::Vector{Float64}, matrix_vcov::AbstractMatrix{Float64}, has_intercept::Bool)
    coefF = copy(coef)
    length(coef) == has_intercept && return NaN
    if has_intercept
        coefF = coefF[2:end]
        matrix_vcov = matrix_vcov[2:end, 2:end]
    end
    try
        return (coefF' * (matrix_vcov \ coefF)) / length(coefF)
    catch
        @info "The variance-covariance matrix is not invertible. F-statistic not computed"
        return NaN
    end
end
