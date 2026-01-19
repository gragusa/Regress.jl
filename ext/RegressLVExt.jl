"""
    RegressLVExt

LoopVectorization extension for Regress.jl.

This extension provides @turbo-optimized implementations for hot loops:
- compute_rss: Residual sum of squares
- _tss_centered: Centered total sum of squares
- _tss_uncentered: Uncentered total sum of squares
- _compute_moment_matrix!: Element-wise M = X .* (residuals * adjustment)
"""
module RegressLVExt

using Regress
using LoopVectorization: @turbo
using StatsBase: AbstractWeights, Weights, UnitWeights

#=============================================================================
# RSS Computation with @turbo
=============================================================================#

"""
    compute_rss_turbo(y, mu) -> T

Compute residual sum of squares using @turbo.
"""
function Regress.compute_rss(y::Vector{T}, mu::Vector{T}) where {T <: Union{Float32, Float64}}
    rss = zero(T)
    @turbo for i in eachindex(y, mu)
        rss += (y[i] - mu[i])^2
    end
    return rss
end

#=============================================================================
# TSS Computation with @turbo
=============================================================================#

"""
    _tss_centered with @turbo optimization for Weights (has .values field).
"""
function Regress._tss_centered(y::Vector{T}, m::T, weights::Weights) where {T <: Union{Float32, Float64}}
    s = zero(T)
    w = weights.values
    @turbo for i in eachindex(y, w)
        s += (y[i] - m)^2 * w[i]
    end
    return s
end

"""
    _tss_centered with @turbo optimization for UnitWeights (no .values field).
"""
function Regress._tss_centered(y::Vector{T}, m::T, weights::UnitWeights) where {T <: Union{Float32, Float64}}
    s = zero(T)
    @turbo for i in eachindex(y)
        s += (y[i] - m)^2
    end
    return s
end

"""
    _tss_uncentered with @turbo optimization for Weights.
"""
function Regress._tss_uncentered(y::Vector{T}, weights::Weights) where {T <: Union{Float32, Float64}}
    s = zero(T)
    w = weights.values
    @turbo for i in eachindex(y, w)
        s += y[i]^2 * w[i]
    end
    return s
end

"""
    _tss_uncentered with @turbo optimization for UnitWeights.
"""
function Regress._tss_uncentered(y::Vector{T}, weights::UnitWeights) where {T <: Union{Float32, Float64}}
    s = zero(T)
    @turbo for i in eachindex(y)
        s += y[i]^2
    end
    return s
end

#=============================================================================
# Moment Matrix Computation with @turbo
=============================================================================#

"""
    compute_moment_matrix_turbo!(M, X, residuals, adjustment)

Compute M = X .* (residuals * adjustment) in-place using @turbo.
This is used in HC1 variance computation.
"""
function Regress.compute_moment_matrix!(M::Matrix{T}, X::Matrix{T},
                                        residuals::Vector{T}, adjustment::T) where {T <: Union{Float32, Float64}}
    n, k = size(X)
    @turbo for j in 1:k
        for i in 1:n
            M[i, j] = X[i, j] * residuals[i] * adjustment
        end
    end
    return M
end

function __init__()
    @info "RegressLVExt: LoopVectorization optimizations enabled"
end

end # module
