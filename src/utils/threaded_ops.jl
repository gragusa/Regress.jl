#=============================================================================
# Threaded Operations with Val-Based Dispatch
#
# This module provides threaded implementations of common operations that
# can be enabled via the `method` parameter in OLS fitting functions.
#
# Operations dispatch on Val{method} to select between:
#   - Val{:cpu} - single-threaded (default)
#   - Val{:threaded} - multi-threaded
=============================================================================#

using LinearAlgebra: I, Symmetric, cholesky!, ldiv!, svd, diagm

#=============================================================================
# Weight Application
=============================================================================#

"""
    apply_weights!(y, X, sqrtw, method_val)

Apply square-root weights to response vector and design matrix in-place.

# Arguments
- `y::Vector{T}`: Response vector (modified in-place)
- `X::Matrix{T}`: Design matrix (modified in-place)
- `sqrtw::Vector{T}`: Square-root of weights
- `method_val::Val`: Val{:cpu} for single-threaded, Val{:threaded} for multi-threaded

# Returns
`nothing` (operates in-place)
"""
function apply_weights!(
        y::Vector{T},
        X::Matrix{T},
        sqrtw::AbstractVector,
        ::Val
) where {T}
    # Default (single-threaded): use fused broadcast
    y .= y .* sqrtw
    X .= X .* sqrtw
    return nothing
end

function apply_weights!(
        y::Vector{T},
        X::Matrix{T},
        sqrtw::AbstractVector,
        ::Val{:threaded}
) where {T}
    n, k = size(X)

    # Parallelize over columns (including y as "column 0")
    Threads.@threads for col in 0:k
        if col == 0
            @inbounds @simd for i in 1:n
                y[i] *= sqrtw[i]
            end
        else
            @inbounds @simd for i in 1:n
                X[i, col] *= sqrtw[i]
            end
        end
    end
    return nothing
end

#=============================================================================
# Cluster-Robust Variance Computation (CR2/CR3)
#
# These operations iterate over clusters independently and can be parallelized.
=============================================================================#

"""
    compute_cluster_adjustments_cr2!(u, X, XX, g, perm, starts, wts, method_val)

Compute CR2 leverage adjustments for each cluster in-place.

# Arguments
- `u::Vector{T}`: Residual vector (modified in-place)
- `X::Matrix{T}`: Model matrix
- `XX::AbstractMatrix{T}`: Bread matrix (X'X)^(-1)
- `g`: Grouping from CovarianceMatrices
- `perm::Vector{Int}`: Permutation array for group indices
- `starts::Vector{Int}`: Start positions for each group
- `wts`: Weights (empty if unweighted)
- `method_val::Val`: Val{:cpu} or Val{:threaded}

# Returns
`nothing` (operates in-place on `u`)
"""
function compute_cluster_adjustments_cr2!(
        u::Vector{T},
        X::Matrix{T},
        XX::AbstractMatrix{T},
        g,
        perm::Vector{Int},
        starts::Vector{Int},
        wts,
        ::Val
) where {T}
    # Sequential version
    for group_id in 1:g.ngroups
        _apply_cr2_to_group!(u, X, XX, perm, starts, group_id, wts)
    end
    return nothing
end

function compute_cluster_adjustments_cr2!(
        u::Vector{T},
        X::Matrix{T},
        XX::AbstractMatrix{T},
        g,
        perm::Vector{Int},
        starts::Vector{Int},
        wts,
        ::Val{:threaded}
) where {T}
    # Parallel version: each cluster is independent
    Threads.@threads for group_id in 1:g.ngroups
        _apply_cr2_to_group!(u, X, XX, perm, starts, group_id, wts)
    end
    return nothing
end

"""
Internal: Apply CR2 adjustment to a single cluster.
"""
function _apply_cr2_to_group!(
        u::Vector{T},
        X::Matrix{T},
        XX::AbstractMatrix{T},
        perm::Vector{Int},
        starts::Vector{Int},
        group_id::Int,
        wts
) where {T}
    ind = @view perm[starts[group_id]:(starts[group_id + 1] - 1)]
    Xg = view(X, ind, :)
    ug = view(u, ind)

    if isempty(wts)
        Hgg = Xg * XX * Xg'
        # CR2: (I - H)^(-1/2) * u
        # Use Cholesky of (I - H) to solve, which gives (I - H)^(-1/2) * u
        ldiv!(ug, cholesky!(Symmetric(I - Hgg); check = false).L, ug)
    else
        Hgg = (Xg * XX * Xg') .* view(wts, ind)'
        # For weighted case, use SVD-based matrix power
        ug .= _matrixpow_neg_half(I - Hgg) * ug
    end
    return nothing
end

"""
    compute_cluster_adjustments_cr3!(u, X, XX, g, perm, starts, wts, method_val)

Compute CR3 leverage adjustments for each cluster in-place.

Similar to CR2 but uses (I - H)^(-1) instead of (I - H)^(-1/2).
"""
function compute_cluster_adjustments_cr3!(
        u::Vector{T},
        X::Matrix{T},
        XX::AbstractMatrix{T},
        g,
        perm::Vector{Int},
        starts::Vector{Int},
        wts,
        ::Val
) where {T}
    # Sequential version
    for group_id in 1:g.ngroups
        _apply_cr3_to_group!(u, X, XX, perm, starts, group_id, wts)
    end
    return nothing
end

function compute_cluster_adjustments_cr3!(
        u::Vector{T},
        X::Matrix{T},
        XX::AbstractMatrix{T},
        g,
        perm::Vector{Int},
        starts::Vector{Int},
        wts,
        ::Val{:threaded}
) where {T}
    # Parallel version: each cluster is independent
    Threads.@threads for group_id in 1:g.ngroups
        _apply_cr3_to_group!(u, X, XX, perm, starts, group_id, wts)
    end
    return nothing
end

"""
Internal: Apply CR3 adjustment to a single cluster.
"""
function _apply_cr3_to_group!(
        u::Vector{T},
        X::Matrix{T},
        XX::AbstractMatrix{T},
        perm::Vector{Int},
        starts::Vector{Int},
        group_id::Int,
        wts
) where {T}
    ind = @view perm[starts[group_id]:(starts[group_id + 1] - 1)]
    Xg = view(X, ind, :)
    ug = view(u, ind)

    if isempty(wts)
        Hgg = Xg * XX * Xg'
        # CR3: (I - H)^(-1) * u
        ldiv!(ug, cholesky!(Symmetric(I - Hgg); check = false), ug)
    else
        Hgg = (Xg * XX * Xg') .* view(wts, ind)'
        # For weighted case, direct inverse
        ug .= (I - Hgg) \ ug
    end
    return nothing
end

"""
Internal: Compute A^(-1/2) via SVD for weighted CR2.
"""
function _matrixpow_neg_half(A; tol = eps()^(1 / 1.5))
    s = svd(A)
    V = s.S
    V[V .< tol] .= 0
    return s.V * diagm(0 => V .^ (-0.5)) * s.Vt
end
