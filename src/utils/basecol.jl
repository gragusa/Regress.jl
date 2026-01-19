#=============================================================================
# Block Matrix Construction Helpers
=============================================================================#

"""
    build_block_upper!(dest, blocks, sizes)

Build the upper triangular part of a block symmetric matrix in-place.
Only fills the upper triangle - lower triangle is undefined.

`blocks` should be a vector of matrices representing the upper triangular blocks:
For a 2x2 block matrix: [A11, A12, A22]
For a 3x3 block matrix: [A11, A12, A13, A22, A23, A33]

Returns `Symmetric(dest, :U)`.
"""
function build_block_upper!(dest::Matrix{T}, blocks::Vector, sizes::Vector{Int}) where {T}
    n_blocks = length(sizes)
    total = sum(sizes)
    @assert size(dest) == (total, total)

    # Fill blocks row by row (upper triangle only)
    block_idx = 1
    row_offset = 0
    for i in 1:n_blocks
        col_offset = row_offset
        for j in i:n_blocks
            block = blocks[block_idx]
            ni, nj = sizes[i], sizes[j]
            if i == j
                # Diagonal block
                @views dest[(row_offset + 1):(row_offset + ni), (col_offset + 1):(col_offset + nj)] .= block
            else
                # Off-diagonal block
                @views dest[(row_offset + 1):(row_offset + ni), (col_offset + 1):(col_offset + nj)] .= block
            end
            col_offset += nj
            block_idx += 1
        end
        row_offset += sizes[i]
    end

    return Symmetric(dest, :U)
end

"""
    build_block_symmetric(blocks, sizes) -> Symmetric

Allocate and build a block symmetric matrix from upper triangular blocks.
"""
function build_block_symmetric(blocks::Vector, sizes::Vector{Int})
    T = eltype(first(blocks))
    total = sum(sizes)
    dest = Matrix{T}(undef, total, total)
    return build_block_upper!(dest, blocks, sizes)
end

#=============================================================================
# Sweep Operator Implementation
=============================================================================#

# generalized 2inverse
#actually return minus the symmetric
function invsym!(X::Symmetric; has_intercept = false, setzeros = false, diagonal = 1:size(X, 2))
    tols = max.(diag(X), 1)
    buffer = zeros(size(X, 1))
    for j in diagonal
        d = X[j, j]
        if setzeros && abs(d) < tols[j] * sqrt(eps())
            X.data[1:j, j] .= 0
            X.data[j, (j + 1):end] .= 0
        else
            # used to mimic SAS; now similar to SweepOperators
            copy!(buffer, view(X, :, j))
            Symmetric(BLAS.syrk!('U', 'N', -1/d, buffer, one(eltype(X)), X.data))
            rmul!(buffer, 1 / d)
            @views copy!(X.data[1:(j - 1), j], buffer[1:(j - 1)])
            @views copy!(X.data[j, (j + 1):end], buffer[(j + 1):end])
            X[j, j] = - 1 / d
        end
        if setzeros && has_intercept && j == 1
            tols = max.(diag(X), 1)
        end
    end
    return X
end

## Returns base of X = [A B C ...]. Takes as input the matrix X'X (actuallyjust its right upper-triangular)
## Important: it must be the case that colinear are first columbs in the bsae in the order of columns
## that is [A B A] returns [true true false] not [false true true]
function basis!(XX::Symmetric; has_intercept = false)
    invXX = invsym!(XX; has_intercept = has_intercept, setzeros = true)
    return diag(invXX) .< 0
end

#=============================================================================
# Optimized Sweep Operator (SIMD-friendly, minimal allocation)
=============================================================================#

"""
    sweep_collinear!(A, tol) -> (basis, rank)

Optimized sweep operator for collinearity detection.
Sweeps the symmetric matrix A in-place and returns a BitVector indicating
which columns are in the basis (non-collinear).

Uses SIMD-optimized loops and minimal allocations.

After sweeping:
- Diagonal elements < 0 indicate swept (non-collinear) columns
- Diagonal elements = 0 indicate collinear columns
- The matrix contains -(X'X)^(-1) in the swept positions

# Arguments
- `A::Matrix{T}`: Upper triangular part of symmetric matrix (modified in-place)
- `tol::Real`: Tolerance for detecting near-zero pivots

# Returns
- `basis::BitVector`: true for non-collinear columns
- `rank::Int`: Number of non-collinear columns
"""
function sweep_collinear!(A::Matrix{T}, tol::Real = sqrt(eps(T))) where {T <: AbstractFloat}
    k = size(A, 1)
    @assert size(A, 2) == k "Matrix must be square"

    basis = trues(k)
    rank = k

    @inbounds for j in 1:k
        d = A[j, j]
        # Use abs(d) for consistency with sweep_solve!
        threshold = max(abs(d), one(T)) * tol

        if abs(d) < threshold
            # Column j is collinear - zero it out
            basis[j] = false
            rank -= 1
            # Zero the column and row
            for i in 1:j
                A[i, j] = zero(T)
            end
            for i in j:k
                A[j, i] = zero(T)
            end
        else
            # Sweep column j
            inv_d = one(T) / d

            # Update upper triangle: A[i,l] -= A[i,j] * A[j,l] / d
            # Split loops allow better SIMD optimization (fixed ranges)
            @inbounds for l in (j + 1):k
                A_jl = A[j, l]
                # Update column l, rows 1 to j-1
                @simd for i in 1:(j - 1)
                    A[i, l] -= A[i, j] * A_jl * inv_d
                end
                # Update diagonal and upper of remaining block
                @simd for i in j:l
                    A[i, l] -= A[i, j] * A_jl * inv_d
                end
            end

            # Update row j (becomes -A[j,:]/d after sweep)
            @simd for l in (j + 1):k
                A[j, l] *= inv_d
            end

            # Update column j above diagonal (becomes -A[:,j]/d)
            @simd for i in 1:(j - 1)
                A[i, j] *= inv_d
            end

            # Diagonal becomes -1/d
            A[j, j] = -inv_d
        end
    end

    return basis, rank
end

"""
    detect_collinearity_sweep(X; tol) -> (basis, X_reduced)

Detect collinear columns using the sweep operator on X'X.
More memory efficient than QR for large n, small k.

# Arguments
- `X::Matrix{T}`: Design matrix (n × k)
- `tol::Real`: Tolerance for detecting collinearity

# Returns
- `basis::BitVector`: Indicator of non-collinear columns
- `X_reduced::Matrix{T}`: Matrix with only non-collinear columns
"""
function detect_collinearity_sweep(X::Matrix{T}; tol::Real = 1e-8) where {T <:
                                                                          AbstractFloat}
    n, k = size(X)

    # Handle edge cases
    if k == 0
        return BitVector(), Matrix{T}(undef, n, 0)
    end

    # Compute X'X (only upper triangle needed)
    # Use BLAS syrk for efficiency
    XX = Matrix{T}(undef, k, k)
    BLAS.syrk!('U', 'T', one(T), X, zero(T), XX)

    # Sweep to detect collinearity
    basis, rank = sweep_collinear!(XX, T(tol))

    # Handle all-collinear case
    if rank == 0
        return falses(k), Matrix{T}(undef, n, 0)
    end

    # Return basis and reduced matrix
    X_reduced = X[:, basis]
    return basis, X_reduced
end

#solve X \ y. Take as input the matrix [X'X, X'y
#                                        y'X, y'y]
# (but only upper matters)
function ls_solve!(Xy::Symmetric, nx)
    if nx > 0
        invsym!(Xy, diagonal = 1:nx)
        return Xy[1:nx, (nx + 1):end]
    else
        return zeros(Float64, 0, size(Xy, 2) - nx)
    end
end
