"""
Core OLS solver utilities for Regress.jl

This file contains functions for:
- Collinearity detection via QR with column pivoting
- Unified fit_ols_core! that builds predictor and solves in one pass
- NaN expansion for rank-deficient cases
"""

using LinearAlgebra
using LinearAlgebra: BlasReal

#=============================================================================
# Collinearity Detection
=============================================================================#

"""
    detect_collinearity(X; tol, method) -> (basis, X_reduced)

Detect collinear columns in design matrix X.

# Arguments
- `X::Matrix{T}`: Design matrix
- `tol::Real=1e-8`: Tolerance for detecting near-zero pivots
- `method::Symbol=:qr`: Detection method - `:qr` or `:sweep`
  - `:qr`: QR with column pivoting (more stable, higher memory for large n)
  - `:sweep`: Sweep operator on X'X (less stable, lower memory, faster for large n)

# Returns
- `basis::BitVector`: Indicator of non-collinear columns
- `X_reduced::Matrix{T}`: Matrix with only non-collinear columns
"""
function detect_collinearity(X::Matrix{T}; tol::Real = 1e-8, method::Symbol = :qr) where {T <:
                                                                                          AbstractFloat}
    if method == :qr
        return detect_collinearity_qr(X; tol = tol)
    elseif method == :sweep
        return detect_collinearity_sweep(X; tol = tol)
    else
        throw(ArgumentError("collinearity method must be :qr or :sweep, got :$method"))
    end
end

"""
    detect_collinearity_qr(X; tol) -> (basis, X_reduced)

Detect collinear columns using QR with column pivoting.
More numerically stable but uses more memory for large n.

# Arguments
- `X::Matrix{T}`: Design matrix
- `tol::Real=1e-8`: Tolerance for detecting near-zero pivots

# Returns
- `basis::BitVector`: Indicator of non-collinear columns
- `X_reduced::Matrix{T}`: Matrix with only non-collinear columns
"""
function detect_collinearity_qr(X::Matrix{T}; tol::Real = 1e-8) where {T <: AbstractFloat}
    n, k = size(X)

    # Handle edge case: no columns
    if k == 0
        return BitVector(), Matrix{T}(undef, n, 0)
    end

    # QR with column pivoting handles zero columns automatically
    F = qr(X, ColumnNorm())

    # Check diagonal of R for near-zero elements
    R_diag = abs.(diag(F.R))
    max_diag = maximum(R_diag)

    # Handle edge case: all zeros
    if max_diag == zero(T)
        return falses(k), Matrix{T}(undef, n, 0)
    end

    threshold = max_diag * tol

    # Find rank
    r = sum(R_diag .> threshold)

    # Build basis indicator
    basis = trues(k)
    if r < k
        # Mark columns beyond rank as collinear (in permuted order)
        for j in (r + 1):k
            basis[F.p[j]] = false
        end
    end

    # Return basis and reduced matrix (subset only once)
    X_reduced = X[:, basis]
    return basis, X_reduced
end

#=============================================================================
# Coefficient Expansion Utilities
=============================================================================#

"""
    expand_coef_nan(coef_reduced, basis) -> Vector

Expand reduced coefficient vector to full size with NaN for collinear columns.
"""
function expand_coef_nan(coef_reduced::Vector{T}, basis::BitVector) where {T}
    k = length(basis)
    coef_full = fill(T(NaN), k)
    coef_full[basis] = coef_reduced
    return coef_full
end

"""
    expand_vcov_nan(vcov_reduced, basis) -> Symmetric

Expand reduced vcov matrix to full size with NaN for collinear columns.
"""
function expand_vcov_nan(vcov_reduced::Symmetric{T}, basis::BitVector) where {T}
    k = length(basis)
    vcov_full = fill(T(NaN), k, k)
    vcov_full[basis, basis] = Matrix(vcov_reduced)
    return Symmetric(vcov_full)
end

#=============================================================================
# BLAS-Optimized Cross-Product
=============================================================================#

"""
    compute_crossproduct(X) -> Symmetric

Compute X'X efficiently using BLAS syrk (symmetric rank-k update).
"""
function compute_crossproduct(X::Matrix{T}) where {T <: BlasReal}
    k = size(X, 2)
    XX = Matrix{T}(undef, k, k)
    BLAS.syrk!('U', 'T', one(T), X, zero(T), XX)
    return Symmetric(XX, :U)
end

# Fallback for non-BLAS types
function compute_crossproduct(X::Matrix{T}) where {T}
    return Symmetric(X' * X)
end

#=============================================================================
# Unified OLS Solver
=============================================================================#

"""
    fit_ols_core!(rr, X, factorization; tol, save_matrices, collinearity) -> (pp, basis_coef, beta_reduced)

Unified OLS solver that detects collinearity, builds predictor, and solves in one pass.

This replaces the separate `build_predictor` + `solve_ols!` pattern to avoid
redundant matrix subsetting and coefficient solving.

# Arguments
- `rr::OLSResponse{T}`: Response object (y will be used, mu will be updated)
- `X::Matrix{T}`: Full design matrix
- `factorization::Symbol`: `:chol` or `:qr`
- `tol::Real=1e-8`: Collinearity detection tolerance
- `save_matrices::Bool=true`: Whether to store X and X_reduced in predictor
- `collinearity::Symbol=:qr`: Collinearity detection method (`:qr` or `:sweep`)

# Returns
- `pp::OLSLinearPredictor{T}`: Predictor with factorization (X/X_reduced may be nothing)
- `basis_coef::BitVector`: Indicator of non-collinear coefficients
- `beta_reduced::Vector{T}`: Coefficients for non-collinear columns only
"""
function fit_ols_core!(rr::OLSResponse{T}, X::Matrix{T},
        factorization::Symbol;
        tol::Real = 1e-8,
        save_matrices::Bool = true,
        collinearity::Symbol = :qr) where {T <: AbstractFloat}

    # Step 1: Detect collinearity and get reduced X (done once)
    basis_coef, X_reduced = detect_collinearity(X; tol = tol, method = collinearity)

    # Step 2: Solve based on factorization method
    if factorization == :chol
        pp, beta_reduced = _fit_cholesky!(rr, X, X_reduced, basis_coef, save_matrices)
    elseif factorization == :qr
        pp, beta_reduced = _fit_qr!(rr, X, X_reduced, basis_coef, save_matrices)
    else
        error("factorization must be :chol or :qr, got :$factorization")
    end

    return pp, basis_coef, beta_reduced
end

"""
Internal Cholesky-based OLS solver.
"""
function _fit_cholesky!(rr::OLSResponse{T}, X::Matrix{T},
        X_reduced::Matrix{T}, basis_coef::BitVector,
        save_matrices::Bool) where {T}

    # Compute X'X using BLAS syrk (more efficient than X_reduced' * X_reduced)
    XX = compute_crossproduct(X_reduced)
    chol_fact = cholesky(XX)

    # Solve: β = (X'X)^(-1) X'y
    Xy = X_reduced' * rr.y
    beta_reduced = chol_fact \ Xy

    # Expand coefficients with NaN for collinear columns
    beta = expand_coef_nan(beta_reduced, basis_coef)

    # Compute fitted values: mu = X_reduced * beta_reduced
    mul!(rr.mu, X_reduced, beta_reduced)

    # Build predictor (conditionally store X and X_reduced)
    if save_matrices
        pp = OLSPredictorChol(X, X_reduced, beta, chol_fact)
    else
        pp = OLSPredictorChol(nothing, nothing, beta, chol_fact)
    end

    return pp, beta_reduced
end

"""
Internal QR-based OLS solver.
"""
function _fit_qr!(rr::OLSResponse{T}, X::Matrix{T},
        X_reduced::Matrix{T}, basis_coef::BitVector,
        save_matrices::Bool) where {T}

    # QR factorization
    qr_fact = qr(X_reduced)

    # Solve: β = R^(-1) Q'y
    beta_reduced = qr_fact \ rr.y

    # Expand coefficients with NaN for collinear columns
    beta = expand_coef_nan(beta_reduced, basis_coef)

    # Compute fitted values
    mul!(rr.mu, X_reduced, beta_reduced)

    # Build predictor (conditionally store X and X_reduced)
    if save_matrices
        pp = OLSPredictorQR(X, X_reduced, beta, qr_fact)
    else
        pp = OLSPredictorQR(nothing, nothing, beta, qr_fact)
    end

    return pp, beta_reduced
end

#=============================================================================
# Legacy Interface (for backward compatibility)
=============================================================================#

"""
    build_predictor(X, y, factorization, has_intercept) -> (predictor, basis_coef)

Build predictor object with factorization and detect collinearity.
This is a legacy interface - prefer `fit_ols_core!` for new code.
"""
function build_predictor(X::Matrix{T}, y::Vector{T},
        factorization::Symbol,
        has_intercept::Bool) where {T <: AbstractFloat}
    # Create temporary response for unified interface
    rr = OLSResponse(y, zeros(T, length(y)), T[], Symbol("y"))
    pp, basis_coef, _ = fit_ols_core!(rr, X, factorization)
    return pp, basis_coef
end

"""
    solve_ols!(rr, pp, basis_coef)

Solve OLS problem - now just computes fitted values since coefficients
are already computed in fit_ols_core!.

This is a legacy interface for backward compatibility.
Note: Requires X_reduced and mu to be stored (not compatible with save=:minimal).
"""
function solve_ols!(rr::OLSResponse{T},
        pp::OLSPredictorChol{T},
        basis_coef::BitVector) where {T}
    pp.X_reduced === nothing &&
        error("X_reduced not stored. Model was fit with save=:minimal.")
    rr.mu === nothing &&
        error("Fitted values not stored. Model was fit with save=:minimal.")
    # Coefficients already solved - just recompute fitted values if needed
    beta_reduced = pp.beta[basis_coef]
    mul!(rr.mu, pp.X_reduced, beta_reduced)
    return pp.beta
end

function solve_ols!(rr::OLSResponse{T},
        pp::OLSPredictorQR{T},
        basis_coef::BitVector) where {T}
    pp.X_reduced === nothing &&
        error("X_reduced not stored. Model was fit with save=:minimal.")
    rr.mu === nothing &&
        error("Fitted values not stored. Model was fit with save=:minimal.")
    beta_reduced = pp.beta[basis_coef]
    mul!(rr.mu, pp.X_reduced, beta_reduced)
    return pp.beta
end

#=============================================================================
# Residual Computation
=============================================================================#

"""
    compute_rss(y, mu) -> T

Compute residual sum of squares without allocating a residuals vector.
"""
function compute_rss(y::AbstractVector{T}, mu::AbstractVector{T}) where {T}
    rss = zero(T)
    @inbounds @simd for i in eachindex(y, mu)
        rss += (y[i] - mu[i])^2
    end
    return rss
end

"""
    compute_residuals!(residuals, y, mu) -> residuals

Compute residuals in-place: residuals = y - mu
"""
function compute_residuals!(residuals::Vector{T}, y::Vector{T}, mu::Vector{T}) where {T}
    @inbounds @simd for i in eachindex(residuals, y, mu)
        residuals[i] = y[i] - mu[i]
    end
    return residuals
end
