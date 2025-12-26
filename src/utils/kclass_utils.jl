##############################################################################
##
## K-Class Estimation Utilities
##
## Core algorithms for LIML, Fuller, and K-class IV estimators.
## Ported and extended from EndogenousLinearModelsEstimators.jl
##
##############################################################################

using LinearAlgebra

"""
    _qr_resid(A::AbstractMatrix, B::AbstractVecOrMat)

Compute residuals B | A using pivoted QR decomposition.
Returns resid(B|A) = B - A * (qr(A, ColumnNorm()) \\ B).

This is numerically more stable than B - A * inv(A'A) * A' * B.
"""
function _qr_resid(A::AbstractMatrix, B::AbstractVecOrMat)
    if size(A, 2) == 0
        return copy(B)
    end
    F = qr(A, ColumnNorm())  # Rank-revealing QR (pivoted)
    return B .- A * (F \ B)
end

"""
    _liml_kappa(y, Xendo, Z, Xexo) -> T

Compute LIML kappa using generalized eigenvalue problem M1 * v = kappa * M2 * v.

Supports multiple endogenous variables (Xendo can be n×k matrix).

# Arguments
- `y::AbstractVector{T}`: Response vector (n,)
- `Xendo::AbstractVecOrMat{T}`: Endogenous variables (n,) or (n × k_endo)
- `Z::AbstractMatrix{T}`: Excluded instruments (n × L)
- `Xexo::AbstractMatrix{T}`: Exogenous variables including intercept (n × k_exo)

# Returns
- `kappa::T`: Minimum eigenvalue of generalized eigenvalue problem

# Details
For k endogenous variables, solves the (k+1) × (k+1) eigenvalue problem:
- M1 = R'R where R = [y_adj, Xendo_adj] (adjusted for exogenous)
- M2 = R_res'R_res where R_res is R residualized on Z_adj
- kappa = min(eigenvalues of M1 * v = kappa * M2 * v)
"""
function _liml_kappa(
        y::AbstractVector{T},
        Xendo::AbstractVecOrMat{T},
        Z::AbstractMatrix{T},
        Xexo::AbstractMatrix{T}
) where {T}
    # Handle single endogenous as matrix
    Xendo_mat = Xendo isa AbstractVector ? reshape(Xendo, :, 1) : Xendo

    # Residualize on exogenous variables
    Yadj = _qr_resid(Xexo, y)
    Xendo_adj = _qr_resid(Xexo, Xendo_mat)
    Zadj = _qr_resid(Xexo, Z)

    # Stack [y, Xendo] into R matrix: n × (1 + k_endo)
    R = hcat(Yadj, Xendo_adj)

    # M1 = R'R (total variation after partialing out exogenous)
    M1 = Symmetric(R' * R)

    # M2 = R_res'R_res (residual variation after partialing out instruments)
    Rres = _qr_resid(Zadj, R)
    M2 = Symmetric(Rres' * Rres)

    # Solve generalized eigenvalue problem: M1 * v = kappa * M2 * v
    # LIML kappa is the smallest eigenvalue
    vals = try
        eigvals(M1, M2)
    catch
        # Add small regularization if M2 is singular/ill-conditioned
        tau = max(eps(T) * tr(M2), T(1e-12))
        eigvals(M1, Symmetric(Matrix(M2) + tau * I))
    end

    return T(minimum(real(vals)))
end

"""
    _kclass_fit(y, Xendo, Z, Xexo, kappa) -> (coef, residuals, invA, Adj)

Core K-class coefficient estimation.

The K-class estimator solves:
    β = [W'W - k*W'W_res]^(-1) * [W'y - k*W'y_res]

where:
- W = [Xendo, Xexo] (all regressors)
- W_res = residualize W on [Z, Xexo]
- y_res = residualize y on [Z, Xexo]

# Arguments
- `y::AbstractVector{T}`: Response vector (n,)
- `Xendo::AbstractVecOrMat{T}`: Endogenous variables (n,) or (n × k_endo)
- `Z::AbstractMatrix{T}`: Excluded instruments (n × L)
- `Xexo::AbstractMatrix{T}`: Exogenous variables (n × k_exo)
- `kappa::T`: K-class parameter

# Returns
- `coef::Vector{T}`: Coefficient vector (k_endo + k_exo,), endogenous coefficients first
- `residuals::Vector{T}`: y - W * coef
- `invA::Matrix{T}`: inv(W'W - k*W'W_res) - for vcov bread
- `Adj::Matrix{T}`: W - k*W_res - for vcov meat (K-class adjustment matrix)
"""
function _kclass_fit(
        y::AbstractVector{T},
        Xendo::AbstractVecOrMat{T},
        Z::AbstractMatrix{T},
        Xexo::AbstractMatrix{T},
        kappa::T
) where {T}
    # Handle single endogenous as matrix
    Xendo_mat = Xendo isa AbstractVector ? reshape(Xendo, :, 1) : Xendo

    # Full instrument set includes exogenous variables
    ZX = hcat(Z, Xexo)

    # Full regressor set: endogenous first, then exogenous
    W = hcat(Xendo_mat, Xexo)

    # Residualize W and y on full instrument set
    Wres = _qr_resid(ZX, W)
    yres = _qr_resid(ZX, y)

    # K-class normal equations
    # A = W'W - kappa * (W'Wres)
    # b = W'y - kappa * (W'yres)
    A = W' * W .- kappa .* (W' * Wres)
    b = W' * y .- kappa .* (W' * yres)

    # Solve for coefficients
    coef = A \ b

    # Residuals using original regressors
    residuals = y .- W * coef

    # Store for vcov calculation
    invA = inv(A)

    # Adjustment matrix for K-class variance: W - k*Wres
    Adj = W .- kappa .* Wres

    return coef, residuals, invA, Adj
end
