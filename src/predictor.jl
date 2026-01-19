"""
Predictor components for OLS estimation.

Similar to GLM.jl's DensePredChol and DensePredQR, these types store
the model matrix, coefficients, and factorizations for linear regression.

When `save=:minimal` is used, X and X_reduced are empty matrices to save memory.
Use `has_predictor_data(pp)` to check if matrices are available.
"""

abstract type OLSLinearPredictor{T <: AbstractFloat} end

"""
    OLSPredictorChol{T}

OLS predictor using Cholesky factorization of X'X.
Faster than QR but less numerically stable for ill-conditioned problems.

Fields:
- `X`: Full model matrix (for public API / post-estimation), empty if save=:minimal
- `X_reduced`: Non-collinear columns only, empty if save=:minimal
- `beta`: Full coefficient vector (with NaN for collinear)
- `chol`: Cholesky factorization of X_reduced'X_reduced
"""
mutable struct OLSPredictorChol{T <: AbstractFloat} <: OLSLinearPredictor{T}
    X::Matrix{T}                            # Full model matrix (empty if save=:minimal)
    X_reduced::Matrix{T}                    # Non-collinear columns only (empty if save=:minimal)
    beta::Vector{T}                         # Coefficient estimates (full, with NaN)
    chol::Cholesky{T, Matrix{T}}            # Cholesky factorization of X_reduced'X_reduced
end

"""
    OLSPredictorQR{T}

OLS predictor using QR factorization of X.
More numerically stable than Cholesky but about 2x slower.

Fields:
- `X`: Full model matrix (for public API / post-estimation), empty if save=:minimal
- `X_reduced`: Non-collinear columns only, empty if save=:minimal
- `beta`: Full coefficient vector (with NaN for collinear)
- `qr`: QR factorization of X_reduced
"""
mutable struct OLSPredictorQR{T <: AbstractFloat} <: OLSLinearPredictor{T}
    X::Matrix{T}                            # Full model matrix (empty if save=:minimal)
    X_reduced::Matrix{T}                    # Non-collinear columns only (empty if save=:minimal)
    beta::Vector{T}                         # Coefficient estimates (full, with NaN)
    qr::LinearAlgebra.QRCompactWY{T, Matrix{T}}   # QR factorization of X_reduced
end

"""
    has_predictor_data(pp::OLSLinearPredictor) -> Bool

Check if predictor has stored matrices (X and X_reduced).
Returns false when model was fit with save=:minimal.
"""
has_predictor_data(pp::OLSLinearPredictor) = !isempty(pp.X)

"""
    clear_predictor_data!(pp::OLSLinearPredictor{T}) where {T}

Clear predictor matrices (X and X_reduced) to save memory. Used for save=:minimal mode.
"""
function clear_predictor_data!(pp::OLSPredictorChol{T}) where {T}
    pp.X = Matrix{T}(undef, 0, 0)
    pp.X_reduced = Matrix{T}(undef, 0, 0)
    return pp
end

function clear_predictor_data!(pp::OLSPredictorQR{T}) where {T}
    pp.X = Matrix{T}(undef, 0, 0)
    pp.X_reduced = Matrix{T}(undef, 0, 0)
    return pp
end

"""
    linpred_rank(pp::OLSLinearPredictor) -> Int

Return the rank of the linear predictor (number of non-collinear coefficients).
"""
linpred_rank(pp::OLSLinearPredictor) = sum(.!isnan.(pp.beta))

"""
    invchol(pp::OLSPredictorChol) -> Symmetric

Compute (X'X)^(-1) from Cholesky factorization.
This is the "bread" of the sandwich variance estimator.
"""
function invchol(pp::OLSPredictorChol{T}) where {T}
    return Symmetric(inv(pp.chol))
end

"""
    invchol(pp::OLSPredictorQR) -> Symmetric

Compute (X'X)^(-1) from QR factorization.
Uses the relationship (X'X)^(-1) = (R'R)^(-1) = R^(-1) R^(-T).
"""
function invchol(pp::OLSPredictorQR{T}) where {T}
    R = pp.qr.R
    R_inv = inv(UpperTriangular(R))
    return Symmetric(R_inv * R_inv')
end

"""
    coefmatrix(pp::OLSLinearPredictor) -> Matrix

Return X'X for the predictor.
"""
function coefmatrix(pp::OLSPredictorChol{T}) where {T}
    return Matrix(pp.chol)
end

function coefmatrix(pp::OLSPredictorQR{T}) where {T}
    R = pp.qr.R
    return R' * R
end
