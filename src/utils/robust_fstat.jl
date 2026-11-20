##############################################################################
##
## Robust F-statistic (Wald test) computation
##
##############################################################################

"""
    compute_robust_fstat(coef, vcov_matrix, has_intercept, dof)

Compute a robust (Wald) F-statistic for testing all non-intercept coefficients = 0.

Uses the Wald test formula:
    F = (R*beta)' * inv(R * V * R') * (R*beta) / q

where R selects non-intercept coefficients, V is the vcov matrix, and q is the
number of restrictions (coefficients being tested).

# Arguments
- `coef::Vector{T}`: Coefficient vector
- `vcov_matrix::AbstractMatrix{T}`: Variance-covariance matrix
- `has_intercept::Bool`: Whether the model has an intercept (first coef)
- `dof::Int`: Denominator degrees of freedom of the F distribution of the p-value:
  `G - 1` for a cluster-robust `vcov_matrix`, the residual degrees of freedom otherwise
  (see [`inference_dof`](@ref))

# Returns
- `(F, p)`: Tuple of F-statistic and p-value

# Notes
- Coefficients with NaN values are excluded from the test
- Returns (NaN, NaN) if no coefficients can be tested or if vcov is singular
"""
function compute_robust_fstat(
        coef::Vector{T},
        vcov_matrix::AbstractMatrix{T},
        has_intercept::Bool,
        dof::Int
) where {T <: AbstractFloat}
    k = length(coef)

    # Identify coefficients to test (all non-intercept, non-NaN)
    test_idx = if has_intercept
        findall(i -> i > 1 && !isnan(coef[i]), 1:k)
    else
        findall(i -> !isnan(coef[i]), 1:k)
    end

    q = length(test_idx)
    q == 0 && return T(NaN), T(NaN)

    # Extract relevant coefficients and vcov submatrix
    beta_test = coef[test_idx]
    V_test = vcov_matrix[test_idx, test_idx]

    # Check for valid vcov submatrix
    if any(isnan, V_test) || any(isinf, V_test)
        return T(NaN), T(NaN)
    end

    # Wald statistic: chi2 = beta' * inv(V) * beta
    # F = chi2 / q
    V_chol = cholesky(Symmetric(V_test); check = false)
    if !issuccess(V_chol)
        return T(NaN), T(NaN)
    end

    chi2 = beta_test' * (V_chol \ beta_test)
    F = chi2 / q

    # p-value from F distribution with (q, dof) degrees of freedom
    p = fdistccdf(q, dof, F)

    return F, p
end

"""
    compute_robust_fstat(m::StatsAPI.RegressionModel, vcov_matrix::AbstractMatrix)

Compute robust F-statistic for a fitted regression model.

Extracts coefficients and intercept status from the model; the degrees of freedom are
those of tests computed with `vcov_matrix` (see [`inference_dof`](@ref)).
"""
function compute_robust_fstat(m::StatsAPI.RegressionModel, vcov_matrix::AbstractMatrix{T}) where {T}
    cc = coef(m)
    has_int = hasintercept(formula(m))
    return compute_robust_fstat(cc, vcov_matrix, has_int, _test_dof(m, vcov_matrix))
end
