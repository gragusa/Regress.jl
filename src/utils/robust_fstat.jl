##############################################################################
##
## Robust F-statistic (Wald test) computation
##
##############################################################################

"""
    compute_robust_fstat(coef, vcov_matrix, has_intercept, dof_residual)

Compute a robust (Wald) F-statistic for testing all non-intercept coefficients = 0.

Uses the Wald test formula:
    F = (R*beta)' * inv(R * V * R') * (R*beta) / q

where R selects non-intercept coefficients, V is the vcov matrix, and q is the
number of restrictions (coefficients being tested).

# Arguments
- `coef::Vector{T}`: Coefficient vector
- `vcov_matrix::AbstractMatrix{T}`: Variance-covariance matrix
- `has_intercept::Bool`: Whether the model has an intercept (first coef)
- `dof_residual::Int`: Residual degrees of freedom for p-value computation

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
    dof_residual::Int
) where T <: AbstractFloat
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

    # p-value from F distribution with (q, dof_residual) degrees of freedom
    p = fdistccdf(q, dof_residual, F)

    return F, p
end

"""
    compute_robust_fstat(m::StatsAPI.RegressionModel, vcov_matrix::AbstractMatrix)

Compute robust F-statistic for a fitted regression model.

Extracts coefficients, intercept status, and degrees of freedom from the model.
"""
function compute_robust_fstat(m::StatsAPI.RegressionModel, vcov_matrix::AbstractMatrix{T}) where T
    cc = coef(m)
    has_int = hasintercept(formula(m))
    dof_res = dof_residual(m)
    return compute_robust_fstat(cc, vcov_matrix, has_int, dof_res)
end
