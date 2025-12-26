##############################################################################
##
## IV Estimator Types
##
##############################################################################

"""
    AbstractIVEstimator

Abstract type for instrumental variables estimators.

Concrete subtypes include:
- `TSLS`: Two-Stage Least Squares (k = 1)
- `LIML`: Limited Information Maximum Likelihood (k = k_LIML)
- `Fuller`: Fuller bias-corrected estimator (k = k_LIML - a/(n-L-p))
- `KClass`: Generic K-class estimator with user-specified k

All estimators are used via the `iv()` function:
```julia
iv(TSLS(), df, @formula(y ~ x + (endo ~ instrument)))
iv(LIML(), df, @formula(y ~ x + (endo ~ instrument)))
iv(Fuller(), df, @formula(y ~ x + (endo ~ instrument)))
iv(Fuller(4.0), df, @formula(y ~ x + (endo ~ instrument)))
iv(KClass(0.5), df, @formula(y ~ x + (endo ~ instrument)))
```
"""
abstract type AbstractIVEstimator end

"""
    TSLS <: AbstractIVEstimator

Two-Stage Least Squares (2SLS) estimator for instrumental variables models.

This is the most common IV estimator, which proceeds in two stages:
1. First stage: Regress endogenous variables on instruments and exogenous variables
2. Second stage: Regress outcome on predicted endogenous variables and exogenous variables

# Usage
```julia
iv(TSLS(), df, @formula(y ~ x + (endo ~ instrument)))
```
"""
struct TSLS <: AbstractIVEstimator end

"""
    LIML <: AbstractIVEstimator

Limited Information Maximum Likelihood estimator for instrumental variables models.

LIML is an alternative to 2SLS that can have better finite-sample properties,
especially when instruments are weak. It uses k = k_LIML, the minimum eigenvalue
of the generalized eigenvalue problem M1 * v = k * M2 * v.

# Usage
```julia
iv(LIML(), df, @formula(y ~ x + (endo ~ instrument)))
```

# References
Anderson, T. W. and Rubin, H. (1949). Estimation of the parameters of a single
equation in a complete system of stochastic equations. Annals of Mathematical
Statistics, 20(1):46-63.
"""
struct LIML <: AbstractIVEstimator end

"""
    Fuller <: AbstractIVEstimator
    Fuller(a::Real = 1.0)

Fuller bias-corrected estimator for instrumental variables models.

Uses k = k_LIML - a/(n - L - p), where:
- k_LIML is the LIML kappa
- n is the sample size
- L is the number of excluded instruments
- p is the number of exogenous variables (including intercept)

Fuller(1) is approximately median-unbiased and has good finite-sample properties.
Fuller(4) minimizes mean squared error under certain conditions.

# Usage
```julia
iv(Fuller(), df, @formula(y ~ x + (endo ~ instrument)))     # a = 1.0 (default)
iv(Fuller(4.0), df, @formula(y ~ x + (endo ~ instrument)))  # a = 4.0
```

# References
Fuller, W. A. (1977). Some properties of a modification of the limited
information estimator. Econometrica, 45(4):939-953.
"""
struct Fuller <: AbstractIVEstimator
    a::Float64
    Fuller(a::Real = 1.0) = new(Float64(a))
end

"""
    KClass <: AbstractIVEstimator
    KClass(kappa::Real)

Generic K-class estimator with user-specified kappa value.

The K-class estimator uses the formula:
    β = [W'W - k*W'W_res]^(-1) * [W'y - k*W'y_res]

where W = [Xendo, Xexo] and W_res, y_res are residualized on [Z, Xexo].

Special cases:
- k = 0: OLS on structural equation (ignores endogeneity)
- k = 1: TSLS (Two-Stage Least Squares)
- k = k_LIML: LIML estimator
- k = k_LIML - a/(n-L-p): Fuller estimator

# Usage
```julia
iv(KClass(1.0), df, @formula(y ~ x + (endo ~ instrument)))  # equivalent to TSLS
iv(KClass(0.5), df, @formula(y ~ x + (endo ~ instrument)))  # custom kappa
```
"""
struct KClass <: AbstractIVEstimator
    kappa::Float64
    KClass(kappa::Real) = new(Float64(kappa))
end
