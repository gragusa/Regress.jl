# Regress.jl

This package estimates linear models with high dimensional categorical variables, potentially including instrumental variables.

This is a fork of [FixedEffectModels.jl](https://github.com/FixedEffects/FixedEffectModels.jl) with additional features and improvements.

## Installation
The package can be installed via:
```julia
using Pkg
Pkg.add(url="https://github.com/gragusa/Regress.jl")
```

## Quick Start

```julia
using Regress, DataFrames

# OLS estimation
model = ols(df, @formula(y ~ x1 + x2))

# OLS with fixed effects
model = ols(df, @formula(y ~ x1 + fe(industry) + fe(year)))

# IV estimation (Two-Stage Least Squares)
model = iv(TSLS(), df, @formula(y ~ x + (endo ~ z1 + z2)))
```

## Robust Variance-Covariance Estimation

Regress.jl integrates with [CovarianceMatrices.jl](https://github.com/gragusa/CovarianceMatrices.jl) for robust inference.

### Heteroskedasticity-Robust (HC) Estimators

```julia
vcov(HC0(), model)  # White's estimator
vcov(HC1(), model)  # DOF-adjusted (default)
vcov(HC2(), model)  # Leverage-adjusted
vcov(HC3(), model)  # Jackknife-like
```

### Cluster-Robust (CR) Estimators

```julia
# Fit model with cluster variable saved
model = ols(df, @formula(y ~ x), save_cluster = :firm)

# Cluster-robust inference
vcov(CR1(:firm), model)
stderror(CR1(:firm), model)

# Two-way clustering
model = ols(df, @formula(y ~ x), save_cluster = (:firm, :year))
vcov(CR1(:firm, :year), model)
```

## `model + vcov(...)` Syntax

A convenient operator syntax allows updating a model's variance-covariance estimator while preserving all other statistics:

```julia
model = ols(df, @formula(y ~ x1 + x2))

# Create a new model with HC3 standard errors
model_hc3 = model + vcov(HC3())

# Access updated statistics
stderror(model_hc3)      # HC3 standard errors
coeftable(model_hc3)     # Coefficient table with HC3 inference
model_hc3.F              # Robust Wald F-statistic
model_hc3.p              # p-value of F-statistic
```

The `+` operator returns a `ModelWithVcov` wrapper that contains:
- The original model
- Precomputed vcov matrix, standard errors, t-statistics, and p-values
- Robust Wald F-statistic for joint significance

This syntax works with all variance estimators (HC0-HC5, CR0-CR3, HAC):

```julia
# Cluster-robust
model_cr = ols(df, @formula(y ~ x), save_cluster = :firm)
model_cr1 = model_cr + vcov(CR1(:firm))

# HAC
model_hac = model + vcov(Bartlett(5))
```

## IV Estimation with First-Stage Diagnostics

For instrumental variables estimation, Regress.jl provides comprehensive first-stage diagnostics:

```julia
model = iv(TSLS(), df, @formula(y ~ x + (endo ~ z1 + z2)))
```

The output automatically displays:
- **Joint Kleibergen-Paap F-statistic**: Tests all first-stage coefficients jointly
- **Per-endogenous F-statistics**: Individual first-stage F-stats for each endogenous variable

```
                         IVEstimator
================================================================
Number of obs:                  1000  Converged:              true
dof (model):                       2  dof (residuals):         997
R²:                            0.892  R² adjusted:           0.892
F-statistic:                 156.234  P-value:               0.000
F (1st stage, joint):        124.673  P (1st stage, joint):  0.000
================================================================
               Estimate  Std. Error   t-stat   Pr(>|t|)  Lower 95%  Upper 95%
----------------------------------------------------------------
x               1.98234     0.05123   38.695     0.0000    1.88176    2.08292
endo            3.01456     0.08234   36.612     0.0000    2.85301    3.17611
(Intercept)     0.98765     0.04321   22.856     0.0000    0.90293    1.07237
================================================================

First-Stage F-Statistics (per endogenous variable):
------------------------------------------------------------
Endogenous                             F-stat        P-value
------------------------------------------------------------
endo                                 124.6735         0.0000
------------------------------------------------------------
```

### Updating IV Models with Different Variance Estimators

The `+ vcov(...)` syntax also works with IV models and recomputes first-stage diagnostics:

```julia
model = iv(TSLS(), df, @formula(y ~ x + (endo ~ z1 + z2)))

# Update to HC3 - recomputes ALL statistics including first-stage F
model_hc3 = model + vcov(HC3())

model_hc3.F_kp           # Joint first-stage F with HC3
model_hc3.F_kp_per_endo  # Per-endogenous F-stats with HC3
```

### `first_stage()` - Extracting First-Stage Diagnostics

The `first_stage()` function returns a `FirstStageResult` struct with all first-stage diagnostics:

```julia
model = iv(TSLS(), df, @formula(y ~ x + (endo ~ z1 + z2)))

# Get first-stage diagnostics
fs = first_stage(model)
fs.F_joint           # Joint Kleibergen-Paap F-statistic
fs.p_joint           # p-value of joint test
fs.F_per_endo        # Per-endogenous F-statistics
fs.p_per_endo        # Per-endogenous p-values

# With different variance estimator
model_hc3 = model + vcov(HC3())
fs_hc3 = first_stage(model_hc3)
fs_hc3.vcov_type     # "HR3" (the internal type name)
```

Pretty-printed output:
```
First-Stage Diagnostics (HC1)
============================================================
Joint Test (Kleibergen-Paap):
  F-statistic:   124.6735       p-value: 0.0000

Per-Endogenous F-Statistics:
------------------------------------------------------------
Endogenous                             F-stat        P-value
------------------------------------------------------------
endo                                  54.6586         0.0000
------------------------------------------------------------

Instruments: 2 excluded, 1 endogenous
============================================================
```

## API Reference

### Main Functions

| Function | Description |
|----------|-------------|
| `ols(df, formula; ...)` | Ordinary Least Squares estimation |
| `iv(method, df, formula; ...)` | Instrumental Variables estimation |
| `fe(var)` | Fixed effect term in formula |
| `first_stage(model)` | Extract first-stage diagnostics from IV model |

### Model Types

| Type | Description |
|------|-------------|
| `OLSEstimator` | Fitted OLS model |
| `IVEstimator` | Fitted IV model |
| `ModelWithVcov` | Model wrapper with precomputed vcov statistics |
| `FirstStageResult` | First-stage diagnostics container |
| `TSLS` | Two-Stage Least Squares estimator |
| `LIML` | Limited Information Maximum Likelihood estimator |

### StatsAPI Methods

All standard StatsAPI methods work with fitted models:

```julia
coef(model)          # Coefficient estimates
stderror(model)      # Standard errors
vcov(model)          # Variance-covariance matrix
confint(model)       # Confidence intervals
coeftable(model)     # Full coefficient table
nobs(model)          # Number of observations
dof(model)           # Degrees of freedom (model)
dof_residual(model)  # Degrees of freedom (residual)
r2(model)            # R-squared
adjr2(model)         # Adjusted R-squared
residuals(model)     # Residual vector
fitted(model)        # Fitted values
```