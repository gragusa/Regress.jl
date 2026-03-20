# Regress.jl

[![CI](https://github.com/gragusa/Regress.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/gragusa/Regress.jl/actions/workflows/ci.yml) [![codecov.io](http://codecov.io/github/gragusa/Regress.jl/coverage.svg?branch=master)](http://codecov.io/github/gragusa/Regress.jl?branch=master) [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) ![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826) ![lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)


High-performance linear models with fixed effects and instrumental variables.

`Regress.jl` is inspired by [FixedEffectModels.jl](https://github.com/FixedEffects/FixedEffectModels.jl). While sharing similar goals, `Regress.jl` takes a different architectural approach, with tight integration with [CovarianceMatrices.jl](https://github.com/gragusa/CovarianceMatrices.jl) and an extended family of IV estimators.

## Key Features

- OLS and IV estimation with high-dimensional fixed effects
- **Tight CovarianceMatrices.jl integration** with `model + vcov()` syntax
- **Extended IV estimators**: TSLS, LIML, Fuller, and KClass
- Comprehensive first-stage diagnostics for IV models
- **Montiel-Olea & Pflueger (2013) robust weak instrument test** with Windmeijer (2025) extensions
- Precomputed inference statistics for fast post-estimation

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/gragusa/Regress.jl")
```

## Quick Start

```julia
using Regress, DataFrames

# OLS estimation
model = Regress.ols(df, @formula(y ~ x1 + x2))

# OLS with fixed effects
model = Regress.ols(df, @formula(y ~ x1 + fe(industry) + fe(year)))

# IV estimation (Two-Stage Least Squares)
model = Regress.iv(Regress.TSLS(), df, @formula(y ~ x + (endo ~ z1 + z2)))
```

## CovarianceMatrices.jl Integration

`Regress.jl` is designed around tight integration with [CovarianceMatrices.jl](https://github.com/gragusa/CovarianceMatrices.jl), providing a seamless workflow for robust inference.

### The `model + vcov()` Syntax

A key feature is the `+` operator for updating a model's variance-covariance estimator. This returns a new model with all inference statistics precomputed:

```julia
model = Regress.ols(df, @formula(y ~ x1 + x2))

# Create a new model with HC3 standard errors
model_hc3 = model + vcov(HC3())

# All statistics are immediately available (precomputed)
stderror(model_hc3)      # HC3 standard errors
coeftable(model_hc3)     # Coefficient table with HC3 inference
model_hc3.F              # Robust Wald F-statistic
model_hc3.p              # p-value of F-statistic
```

The returned model has:

- The same underlying data and coefficients
- Precomputed vcov matrix, standard errors, t-statistics, and p-values
- Robust Wald F-statistic for joint significance

All the estimators defined in `CovarianceMatrices.jl` are supported.

## IV Estimators

`Regress.jl` provides a family of IV estimators unified under the K-class framework:

```julia
# Two-Stage Least Squares (most common)
model_tsls = Regress.iv(Regress.TSLS(), df, @formula(y ~ x + (endo ~ z1 + z2)))

# LIML - better finite-sample properties, especially with weak instruments
model_liml = Regress.iv(Regress.LIML(), df, @formula(y ~ x + (endo ~ z1 + z2)))

# Fuller - bias-corrected estimator
# Fuller(1.0) is approximately median-unbiased
# Fuller(4.0) minimizes mean squared error
model_fuller = Regress.iv(Regress.Fuller(1.0), df, @formula(y ~ x + (endo ~ z1 + z2)))

# Generic K-class with custom kappa
model_kclass = Regress.iv(Regress.KClass(0.9), df, @formula(y ~ x + (endo ~ z1 + z2)))
```

The `+ vcov()` syntax also works with IV models and automatically recomputes first-stage diagnostics:

```julia
model = Regress.iv(Regress.TSLS(), df, @formula(y ~ x + (endo ~ z1 + z2)))

# Update to HC3 - recomputes ALL statistics, including first-stage F
model_hc3 = model + vcov(HC3())
model_hc3.F_kp           # Joint first-stage F with HC3
model_hc3.F_kp_per_endo  # Per-endogenous F-stats with HC3
```

## First-Stage Diagnostics

For IV estimation, Regress.jl provides comprehensive first-stage diagnostics:

```julia
model = Regress.iv(Regress.TSLS(), df, @formula(y ~ x + (endo ~ z1 + z2)))
```

The output automatically displays:
- **Joint Kleibergen-Paap F-statistic**: Tests all first-stage coefficients jointly
- **Per-endogenous F-statistics**: Individual first-stage F-stats for each endogenous variable

```
                                    TSLS
────────────────────────────────────────────────────────────────────────────
Number of obs:                   1000   Converged:                      true
dof (model):                        2   dof (residuals):                 997
R²:                             0.892   R² adjusted:                   0.892
F-statistic:                  156.234   P-value:                       0.000
F (1st stage, joint):         124.673   P (1st stage, joint):          0.000
────────────────────────────────────────────────────────────────────────────
               Estimate  Std. Error   t-stat   Pr(>|t|)  Lower 95%  Upper 95%
────────────────────────────────────────────────────────────────────────────
x               1.98234     0.05123   38.695     0.0000    1.88176    2.08292
endo            3.01456     0.08234   36.612     0.0000    2.85301    3.17611
(Intercept)     0.98765     0.04321   22.856     0.0000    0.90293    1.07237
────────────────────────────────────────────────────────────────────────────

First-Stage F-Statistics (per endogenous variable):
────────────────────────────────────────────────────────────────────────────
Endogenous                             F-stat        P-value
────────────────────────────────────────────────────────────────────────────
endo                                 124.6735         0.0000
────────────────────────────────────────────────────────────────────────────
Note: Std. errors computed using HC1 variance estimator; 2 excluded instruments, 1 endogenous
```

### `Regress.first_stage()` - Extracting First-Stage Diagnostics

The `Regress.first_stage()` function returns a `FirstStageResult` struct for programmatic access:

```julia
model = Regress.iv(Regress.TSLS(), df, @formula(y ~ x + (endo ~ z1 + z2)))

fs = Regress.first_stage(model)
fs.F_joint           # Joint Kleibergen-Paap F-statistic
fs.p_joint           # p-value of joint test
fs.F_per_endo        # Per-endogenous F-statistics
fs.p_per_endo        # Per-endogenous p-values

# With a different variance estimator
model_hc3 = model + vcov(HC3())
fs_hc3 = Regress.first_stage(model_hc3)
```

## Weak Instrument Test

The Kleibergen-Paap F-statistic is a useful first-stage diagnostic, but it does not provide formal critical values for detecting weak instruments under heteroskedasticity. `Regress.weakivtest` implements the **Montiel-Olea & Pflueger (2013)** robust weak instrument test, along with the **Windmeijer (2025)** robust F-statistic for the GMMf estimator.

```julia
model = Regress.iv(Regress.TSLS(), df, @formula(y ~ exper + expersq + (educ ~ age + kidslt6 + kidsge6)))

result = Regress.weakivtest(model)
```

```
Montiel-Pflueger robust weak instrument test
──────────────────────────────────────────────────────
btsls:                              0.0964
sebtsls:                            0.0865
bliml:                              0.0958
sebliml:                            0.0913
kappa:                              1.0016
Non-Robust F statistic:              4.342
Effective F statistic:               4.552
Confidence level alpha:               5%
──────────────────────────────────────────────────────

──────────────────────────────────────────────────────
Critical Values                  TSLS         LIML
──────────────────────────────────────────────────────
% of Worst Case Bias
tau=5%                         15.711       15.406
tau=10%                         9.957        9.789
tau=20%                         6.749        6.654
tau=30%                         5.562        5.494
──────────────────────────────────────────────────────
```

**Decision rule:** Reject weak instruments at threshold $\tau$ if the effective F exceeds the critical value. In the example above, the effective F of 4.552 is below all critical values --- the instruments are weak.

The test supports two bias benchmarks:

```julia
# Nagar bias benchmark (default, Montiel-Olea & Pflueger 2013)
result = Regress.weakivtest(model)

# OLS bias benchmark (Windmeijer 2025)
result = Regress.weakivtest(model; benchmark=:ols)
```

The result struct provides programmatic access to all quantities:

```julia
result.F_eff          # Effective F-statistic
result.F_robust       # Robust F-statistic (Windmeijer)
result.cv_TSLS        # TSLS critical values at tau = 5%, 10%, 20%, 30%
result.cv_LIML        # LIML critical values
result.cv_GMMf        # GMMf critical values
result.btsls          # TSLS coefficient
result.bliml          # LIML coefficient
result.kappa          # LIML kappa
```

> **Note:** The test requires a single endogenous regressor, matching the Stata `gfweakivtest` command. Results have been validated against Stata.

## Large-Scale IV Estimation

`Regress.jl` efficiently handles IV estimation with many instruments. This example uses the Angrist-Krueger (1991) returns-to-schooling data with quarter-of-birth instruments.

### Example: Returns to Schooling with Many Instruments

```julia
using Regress, CSV, DataFrames, CategoricalArrays

# Load Angrist-Krueger data (~330k observations)
data = CSV.read("path/to/JIVE.txt", DataFrame)
data.sob = categorical(data.sob)  # State of birth
data.yob = categorical(data.yob)  # Year of birth
data.qob = categorical(data.qob)  # Quarter of birth

# Large model: 180 excluded instruments
# Education is endogenous, instrumented by yob*qob and sob*qob interactions
model = Regress.iv(Regress.TSLS(), data,
  @formula(lwage ~ (educ ~ fe(yob)&fe(qob) + fe(sob)&fe(qob)) + fe(yob) + fe(sob)))
```

```
                                TSLS
────────────────────────────────────────────────────────────────────
Number of obs:             329509  Converged:                   true
dof (model):                    1  dof (residuals):           329446
R²:                         0.114  R² adjusted:                0.114
F-statistic:              92.2266  P-value:                    0.000
F (1st stage, joint):     2.38722  P (1st stage, joint):       0.000
────────────────────────────────────────────────────────────────────
       Estimate  Std. Error   t-stat  Pr(>|t|)  Lower 95%  Upper 95%
────────────────────────────────────────────────────────────────────
educ  0.0928181  0.00966506  9.60347    <1e-21  0.0738748   0.111761
────────────────────────────────────────────────────────────────────

```

## API Reference

### Main Functions

| Function | Description |
|----------|-------------|
| `Regress.ols(df, formula; ...)` | Ordinary Least Squares estimation |
| `Regress.iv(method, df, formula; ...)` | Instrumental Variables estimation |
| `Regress.first_stage(model)` | Extract first-stage diagnostics from IV model |
| `Regress.weakivtest(model)` | Montiel-Olea & Pflueger robust weak instrument test |
| `fe(var)` | Fixed effect term in formula |

### Model Types

| Type | Description |
|------|-------------|
| `Regress.OLSEstimator` | Fitted OLS model |
| `Regress.IVEstimator` | Fitted IV model |
| `Regress.FirstStageResult` | First-stage diagnostics container |
| `Regress.WeakIVTestResult` | Weak instrument test results |

### IV Estimator Types

| Type | Description |
|------|-------------|
| `Regress.TSLS` | Two-Stage Least Squares (k = 1) |
| `Regress.LIML` | Limited Information Maximum Likelihood |
| `Regress.Fuller(a)` | Fuller bias-corrected estimator (default a = 1.0) |
| `Regress.KClass(kappa)` | Generic K-class with custom kappa |

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
