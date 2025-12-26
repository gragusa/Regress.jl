# API Reference

## Main Functions

```@docs
ols
iv
fe
partial_out
first_stage
```

## Model Types

```@docs
OLSEstimator
IVEstimator
FirstStageResult
```

## IV Estimators

```@docs
AbstractIVEstimator
TSLS
LIML
```

## Variance Specification

```@docs
VcovSpec
```

## StatsAPI Interface

All fitted models (`OLSEstimator`, `IVEstimator`) implement the StatsAPI interface:

### Coefficients and Inference

| Function | Description |
|----------|-------------|
| `coef(model)` | Coefficient estimates |
| `coefnames(model)` | Coefficient names |
| `stderror(model)` | Standard errors (default HC1) |
| `stderror(vcov_type, model)` | Standard errors with specific vcov |
| `vcov(model)` | Variance-covariance matrix |
| `vcov(vcov_type, model)` | Vcov with specific estimator |
| `confint(model)` | 95% confidence intervals |
| `coeftable(model)` | Full coefficient table |

### Model Statistics

| Function | Description |
|----------|-------------|
| `nobs(model)` | Number of observations |
| `dof(model)` | Degrees of freedom (model) |
| `dof_residual(model)` | Degrees of freedom (residual) |
| `r2(model)` | R-squared |
| `adjr2(model)` | Adjusted R-squared |
| `deviance(model)` | Residual sum of squares |
| `nulldeviance(model)` | Null model deviance |
| `loglikelihood(model)` | Log-likelihood |
| `nullloglikelihood(model)` | Null model log-likelihood |

### Prediction and Residuals

| Function | Description |
|----------|-------------|
| `fitted(model)` | Fitted values (if saved) |
| `residuals(model)` | Residuals (if saved) |
| `predict(model, newdata)` | Predict for new data |
| `responsename(model)` | Name of response variable |

## Model Fields

### OLSEstimator Fields

| Field | Type | Description |
|-------|------|-------------|
| `coef` | `Vector{T}` | Coefficient estimates |
| `vcov_matrix` | `Matrix{T}` | Variance-covariance matrix |
| `F` | `T` | Robust Wald F-statistic |
| `p` | `T` | p-value of F-statistic |
| `esample` | `BitVector` | Estimation sample indicator |
| `nobs` | `Int` | Number of observations |
| `dof` | `Int` | Model degrees of freedom |
| `dof_residual` | `Int` | Residual degrees of freedom |
| `dof_fes` | `Int` | Fixed effects degrees of freedom |
| `rss` | `T` | Residual sum of squares |
| `tss` | `T` | Total sum of squares |
| `r2` | `T` | R-squared |
| `r2_within` | `T` | Within R-squared (if FE) |
| `adjr2` | `T` | Adjusted R-squared |
| `converged` | `Bool` | Convergence status |
| `iterations` | `Int` | Number of iterations |

### IVEstimator Fields

All fields from `OLSEstimator`, plus:

| Field | Type | Description |
|-------|------|-------------|
| `F_kp` | `T` | Joint first-stage F-statistic |
| `p_kp` | `T` | p-value of joint F |
| `F_kp_per_endo` | `Vector{T}` | Per-endogenous F-statistics |
| `p_kp_per_endo` | `Vector{T}` | Per-endogenous p-values |

## Variance Estimators

### Heteroskedasticity-Robust

From CovarianceMatrices.jl:

```julia
HC0()   # White's estimator
HC1()   # DOF-adjusted (default)
HC2()   # Leverage-adjusted
HC3()   # Jackknife-like
HC4()   # Cribari-Neto
HC5()   # Cribari-Neto modified
```

### Cluster-Robust

```julia
CR0(:cluster)           # No adjustment
CR1(:cluster)           # G/(G-1) adjustment
CR2(:cluster)           # Bell-McCaffrey
CR3(:cluster)           # Squared leverage

# Multi-way
CR1(:cluster1, :cluster2)
```

### HAC

```julia
Bartlett(bandwidth)
Parzen(bandwidth)
QuadraticSpectral(bandwidth)
TukeyHanning(bandwidth)
Truncated(bandwidth)
```

## Index

```@index
```
