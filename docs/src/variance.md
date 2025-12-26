# Variance Estimation

Regress.jl integrates with [CovarianceMatrices.jl](https://github.com/gragusa/CovarianceMatrices.jl) for robust variance-covariance estimation.

## The `model + vcov(...)` Syntax

A convenient operator syntax allows updating a model's variance estimator:

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

The `+` operator returns a wrapper containing:
- The original model
- Precomputed vcov matrix, standard errors, t-statistics, and p-values
- Robust Wald F-statistic for joint significance

## Heteroskedasticity-Robust Estimators

### Available HC Estimators

| Estimator | Description |
|-----------|-------------|
| `HC0()` | White's (1980) estimator, no degrees of freedom adjustment |
| `HC1()` | DOF-adjusted (n/(n-k)), default in many packages |
| `HC2()` | Leverage-adjusted |
| `HC3()` | Jackknife-like, squared leverage adjustment |
| `HC4()` | Cribari-Neto (2004) |
| `HC5()` | Cribari-Neto modified |

### Usage

```julia
# Via + operator
model_hc3 = model + vcov(HC3())

# Direct computation
V = vcov(HC3(), model)
se = stderror(HC3(), model)
```

## Cluster-Robust Estimators

Cluster-robust variance estimation accounts for within-cluster correlation.

### Setup

Save cluster variables during estimation:

```julia
model = ols(df, @formula(y ~ x), save_cluster = :firm)
```

For multi-way clustering:

```julia
model = ols(df, @formula(y ~ x), save_cluster = (:firm, :year))
```

### Available CR Estimators

| Estimator | Description |
|-----------|-------------|
| `CR0()` | No small-sample adjustment |
| `CR1()` | G/(G-1) adjustment (Stata/R default) |
| `CR2()` | Bell-McCaffrey leverage adjustment |
| `CR3()` | Squared leverage adjustment |

### Usage

```julia
# One-way clustering
model_cr1 = model + vcov(CR1(:firm))

# Two-way clustering
model_cr12 = model + vcov(CR1(:firm, :year))

# Direct computation
V = vcov(CR1(:firm), model)
se = stderror(CR1(:firm), model)
```

### Small-Sample Correction

The default small-sample correction follows R fixest's formula:
- Scale: `G/(G-1) * (n-1)/(n-K)`
- Fixed effects nested in clusters are not counted in K

## HAC Estimators

Heteroskedasticity and Autocorrelation Consistent (HAC) estimators for time series:

### Available HAC Estimators

| Estimator | Kernel |
|-----------|--------|
| `Bartlett(bw)` | Bartlett (Newey-West) |
| `Parzen(bw)` | Parzen |
| `QuadraticSpectral(bw)` | Quadratic Spectral (Andrews) |
| `TukeyHanning(bw)` | Tukey-Hanning |
| `Truncated(bw)` | Truncated |

The `bw` parameter is the bandwidth (number of lags).

### Usage

```julia
# Bartlett kernel with 5 lags (Newey-West)
model_hac = model + vcov(Bartlett(5))

# Quadratic spectral kernel
model_qs = model + vcov(QuadraticSpectral(10))
```

## Combining with IV Models

All variance estimators work with IV models. The `+ vcov(...)` syntax recomputes first-stage diagnostics:

```julia
model = iv(TSLS(), df, @formula(y ~ x + (endo ~ z1 + z2)))

# HC3 for IV
model_hc3 = model + vcov(HC3())

# First-stage F-stats are recomputed with HC3
model_hc3.F_kp           # Joint first-stage F with HC3
model_hc3.F_kp_per_endo  # Per-endogenous F-stats with HC3
```

## Robust Wald F-Statistic

The Wald F-statistic tests the joint null hypothesis that all non-intercept coefficients are zero:

```julia
model_hc3 = model + vcov(HC3())

model_hc3.F   # Robust Wald F-statistic
model_hc3.p   # p-value
```

The F-statistic is computed as:
```
F = (1/q) * beta' * inv(V[non-intercept, non-intercept]) * beta
```

where `q` is the number of non-intercept coefficients.

## Performance Considerations

- `vcov()` is computed lazily when called on a model
- The `+ vcov(...)` syntax precomputes and caches all statistics
- For multiple variance estimators, use direct `vcov(estimator, model)` calls
- HAC estimators are more expensive than HC estimators

## Example

```julia
using Regress, DataFrames

# Create data
n = 1000
df = DataFrame(
    y = randn(n),
    x1 = randn(n),
    x2 = randn(n),
    firm = rand(1:50, n),
    year = rand(2010:2020, n)
)

# Fit model with cluster variable
model = ols(df, @formula(y ~ x1 + x2), save_cluster = (:firm, :year))

# Compare variance estimators
println("HC1: ", stderror(model))
println("HC3: ", stderror(model + vcov(HC3())))
println("CR1 (firm): ", stderror(model + vcov(CR1(:firm))))
println("CR1 (two-way): ", stderror(model + vcov(CR1(:firm, :year))))

# Full inference with cluster-robust SE
model_cr = model + vcov(CR1(:firm))
coeftable(model_cr)
println("Robust Wald F: ", model_cr.F)
```
