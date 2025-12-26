# Getting Started

## Installation

Regress.jl can be installed from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/gragusa/Regress.jl")
```

## Basic Usage

### Loading the Package

```julia
using Regress, DataFrames
```

Regress.jl re-exports [StatsModels.jl](https://github.com/JuliaStats/StatsModels.jl) for formula syntax and [CovarianceMatrices.jl](https://github.com/gragusa/CovarianceMatrices.jl) for variance estimators.

### Creating a Model

Use `ols()` for Ordinary Least Squares:

```julia
df = DataFrame(y = randn(100), x = randn(100))
model = ols(df, @formula(y ~ x))
```

Use `iv()` for Instrumental Variables:

```julia
model = iv(TSLS(), df, @formula(y ~ (endo ~ instrument)))
```

### Accessing Results

All standard StatsAPI methods work:

```julia
coef(model)          # Coefficient estimates
stderror(model)      # Standard errors
vcov(model)          # Variance-covariance matrix
confint(model)       # Confidence intervals
coeftable(model)     # Full coefficient table
nobs(model)          # Number of observations
r2(model)            # R-squared
```

### Robust Standard Errors

Update the variance estimator using `+ vcov(...)`:

```julia
model_hc3 = model + vcov(HC3())
stderror(model_hc3)  # HC3 standard errors
```

Or compute directly:

```julia
vcov(HC3(), model)
stderror(HC3(), model)
```

## Formula Syntax

### Basic Terms

```julia
@formula(y ~ x1 + x2)           # Multiple regressors
@formula(y ~ x1 + x2 + x1&x2)   # Interaction
@formula(y ~ x1 * x2)           # Full factorial (x1 + x2 + x1&x2)
```

### Fixed Effects

Use `fe()` to absorb high-dimensional categorical variables:

```julia
@formula(y ~ x + fe(firm))                    # One-way FE
@formula(y ~ x + fe(firm) + fe(year))         # Two-way FE
@formula(y ~ x + fe(firm)&fe(year))           # Firm-year FE
@formula(y ~ x + fe(firm)&year)               # Firm-specific time trends
```

### Instrumental Variables

Use parentheses `(endo ~ instruments)` for IV specification:

```julia
@formula(y ~ x + (endo ~ z1 + z2))              # One endogenous variable
@formula(y ~ x + (endo1 + endo2 ~ z1 + z2 + z3)) # Multiple endogenous
@formula(y ~ (endo ~ z) + fe(firm))             # IV with fixed effects
```

## Common Options

### Keyword Arguments for `ols()` and `iv()`

| Argument | Type | Description |
|----------|------|-------------|
| `weights` | `Symbol` | Column for weighted regression |
| `save_cluster` | `Symbol` or `Tuple{Symbol,...}` | Save cluster variables for post-estimation |
| `subset` | `AbstractVector` | Boolean vector for subsetting |
| `contrasts` | `Dict` | Contrast codings for categorical variables |
| `method` | `Symbol` | `:cpu`, `:CUDA`, or `:Metal` |
| `nthreads` | `Integer` | Number of threads for fixed effects |
| `double_precision` | `Bool` | Use Float64 (default true for CPU) |
| `tol` | `Real` | Tolerance for FE demeaning (default 1e-6) |
| `drop_singletons` | `Bool` | Drop singleton groups (default true) |

### Example with Options

```julia
model = ols(df, @formula(y ~ x + fe(firm)),
    weights = :pop,
    save_cluster = :firm,
    method = :cpu,
    nthreads = 4
)
```
