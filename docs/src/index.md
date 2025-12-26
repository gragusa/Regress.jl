# Regress.jl

**Regress.jl** is a Julia package for estimating linear regression models with high-dimensional fixed effects and instrumental variables.

## Features

- **OLS Estimation** with support for high-dimensional fixed effects
- **Instrumental Variables** estimation (Two-Stage Least Squares)
- **Robust Variance Estimation** via integration with [CovarianceMatrices.jl](https://github.com/gragusa/CovarianceMatrices.jl)
  - Heteroskedasticity-robust (HC0-HC5)
  - Cluster-robust (CR0-CR3)
  - HAC estimators (Bartlett, Parzen, etc.)
- **GPU Acceleration** for large-scale fixed effects (CUDA, Metal)
- **Convenient API** with `model + vcov(...)` syntax for updating variance estimators

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/gragusa/Regress.jl")
```

## Quick Example

```julia
using Regress, DataFrames

# Create sample data
df = DataFrame(
    y = randn(1000),
    x1 = randn(1000),
    x2 = randn(1000),
    firm = rand(1:50, 1000),
    year = rand(2010:2020, 1000)
)

# OLS estimation
model = ols(df, @formula(y ~ x1 + x2))

# OLS with fixed effects
model_fe = ols(df, @formula(y ~ x1 + fe(firm) + fe(year)))

# Robust standard errors
model_hc3 = model + vcov(HC3())
stderror(model_hc3)

# Cluster-robust standard errors
model_cr = ols(df, @formula(y ~ x1), save_cluster = :firm)
model_cr1 = model_cr + vcov(CR1(:firm))
```

## Contents

```@contents
Pages = ["getting_started.md", "ols.md", "iv.md", "fixed_effects.md", "variance.md", "api.md"]
Depth = 2
```

## Credits

This package is a fork of [FixedEffectModels.jl](https://github.com/FixedEffects/FixedEffectModels.jl) by Matthieu Gomez, with additional features and improvements.
