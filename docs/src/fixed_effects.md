# Fixed Effects

## Overview

Regress.jl efficiently handles high-dimensional fixed effects by absorbing (partialing out) categorical variables from the regression.

## Specifying Fixed Effects

Use `fe()` in the formula:

```julia
# One-way fixed effects
model = ols(df, @formula(y ~ x + fe(firm)))

# Two-way fixed effects
model = ols(df, @formula(y ~ x + fe(firm) + fe(year)))

# Three-way (or more)
model = ols(df, @formula(y ~ x + fe(firm) + fe(year) + fe(industry)))
```

## Interacted Fixed Effects

### Nested Fixed Effects

Combine multiple categoricals into a single fixed effect:

```julia
# Firm-year fixed effects
model = ols(df, @formula(y ~ x + fe(firm)&fe(year)))
```

### Continuous Interactions (Slopes)

Allow for heterogeneous slopes:

```julia
# Firm-specific time trends
model = ols(df, @formula(y ~ x + fe(firm)&year))

# Industry-specific effects of x
model = ols(df, @formula(y ~ fe(industry)&x))
```

### Full Factorial

Use `*` for full factorial expansion:

```julia
# Expands to: fe(firm) + fe(year) + fe(firm)&fe(year)
model = ols(df, @formula(y ~ x + fe(firm)*fe(year)))
```

## Algorithm

Regress.jl uses an iterative demeaning algorithm from [FixedEffects.jl](https://github.com/gragusa/FixedEffects.jl):

1. For each fixed effect, compute group means
2. Subtract group means from all variables
3. Repeat until convergence

### Convergence Options

```julia
model = ols(df, @formula(y ~ x + fe(firm)),
    tol = 1e-8,        # Convergence tolerance (default 1e-6)
    maxiter = 50000    # Maximum iterations (default 10000)
)
```

### Checking Convergence

```julia
model.converged     # Boolean
model.iterations    # Number of iterations
```

## Singleton Groups

By default, observations in groups with only one observation are dropped:

```julia
model = ols(df, @formula(y ~ x + fe(firm)),
    drop_singletons = true   # Default
)
```

Disable to keep singletons:

```julia
model = ols(df, @formula(y ~ x + fe(firm)),
    drop_singletons = false
)
```

## Within R-squared

For models with fixed effects, two R-squared values are reported:

- `r2(model)`: Overall R-squared (including fixed effects)
- `model.r2_within`: Within R-squared (variation explained by x only)

## Degrees of Freedom

Fixed effects consume degrees of freedom. Check:

```julia
dof_residual(model)  # n - k - number of FE groups
model.dof_fes        # Degrees of freedom absorbed by FE
```

## GPU Acceleration

For very large datasets, enable GPU acceleration:

```julia
# CUDA (NVIDIA GPUs)
model = ols(df, @formula(y ~ x + fe(firm)),
    method = :CUDA
)

# Metal (Apple Silicon)
model = ols(df, @formula(y ~ x + fe(firm)),
    method = :Metal
)
```

GPU acceleration is beneficial for n > 10 million observations.

## Multi-threading

Enable multi-threading for CPU computation:

```julia
model = ols(df, @formula(y ~ x + fe(firm)),
    method = :cpu,
    nthreads = 8   # Number of threads
)
```

## Extracting Fixed Effects

To recover the fixed effect estimates:

```julia
# Save fixed effects during estimation
model = ols(df, @formula(y ~ x + fe(firm)),
    save = :fe   # or :all for both residuals and FE
)

# Access via fes component
model.fes.values  # Named tuple of fixed effect values
```

## Partial Out

Use `partial_out()` to remove fixed effects from variables without running a regression:

```julia
# Partial out firm effects from y and x
result, converged = partial_out(df, @formula(y + x ~ fe(firm)))
```

## Example

```julia
using Regress, DataFrames

# Create panel data
n_firms = 100
n_years = 10
n = n_firms * n_years

df = DataFrame(
    firm = repeat(1:n_firms, inner = n_years),
    year = repeat(2010:2019, outer = n_firms),
    y = randn(n),
    x = randn(n)
)

# One-way FE
m1 = ols(df, @formula(y ~ x + fe(firm)))
println("Within R2: ", m1.r2_within)

# Two-way FE
m2 = ols(df, @formula(y ~ x + fe(firm) + fe(year)))
println("Within R2: ", m2.r2_within)

# Firm-specific trends
m3 = ols(df, @formula(y ~ x + fe(firm)&year))

# Cluster-robust SE
m2_cr = ols(df, @formula(y ~ x + fe(firm) + fe(year)),
    save_cluster = :firm)
m2_cr1 = m2_cr + vcov(CR1(:firm))
coeftable(m2_cr1)
```
