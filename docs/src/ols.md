# OLS Estimation

## Basic OLS

The `ols()` function estimates linear models using Ordinary Least Squares:

```julia
model = ols(df, @formula(y ~ x1 + x2))
```

### Return Type

Returns an `OLSEstimator{T, P, V}` where:
- `T` is the numeric type (Float64 or Float32)
- `P` is the predictor type (Cholesky or QR factorization)
- `V` is the variance estimator type

### Accessing Results

```julia
coef(model)          # Coefficient vector
stderror(model)      # Standard errors (default HC1)
vcov(model)          # Variance-covariance matrix
confint(model)       # 95% confidence intervals
coeftable(model)     # Full coefficient table with statistics

nobs(model)          # Number of observations
dof(model)           # Degrees of freedom (model)
dof_residual(model)  # Degrees of freedom (residual)
r2(model)            # R-squared
adjr2(model)         # Adjusted R-squared

residuals(model)     # Residual vector (if saved)
fitted(model)        # Fitted values (if saved)
```

## Weighted Regression

Specify weights using a column name:

```julia
model = ols(df, @formula(y ~ x), weights = :pop)
```

## Categorical Variables

Categorical variables are automatically handled with appropriate dummy coding:

```julia
using CategoricalArrays
df.region = categorical(df.region)

model = ols(df, @formula(y ~ x + region))
```

Control contrast coding with the `contrasts` argument:

```julia
model = ols(df, @formula(y ~ x + region),
    contrasts = Dict(:region => DummyCoding(base = "North"))
)
```

## Prediction

### In-Sample Prediction

```julia
# Using stored fitted values (requires save = :residuals or :all)
fitted(model)

# Recompute predictions
predict(model, df)
```

### Out-of-Sample Prediction

```julia
new_data = DataFrame(x1 = [1.0, 2.0], x2 = [0.5, 1.5])
predict(model, new_data)
```

## Collinearity Handling

Regress.jl automatically detects and handles collinear variables:

```julia
# x2 is perfectly collinear with x1
df.x2 = 2 * df.x1

model = ols(df, @formula(y ~ x1 + x2))
# Coefficient for x2 will be 0 with NaN standard error
```

Check which coefficients are identified:

```julia
model.basis_coef  # BitVector indicating non-collinear coefficients
```

## Solver Options

### Cholesky vs QR

By default, Regress.jl uses Cholesky factorization for speed. For numerically ill-conditioned problems, QR factorization is more stable:

```julia
# Automatic choice based on condition number
model = ols(df, @formula(y ~ x))

# The predictor type is stored in the model
model.predictor  # OLSPredictorChol or OLSPredictorQR
```

## Example

```julia
using Regress, DataFrames, CategoricalArrays

# Load data
df = DataFrame(
    sales = randn(1000) .+ 100,
    price = randn(1000) .+ 50,
    advertising = randn(1000) .+ 20,
    region = rand(["North", "South", "East", "West"], 1000),
    pop = rand(1000:5000, 1000)
)
df.region = categorical(df.region)

# Basic OLS
m1 = ols(df, @formula(sales ~ price + advertising))
coeftable(m1)

# With categorical variable
m2 = ols(df, @formula(sales ~ price + advertising + region))
coeftable(m2)

# Weighted regression
m3 = ols(df, @formula(sales ~ price + advertising), weights = :pop)
coeftable(m3)

# Robust standard errors
m3_hc3 = m3 + vcov(HC3())
coeftable(m3_hc3)
```
