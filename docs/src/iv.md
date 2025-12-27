# IV Estimation

## Two-Stage Least Squares

The `iv()` function estimates instrumental variables models:

```julia
model = iv(TSLS(), df, @formula(y ~ x + (endo ~ z1 + z2)))
```

### Formula Syntax

Use parentheses to specify the endogenous variable and its instruments:

```julia
# Single endogenous variable
@formula(y ~ x + (endo ~ z1 + z2))

# Multiple endogenous variables
@formula(y ~ x + (endo1 + endo2 ~ z1 + z2 + z3))

# With fixed effects
@formula(y ~ x + (endo ~ z) + fe(firm))
```

### Return Type

Returns an `IVEstimator{T, V}` where:
- `T` is the numeric type (Float64 or Float32)
- `V` is the variance estimator type

## First-Stage Diagnostics

Regress.jl automatically computes first-stage diagnostics for weak instrument detection.

### Joint F-Statistic

The Kleibergen-Paap rk Wald F-statistic tests all first-stage coefficients jointly:

```julia
model = iv(TSLS(), df, @formula(y ~ x + (endo ~ z1 + z2)))

model.F_kp    # Joint first-stage F-statistic
model.p_kp    # p-value
```

### Per-Endogenous F-Statistics

For models with multiple endogenous variables, individual first-stage F-statistics are computed:

```julia
model.F_kp_per_endo   # Vector of F-statistics
model.p_kp_per_endo   # Vector of p-values
```

The output displays a table with weak instrument warnings (F < 10):

```
First-Stage F-Statistics (per endogenous variable):
------------------------------------------------------------
Endogenous                         F-stat      P-value  Weak?
------------------------------------------------------------
Price                             23.4567       0.0000
NDI                                8.1234       0.0012   Yes
------------------------------------------------------------
Note: F < 10 may indicate weak instruments (Stock-Yogo)
```

### `first_stage()` Function

Extract first-stage diagnostics as a structured object:

```julia
fs = first_stage(model)

fs.F_joint        # Joint F-statistic
fs.p_joint        # p-value
fs.F_per_endo     # Per-endogenous F-stats
fs.p_per_endo     # Per-endogenous p-values
fs.vcov_type      # Variance estimator name
fs.n_instruments  # Number of excluded instruments
fs.n_endogenous   # Number of endogenous variables
```

Pretty print the diagnostics:

```julia
show(fs)
```

## Updating Variance Estimators

The `+ vcov(...)` syntax works with IV models and recomputes all diagnostics:

```julia
model = iv(TSLS(), df, @formula(y ~ x + (endo ~ z1 + z2)))

# Update to HC3 - recomputes first-stage F-stats
model_hc3 = model + vcov(HC3())

model_hc3.F_kp           # First-stage F with HC3
model_hc3.F_kp_per_endo  # Per-endogenous F-stats with HC3
```

## Example

```julia
using Regress, DataFrames

# Simulate IV data
n = 1000
z1 = randn(n)
z2 = randn(n)
x = randn(n)
endo = 0.5 * z1 + 0.5 * z2 + 0.3 * randn(n)  # Endogenous
y = 1.0 + 2.0 * x + 3.0 * endo + randn(n)

df = DataFrame(y = y, x = x, endo = endo, z1 = z1, z2 = z2)

# TSLS estimation
model = iv(TSLS(), df, @formula(y ~ x + (endo ~ z1 + z2)))

# View results
coeftable(model)

# First-stage diagnostics
fs = first_stage(model)
println("First-stage F: ", fs.F_joint)

# Robust standard errors
model_hc3 = model + vcov(HC3())
coeftable(model_hc3)

# First-stage with robust vcov
fs_hc3 = first_stage(model_hc3)
```

## Available Estimators

| Estimator | Description |
|-----------|-------------|
| `TSLS()` | Two-Stage Least Squares |
| `LIML()` | Limited Information Maximum Likelihood |
| `Fuller(a)` | Fuller bias-corrected estimator (default a=1) |
| `KClass(κ)` | Generic K-class estimator with user-specified κ |

## K-Class Estimators: LIML, Fuller, KClass

K-class estimators are a family of IV estimators indexed by a parameter κ:
- κ = 0: OLS (ignores endogeneity)
- κ = 1: TSLS
- κ = κ_LIML: Limited Information Maximum Likelihood

### LIML

LIML is asymptotically more efficient than TSLS and more robust to weak instruments:

```julia
model = iv(LIML(), df, @formula(y ~ x + (endo ~ z1 + z2)))

# Access the computed kappa
model.postestimation.kappa  # Should be ≥ 1
```

### Fuller

Fuller's modification reduces small-sample bias by subtracting a correction:

```julia
# Default a=1 (minimizes bias)
model = iv(Fuller(), df, @formula(y ~ x + (endo ~ z1 + z2)))

# Custom a parameter
model = iv(Fuller(4.0), df, @formula(y ~ x + (endo ~ z1 + z2)))
```

Fuller(1) is approximately unbiased. Fuller(4) minimizes MSE in some scenarios.

### KClass

For direct control over κ:

```julia
# Equivalent to TSLS
model = iv(KClass(1.0), df, @formula(y ~ x + (endo ~ z1 + z2)))
```

### Variance-Covariance for K-Class

All robust variance estimators work with K-class:

```julia
# Default (HC1)
vcov(model)

# Heteroskedasticity-robust
vcov(HC3(), model)

# Cluster-robust (requires save_cluster during fit)
model = iv(LIML(), df, @formula(y ~ (endo ~ z1 + z2)), save_cluster=:firm)
vcov(CR1(:firm), model)

# Using + operator
model_robust = model + vcov(HC3())
model_clustered = model + vcov(CR1(:firm))
```

## Example: LIML with Fixed Effects and Many Instruments

A complete example with one endogenous variable, multiple instruments, and fixed effects:

```julia
using Regress, DataFrames, StableRNGs

# Simulate panel data with endogeneity
rng = StableRNG(123)
n = 1000
n_firms = 100

# Firm fixed effects
firm_id = repeat(1:n_firms, inner=n ÷ n_firms)
firm_effect = randn(rng, n_firms)[firm_id]

# Multiple instruments (e.g., Bartik-style shifters)
z1 = randn(rng, n)  # Instrument 1
z2 = randn(rng, n)  # Instrument 2
z3 = randn(rng, n)  # Instrument 3
z4 = randn(rng, n)  # Instrument 4

# Exogenous controls
x1 = randn(rng, n)
x2 = randn(rng, n)

# Endogenous variable (correlated with error)
e = randn(rng, n)
u = 0.5 .* e .+ randn(rng, n)
endo = 0.3 .* z1 .+ 0.25 .* z2 .+ 0.2 .* z3 .+ 0.15 .* z4 .+ u

# Outcome
y = 2.0 .* endo .+ 1.0 .* x1 .+ 0.5 .* x2 .+ firm_effect .+ e

df = DataFrame(
    y = y, endo = endo, x1 = x1, x2 = x2,
    z1 = z1, z2 = z2, z3 = z3, z4 = z4,
    firm_id = firm_id
)

# LIML with fixed effects and 4 instruments
model_liml = iv(LIML(), df,
    @formula(y ~ x1 + x2 + (endo ~ z1 + z2 + z3 + z4) + fe(firm_id)))

# View coefficients
coeftable(model_liml)

# Check LIML kappa (should be close to 1 with strong instruments)
println("LIML κ = ", model_liml.postestimation.kappa)

# Compare with TSLS
model_tsls = iv(TSLS(), df,
    @formula(y ~ x1 + x2 + (endo ~ z1 + z2 + z3 + z4) + fe(firm_id)))

# Coefficients should be similar but not identical
println("LIML coef: ", coef(model_liml))
println("TSLS coef: ", coef(model_tsls))

# Fuller for reduced bias
model_fuller = iv(Fuller(), df,
    @formula(y ~ x1 + x2 + (endo ~ z1 + z2 + z3 + z4) + fe(firm_id)))

# With cluster-robust standard errors
model_liml_cluster = iv(LIML(), df,
    @formula(y ~ x1 + x2 + (endo ~ z1 + z2 + z3 + z4) + fe(firm_id)),
    save_cluster = :firm_id)

model_clustered = model_liml_cluster + vcov(CR1(:firm_id))
coeftable(model_clustered)
```

### Comparing Estimators

```julia
# All three estimators on the same data
df = DataFrame(y=y, x=x, endo=endo, z1=z1, z2=z2, z3=z3)

m_tsls = iv(TSLS(), df, @formula(y ~ x + (endo ~ z1 + z2 + z3)))
m_liml = iv(LIML(), df, @formula(y ~ x + (endo ~ z1 + z2 + z3)))
m_fuller = iv(Fuller(), df, @formula(y ~ x + (endo ~ z1 + z2 + z3)))

# Compare point estimates
println("TSLS:   ", coef(m_tsls))
println("LIML:   ", coef(m_liml))
println("Fuller: ", coef(m_fuller))

# LIML and Fuller should be very close with strong instruments
# Fuller's kappa is slightly smaller than LIML's
println("LIML κ:   ", m_liml.postestimation.kappa)
println("Fuller κ: ", m_fuller.postestimation.kappa)  # κ_LIML - 1/(n-L-p)
```

### When to Use Each Estimator

| Scenario | Recommended Estimator |
|----------|----------------------|
| Strong instruments (F > 10) | TSLS or LIML (similar results) |
| Moderate instruments (F ∈ [5, 10]) | LIML or Fuller(1) |
| Many instruments | LIML (more robust) |
| Finite-sample bias concerns | Fuller(1) or Fuller(4) |
| Exact-identification | TSLS = LIML (identical) |

## Combining with Fixed Effects

IV estimation works seamlessly with fixed effects:

```julia
model = iv(TSLS(), df, @formula(y ~ x + (endo ~ z) + fe(firm) + fe(year)))
```

Fixed effects are partialed out from all variables (y, x, endo, z) before IV estimation.

## Large-Scale IV Estimation: Angrist-Krueger Returns to Schooling

This tutorial demonstrates Regress.jl's capability to efficiently estimate IV models with many instruments using the classic Angrist and Krueger (1991) returns to schooling dataset.

### Background

Angrist and Krueger's seminal paper uses quarter of birth as an instrument for education. The identification strategy exploits compulsory schooling laws: individuals born earlier in the year can legally drop out of school at a younger age, leading to less education on average. Since quarter of birth is (arguably) unrelated to ability, it provides valid instruments for education.

The "large model" specification interacts quarter of birth with year of birth and state of birth, generating approximately 180 excluded instruments.

### Data

The JIVE dataset contains approximately 330,000 observations from the 1980 Census:

| Variable | Description |
|----------|-------------|
| `lwage` | Log weekly wage |
| `educ` | Years of education |
| `yob` | Year of birth (categorical) |
| `sob` | State of birth (categorical) |
| `qob` | Quarter of birth (1-4) |

### Loading and Preparing Data

```julia
using Regress, CSV, DataFrames, CategoricalArrays

# Load data
data = CSV.read("path/to/JIVE.txt", DataFrame)

# Convert to categorical for proper dummy variable creation
data.sob = categorical(data.sob)  # 51 states
data.yob = categorical(data.yob)  # ~10 birth years
data.qob = categorical(data.qob)  # 4 quarters

println("Observations: ", nrow(data))  # ~329,509
```

### The Large Model Specification

The large model includes:
- **Outcome**: `lwage` (log weekly wage)
- **Endogenous variable**: `educ` (years of education)
- **Excluded instruments**: `yob×qob` and `sob×qob` interactions (~180 instruments)
- **Fixed effects**: Year of birth and state of birth

```julia
# Estimate large model with many instruments
@time model = iv(TSLS(), data,
    @formula(lwage ~ (educ ~ yob&qob + sob&qob) + fe(yob) + fe(sob)))
```

Output:
```
  3.96 seconds (163.91 k allocations: 5.974 GiB)
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
Note: 180 excluded instruments, 1 endogenous
```

### Interpretation

The coefficient on education is **0.093**, implying a 9.3% increase in weekly wages for each additional year of schooling. This is consistent with the original Angrist-Krueger findings.

### Performance Notes

Regress.jl efficiently handles this large-scale problem:
- **329,509 observations**
- **180 excluded instruments**
- **~62 fixed effects** (year of birth + state of birth)
- **~4 seconds** estimation time (after Julia compilation)

The fixed effects are partialed out using the iterative demeaning algorithm from FixedEffects.jl, which scales well to high-dimensional problems.

### Robust Inference

With many instruments, robust standard errors are recommended:

```julia
# HC3 robust standard errors
model_hc3 = model + vcov(HC3())
coeftable(model_hc3)

# Extract first-stage diagnostics with robust vcov
fs = first_stage(model_hc3)
println("First-stage F (HC3): ", fs.F_joint)
```

### Weak Instruments Caveat

Note that the first-stage F-statistic (2.39) is below the Stock-Yogo critical value of 10, suggesting potential weak instrument bias. This is a known issue with the many-instruments Angrist-Krueger specification. In practice, you might consider:

1. **LIML estimator**: More robust to weak instruments than TSLS

```julia
model_liml = iv(LIML(), data,
    @formula(lwage ~ (educ ~ yob&qob + sob&qob) + fe(yob) + fe(sob)))
```

2. **Fuller estimator**: Reduces finite-sample bias

```julia
model_fuller = iv(Fuller(1.0), data,
    @formula(lwage ~ (educ ~ yob&qob + sob&qob) + fe(yob) + fe(sob)))
```

3. **Fewer instruments**: Use only `yob×qob` (the "small model")

```julia
model_small = iv(TSLS(), data,
    @formula(lwage ~ (educ ~ yob&qob) + fe(yob) + fe(sob)))
```

### Complete Example

```julia
using Regress, CSV, DataFrames, CategoricalArrays

# Load and prepare data
data = CSV.read("path/to/JIVE.txt", DataFrame)
data.sob = categorical(data.sob)
data.yob = categorical(data.yob)
data.qob = categorical(data.qob)

# Large model (many instruments)
println("=== Large Model (180 instruments) ===")
@time model_large = iv(TSLS(), data,
    @formula(lwage ~ (educ ~ yob&qob + sob&qob) + fe(yob) + fe(sob)))
println(model_large)

# Small model (fewer instruments)
println("\n=== Small Model (40 instruments) ===")
@time model_small = iv(TSLS(), data,
    @formula(lwage ~ (educ ~ yob&qob) + fe(yob) + fe(sob)))
println(model_small)

# Compare with LIML (more robust to weak instruments)
println("\n=== LIML (Large Model) ===")
model_liml = iv(LIML(), data,
    @formula(lwage ~ (educ ~ yob&qob + sob&qob) + fe(yob) + fe(sob)))
println("LIML κ = ", model_liml.postestimation.kappa)
println("LIML coef on educ: ", coef(model_liml)[1])

# Robust standard errors
model_robust = model_large + vcov(HC3())
println("\n=== Robust Standard Errors (HC3) ===")
coeftable(model_robust)
```

### References

- Angrist, J. D., & Krueger, A. B. (1991). Does Compulsory School Attendance Affect Schooling and Earnings? *The Quarterly Journal of Economics*, 106(4), 979-1014.
- Stock, J. H., & Yogo, M. (2005). Testing for Weak Instruments in Linear IV Regression. In *Identification and Inference for Econometric Models: Essays in Honor of Thomas Rothenberg*.
