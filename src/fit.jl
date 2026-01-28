##############################################################################
##
## Main User-Facing API: ols() and iv()
##
##############################################################################

"""
    ols(df, formula; kwargs...) -> OLSEstimator

Estimate a linear model using Ordinary Least Squares (OLS).

Supports high-dimensional categorical variables (fixed effects) but not instrumental variables.
For IV models, use `iv(estimator, df, formula)`.

# Arguments
- `df`: a Table (e.g., DataFrame)
- `formula`: A formula created using `@formula(y ~ x1 + x2 + fe(group))`

# Keyword Arguments
- `contrasts::Dict = Dict()`: Contrast codings for categorical variables
- `weights::Union{Nothing, Symbol}`: Column name for weights
- `save::Symbol = :residuals`: Save residuals (`:residuals`), fixed effects (`:fe`), or both (`:all`)
- `save_cluster::Union{Symbol, Vector{Symbol}, Nothing}`: Cluster variables to save for post-estimation vcov
- `dof_add::Integer = 0`: Manual adjustment to degrees of freedom
- `method::Symbol = :cpu`: Computation method (`:cpu`, `:CUDA`, `:Metal`)
- `nthreads::Integer`: Number of threads (default: `Threads.nthreads()` for CPU)
- `double_precision::Bool = true` for CPU, `false` otherwise: Use Float64 vs Float32
- `tol::Real = 1e-6`: Tolerance for fixed effects demeaning
- `maxiter::Integer = 10000`: Maximum iterations for fixed effects
- `drop_singletons::Bool = true`: Drop singleton observations
- `progress_bar::Bool = true`: Show progress bar during estimation
- `subset::Union{Nothing, AbstractVector}`: Select specific rows

# Returns
- `OLSEstimator{T}`: Fitted model (T is Float64 or Float32 depending on `double_precision`)

# Examples
```julia
using DataFrames, RDatasets, Regress

df = dataset("plm", "Cigar")

# Simple OLS
model = ols(df, @formula(Sales ~ NDI + Pop))

# With fixed effects
model = ols(df, @formula(Sales ~ NDI + fe(State) + fe(Year)))

# Post-estimation robust standard errors
vcov(HC3(), model)
vcov(:State, :CR1, model)  # cluster-robust

# With weights
model = ols(df, @formula(Sales ~ NDI), weights = :Pop)
```

# Post-Estimation Variance Calculations

After fitting a model, you can compute different variance-covariance matrices without re-running the regression:

```julia
model = ols(df, @formula(y ~ x1 + x2 + fe(firm_id)))

# Different robust estimators
vcov(HC3(), model)           # Heteroskedasticity-robust (HC3)
vcov(HC1(), model)           # Default (HC1)

# Cluster-robust (using stored cluster variable from fe())
vcov(:firm_id, :CR1, model)

# Two-way clustering
vcov((:firm_id, :year), :CR1, model)

# HAC (time series)
vcov(Bartlett(5), model)

# Standard errors and coefficient table
stderror(HC3(), model)
coeftable(model, :firm_id, :CR1)
```

See also: [`iv`](@ref), [`OLSEstimator`](@ref)
"""
function ols(@nospecialize(df), formula::FormulaTerm; kwargs...)
    has_iv(formula) &&
        throw(ArgumentError("Formula contains instrumental variables. Use `iv(TSLS(), df, formula)` instead."))
    return fit_ols(df, formula; kwargs...)
end

##############################################################################
##
## Matrix-Based OLS API
##
##############################################################################

"""
    ols(X::AbstractMatrix, y::AbstractVector; kwargs...) -> OLSMatrixEstimator

Fit an OLS model from matrix inputs (no formula required).

This is a lightweight, efficient interface for cases where you have pre-computed
design matrices. Compatible with CovarianceMatrices.jl for robust inference.

# Arguments
- `X::AbstractMatrix`: Design matrix (n x k). Include an intercept column if desired.
- `y::AbstractVector`: Response vector (length n)

# Keyword Arguments
- `factorization::Symbol = :auto`: Factorization method
  - `:auto`: Cholesky for k < 100, QR for k >= 100
  - `:chol`: Faster Cholesky factorization
  - `:qr`: More numerically stable QR factorization
- `collinearity::Symbol = :qr`: Collinearity detection method (`:qr` or `:sweep`)
- `tol::Real = 1e-8`: Tolerance for collinearity detection
- `weights::Union{Nothing, AbstractVector} = nothing`: Observation weights
- `has_intercept::Bool = true`: Whether model includes an intercept (for R² computation)

# Returns
- `OLSMatrixEstimator{T, P, V}`: Fitted model with precomputed HC1 standard errors

# Example
```julia
using Regress, CovarianceMatrices

# Create design matrix with intercept
n = 100
X = hcat(ones(n), randn(n, 2))
beta_true = [1.0, 2.0, -1.5]
y = X * beta_true + 0.1 * randn(n)

# Fit model
model = ols(X, y)

# Extract results
coef(model)           # Coefficients
stderror(model)       # Standard errors (HC1)
r2(model)             # R-squared
residuals(model)      # Residuals
nobs(model)           # Number of observations

# Post-estimation robust standard errors
vcov(HC3(), model)    # HC3 vcov matrix
stderror(HC3(), model)  # HC3 standard errors

# Update model with different vcov
model_hc3 = model + vcov(HC3())
stderror(model_hc3)   # Now uses HC3

# Confidence intervals
confint(model)        # Using precomputed HC1
confint(HC3(), model) # Using HC3
```

# Collinearity Handling
Linearly dependent columns are automatically detected. Coefficients for
collinear columns are set to 0, and their standard errors to NaN.

```julia
# With collinear column
X = hcat(ones(100), randn(100, 2), randn(100, 2)[:, 1])  # Last column duplicated
model = ols(X, randn(100))
coef(model)  # One coefficient will be 0
```

See also: [`ols(df, formula)`](@ref), [`OLSMatrixEstimator`](@ref)
"""
function ols(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real};
        factorization::Symbol = :auto,
        collinearity::Symbol = :qr,
        tol::Real = 1e-8,
        weights::Union{Nothing, AbstractVector} = nothing,
        has_intercept::Bool = true)

    # Validate inputs
    n, k = size(X)
    length(y) == n ||
        throw(DimensionMismatch("X has $n rows but y has $(length(y)) elements"))

    # Validate keywords
    factorization in (:auto, :chol, :qr) ||
        throw(ArgumentError("factorization must be :auto, :chol, or :qr, got :$factorization"))
    collinearity in (:qr, :sweep) ||
        throw(ArgumentError("collinearity must be :qr or :sweep, got :$collinearity"))

    # Determine numeric type
    T = promote_type(eltype(X), eltype(y))
    T <: AbstractFloat || (T = Float64)

    # Convert to Matrix{T} and Vector{T} (materializes views)
    X_mat = convert(Matrix{T}, X)
    y_vec = convert(Vector{T}, y)

    # Handle weights
    has_weights = weights !== nothing
    if has_weights
        length(weights) == n || throw(DimensionMismatch("weights must have length $n"))
        wts_vec = convert(Vector{T}, weights)
        sqrtw = sqrt.(wts_vec)
        X_mat = X_mat .* sqrtw
        y_vec = y_vec .* sqrtw
    else
        wts_vec = T[]
    end

    # Choose factorization
    if factorization == :auto
        factorization = k < 100 ? :chol : :qr
    end

    # Build response object
    mu = similar(y_vec)
    rr = OLSResponse(y_vec, mu, wts_vec, T[], :y)

    # Compute TSS (before fitting, for R² calculation)
    if has_intercept
        ymean = mean(y_vec)
        tss = sum(abs2, y_vec .- ymean)
    else
        # For models without intercept, TSS = sum(y^2)
        tss = sum(abs2, y_vec)
    end

    # Fit using unified solver
    pp, basis_coef,
    _ = fit_ols_core!(rr, X_mat, factorization;
        tol = tol, save_matrices = true, collinearity = collinearity)

    # Compute RSS efficiently
    rss = compute_rss(rr.y, rr.mu)

    # Degrees of freedom
    dof_model = sum(basis_coef)
    dof_res = max(1, n - dof_model)

    # R-squared
    r2_val = 1 - rss / tss

    # Compute default HC1 vcov and statistics
    residuals_vcov = rr.y .- rr.mu
    invXX = invchol(pp)

    vcov_matrix = compute_hc1_vcov_direct(
        pp.X, residuals_vcov, invXX, basis_coef,
        n, dof_model, 0, dof_res  # No fixed effects
    )

    # Standard errors
    se = sqrt.(diag(vcov_matrix))

    # t-statistics and p-values
    coef_vec = copy(pp.beta)
    coef_vec[.!basis_coef] .= zero(T)
    t_stats = coef_vec ./ se
    p_values = 2 .* tdistccdf.(dof_res, abs.(t_stats))

    # Default vcov estimator (HC1)
    default_vcov = CovarianceMatrices.HC1()

    return OLSMatrixEstimator{T, typeof(pp), typeof(default_vcov)}(
        rr, pp, basis_coef,
        n, dof_model, dof_res,
        T(rss), T(tss), T(r2_val), has_intercept,
        default_vcov, vcov_matrix, se, t_stats, p_values
    )
end

##############################################################################
##
## IV Function - Instrumental Variables
##
##############################################################################

"""
    iv(estimator::AbstractIVEstimator, df, formula; kwargs...) -> IVEstimator

Estimate an instrumental variables model using the specified estimator.

# Arguments
- `estimator::AbstractIVEstimator`: Estimator type (`TSLS()`, `LIML()`, `Fuller()`, `KClass()`)
- `df`: a Table (e.g., DataFrame)
- `formula`: A formula with IV syntax: `@formula(y ~ x + (endo ~ instrument))`

# Keyword Arguments
Same as `ols()`, plus:
- `first_stage::Bool = true`: Compute first-stage F-statistics

# Returns
- `IVEstimator{T}`: Fitted IV model (T is Float64 or Float32 depending on `double_precision`)

# Available Estimators
- `TSLS()`: Two-Stage Least Squares (k = 1)
- `LIML()`: Limited Information Maximum Likelihood (k = k_LIML)
- `Fuller(a)`: Fuller bias-corrected estimator (k = k_LIML - a/(n-L-p)), default a=1
- `KClass(k)`: Generic K-class estimator with user-specified k

# Examples
```julia
using DataFrames, RDatasets, Regress

df = dataset("plm", "Cigar")

# Two-stage least squares
model = iv(TSLS(), df, @formula(Sales ~ NDI + (Price ~ Pimin)))

# LIML (better with weak instruments)
model = iv(LIML(), df, @formula(Sales ~ NDI + (Price ~ Pimin)))

# Fuller bias-corrected (approximately median-unbiased)
model = iv(Fuller(), df, @formula(Sales ~ NDI + (Price ~ Pimin)))
model = iv(Fuller(4.0), df, @formula(Sales ~ NDI + (Price ~ Pimin)))  # minimize MSE

# Generic K-class
model = iv(KClass(0.5), df, @formula(Sales ~ NDI + (Price ~ Pimin)))

# With fixed effects
model = iv(TSLS(), df, @formula(Sales ~ (Price ~ Pimin) + fe(State)))

# Post-estimation
vcov(HC3(), model)
coeftable(model)
```

# Post-Estimation Variance Calculations

IV models support the same post-estimation vcov calculations as OLS:

```julia
model = iv(TSLS(), df, @formula(y ~ x + (endo ~ instrument)))

# Heteroskedasticity-robust
vcov(HC3(), model)

# Update model with different vcov
model_hc3 = model + vcov(HC3())

# Cluster-robust (save cluster variable when fitting)
model = iv(TSLS(), df, @formula(y ~ x + (endo ~ inst)), save_cluster = :firm_id)
vcov(CR1(:firm_id), model)

# Two-way clustering
model = iv(TSLS(), df, @formula(...), save_cluster = [:firm_id, :year])
vcov(CR1(:firm_id, :year), model)
```

See also: [`ols`](@ref), [`TSLS`](@ref), [`LIML`](@ref), [`Fuller`](@ref), [`KClass`](@ref), [`IVEstimator`](@ref)
"""
# Check formula before dispatching to specific estimator
function _check_iv_formula(formula::FormulaTerm)
    !has_iv(formula) &&
        throw(ArgumentError("Formula does not contain instrumental variables. Use `ols(df, formula)` instead."))
end

function iv(::TSLS, @nospecialize(df), formula::FormulaTerm; kwargs...)
    _check_iv_formula(formula)
    return fit_tsls(df, formula; kwargs...)
end

function iv(::LIML, @nospecialize(df), formula::FormulaTerm; kwargs...)
    _check_iv_formula(formula)
    return fit_liml(df, formula; kwargs...)
end

function iv(estimator::Fuller, @nospecialize(df), formula::FormulaTerm; kwargs...)
    _check_iv_formula(formula)
    return fit_fuller(df, formula; a = estimator.a, kwargs...)
end

function iv(estimator::KClass, @nospecialize(df), formula::FormulaTerm; kwargs...)
    _check_iv_formula(formula)
    return fit_kclass(df, formula; kappa = estimator.kappa, kwargs...)
end

##############################################################################
##
## Matrix-based IV Function
##
##############################################################################

"""
    iv(::TSLS, Z::AbstractMatrix, X::AbstractMatrix, y::AbstractVector;
       has_intercept::Bool=true, n_endogenous::Int=1) -> IVMatrixEstimator

Estimate an IV model using Two-Stage Least Squares with matrix inputs.

# Arguments
- `Z::AbstractMatrix`: Instrument matrix, typically [Xexo, Zinstr] where Zinstr are excluded instruments
- `X::AbstractMatrix`: Regressor matrix, typically [Xexo, Xendo] where Xendo are endogenous variables
- `y::AbstractVector`: Response vector

# Keyword Arguments
- `has_intercept::Bool=true`: Whether the model includes an intercept (affects DOF calculation)
- `n_endogenous::Int=1`: Number of endogenous variables (last n_endogenous columns of X)

# Returns
- `IVMatrixEstimator{T}`: Fitted IV model supporting `coef()`, `vcov()`, `model + vcov()`, etc.

# Details

The TSLS estimator is computed as:
```math
\\hat{\\beta}_{TSLS} = (X'P_Z X)^{-1} X'P_Z y
```
where ``P_Z = Z(Z'Z)^{-1}Z'`` is the projection matrix onto the instrument space.

# Example
```julia
using Regress

# Generate data
n = 200
z = randn(n)           # Instrument
u = randn(n)           # Error
x = 0.5*z + 0.5*u      # Endogenous (correlated with u)
y = 1.0 .+ 2.0*x .+ u  # True coefficient = 2.0

# Build matrices (no intercept case for simplicity)
Z = hcat(ones(n), z)   # [constant, instrument]
X = hcat(ones(n), x)   # [constant, endogenous]

model = iv(TSLS(), Z, X, y; has_intercept=true, n_endogenous=1)
coef(model)  # Should be approximately [1.0, 2.0]

# Robust standard errors
using CovarianceMatrices
vcov(HC1(), model)
model_robust = model + vcov(HC1())
```

See also: [`iv(::TSLS, df, formula)`](@ref), [`IVMatrixEstimator`](@ref)
"""
function iv(::TSLS, Z::AbstractMatrix{<:Real}, X::AbstractMatrix{<:Real},
        y::AbstractVector{<:Real};
        has_intercept::Bool = true,
        n_endogenous::Int = 1)

    # Validate inputs
    n = length(y)
    size(X, 1) == n ||
        throw(DimensionMismatch("X has $(size(X,1)) rows but y has $n elements"))
    size(Z, 1) == n ||
        throw(DimensionMismatch("Z has $(size(Z,1)) rows but y has $n elements"))
    n_endogenous >= 0 || throw(ArgumentError("n_endogenous must be non-negative"))
    n_endogenous <= size(X, 2) ||
        throw(ArgumentError("n_endogenous ($n_endogenous) > number of X columns ($(size(X,2)))"))

    # Check order condition
    k_x = size(X, 2)
    k_z = size(Z, 2)
    k_exo = k_x - n_endogenous
    k_instr = k_z - k_exo  # Excluded instruments
    k_instr >= n_endogenous || throw(ArgumentError(
        "Not enough instruments: $k_instr excluded instruments < $n_endogenous endogenous variables"))

    # Determine numeric type
    T = promote_type(eltype(X), eltype(Z), eltype(y))
    T <: AbstractFloat || (T = Float64)

    # Convert to concrete types
    X_mat = convert(Matrix{T}, X)
    Z_mat = convert(Matrix{T}, Z)
    y_vec = convert(Vector{T}, y)

    # Compute projection matrix P_Z = Z(Z'Z)^{-1}Z'
    ZZ = Z_mat' * Z_mat
    ZZ_chol = try
        cholesky(Symmetric(ZZ))
    catch e
        throw(ArgumentError("Instrument matrix Z is rank deficient or nearly singular"))
    end
    ZZ_inv = inv(ZZ_chol)

    # Projected regressors: X̂ = P_Z * X = Z * (Z'Z)^{-1} * Z' * X
    ZX = Z_mat' * X_mat
    X_hat = Z_mat * (ZZ_inv * ZX)

    # TSLS estimator: β = (X̂'X)^{-1} X̂'y = (X'P_Z X)^{-1} X'P_Z y
    # Using X̂'X = X'P_Z X and X̂'y = X'P_Z y
    XhatX = X_hat' * X_mat
    Xhaty = X_hat' * y_vec

    XhatX_chol = try
        cholesky(Symmetric(XhatX))
    catch e
        throw(ArgumentError("X̂'X is singular - check for perfect collinearity"))
    end

    beta = XhatX_chol \ Xhaty

    # For vcov, we need (X̂'X̂)^{-1}, not (X̂'X)^{-1}
    XhatXhat = X_hat' * X_hat
    XhatXhat_chol = try
        cholesky(Symmetric(XhatXhat))
    catch
        # Fall back to pseudo-inverse
        Symmetric(pinv(XhatXhat))
    end
    invXhatXhat = if XhatXhat_chol isa Cholesky
        inv(XhatXhat_chol)
    else
        XhatXhat_chol  # Already a matrix from pinv
    end

    # Residuals
    residuals = y_vec - X_mat * beta

    # Degrees of freedom
    k = size(X_mat, 2)
    basis_coef = trues(k)  # No collinearity detection for simplicity
    dof_model = sum(basis_coef)
    dof_res = max(1, n - dof_model)

    # Sum of squares
    rss = sum(abs2, residuals)
    ymean = mean(y_vec)
    tss = has_intercept ? sum(abs2, y_vec .- ymean) : sum(abs2, y_vec)
    r2_val = 1 - rss / tss

    # Post-estimation data for vcov
    postestimation = PostEstimationDataIVMatrix{T}(
        X_mat, Z_mat, X_hat, y_vec, residuals, Matrix{T}(invXhatXhat), n_endogenous
    )

    # Default vcov (HC1)
    default_vcov = CovarianceMatrices.HC1()

    # Compute HC1 variance-covariance matrix
    # Sandwich: V = n * (X̂'X̂)^{-1} * (X̂'diag(e²)X̂ * n/(n-k)) / n * (X̂'X̂)^{-1}
    #           = (n/(n-k)) * (X̂'X̂)^{-1} * X̂'diag(e²)X̂ * (X̂'X̂)^{-1}
    scale_hc1 = T(n) / T(n - k)
    Xe = X_hat .* residuals
    meat = Xe' * Xe
    vcov_matrix = Symmetric(scale_hc1 .* invXhatXhat * meat * invXhatXhat)

    # Standard errors, t-stats, p-values
    se = sqrt.(diag(vcov_matrix))
    t_stats = beta ./ se
    p_values = 2 .* tdistccdf.(dof_res, abs.(t_stats))

    return IVMatrixEstimator{T, typeof(default_vcov)}(
        beta,
        postestimation,
        basis_coef,
        n,
        dof_model,
        dof_res,
        T(rss),
        T(tss),
        T(r2_val),
        has_intercept,
        default_vcov,
        vcov_matrix,
        se,
        t_stats,
        p_values
    )
end
