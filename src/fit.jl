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
- `weights::Union{Nothing, Symbol}`: Column name for weights, finite and nonnegative;
  rows with zero or missing weight are dropped
- `save::Symbol = :residuals`: Save residuals (`:residuals`), fixed effects (`:fe`), or both (`:all`)
- `save_cluster::Union{Symbol, Vector{Symbol}, Nothing}`: Cluster variables to save for post-estimation vcov
- `dof_add::Integer = 0`: Manual adjustment to degrees of freedom
- `method::Symbol = :cpu`: Computation method (`:cpu`, `:CUDA`, `:Metal`)
- `nthreads::Integer`: Number of threads (default: `Threads.nthreads()` for CPU)
- `double_precision::Bool = true` for CPU, `false` otherwise: Use Float64 vs Float32
- `factorization::Symbol = :auto`: `:chol`, `:qr` or `:sweep`; `:auto` is Cholesky for
  fewer than 100 coefficients and QR otherwise
- `collinearity::Symbol = :sweep`: Collinearity detection; `:sweep` drops a column that
  the earlier ones explain, `:qr` uses QR with column pivoting
- `collinearity_tol::Real = 1e-8`: Tolerance for collinearity detection
- `fe_tol::Real = 1e-8`: Tolerance for fixed effects demeaning (formerly `tol`, now
  deprecated)
- `fe_maxiter::Integer = 10000`: Maximum iterations for fixed effects demeaning (formerly
  `maxiter`, now deprecated); failure to converge throws an error
- `drop_singletons::Bool = true`: Drop the observations of fixed-effect groups with one
  observation, repeatedly across fixed effects until none is left
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
  - `:sweep`: Sweep operator on the cross-product matrix
- `collinearity::Symbol = :sweep`: Collinearity detection. `:sweep` walks the columns in
  order and drops a column that the earlier ones explain, as `ols(df, formula)`, the IV
  fits and fixest do; `:qr` uses QR with column pivoting, which keeps the columns of
  largest norm first
- `collinearity_tol::Real = 1e-8`: Tolerance for collinearity detection (formerly `tol`,
  now deprecated)
- `weights::Union{Nothing, AbstractVector} = nothing`: Observation weights, finite and
  nonnegative; rows with zero weight are dropped from the estimation sample
- `fe = nothing`: Fixed effects to absorb: one grouping vector of length n, or a tuple of
  them. `y` and the columns of `X` are demeaned on the groups, as `fe(...)` terms do in
  `ols(df, formula)`, and the absorbed levels enter the degrees of freedom and the
  cluster-robust small-sample correction. The fixed effects absorb the intercept, so `X`
  must not contain an intercept column.
- `fe_tol::Real = 1e-8`, `fe_maxiter::Integer = 10000`: Tolerance and iteration limit of
  the demeaning; failure to converge throws an error
- `drop_singletons::Bool = true`: Drop the observations of fixed-effect groups with one
  observation, repeatedly across fixed effects until none is left, as `ols(df, formula)`
  and fixest do. A singleton fits its observation exactly, so dropping it leaves the
  coefficients unchanged and corrects `nobs` and the degrees of freedom.
- `dof_add::Integer = 0`: Subtracted from the residual degrees of freedom
- `has_intercept::Bool = fe === nothing`: Whether model includes an intercept (for R²
  computation); `true` together with `fe` throws an `ArgumentError`
- `coefnames = nothing`: One name per column of `X`, returned by `coefnames(model)` and
  shown by `coeftable(model)`; the default is `["x1", ..., "xk"]`

# Returns
- `OLSMatrixEstimator{T, P, V, C}`: Fitted model with precomputed HC1 standard errors

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
        collinearity::Symbol = :sweep,
        collinearity_tol::Union{Real, Nothing} = nothing,
        tol::Union{Real, Nothing} = nothing,
        weights::Union{Nothing, AbstractVector} = nothing,
        fe = nothing,
        fe_tol::Real = 1e-8,
        fe_maxiter::Integer = 10000,
        drop_singletons::Bool = true,
        dof_add::Integer = 0,
        has_intercept::Bool = fe === nothing,
        coefnames::Union{Nothing, AbstractVector} = nothing)
    collinearity_tol = _renamed_keyword(collinearity_tol, tol, :collinearity_tol, :tol,
        :ols, 1e-8)

    # Fit is 1-based: X and y are materialized into `Matrix`/`Vector` and indexed
    # positionally. Reject offset axes up front with a clear message rather than
    # failing deep inside the materialization.
    Base.require_one_based_indexing(X, y)

    n, k = size(X)
    length(y) == n ||
        throw(DimensionMismatch("X has $n rows but y has $(length(y)) elements"))
    names = _matrix_coefnames(coefnames, k)
    factorization in (:auto, :chol, :qr, :sweep) ||
        throw(ArgumentError("factorization must be :auto, :chol, :qr, or :sweep, got :$factorization"))
    collinearity in (:qr, :sweep) ||
        throw(ArgumentError("collinearity must be :qr or :sweep, got :$collinearity"))

    T = promote_type(eltype(X), eltype(y))
    T <: AbstractFloat || (T = Float64)

    esample, fes, wts = _matrix_sample(n, fe, weights, has_intercept, drop_singletons, T)

    # The kernel demeans and weights in place, so the inputs are copied unless they are
    # used as given.
    as_given = isempty(fes) && wts isa UnitWeights && all(esample)
    X_mat = as_given ? convert(Matrix{T}, X) : Matrix{T}(X[esample, :])
    y_vec = as_given ? convert(Vector{T}, y) : Vector{T}(y[esample])

    fit, _ = _fit_ols_kernel(y_vec, X_mat, wts, fes;
        response_name = :y, coef_names = names, has_intercept,
        has_fe_intercept = !isempty(fes), factorization, collinearity, collinearity_tol,
        fe_tol, fe_maxiter, method = :cpu, nthreads = Threads.nthreads(),
        progress_bar = false, save_fes = false, save_matrices = true, dof_add)

    fe_groups, fe_ngroups = _matrix_fe_groups(fes)
    return OLSMatrixEstimator(fit.rr, fit.pp, fit.basis_coef,
        fit.nobs, fit.dof_model, fit.dof_residual,
        fit.rss, fit.tss_total, fit.r2, has_intercept, names, fe_groups, fe_ngroups,
        esample, fit.vcov_estimator, fit.vcov_matrix, fit.se, fit.t_stats, fit.p_values)
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
    iv(estimator, Z::AbstractMatrix, X::AbstractMatrix, y::AbstractVector;
       n_endogenous::Int=1, weights=nothing, fe=nothing, drop_singletons=true,
       dof_add=0, has_intercept::Bool=fe === nothing, coefnames=nothing) -> IVMatrixEstimator

Estimate an IV model with matrix inputs, by Two-Stage Least Squares (`TSLS()`) or a
k-class estimator (`LIML()`, `Fuller(a)`, `KClass(kappa)`). The estimation is that of
`iv(estimator, df, formula)`: the same data and settings give identical results, with
the same treatment of collinear columns (a column that the earlier ones explain is
dropped; an endogenous regressor that the instruments explain becomes exogenous).

# Arguments
- `estimator`: `TSLS()`, `LIML()`, `Fuller(a)` or `KClass(kappa)`
- `Z::AbstractMatrix`: Instrument matrix [Xexo, Zinstr] where Zinstr are excluded
  instruments. The k-class estimators require its first columns to equal those of `X`
  and throw an `ArgumentError` otherwise. For TSLS, when they do not, every column of
  `Z` is an excluded instrument, and columns of `Z` that repeat exogenous regressors are
  dropped as collinear.
- `X::AbstractMatrix`: Regressor matrix, typically [Xexo, Xendo] where Xendo are endogenous variables
- `y::AbstractVector`: Response vector

# Keyword Arguments
- `n_endogenous::Int=1`: Number of endogenous variables (last n_endogenous columns of X)
- `weights=nothing`: Observation weights, finite and nonnegative; rows with zero weight
  are dropped from the estimation sample
- `fe=nothing`, `fe_tol=1e-8`, `fe_maxiter=10000`: Fixed effects to absorb, as in
  [`ols(X, y)`](@ref); `y` and the columns of `X` and `Z` are demeaned on the groups.
  `first_stage` and `weakivtest` count the absorbed levels.
- `drop_singletons=true`: Drop fixed-effect singletons, as in [`ols(X, y)`](@ref)
- `dof_add=0`: Subtracted from the residual degrees of freedom
- `has_intercept::Bool=fe === nothing`: Whether the model includes an intercept (for R²
  computation); `true` together with `fe` throws an `ArgumentError`
- `coefnames=nothing`: One name per column of `X`, returned by `coefnames(model)` and
  shown by `coeftable(model)`; the default is `["x1", ..., "xk"]`. `first_stage(model)`
  names the endogenous variables and the exogenous regressors from it.

# Returns
- `IVMatrixEstimator{T}`: Fitted IV model supporting `coef()`, `vcov()`, `model + vcov()`, etc.

# Details

The TSLS estimator is computed as:
```math
\\hat{\\beta}_{TSLS} = (X'P_Z X)^{-1} X'P_Z y
```
where ``P_Z = Z(Z'Z)^{-1}Z'`` is the projection matrix onto the instrument space.

A k-class estimator solves
```math
\\hat{\\beta}_\\kappa = (X'X - \\kappa X'M_Z X)^{-1} (X'y - \\kappa X'M_Z y)
```
with ``M_Z = I - P_Z``; ``\\kappa`` is the LIML eigenvalue for `LIML()`, that eigenvalue
minus ``a/(n - L - p)`` for `Fuller(a)` (``L`` excluded instruments, ``p`` exogenous
regressors), and `kappa` for `KClass(kappa)`, as in the formula fits. The sandwich
variance replaces ``\\hat X`` by ``X - \\kappa M_Z X`` and ``(\\hat X'\\hat X)^{-1}`` by
``(X'X - \\kappa X'M_Z X)^{-1}``. `first_stage` and `weakivtest` depend only on the data
and the variance estimator, not on the second-stage estimator.

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
function iv(estimator::AbstractIVEstimator, Z::AbstractMatrix{<:Real},
        X::AbstractMatrix{<:Real},
        y::AbstractVector{<:Real};
        n_endogenous::Int = 1,
        weights::Union{Nothing, AbstractVector} = nothing,
        fe = nothing,
        fe_tol::Real = 1e-8,
        fe_maxiter::Integer = 10000,
        drop_singletons::Bool = true,
        dof_add::Integer = 0,
        has_intercept::Bool = fe === nothing,
        coefnames::Union{Nothing, AbstractVector} = nothing)

    # Fit is 1-based: Z, X, y are materialized into `Matrix`/`Vector` and indexed
    # positionally. Reject offset axes up front with a clear message rather than
    # failing deep inside the materialization.
    Base.require_one_based_indexing(Z, X, y)

    # Validate inputs
    n = length(y)
    size(X, 1) == n ||
        throw(DimensionMismatch("X has $(size(X,1)) rows but y has $n elements"))
    size(Z, 1) == n ||
        throw(DimensionMismatch("Z has $(size(Z,1)) rows but y has $n elements"))
    n_endogenous >= 0 || throw(ArgumentError("n_endogenous must be non-negative"))
    n_endogenous <= size(X, 2) ||
        throw(ArgumentError("n_endogenous ($n_endogenous) > number of X columns ($(size(X,2)))"))
    names = _matrix_coefnames(coefnames, size(X, 2))

    # Check order condition
    k_x = size(X, 2)
    k_z = size(Z, 2)
    k_exo = k_x - n_endogenous
    k_instr = k_z - k_exo  # Excluded instruments
    k_instr >= n_endogenous || throw(ArgumentError(
        "Not enough instruments: $k_instr excluded instruments < $n_endogenous endogenous variables"))
    _check_matrix_iv_layout(estimator, Z, X, k_exo)

    T = promote_type(eltype(X), eltype(Z), eltype(y))
    T <: AbstractFloat || (T = Float64)
    esample, fes, wts = _matrix_sample(n, fe, weights, has_intercept, drop_singletons, T)

    # The excluded instruments: the columns of Z after its copy of the exogenous
    # regressors, or all of Z when it does not lead with them (TSLS only).
    leads_with_exo = view(Z, :, 1:k_exo) == view(X, :, 1:k_exo)
    instr_cols = leads_with_exo ? ((k_exo + 1):k_z) : (1:k_z)
    Xexo = Matrix{T}(X[esample, 1:k_exo])
    Xendo = Matrix{T}(X[esample, (k_exo + 1):k_x])
    Zinstr = Matrix{T}(Z[esample, instr_cols])
    y_vec = Vector{T}(y[esample])

    fit, _ = _iv_kernel(estimator, y_vec, Xexo, Xendo, Zinstr, wts, fes;
        response_name = :y, coefnames_exo = names[1:k_exo],
        coefnames_endo = names[(k_exo + 1):k_x],
        coefnames_iv = ["z$i" for i in eachindex(instr_cols)],
        has_intercept, has_fe_intercept = !isempty(fes), fe_tol, fe_maxiter,
        method = :cpu, nthreads = Threads.nthreads(), progress_bar = false,
        save_fes = false, dof_add, first_stage = false)

    fe_groups, fe_ngroups = _matrix_fe_groups(fes)
    postestimation = PostEstimationDataIVMatrix{T}(
        fit.X, fit.Z, fit.Xhat, fit.y, fit.residuals, Matrix(fit.bread), fit.endo_cols,
        wts isa UnitWeights ? T[] : collect(wts))
    default_vcov = CovarianceMatrices.HC1()
    return IVMatrixEstimator{T, typeof(estimator), typeof(default_vcov),
        typeof(fit.vcov_matrix)}(
        estimator, fit.coef, postestimation, fit.basis_coef,
        fit.nobs, fit.dof_model, fit.dof_residual,
        fit.rss, fit.tss_total, T(1 - fit.rss / fit.tss_total),
        has_intercept, names, fe_groups, fe_ngroups, esample,
        default_vcov, fit.vcov_matrix, fit.se, fit.t_stats, fit.p_values)
end

# The kernel of each IV estimator.
_iv_kernel(::TSLS, args...; kwargs...) = _fit_tsls_kernel(args...; kwargs...)
function _iv_kernel(estimator::AbstractIVEstimator, args...; kwargs...)
    return _fit_kclass_kernel(estimator, args...; kwargs...)
end

# The k-class fit separates the exogenous regressors from the excluded instruments, so
# it needs Z = [Xexo, Zinstr] with the same leading columns as X.
_check_matrix_iv_layout(::TSLS, Z, X, k_exo) = nothing
function _check_matrix_iv_layout(estimator::AbstractIVEstimator, Z, X, k_exo)
    view(Z, :, 1:k_exo) == view(X, :, 1:k_exo) || throw(ArgumentError(
        "$(_estimator_name(estimator)) needs Z = [Xexo, Zinstr] and X = [Xexo, Xendo] " *
        "with the same first $k_exo (exogenous) columns"))
    return nothing
end

# Pivoted QR of `A`; throws `ArgumentError(msg)` when `A` has numerical rank below its
# column count, that is, when a diagonal entry of R is at most `tol` times the largest,
# the rule `detect_collinearity_qr` applies. A tolerance, rather than a failed Cholesky
# factorization of `A'A`, decides singularity: an exactly collinear `A` can leave a
# small positive pivot after rounding, depending on the BLAS kernel.
function _full_rank_qr(A, msg; tol = 1e-8)
    F = qr(A, ColumnNorm())
    R_diag = abs.(diag(F.R))
    threshold = isempty(R_diag) ? zero(eltype(R_diag)) : maximum(R_diag) * tol
    count(>(threshold), R_diag) == size(A, 2) || throw(ArgumentError(msg))
    return F
end

"""
    probit(df::DataFrame, formula; beta0=nothing, max_iter=100, tolerance=1e-6)
        -> BinaryEstimator

Fit a Bernoulli probit model by Fisher scoring. The response must be binary;
use `fe(...)` terms in `formula` to absorb fixed effects.

`beta0=nothing` starts coefficients at zero. Without fixed effects the design
includes an added intercept, which must also be included in a supplied `beta0`.
Stop when the relative deviance change is below `tolerance`.
Reaching `max_iter` without convergence raises an error.

Equivalent to `glmfe(formula, df; link=ProbitLink(), ...)`.
See [`glmfe`](@ref) for the convergence criterion and fitting details.
"""
function probit(@nospecialize(df), formula::FormulaTerm;
        beta0 = nothing, max_iter = 100, tolerance = 1e-6)
    return glmfe(formula, df; link = ProbitLink(), beta0, max_iter, tolerance)
end

"""
    glmfe(formula, data::DataFrame; link=ProbitLink(), beta0=nothing,
          max_iter=100, tolerance=1e-6) -> BinaryEstimator

Fit a Bernoulli model by Fisher scoring, optionally absorbing `fe(...)` terms.
Use `ProbitLink()` for probit or `LogitLink()` for logistic regression. The
response must contain only zeros and ones; other response families are not
currently supported.

Observations in fixed-effect intercept groups with only zeros or only ones
are removed recursively, including groups that become constant after other
groups are removed. These groups have infinite intercept estimates and do
not identify slopes in the profiled likelihood. The returned model stores
an `esample` mask and `drop_reasons` over the original rows. `nobs`, `fitted`,
`response`, and `residuals` refer to the retained sample. An empty retained
sample raises an error. Other forms of separation are not detected by this
group filter.

`beta0=nothing` starts at zero. Without fixed effects an intercept is prepended
to the design and must be included in a supplied coefficient vector. Stop when
`abs(Dnew - Dold) / (0.1 + abs(Dnew)) < tolerance`. Step-halving shrinks slope
and fixed-effect updates together to obtain a finite, non-increasing deviance.
Failure of step-halving, fixed-effect absorption, or convergence within
`max_iter` raises an error. Working quantities outside the representable
floating-point range raise a `DomainError`; probabilities are not clipped.
`max_iter` must be positive and `tolerance` must be finite and positive.

# Examples
```julia
using Regress: fe
m = glmfe(@formula(y ~ x + fe(id)), data; link=ProbitLink())
m = glmfe(@formula(y ~ x + fe(id)), data; link=LogitLink())
```

`probit(data, formula; kwargs...)` remains available as a compatibility wrapper.
"""
function glmfe(formula::FormulaTerm, @nospecialize(data);
        link = ProbitLink(), beta0 = nothing, max_iter = 100, tolerance = 1e-6)
    return fit_glmfe(data, formula, beta0, max_iter, tolerance, link)
end
