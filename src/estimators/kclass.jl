##############################################################################
##
## K-Class Estimation (LIML, Fuller, KClass)
##
## Uses the K-class formula: β = [W'W - k*W'Wres]^(-1) * [W'y - k*W'yres]
##
##############################################################################

"""
    _kclass_kappa(est, y, Xendo, Z, Xexo) -> T

The k-class parameter of the estimator `est`: the LIML eigenvalue for `LIML()`, the LIML
eigenvalue minus `a / (n - L - p)` for `Fuller(a)` (`n` rows, `L` excluded instruments
in `Z`, `p` exogenous regressors in `Xexo`), and `kappa` for `KClass(kappa)`.

The LIML eigenvalue is at least 1, and equals 1 when the model is exactly identified, so
Fuller's `kappa` is below 1 whenever the LIML eigenvalue is within `a / (n - L - p)` of
1. A `kappa` below 1 is part of these estimators, not a sign of weak identification.
"""
function _kclass_kappa(est::AbstractIVEstimator, y::AbstractVector{T},
        Xendo::AbstractVecOrMat{T}, Z::AbstractMatrix{T},
        Xexo::AbstractMatrix{T}) where {T}
    kappa = if est isa LIML
        _liml_kappa(y, Xendo, Z, Xexo)
    elseif est isa Fuller
        adj_denom = length(y) - size(Z, 2) - size(Xexo, 2)
        if adj_denom <= 0
            throw(ArgumentError("Fuller adjustment undefined: n - L - p = $adj_denom <= 0"))
        end
        _liml_kappa(y, Xendo, Z, Xexo) - T(est.a) / adj_denom
    elseif est isa KClass
        T(est.kappa)
    else
        throw(ArgumentError("Unknown K-class estimator: $(typeof(est))"))
    end

    return kappa
end

"""
    _fit_kclass_kernel(est, y, Xexo, Xendo, Z, wts, fes; kwargs...) -> (fit, fe_solve)

The k-class estimator `est` (`LIML()`, `Fuller(a)` or `KClass(kappa)`) of `y` on
`[Xexo, Xendo]` with instruments `[Xexo, Z]` and weights `wts`, absorbing the fixed
effects `fes`. Both `iv(estimator, df, formula)` and `iv(estimator, Z, X, y)` estimate
through this function, so equal data and settings give identical results. See
[`_fit_tsls_kernel`](@ref) for the inputs and the layout of `fit`; `fit.Xhat` and
`fit.bread` hold the k-class adjustment matrix `Adj` and `inv(A)`.
"""
function _fit_kclass_kernel(est::AbstractIVEstimator, y::Vector{T}, Xexo::Matrix{T},
        Xendo::Matrix{T}, Z::Matrix{T}, wts::AbstractWeights, fes::Vector{<:FixedEffect};
        iv_fes::Vector{FixedEffect} = FixedEffect[], has_intercept::Bool,
        dof_add::Integer, first_stage::Bool, kwargs...) where {T <: AbstractFloat}
    isempty(iv_fes) || throw(ArgumentError(
        "fixed effects as excluded instruments are supported by TSLS only, not by " *
        "$(_estimator_name(est))"))
    prep, fe_solve = _iv_prepare!(y, Xexo, Xendo, Z, wts, fes; has_intercept, kwargs...)
    Xexo, Xendo, Z = prep.Xexo, prep.Xendo, prep.Z
    size(Z, 2) >= size(Xendo, 2) ||
        throw(ArgumentError("Model not identified. There must be at least as many instruments as endogenous variables"))
    n = length(y)
    n_exo, n_endo = size(Xexo, 2), size(Xendo, 2)

    kappa = _kclass_kappa(est, y, Xendo, Z, Xexo)
    coef_kclass, residuals, invA, Adj = _kclass_fit(y, Xendo, Z, Xexo, kappa)

    # `_kclass_fit` orders the coefficients [endogenous; exogenous]
    idx = vcat((n_endo + 1):(n_endo + n_exo), 1:n_endo)
    coef_reduced = coef_kclass[idx]
    X = hcat(Xexo, Xendo)

    newZ = hcat(Xexo, Z)
    Pi = first_stage ? qr(newZ) \ Xendo : Matrix{T}(undef, size(newZ, 2), 0)
    fs_stats = _iv_first_stage_stats(first_stage, Xendo, newZ * Pi, newZ, Pi, prep.XexoZ,
        prep.ZZ, prep, iv_fes, n, has_intercept)

    coef, basis_coef, order, endo_cols = _iv_formula_order(prep.coll, coef_reduced, n_exo)
    issorted(order) || (X = X[:, order])
    Adj = Adj[:, idx[order]]
    invA = Symmetric(invA[idx[order], idx[order]])

    return _iv_kernel_inference(; coef, basis_coef, X, Xhat = Adj, bread = invA,
        crossx = cholesky(Symmetric(X' * X)), y, Z = newZ, endo_cols, residuals, prep,
        dof_add, has_intercept, fs_stats, kappa),
    fe_solve
end

"""
    fit_kclass_estimator(estimator, df, formula; kwargs...) -> IVEstimator{T}

Fit a k-class IV estimator (`LIML()`, `Fuller(a)` or `KClass(kappa)`) from a table and
a formula; see [`iv`](@ref) for the keywords.
"""
function fit_kclass_estimator(
        estimator::AbstractIVEstimator,
        @nospecialize(df),
        formula::FormulaTerm;
        contrasts::Dict = Dict{Symbol, Any}(),
        weights::Union{Symbol, Nothing} = nothing,
        save::Union{Bool, Symbol} = :residuals,
        save_cluster::Union{Symbol, Vector{Symbol}, Nothing} = nothing,
        dof_add::Integer = 0,
        method::Symbol = :cpu,
        nthreads::Integer = method == :cpu ? Threads.nthreads() : 256,
        double_precision::Bool = method == :cpu,
        fe_tol::Union{Real, Nothing} = nothing,
        fe_maxiter::Union{Integer, Nothing} = nothing,
        tol::Union{Real, Nothing} = nothing,
        maxiter::Union{Integer, Nothing} = nothing,
        drop_singletons::Bool = true,
        progress_bar::Bool = true,
        subset::Union{Nothing, AbstractVector} = nothing,
        first_stage::Bool = true)
    fe_tol = _renamed_keyword(fe_tol, tol, :fe_tol, :tol, :iv, 1e-8)
    fe_maxiter = _renamed_keyword(fe_maxiter, maxiter, :fe_maxiter, :maxiter, :iv, 10000)
    prep = _iv_formula_data(df, formula; contrasts, weights, save, save_cluster, method,
        nthreads, double_precision, drop_singletons, subset)
    return _fit_kclass_formula(estimator, prep.y, prep.Xexo, prep.Xendo, prep.Z,
        prep.wts, prep.subfes; prep, fe_tol, fe_maxiter, method, nthreads = prep.nthreads,
        progress_bar, dof_add, first_stage)
end

# The formula k-class fit from its model matrices onward, compiled for the concrete `T`.
function _fit_kclass_formula(est::AbstractIVEstimator, y::Vector{T}, Xexo::Matrix{T},
        Xendo::Matrix{T}, Z::Matrix{T}, wts, subfes; prep, fe_tol, fe_maxiter, method,
        nthreads, progress_bar, dof_add, first_stage) where {T}
    data_prep = prep.data_prep
    fit, fe_solve = _fit_kclass_kernel(est, y, Xexo, Xendo, Z, wts, subfes;
        iv_fes = prep.iv_fes, _iv_kernel_names(prep)..., fe_tol, fe_maxiter, method,
        nthreads, progress_bar, has_intercept = data_prep.has_intercept,
        has_fe_intercept = data_prep.has_fe_intercept, save_fes = data_prep.save_fes,
        dof_add, first_stage)
    return _iv_formula_result(est, fit, fe_solve, prep, wts, subfes, fe_tol, fe_maxiter)
end

# Convenience wrappers

"""
    fit_liml(df, formula; kwargs...) -> IVEstimator{T}

Fit an instrumental variables model using LIML (Limited Information Maximum Likelihood).
"""
function fit_liml(@nospecialize(df), formula::FormulaTerm; kwargs...)
    return fit_kclass_estimator(LIML(), df, formula; kwargs...)
end

"""
    fit_fuller(df, formula; a=1.0, kwargs...) -> IVEstimator{T}

Fit an instrumental variables model using Fuller's bias-corrected estimator.

# Keyword Arguments
- `a::Real = 1.0`: Fuller bias correction parameter. Fuller(1) is median-unbiased.
"""
function fit_fuller(@nospecialize(df), formula::FormulaTerm; a::Real = 1.0, kwargs...)
    return fit_kclass_estimator(Fuller(a), df, formula; kwargs...)
end

"""
    fit_kclass(df, formula; kappa, kwargs...) -> IVEstimator{T}

Fit an instrumental variables model using generic K-class estimator.

# Keyword Arguments
- `kappa::Real`: K-class parameter. k=1 gives TSLS.
"""
function fit_kclass(@nospecialize(df), formula::FormulaTerm; kappa::Real, kwargs...)
    return fit_kclass_estimator(KClass(kappa), df, formula; kwargs...)
end
