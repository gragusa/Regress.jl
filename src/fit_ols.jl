##############################################################################
##
## OLS Estimation - New Architecture with GLM-compatible Structure
##
##############################################################################

"""
    fit_ols(df, formula; kwargs...) -> OLSEstimator{T}

Fit a linear model using Ordinary Least Squares (OLS).

This is the lean, fast workhorse function for linear regression.
Supports fixed effects but NOT instrumental variables.

# New Features
- `factorization::Symbol`: Choose `:auto` (default), `:chol`, or `:qr`
  - `:auto`: Cholesky for k < 100, QR for k >= 100
  - `:chol`: Faster but less stable (uses Cholesky decomposition)
  - `:qr`: More stable but ~2x slower (uses QR decomposition)
- `lazy_vcov::Bool`: If `true`, defer vcov computation until accessed (faster fitting).
  Default is `true`. Set to `false` to precompute vcov at fit time.
"""
function fit_ols(@nospecialize(df),
        formula::FormulaTerm;
        contrasts::Dict = Dict{Symbol, Any}(),
        weights::Union{Symbol, Nothing} = nothing,
        save::Union{Bool, Symbol} = :residuals,
        save_cluster::Union{Symbol, Vector{Symbol}, Nothing} = nothing,
        dof_add::Integer = 0,
        method::Symbol = :cpu,
        factorization::Symbol = :auto,  # NEW: :auto, :chol, or :qr
        collinearity::Symbol = :qr,     # :qr or :sweep
        nthreads::Integer = method == :cpu ? Threads.nthreads() : 256,
        double_precision::Bool = method == :cpu,
        tol::Real = 1e-6,
        maxiter::Integer = 10000,
        drop_singletons::Bool = true,
        progress_bar::Bool = true,
        subset::Union{Nothing, AbstractVector} = nothing,
        lazy_vcov::Bool = true)  # Defer vcov computation (saves time when using custom vcov)

    # Validate keywords
    save, save_residuals = validate_save_keyword(save)
    nthreads = validate_nthreads(method, nthreads)

    # Validate factorization choice
    factorization in (:auto, :chol, :qr, :sweep) ||
        throw(ArgumentError("factorization must be :auto, :chol, :qr, or :sweep, got :$factorization"))

    # Validate collinearity method
    collinearity in (:qr, :sweep) ||
        throw(ArgumentError("collinearity must be :qr or :sweep, got :$collinearity"))

    # Convert contrasts to Dict{Symbol, Any}
    contrasts = convert(Dict{Symbol, Any}, contrasts)

    # Convert to DataFrame
    df = DataFrame(df; copycols = false)

    ##############################################################################
    ## Prepare Data
    ##############################################################################

    data_prep = prepare_data(df, formula, weights, subset, save, drop_singletons, nthreads)

    # Extract cluster variables (skip for :minimal mode to save memory)
    if save == :minimal
        cluster_data = NamedTuple()
    else
        cluster_data = extract_cluster_variables(df, data_prep.fe_vars, save_cluster, data_prep.esample)
    end

    # Create subsetted dataframe and weights
    if data_prep.has_weights
        wts = Weights(disallowmissing(view(df[!, weights], data_prep.esample)))
    else
        wts = uweights(data_prep.nobs)
    end

    # Create subsetted dataframe - optimize by checking if disallowmissing is needed
    subdf = _create_subdf(df, data_prep.all_vars, data_prep.esample)
    subfes = isempty(data_prep.fes) ? FixedEffect[] :
             FixedEffect[fe[data_prep.esample] for fe in data_prep.fes]

    ##############################################################################
    ## Create Model Matrices
    ##############################################################################

    # Determine numeric type
    T = double_precision ? Float64 : Float32

    # Apply schema
    s = schema(data_prep.formula, subdf, contrasts)
    formula_schema = apply_schema(data_prep.formula, s, OLSEstimator, data_prep.has_fe_intercept)

    # Create matrices
    y = convert(Vector{T}, response(formula_schema, subdf))
    X = convert(Matrix{T}, modelmatrix(formula_schema, subdf))
    response_name, coef_names = coefnames(formula_schema)

    # Build response object (fitted values will be set after solve)
    rr = build_response(y, wts, Symbol(response_name))

    # Compute total sum of squares before demeaning
    tss_total = compute_tss_total(rr.y, wts, data_prep.has_intercept |
                                             data_prep.has_fe_intercept)

    # Validate finite values
    all(isfinite, wts) || throw("Weights are not finite")
    all(isfinite, rr.y) ||
        throw("Some observations for the dependent variable are infinite")
    all(isfinite, X) || throw("Some observations for the exogenous variables are infinite")

    ##############################################################################
    ## Partial Out Fixed Effects
    ##############################################################################

    oldy, oldX = nothing, nothing
    feM = nothing

    if data_prep.has_fes
        # Combine columns for demeaning
        # Direct construction avoids vcat overhead and extra allocations
        n_cols = 1 + size(X, 2)
        cols = Vector{AbstractVector{T}}(undef, n_cols)
        cols[1] = rr.y
        @inbounds for j in 1:size(X, 2)
            cols[j + 1] = view(X, :, j)
        end
        colnames = vcat(response_name, coef_names)

        # Partial out fixed effects (modifies cols in-place)
        feM, iterations,
        converged,
        tss_partial,
        oldy,
        oldX = partial_out_fixed_effects!(cols, colnames, subfes, wts, method, nthreads,
            tol, maxiter, progress_bar, data_prep.save_fes,
            data_prep.has_intercept, data_prep.has_fe_intercept, T)
    else
        iterations, converged = 0, true
        tss_partial = tss_total
    end

    ##############################################################################
    ## Apply Weights
    ##############################################################################

    if data_prep.has_weights
        sqrtw = sqrt.(wts)
        rr.y .= rr.y .* sqrtw
        X .= X .* sqrtw
    end

    ##############################################################################
    ## Choose Factorization Method
    ##############################################################################

    if factorization == :auto
        k = size(X, 2)
        # Sweep is fastest (matches FEM) for small k
        # For k >= 100, use QR for numerical stability
        factorization = k < 100 ? :sweep : :qr
    end

    ##############################################################################
    ## Build Predictor and Solve (Unified)
    ##############################################################################

    # Determine whether to save matrices (X, y, mu)
    # :minimal mode discards them to save memory
    save_matrices = (save != :minimal)

    # Unified solver: detect collinearity, factorize, and solve in one pass
    # This avoids redundant matrix subsetting and double-solving
    pp, basis_coef,
    _ = fit_ols_core!(rr, X, factorization;
        save_matrices = save_matrices,
        collinearity = collinearity,
        has_intercept = data_prep.has_intercept)

    ##############################################################################
    ## Compute RSS (without allocating residuals vector)
    ##############################################################################

    # Compute RSS directly using SIMD-optimized loop
    rss = compute_rss(rr.y, rr.mu)

    # Degrees of freedom
    ngroups_fes = [nunique(fe) for fe in subfes]
    dof_fes = sum(ngroups_fes)
    dof_model = sum(basis_coef)  # Only non-collinear coefficients
    dof_base = data_prep.nobs - dof_model - dof_fes - dof_add
    dof_residual = max(1, dof_base - (data_prep.has_intercept | data_prep.has_fe_intercept))

    # R-squared
    r2 = 1 - rss / tss_total
    r2_within = data_prep.has_fes ? 1 - rss / tss_partial : r2

    # F-statistic and p-value
    # Numerator DOF for F-test excludes intercept (tests joint significance of slopes)
    # For models with FE, use within TSS; otherwise use total TSS
    tss_for_fstat = data_prep.has_fes ? tss_partial : tss_total
    mss_val = tss_for_fstat - rss
    dof_model_ftest = dof_model - (data_prep.has_intercept & !data_prep.has_fe_intercept)
    F_stat = dof_model_ftest > 0 ? (mss_val / dof_model_ftest) / (rss / dof_residual) :
             T(NaN)
    p_val = dof_model_ftest > 0 ? fdistccdf(dof_model_ftest, dof_residual, F_stat) : T(NaN)

    ##############################################################################
    ## Solve for Fixed Effects (if requested)
    ##############################################################################

    augmentdf = DataFrame()
    if data_prep.save_fes && oldy !== nothing
        # Solve for FE estimates
        # Note: oldX was saved before collinearity detection, so filter by basis_coef
        newfes, b,
        c = solve_coefficients!(oldy - oldX[:, basis_coef] * pp.beta[basis_coef], feM;
            tol = tol, maxiter = maxiter)

        # Create DataFrame with FE estimates
        for fekey in data_prep.fekeys
            augmentdf[!, fekey] = df[!, fekey]
        end
        for j in eachindex(subfes)
            augmentdf[!, data_prep.feids[j]] = Vector{Union{T, Missing}}(missing, data_prep.nrows)
            augmentdf[data_prep.esample, data_prep.feids[j]] = newfes[j]
        end
    end

    ##############################################################################
    ## Build Fixed Effects Component
    ##############################################################################

    # Use cluster_data which was already extracted and subsetted
    cluster_vars_nt = cluster_data

    fes = OLSFixedEffects{T}(
        augmentdf,
        data_prep.fekeys,
        cluster_vars_nt,
        dof_fes,
        ngroups_fes,
        iterations,
        converged,
        method
    )

    ##############################################################################
    ## Handle esample
    ##############################################################################

    # Handle Colon case for esample
    esample_final = data_prep.esample == Colon() ? trues(data_prep.nrows) :
                    data_prep.esample

    ##############################################################################
    ## Convert coefficient names to strings
    ##############################################################################

    coef_names_str = String[string(name) for name in coef_names]

    ##############################################################################
    ## Compute Default Vcov (HC1) and Related Statistics
    ## (Must be done before clearing y/mu for minimal mode, unless lazy_vcov=true)
    ##############################################################################

    # Default vcov estimator (HC1)
    default_vcov = CovarianceMatrices.HC1()

    # Lazy vcov: defer computation until accessed
    # This speeds up fitting when the user will call vcov(CR1(...), m) separately
    if lazy_vcov && save_matrices
        # Store nothing for lazy fields - will be computed on first access
        vcov_matrix = nothing
        se = nothing
        t_stats_val = nothing
        p_values_val = nothing
        F_stat_robust = nothing
        p_val_robust = nothing
    else
        # Eager vcov computation (original behavior)
        # Compute residuals for vcov (weighted: y - mu)
        residuals_vcov = rr.y .- rr.mu

        # Get invXX from the predictor
        invXX = invchol(pp)

        # Compute HC1 vcov matrix directly from components
        # Note: X and residuals are weighted if model has weights
        vcov_matrix = if save_matrices
            compute_hc1_vcov_direct(
                pp.X, residuals_vcov, invXX, basis_coef,
                data_prep.nobs, dof_model, dof_fes, dof_residual
            )
        else
            # Minimal mode: can't compute vcov without X
            # Create a placeholder zero matrix
            Symmetric(zeros(T, length(basis_coef), length(basis_coef)))
        end

        # Compute standard errors
        se = sqrt.(diag(vcov_matrix))

        # Compute t-statistics and p-values
        coef_vec = copy(pp.beta)
        coef_vec[.!basis_coef] .= zero(T)
        t_stats_val = coef_vec ./ se
        p_values_val = 2 .* tdistccdf.(dof_residual, abs.(t_stats_val))

        # Compute robust F-statistic (Wald test) using vcov
        has_int = data_prep.has_intercept
        F_stat_robust,
        p_val_robust = compute_robust_fstat(
            coef_vec, vcov_matrix, has_int, dof_residual
        )
    end

    ##############################################################################
    ## Clear y/mu if minimal mode (to save memory)
    ##############################################################################

    if !save_matrices
        # Clear data from response and predictor to save memory
        clear_response_data!(rr)
        clear_predictor_data!(pp)
    end

    ##############################################################################
    ## Return OLSEstimator
    ##############################################################################

    # Ensure all statistics are of type T for type stability
    # Note: F_stat_robust and p_val_robust may be Nothing (lazy) or T (eager)
    return OLSEstimator{T, typeof(pp), typeof(default_vcov)}(
        rr, pp, fes,
        data_prep.formula_origin, formula_schema, contrasts,
        esample_final,
        coef_names_str, basis_coef,
        data_prep.nobs, dof_model, dof_fes, dof_residual,
        T(tss_total), T(tss_partial), T(rss),
        T(r2), T(r2_within),
        data_prep.has_intercept,
        default_vcov, vcov_matrix, se, t_stats_val, p_values_val,
        F_stat_robust === nothing ? nothing : T(F_stat_robust),
        p_val_robust === nothing ? nothing : T(p_val_robust)
    )
end
