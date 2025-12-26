##############################################################################
##
## K-Class Estimation (LIML, Fuller, KClass)
##
## Uses the K-class formula: β = [W'W - k*W'Wres]^(-1) * [W'y - k*W'yres]
##
##############################################################################

"""
    fit_kclass_estimator(estimator, df, formula; kwargs...) -> IVEstimator{T}

Internal function to fit K-class IV estimators (LIML, Fuller, KClass).
This function shares most of the data preparation with TSLS, but uses
the K-class coefficient formula instead of two-stage least squares.

# Arguments
- `estimator::AbstractIVEstimator`: One of LIML(), Fuller(a), or KClass(k)
- `df`: DataFrame containing the data
- `formula`: Formula with IV syntax

# Returns
- `IVEstimator{T, V}`: Fitted model with parametric vcov type
"""
function fit_kclass_estimator(
        estimator::AbstractIVEstimator,
        df,
        formula::FormulaTerm;
        contrasts::Dict = Dict{Symbol, Any}(),
        weights::Union{Symbol, Nothing} = nothing,
        save::Union{Bool, Symbol} = :residuals,
        save_cluster::Union{Symbol, Vector{Symbol}, Nothing} = nothing,
        dof_add::Integer = 0,
        method::Symbol = :cpu,
        nthreads::Integer = method == :cpu ? Threads.nthreads() : 256,
        double_precision::Bool = method == :cpu,
        tol::Real = 1e-6,
        maxiter::Integer = 10000,
        drop_singletons::Bool = true,
        progress_bar::Bool = true,
        subset::Union{Nothing, AbstractVector} = nothing,
        first_stage::Bool = true
)
    # Validate save keyword
    save, save_residuals = validate_save_keyword(save)

    # Check nthreads
    nthreads = validate_nthreads(method, nthreads)

    # Convert to DataFrame
    df = DataFrame(df; copycols = false)

    ##############################################################################
    ## Prepare Data (same as TSLS)
    ##############################################################################

    data_prep = prepare_data(df, formula, weights, subset, save, drop_singletons, nthreads)

    # Extract cluster variables
    cluster_data = extract_cluster_variables(df, data_prep.fe_vars, save_cluster, data_prep.esample)

    # Create subsetted dataframe and weights
    if data_prep.has_weights
        wts = Weights(disallowmissing(view(df[!, weights], data_prep.esample)))
    else
        wts = uweights(data_prep.nobs)
    end

    # Create subsetted dataframe
    subdf = _create_subdf(df, data_prep.all_vars, data_prep.esample)
    subfes = isempty(data_prep.fes) ? FixedEffect[] :
             FixedEffect[fe[data_prep.esample] for fe in data_prep.fes]

    ##############################################################################
    ## Create Model Matrices
    ##############################################################################

    T = double_precision ? Float64 : Float32

    # Apply schema for exogenous variables
    s = schema(data_prep.formula, subdf, contrasts)
    formula_schema = apply_schema(data_prep.formula, s, IVEstimator, data_prep.has_fe_intercept)

    y = convert(Vector{T}, response(formula_schema, subdf))
    Xexo = convert(Matrix{T}, modelmatrix(formula_schema, subdf))
    response_name, coefnames_exo = coefnames(formula_schema)

    # Create endogenous and instrument matrices
    formula_endo_schema = apply_schema(data_prep.formula_endo,
        schema(data_prep.formula_endo, subdf, contrasts),
        StatisticalModel)
    Xendo = convert(Matrix{T}, modelmatrix(formula_endo_schema, subdf))
    _, coefnames_endo = coefnames(formula_endo_schema)

    formula_iv_schema = apply_schema(data_prep.formula_iv,
        schema(data_prep.formula_iv, subdf, contrasts),
        StatisticalModel)
    Z = convert(Matrix{T}, modelmatrix(formula_iv_schema, subdf))
    _, coefnames_iv = coefnames(formula_iv_schema)

    # Modify formula schema for prediction
    formula_schema = FormulaTerm(formula_schema.lhs,
        MatrixTerm(tuple(eachterm(formula_schema.rhs)...,
            (term
            for term in eachterm(formula_endo_schema.rhs)
            if term != ConstantTerm(0))...)))

    coef_names = vcat(coefnames_exo, coefnames_endo)

    # Compute total sum of squares
    tss_total = tss(y, data_prep.has_intercept | data_prep.has_fe_intercept, wts)

    # Validate finite values
    all(isfinite, wts) || throw("Weights are not finite")
    all(isfinite, y) || throw("Some observations for the dependent variable are infinite")
    all(isfinite, Xexo) ||
        throw("Some observations for the exogenous variables are infinite")
    all(isfinite, Xendo) ||
        throw("Some observations for the endogenous variables are infinite")
    all(isfinite, Z) ||
        throw("Some observations for the instrumental variables are infinite")

    ##############################################################################
    ## Partial Out Fixed Effects
    ##############################################################################

    oldy, oldX = nothing, nothing
    feM = nothing

    if data_prep.has_fes
        cols = vcat(eachcol(y), eachcol(Xexo), eachcol(Xendo), eachcol(Z))
        colnames = vcat(response_name, coefnames_exo, coefnames_endo, coefnames_iv)

        feM, iterations,
        converged,
        tss_partial,
        oldy_temp,
        oldX_temp = partial_out_fixed_effects!(
            cols, colnames, subfes, wts, method, nthreads,
            tol, maxiter, progress_bar, data_prep.save_fes,
            data_prep.has_intercept, data_prep.has_fe_intercept, T)

        if data_prep.save_fes && oldX_temp !== nothing
            n_exo = size(Xexo, 2)
            oldX = hcat(oldX_temp[:, 1:n_exo], oldX_temp[:, (n_exo + 1):(n_exo + size(Xendo, 2))])
            oldy = oldy_temp
        end
    else
        iterations, converged = 0, true
        tss_partial = tss_total
    end

    ##############################################################################
    ## Apply Weights
    ##############################################################################

    if data_prep.has_weights
        sqrtw = sqrt.(wts)
        y .= y .* sqrtw
        Xexo .= Xexo .* sqrtw
        Xendo .= Xendo .* sqrtw
        Z .= Z .* sqrtw
    end

    ##############################################################################
    ## Collinearity Detection (same as TSLS)
    ##############################################################################

    perm = nothing

    # First pass: check collinearity within Xendo
    XendoXendo = compute_crossproduct(Xendo)
    XendoXendo_check = copy(XendoXendo.data)
    basis_endo = basis!(Symmetric(XendoXendo_check); has_intercept = false)
    if !all(basis_endo)
        Xendo = Xendo[:, basis_endo]
        XendoXendo = Symmetric(XendoXendo.data[basis_endo, basis_endo])
    end

    # Compute cross-products
    XexoXexo = compute_crossproduct(Xexo)
    ZZ = compute_crossproduct(Z)
    XexoZ = Xexo' * Z
    XexoXendo = Xexo' * Xendo
    ZXendo = Z' * Xendo

    # Second pass: joint collinearity check
    k_exo, k_z, k_endo = size(Xexo, 2), size(Z, 2), size(Xendo, 2)
    XexoZXendo = build_block_symmetric(
        [XexoXexo.data, XexoZ, XexoXendo, ZZ.data, ZXendo, XendoXendo.data],
        [k_exo, k_z, k_endo]
    )

    basis_all = basis!(XexoZXendo; has_intercept = data_prep.has_intercept)
    basis_Xexo = basis_all[1:k_exo]
    basis_Z = basis_all[(k_exo + 1):(k_exo + k_z)]
    basis_endo_small = basis_all[(k_exo + k_z + 1):end]

    # Handle recategorization if needed
    if !all(basis_endo_small)
        Xexo = hcat(Xexo, Xendo[:, .!basis_endo_small])
        Xendo = Xendo[:, basis_endo_small]

        XexoXexo = compute_crossproduct(Xexo)
        XexoZ = Xexo' * Z
        XexoXendo = Xexo' * Xendo
        ZXendo = Z' * Xendo
        XendoXendo = compute_crossproduct(Xendo)

        basis_endo2 = trues(length(basis_endo))
        basis_endo2[basis_endo] = basis_endo_small
        ans = collect(1:length(basis_endo))
        ans = vcat(ans[.!basis_endo2], ans[basis_endo2])
        perm = vcat(1:(size(Xexo, 2) - count(.!basis_endo_small)),
            (size(Xexo, 2) - count(.!basis_endo_small)) .+ ans)

        out = join(coefnames_endo[.!basis_endo2], " ")
        @info "Endogenous vars collinear with ivs. Recategorized as exogenous: $(out)"

        k_exo, k_z, k_endo = size(Xexo, 2), size(Z, 2), size(Xendo, 2)
        XexoZXendo = build_block_symmetric(
            [XexoXexo.data, XexoZ, XexoXendo, ZZ.data, ZXendo, XendoXendo.data],
            [k_exo, k_z, k_endo]
        )
        basis_all = basis!(XexoZXendo; has_intercept = data_prep.has_intercept)
        basis_Xexo = basis_all[1:k_exo]
        basis_Z = basis_all[(k_exo + 1):(k_exo + k_z)]
        basis_endo_small2 = basis_all[(k_exo + k_z + 1):end]
    end

    # Apply basis to matrices
    if !all(basis_Xexo)
        Xexo = Xexo[:, basis_Xexo]
        XexoXexo = Symmetric(XexoXexo.data[basis_Xexo, basis_Xexo])
        XexoXendo = XexoXendo[basis_Xexo, :]
    end
    if !all(basis_Z)
        Z = Z[:, basis_Z]
        ZZ = Symmetric(ZZ.data[basis_Z, basis_Z])
        ZXendo = ZXendo[basis_Z, :]
    end
    XexoZ = XexoZ[basis_Xexo, basis_Z]

    # Check identification
    size(ZXendo, 1) >= size(ZXendo, 2) ||
        throw("Model not identified. There must be at least as many instruments as endogenous variables")

    # Compute final basis for coefficients
    basis_endo2 = trues(length(basis_endo))
    basis_endo2[basis_endo] = basis_endo_small
    basis_coef = vcat(basis_Xexo, basis_endo[basis_endo2])

    ##############################################################################
    ## Compute K-class Kappa
    ##############################################################################

    k_exo_final = size(Xexo, 2)
    k_z_final = size(Z, 2)
    k_endo_final = size(Xendo, 2)

    kappa = if estimator isa LIML
        _liml_kappa(y, Xendo, Z, Xexo)
    elseif estimator isa Fuller
        kappa_liml = _liml_kappa(y, Xendo, Z, Xexo)
        # Fuller adjustment: kappa = kappa_liml - a/(n - L - p)
        n = data_prep.nobs
        L = k_z_final  # Number of excluded instruments
        p = k_exo_final  # Number of exogenous variables
        adj_denom = n - L - p
        if adj_denom <= 0
            throw(ArgumentError("Fuller adjustment undefined: n - L - p = $adj_denom <= 0"))
        end
        kappa_liml - T(estimator.a) / adj_denom
    elseif estimator isa KClass
        T(estimator.kappa)
    else
        throw(ArgumentError("Unknown K-class estimator: $(typeof(estimator))"))
    end

    # Warn if kappa < 1 (poor identification)
    if kappa < one(T)
        @warn "K-class kappa < 1 ($kappa), model may be poorly identified"
    end

    ##############################################################################
    ## K-Class Coefficient Estimation
    ##############################################################################

    # Use the core K-class fitting algorithm
    coef_kclass, residuals_kclass, invA, Adj = _kclass_fit(y, Xendo, Z, Xexo, kappa)

    # Reorder coefficients: exogenous first, then endogenous
    # _kclass_fit returns [endogenous, exogenous], but we want [exogenous, endogenous]
    n_endo = size(Xendo, 2)
    n_exo = size(Xexo, 2)
    coef = vcat(coef_kclass[(n_endo + 1):end], coef_kclass[1:n_endo])

    # Also reorder invA and Adj accordingly
    reorder_idx = vcat((n_endo + 1):(n_endo + n_exo), 1:n_endo)
    invA_reordered = Symmetric(invA[reorder_idx, reorder_idx])
    Adj_reordered = Adj[:, reorder_idx]

    # Build X = [Xexo, Xendo] for storing
    X = hcat(Xexo, Xendo)

    # For K-class, Xhat is not meaningful in the same way as TSLS
    # We use X directly but store Adj for vcov calculation
    Xhat = X  # Use original X for prediction purposes

    # Compute XhatXhat for storage (used in some calculations)
    XhatXhat = Symmetric(X' * X)

    ##############################################################################
    ## First-Stage F-Statistics (if requested)
    ##############################################################################

    F_kp, p_kp = T(NaN), T(NaN)
    F_kp_per_endo = T[]
    p_kp_per_endo = T[]
    first_stage_data = nothing

    if first_stage && k_endo_final > 0
        # Compute first-stage quantities (same as TSLS)
        newZ = hcat(Xexo, Z)
        newZXendo = vcat(XexoXendo, ZXendo)

        newZnewZ_aug = build_block_symmetric(
            [XexoXexo.data, XexoZ, XexoXendo, ZZ.data,
                ZXendo, zeros(T, k_endo_final, k_endo_final)],
            [k_exo_final, k_z_final, k_endo_final]
        )
        Pi = ls_solve!(newZnewZ_aug, k_exo_final + k_z_final)

        # Compute residuals for first-stage F-stat
        Xendo_res = BLAS.gemm!('N', 'N', -one(T), newZ, Pi, one(T), copy(Xendo))

        XexoZ_aug = build_block_symmetric(
            [XexoXexo.data, XexoZ, ZZ.data],
            [k_exo_final, k_z_final]
        )
        Pi2 = ls_solve!(XexoZ_aug, k_exo_final)
        Z_res = BLAS.gemm!('N', 'N', -one(T), Xexo, Pi2, one(T), copy(Z))

        Pip = Pi[(k_exo_final + 1):end, :]

        dof_fes_local = sum(nunique(fe) for fe in subfes; init = 0)

        F_kp,
        p_kp = compute_first_stage_fstat(
            Xendo_res, Z_res, Pip,
            CovarianceMatrices.HR1(),
            data_prep.nobs,
            size(X, 2),
            dof_fes_local
        )

        # Pass original data for robust Wald F computation (matches R's approach)
        F_kp_per_endo,
        p_kp_per_endo = compute_per_endogenous_fstats(
            Xendo_res, Z_res, Pip,
            CovarianceMatrices.HR1(),
            data_prep.nobs,
            size(X, 2),
            dof_fes_local;
            Xendo_orig = Xendo,
            newZ = newZ
        )

        endo_names_final = if all(basis_endo)
            collect(String.(coefnames_endo))
        else
            endo_idx = findall(basis_endo)
            if !all(basis_endo_small)
                endo_idx = endo_idx[basis_endo_small]
            end
            [String(coefnames_endo[i]) for i in endo_idx]
        end

        first_stage_data = FirstStageData{T}(
            copy(Pip),
            copy(Xendo_res),
            copy(Z_res),
            endo_names_final,
            k_exo_final,
            copy(Xendo),          # Original endogenous variables
            copy(newZ),           # Full first-stage design [Xexo, Z]
            data_prep.has_intercept
        )
    end

    ##############################################################################
    ## Compute Statistics
    ##############################################################################

    residuals = residuals_kclass
    residuals2 = nothing
    if save_residuals
        residuals2 = Vector{Union{T, Missing}}(missing, data_prep.nrows)
        if data_prep.has_weights
            residuals2[data_prep.esample] .= residuals ./ sqrt.(wts)
        else
            residuals2[data_prep.esample] .= residuals
        end
    end

    # Degrees of freedom
    ngroups_fes = [nunique(fe) for fe in subfes]
    dof_fes = sum(ngroups_fes)
    dof_base = data_prep.nobs - size(X, 2) - dof_fes - dof_add
    dof_residual = max(1, dof_base - (data_prep.has_intercept | data_prep.has_fe_intercept))

    # R-squared
    rss = sum(abs2, residuals)
    r2_within = data_prep.has_fes ? one(T) - rss / tss_partial : one(T) - rss / tss_total

    # F-statistic and p-value
    tss_for_fstat = data_prep.has_fes ? tss_partial : tss_total
    mss_val = tss_for_fstat - rss
    dof_model = size(X, 2) - (data_prep.has_intercept & !data_prep.has_fe_intercept)
    F_stat = dof_model > 0 ? (mss_val / dof_model) / (rss / dof_residual) : T(NaN)
    p_val = dof_model > 0 ? fdistccdf(dof_model, dof_residual, F_stat) : T(NaN)

    ##############################################################################
    ## Handle Omitted Variables
    ##############################################################################

    if !all(basis_coef)
        newcoef = zeros(T, length(basis_coef))
        newindex = [searchsortedfirst(cumsum(basis_coef), i) for i in 1:length(coef)]
        for i in eachindex(newindex)
            newcoef[newindex[i]] = coef[i]
        end
        newcoef[.!basis_coef] .= T(NaN)
        coef = newcoef

        # Also expand invA_reordered and Adj_reordered
        # For omitted variables, set corresponding rows/cols to 0
        k_total = length(basis_coef)
        k_active = sum(basis_coef)

        invA_full = zeros(T, k_total, k_total)
        active_idx = findall(basis_coef)
        invA_full[active_idx, active_idx] .= invA_reordered
        invA_reordered = Symmetric(invA_full)

        Adj_full = zeros(T, size(Adj_reordered, 1), k_total)
        Adj_full[:, active_idx] .= Adj_reordered
        Adj_reordered = Adj_full
    end

    # Handle permutation from recategorization
    if perm !== nothing
        _invperm = invperm(perm)
        coef = coef[_invperm]
        basis_coef = basis_coef[_invperm]
        invA_reordered = Symmetric(invA_reordered.data[_invperm, _invperm])
        Adj_reordered = Adj_reordered[:, _invperm]
    end

    ##############################################################################
    ## Create PostEstimationData
    ##############################################################################

    # For K-class, we store:
    # - X: the adjustment matrix Adj for vcov calculation
    # - Xhat: original X with actual variables
    # - invXX: inv(A) from K-class formula
    # - Adj: the adjustment matrix
    # - kappa: the K-class parameter

    postestimation_data = PostEstimationDataIV(
        convert(Matrix{T}, Adj_reordered),  # X: use Adj for vcov moment matrix
        convert(Matrix{T}, X),               # Xhat: original X
        cholesky(Symmetric(X' * X)),        # crossx
        invA_reordered,                      # invXX: inv(A) for K-class
        wts,
        cluster_data,
        basis_coef,
        first_stage_data,
        convert(Matrix{T}, Adj_reordered),  # Adj: K-class adjustment matrix
        kappa                                # kappa: K-class parameter
    )

    ##############################################################################
    ## Solve for Fixed Effects (if requested)
    ##############################################################################

    augmentdf = DataFrame()
    if data_prep.save_fes && oldy !== nothing
        coef_nonnan = coef[basis_coef]
        newfes, b,
        c = solve_coefficients!(oldy - oldX * coef_nonnan, feM;
            tol = tol, maxiter = maxiter)
        for fekey in data_prep.fekeys
            augmentdf[!, fekey] = df[!, fekey]
        end
        for j in eachindex(subfes)
            augmentdf[!, data_prep.feids[j]] = Vector{Union{T, Missing}}(missing, data_prep.nrows)
            augmentdf[data_prep.esample, data_prep.feids[j]] = newfes[j]
        end
    end

    esample_final = data_prep.esample == Colon() ? trues(data_prep.nrows) :
                    data_prep.esample

    ##############################################################################
    ## Compute Default Vcov (HC1) and Related Statistics
    ##############################################################################

    # For K-class, the vcov "bread" is invA (stored in invXX)
    # and the "meat" uses Adj (stored in X field of postestimation_data)

    # Compute HC1 vcov: invA * (Adj .* u)' * (Adj .* u) * invA * scale
    n = data_prep.nobs
    k = size(X, 2)
    scale = n / dof_residual

    M = Adj_reordered .* residuals
    meat = M' * M
    vcov_matrix = Symmetric(scale .* invA_reordered * meat * invA_reordered)

    # Compute standard errors
    se = sqrt.(diag(vcov_matrix))

    # Compute t-statistics and p-values
    coef_full = copy(coef)
    coef_full[.!basis_coef] .= zero(T)
    t_stats = coef_full ./ se
    p_values = 2 .* tdistccdf.(dof_residual, abs.(t_stats))

    # Compute robust F-statistic (Wald test)
    has_int = hasintercept(data_prep.formula_origin)
    F_stat_robust,
    p_val_robust = compute_robust_fstat(coef_full, vcov_matrix, has_int, dof_residual)

    # Default vcov estimator (HC1)
    default_vcov = CovarianceMatrices.HC1()

    ##############################################################################
    ## Return IVEstimator
    ##############################################################################

    return IVEstimator{T, typeof(default_vcov)}(
        estimator,  # Store the actual estimator (LIML, Fuller, KClass)
        coef,
        esample_final, residuals2, augmentdf,
        postestimation_data,
        data_prep.fekeys, coef_names, response_name,
        data_prep.formula_origin, formula_schema, contrasts,
        data_prep.nobs, dof_model, dof_fes, dof_residual,
        rss, tss_total,
        iterations, converged, r2_within,
        default_vcov, vcov_matrix, se, t_stats, p_values,
        F_stat_robust, p_val_robust, F_kp, p_kp,
        F_kp_per_endo, p_kp_per_endo
    )
end

# Convenience wrappers

"""
    fit_liml(df, formula; kwargs...) -> IVEstimator{T}

Fit an instrumental variables model using LIML (Limited Information Maximum Likelihood).
"""
function fit_liml(df, formula::FormulaTerm; kwargs...)
    return fit_kclass_estimator(LIML(), df, formula; kwargs...)
end

"""
    fit_fuller(df, formula; a=1.0, kwargs...) -> IVEstimator{T}

Fit an instrumental variables model using Fuller's bias-corrected estimator.

# Keyword Arguments
- `a::Real = 1.0`: Fuller bias correction parameter. Fuller(1) is median-unbiased.
"""
function fit_fuller(df, formula::FormulaTerm; a::Real = 1.0, kwargs...)
    return fit_kclass_estimator(Fuller(a), df, formula; kwargs...)
end

"""
    fit_kclass(df, formula; kappa, kwargs...) -> IVEstimator{T}

Fit an instrumental variables model using generic K-class estimator.

# Keyword Arguments
- `kappa::Real`: K-class parameter. k=1 gives TSLS.
"""
function fit_kclass(df, formula::FormulaTerm; kappa::Real, kwargs...)
    return fit_kclass_estimator(KClass(kappa), df, formula; kwargs...)
end
