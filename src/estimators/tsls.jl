##############################################################################
##
## Two-Stage Least Squares (TSLS) Estimation
##
##############################################################################

"""
    fit_tsls(df, formula; kwargs...) -> IVEstimator{T}

Fit an instrumental variables model using Two-Stage Least Squares (TSLS).

This function handles the complete TSLS estimation including:
- Multi-stage collinearity detection
- First-stage regression
- Second-stage regression
- First-stage F-statistics
"""
function fit_tsls(df,
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
        first_stage::Bool = true)

    # Validate save keyword
    save, save_residuals = validate_save_keyword(save)

    # Check nthreads
    nthreads = validate_nthreads(method, nthreads)

    # Convert to DataFrame
    df = DataFrame(df; copycols = false)

    ##############################################################################
    ## Prepare Data
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

    # Create subsetted dataframe - use optimized function
    subdf = _create_subdf(df, data_prep.all_vars, data_prep.esample)
    subfes = isempty(data_prep.fes) ? FixedEffect[] :
             FixedEffect[fe[data_prep.esample] for fe in data_prep.fes]

    ##############################################################################
    ## Create Model Matrices
    ##############################################################################

    # Determine numeric type
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
        # Combine all columns for demeaning
        cols = vcat(eachcol(y), eachcol(Xexo), eachcol(Xendo), eachcol(Z))
        colnames = vcat(response_name, coefnames_exo, coefnames_endo, coefnames_iv)

        # Partial out fixed effects
        feM, iterations,
        converged,
        tss_partial,
        oldy_temp,
        oldX_temp = partial_out_fixed_effects!(
            cols, colnames, subfes, wts, method, nthreads,
            tol, maxiter, progress_bar, data_prep.save_fes,
            data_prep.has_intercept, data_prep.has_fe_intercept, T)

        # For TSLS, we need to combine Xexo and Xendo for oldX
        if data_prep.save_fes && oldX_temp !== nothing
            # oldX_temp contains all X columns; reconstruct as hcat(Xexo, Xendo)
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
    ## Multi-Stage Collinearity Detection for IV
    ##############################################################################

    perm = nothing

    # First pass: check collinearity within Xendo
    XendoXendo = compute_crossproduct(Xendo)
    # Note: basis! modifies its argument, so we need a copy for the first check
    XendoXendo_check = copy(XendoXendo.data)
    basis_endo = basis!(Symmetric(XendoXendo_check); has_intercept = false)
    if !all(basis_endo)
        Xendo = Xendo[:, basis_endo]
        XendoXendo = Symmetric(XendoXendo.data[basis_endo, basis_endo])
    end

    # Compute cross-products (using BLAS where beneficial)
    XexoXexo = compute_crossproduct(Xexo)
    ZZ = compute_crossproduct(Z)
    XexoZ = Xexo' * Z
    XexoXendo = Xexo' * Xendo
    ZXendo = Z' * Xendo

    # Second pass: joint collinearity check for (Xexo, Z, Xendo)
    k_exo, k_z, k_endo = size(Xexo, 2), size(Z, 2), size(Xendo, 2)
    XexoZXendo = build_block_symmetric(
        [XexoXexo.data, XexoZ, XexoXendo, ZZ.data, ZXendo, XendoXendo.data],
        [k_exo, k_z, k_endo]
    )

    basis_all = basis!(XexoZXendo; has_intercept = data_prep.has_intercept)
    basis_Xexo = basis_all[1:k_exo]
    basis_Z = basis_all[(k_exo + 1):(k_exo + k_z)]
    basis_endo_small = basis_all[(k_exo + k_z + 1):end]

    # If adding Xexo and Z makes Xendo collinear, recategorize as exogenous
    if !all(basis_endo_small)
        Xexo = hcat(Xexo, Xendo[:, .!basis_endo_small])
        Xendo = Xendo[:, basis_endo_small]

        # Recompute cross-products for new Xexo
        XexoXexo = compute_crossproduct(Xexo)
        XexoZ = Xexo' * Z
        XexoXendo = Xexo' * Xendo
        ZXendo = Z' * Xendo
        XendoXendo = compute_crossproduct(Xendo)

        # Track permutation for coefficient reordering
        basis_endo2 = trues(length(basis_endo))
        basis_endo2[basis_endo] = basis_endo_small
        ans = collect(1:length(basis_endo))
        ans = vcat(ans[.!basis_endo2], ans[basis_endo2])
        perm = vcat(1:(size(Xexo, 2) - count(.!basis_endo_small)),
            (size(Xexo, 2) - count(.!basis_endo_small)) .+ ans)

        out = join(coefnames_endo[.!basis_endo2], " ")
        @info "Endogenous vars collinear with ivs. Recategorized as exogenous: $(out)"

        # Third pass after recategorization
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
    ## First-Stage Regression: Compute Pi = (Xexo, Z) \ Xendo
    ##############################################################################

    newZ = hcat(Xexo, Z)
    k_exo_final, k_z_final, k_endo_final = size(Xexo, 2), size(Z, 2), size(Xendo, 2)
    newZXendo = vcat(XexoXendo, ZXendo)

    # Build augmented system for ls_solve: [newZ'newZ, newZ'Xendo; Xendo'newZ, 0]
    # Using block symmetric builder (only upper triangle matters for ls_solve!)
    newZnewZ_aug = build_block_symmetric(
        [XexoXexo.data, XexoZ, XexoXendo, ZZ.data,
            ZXendo, zeros(T, k_endo_final, k_endo_final)],
        [k_exo_final, k_z_final, k_endo_final]
    )
    Pi = ls_solve!(newZnewZ_aug, k_exo_final + k_z_final)

    # Predicted endogenous variables
    newnewZ = newZ * Pi

    ##############################################################################
    ## Second-Stage Regression using Xhat
    ##############################################################################

    Xhat = hcat(Xexo, newnewZ)
    XexoNewnewZ = Xexo' * newnewZ
    newnewZnewnewZ = compute_crossproduct(newnewZ)

    # Original X with actual endogenous variables (for storing)
    X = hcat(Xexo, Xendo)

    # Build augmented system for second stage: [Xhat'Xhat, Xhat'y; y'Xhat, 0]
    Xhaty = Xhat' * y
    k_xhat = size(Xhat, 2)
    XhatXhat_aug = build_block_symmetric(
        [XexoXexo.data, XexoNewnewZ, reshape(Xhaty[1:k_exo_final], :, 1),
            newnewZnewnewZ.data, reshape(Xhaty[(k_exo_final + 1):end], :, 1),
            zeros(T, 1, 1)],
        [k_exo_final, k_endo_final, 1]
    )
    invsym!(XhatXhat_aug; diagonal = 1:k_xhat)
    invXhatXhat = Symmetric(.- XhatXhat_aug.data[1:k_xhat, 1:k_xhat])
    coef = XhatXhat_aug.data[1:k_xhat, end]

    # Also build XhatXhat for storage
    XhatXhat = build_block_symmetric(
        [XexoXexo.data, XexoNewnewZ, newnewZnewnewZ.data],
        [k_exo_final, k_endo_final]
    )

    ##############################################################################
    ## First-Stage F-Statistics (if requested)
    ##############################################################################

    F_kp, p_kp = T(NaN), T(NaN)
    F_kp_per_endo = T[]
    p_kp_per_endo = T[]
    first_stage_data = nothing

    if first_stage && size(Xendo, 2) > 0
        # Compute residuals for first-stage F-stat: Xendo - newZ * Pi
        Xendo_res = BLAS.gemm!('N', 'N', -one(T), newZ, Pi, one(T), copy(Xendo))

        # Partial out Z w.r.t. Xexo using block system
        XexoZ_aug = build_block_symmetric(
            [XexoXexo.data, XexoZ, ZZ.data],
            [k_exo_final, k_z_final]
        )
        Pi2 = ls_solve!(XexoZ_aug, k_exo_final)
        Z_res = BLAS.gemm!('N', 'N', -one(T), Xexo, Pi2, one(T), copy(Z))

        # Extract the relevant part of Pi (instruments only)
        Pip = Pi[(k_exo_final + 1):end, :]

        # Compute DOF for fixed effects
        dof_fes_local = sum(nunique(fe) for fe in subfes; init = 0)

        # Compute joint first-stage F-statistic using Kleibergen-Paap rank test
        F_kp,
        p_kp = compute_first_stage_fstat(
            Xendo_res, Z_res, Pip,
            CovarianceMatrices.HR1(),
            data_prep.nobs,
            size(X, 2),
            dof_fes_local
        )

        # Compute per-endogenous first-stage F-statistics
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

        # Store first-stage data for recomputation with different vcov
        # Get endogenous variable names (handle basis filtering)
        endo_names_final = if all(basis_endo)
            collect(String.(coefnames_endo))
        else
            # Filter names to match the retained endogenous variables
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

    # Residuals
    residuals = y - X * coef
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

    # F-statistic and p-value for the model
    # For models with FE, use within TSS; otherwise use total TSS
    tss_for_fstat = data_prep.has_fes ? tss_partial : tss_total
    mss_val = tss_for_fstat - rss
    # Numerator DOF excludes intercept only if explicit intercept (not absorbed by FE)
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
        # Set omitted coefficients to NaN
        newcoef[.!basis_coef] .= T(NaN)
        coef = newcoef
    end

    # Handle permutation from recategorization
    if perm !== nothing
        _invperm = invperm(perm)
        coef = coef[_invperm]
        basis_coef = basis_coef[_invperm]
    end

    ##############################################################################
    ## Create PostEstimationData for vcov calculations
    ##############################################################################

    postestimation_data = PostEstimationDataIV(
        convert(Matrix{T}, Xhat),  # Store Xhat for inference
        convert(Matrix{T}, X),     # Store original X with actual endogenous
        cholesky(Symmetric(XhatXhat)),
        invXhatXhat,
        wts,
        cluster_data,
        basis_coef,
        first_stage_data,  # First-stage data for F-stat recomputation
        nothing,  # Adj (K-class only, not needed for TSLS)
        nothing   # kappa (K-class only, not needed for TSLS)
    )

    ##############################################################################
    ## Solve for Fixed Effects (if requested)
    ##############################################################################

    augmentdf = DataFrame()
    if data_prep.save_fes && oldy !== nothing
        # Use non-NaN coefficients for FE computation
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

    # Handle Colon case for esample
    esample_final = data_prep.esample == Colon() ? trues(data_prep.nrows) :
                    data_prep.esample

    ##############################################################################
    ## Compute Default Vcov (HC1) and Related Statistics
    ##############################################################################

    # Compute HC1 vcov matrix directly from components
    # Use residuals (Vector{T}) not residuals2 (Vector{Union{T, Missing}})
    vcov_matrix = compute_hc1_vcov_direct_iv(
        convert(Matrix{T}, Xhat), residuals, invXhatXhat, basis_coef,
        data_prep.nobs, dof_model, dof_fes, dof_residual
    )

    # Compute standard errors
    se = sqrt.(diag(vcov_matrix))

    # Compute t-statistics and p-values
    # Use coef with zeros for collinear entries
    coef_full = copy(coef)
    coef_full[.!basis_coef] .= zero(T)
    t_stats = coef_full ./ se
    p_values = 2 .* tdistccdf.(dof_residual, abs.(t_stats))

    # Compute robust F-statistic (Wald test) using vcov
    has_int = hasintercept(data_prep.formula_origin)
    F_stat_robust,
    p_val_robust = compute_robust_fstat(
        coef_full, vcov_matrix, has_int, dof_residual
    )

    # Default vcov estimator (HC1)
    default_vcov = CovarianceMatrices.HC1()

    ##############################################################################
    ## Return IVEstimator
    ##############################################################################

    return IVEstimator{T, typeof(default_vcov)}(
        TSLS(),
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
        F_kp_per_endo, p_kp_per_endo  # Per-endogenous first-stage F-stats
    )
end
