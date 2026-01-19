##############################################################################
##
## Two-Stage Least Squares (TSLS) Estimation
##
##############################################################################

##############################################################################
## Helper functions for IV estimation (extracted for faster compilation)
##############################################################################

"""
    _iv_collinearity_detection!(Xexo, Xendo, Z, XexoXexo, XendoXendo, ZZ,
                                 has_intercept, coefnames_endo)

Multi-stage collinearity detection for IV models.
Returns updated matrices and basis vectors.
"""
function _iv_collinearity_detection!(
        Xexo::Matrix{T}, Xendo::Matrix{T}, Z::Matrix{T},
        XexoXexo::Symmetric{T}, XendoXendo::Symmetric{T}, ZZ::Symmetric{T},
        has_intercept::Bool, coefnames_endo::Vector
) where {T}
    perm = nothing

    # First pass: check collinearity within Xendo
    XendoXendo_check = copy(XendoXendo.data)
    basis_endo = basis!(Symmetric(XendoXendo_check); has_intercept = false)
    if !all(basis_endo)
        Xendo = Xendo[:, basis_endo]
        XendoXendo = Symmetric(XendoXendo.data[basis_endo, basis_endo])
    end

    # Compute cross-products
    XexoZ = Xexo' * Z
    XexoXendo = Xexo' * Xendo
    ZXendo = Z' * Xendo

    # Second pass: joint collinearity check for (Xexo, Z, Xendo)
    k_exo, k_z, k_endo = size(Xexo, 2), size(Z, 2), size(Xendo, 2)
    XexoZXendo = build_block_symmetric(
        [XexoXexo.data, XexoZ, XexoXendo, ZZ.data, ZXendo, XendoXendo.data],
        [k_exo, k_z, k_endo]
    )

    basis_all = basis!(XexoZXendo; has_intercept = has_intercept)
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
        basis_all = basis!(XexoZXendo; has_intercept = has_intercept)
        basis_Xexo = basis_all[1:k_exo]
        basis_Z = basis_all[(k_exo + 1):(k_exo + k_z)]
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

    # Compute final basis for coefficients
    basis_endo2 = trues(length(basis_endo))
    basis_endo2[basis_endo] = basis_endo_small
    basis_coef = vcat(basis_Xexo, basis_endo[basis_endo2])

    return (
        Xexo = Xexo, Xendo = Xendo, Z = Z,
        XexoXexo = XexoXexo, XendoXendo = XendoXendo, ZZ = ZZ,
        XexoZ = XexoZ, XexoXendo = XexoXendo, ZXendo = ZXendo,
        basis_endo = basis_endo, basis_endo_small = basis_endo_small,
        basis_Xexo = basis_Xexo, basis_Z = basis_Z,
        basis_coef = basis_coef, perm = perm
    )
end

"""
    _iv_first_stage_standard(Xexo, Z, Xendo, XexoXexo, ZZ, XexoXendo, ZXendo, T)

Standard first-stage regression: Pi = (Xexo, Z) \\ Xendo
"""
function _iv_first_stage_standard(
        Xexo::Matrix{T}, Z::Matrix{T}, Xendo::Matrix{T},
        XexoXexo::Symmetric{T}, ZZ::Symmetric{T},
        XexoXendo::Matrix{T}, ZXendo::Matrix{T}
) where {T}
    k_exo, k_z, k_endo = size(Xexo, 2), size(Z, 2), size(Xendo, 2)
    XexoZ = Xexo' * Z

    newZ = hcat(Xexo, Z)

    # Build augmented system for ls_solve
    newZnewZ_aug = build_block_symmetric(
        [XexoXexo.data, XexoZ, XexoXendo, ZZ.data, ZXendo, zeros(T, k_endo, k_endo)],
        [k_exo, k_z, k_endo]
    )
    Pi = ls_solve!(newZnewZ_aug, k_exo + k_z)

    # Predicted endogenous variables
    Xendo_hat = newZ * Pi

    return (newZ = newZ, Pi = Pi, Xendo_hat = Xendo_hat, XexoZ = XexoZ)
end

"""
    _iv_first_stage_fe(Xexo, Xendo, iv_fes, wts, method, nthreads, maxiter, tol, progress_bar, T)

FE-based first-stage regression using Frisch-Waugh-Lovell.
"""
function _iv_first_stage_fe(
        Xexo::Matrix{T}, Xendo::Matrix{T},
        iv_fes::Vector{FixedEffect},
        wts::AbstractWeights,
        method::Symbol, nthreads::Int,
        maxiter::Int, tol::Real, progress_bar::Bool
) where {T}
    k_exo, k_endo = size(Xexo, 2), size(Xendo, 2)
    nobs = size(Xexo, 1)

    # Create FE solver for instrument FEs
    iv_feM = AbstractFixedEffectSolver{T}(iv_fes, wts, Val{method}, nthreads)

    # Step 1: Demean Xexo by instrument FEs
    Xexo_demeaned = copy(Xexo)
    if k_exo > 0
        Xexo_cols = [Xexo_demeaned[:, j] for j in 1:k_exo]
        solve_residuals!(
            Xexo_cols, iv_feM; maxiter = maxiter, tol = tol, progress_bar = progress_bar)
        for j in 1:k_exo
            Xexo_demeaned[:, j] .= Xexo_cols[j]
        end
    end

    # Step 2: Demean Xendo by instrument FEs
    Xendo_demeaned = copy(Xendo)
    Xendo_cols = [Xendo_demeaned[:, j] for j in 1:k_endo]
    solve_residuals!(
        Xendo_cols, iv_feM; maxiter = maxiter, tol = tol, progress_bar = progress_bar)
    for j in 1:k_endo
        Xendo_demeaned[:, j] .= Xendo_cols[j]
    end

    # Step 3: Within-FE regression of Xendo on Xexo
    if k_exo > 0
        XexoXexo_within = compute_crossproduct(Xexo_demeaned)
        XexoXendo_within = Xexo_demeaned' * Xendo_demeaned
        gamma = XexoXexo_within \ XexoXendo_within
        Xendo_within_res = Xendo_demeaned .- Xexo_demeaned * gamma
    else
        gamma = zeros(T, 0, k_endo)
        Xendo_within_res = Xendo_demeaned
    end

    # Step 4: Predicted endogenous = original Xendo - within residuals
    Xendo_hat = Xendo .- Xendo_within_res

    return (newZ = Xexo, Pi = gamma, Xendo_hat = Xendo_hat,
        XexoZ = zeros(T, k_exo, 0),
        Z = Matrix{T}(undef, nobs, 0),
        ZZ = Symmetric(zeros(T, 0, 0)),
        ZXendo = zeros(T, 0, k_endo))
end

"""
    _iv_second_stage(Xexo, Xendo_hat, y, XexoXexo, T)

Second-stage IV regression using predicted endogenous variables.
"""
function _iv_second_stage(
        Xexo::Matrix{T}, Xendo_hat::Matrix{T}, y::Vector{T},
        XexoXexo::Symmetric{T}
) where {T}
    k_exo, k_endo = size(Xexo, 2), size(Xendo_hat, 2)

    Xhat = hcat(Xexo, Xendo_hat)
    XexoXendo_hat = Xexo' * Xendo_hat
    Xendo_hatXendo_hat = compute_crossproduct(Xendo_hat)

    # Build augmented system for second stage
    Xhaty = Xhat' * y
    k_xhat = k_exo + k_endo
    XhatXhat_aug = build_block_symmetric(
        [XexoXexo.data, XexoXendo_hat, reshape(Xhaty[1:k_exo], :, 1),
            Xendo_hatXendo_hat.data, reshape(Xhaty[(k_exo + 1):end], :, 1),
            zeros(T, 1, 1)],
        [k_exo, k_endo, 1]
    )
    invsym!(XhatXhat_aug; diagonal = 1:k_xhat)
    invXhatXhat = Symmetric(.- XhatXhat_aug.data[1:k_xhat, 1:k_xhat])
    coef = XhatXhat_aug.data[1:k_xhat, end]

    # Build XhatXhat for storage
    XhatXhat = build_block_symmetric(
        [XexoXexo.data, XexoXendo_hat, Xendo_hatXendo_hat.data],
        [k_exo, k_endo]
    )

    return (Xhat = Xhat, coef = coef, invXhatXhat = invXhatXhat, XhatXhat = XhatXhat)
end

"""
    _iv_first_stage_fstats_standard(...)

Compute first-stage F-statistics for standard IV.
"""
function _iv_first_stage_fstats_standard(
        Xendo::Matrix{T}, newZ::Matrix{T}, Pi::Matrix{T},
        XexoXexo::Symmetric{T}, XexoZ::Matrix{T}, ZZ::Symmetric{T},
        X::Matrix{T}, nobs::Int, dof_fes::Int,
        endo_names::Vector{String}, k_exo::Int, has_intercept::Bool
) where {T}
    n = size(Xendo, 1)
    k_endo = size(Xendo, 2)
    k_z = size(ZZ, 1)

    # Compute residuals for first-stage F-stat: Xendo_res = Xendo - newZ * Pi
    # Pre-allocate and compute in one step
    Xendo_res = copy(Xendo)
    BLAS.gemm!('N', 'N', -one(T), newZ, Pi, one(T), Xendo_res)

    # Partial out Z w.r.t. Xexo: Z_res = Z - Xexo * Pi2
    XexoZ_aug = build_block_symmetric([XexoXexo.data, XexoZ, ZZ.data], [k_exo, k_z])
    Pi2 = ls_solve!(XexoZ_aug, k_exo)

    # Extract Z from newZ without allocating a new matrix
    # newZ = [Xexo, Z], so Z = newZ[:, (k_exo+1):end]
    # Compute Z_res = Z - Xexo * Pi2 using views where possible
    Xexo_view = view(newZ, :, 1:k_exo)
    Z_res = newZ[:, (k_exo + 1):end]  # This creates a copy (needed for gemm!)
    BLAS.gemm!('N', 'N', -one(T), Xexo_view, Pi2, one(T), Z_res)

    # Extract the relevant part of Pi (instruments only)
    Pip = Pi[(k_exo + 1):end, :]

    # Compute joint first-stage F-statistic
    F_kp,
    p_kp = compute_first_stage_fstat(
        Xendo_res, Z_res, Pip,
        CovarianceMatrices.HR1(),
        nobs, size(X, 2), dof_fes
    )

    # Compute per-endogenous F-statistics
    F_kp_per_endo,
    p_kp_per_endo = compute_per_endogenous_fstats(
        Xendo_res, Z_res, Pip,
        CovarianceMatrices.HR1(),
        nobs, size(X, 2), dof_fes;
        Xendo_orig = Xendo, newZ = newZ
    )

    # Store first-stage data - reuse existing matrices where possible
    # Pip, Xendo_res, Z_res are already computed and not needed elsewhere
    # Xendo and newZ are passed by reference, only copy if we must preserve them
    first_stage_data = FirstStageData{T}(
        Pip, Xendo_res, Z_res,
        endo_names, k_exo, Xendo, newZ, has_intercept
    )

    return (F_kp = F_kp, p_kp = p_kp,
        F_kp_per_endo = F_kp_per_endo, p_kp_per_endo = p_kp_per_endo,
        first_stage_data = first_stage_data)
end

"""
    _iv_first_stage_fstats_fe(...)

Compute first-stage F-statistics for FE-based IV.
"""
function _iv_first_stage_fstats_fe(
        Xendo::Matrix{T}, Xendo_hat::Matrix{T},
        iv_fes::Vector{FixedEffect},
        newZ::Matrix{T}, nobs::Int, k_exo::Int,
        endo_names::Vector{String}, has_intercept::Bool
) where {T}
    k_endo = size(Xendo, 2)
    Xendo_res = Xendo .- Xendo_hat
    n_iv_fe_groups = sum(nunique(fe) for fe in iv_fes; init = 0)

    F_kp_per_endo = T[]
    p_kp_per_endo = T[]

    for j in 1:k_endo
        rss_j = sum(abs2, Xendo_res[:, j])
        tss_j = sum(abs2, Xendo[:, j] .- mean(Xendo[:, j]))
        ess_j = tss_j - rss_j

        df1 = n_iv_fe_groups - 1
        df2 = nobs - n_iv_fe_groups - k_exo

        if df1 > 0 && df2 > 0
            F_j = (ess_j / df1) / (rss_j / df2)
            p_j = fdistccdf(df1, df2, F_j)
            push!(F_kp_per_endo, F_j)
            push!(p_kp_per_endo, p_j)
        else
            push!(F_kp_per_endo, T(NaN))
            push!(p_kp_per_endo, T(NaN))
        end
    end

    # Joint F
    if k_endo == 1
        F_kp, p_kp = F_kp_per_endo[1], p_kp_per_endo[1]
    else
        F_kp = k_endo / sum(1 ./ F_kp_per_endo)
        df1 = n_iv_fe_groups - 1
        df2 = nobs - n_iv_fe_groups - k_exo
        p_kp = fdistccdf(df1, df2, F_kp)
    end

    # Reuse existing matrices - Xendo_res is already a copy, Xendo and newZ can be shared
    first_stage_data = FirstStageData{T}(
        zeros(T, 0, k_endo), Xendo_res, zeros(T, nobs, 0),
        endo_names, k_exo, Xendo, newZ, has_intercept
    )

    return (F_kp = F_kp, p_kp = p_kp,
        F_kp_per_endo = F_kp_per_endo, p_kp_per_endo = p_kp_per_endo,
        first_stage_data = first_stage_data)
end

"""
    _iv_compute_inference(Xhat, residuals_raw, invXhatXhat, basis_coef, coef,
                          nobs, dof_model, dof_fes, dof_residual, formula_origin, T)

Compute vcov matrix, standard errors, t-stats, p-values, and robust F-stat.
"""
function _iv_compute_inference(
        Xhat::Matrix{T}, residuals_raw::Vector{T},
        invXhatXhat::Symmetric{T}, basis_coef::BitVector, coef::Vector{T},
        nobs::Int, dof_model::Int, dof_fes::Int, dof_residual::Int,
        formula_origin
) where {T}
    # Compute HC1 vcov
    vcov_matrix = compute_hc1_vcov_direct_iv(
        Xhat, residuals_raw, invXhatXhat, basis_coef,
        nobs, dof_model, dof_fes, dof_residual
    )

    # Standard errors
    se = sqrt.(diag(vcov_matrix))

    # t-stats and p-values
    coef_full = copy(coef)
    coef_full[.!basis_coef] .= zero(T)
    t_stats = coef_full ./ se
    p_values = 2 .* tdistccdf.(dof_residual, abs.(t_stats))

    # Robust F-statistic
    has_int = hasintercept(formula_origin)
    F_stat_robust,
    p_val_robust = compute_robust_fstat(coef_full, vcov_matrix, has_int, dof_residual)

    return (vcov_matrix = vcov_matrix, se = se, t_stats = t_stats,
        p_values = p_values, F_stat_robust = F_stat_robust,
        p_val_robust = p_val_robust)
end

##############################################################################
##
## Main fit_tsls function
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
function fit_tsls(@nospecialize(df),
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

    # Validate inputs
    save, save_residuals = validate_save_keyword(save)
    nthreads = validate_nthreads(method, nthreads)
    contrasts = convert(Dict{Symbol, Any}, contrasts)
    df = DataFrame(df; copycols = false)

    # Prepare data
    data_prep = prepare_data(df, formula, weights, subset, save, drop_singletons, nthreads)
    cluster_data = extract_cluster_variables(df, data_prep.fe_vars, save_cluster, data_prep.esample)

    # Create weights
    wts = data_prep.has_weights ?
          Weights(disallowmissing(view(df[!, weights], data_prep.esample))) :
          uweights(data_prep.nobs)

    # Create subsetted data
    subdf = _create_subdf(df, data_prep.all_vars, data_prep.esample)
    subfes = isempty(data_prep.fes) ? FixedEffect[] :
             FixedEffect[fe[data_prep.esample] for fe in data_prep.fes]

    # Determine numeric type
    T = double_precision ? Float64 : Float32

    ##########################################################################
    ## Create Model Matrices
    ##########################################################################

    s = schema(data_prep.formula, subdf, contrasts)
    formula_schema = apply_schema(data_prep.formula, s, IVEstimator, data_prep.has_fe_intercept)

    y = convert(Vector{T}, response(formula_schema, subdf))
    Xexo = convert(Matrix{T}, modelmatrix(formula_schema, subdf))
    response_name, coefnames_exo = coefnames(formula_schema)

    # Endogenous variables
    formula_endo_schema = apply_schema(data_prep.formula_endo,
        schema(data_prep.formula_endo, subdf, contrasts), StatisticalModel)
    Xendo = convert(Matrix{T}, modelmatrix(formula_endo_schema, subdf))
    _, coefnames_endo = coefnames(formula_endo_schema)

    # Instruments
    has_regular_iv = data_prep.formula_iv != FormulaTerm(ConstantTerm(0), ConstantTerm(0))
    if has_regular_iv
        formula_iv_schema = apply_schema(data_prep.formula_iv,
            schema(data_prep.formula_iv, subdf, contrasts), StatisticalModel)
        Z = convert(Matrix{T}, modelmatrix(formula_iv_schema, subdf))
        _, coefnames_iv = coefnames(formula_iv_schema)
    else
        Z = Matrix{T}(undef, data_prep.nobs, 0)
        coefnames_iv = String[]
    end

    # FE instruments
    has_fe_iv = data_prep.has_fe_iv
    iv_fes = FixedEffect[]
    if has_fe_iv
        iv_fes, _, _ = parse_fixedeffect(subdf, data_prep.formula_iv_fe)
        has_fe_iv = !isempty(iv_fes)
    end

    # Update formula schema for prediction
    # Use concrete vector to avoid runtime dispatch
    schema_terms = collect(AbstractTerm, eachterm(formula_schema.rhs))
    for term in eachterm(formula_endo_schema.rhs)
        if term != ConstantTerm(0)
            push!(schema_terms, term)
        end
    end
    formula_schema = FormulaTerm(formula_schema.lhs, MatrixTerm(Tuple(schema_terms)))
    coef_names = vcat(coefnames_exo, coefnames_endo)

    # Compute TSS and validate
    tss_total = tss(y, data_prep.has_intercept | data_prep.has_fe_intercept, wts)
    all(isfinite, wts) || throw("Weights are not finite")
    all(isfinite, y) || throw("Some observations for the dependent variable are infinite")
    all(isfinite, Xexo) ||
        throw("Some observations for the exogenous variables are infinite")
    all(isfinite, Xendo) ||
        throw("Some observations for the endogenous variables are infinite")
    all(isfinite, Z) ||
        throw("Some observations for the instrumental variables are infinite")

    ##########################################################################
    ## Partial Out Fixed Effects
    ##########################################################################

    oldy, oldX, feM = nothing, nothing, nothing
    iterations, converged, tss_partial = 0, true, tss_total

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
    end

    # Apply weights
    if data_prep.has_weights
        sqrtw = sqrt.(wts)
        y .= y .* sqrtw
        Xexo .= Xexo .* sqrtw
        Xendo .= Xendo .* sqrtw
        Z .= Z .* sqrtw
    end

    ##########################################################################
    ## Collinearity Detection
    ##########################################################################

    XexoXexo = compute_crossproduct(Xexo)
    XendoXendo = compute_crossproduct(Xendo)
    ZZ = compute_crossproduct(Z)

    coll = _iv_collinearity_detection!(
        Xexo, Xendo, Z, XexoXexo, XendoXendo, ZZ,
        data_prep.has_intercept, coefnames_endo
    )

    Xexo, Xendo, Z = coll.Xexo, coll.Xendo, coll.Z
    XexoXexo, ZZ = coll.XexoXexo, coll.ZZ
    XexoZ, XexoXendo, ZXendo = coll.XexoZ, coll.XexoXendo, coll.ZXendo
    basis_coef, perm = coll.basis_coef, coll.perm
    basis_endo, basis_endo_small = coll.basis_endo, coll.basis_endo_small

    # Check identification
    if !has_fe_iv
        size(ZXendo, 1) >= size(ZXendo, 2) ||
            throw("Model not identified. There must be at least as many instruments as endogenous variables")
    end

    ##########################################################################
    ## First-Stage Regression
    ##########################################################################

    k_exo_final, k_z_final, k_endo_final = size(Xexo, 2), size(Z, 2), size(Xendo, 2)

    if has_fe_iv
        fs = _iv_first_stage_fe(
            Xexo, Xendo, iv_fes, wts, method, nthreads, maxiter, tol, progress_bar)
        newZ, Pi, Xendo_hat = fs.newZ, fs.Pi, fs.Xendo_hat
        Z, ZZ, XexoZ, ZXendo = fs.Z, fs.ZZ, fs.XexoZ, fs.ZXendo
        k_z_final = 0
    else
        fs = _iv_first_stage_standard(Xexo, Z, Xendo, XexoXexo, ZZ, XexoXendo, ZXendo)
        newZ, Pi, Xendo_hat, XexoZ = fs.newZ, fs.Pi, fs.Xendo_hat, fs.XexoZ
    end

    ##########################################################################
    ## Second-Stage Regression
    ##########################################################################

    ss = _iv_second_stage(Xexo, Xendo_hat, y, XexoXexo)
    Xhat, coef, invXhatXhat, XhatXhat = ss.Xhat, ss.coef, ss.invXhatXhat, ss.XhatXhat
    X = hcat(Xexo, Xendo)

    ##########################################################################
    ## First-Stage F-Statistics
    ##########################################################################

    F_kp, p_kp = T(NaN), T(NaN)
    F_kp_per_endo, p_kp_per_endo = T[], T[]
    first_stage_data = empty_first_stage_data(T)

    if first_stage && k_endo_final > 0
        endo_names_final = all(basis_endo) ?
                           collect(String.(coefnames_endo)) :
                           [String(coefnames_endo[i])
                            for i in findall(basis_endo)[basis_endo_small]]

        dof_fes_local = sum(nunique(fe) for fe in subfes; init = 0)

        if has_fe_iv
            fstats = _iv_first_stage_fstats_fe(
                Xendo, Xendo_hat, iv_fes, newZ, data_prep.nobs, k_exo_final,
                endo_names_final, data_prep.has_intercept)
        else
            fstats = _iv_first_stage_fstats_standard(
                Xendo, newZ, Pi, XexoXexo, XexoZ, ZZ, X,
                data_prep.nobs, dof_fes_local, endo_names_final,
                k_exo_final, data_prep.has_intercept)
        end

        F_kp, p_kp = fstats.F_kp, fstats.p_kp
        F_kp_per_endo, p_kp_per_endo = fstats.F_kp_per_endo, fstats.p_kp_per_endo
        first_stage_data = fstats.first_stage_data
    end

    ##########################################################################
    ## Compute Statistics
    ##########################################################################

    residuals_raw = y - X * coef
    residuals_esample = data_prep.has_weights ? residuals_raw ./ sqrt.(wts) : residuals_raw

    ngroups_fes = [nunique(fe) for fe in subfes]
    dof_fes = sum(ngroups_fes)
    dof_base = data_prep.nobs - size(X, 2) - dof_fes - dof_add
    dof_residual = max(1, dof_base - (data_prep.has_intercept | data_prep.has_fe_intercept))

    rss = sum(abs2, residuals_raw)
    r2_within = data_prep.has_fes ? one(T) - rss / tss_partial : one(T) - rss / tss_total

    dof_model = size(X, 2) - (data_prep.has_intercept & !data_prep.has_fe_intercept)
    tss_for_fstat = data_prep.has_fes ? tss_partial : tss_total
    mss_val = tss_for_fstat - rss
    F_stat = dof_model > 0 ? (mss_val / dof_model) / (rss / dof_residual) : T(NaN)
    p_val = dof_model > 0 ? fdistccdf(dof_model, dof_residual, F_stat) : T(NaN)

    ##########################################################################
    ## Handle Omitted Variables
    ##########################################################################

    if !all(basis_coef)
        newcoef = zeros(T, length(basis_coef))
        newindex = [searchsortedfirst(cumsum(basis_coef), i) for i in 1:length(coef)]
        for i in eachindex(newindex)
            newcoef[newindex[i]] = coef[i]
        end
        newcoef[.!basis_coef] .= T(NaN)
        coef = newcoef
    end

    if perm !== nothing
        _invperm = invperm(perm)
        coef = coef[_invperm]
        basis_coef = basis_coef[_invperm]
    end

    ##########################################################################
    ## Create PostEstimationData
    ##########################################################################

    postestimation_data = PostEstimationDataIV(
        convert(Matrix{T}, Xhat), convert(Matrix{T}, X),
        cholesky(Symmetric(XhatXhat)), invXhatXhat, wts, cluster_data,
        basis_coef, first_stage_data,
        Matrix{T}(undef, 0, 0), T(NaN)
    )

    ##########################################################################
    ## Solve for Fixed Effects (if requested)
    ##########################################################################

    augmentdf = DataFrame()
    if data_prep.save_fes && oldy !== nothing
        coef_nonnan = coef[basis_coef]
        newfes, _,
        _ = solve_coefficients!(oldy - oldX * coef_nonnan, feM; tol = tol, maxiter = maxiter)
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

    ##########################################################################
    ## Compute Inference
    ##########################################################################

    inf = _iv_compute_inference(
        convert(Matrix{T}, Xhat), residuals_raw, invXhatXhat, basis_coef, coef,
        data_prep.nobs, dof_model, dof_fes, dof_residual, data_prep.formula_origin)

    ##########################################################################
    ## Return IVEstimator
    ##########################################################################

    return IVEstimator{
        T, TSLS, typeof(CovarianceMatrices.HC1()), typeof(postestimation_data)}(
        TSLS(), coef,
        esample_final, residuals_esample, save_residuals, augmentdf,
        postestimation_data,
        data_prep.fekeys, coef_names, response_name,
        data_prep.formula_origin, formula_schema, contrasts,
        data_prep.nobs, dof_model, dof_fes, dof_residual,
        rss, tss_total,
        iterations, converged, r2_within,
        CovarianceMatrices.HC1(), inf.vcov_matrix, inf.se, inf.t_stats, inf.p_values,
        inf.F_stat_robust, inf.p_val_robust, F_kp, p_kp,
        F_kp_per_endo, p_kp_per_endo
    )
end
