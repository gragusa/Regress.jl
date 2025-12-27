##############################################################################
##
## Common Utilities for OLS and IV Fitting
##
##############################################################################

##############################################################################
##
## Validation Functions for Common Keywords
##
##############################################################################

"""
    validate_save_keyword(save)

Validate and normalize the save keyword argument.
Returns a normalized Symbol and a boolean indicating whether residuals should be saved.

Valid options:
- `:all` or `true`: Save residuals, FE estimates, and model matrices (X, y, mu)
- `:residuals`: Save residuals and model matrices
- `:fe`: Save FE estimates and model matrices
- `:none` or `false`: Don't save residuals or FE estimates, but keep model matrices
- `:minimal`: Don't store model matrices (X, y, mu) - smallest memory footprint
"""
function validate_save_keyword(save::Union{Bool, Symbol})
    # Normalize save keyword
    if save == true
        save = :all
    elseif save == false
        save = :none
    end

    if save ∉ (:all, :residuals, :fe, :none, :minimal)
        throw(ArgumentError("save keyword must be :all, :none, :residuals, :fe, or :minimal"))
    end

    save_residuals = (save == :residuals) | (save == :all)
    return save, save_residuals
end

"""
    validate_nthreads(method, nthreads)

Validate and adjust nthreads based on method and available threads.
Returns the validated nthreads value.
"""
function validate_nthreads(method::Symbol, nthreads::Integer)
    if method == :cpu && nthreads > Threads.nthreads()
        @warn "nthreads = $(nthreads) is ignored (Julia started with only $(Threads.nthreads()) threads)"
        nthreads = Threads.nthreads()
    end
    return nthreads
end

##############################################################################
##
## Helper Functions for Predict - Check for FE/continuous interactions
##
##############################################################################

"""
    is_cont_fe_int(x)

Check if a term is an interaction between a continuous variable and a fixed effect.
Used by predict() to detect unsupported formula structures.
"""
function is_cont_fe_int(x)
    x isa InteractionTerm || return false
    any(x -> isa(x, Term), x.terms) &&
        any(x -> isa(x, FunctionTerm{typeof(fe), Vector{Term}}), x.terms)
end

"""
    has_cont_fe_interaction(x::FormulaTerm)

Check if a formula has interactions between continuous variables and fixed effects.
Returns true if such interactions exist (currently not supported in predict).
"""
function has_cont_fe_interaction(x::FormulaTerm)
    if x.rhs isa Term
        is_cont_fe_int(x)
    elseif hasfield(typeof(x.rhs), :lhs)
        false
    else
        any(is_cont_fe_int, x.rhs)
    end
end

"""
    prepare_data(df, formula, weights, subset, save, drop_singletons, nthreads)

Prepare data for estimation: parse formula, create esample, handle weights, etc.
Returns a named tuple with all prepared data.
"""
function prepare_data(df::DataFrame,
        formula::FormulaTerm,
        weights::Union{Symbol, Nothing},
        subset::Union{Nothing, AbstractVector},
        save::Symbol,
        drop_singletons::Bool,
        nthreads::Integer)
    nrows = size(df, 1)

    # Parse formula components
    formula_origin = formula
    if !omitsintercept(formula) & !hasintercept(formula)
        formula = FormulaTerm(formula.lhs, InterceptTerm{true}() + formula.rhs)
    end

    formula, formula_endo, formula_iv, formula_iv_fe = parse_iv(formula)
    has_iv_flag = formula_iv != FormulaTerm(ConstantTerm(0), ConstantTerm(0)) ||
                  formula_iv_fe != FormulaTerm(ConstantTerm(0), ConstantTerm(0))
    has_fe_iv = formula_iv_fe != FormulaTerm(ConstantTerm(0), ConstantTerm(0))

    formula, formula_fes = parse_fe(formula)
    has_fes = formula_fes != FormulaTerm(ConstantTerm(0), ConstantTerm(0))
    save_fes = (save == :fe) | ((save == :all) & has_fes)
    has_weights_flag = weights !== nothing

    # Parse fixed effects
    fes, feids, fekeys = parse_fixedeffect(df, formula_fes)
    has_fe_intercept = any(fe.interaction isa UnitWeights for fe in fes)

    # Remove intercept if absorbed by fixed effects
    if has_fe_intercept
        formula = FormulaTerm(formula.lhs,
            tuple(InterceptTerm{false}(),
                (term
                for term in eachterm(formula.rhs)
                if !isa(term, Union{ConstantTerm, InterceptTerm}))...))
    end
    has_intercept = hasintercept(formula)

    # Collect all variables
    exo_vars = unique(StatsModels.termvars(formula))
    iv_vars = unique(StatsModels.termvars(formula_iv))
    iv_fe_vars = unique(StatsModels.termvars(formula_iv_fe))
    endo_vars = unique(StatsModels.termvars(formula_endo))
    fe_vars = unique(StatsModels.termvars(formula_fes))
    all_vars = unique(vcat(exo_vars, endo_vars, iv_vars, iv_fe_vars, fe_vars))

    # Create estimation sample
    esample = completecases(df, all_vars)
    if has_weights_flag
        esample .&= BitArray(!ismissing(x) && (x > 0) for x in df[!, weights])
    end
    if subset !== nothing
        if length(subset) != nrows
            throw("df has $(nrows) rows but the subset vector has $(length(subset)) elements")
        end
        esample .&= BitArray(!ismissing(x) && x for x in subset)
    end

    # Drop singletons
    n_singletons = 0
    if drop_singletons
        n_singletons = drop_singletons!(esample, fes, nthreads)
    end

    nobs = sum(esample)
    (nobs > 0) || throw("sample is empty")
    (nobs < nrows) || (esample = Colon())

    return (formula = formula,
        formula_origin = formula_origin,
        formula_endo = formula_endo,
        formula_iv = formula_iv,
        formula_iv_fe = formula_iv_fe,
        formula_fes = formula_fes,
        has_iv = has_iv_flag,
        has_fe_iv = has_fe_iv,
        has_fes = has_fes,
        has_intercept = has_intercept,
        has_fe_intercept = has_fe_intercept,
        has_weights = has_weights_flag,
        save_fes = save_fes,
        fes = fes,
        feids = feids,
        fekeys = fekeys,
        exo_vars = exo_vars,
        iv_vars = iv_vars,
        iv_fe_vars = iv_fe_vars,
        endo_vars = endo_vars,
        fe_vars = fe_vars,
        all_vars = all_vars,
        esample = esample,
        nobs = nobs,
        nrows = nrows,
        n_singletons = n_singletons)
end

"""
    _create_subdf(df, all_vars, esample) -> DataFrame

Create a subsetted DataFrame with only the necessary columns.
Optimized to avoid unnecessary disallowmissing calls when column already non-missing.
"""
function _create_subdf(df::DataFrame,
        all_vars::Vector{Symbol},
        esample::Union{BitVector, Colon})
    # Pre-allocate column storage
    cols = Vector{Any}(undef, length(all_vars))

    @inbounds for (i, x) in enumerate(all_vars)
        col = df[!, x]
        # Subset if needed
        subcol = esample isa Colon ? col : view(col, esample)
        # Only call disallowmissing if column type includes Missing
        if eltype(subcol) >: Missing
            cols[i] = disallowmissing(subcol)
        else
            # Already non-missing, just materialize the view
            cols[i] = collect(subcol)
        end
    end

    # Construct DataFrame directly from columns (avoids NamedTuple overhead)
    return DataFrame(cols, all_vars; copycols = false)
end

"""
    extract_cluster_variables(df, fe_vars, save_cluster, esample)

Extract and store cluster variables for post-estimation vcov calculations.
"""
function extract_cluster_variables(df::DataFrame,
        fe_vars::Vector{Symbol},
        save_cluster::Union{Symbol, Vector{Symbol}, Nothing},
        esample::Union{BitVector, Colon})

    # Auto-detect cluster variables from fe() terms
    cluster_vars_to_save = copy(fe_vars)

    # Add user-specified cluster variables
    if save_cluster !== nothing
        if save_cluster isa Symbol
            push!(cluster_vars_to_save, save_cluster)
        else
            append!(cluster_vars_to_save, save_cluster)
        end
    end
    cluster_vars_to_save = unique(cluster_vars_to_save)

    # Store cluster variables (subsetted to esample)
    if isempty(cluster_vars_to_save)
        return NamedTuple()
    else
        cluster_arrays = []
        cluster_names = Symbol[]
        for var in cluster_vars_to_save
            if hasproperty(df, var)
                # Subset to esample (handle Colon case)
                if esample isa Colon
                    push!(cluster_arrays, Vector(df[!, var]))
                else
                    push!(cluster_arrays, Vector(view(df, esample, var)))
                end
                push!(cluster_names, var)
            else
                @warn "Cluster variable :$var not found in dataframe, skipping"
            end
        end
        return NamedTuple{Tuple(cluster_names)}(Tuple(cluster_arrays))
    end
end

"""
    create_model_matrices(subdf, formula, formula_endo, formula_iv, contrasts, has_intercept, has_fe_intercept)

Create model matrices X, y, Z (instruments), Xendo (endogenous) from the prepared data.
Returns a named tuple with matrices and metadata.
"""
function create_model_matrices(subdf::DataFrame,
        formula::FormulaTerm,
        formula_endo::FormulaTerm,
        formula_iv::FormulaTerm,
        contrasts::Dict,
        has_iv::Bool,
        OLSEstimatorType::Type)

    # Apply schema and create main model matrix
    formula_schema = apply_schema(formula, schema(formula, subdf, contrasts), OLSEstimatorType, false)
    formula_schema_fe = apply_schema(formula, schema(formula, subdf, contrasts), OLSEstimatorType, true)

    # Response and exogenous variables
    y = response(formula_schema, subdf)
    Xexo = modelmatrix(formula_schema, subdf)

    response_name = coefnames(formula_schema)[1]
    coef_names = coefnames(formula_schema)[2:end]

    if has_iv
        # IV model matrices
        formula_iv_schema = apply_schema(
            formula_iv, schema(formula_iv, subdf, contrasts), OLSEstimatorType, false)
        Z = modelmatrix(formula_iv_schema, subdf)
        coef_names_iv = coefnames(formula_iv_schema)

        formula_endo_schema = apply_schema(
            formula_endo, schema(formula_endo, subdf, contrasts), OLSEstimatorType, false)
        Xendo = modelmatrix(formula_endo_schema, subdf)
        coef_names_endo = coefnames(formula_endo_schema)

        return (y = y,
            Xexo = Xexo,
            Z = Z,
            Xendo = Xendo,
            formula_schema = formula_schema,
            formula_schema_fe = formula_schema_fe,
            response_name = response_name,
            coef_names = coef_names,
            coef_names_iv = coef_names_iv,
            coef_names_endo = coef_names_endo)
    else
        # OLS model matrices
        return (y = y,
            Xexo = Xexo,
            formula_schema = formula_schema,
            formula_schema_fe = formula_schema_fe,
            response_name = response_name,
            coef_names = coef_names)
    end
end

"""
    create_fixed_effect_solver(fes, weights, nobs, method, double_precision, nthreads)

Create the fixed effects solver object.
"""
function create_fixed_effect_solver(subfes::Vector{<:FixedEffect},
        weights::AbstractWeights,
        nobs::Int,
        method::Symbol,
        double_precision::Bool,
        nthreads::Integer)
    if length(subfes) > 0
        feM = AbstractFixedEffectSolver{double_precision ? Float64 : Float32}(
            subfes, weights, Val{method}, nthreads)
    else
        feM = nothing
    end
    return feM
end

"""
    demean_variables!(y, X, feM, tol, maxiter, progress_bar)

Demean variables with respect to fixed effects in place.
Returns (y, X, iterations, converged, tss_partial).
"""
function demean_variables!(y::AbstractVector{T},
        X::AbstractMatrix{T},
        feM::Union{AbstractFixedEffectSolver, Nothing},
        tol::Real,
        maxiter::Integer,
        progress_bar::Bool) where {T}
    if feM !== nothing
        oldy = copy(y)
        oldX = copy(X)

        y, b,
        c = solve_residuals!(
            y, feM; tol = tol, maxiter = maxiter, progress_bar = progress_bar)
        iterations = b
        converged = c

        X, b,
        c = solve_residuals!(
            X, feM; tol = tol, maxiter = maxiter, progress_bar = progress_bar)

        # Compute TSS after partialling out fixed effects
        tss_partial = zero(T)
        @inbounds @simd for i in eachindex(y)
            tss_partial += abs2(y[i])
        end

        return y, X, oldy, oldX, iterations, converged, tss_partial
    else
        iterations = 0
        converged = true
        tss_partial = zero(T)
        @inbounds @simd for i in eachindex(y)
            tss_partial += abs2(y[i])
        end

        return y, X, nothing, nothing, iterations, converged, tss_partial
    end
end

"""
    compute_tss_total(y, weights, has_intercept)

Compute total sum of squares, accounting for intercept and weights.
"""
function compute_tss_total(y::AbstractVector{T},
        weights::AbstractWeights,
        has_intercept::Bool) where {T}
    if has_intercept
        return tss(y, weights)
    else
        tss_total = zero(T)
        @inbounds @simd for i in eachindex(y)
            tss_total += abs2(y[i]) * weights[i]
        end
        return tss_total
    end
end

"""
    compute_vcov_and_fstat(coef, X, residuals, dof_residual, has_intercept)

Compute variance-covariance matrix and F-statistic.
"""
function compute_vcov_and_fstat(coef::Vector{T},
        invXX::Symmetric{T, Matrix{T}},
        residuals::AbstractVector{T},
        dof_residual::Int,
        has_intercept::Bool) where {T}

    # Compute variance
    rss_value = zero(T)
    @inbounds @simd for i in eachindex(residuals)
        rss_value += abs2(residuals[i])
    end

    s2 = rss_value / dof_residual
    matrix_vcov = Symmetric(s2 * invXX)

    # Compute F-statistic
    F = Fstat(coef, matrix_vcov, has_intercept)

    return matrix_vcov, F
end

"""
    solve_fixed_effects!(residuals, feM, oldy, oldX, coef, feids, tol, maxiter)

Solve for fixed effects and create augmented dataframe.
"""
function solve_fixed_effects!(residuals::AbstractVector,
        feM::Union{AbstractFixedEffectSolver, Nothing},
        oldy::Union{AbstractVector, Nothing},
        oldX::Union{AbstractMatrix, Nothing},
        coef::Vector,
        fekeys::Vector{Symbol},
        feids::Vector,
        subfes::Vector{<:FixedEffect},
        save_fes::Bool,
        df::DataFrame,
        esample::Union{BitVector, Colon},
        nrows::Int,
        tol::Real,
        maxiter::Integer)
    augmentdf = DataFrame()
    if save_fes && feM !== nothing
        newfes, b,
        c = solve_coefficients!(oldy - oldX * coef, feM; tol = tol, maxiter = maxiter)
        for fekey in fekeys
            augmentdf[!, fekey] = df[!, fekey]
        end
        for j in eachindex(subfes)
            augmentdf[!, feids[j]] = Vector{Union{Float64, Missing}}(missing, nrows)
            augmentdf[esample, feids[j]] = newfes[j]
        end
    end
    return augmentdf
end

"""
    partial_out_fixed_effects!(cols, colnames, subfes, wts, method, nthreads,
                                tol, maxiter, progress_bar, save_fes, has_intercept, has_fe_intercept)

Partial out fixed effects from a collection of columns (y and X matrices).
This is the common FE demeaning logic shared by both fit_ols and fit_tsls.

Returns:
- feM: Fixed effect solver object (or nothing if no FEs)
- iterations: Maximum number of iterations
- converged: Boolean indicating convergence
- tss_partial: TSS after partialing out FEs
- oldy, oldX: Copies of y and X before demeaning (for FE estimation, if save_fes=true)
"""
function partial_out_fixed_effects!(cols::Vector,
        colnames::Vector,
        subfes::Vector{<:FixedEffect},
        wts::AbstractWeights,
        method::Symbol,
        nthreads::Integer,
        tol::Real,
        maxiter::Integer,
        progress_bar::Bool,
        save_fes::Bool,
        has_intercept::Bool,
        has_fe_intercept::Bool,
        T::Type)

    # Initialize return values
    iterations, converged = 0, true
    oldy, oldX = nothing, nothing
    feM = nothing

    if !isempty(subfes)
        # Save copies if needed for FE estimation
        if save_fes
            oldy = deepcopy(cols[1])
            if length(cols) > 1
                oldX = hcat([deepcopy(col) for col in cols[2:end]]...)
            end
        end

        # Store pre-demeaning sum of squares for collinearity detection
        sumsquares_pre = [sum(abs2, x) for x in cols]

        # Create FE solver
        feM = AbstractFixedEffectSolver{T}(subfes, wts, Val{method}, nthreads)

        # Partial out fixed effects
        _, iterations,
        convergeds = solve_residuals!(cols, feM;
            maxiter = maxiter,
            tol = tol,
            progress_bar = progress_bar)

        # Check for collinearity with FEs
        for i in 1:length(cols)
            if sum(abs2, cols[i]) < tol * sumsquares_pre[i]
                if i == 1
                    @info "Dependent variable $(colnames[1]) is probably perfectly explained by fixed effects."
                else
                    @info "RHS-variable $(colnames[i]) is collinear with the fixed effects."
                    cols[i] .= zero(T)
                end
            end
        end

        iterations = maximum(iterations)
        converged = all(convergeds)
        if !converged
            @info "Convergence not achieved in $(iterations) iterations; try increasing maxiter or decreasing tol."
        end

        # Compute TSS after partialing out
        tss_partial = tss(cols[1], has_intercept | has_fe_intercept, wts)
    else
        tss_partial = tss(cols[1], has_intercept | has_fe_intercept, wts)
    end

    return feM, iterations, converged, tss_partial, oldy, oldX
end
