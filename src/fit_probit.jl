#######################################
### HELPER FUNCTION TO CLEAN DATA
#######################################
"""
    select_columns(df::DataFrame, formula::FormulaTerm)
        -> (schema, formula_schema, data, X_without_fe, y)

Parse `formula` against `df` and return:
- `schema`: the `StatsModels` schema
- `formula_schema`: the schema-applied formula
- `data`: `DataFrame` containing only the columns referenced by `formula`
- `X_without_fe`: model matrix with `fe(...)` terms excluded
- `y`: response vector
"""
function select_columns(df::DataFrame, formula::FormulaTerm)
    formula_without_fe = remove_fixedeffects(formula)
    formula = ignore_fe(formula)
    
    y = modelcols(formula.lhs,df)
    X = modelcols(MatrixTerm(formula.rhs),df)
    
    X_without_fe = modelcols(MatrixTerm(formula_without_fe.rhs),df)

    schema = StatsModels.schema(formula, df)
    formula_schema = apply_schema(formula,schema)

    response_name = coefnames(formula_schema.lhs) 
    Xnames = coefnames(formula_schema.rhs)

    out = DataFrame()

    out[!, response_name] = vec(y)
    for (j,name) in enumerate(Xnames)
        out[!, name] = X[:, j]
    end
    X_final = convert(Matrix{Float64}, X_without_fe) #FIXME it is always float here
    return (schema, formula_schema,out, X_final, vec(y))
    
end

"""
    ignore_fe(f::FormulaTerm) -> FormulaTerm

Return a formula where each `fe(x)` term is converted to the ordinary term `x`.
"""
function ignore_fe(f::FormulaTerm)
    rhs_terms = if hasproperty(f.rhs,:terms)
        f.rhs.terms
    elseif f.rhs isa Tuple
        f.rhs
    else
        (f.rhs,)
    end

    new_rhs = map(rhs_terms) do t
        if t isa FunctionTerm{typeof(fe)}
            term(Symbol(t.args[1]))
        else
            t
        end
    end

    return FormulaTerm(f.lhs, Tuple(new_rhs))
end

"""
    remove_fixedeffects(f::FormulaTerm) -> FormulaTerm

Return a formula with all `fe(...)` terms removed from the right-hand side.
"""
function remove_fixedeffects(f::FormulaTerm)
    rhs_terms = if hasproperty(f.rhs,:terms)
        f.rhs.terms
    elseif f.rhs isa Tuple
        f.rhs
    else
        (f.rhs,)
    end

    new_rhs = filter(rhs_terms) do t
        !(t isa FunctionTerm{typeof(fe)})
    end

    return FormulaTerm(f.lhs, Tuple(new_rhs))
end


"""
    get_coefficient_names_nofe(formula::FormulaTerm, data::DataFrame)

Return the response name and coefficient names after excluding fixed-effect
terms from `formula`.
"""
function get_coefficient_names_nofe(formula::FormulaTerm, data::DataFrame)
    formula = remove_fixedeffects(formula)
    schema = StatsModels.schema(formula, data)
    formula_schema = apply_schema(formula,schema)
    response_name, coef_names = coefnames(formula_schema.lhs), coefnames(formula_schema.rhs)
    coef_names_str = String[string(name) for name = coef_names] 
    return (Symbol(response_name),coef_names_str)
end


############################################################
### OLS SOLVER
############################################################
"""
    ils_solver(X, y; factorization = :auto, collinearity = :qr, tol = 1e-8,
               weights = nothing, has_intercept = true) -> ILSEstimator

Fit the weighted least-squares step used by iterative least squares (ILS).  
This is a lightweight WLS solver for internal probit iterations, returning
an `ILSEstimator` containing the response object, predictor object, and
basis coefficient mask, without full inference results.
"""
function ils_solver(X::AbstractVecOrMat{<:Real}, y::AbstractVector{<:Real};
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

    # Fit using unified solver
    pp, basis_coef,
    _ = fit_ols_core!(rr, X_mat, factorization;
        tol = tol, save_matrices = true, collinearity = collinearity)


    return ILSEstimator{T, typeof(pp)}(
        rr, pp, basis_coef
    )
end

############################################################
### Probit specific likelihood helper
############################################################
"""
    log_likelihood_probit(y, eta)

Return the score, observed information, and log-likelihood contribution for a
single probit observation with response `y` and linear predictor `eta`.
"""
function log_likelihood_probit(y,eta)
    if y == 1
        gi = exp(normlogpdf(eta)-normlogcdf(eta))
        hi = gi^2 + eta * gi
        di = normlogcdf(eta)
    else
        gi = - exp(normlogpdf(eta)-normlogccdf(eta)) 
        hi = gi^2 + eta*gi
        di = normlogccdf(eta)
    end
    return (gi, hi,di)
end

"""
    update_predictor!(m::BinaryEstimator, fes) -> feM

Perform one IRLS predictor update. Computes the working response `tildaz` and
working weights from the current score and observed information, partials out
fixed effects `fes` from both the working response and model matrix, solves the
weighted least-squares problem, and stores the result in `pp.beta_new`.
Returns the `FixedEffectMatrix` `feM` from the demeaning step.
"""
function update_predictor!(m::BinaryEstimator,fes)
        rr = m.rr
        pp = m.pp
        gi = getindex.(rr.v, 1)
        hi = getindex.(rr.v, 2)
        copyto!(pp.tildaX,pp.X)
        pp.tildaz .= rr.eta .+ (gi ./ hi) 
        copyto!(pp.z,pp.tildaz) #dest,source
        cols = Vector{AbstractVector{Float64}}(collect(eachcol(pp.tildaX))) #this is a view so it does not allocate
        pushfirst!(cols, pp.tildaz) 
        feM, _,_,_,_,_ = partial_out_fixed_effects!(
                        cols,
            m.coefnames,
            fes,
            Weights(hi),
            :cpu, # TODO make this an argument,
            Threads.nthreads(),
            1e-6,
            10000,
            true,
            false,
            true, 
            true, 
            Float64,
            false
        ) # this modifies X and z in place

        wls = ils_solver(
            pp.tildaX,
            pp.tildaz,
            weights= hi
        )
        pp.beta_new = coef(wls)
        m.basis_coef = basis_coef(wls)
        return feM
end



"""
    stephalving!(m::BinaryEstimator, alpha_sum)

Apply step-halving to ensure the deviance does not increase. Repeatedly
bisects the step from `pp.beta` to `pp.beta_new` (up to 26 halvings) until
`rr.deviance_new ≤ rr.deviance`.
"""
function stephalving!(m::BinaryEstimator,alpha_sum)
        rr = m.rr
        pp = m.pp
        steps = 0
        while rr.deviance < rr.deviance_new && steps < 26
            @info "Step-halving iteration $(steps), deviance = $(rr.deviance), new deviance = $(rr.deviance_new)"
            pp.beta_new = (pp.beta .+ pp.beta_new) ./ 2
            rr.eta = pp.X * pp.beta_new .+ alpha_sum
            rr.v = log_likelihood_probit.(rr.y,rr.eta)
            rr.deviance_new = deviance(rr)
            steps += 1
        end
end

"""
    update_response!(m, alphanew)

Update the response object after a predictor step. Recomputes `rr.eta` using
`pp.beta_new` and the new fixed-effect sum `alphanew`, refreshes the
log-likelihood contributions `rr.v` and `rr.deviance_new`, then calls
`stephalving!` if needed to enforce deviance decrease.
"""
function update_response!(m, alphanew)
    rr = m.rr
    pp = m.pp
    rr.eta = pp.X * pp.beta_new .+ alphanew
    rr.v = log_likelihood_probit.(rr.y,rr.eta)
    rr.deviance_new = deviance(rr)
    stephalving!(m,alphanew)
end


############################################################
### FIT PROBIT
############################################################
"""
    fit_probit(data, formula, beta0, max_iter, tolerance) -> BinaryEstimator

Estimate a binary-response probit model, optionally absorbing fixed effects
specified with `fe(...)` terms in `formula`.

The estimator uses iterative reweighted least squares. At each iteration, fixed
effects are partialled out before solving the weighted least-squares update for
the slope coefficients.

# Arguments
- `data`: Input table containing the response, regressors, and fixed-effect
  variables.
- `formula::FormulaTerm`: A `StatsModels.jl` formula, for example
  `@formula(y ~ x1 + x2 + fe(group))`.
- `beta0::Vector`: Initial coefficient vector for the non-fixed-effect
  regressors.
- `max_iter::Integer`: Maximum number of IRLS iterations.
- `tolerance::Real`: Convergence tolerance for the deviance update.

# Returns
- `BinaryEstimator`: Fitted model containing the response, fitted
  probabilities, coefficient estimates, model matrix, formula, and coefficient
  names.
"""
function fit_probit(
    @nospecialize(data),
    formula::FormulaTerm,
    beta0::Union{Nothing, Vector},
    max_iter::Integer,
    tolerance::Real #if the difference between the old beta and the new one is below the tolerance stop 
)
    ###############################################
    ###### FORMULA PARSING AND DATA CLEANING ######
    ###############################################



    #parse the formula and return a dataframe with only the needed columns, X::Matrix, y::Vector.
    schema, formula_schema, data, X, y = select_columns(data, formula)
    
    # Put all 0s in beta if user did not input an initial guess
    if beta0 === nothing
        beta0 = zeros(size(X, 2))
    end
    # store coefficient names for model summary, ignroes fe variables
    response_name, coef_names_str = get_coefficient_names_nofe(formula, data)

    #initialize alpha to all 0 and append it to the dataframe
    alpha = zeros(size(X,1))
    
    # vector to store fitted values, now empty
    fitted_probabilities = Vector{Float64}()

    n = size(data,1)  
    k = count(==(1),y) 
    # total sum of squares
    Lnull = k * log(k/n) + (n-k) * log(1 - k/n)
    nulldeviance = -2 * Lnull


    # formula parsing
    formula, formula_fes = parse_fe(formula)
    fes, feids, fekeys = parse_fixedeffect(data, formula_fes)


    ## Instantiate predictor object
    pp = BinaryPredictorQR{Float64,Weights}(
            X,similar(X),
            beta0,similar(beta0),
            similar(beta0),Weights(ones(length(y))),
            similar(X),similar(y), similar(y)
        )


    ## Instantiate response object
    rr = BinaryResponse(y,pp,response_name)
    
    beta = copy(pp.beta)

    m = BinaryEstimator{Float64}(
            rr,                             # rr
            pp,                             # pp
            formula,                        # formula
            formula_schema,                 # formula_schema
            formula_fes,                    # formula_fes
            size(data,1),                   # n_observations
            0,                              # n_parameters
            0,                              # deviance
            nulldeviance,                   # nulldeviance
            has_fe(formula_fes),            # has_fixed_effects
            0,                              # fixed_effects_dof
            coef_names_str,                 # coefnames
            trues(length(coef_names_str))   # basis_coef
        )
        m.deviance = deviance(m)
    
    

    ###############################################
    ############ ESTIMATION LOOP ##################
    ###############################################
    i = 0
    for _ in 1:max_iter
        feM = update_predictor!(m,fes)

        if m.has_fixed_effects
            newfes, _ , _ = solve_coefficients!(
                pp.z - pp.X * pp.beta_new,
                feM;
                tol = 1e-6,
                maxiter = 1000
            )
            alpha = stack(newfes)
            #if there are more than one fe we are interested only in the sum of them
            alpha_sum = alpha isa AbstractVector ? alpha : vec(sum(alpha, dims = 2)) 
        else
            alpha_sum = alpha
        end

        update_response!(m,alpha_sum)
        
        if norm(rr.deviance_new - rr.deviance) / (0.1 + norm(rr.deviance_new))  < tolerance
            pp.beta = pp.beta_new
            break
        end
        rr.deviance = rr.deviance_new
        pp.beta = pp.beta_new
        i += 1
    end

    ################################
    ## Summary statistics
    ################################

    ngroups_fes = [nunique(fe) for fe in fes]
    dof_fes = sum(ngroups_fes)
    dof_model = sum(basis_coef(m))
    m.n_parameters = dof_model
    m.fixed_effects_dof = dof_fes

    rr.mu = normcdf.(rr.eta) #FIXME need to also use alpha??
    println("Convergence reached after $i iterations")
    return m
end
