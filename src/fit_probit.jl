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

    y = modelcols(formula.lhs, df)
    X = modelcols(MatrixTerm(formula.rhs), df)

    X_without_fe = modelcols(MatrixTerm(formula_without_fe.rhs), df)

    schema = StatsModels.schema(formula, df)
    formula_schema = apply_schema(formula, schema)

    response_name = coefnames(formula_schema.lhs)
    Xnames = coefnames(formula_schema.rhs)

    out = DataFrame()

    out[!, response_name] = vec(y)
    for (j, name) in enumerate(Xnames)
        out[!, name] = X[:, j]
    end
    X_final = convert(Matrix{Float64}, X_without_fe)
    return (schema, formula_schema, out, X_final, vec(y))
end

"""
    ignore_fe(f::FormulaTerm) -> FormulaTerm

Return a formula where each `fe(x)` term is converted to the ordinary term `x`.
"""
function ignore_fe(f::FormulaTerm)
    rhs_terms = if hasproperty(f.rhs, :terms)
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
    rhs_terms = if hasproperty(f.rhs, :terms)
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
    formula_schema = apply_schema(formula, schema)
    response_name, coef_names = coefnames(formula_schema.lhs), coefnames(formula_schema.rhs)
    coef_names_str = String[string(name) for name in coef_names]
    return (Symbol(response_name), coef_names_str)
end

############################################################
### Probit-specific likelihood helper
############################################################
"""
    log_likelihood_probit(y, eta)

Return the score, observed information, and log-likelihood contribution for a
single probit observation with response `y` and linear predictor `eta`.
"""
function log_likelihood_probit(y, eta)
    if y == 1
        gi = exp(normlogpdf(eta) - normlogcdf(eta))
        hi = gi^2 + eta * gi
        di = normlogcdf(eta)
    else
        gi = -exp(normlogpdf(eta) - normlogccdf(eta))
        hi = gi^2 + eta * gi
        di = normlogccdf(eta)
    end
    return (gi, hi, di)
end

############################################################
### IRLS workspace
############################################################
"""
    ProbitWorkspace{T, C}

Preallocated buffers reused across IRLS iterations so that the inner loop makes
no per-iteration allocations for the working response, weights, or demeaned
design.

The working response and design columns to be absorbed share a single
`(n × (k+1))` buffer `M`: column 1 holds the working response, columns `2:k+1`
hold the design. `cols` is a concretely typed vector of views into `M`, so
`solve_residuals!` demeans them without dynamic dispatch.
"""
struct ProbitWorkspace{T <: AbstractFloat, C <: AbstractVector{T}}
    g::Vector{T}       # per-observation score
    h::Vector{T}       # per-observation observed information (working weights)
    z::Vector{T}       # working response η + g/h
    M::Matrix{T}       # [working response | design], absorbed in place
    cols::Vector{C}    # views into M's columns for solve_residuals!
    resid::Vector{T}   # z - Xβ, input to the fixed-effect contribution recovery
end

function ProbitWorkspace(X::Matrix{T}) where {T <: AbstractFloat}
    n, k = size(X)
    M = Matrix{T}(undef, n, k + 1)
    C = typeof(view(M, :, 1))
    cols = C[view(M, :, j) for j in 1:(k + 1)]
    return ProbitWorkspace{T, C}(
        Vector{T}(undef, n), Vector{T}(undef, n), Vector{T}(undef, n),
        M, cols, Vector{T}(undef, n))
end

# Views onto the working response and the absorbed design within the shared buffer.
zbuf(ws::ProbitWorkspace) = view(ws.M, :, 1)
Xdbuf(ws::ProbitWorkspace) = view(ws.M, :, 2:size(ws.M, 2))

############################################################
### Weighted least squares on the absorbed design
############################################################
"""
    wls_absorbed!(beta, Xd, z, h) -> beta

Solve the weighted least-squares problem `min_β Σ hᵢ (zᵢ - Xdᵢ'β)²` on the
FE-absorbed design `Xd` and working response `z`, writing the coefficients into
`beta`. Columns of `Xd` that are collinear with the fixed effects have already
been zeroed by the caller, so their coefficients come back as zero.
"""
function wls_absorbed!(beta::Vector{T}, Xd::AbstractMatrix{T}, z::AbstractVector{T},
        h::Vector{T}, basis::BitVector) where {T <: AbstractFloat}
    hz = h .* z
    hX = h .* Xd
    XtWX = Xd' * hX
    XtWz = Xd' * hz
    # FE-collinear columns (basis[j] == false) are excluded from the solve: pin
    # their row/column to a unit diagonal with a zero right-hand side so their
    # coefficient comes back as zero.
    @inbounds for j in axes(XtWX, 1)
        if !basis[j]
            XtWX[:, j] .= zero(T)
            XtWX[j, :] .= zero(T)
            XtWX[j, j] = one(T)
            XtWz[j] = zero(T)
        end
    end
    copyto!(beta, cholesky!(Symmetric(XtWX)) \ XtWz)
    return beta
end

"""
    _probit_irls!(m, ws, feM, has_fes, max_iter, tolerance) -> basis

Run the IRLS loop in place on the model `m`, using the preallocated workspace
`ws` and (optionally) the fixed-effect solver `feM`. Returns the basis mask of
slope columns that survive fixed-effect absorption.

A separate function from `fit_probit` so the hot loop specializes on the
concrete types of `ws` and `feM`.
"""
function _probit_irls!(m::BinaryEstimator, ws::ProbitWorkspace, feM, has_fes::Bool,
        max_iter::Integer, tolerance::Real)
    rr = m.rr
    pp = m.pp
    n = length(rr.y)
    k = size(pp.X, 2)
    # Basis of slope columns that survive absorption. A column collinear with the
    # fixed effects collapses to (near) zero after demeaning; it is detected once
    # on the first iteration and excluded thereafter.
    basis = trues(k)
    basis_detected = !has_fes
    alpha = zeros(n)
    sspre = zeros(k)
    zd = zbuf(ws)
    Xd = Xdbuf(ws)

    for _ in 1:max_iter
        # Score and observed information from the current linear predictor.
        @inbounds for i in eachindex(rr.v)
            gi, hi, _ = rr.v[i]
            ws.g[i] = gi
            ws.h[i] = hi
        end

        # Working response, and the absorption buffer holding [z | X].
        ws.z .= rr.eta .+ ws.g ./ ws.h
        zd .= ws.z
        copyto!(Xd, pp.X)

        if has_fes
            if !basis_detected
                @inbounds for j in 1:k
                    sspre[j] = sum(abs2, view(Xd, :, j))
                end
            end
            FixedEffects.update_weights!(feM, Weights(ws.h))
            solve_residuals!(
                ws.cols, feM; tol = 1e-6, maxiter = 10000, progress_bar = false)
            if !basis_detected
                @inbounds for j in 1:k
                    if sum(abs2, view(Xd, :, j)) < 1e-6 * sspre[j]
                        basis[j] = false
                    end
                end
                basis_detected = true
            end
            # Zero any FE-collinear columns so the WLS coefficient is exactly zero.
            @inbounds for j in 1:k
                basis[j] || (view(Xd, :, j) .= zero(eltype(Xd)))
            end
        end

        wls_absorbed!(pp.beta_new, Xd, zd, ws.h, basis)

        if has_fes
            # The fixed-effect contribution to the linear predictor is the part of
            # the working residual removed by absorption:
            #   α = (z − Xβ) − (z̃ − X̃β)
            # where z̃, X̃ are the demeaned working response and design. Both
            # residuals are already available, so no extra solve is needed.
            mul!(ws.resid, pp.X, pp.beta_new)
            mul!(alpha, Xd, pp.beta_new)
            @inbounds for i in eachindex(alpha)
                alpha[i] = (ws.z[i] - ws.resid[i]) - (zd[i] - alpha[i])
            end
        else
            fill!(alpha, zero(eltype(alpha)))
        end

        update_response!(m, alpha)

        # beta and beta_new are kept as distinct buffers: stephalving! mutates
        # beta_new in place, so beta must be an independent copy.
        converged = abs(rr.deviance_new - rr.deviance) / (0.1 + abs(rr.deviance_new)) <
                    tolerance
        rr.deviance = rr.deviance_new
        copyto!(pp.beta, pp.beta_new)
        converged && break
    end

    return basis
end

############################################################
### FIT PROBIT
############################################################
"""
    fit_probit(data, formula, beta0, max_iter, tolerance) -> BinaryEstimator

Estimate a binary-response probit model, optionally absorbing fixed effects
specified with `fe(...)` terms in `formula`.

The estimator uses iteratively reweighted least squares. Each iteration forms
the working response and weights from the probit score and observed information,
absorbs the fixed effects from both the working response and the design, solves
the weighted least-squares update for the slope coefficients, and recovers the
fixed-effect contribution to the linear predictor. Step-halving guards against
deviance increases.

The fixed-effect solver and the working buffers are constructed once and reused
across iterations; only the observation weights are refreshed each iteration.

# Arguments
- `data`: Input table containing the response, regressors, and fixed-effect
  variables.
- `formula::FormulaTerm`: A `StatsModels.jl` formula, for example
  `@formula(y ~ x1 + x2 + fe(group))`.
- `beta0::Union{Nothing, Vector}`: Initial coefficient vector for the
  non-fixed-effect regressors, or `nothing` for a zero start.
- `max_iter::Integer`: Maximum number of IRLS iterations.
- `tolerance::Real`: Convergence tolerance for the relative deviance change.

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
        tolerance::Real)
    schema, formula_schema, data, X, y = select_columns(data, formula)

    response_name, coef_names_str = get_coefficient_names_nofe(formula, data)

    n = size(data, 1)
    k = count(==(1), y)
    Lnull = k * log(k / n) + (n - k) * log(1 - k / n)
    nulldeviance = -2 * Lnull

    formula, formula_fes = parse_fe(formula)
    fes, feids, fekeys = parse_fixedeffect(data, formula_fes)
    has_fes = has_fe(formula_fes)

    # Fixed effects absorb the constant; without them the model needs an explicit
    # intercept, which the fe-excluded model matrix does not carry.
    if !has_fes
        X = hcat(ones(eltype(X), size(X, 1)), X)
        pushfirst!(coef_names_str, "(Intercept)")
    end

    if beta0 === nothing
        beta0 = zeros(size(X, 2))
    end

    pp = BinaryPredictorQR{Float64}(X, beta0, similar(beta0))

    rr = BinaryResponse(y, pp, response_name)

    m = BinaryEstimator{Float64}(
        rr,
        pp,
        formula,
        formula_schema,
        formula_fes,
        n,
        0,
        0,
        nulldeviance,
        has_fes,
        0,
        coef_names_str,
        trues(length(coef_names_str))
    )
    m.deviance = deviance(m)

    ###############################################
    ############ ESTIMATION LOOP ##################
    ###############################################
    ws = ProbitWorkspace(X)
    # Build the fixed-effect solver once; only its weights change per iteration.
    feM = has_fes ?
          AbstractFixedEffectSolver{Float64}(fes, Weights(ones(n)), Val{:cpu}) : nothing
    basis = _probit_irls!(m, ws, feM, has_fes, max_iter, tolerance)

    ################################
    ## Summary statistics
    ################################
    ngroups_fes = [nunique(fe) for fe in fes]
    dof_fes = sum(ngroups_fes; init = 0)
    m.basis_coef = basis
    m.n_parameters = sum(basis)
    m.fixed_effects_dof = dof_fes

    rr.mu = normcdf.(rr.eta)
    return m
end

"""
    refresh_response!(rr, X, beta, alpha) -> deviance

Recompute the linear predictor `rr.eta = X·β + α` and the per-observation
log-likelihood contributions `rr.v` in place, and return the deviance. Performs
no allocations beyond the reused buffers.
"""
function refresh_response!(rr::BinaryResponse{T}, X, beta, alpha) where {T}
    mul!(rr.eta, X, beta)
    dev = zero(T)
    @inbounds for i in eachindex(rr.eta)
        rr.eta[i] += alpha[i]
        gi, hi, di = log_likelihood_probit(rr.y[i], rr.eta[i])
        rr.v[i] = (gi, hi, di)
        dev -= 2 * di
    end
    return dev
end

"""
    stephalving!(m::BinaryEstimator, alpha)

Apply step-halving to ensure the deviance does not increase. Repeatedly bisects
the step from `pp.beta` to `pp.beta_new` (up to 26 halvings) until
`rr.deviance_new ≤ rr.deviance`.
"""
function stephalving!(m::BinaryEstimator, alpha)
    rr = m.rr
    pp = m.pp
    steps = 0
    while rr.deviance < rr.deviance_new && steps < 26
        pp.beta_new .= (pp.beta .+ pp.beta_new) ./ 2
        rr.deviance_new = refresh_response!(rr, pp.X, pp.beta_new, alpha)
        steps += 1
    end
end

"""
    update_response!(m, alpha)

Update the response object after a predictor step. Recomputes `rr.eta` using
`pp.beta_new` and the fixed-effect contribution `alpha`, refreshes the
log-likelihood contributions `rr.v` and `rr.deviance_new`, then calls
`stephalving!` if needed to enforce a deviance decrease.
"""
function update_response!(m, alpha)
    rr = m.rr
    pp = m.pp
    rr.deviance_new = refresh_response!(rr, pp.X, pp.beta_new, alpha)
    stephalving!(m, alpha)
end
