#######################################
### HELPER FUNCTION TO CLEAN DATA
#######################################
"""
    select_columns(df::DataFrame, formula::FormulaTerm)
        -> (schema, formula_schema, data, X_without_fe, y)

Build model columns and formula metadata from `df`. Return the schema and
schema-applied formula after converting standalone `fe(...)` terms to ordinary
terms, a DataFrame reconstructed from the response and model columns, the
`Matrix{Float64}` design excluding standalone fixed-effect terms, and the
response vector. The returned DataFrame is not a direct subset of `df`.
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

# Constant-outcome intercept groups have unbounded fixed effects (see #25).
function _binary_sample(y::AbstractVector, fes)
    Base.require_one_based_indexing(y)
    for yi in y
        yi in (0, 1) || throw(DomainError(yi, "Bernoulli response must be zero or one"))
    end
    sample = trues(length(y))
    reasons = fill(:none, length(y))
    intercept_fes = filter(fe -> fe.interaction isa UnitWeights, fes)
    caches = [zeros(UInt8, fe.n) for fe in intercept_fes]
    while true
        removed = 0
        for (fe, cache) in zip(intercept_fes, caches)
            removed += _drop_binary_separated!(sample, reasons, y, fe, cache)
        end
        removed == 0 && return sample, reasons
    end
end

function _drop_binary_separated!(sample, reasons, y, fe::FixedEffect, cache)
    refs = fe.refs
    fill!(cache, 0)
    # The two bits record whether a retained group contains zeros and ones.
    for i in eachindex(sample, y, refs)
        if sample[i]
            cache[refs[i]] |= y[i] == 0 ? 0x01 : 0x02
        end
    end
    removed = 0
    for i in eachindex(sample, reasons, refs)
        if sample[i]
            state = cache[refs[i]]
            if state == 0x01 || state == 0x02
                sample[i] = false
                reasons[i] = state == 0x01 ? :fe_all_zero : :fe_all_one
                removed += 1
            end
        end
    end
    return removed
end

############################################################
### IRLS workspace for Fisher Scoring
############################################################
"""
    GLMWorkspace(X::Matrix{T})

Allocate buffers for Bernoulli Fisher scoring with `n` observations and `k`
design columns. Buffers are uninitialized until filled by the fitting loop.

- `g`: response residuals `y - mu`.
- `d`: inverse-link derivatives `dmu_deta`.
- `h`: Fisher weights `d^2 / (mu * one_minus_mu)`.
- `z`: working response `eta + g / d`.
- `M`: shared `n × (k + 1)` absorption buffer containing `[z X]`.
- `cols`: views of the columns of `M` for the fixed-effect solver.
- `resid`: temporary storage for `h .* z`, fixed-effect recovery, and the
  accepted predictor during step-halving;
  unlike `rr.resid`, this is not the response residual vector.
- `hX`: weighted design buffer with size `n × k`.
- `XtWX`, `XtWz`: normal-equation buffers with sizes `k × k` and `k`.

The response and residual buffers have length `n`. Arithmetic in the formulas
is elementwise.
"""
struct GLMWorkspace{T <: AbstractFloat, C <: AbstractVector{T}}
    g::Vector{T}       # response residual y - mu
    d::Vector{T}       # inverse-link derivative dmu_deta
    h::Vector{T}       # Fisher weights dmu_deta^2 / Var(y)
    z::Vector{T}       # working response η + g/d
    M::Matrix{T}       # [working response | design], absorbed in place
    cols::Vector{C}    # views into M's columns for solve_residuals!
    resid::Vector{T}   # scratch for weighted least squares, fixed effects, and step-halving
    hX::Matrix{T}
    XtWX::Matrix{T}
    XtWz::Vector{T}
end

function GLMWorkspace(X::Matrix{T}) where {T <: AbstractFloat}
    n, k = size(X)
    M = Matrix{T}(undef, n, k + 1)
    C = typeof(view(M, :, 1))
    cols = C[view(M, :, j) for j in 1:(k + 1)]
    return GLMWorkspace{T, C}(
        Vector{T}(undef, n), Vector{T}(undef, n), Vector{T}(undef, n), Vector{T}(undef, n),
        M, cols, Vector{T}(undef, n), Matrix{T}(undef, n, k),
        Matrix{T}(undef, k, k), Vector{T}(undef, k))
end

############################################################
### IRLS workspace
############################################################
"""
    ProbitWorkspace{T, C}

Legacy workspace for score/observed-information updates, with working response
`eta + g / h`. The current `probit` fitting path uses `GLMWorkspace`
instead. Buffers are allocated once and are initially uninitialized.

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
zbuf(ws::Union{ProbitWorkspace, GLMWorkspace}) = view(ws.M, :, 1)
Xdbuf(ws::Union{ProbitWorkspace, GLMWorkspace}) = view(ws.M, :, 2:size(ws.M, 2))

# Bernoulli residuals use the complement directly when the response is one.
function _working_values(y, eta, mu, one_minus_mu, d, ::ProbitLink)
    if mu > 0 && one_minus_mu > 0 && d > 0
        # Dividing before multiplying avoids underflow in d^2 (see #25).
        h = (d / mu) * (d / one_minus_mu)
        correction = (y == 1 ? one_minus_mu : -mu) / d
    else
        log_mu = StatsFuns.normlogcdf(eta)
        log_complement = StatsFuns.normlogccdf(eta)
        log_d = normlogpdf(eta)
        h = exp(2 * log_d - log_mu - log_complement)
        correction = y == 1 ? exp(log_complement - log_d) : -exp(log_mu - log_d)
    end
    return h, correction
end

function _working_values(y, eta, mu, one_minus_mu, d, ::LogitLink)
    return d, y == 1 ? inv(mu) : -inv(one_minus_mu)
end

function _working_response!(ws::GLMWorkspace, rr::BinaryResponse)
    copyto!(ws.g, rr.resid)
    copyto!(ws.d, rr.dmu_deta)
    for i in eachindex(ws.h, ws.z, rr.y, rr.eta, rr.mu, rr.one_minus_mu, rr.dmu_deta)
        h, correction = _working_values(rr.y[i], rr.eta[i], rr.mu[i],
            rr.one_minus_mu[i], rr.dmu_deta[i], rr.link)
        z = rr.eta[i] + correction
        isfinite(h) && h > 0 && isfinite(z) || throw(DomainError(rr.eta[i],
            "Fisher weight or working response is outside the representable range; rescale predictors or use a different beta0"))
        ws.h[i] = h
        ws.z[i] = z
    end
    return ws
end

############################################################
### Weighted least squares on the absorbed design
############################################################
"""
    wls_absorbed!(beta, Xd, z, h, basis) -> beta

Solve `min_β Σ hᵢ (zᵢ - Xdᵢ'β)²` and write the coefficients into `beta`.
`Xd` and `z` must already have fixed effects absorbed, if applicable.
`basis` selects retained columns; excluded coefficients are set to zero.

Form the weighted normal equations and solve them with Cholesky factorization.
The retained system must be positive definite. Optional scratch buffers `hz`,
`hX`, `XtWX`, and `XtWz` are overwritten; omitted buffers are allocated. Buffers
must match the input axes and must not alias the inputs. Inputs must use
one-based indexing. Only `beta` and the scratch buffers are mutated.
"""
function wls_absorbed!(beta::Vector{T}, Xd::AbstractMatrix{T}, z::AbstractVector{T},
        h::Vector{T}, basis::BitVector;
        hz = similar(z), hX = similar(Xd),
        XtWX = similar(beta, T, (length(beta), length(beta))),
        XtWz = similar(beta)) where {T <: AbstractFloat}
    Base.require_one_based_indexing(beta, Xd, z, h, basis, hz, hX, XtWX, XtWz)
    axes(Xd, 1) == axes(z, 1) == axes(h, 1) == axes(hz, 1) ||
        throw(DimensionMismatch("The design, response, weights, and weighted response buffer must have matching row axes"))
    axes(Xd, 2) == axes(beta, 1) == axes(basis, 1) == axes(XtWz, 1) ||
        throw(DimensionMismatch("The design, coefficients, basis, and normal-equation response buffer must have matching column axes"))
    axes(hX) == axes(Xd) ||
        throw(DimensionMismatch("The weighted design buffer must match the design axes"))
    axes(XtWX) == (axes(Xd, 2), axes(Xd, 2)) ||
        throw(DimensionMismatch("The normal-equation matrix buffer must match the design column axes"))
    hz .= h .* z
    hX .= h .* Xd
    mul!(XtWX, Xd', hX)
    mul!(XtWz, Xd', hz)
    # FE-collinear columns (basis[j] == false) are excluded from the solve: pin
    # their row/column to a unit diagonal with a zero right-hand side so their
    # coefficient comes back as zero.
    for j in eachindex(beta, basis)
        if !basis[j]
            view(XtWX, :, j) .= zero(T)
            view(XtWX, j, :) .= zero(T)
            XtWX[j, j] = one(T)
            XtWz[j] = zero(T)
        end
    end
    copyto!(beta, XtWz)
    ldiv!(cholesky!(Symmetric(XtWX)), beta)
    return beta
end

"""
    _glm_irls!(m, ws, feM, has_fes, max_iter, tolerance) -> basis

Run Bernoulli Fisher scoring in place using `GLMWorkspace` and the optional
fixed-effect solver `feM`. At each iteration, form the working response
`z = eta + (y - mu) / dmu_deta` and weights
`h = dmu_deta^2 / (mu * one_minus_mu)`, elementwise.

Absorb fixed effects when present, solve weighted least squares, recover the
fixed-effect contribution, and update the response with step-halving.
Stop when `abs(Dnew - Dold) / (0.1 + abs(Dnew)) < tolerance`.
Throw if the iteration limit is reached, fixed-effect absorption fails to
converge, or working quantities exceed the representable range.

Return a Boolean mask of retained design columns. Columns nearly annihilated
by fixed-effect absorption are detected on the first iteration and excluded
thereafter; without fixed effects the mask contains only `true` values.
The separate function permits specialization on workspace and solver types.
"""
function _glm_irls!(m::BinaryEstimator, ws::GLMWorkspace, feM, has_fes::Bool,
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
        _working_response!(ws, rr)

        # Working response, and the absorption buffer holding [z | X].
        zd .= ws.z
        copyto!(Xd, pp.X)

        if has_fes
            if !basis_detected
                @inbounds for j in 1:k
                    sspre[j] = sum(abs2, view(Xd, :, j))
                end
            end
            FixedEffects.update_weights!(feM, Weights(ws.h))
            _, iterations, convergeds = solve_residuals!(
                ws.cols, feM; tol = 1e-6, maxiter = 10000, progress_bar = false)
            all(convergeds) || error(
                "Fixed-effect absorption did not converge in $(maximum(iterations)) iterations")
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

        wls_absorbed!(pp.beta_new, Xd, zd, ws.h, basis;
            hz = ws.resid, hX = ws.hX, XtWX = ws.XtWX, XtWz = ws.XtWz)

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

        update_response!(m, alpha; eta_old = ws.resid)

        # beta and beta_new are kept as distinct buffers: stephalving! mutates
        # beta_new in place, so beta must be an independent copy.
        converged = abs(rr.deviance_new - rr.deviance) / (0.1 + abs(rr.deviance_new)) <
                    tolerance
        rr.deviance = rr.deviance_new
        copyto!(pp.beta, pp.beta_new)
        if converged
            m.deviance = rr.deviance
            return basis
        end
    end

    error("Fisher scoring did not converge in $max_iter iterations; increase max_iter or inspect the model for separation")
end

############################################################
### FIT BERNOULLI GLM
############################################################
"""
    fit_glmfe(data::DataFrame, formula, beta0, max_iter, tolerance, link) -> BinaryEstimator

Fit a Bernoulli model with the selected probit or logit link by Fisher scoring,
optionally absorbing fixed effects specified by `fe(...)` terms. The response must contain only zeros
and ones. Data preparation currently requires a `DataFrame`.

Each iteration forms working responses and Fisher weights, absorbs fixed
effects, solves weighted least squares, and recovers the fixed-effect
contribution. Workspace buffers and the fixed-effect solver are reused;
solver weights are updated each iteration. Step-halving shrinks the slope
and fixed-effect updates together until the deviance is finite and does not
increase, with at most 1024 halvings per update.

# Arguments
- `link`: `ProbitLink()` or `LogitLink()`.
- `data`: DataFrame containing the response, regressors, and fixed-effect variables.
- `formula`: Model formula, for example `@formula(y ~ x + fe(group))`.
- `beta0`: Initial coefficient vector, or `nothing` for zeros. Its length must
  match the design columns, including the intercept added without fixed effects.
- `max_iter`: Maximum number of Fisher-scoring iterations.
- `tolerance`: Threshold for `abs(Dnew - Dold) / (0.1 + abs(Dnew))`.

Without fixed effects, an explicit intercept is prepended to the design.
With fixed effects, the constant is absorbed. Return a `BinaryEstimator` with
coefficients, response buffers, formula metadata, and parameter counts.
Remove observations in all-zero or all-one fixed-effect intercept groups
recursively before fitting. `esample` and `drop_reasons` refer to the original
rows; response and predictor buffers contain only the estimation sample.
Failure of step-halving or convergence raises an error. Working quantities
outside the representable floating-point range raise a `DomainError`.
"""
function fit_glmfe(
        @nospecialize(data),
        formula::FormulaTerm,
        beta0::Union{Nothing, Vector},
        max_iter::Integer,
        tolerance::Real,
        link::Link)
    max_iter > 0 || throw(ArgumentError("max_iter must be positive"))
    isfinite(tolerance) && tolerance > 0 ||
        throw(ArgumentError("tolerance must be finite and positive"))
    schema, formula_schema, data, X, y = select_columns(data, formula)

    response_name, coef_names_str = get_coefficient_names_nofe(formula, data)

    formula, formula_fes = parse_fe(formula)
    fes, feids, fekeys = parse_fixedeffect(data, formula_fes)
    has_fes = has_fe(formula_fes)

    sample, drop_reasons = _binary_sample(y, fes)
    n = count(sample)
    n > 0 || throw(ArgumentError(
        "No observations remain after removing constant-outcome fixed-effect groups"))
    if !all(sample)
        X = X[sample, :]
        y = y[sample]
        fes = FixedEffect[FixedEffect(fe.refs[sample]; interaction = fe.interaction[sample])
                          for fe in fes]
    end
    k = count(==(1), y)
    0 < k < n || throw(ArgumentError("Bernoulli response must contain both zeros and ones"))
    Lnull = k * log(k / n) + (n - k) * log1p(-k / n)
    nulldeviance = -2 * Lnull

    # Fixed effects absorb the constant; without them the model needs an explicit
    # intercept, which the fe-excluded model matrix does not carry.
    if !has_fes
        X = hcat(ones(eltype(X), size(X, 1)), X)
        pushfirst!(coef_names_str, "(Intercept)")
    end

    if beta0 === nothing
        beta0 = zeros(size(X, 2))
    end
    all(isfinite, beta0) || throw(ArgumentError("beta0 must contain only finite values"))

    pp = BinaryPredictorQR{Float64}(X, beta0, similar(beta0))

    rr = BinaryResponse(y, pp, response_name, link)
    isfinite(rr.deviance) || throw(ArgumentError(
        "Initial deviance must be finite; check predictors and beta0"))

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
        trues(length(coef_names_str)),
        sample,
        drop_reasons
    )
    m.deviance = deviance(m)

    ###############################################
    ############ ESTIMATION LOOP ##################
    ###############################################
    ws = GLMWorkspace(X)
    # Build the fixed-effect solver once; only its weights change per iteration.
    feM = has_fes ?
          AbstractFixedEffectSolver{Float64}(fes, Weights(ones(n)), Val{:cpu}) : nothing
    basis = _glm_irls!(m, ws, feM, has_fes, max_iter, tolerance)

    ################################
    ## Summary statistics
    ################################
    dof_fes = _fe_dof(fes)
    m.basis_coef = basis
    m.n_parameters = sum(basis)
    m.fixed_effects_dof = dof_fes

    return m
end

"""
    refresh_response!(rr, X, beta, alpha) -> deviance

Set `rr.eta = X * beta + alpha`, then update `mu`, `one_minus_mu`,
`dmu_deta`, `resid`, and `deviance_contributions` using the selected inverse link
and Bernoulli deviance. Reuse the response buffers and return the sum of the
per-observation deviance contributions.

Do not assign `rr.deviance` or `rr.deviance_new`; the caller stores the returned
value so an accepted deviance remains available during candidate evaluation.
"""
function refresh_response!(rr::BinaryResponse{T}, X, beta, alpha) where {T}
    mul!(rr.eta, X, beta)
    rr.eta .+= alpha
    return refresh_response!(rr)
end

function refresh_response!(rr::BinaryResponse)
    inverse_link!(rr)
    rr.resid .= ifelse.(rr.y .== 1, rr.one_minus_mu, .-rr.mu)
    rr.deviance_contributions .= deviance_bernoulli.(rr.y, rr.eta, Ref(rr.link))
    return sum(rr.deviance_contributions)
end

"""
    stephalving!(m::BinaryEstimator, eta_old; max_halvings=1024)

While the candidate deviance is nonfinite or exceeds the accepted deviance,
halve the coefficient update and the full predictor update relative to
`eta_old`. This shrinks both slope and fixed-effect contributions.

Attempt at most `max_halvings` halvings. Update the candidate coefficients, response
buffers, and `rr.deviance_new`, leaving `pp.beta` and `rr.deviance` unchanged.
Throw if no finite, non-increasing deviance is obtained.
"""
function stephalving!(m::BinaryEstimator, eta_old; max_halvings = 1024)
    rr = m.rr
    pp = m.pp
    steps = 0
    while !isfinite(rr.deviance_new) || rr.deviance_new > rr.deviance
        steps < max_halvings || error(
            "Step-halving failed to find a finite, non-increasing deviance after $max_halvings halvings")
        pp.beta_new .= pp.beta ./ 2 .+ pp.beta_new ./ 2
        rr.eta .= eta_old ./ 2 .+ rr.eta ./ 2
        rr.deviance_new = refresh_response!(rr)
        steps += 1
    end
    return m
end

"""
    update_response!(m, alpha; eta_old=similar(m.rr.eta), max_halvings=1024)

Evaluate `pp.beta_new` with the supplied fixed-effect contribution `alpha`.
Refresh the linear predictor, probabilities, derivative, residuals, and
per-observation deviance contributions, and store their sum in `rr.deviance_new`.

Overwrite the scratch buffer `eta_old` with the accepted predictor before
candidate evaluation. Call `stephalving!` to shrink the full update if needed.
Leave acceptance of the candidate coefficients and deviance to the caller.
"""
function update_response!(m, alpha; eta_old = similar(m.rr.eta), max_halvings = 1024)
    rr = m.rr
    pp = m.pp
    copyto!(eta_old, rr.eta)
    rr.deviance_new = refresh_response!(rr, pp.X, pp.beta_new, alpha)
    stephalving!(m, eta_old; max_halvings)
end

"""
    inverse_link!(rr::BinaryResponse{T, ProbitLink}) -> rr

Update `rr.mu`, `rr.one_minus_mu`, and `rr.dmu_deta` from `rr.eta` in place.
Use the standard normal CDF, complementary CDF, and PDF, respectively.
The complementary CDF avoids cancellation when `mu` is close to one.
"""
function inverse_link!(rr::BinaryResponse{T, ProbitLink}) where {T}
    rr.mu .= normcdf.(rr.eta)
    rr.one_minus_mu .= StatsFuns.normccdf.(rr.eta)
    rr.dmu_deta .= exp.(normlogpdf.(rr.eta))
    return rr
end

"""
    inverse_link!(rr::BinaryResponse{T, LogitLink}) -> rr

Update the response probabilities and inverse-link derivative for the logit
link: `mu = logistic(eta)`, `one_minus_mu = logistic(-eta)`, and
`dmu_deta = mu * one_minus_mu`, elementwise. The complement formula avoids
subtraction from one. Return the response object.

Fill the existing `rr.mu`, `rr.one_minus_mu`, and `rr.dmu_deta` buffers in place.
"""
function inverse_link!(rr::BinaryResponse{T, LogitLink}) where {T}
    rr.mu .= StatsFuns.logistic.(rr.eta)
    rr.one_minus_mu .= StatsFuns.logistic.(-rr.eta)
    rr.dmu_deta .= rr.mu .* rr.one_minus_mu
    return rr
end

"""
    link_probit(mu) -> eta

Evaluate the probit link `eta = Φ⁻¹(mu)` for a scalar probability `mu` in
`[0, 1]`, where `Φ` is the standard normal CDF. The endpoints `0` and `1`
map to `-Inf` and `Inf`, respectively.
"""
function link_probit(mu)
    return StatsFuns.norminvcdf(mu)
end

"""
    link_logit(mu) -> eta

Evaluate the logit link `eta = log(mu / (1 - mu))` for a scalar probability
`mu` in `[0, 1]`. The endpoints `0` and `1` map to `-Inf` and `Inf`,
respectively. Use `log1p(-mu)` to evaluate the logarithm of the complement.
"""
function link_logit(mu)
    return log(mu) - log1p(-mu)
end

"""
    var_bernoulli(rr) -> variances

Return a vector of Bernoulli variances `rr.mu .* rr.one_minus_mu`.
The probability buffers must be updated before calling this function.
"""
function var_bernoulli(rr::BinaryResponse)
    return rr.mu .* rr.one_minus_mu
end

"""
    deviance_bernoulli(y, mu, one_minus_mu) -> deviance
    deviance_bernoulli(y, eta, link::Link) -> deviance

Return the deviance contribution of one Bernoulli observation `y` (zero or
one), given its success probability `mu` and complement `one_minus_mu`.
The contribution is `-2 * log(mu)` when `y == 1` and
`-2 * log(one_minus_mu)` when `y == 0`. Sum the contributions to obtain
the total deviance.

Accepting the complement separately allows callers to supply a value computed
with `ccdf`. Branching on `y` avoids `0 * log(0)` at probability endpoints;
an observation assigned zero probability has infinite deviance.

The predictor-and-link form evaluates log probabilities directly, preserving
finite deviance when a tail probability underflows to zero.
"""
function deviance_bernoulli(y, mu, one_minus_mu)
    if y == 1
        return -2 * log(mu)
    elseif y == 0
        return -2 * log(one_minus_mu)
    end
    throw(DomainError(y, "Bernoulli response must be zero or one"))
end

# Evaluate log probabilities before exponentiation can underflow (see #25).
function deviance_bernoulli(y, eta, ::ProbitLink)
    if y == 1
        return -2 * StatsFuns.normlogcdf(eta)
    elseif y == 0
        return -2 * StatsFuns.normlogccdf(eta)
    end
    throw(DomainError(y, "Bernoulli response must be zero or one"))
end

function deviance_bernoulli(y, eta, ::LogitLink)
    if y == 1
        return 2 * StatsFuns.log1pexp(-eta)
    elseif y == 0
        return 2 * StatsFuns.log1pexp(eta)
    end
    throw(DomainError(y, "Bernoulli response must be zero or one"))
end

"""
    fit_probit(data, formula, beta0, max_iter, tolerance) -> BinaryEstimator

Compatibility wrapper for `fit_glmfe` with `ProbitLink()`.
See [`glmfe`](@ref) for the public formula-first interface.
"""
function fit_probit(data, formula, beta0, max_iter, tolerance)
    return fit_glmfe(data, formula, beta0, max_iter, tolerance, ProbitLink())
end
