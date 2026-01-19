##############################################################################
##
## Type OLSEstimator (for OLS estimation) - GLM.jl compatible structure
##
##############################################################################

"""
    OLSEstimator <: AbstractRegressModel

Model type for linear regression estimated by ordinary least squares (OLS).

Uses a GLM.jl-compatible structure with separate response (`rr`) and
predictor (`pp`) components, plus econometrics-specific fixed effects (`fes`).

Use `ols(df, formula)` to fit this model type.

# Type Parameters
- `T`: Float type (Float64 or Float32)
- `P`: Predictor type (OLSPredictorChol or OLSPredictorQR)
- `V`: Variance-covariance estimator type (HC1, HC3, CR1, etc.)

# Fields
- `rr::OLSResponse{T}`: Response object (y, fitted values, weights)
- `pp::OLSLinearPredictor{T}`: Predictor object (X, coefficients, factorization)
- `fes::OLSFixedEffects{T}`: Fixed effects component
- `formula::FormulaTerm`: Original formula
- `formula_schema::FormulaTerm`: Schema for predict
- `contrasts::Dict`: Contrasts used
- `esample::BitVector`: Estimation sample indicator
- `coefnames::Vector{String}`: Coefficient names
- `basis_coef::BitVector`: Non-collinear coefficients indicator
- `nobs::Int`: Number of observations
- `dof::Int`: Degrees of freedom (parameters estimated)
- `dof_fes::Int`: DOF absorbed by fixed effects
- `dof_residual::Int`: Residual degrees of freedom
- `tss_total::T`: Total sum of squares (before FE)
- `tss_partial::T`: TSS after partialing out FEs
- `rss::T`: Residual sum of squares
- `r2::T`: R-squared
- `r2_within::T`: Within R-squared (with FEs)
- `has_intercept::Bool`: Whether model has intercept
- `vcov_estimator::V`: Variance-covariance estimator (deep copy)
- `vcov_matrix::Symmetric{T, Matrix{T}}`: Precomputed variance-covariance matrix
- `se::Vector{T}`: Standard errors
- `t_stats::Vector{T}`: t-statistics
- `p_values::Vector{T}`: p-values
- `F::T`: F-statistic (computed with vcov_estimator)
- `p::T`: P-value of F-statistic
"""
struct OLSEstimator{T <: AbstractFloat, P <: OLSLinearPredictor{T}, V} <:
       AbstractRegressModel
    # Core GLM-style components
    rr::OLSResponse{T}              # Response object
    pp::P                           # Predictor object (Chol or QR)

    # Fixed effects component
    fes::OLSFixedEffects{T}         # Fixed effects (empty if no FEs)

    # Formula and metadata
    formula::FormulaTerm
    formula_schema::FormulaTerm
    contrasts::Dict{Symbol, Any}

    # Sample information
    esample::BitVector              # Which observations used

    # Coefficient metadata
    coefnames::Vector{String}
    basis_coef::BitVector           # Which coefficients are not collinear

    # Degrees of freedom
    nobs::Int                       # Number of observations
    dof::Int                        # Number of estimated parameters
    dof_fes::Int                    # DOF absorbed by fixed effects
    dof_residual::Int               # Residual degrees of freedom

    # Model fit statistics
    tss_total::T                    # Total sum of squares (before FE)
    tss_partial::T                  # TSS after partialing out FEs
    rss::T                          # Residual sum of squares
    r2::T                           # R-squared
    r2_within::T                    # Within R-squared (if FEs present)

    # Additional metadata
    has_intercept::Bool

    # Variance-covariance estimator and precomputed statistics
    # NOTE: These can be Nothing for lazy computation (deferred until accessed).
    # When lazy=true at fit time, vcov is computed on first access via StatsAPI.vcov(m).
    vcov_estimator::V                                           # Deep copy of the estimator
    vcov_matrix::Union{Nothing, Symmetric{T, Matrix{T}}}       # Precomputed vcov (or Nothing for lazy)
    se::Union{Nothing, Vector{T}}                              # Standard errors (or Nothing for lazy)
    t_stats::Union{Nothing, Vector{T}}                         # t-statistics (or Nothing for lazy)
    p_values::Union{Nothing, Vector{T}}                        # p-values (or Nothing for lazy)

    # Test statistics (computed with vcov_estimator)
    F::Union{Nothing, T}            # F-statistic (or Nothing for lazy)
    p::Union{Nothing, T}            # P-value of F-stat (or Nothing for lazy)
end

has_iv(::OLSEstimator) = false
has_fe(m::OLSEstimator) = has_fe(m.formula)
dof_fes(m::OLSEstimator) = m.dof_fes
r2_within(m::OLSEstimator) = m.r2_within
model_hasintercept(m::OLSEstimator) = m.has_intercept

##############################################################################
##
## Type OLSMatrixEstimator (for matrix-based OLS estimation)
##
##############################################################################

"""
    OLSMatrixEstimator <: AbstractRegressModel

Lightweight OLS model for matrix-based estimation without formula machinery.
Compatible with CovarianceMatrices.jl for post-estimation robust vcov.

Use `ols(X, y)` to fit this model type.

# Type Parameters
- `T`: Float type (Float64 or Float32)
- `P`: Predictor type (OLSPredictorChol or OLSPredictorQR)
- `V`: Variance-covariance estimator type (HC1, HC3, CR1, etc.)

# Fields
- `rr::OLSResponse{T}`: Response object (y, fitted values, weights)
- `pp::P`: Predictor object (X, coefficients, factorization)
- `basis_coef::BitVector`: Non-collinear coefficients indicator
- `nobs::Int`: Number of observations
- `dof::Int`: Degrees of freedom (parameters estimated)
- `dof_residual::Int`: Residual degrees of freedom
- `rss::T`: Residual sum of squares
- `tss::T`: Total sum of squares
- `r2::T`: R-squared
- `has_intercept::Bool`: Whether model has intercept (assumed from first column)
- `vcov_estimator::V`: Variance-covariance estimator (deep copy)
- `vcov_matrix::Symmetric{T, Matrix{T}}`: Precomputed variance-covariance matrix
- `se::Vector{T}`: Standard errors
- `t_stats::Vector{T}`: t-statistics
- `p_values::Vector{T}`: p-values

# Example
```julia
X = hcat(ones(100), randn(100, 2))
y = randn(100)
model = ols(X, y)

coef(model)
residuals(model)
r2(model)

# Robust standard errors
using CovarianceMatrices
vcov(HC1(), model)
stderror(model)  # Uses precomputed vcov
```
"""
struct OLSMatrixEstimator{T <: AbstractFloat, P <: OLSLinearPredictor{T}, V} <:
       AbstractRegressModel
    rr::OLSResponse{T}              # Response object
    pp::P                           # Predictor object (Chol or QR)
    basis_coef::BitVector           # Which coefficients are not collinear
    nobs::Int                       # Number of observations
    dof::Int                        # Number of estimated parameters
    dof_residual::Int               # Residual degrees of freedom
    rss::T                          # Residual sum of squares
    tss::T                          # Total sum of squares
    r2::T                           # R-squared
    has_intercept::Bool             # Whether model has intercept

    # Variance-covariance estimator and precomputed statistics
    vcov_estimator::V                        # Deep copy of the estimator
    vcov_matrix::Symmetric{T, Matrix{T}}    # Precomputed vcov matrix
    se::Vector{T}                            # Standard errors
    t_stats::Vector{T}                       # t-statistics
    p_values::Vector{T}                      # p-values
end

has_iv(::OLSMatrixEstimator) = false
has_fe(::OLSMatrixEstimator) = false
dof_fes(::OLSMatrixEstimator) = 0
model_hasintercept(m::OLSMatrixEstimator) = m.has_intercept

"""
    has_matrices(m::OLSMatrixEstimator) -> Bool

Check if model has stored matrices (X, y, mu).
"""
has_matrices(m::OLSMatrixEstimator) = has_predictor_data(m.pp)

##############################################################################
##
## StatsAPI Interface for OLSMatrixEstimator
##
##############################################################################

# Basic accessors
function StatsAPI.coef(m::OLSMatrixEstimator)
    beta = copy(m.pp.beta)
    beta[.!m.basis_coef] .= zero(eltype(beta))
    return beta
end

function StatsAPI.response(m::OLSMatrixEstimator)
    has_response_data(m.rr) ||
        error("Response vector not stored.")
    return m.rr.y
end

function StatsAPI.fitted(m::OLSMatrixEstimator)
    has_response_data(m.rr) ||
        error("Fitted values not stored.")
    return m.rr.mu
end

function StatsAPI.modelmatrix(m::OLSMatrixEstimator)
    has_predictor_data(m.pp) ||
        error("Model matrix not stored.")
    return m.pp.X
end

function StatsAPI.residuals(m::OLSMatrixEstimator)
    has_response_data(m.rr) ||
        error("Response vector not stored. Cannot compute residuals.")
    resid = m.rr.y .- m.rr.mu
    # Unweight residuals if model was estimated with weights
    if isweighted(m.rr)
        sqrtw = sqrt.(m.rr.wts)
        resid = resid ./ sqrtw
    end
    return resid
end

# Sample size and DOF
StatsAPI.nobs(m::OLSMatrixEstimator) = m.nobs
StatsAPI.dof(m::OLSMatrixEstimator) = m.dof
StatsAPI.dof_residual(m::OLSMatrixEstimator) = m.dof_residual

# Model fit
StatsAPI.deviance(m::OLSMatrixEstimator) = m.rss
StatsAPI.nulldeviance(m::OLSMatrixEstimator) = m.tss
StatsAPI.rss(m::OLSMatrixEstimator) = m.rss
StatsAPI.r2(m::OLSMatrixEstimator) = m.r2

# Variance-covariance (returns precomputed matrix)
StatsAPI.vcov(m::OLSMatrixEstimator) = m.vcov_matrix

# Standard errors (returns precomputed values)
StatsAPI.stderror(m::OLSMatrixEstimator) = m.se

# Adjusted R2 - kept specialized because AbstractRegressModel relies on formula()
function StatsAPI.adjr2(m::OLSMatrixEstimator)
    n = nobs(m)
    k = dof(m)
    dev = deviance(m)
    dev0 = nulldeviance(m)
    return 1 - (dev * (n - m.has_intercept)) / (dev0 * max(n - k, 1))
end

##############################################################################
##
## + Operator for OLSMatrixEstimator + VcovSpec
##
##############################################################################

"""
    Base.:+(m::OLSMatrixEstimator, v::VcovSpec)

Create a new model with a different variance-covariance estimator.
"""
function Base.:+(m::OLSMatrixEstimator{T, P, V1}, v::VcovSpec{V2}) where {T, P, V1, V2}
    vcov_mat = StatsBase.vcov(v.source, m)
    se, t_stats, p_values, _, _ = _calculate_vcov_stats(m, vcov_mat)
    vcov_copy = deepcopy_vcov(v.source)

    return OLSMatrixEstimator{T, P, V2}(
        m.rr, m.pp, m.basis_coef,
        m.nobs, m.dof, m.dof_residual,
        m.rss, m.tss, m.r2, m.has_intercept,
        vcov_copy, Symmetric(vcov_mat), se, t_stats, p_values
    )
end

"""
    has_matrices(m::OLSEstimator) -> Bool

Check if model has stored matrices (X, y, mu). Returns false if fit with save=:minimal.
"""
has_matrices(m::OLSEstimator) = has_predictor_data(m.pp)

# Property accessor for backward compatibility and lazy vcov support
function Base.getproperty(m::OLSEstimator, s::Symbol)
    if s === :iterations
        return getfield(m, :fes).iterations
    elseif s === :converged
        return getfield(m, :fes).converged
    elseif s === :F
        # Lazy F-stat: compute on demand if nothing
        F_val = getfield(m, :F)
        F_val !== nothing && return F_val
        return _get_fstat(m)
    elseif s === :p
        # Lazy p-value: compute on demand if nothing
        p_val = getfield(m, :p)
        p_val !== nothing && return p_val
        return _get_pval(m)
    else
        return getfield(m, s)
    end
end

##############################################################################
##
## StatsAPI Interface - GLM-compatible
##
##############################################################################

# Basic accessors
function StatsAPI.coef(m::OLSEstimator)
    # Return 0.0 for collinear coefficients (backward compatibility)
    beta = copy(m.pp.beta)
    beta[.!m.basis_coef] .= zero(eltype(beta))
    return beta
end
StatsAPI.coefnames(m::OLSEstimator) = m.coefnames
StatsAPI.responsename(m::OLSEstimator) = m.rr.response_name

function StatsAPI.response(m::OLSEstimator)
    has_response_data(m.rr) ||
        error("Response vector not stored. Model was fit with save=:minimal.")
    return m.rr.y
end

function StatsAPI.fitted(m::OLSEstimator)
    has_response_data(m.rr) ||
        error("Fitted values not stored. Model was fit with save=:minimal.")
    return m.rr.mu
end

function StatsAPI.modelmatrix(m::OLSEstimator)
    has_predictor_data(m.pp) ||
        error("Model matrix not stored. Model was fit with save=:minimal.")
    return m.pp.X
end

# Residuals (compute from y - mu, unweighted)
function StatsAPI.residuals(m::OLSEstimator)
    has_response_data(m.rr) ||
        error("Response vector not stored. Model was fit with save=:minimal. Cannot compute residuals.")
    resid = m.rr.y .- m.rr.mu
    # Unweight residuals if model was estimated with weights
    if isweighted(m.rr)
        sqrtw = sqrt.(m.rr.wts)
        resid = resid ./ sqrtw
    end
    return resid
end

# Sample size and DOF
StatsAPI.nobs(m::OLSEstimator) = m.nobs
StatsAPI.dof(m::OLSEstimator) = m.dof
StatsAPI.dof_residual(m::OLSEstimator) = m.dof_residual

# Model fit
StatsAPI.deviance(m::OLSEstimator) = m.rss
StatsAPI.nulldeviance(m::OLSEstimator) = m.tss_total
StatsAPI.rss(m::OLSEstimator) = m.rss
StatsAPI.r2(m::OLSEstimator) = m.r2

# Variance-covariance (returns precomputed matrix or computes on-demand)
function StatsAPI.vcov(m::OLSEstimator)
    m.vcov_matrix !== nothing && return m.vcov_matrix
    # Lazy computation: compute vcov using the stored estimator
    return _compute_vcov_lazy(m)
end

# Standard errors (returns precomputed values or computes on-demand)
function StatsAPI.stderror(m::OLSEstimator)
    m.se !== nothing && return m.se
    # Lazy computation: compute from vcov
    return sqrt.(diag(vcov(m)))
end

# t-statistics (returns precomputed values or computes on-demand)
function t_stats(m::OLSEstimator)
    m.t_stats !== nothing && return m.t_stats
    # Lazy computation: compute from coef and se
    return coef(m) ./ stderror(m)
end

# p-values (returns precomputed values or computes on-demand)
function p_values(m::OLSEstimator)
    m.p_values !== nothing && return m.p_values
    # Lazy computation: compute from t-stats
    return 2 .* tdistccdf.(dof_residual(m), abs.(t_stats(m)))
end

"""
    _compute_vcov_lazy(m::OLSEstimator)

Compute vcov matrix on-demand for models created with lazy vcov.
Uses the stored vcov_estimator (typically HC1).
"""
function _compute_vcov_lazy(m::OLSEstimator{T}) where {T}
    # Compute vcov using the stored estimator via CovarianceMatrices.vcov
    return Symmetric(CovarianceMatrices.vcov(m.vcov_estimator, m))
end

# Override for lazy F-stat computation
# Use getfield to avoid infinite recursion through getproperty
function _get_fstat(m::OLSEstimator)
    F_val = getfield(m, :F)
    F_val !== nothing && return F_val
    # Lazy computation: compute F-stat from vcov
    has_int = getfield(m, :has_intercept)
    F, _ = compute_robust_fstat(coef(m), vcov(m), has_int, dof_residual(m))
    return F
end

function _get_pval(m::OLSEstimator)
    p_val = getfield(m, :p)
    p_val !== nothing && return p_val
    # Lazy computation: compute p-val from vcov
    has_int = getfield(m, :has_intercept)
    _, p = compute_robust_fstat(coef(m), vcov(m), has_int, dof_residual(m))
    return p
end

# Formula
StatsModels.formula(m::OLSEstimator) = m.formula_schema

##############################################################################
##
## Predict and Residuals (with new data)
##
##############################################################################

function StatsAPI.predict(m::OLSEstimator{T}, data) where {T}
    Tables.istable(data) ||
        throw(ArgumentError("expected second argument to be a Table, got $(typeof(data))"))

    has_cont_fe_interaction(m.formula) &&
        throw(ArgumentError("Interaction of fixed effect and continuous variable detected in formula; this is currently not supported in `predict`"))

    cdata = Tables.columntable(data)
    cols, nonmissings = StatsModels.missing_omit(cdata, m.formula_schema.rhs)
    Xnew = modelmatrix(m.formula_schema, cols)

    # Use only non-collinear coefficients
    coef_valid = coef(m)[m.basis_coef]

    # Type-stable inner function via function barrier
    return _predict_ols_impl(Xnew, coef_valid, nonmissings, m.basis_coef, m, data, T)
end

# Type-stable inner function for OLS predict
function _predict_ols_impl(
        Xnew::AbstractMatrix, coef_valid::AbstractVector{T},
        nonmissings::AbstractVector{Bool}, basis_coef::BitVector,
        m::OLSEstimator{T}, data, ::Type{T}
) where {T}
    n = length(nonmissings)
    # Always allocate with Union type for consistent return type
    out = Vector{Union{T, Missing}}(missing, n)
    @views out[nonmissings] .= Xnew[:, basis_coef] * coef_valid

    if has_fe(m)
        nrow(fe(m)) > 0 ||
            throw(ArgumentError("Model has no estimated fixed effects. To store estimates of fixed effects, run `ols` with the option save = :fe"))

        df = DataFrame(data; copycols = false)
        fes = leftjoin(select(df, m.fes.fe_names), dropmissing(unique(m.fes.fe));
            on = m.fes.fe_names, makeunique = true, matchmissing = :equal, order = :left)
        fes = combine(fes, AsTable(Not(m.fes.fe_names)) => sum)

        @views out[nonmissings] .+= fes[nonmissings, 1]
    end

    return out
end

function StatsAPI.residuals(m::OLSEstimator{T}, data) where {T}
    Tables.istable(data) ||
        throw(ArgumentError("expected second argument to be a Table, got $(typeof(data))"))
    has_fe(m) &&
        throw("To access residuals for a model with high-dimensional fixed effects, access them directly with `residuals(m)`.")

    cdata = Tables.columntable(data)
    cols, nonmissings = StatsModels.missing_omit(cdata, m.formula_schema.rhs)
    Xnew = modelmatrix(m.formula_schema, cols)
    y = response(m.formula_schema, cdata)

    # Use only non-collinear coefficients
    coef_valid = coef(m)[m.basis_coef]

    # Type-stable inner function via function barrier
    return _residuals_ols_impl(y, Xnew, coef_valid, nonmissings, m.basis_coef, T)
end

# Type-stable inner function for OLS residuals
function _residuals_ols_impl(
        y::AbstractVector, Xnew::AbstractMatrix, coef_valid::AbstractVector{T},
        nonmissings::AbstractVector{Bool}, basis_coef::BitVector, ::Type{T}
) where {T}
    n = length(nonmissings)
    # Always allocate with Union type for consistent return type
    out = Vector{Union{T, Missing}}(missing, n)
    @views out[nonmissings] .= y .- Xnew[:, basis_coef] * coef_valid
    return out
end

"""
   fe(m::OLSEstimator; keepkeys = false)

Return a DataFrame with fixed effects estimates.
The output is aligned with the original DataFrame used in `ols`.

### Keyword arguments
* `keepkeys::Bool` : Should the returned DataFrame include the original variables used to define groups? Default to false
"""
function fe(m::OLSEstimator; keepkeys = false)
    !has_fe(m) && throw("fe() is not defined for models without fixed effects")
    if keepkeys
        m.fes.fe
    else
        m.fes.fe[!, (length(m.fes.fe_names) + 1):end]
    end
end

function StatsAPI.coeftable(m::OLSEstimator; level = 0.95)
    cc = coef(m)
    se = stderror(m)      # Use accessor for lazy support
    tt = t_stats(m)       # Use accessor for lazy support
    pv = p_values(m)      # Use accessor for lazy support
    coefnms = coefnames(m)

    # Compute confidence intervals using precomputed se
    scale = tdistinvcdf(dof_residual(m), 1 - (1 - level) / 2)
    conf_int = hcat(cc .- scale .* se, cc .+ scale .* se)

    # put (intercept) last
    if !isempty(coefnms) &&
       ((coefnms[1] == Symbol("(Intercept)")) || (coefnms[1] == "(Intercept)"))
        newindex = vcat(2:length(cc), 1)
        cc = cc[newindex]
        se = se[newindex]
        tt = tt[newindex]
        pv = pv[newindex]
        conf_int = conf_int[newindex, :]
        coefnms = coefnms[newindex]
    end

    CoefTable(
        hcat(cc, se, tt, pv, conf_int[:, 1:2]),
        ["Estimate", "Std. Error", "t-stat", "Pr(>|t|)", "Lower 95%", "Upper 95%"],
        ["$(coefnms[i])" for i in 1:length(cc)], 4)
end

##############################################################################
##
## Display Result
##
##############################################################################

function top(m::OLSEstimator)
    # Use shared summary plus OLS specific fields
    out_common = _summary_table_common(m) # Matrix

    # We need to insert "Converged" at index 2
    # "Converged" row
    row_converged = ["Converged" m.fes.converged]

    # Reconstruct matrix
    part1 = out_common[1:1, :]
    part2 = out_common[2:end, :]
    out = vcat(part1, row_converged, part2)

    # Add Iterations if FE
    if has_fe(m)
        row_iter = ["Iterations" sprint(show, m.fes.iterations, context = :compact => true)]
        out = vcat(out, row_iter)
    end
    return out
end

import StatsBase: NoQuote, PValue
function Base.show(io::IO, m::OLSEstimator)
    ct = coeftable(m)
    cols = ct.cols;
    rownms = ct.rownms;
    colnms = ct.colnms;
    nc = length(cols)
    nr = length(cols[1])
    if length(rownms) == 0
        rownms = [lpad("[$i]", floor(Integer, log10(nr))+3) for i in 1:nr]
    end
    mat = [j == 1 ? NoQuote(rownms[i]) :
           j-1 == ct.pvalcol ? NoQuote(sprint(show, PValue(cols[j - 1][i]))) :
           j-1 in ct.teststatcol ? TestStat(cols[j - 1][i]) :
           cols[j - 1][i] isa AbstractString ? NoQuote(cols[j - 1][i]) : cols[j - 1][i]
           for i in 1:nr, j in 1:(nc + 1)]
    io = IOContext(io, :compact=>true, :limit=>false)
    A = Base.alignment(io, mat, 1:size(mat, 1), 1:size(mat, 2),
        typemax(Int), typemax(Int), 3)
    nmswidths = pushfirst!(length.(colnms), 0)
    A = [nmswidths[i] > sum(A[i]) ? (A[i][1]+nmswidths[i]-sum(A[i]), A[i][2]) : A[i]
         for i in 1:length(A)]
    totwidth = compute_table_width(A, colnms)

    # Title: just "OLS"
    ctitle = "OLS"
    halfwidth = max(0, div(totwidth - length(ctitle), 2))
    print(io, " " ^ halfwidth * ctitle * " " ^ halfwidth)
    ctop = top(m)
    for i in 1:size(ctop, 1)
        ctop[i, 1] = ctop[i, 1] * ":"
    end
    println(io)
    println_horizontal_line(io, totwidth)
    halfwidth = div(totwidth, 2) - 1
    interwidth = 2 + mod(totwidth, 2)
    for i in 1:(div(size(ctop, 1) - 1, 2) + 1)
        print(io, ctop[2 * i - 1, 1])
        print(io, lpad(ctop[2 * i - 1, 2], halfwidth - length(ctop[2 * i - 1, 1])))
        print(io, " " ^ interwidth)
        if size(ctop, 1) >= 2*i
            print(io, ctop[2 * i, 1])
            print(io, lpad(ctop[2 * i, 2], halfwidth - length(ctop[2 * i, 1])))
        end
        println(io)
    end

    println_horizontal_line(io, totwidth)
    print(io, repeat(' ', sum(A[1])))
    for j in 1:length(colnms)
        print(io, "  ", lpad(colnms[j], sum(A[j + 1])))
    end
    println(io)
    println_horizontal_line(io, totwidth)
    for i in 1:size(mat, 1)
        Base.print_matrix_row(io, mat, A, i, 1:size(mat, 2), "  ")
        i != size(mat, 1) && println(io)
    end
    println(io)
    println_horizontal_line(io, totwidth)

    # Note: variance-covariance type
    vcov_name = vcov_type_name(m.vcov_estimator)
    println(io, "Note: Std. errors computed using $vcov_name variance estimator")
    nothing
end

function Base.show(io::IO, ::MIME"text/html", m::OLSEstimator)
    ct = coeftable(m)
    cols = ct.cols;
    rownms = ct.rownms;
    colnms = ct.colnms;

    # Start table with "OLS" as caption
    html_table_start(io; class = "regress-table regress-ols", caption = "OLS")

    # Summary statistics section
    ctop = top(m)
    html_thead_start(io; class = "regress-summary")
    for i in 1:size(ctop, 1)
        html_row(io, [ctop[i, 1], ctop[i, 2]]; class = "regress-summary-row")
    end
    html_thead_end(io)

    # Coefficient table header
    html_thead_start(io; class = "regress-coef-header")
    html_row(io, vcat([""], colnms); is_header = true)
    html_thead_end(io)

    # Coefficient table body
    html_tbody_start(io; class = "regress-coef-body")
    for i in 1:length(rownms)
        row_data = [rownms[i]]
        for j in 1:length(cols)
            if j == ct.pvalcol
                push!(row_data, format_pvalue(cols[j][i]))
            else
                push!(row_data, format_number(cols[j][i]))
            end
        end
        html_row(io, row_data)
    end
    html_tbody_end(io)

    # Footer with vcov type note
    vcov_name = vcov_type_name(m.vcov_estimator)
    html_tfoot_start(io; class = "regress-footer")
    html_row(io, ["Note: Std. errors computed using $vcov_name variance estimator",
        "", "", "", "", "", ""])
    html_tfoot_end(io)

    html_table_end(io)
end

##############################################################################
##
## Schema
##
##############################################################################
function StatsModels.apply_schema(t::FormulaTerm, schema::StatsModels.Schema,
        Mod::Type{<:OLSEstimator}, has_fe_intercept)
    schema = StatsModels.FullRank(schema)
    if has_fe_intercept
        push!(schema.already, InterceptTerm{true}())
    end
    FormulaTerm(apply_schema(t.lhs, schema.schema, StatisticalModel),
        StatsModels.collect_matrix_terms(apply_schema(t.rhs, schema, StatisticalModel)))
end

##############################################################################
##
## + Operator for OLSEstimator + VcovSpec
##
##############################################################################

"""
    Base.:+(m::OLSEstimator, v::VcovSpec)

Create a new model with a different variance-covariance estimator.

Returns a new `OLSEstimator` with the same underlying data but
precomputed vcov statistics using the specified variance estimator.
The vcov estimator is deep-copied to avoid aliasing issues.

# Examples
```julia
model = ols(df, @formula(y ~ x))

# Heteroskedasticity-robust
model_hc3 = model + vcov(HC3())
stderror(model_hc3)  # Uses precomputed HC3 standard errors
model_hc3.F          # Robust Wald F-statistic

# Cluster-robust
model_cr = ols(df, @formula(y ~ x), save_cluster = :firm)
model_cr1 = model_cr + vcov(CR1(:firm))
```

See also: [`VcovSpec`](@ref)
"""
function Base.:+(m::OLSEstimator{T, P, V1}, v::VcovSpec{V2}) where {T, P, V1, V2}
    # Compute vcov matrix using StatsBase.vcov (which dispatches to covariance.jl methods)
    vcov_mat = StatsBase.vcov(v.source, m)

    # Use shared helper for stats
    se, t_stats, p_values, F_stat, p_val = _calculate_vcov_stats(m, vcov_mat)

    # Deep copy the vcov estimator to avoid aliasing
    vcov_copy = deepcopy_vcov(v.source)

    # Return new OLSEstimator with same data but different vcov type
    return OLSEstimator{T, P, V2}(
        m.rr, m.pp, m.fes,
        m.formula, m.formula_schema, m.contrasts,
        m.esample,
        m.coefnames, m.basis_coef,
        m.nobs, m.dof, m.dof_fes, m.dof_residual,
        m.tss_total, m.tss_partial, m.rss, m.r2, m.r2_within,
        m.has_intercept,
        vcov_copy, Symmetric(vcov_mat), se, t_stats, p_values,
        F_stat, p_val
    )
end
