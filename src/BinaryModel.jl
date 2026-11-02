import StatsAPI: coef, coefnames, coeftable, confint, deviance,
                 dof, dof_residual, fitted, loglikelihood,
                 modelmatrix, nobs, nulldeviance, nullloglikelihood,
                 predict, residuals, response, responsename
using Printf: @sprintf

# ── Response ─────────────────────────────────────────────────────────────────
"""Abstract supertype for binary-response link functions supported by `glmfe`."""
abstract type Link end

"""Standard normal CDF inverse link for Bernoulli probit regression."""
struct ProbitLink <: Link end
"""Logistic inverse link for Bernoulli logistic regression."""
struct LogitLink <: Link end
"""
    BinaryResponse{T <: AbstractFloat, L <: Link}

Working storage for a binary response and its current predictor evaluation.

# Fields
- `link`: concrete link instance selecting inverse-link dispatch.
- `y`: observed binary response vector.
- `deviance`: accepted total deviance used as the step-halving reference.
- `deviance_new`: candidate total deviance.
- `eta`: linear predictor, including the fixed-effect contribution after refresh.
- `mu`: fitted probabilities.
- `one_minus_mu`: complementary probabilities computed directly by the inverse link.
- `dmu_deta`: inverse-link derivatives.
- `resid`: response residuals `y - mu`.
- `deviance_contributions`: deviance contribution of each observation.
- `wts`, `offset`: reserved buffers, currently unused and uninitialized.
- `response_name`: response variable name as a symbol.

During a candidate evaluation, vector fields describe the candidate while
`deviance` retains the accepted reference value.
"""
mutable struct BinaryResponse{T <: AbstractFloat, L <: Link}
    link::L
    y::Vector{T}
    deviance::T
    deviance_new::T
    eta::Vector{T}
    mu::Vector{T}
    one_minus_mu::Vector{T}
    dmu_deta::Vector{T}
    resid::Vector{T}
    deviance_contributions::Vector{T}
    wts::Vector{T}
    offset::Vector{T}
    response_name::Symbol
end

function StatsAPI.deviance(rr::BinaryResponse)
    return sum(rr.deviance_contributions)
end

# ── Predictor ─────────────────────────────────────────────────────────────────

"""
    BinaryPredictorQR{T <: AbstractFloat}

Predictor for binary models: the design matrix and the current and candidate
coefficient vectors updated at each IRLS step.

# Fields
- `X`: design matrix (fixed-effect terms excluded)
- `beta`: current coefficient estimates
- `beta_new`: candidate coefficients from the latest weighted least-squares step
"""
mutable struct BinaryPredictorQR{T <: AbstractFloat}
    X::Matrix{T}
    beta::Vector{T}
    beta_new::Vector{T}
end

"""
    BinaryResponse(yi, pp::BinaryPredictorQR, responsename, link::Link=ProbitLink()) -> BinaryResponse

Convert the binary response to the coefficient element type and initialize
`eta = pp.X * pp.beta` without a fixed-effect contribution. Use the selected
inverse link to fill probabilities, complements, and derivatives, then compute
residuals and Bernoulli deviance contributions.

Set `deviance` to their sum and initialize `deviance_new` to zero. Allocate
`wts` and `offset` without initializing or using them.
"""
function BinaryResponse(yi, pp::BinaryPredictorQR, responsename, link::Link = ProbitLink())
    T = eltype(pp.beta)
    yi = T.(yi)
    eta = pp.X * pp.beta
    mu = similar(yi)
    rr = BinaryResponse(
        link,
        yi,
        zero(T),
        zero(T),
        eta,
        mu, # fitted probabilities
        similar(yi), # one_minus_mu
        similar(yi), # dmu_deta
        similar(yi), # resid
        similar(yi), # deviance_contributions
        similar(yi), # weights
        similar(yi), # offset
        responsename
    )
    rr.deviance = refresh_response!(rr)
    return rr
end

# ── Fitted model ──────────────────────────────────────────────────────────────

"""
    BinaryEstimator{T <: AbstractFloat, R <: BinaryResponse{T}}

Bernoulli model storage used by `glmfe` for probit and logit fitting. Combine a
`BinaryResponse`, a `BinaryPredictorQR`, formula metadata, and parameter counts.

Fitting returns a model only when the deviance convergence criterion is met.
The concrete response type preserves its link parameter for specialization.

`esample` is a Boolean mask over the original input rows. `drop_reasons` has
one symbol per input row: `:none` for retained rows, `:fe_all_zero` or
`:fe_all_one` for rows removed from constant-outcome fixed-effect groups.
Response and predictor buffers, `fitted`, `residuals`, and `nobs` describe
only the retained estimation sample.
"""
mutable struct BinaryEstimator{T <: AbstractFloat, R <: BinaryResponse{T}} <:
               AbstractRegressModel
    rr::R
    pp::BinaryPredictorQR{T}

    formula::FormulaTerm
    formula_schema::FormulaTerm
    formula_fes::FormulaTerm

    n_observations::Int
    n_parameters::Int

    deviance::Float64
    nulldeviance::Float64

    has_fixed_effects::Bool
    fixed_effects_dof::Int

    coefnames::Vector{String}
    basis_coef::BitVector
    esample::BitVector
    drop_reasons::Vector{Symbol}
end

function BinaryEstimator{T, R}(rr, pp, formula, formula_schema, formula_fes,
        n_observations, n_parameters, deviance, nulldeviance, has_fixed_effects,
        fixed_effects_dof, coefnames, basis_coef) where {T, R <: BinaryResponse{T}}
    return BinaryEstimator{T, R}(rr, pp, formula, formula_schema, formula_fes,
        n_observations, n_parameters, deviance, nulldeviance, has_fixed_effects,
        fixed_effects_dof, coefnames, basis_coef, trues(length(rr.y)),
        fill(:none, length(rr.y)))
end

function BinaryEstimator(rr::BinaryResponse{T}, args::Vararg{Any, 12}) where {T}
    return BinaryEstimator{T}(rr, args...)
end

# Preserve construction with only the numeric type specified.
function BinaryEstimator{T}(rr::R, args...) where {T, R <: BinaryResponse{T}}
    return BinaryEstimator{T, R}(rr, args...)
end

has_iv(::BinaryEstimator) = false
has_fe(m::BinaryEstimator) = has_fe(m.formula_fes)

basis_coef(m::BinaryEstimator) = m.basis_coef

function esample(m::BinaryEstimator, v::AbstractVector)
    Base.require_one_based_indexing(v)
    length(v) == length(m.esample) ||
        throw(ArgumentError("Vector length ($(length(v))) must match original data length ($(length(m.esample)))"))
    return v[m.esample]
end

StatsAPI.islinear(::BinaryEstimator) = false
StatsAPI.coefnames(m::BinaryEstimator) = m.coefnames
StatsAPI.responsename(m::BinaryEstimator) = m.rr.response_name
StatsAPI.nulldeviance(m::BinaryEstimator) = m.nulldeviance
StatsAPI.nobs(m::BinaryEstimator) = m.n_observations
StatsAPI.dof(m::BinaryEstimator) = m.n_parameters
dof_fes(m::BinaryEstimator) = m.fixed_effects_dof
StatsAPI.dof_residual(m::BinaryEstimator) = max(1, nobs(m) - dof(m) - dof_fes(m))

function StatsAPI.coef(m::BinaryEstimator)
    m.pp.beta
end

function StatsAPI.fitted(m::BinaryEstimator)
    m.rr.mu
end

function StatsAPI.response(m::BinaryEstimator)
    m.rr.y
end

function StatsAPI.residuals(m::BinaryEstimator)
    copy(m.rr.resid)
end

function StatsAPI.modelmatrix(m::BinaryEstimator)
    m.pp.X
end

function StatsAPI.coeftable(m::BinaryEstimator)
    CoefTable(
        [coef(m)],
        ["Estimate"],
        coefnames(m)
    )
end

function _format_binary_summary_value(x::Real)
    return isfinite(x) ? @sprintf("%.4f", x) : "."
end

function _binary_summary_table(m::BinaryEstimator)
    out = ["Number of obs" sprint(show, nobs(m), context = :compact => true);
           "dof (model)" sprint(show, dof(m), context = :compact => true);
           "dof (residuals)" sprint(show, dof_residual(m), context = :compact => true);
           "Log likelihood" _format_binary_summary_value(loglikelihood(m));
           "Null log likelihood" _format_binary_summary_value(nullloglikelihood(m));
           "Deviance" _format_binary_summary_value(deviance(m));
           "Null deviance" _format_binary_summary_value(nulldeviance(m));
           "Pseudo R² (McFadden)" _format_binary_summary_value(r2(m));]

    removed = count(!, m.esample)
    if removed > 0
        out = vcat(out, ["Observations removed" string(removed)])
    end
    if has_fe(m)
        out = vcat(out,
            ["Fixed effects" "yes";
             "dof (fixed effects)" sprint(show, dof_fes(m), context = :compact => true);])
    end
    return out
end

function Base.show(io::IO, m::BinaryEstimator)
    show(io, MIME"text/plain"(), m)
end

function Base.show(io::IO, ::MIME"text/plain", m::BinaryEstimator)
    ctop = _binary_summary_table(m)
    label_width = maximum(length.(ctop[:, 1])) + 2

    println(io, "Binary Model")
    println(io, "─" ^ max(40, label_width + 18))
    for i in 1:size(ctop, 1)
        print(io, rpad(ctop[i, 1] * ":", label_width))
        println(io, ctop[i, 2])
    end
    println(io)
    show(io, MIME"text/plain"(), coeftable(m))
end

function StatsAPI.deviance(m::BinaryEstimator)
    return deviance(m.rr)
end

StatsAPI.loglikelihood(m::BinaryEstimator) = -deviance(m) / 2
StatsAPI.nullloglikelihood(m::BinaryEstimator) = -nulldeviance(m) / 2

function StatsAPI.r2(m::BinaryEstimator)
    return 1 - (deviance(m)/nulldeviance(m))
end
