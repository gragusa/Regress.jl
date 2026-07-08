import StatsAPI: coef, coefnames, coeftable, confint, deviance,
                 dof, dof_residual, fitted, loglikelihood,
                 modelmatrix, nobs, nulldeviance, nullloglikelihood,
                 predict, residuals, response, responsename
using Printf: @sprintf

# ── Response ─────────────────────────────────────────────────────────────────

"""
    BinaryResponse{T <: AbstractFloat}

Working storage for binary model response. Holds observed data, fitted values,
and per-observation scores/hessians/log-likelihoods updated at each IRLS iteration.

# Fields
- `y`: observed binary response vector
- `distribution`: assumed link distribution (e.g. Normal for probit)
- `v`: per-observation `(score, hessian, log-likelihood)` tuples
- `deviance`, `deviance_new`: current and candidate deviance (convergence check)
- `eta`: linear predictor 
- `mu`: fitted probabilities (inverse-link of η)
- `wts`: observation weights
- `offset`: optional offset vector (empty = no offset)
- `response_name`: symbol name of the response variable
"""
mutable struct BinaryResponse{T <: AbstractFloat}
    y::Vector{T}
    distribution::Distribution
    v::Vector{Tuple{T, T, T}}
    deviance::T
    deviance_new::T
    eta::Vector{T}
    mu::Vector{T}
    wts::Vector{T}
    offset::Vector{T}
    response_name::Symbol
end

function StatsAPI.deviance(rr::BinaryResponse)
    total_log_likelihood = sum(getindex.(rr.v, 3))
    return -2 * total_log_likelihood
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
    BinaryResponse(yi, pp::BinaryPredictorQR, responsename) -> BinaryResponse

Construct a `BinaryResponse` from initial response vector `yi`, predictor `pp`,
and the response variable name. Computes the initial linear predictor `eta`,
log-likelihood contributions, and deviance.
"""
function BinaryResponse(yi, pp::BinaryPredictorQR, responsename)
    T = eltype(pp.beta)
    yi = T.(yi)
    eta = pp.X * pp.beta
    v = log_likelihood_probit.(yi, eta)
    total_log_likelihood = sum(getindex.(v, 3))
    deviance = -2 * total_log_likelihood
    return BinaryResponse(
        yi,
        Normal(0, 1),
        v,
        deviance,
        0.0,
        eta,
        similar(yi), # fitted probabilities
        similar(yi), # weights
        similar(yi), # offset
        responsename
    )
end

# ── Fitted model ──────────────────────────────────────────────────────────────

"""
    BinaryEstimator{T <: AbstractFloat}

Fitted binary regression model (e.g. probit, logit). Combines a `BinaryResponse`
and `BinaryPredictorQR` with formula metadata and summary statistics produced
after IRLS convergence.
"""
mutable struct BinaryEstimator{T <: AbstractFloat} <: AbstractRegressModel
    rr::BinaryResponse{T}
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
end

has_iv(::BinaryEstimator) = false
has_fe(m::BinaryEstimator) = has_fe(m.formula_fes)

basis_coef(m::BinaryEstimator) = m.basis_coef

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
    m.rr.y - m.rr.mu
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
    total_log_likelihood = sum(getindex.(m.rr.v, 3))
    return -2 * total_log_likelihood
end

StatsAPI.loglikelihood(m::BinaryEstimator) = -deviance(m) / 2
StatsAPI.nullloglikelihood(m::BinaryEstimator) = -nulldeviance(m) / 2

function StatsAPI.r2(m::BinaryEstimator)
    return 1 - (deviance(m)/nulldeviance(m))
end
