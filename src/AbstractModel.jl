##############################################################################
##
## AbstractRegressModel
##
## Shared behavior for OLSEstimator, OLSMatrixEstimator, and IVEstimator.
##
##############################################################################

abstract type AbstractRegressModel <: StatsAPI.RegressionModel end

##############################################################################
##
## Required Interface Methods (subtypes must implement)
##
## - nobs(m)
## - dof(m)
## - dof_residual(m)
## - deviance(m) (usually rss)
## - nulldeviance(m) (usually tss)
## - r2(m)
## - has_fe(m)
## - dof_fes(m)
##
##############################################################################

##############################################################################
##
## Accessor Methods with Defaults
##
## These provide default implementations that subtypes can override.
##
##############################################################################

"""
    r2_within(m::AbstractRegressModel)

Return the within R² for models with fixed effects, or NaN for models without.
Subtypes with fixed effects should override this method.
"""
r2_within(::AbstractRegressModel) = NaN

"""
    model_hasintercept(m::AbstractRegressModel)

Return whether the model has an intercept term.
Default implementation checks the formula if available.
"""
model_hasintercept(m::AbstractRegressModel) = hasintercept(formula(m))

# Override for OLSMatrixEstimator which stores has_intercept directly
# (defined in LinearModel.jl after the type is created)

"""
    t_stats(m::AbstractRegressModel)

Return the t-statistics vector for coefficients.
"""
t_stats(m::AbstractRegressModel) = m.t_stats

"""
    p_values(m::AbstractRegressModel)

Return the p-values vector for coefficients.
"""
p_values(m::AbstractRegressModel) = m.p_values

##############################################################################
##
## Shared StatsAPI Interface
##
##############################################################################

StatsAPI.islinear(::AbstractRegressModel) = true

function StatsAPI.mss(m::AbstractRegressModel)
    return nulldeviance(m) - deviance(m)
end

function StatsAPI.loglikelihood(m::AbstractRegressModel)
    n = nobs(m)
    σ² = deviance(m) / n
    return -n/2 * (log(2π) + log(σ²) + 1)
end

function StatsAPI.nullloglikelihood(m::AbstractRegressModel)
    n = nobs(m)
    σ² = nulldeviance(m) / n
    return -n/2 * (log(2π) + log(σ²) + 1)
end

function nullloglikelihood_within(m::AbstractRegressModel)
    n = nobs(m)
    r2w = r2_within(m)
    if isnan(r2w)
        return nullloglikelihood(m)
    else
        tss_within = deviance(m) / (1 - r2w)
        return -n/2 * (log(2π * tss_within / n) + 1)
    end
end

# The parameter count is `nobs - dof_residual`: estimated coefficients plus
# absorbed fixed-effect levels. `dof(model)` counts only the coefficients.
function StatsAPI.adjr2(model::AbstractRegressModel, variant::Symbol = :devianceratio)
    n = nobs(model)
    # The null model fits one parameter when the null deviance is centered
    centered = model_hasintercept(model) | has_fe(model)

    if variant == :McFadden
        k = n - dof_residual(model) - centered
        ll = loglikelihood(model)
        ll0 = nullloglikelihood(model)
        1 - (ll - k)/ll0
    elseif variant == :devianceratio
        dev = deviance(model)
        dev0 = nulldeviance(model)
        1 - (dev * (n - centered)) / (dev0 * dof_residual(model))
    else
        throw(ArgumentError("variant must be one of :McFadden or :devianceratio"))
    end
end

##############################################################################
##
## Degrees of freedom of t and F tests
##
##############################################################################

"""
    inference_dof(m) -> Int

Degrees of freedom of the t and F distributions behind the p-values, the confidence
intervals and the F test of `m`. With cluster-robust standard errors (`CR0`–`CR3`,
`Cluster`) it is `G - 1`, where `G` is the number of clusters, or the smallest number of
clusters over several cluster dimensions; this is R fixest's default,
`ssc(t.df = "min")`. Otherwise it is `dof_residual(m)`.
"""
inference_dof(m::AbstractRegressModel) = _test_dof(m, vcov(m))

# Degrees of freedom of the tests of `m` computed with the covariance matrix `V`. The
# linear and IV models apply `_cluster_test_dof` (utils/covariance.jl); other models,
# binary-response models among them, keep `dof_residual(m)`.
_test_dof(m, V) = dof_residual(m)

# `G - 1` for a cluster-robust covariance matrix or estimator `V`, and `dof` otherwise.
_cluster_test_dof(V, dof) = dof
function _cluster_test_dof(V::CovarianceMatrix, dof)
    return _cluster_test_dof(CovarianceMatrices.estimator(V), dof)
end
function _cluster_test_dof(k::CovarianceMatrices.CR, dof)
    minimum(CovarianceMatrices.nclusters(k)) - 1
end

function StatsAPI.confint(m::AbstractRegressModel; level::Real = 0.95)
    scale = tdistinvcdf(inference_dof(m), 1 - (1 - level) / 2)
    se_val = stderror(m)
    coef_val = coef(m)
    hcat(coef_val .- scale .* se_val, coef_val .+ scale .* se_val)
end

function StatsAPI.confint(ve::CovarianceMatrices.AbstractAsymptoticVarianceEstimator,
        m::AbstractRegressModel; level::Real = 0.95)
    V = vcov(ve, m)
    scale = tdistinvcdf(_test_dof(m, V), 1 - (1 - level) / 2)
    se = sqrt.(diag(V))
    coef_val = coef(m)
    hcat(coef_val .- scale .* se, coef_val .+ scale .* se)
end

##############################################################################
##
## Shared Coeftable Implementation
##
##############################################################################

function StatsAPI.coeftable(m::AbstractRegressModel; level = 0.95)
    cc = coef(m)
    se = stderror(m)
    tt = t_stats(m)
    pv = p_values(m)
    coefnms = String[string(c) for c in coefnames(m)]

    # Compute confidence intervals using precomputed se
    scale = tdistinvcdf(inference_dof(m), 1 - (1 - level) / 2)
    conf_int = hcat(cc .- scale .* se, cc .+ scale .* se)

    # Put (intercept) last if present
    if !isempty(coefnms) && coefnms[1] == "(Intercept)"
        newindex = vcat(2:length(cc), 1)
        cc = cc[newindex]
        se = se[newindex]
        tt = tt[newindex]
        pv = pv[newindex]
        conf_int = conf_int[newindex, :]
        coefnms = coefnms[newindex]
    end

    levstr = isinteger(100 * level) ? string(Integer(100 * level)) : string(100 * level)
    CoefTable(
        hcat(cc, se, tt, pv, conf_int),
        ["Estimate", "Std. Error", "t-stat", "Pr(>|t|)",
            "Lower $levstr%", "Upper $levstr%"],
        coefnms, 4)
end

##############################################################################
##
## Shared Show Methods
##
##############################################################################

# Helper for top summary
function _summary_table_common(m::AbstractRegressModel)
    F_val = m.F
    p_val = m.p

    out = ["Number of obs" sprint(show, nobs(m), context = :compact => true);
           "dof (model)" sprint(show, dof(m), context = :compact => true);
           "dof (residuals)" sprint(show, dof_residual(m), context = :compact => true);
           "R²" @sprintf("%.3f", r2(m));
           "R² adjusted" @sprintf("%.3f", adjr2(m));
           "F-statistic" sprint(show, F_val, context = :compact => true);
           "P-value" @sprintf("%.3f", p_val);]

    if has_fe(m)
        out = vcat(out, ["R² within" @sprintf("%.3f", m.r2_within)])
    end
    return out
end

# We can't fully unify show() yet because OLS and IV have different "top" sections 
# (IV has First Stage stats, OLS has converged/iterations for FE).
# But we can share the CoefTable printing logic if we want.
# For now, let's keep the show() methods specialized but maybe use the helper above.

##############################################################################
##
## Shared Helper for Vcov Refitting
##
##############################################################################

"""
    _calculate_vcov_stats(m, vcov_mat)

Helper to calculate stats (se, t_stats, p_values, F) given a new vcov matrix. The tests
use `_test_dof(m, vcov_mat)` degrees of freedom (see [`inference_dof`](@ref)).
"""
function _calculate_vcov_stats(m::AbstractRegressModel, vcov_mat::AbstractMatrix)
    se = sqrt.(diag(vcov_mat))
    cc = coef(m)
    t_stat = cc ./ se
    df = _test_dof(m, vcov_mat)
    p_val = 2 .* tdistccdf.(df, abs.(t_stat))

    # Compute robust F-statistic (Wald test)
    has_int = model_hasintercept(m)
    F_stat, F_p_val = compute_robust_fstat(cc, vcov_mat, has_int, df)

    return se, t_stat, p_val, F_stat, F_p_val
end
