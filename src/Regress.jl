
module Regress

using DataFrames
using FixedEffects
using LinearAlgebra
using Printf
using Reexport
using PrecompileTools
using StableRNGs
using Statistics
using StatsAPI
using StatsBase
using StatsFuns
@reexport using StatsModels
using Tables

# CovarianceMatrices.jl for post-estimation vcov
@reexport using CovarianceMatrices
using CovarianceMatrices: AbstractAsymptoticVarianceEstimator
using CovarianceMatrices: HC0, HC1, HC2, HC3, HC4, HC5
using CovarianceMatrices: CR0, CR1, CR2, CR3
using CovarianceMatrices: Bartlett, Parzen, QuadraticSpectral, TukeyHanning, Truncated
using CovarianceMatrices: Information, Misspecified
using CovarianceMatrices: Uncorrelated

include("utils/fixedeffects.jl")
include("utils/basecol.jl")
include("utils/tss.jl")
include("utils/formula.jl")
include("utils/fit_common.jl")  # Shared utilities for fitting
include("utils/ranktest.jl")    # Kleibergen-Paap rank test for IV
include("utils/vcov_spec.jl")   # VcovSpec wrapper for model + vcov() syntax
include("utils/robust_fstat.jl")  # Robust Wald F-statistic computation
include("utils/vcov_copy.jl")   # Deep copy utilities for vcov estimators
include("utils/kclass_utils.jl")  # K-class estimation utilities (LIML, Fuller)
include("utils/show_utils.jl")   # Display formatting utilities

# New component types (GLM-style architecture)
include("response.jl")              # OLSResponse
include("predictor.jl")             # OLSLinearPredictor (Chol and QR)
include("fixedeffects_component.jl") # OLSFixedEffects
include("ols_solver.jl")            # Solver utilities

# Model types and estimators
include("estimators.jl")
include("LinearModel.jl")
include("IVModel.jl")

# Covariance utilities (must be before fitting functions)
include("utils/covariance.jl")  # Helper functions for covariance calculations

# Estimator implementations
include("estimators/tsls.jl")  # TSLS implementation
include("estimators/liml.jl")  # LIML note (implementation in kclass.jl)
include("estimators/kclass.jl")  # K-class estimators: LIML, Fuller, KClass

# Keep FixedEffectModel for backwards compatibility (if needed)
# include("FixedEffectModel.jl")

# Fitting functions
include("fit_ols.jl")  # NEW: Pure OLS implementation
include("fit.jl")      # REFACTORED: Just thin wrappers now
include("partial_out.jl")
# Export from StatsBase
# export coef, coefnames, coeftable, responsename, vcov, stderror, nobs, dof, dof_residual, r2,  r², adjr2, adjr², islinear, deviance, nulldeviance, rss, mss, confint, predict, residuals, fit,
#     loglikelihood, nullloglikelihood, dof_fes

# Main estimation functions
export ols, iv, fe

# Model types
export OLSEstimator, OLSMatrixEstimator, IVEstimator

# VcovSpec for model + vcov() syntax
export VcovSpec

# # Component types (for advanced usage)
# export OLSResponse, OLSLinearPredictor, OLSPredictorChol, OLSPredictorQR, OLSFixedEffects

# IV Estimators
export AbstractIVEstimator, TSLS, LIML, Fuller, KClass

# First-stage diagnostics
export FirstStageResult, first_stage

# Utility functions
export partial_out
# fe,
# has_iv,
# has_fe,
# Vcov,
# esample  # Helper for subsetting vectors to estimation sample

# Re-export commonly used CovarianceMatrices.jl estimators

##############################################################################
##
## Helper Function for Subsetting to Estimation Sample
##
##############################################################################

"""
    esample(model::Union{OLSEstimator, IVEstimator, FixedEffectModel}, v::AbstractVector)

Return a boolean vector indicating which observations in `v` were used in fitting `model`.

# Arguments
- `model`: A fitted model (OLSEstimator, IVEstimator, or FixedEffectModel)
"""
function esample(m::Union{OLSEstimator, IVEstimator}, v::AbstractVector)
    length(v) == length(m.esample) ||
        throw(ArgumentError("Vector length ($(length(v))) must match original data length ($(length(m.esample)))"))
    return v[m.esample]
end

@compile_workload begin
    df = DataFrame(
        x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], x2 = [1.0, 2.0, 4.0, 4.0, 3.0, 5.0],
        y = [3.0, 4.0, 4.0, 5.0, 1.0, 2.0], id = [1, 1, 2, 2, 3, 3])
    model = ols(df, @formula(y ~ x1 + x2))
    ols(df, @formula(y ~ x1 + fe(id)))
    # Post-estimation vcov with + operator syntax
    model_hc3 = model + vcov(HC3())
    vcov(model_hc3)
    stderror(model_hc3)
end

end
