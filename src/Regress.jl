module Regress

using CovarianceMatrices: CovarianceMatrices, AbstractAsymptoticVarianceEstimator,
                          aVar, momentmatrix, stderror, vcov,
                          HC0, HC1, HC2, HC3, HC4, HC5,
                          CR0, CR1, CR2, CR3,
                          Bartlett, Parzen, QuadraticSpectral, TukeyHanning, Truncated,
                          Information, Misspecified, Uncorrelated, VcovSpec
using DataFrames: DataFrames, AsTable, DataFrame, Not, combine, completecases,
                  disallowmissing, disallowmissing!, dropmissing, leftjoin,
                  nrow, select
using FixedEffects: FixedEffects, AbstractFixedEffectSolver, FixedEffect,
                    solve_coefficients!, solve_residuals!
using LinearAlgebra: LinearAlgebra, BLAS, Cholesky, ColumnNorm, Hermitian, I,
                     Symmetric, UpperTriangular, cholesky, cholesky!, diag,
                     diagm, eigvals, issuccess, ldiv!, mul!, qr, rank, rmul!,
                     svd, tr
using PrecompileTools: PrecompileTools, @compile_workload
using Printf: Printf, @printf, @sprintf
using Reexport: Reexport, @reexport
using StableRNGs: StableRNGs
using Statistics: Statistics
using StatsAPI: StatsAPI, adjr2, coef, coefnames, coeftable, confint, deviance,
                dof, dof_residual, fitted, islinear, leverage, loglikelihood,
                modelmatrix, nobs, nulldeviance, nullloglikelihood, predict,
                r2, residuals, response, responsename, rss, stderror, vcov, weights
using StatsBase: StatsBase, AbstractWeights, CoefTable, UnitWeights, Weights,
                 mean, uweights
using StatsFuns: StatsFuns, chisqccdf, fdistccdf, tdistccdf, tdistinvcdf
@reexport using StatsModels
using StatsModels: StatsModels, @formula, AbstractTerm, ConstantTerm,
                   FormulaTerm, FunctionTerm, InteractionTerm, InterceptTerm,
                   MatrixTerm, StatisticalModel, Term, apply_schema, coefnames,
                   formula, hasintercept, modelmatrix, omitsintercept,
                   response, schema, term
using Tables: Tables

# Re-export CovarianceMatrices.jl for post-estimation vcov
@reexport using CovarianceMatrices

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
include("AbstractModel.jl")
include("LinearModel.jl")
include("IVModel.jl")

# Covariance utilities (must be before fitting functions)
include("utils/covariance.jl")  # Helper functions for covariance calculations

# Estimator implementations
include("estimators/tsls.jl")  # TSLS implementation
include("estimators/kclass.jl")  # K-class estimators: LIML, Fuller, KClass

# Fitting functions
include("fit_ols.jl")  # NEW: Pure OLS implementation
include("fit.jl")      # REFACTORED: Just thin wrappers now
include("partial_out.jl")

# Main estimation functions
export ols, iv, fe

# Model types
export OLSEstimator, OLSMatrixEstimator, IVEstimator, IVMatrixEstimator

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

# Re-export StatsAPI functions for user convenience
export coef, coefnames, coeftable, confint, stderror, vcov
export nobs, dof, dof_residual
export r2, adjr2, deviance, nulldeviance, loglikelihood, nullloglikelihood, rss
export residuals, fitted, response, predict, modelmatrix, weights
export islinear, responsename, leverage

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
        y = [3.0, 4.0, 4.0, 5.0, 1.0, 2.0], id = [1, 1, 2, 2, 3, 3],
        z = [2.0, 3.0, 1.0, 4.0, 2.0, 5.0])  # instrument

    # OLS precompilation
    model = ols(df, @formula(y ~ x1 + x2))
    ols(df, @formula(y ~ x1 + fe(id)))

    # Post-estimation vcov with + operator syntax
    model_hc3 = model + vcov(HC3())
    vcov(model_hc3)
    stderror(model_hc3)

    # IV precompilation (TSLS)
    iv_model = iv(TSLS(), df, @formula(y ~ x2 + (x1 ~ z)))
    vcov(iv_model)
    stderror(iv_model)
end

end
