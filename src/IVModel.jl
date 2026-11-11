##############################################################################
##
## Abstract Test Types
##
##############################################################################

"""
    AbstractTest

Abstract supertype for test result types (first-stage F-tests, weak IV tests,
endogeneity and overidentification tests).

# Interface
- `StatsAPI.pvalue(t)` returns the test's p-value where one is meaningful.
  Single-statistic tests return a scalar; per-endogenous first-stage tests
  return a vector aligned with the endogenous variables. The composite
  first-stage results (`FirstStageResult`, `FirstStageIV`) return the robust
  p-values; their non-robust counterparts remain accessible as fields.
  `WeakIVTestResult` implements no `pvalue` — it reports against critical
  values rather than a p-value.
- `StatsAPI.dof(t)` returns a scalar degrees of freedom where one is
  unambiguous (`SarganResult`). F-based tests carry `df1`/`df2` fields instead
  and implement no scalar `dof`.
"""
abstract type AbstractTest end

"""
    Homoskedastic

Sentinel type indicating the IID (homoskedastic) variance assumption.
Used as the `K` type parameter in `FirstStageFTest{T, Homoskedastic}`.
"""
struct Homoskedastic end

"""
    FirstStageFTest{T, K} <: AbstractTest

First-stage F-test result. Parametric on:
- `T`: `Float64` (single endogenous or joint KP) or `Vector{Float64}` (multiple endogenous)
- `K`: variance estimator type — `Homoskedastic` for IID, or a CovarianceMatrices type for robust/KP

# Fields
- `stat::T`: F-statistic value(s)
- `df1::Int`: numerator degrees of freedom
- `df2::Int`: denominator degrees of freedom (0 for asymptotic KP)
- `p::T`: p-value(s)
- `type::Symbol`: `:iid`, `:robust`, or `:kp`
- `vcov_estimator::K`: variance estimator used
- `endo_names::Vector{String}`: endogenous variable names

# Examples
```julia
m = iv(TSLS(), df, @formula(y ~ x + (endo ~ z1 + z2)))
first_stage_F_iid(m)      # IID F-test
first_stage_F_robust(m)   # Robust Wald F-test
first_stage_F_KP(m)       # Kleibergen-Paap rk Wald F-test
```
"""
struct FirstStageFTest{T, K} <: AbstractTest
    stat::T
    df1::Int
    df2::Int
    p::T
    type::Symbol
    vcov_estimator::K
    endo_names::Vector{String}
end

function Base.show(io::IO, f::FirstStageFTest{<:Real})
    type_str = if f.type == :iid
        "First-stage F-test (IID)"
    elseif f.type == :kp
        "Kleibergen-Paap rk Wald F-test"
    else
        vcov_name = string(typeof(f.vcov_estimator).name.name)
        "First-stage F-test (robust, $vcov_name)"
    end
    println(io, type_str)
    name = isempty(f.endo_names) ? "joint" : f.endo_names[1]
    p_str = f.p < 0.001 ? "p < 0.001" : @sprintf("p = %.4f", f.p)
    if f.df2 > 0
        @printf(io, "  %s: F(%d, %d) = %.4f  (%s)\n", name, f.df1, f.df2, f.stat, p_str)
    else
        @printf(io, "  %s: F = %.4f, df = %d  (%s)\n", name, f.stat, f.df1, p_str)
    end
end

function Base.show(io::IO, f::FirstStageFTest{<:AbstractVector})
    type_str = if f.type == :iid
        "First-stage F-test (IID)"
    else
        vcov_name = string(typeof(f.vcov_estimator).name.name)
        "First-stage F-test (robust, $vcov_name)"
    end
    if f.df2 > 0
        println(io, "$type_str, F($(f.df1), $(f.df2)):")
    else
        println(io, "$type_str:")
    end
    for (j, name) in enumerate(f.endo_names)
        p_str = f.p[j] < 0.001 ? "p < 0.001" : @sprintf("p = %.4f", f.p[j])
        @printf(io, "  %s: F = %.4f  (%s)\n", name, f.stat[j], p_str)
    end
end

StatsAPI.pvalue(f::FirstStageFTest) = f.p

##############################################################################
##
## Type IVEstimator (for IV estimation)
##
##############################################################################

"""
    FirstStageData{T}

Container for first-stage regression data needed for per-endogenous F-statistics
and recomputation with different variance estimators.

# Fields
- `Pi::Matrix{T}`: First-stage coefficients (instruments portion only), l × k matrix
- `Xendo_res::Matrix{T}`: Residualized endogenous variables (n × k)
- `Z_res::Matrix{T}`: Residualized instruments (n × l)
- `endogenous_names::Vector{String}`: Names of endogenous variables
- `n_exo::Int`: Number of exogenous variables in first stage
- `Xendo_orig::Matrix{T}`: Original endogenous variables (n × k, not residualized)
- `newZ::Matrix{T}`: Full first-stage design matrix [Xexo, Z] (n × (k_exo + l))
- `has_intercept::Bool`: Whether model has intercept
"""
struct FirstStageData{T}
    Pi::Matrix{T}
    Xendo_res::Matrix{T}
    Z_res::Matrix{T}
    endogenous_names::Vector{String}
    n_exo::Int
    # For correct robust F-stat computation (per-endogenous), we need original data
    # The K-P rank test uses residualized data, but robust Wald F needs original
    Xendo_orig::Matrix{T}  # Original endogenous variables (n × k, not residualized)
    newZ::Matrix{T}        # Full first-stage design matrix [Xexo, Z] (n × (k_exo + l))
    has_intercept::Bool    # Whether model has intercept
end

"""
    FirstStageResult{T}

Container for first-stage regression diagnostics from IV estimation.
Returned by `first_stage(model)`.

# Fields
- `endogenous_names::Vector{String}`: Names of endogenous variables
- `F_nonrobust::Vector{T}`: Homoskedastic first-stage F-statistic for each endogenous variable
- `p_nonrobust::Vector{T}`: P-value of the homoskedastic F-statistic
- `F_robust::Vector{T}`: Robust first-stage F-statistic for each endogenous variable
- `p_robust::Vector{T}`: P-value of the robust F-statistic
- `df1::Int`: Numerator degrees of freedom (number of excluded instruments)
- `df2::Int`: Denominator degrees of freedom (first-stage residual df)
- `n_endogenous::Int`: Number of endogenous variables
- `n_instruments::Int`: Number of excluded instruments
- `vcov_type::String`: Variance estimator used for `F_robust`

# Examples
```julia
model = iv(TSLS(), df, @formula(y ~ x + (endo ~ z1 + z2)))
fs = first_stage(model)
fs.F_nonrobust       # Homoskedastic first-stage F-statistics
fs.F_robust          # Robust first-stage F-statistics
```
"""
struct FirstStageResult{T <: AbstractFloat} <: AbstractTest
    endogenous_names::Vector{String}
    F_nonrobust::Vector{T}
    p_nonrobust::Vector{T}
    F_robust::Vector{T}
    p_robust::Vector{T}
    df1::Int
    df2::Int
    n_endogenous::Int
    n_instruments::Int
    vcov_type::String
end

StatsAPI.pvalue(fs::FirstStageResult) = fs.p_robust

"""
    FirstStageIV{T, M}

First-stage regression results for matrix-based IV estimation.

Contains full OLS regression models for each endogenous variable,
plus fixest-style non-robust and robust first-stage F-statistics.

# Fields
- `models::Vector{M}`: OLS models (one per endogenous variable)
- `endogenous_names::Vector{String}`: Names of endogenous variables
- `instrument_names::Vector{String}`: Names of excluded instruments
- `F_nonrobust::Vector{T}`: Non-robust F per endogenous variable
- `p_nonrobust::Vector{T}`: P-values for the non-robust F-statistics
- `F_robust::Vector{T}`: Robust F per endogenous variable using the model's vcov
- `p_robust::Vector{T}`: P-values for the robust F-statistics
- `df1::Int`: Numerator degrees of freedom
- `df2::Int`: Denominator degrees of freedom
- `vcov_type::String`: Variance estimator used for `F_robust`

# Example
```julia
m = iv(TSLS(), Z, X, y; n_endogenous=1)
fs = first_stage(m)
coef(fs.models[1])       # first-stage coefficients
stderror(fs.models[1])   # standard errors
fitted(fs.models[1])     # predicted endogenous
fs.F_nonrobust           # non-robust F
fs.F_robust              # robust F
```
"""
struct FirstStageIV{T <: AbstractFloat, M <: OLSMatrixEstimator} <: AbstractTest
    models::Vector{M}
    endogenous_names::Vector{String}
    instrument_names::Vector{String}
    F_nonrobust::Vector{T}
    p_nonrobust::Vector{T}
    F_robust::Vector{T}
    p_robust::Vector{T}
    df1::Int
    df2::Int
    vcov_type::String
end

function Base.show(io::IO, fs::FirstStageIV{T}) where {T}
    n_endo = length(fs.models)
    println(io, "First-Stage IV Regressions ($n_endo endogenous)")
    println(io, "─" ^ 60)

    for (j, name) in enumerate(fs.endogenous_names)
        m = fs.models[j]
        @printf(io, "  %-20s  R² = %.4f   F(nonrobust) = %.3f   F(%s) = %.3f\n",
            name, r2(m), fs.F_nonrobust[j], fs.vcov_type, fs.F_robust[j])
    end

    println(io, "─" ^ 60)
    @printf(io, "DoF: (%d, %d)\n", fs.df1, fs.df2)
    println(io, "Excluded instruments: ", join(fs.instrument_names, ", "))
end

StatsAPI.pvalue(fs::FirstStageIV) = fs.p_robust

"""
    has_first_stage_data(fsd::FirstStageData) -> Bool

Check if FirstStageData has actual data (not an empty sentinel).
"""
has_first_stage_data(fsd::FirstStageData) = !isempty(fsd.Pi)

"""
    empty_first_stage_data(::Type{T}) where T

Create an empty FirstStageData sentinel.
"""
function empty_first_stage_data(::Type{T}) where {T <: AbstractFloat}
    FirstStageData{T}(
        Matrix{T}(undef, 0, 0),
        Matrix{T}(undef, 0, 0),
        Matrix{T}(undef, 0, 0),
        String[],
        0,
        Matrix{T}(undef, 0, 0),
        Matrix{T}(undef, 0, 0),
        false
    )
end

"""
    PostEstimationDataIV{T}

Container for post-estimation data required for IV variance-covariance calculations.

# Fields
- `X_fitted::Matrix{T}`: Design matrix used for inference (predicted endogenous for TSLS, Adj for K-class)
- `X_original::Matrix{T}`: Original regressor matrix [Xexo, Xendo]
- `y::Vector{T}`: Response vector (demeaned if FE present)
- `crossx::Cholesky{T, Matrix{T}}`: Cholesky factorization of X'X
- `invXX::Symmetric{T, Matrix{T}}`: Inverse of X'X (or inv(A) for K-class)
- `weights::AbstractWeights`: Weights used in estimation
- `cluster_vars::NamedTuple`: Cluster variables (subsetted to esample)
- `basis_coef::BitVector`: Indicator of which coefficients are not collinear
- `first_stage_data::FirstStageData{T}`: First-stage data for F-statistics (empty if not computed)
- `Adj::Matrix{T}`: K-class adjustment matrix (empty 0×0 for TSLS)
- `kappa::T`: K-class parameter (NaN for TSLS, k_LIML for LIML, etc.)
- `fe_groups::Vector{Vector{Int}}`: FE grouping vectors for nesting detection (one per FE dimension)
- `fe_names::Vector{Symbol}`: Names of FE variables
- `ngroups::Vector{Int}`: Number of groups per FE dimension

Use `has_first_stage_data(pe.first_stage_data)` to check if first-stage data is available.
Use `has_kclass_adj(pe)` to check if K-class adjustment matrix is available.
"""
struct PostEstimationDataIV{T}
    X_fitted::Matrix{T}         # Predicted/adjusted endogenous used for inference (X̂ for TSLS, Adj for K-class)
    X_original::Matrix{T}       # Original regressors [Xexo, Xendo]
    y::Vector{T}                # Response vector (demeaned if FE present)
    crossx::Cholesky{T, Matrix{T}}
    invXX::Symmetric{T, Matrix{T}}  # (X̂'X̂)^{-1}
    weights::AbstractWeights
    cluster_vars::NamedTuple
    basis_coef::BitVector
    first_stage_data::FirstStageData{T}
    Adj::Matrix{T}              # Empty (0×0) for TSLS, filled for K-class
    kappa::T                    # NaN for TSLS
    # For leverage computation (AER formula)
    Z::Matrix{T}                # Instruments matrix
    invZZ::Symmetric{T, Matrix{T}}  # (Z'Z)^{-1}
    # FE nesting detection fields
    fe_groups::Vector{Vector{Int}}   # One ref vector per FE dimension
    fe_names::Vector{Symbol}          # Names of FE variables
    ngroups::Vector{Int}              # Number of groups per FE dimension
end

"""
    has_kclass_adj(pe::PostEstimationDataIV) -> Bool

Check if the post-estimation data has K-class adjustment matrix.
"""
has_kclass_adj(pe::PostEstimationDataIV) = !isempty(pe.Adj)

"""
    IVEstimator <: AbstractRegressModel

Model type for instrumental variables regression.

Use `iv(estimator, df, formula)` to fit this model type, where `estimator` is
one of `TSLS()`, `LIML()`, etc.

# Type Parameters
- `T`: Float type (Float64 or Float32)
- `E`: IV Estimator type (TSLS, LIML, etc.)
- `V`: Variance-covariance estimator type (HC1, HC3, CR1, etc.)
- `P`: Post-estimation data type (PostEstimationDataIV or Nothing)

# Examples
```julia
iv(TSLS(), df, @formula(y ~ x + (endo ~ instrument)))
```
"""
struct IVEstimator{
    T, E <: AbstractIVEstimator, V, P <: Union{PostEstimationDataIV{T}, Nothing},
    C <: AbstractMatrix{T}} <:
       AbstractRegressModel
    estimator::E  # Which IV estimator was used

    coef::Vector{T}   # Vector of coefficients

    esample::BitVector      # Is the row of the original dataframe part of the estimation sample?
    residuals_esample::Vector{T}  # Unweighted residuals for esample rows only (see `residuals_for_vcov`)
    has_residuals::Bool           # Whether residuals were saved
    fe::DataFrame

    # Post-estimation data for CovarianceMatrices.jl
    postestimation::P

    fekeys::Vector{Symbol}

    coefnames::Vector{String}       # Name of coefficients
    responsename::Union{String, Symbol} # Name of dependent variable
    formula::FormulaTerm        # Original formula
    formula_schema::FormulaTerm # Schema for predict
    contrasts::Dict{Symbol, Any}

    nobs::Int64             # Number of observations
    dof::Int64              # Number parameters estimated - has_intercept. Used for p-value of F-stat.
    dof_fes::Int64          # Number of fixed effects
    dof_residual::Int64     # dof used for t-test and p-value of F-stat. nobs - degrees of freedoms with simple std
    rss::T            # Sum of squared residuals
    tss::T            # Total sum of squares

    # for FE
    iterations::Int         # Number of iterations
    converged::Bool         # Has the demeaning algorithm converged?
    r2_within::T      # within r2 (with fixed effect)

    # Variance-covariance estimator and precomputed statistics
    vcov_estimator::V                        # Estimator used to compute vcov_matrix
    vcov_matrix::C                           # Precomputed vcov matrix
    se::Vector{T}                            # Standard errors
    t_stats::Vector{T}                       # t-statistics
    p_values::Vector{T}                      # p-values

    # Test statistics (computed with vcov_estimator)
    F::T                    # F-statistic (Wald test)
    p::T                    # P-value of F-stat
    F_first_stage_nonrobust::Vector{T}
    p_first_stage_nonrobust::Vector{T}
    F_first_stage_robust::Vector{T}
    p_first_stage_robust::Vector{T}
    F_kp::T                 # Joint Kleibergen-Paap rk Wald F-statistic
    p_kp::T                 # P-value of joint KP test
end

has_iv(::IVEstimator) = true
has_fe(m::IVEstimator) = has_fe(m.formula)
r2_within(m::IVEstimator) = m.r2_within
model_hasintercept(m::IVEstimator) = hasintercept(m.formula_schema)

"""
    has_residuals_data(m::IVEstimator) -> Bool

Check if residuals were saved during fitting.
"""
has_residuals_data(m::IVEstimator) = m.has_residuals

"""
    residuals_for_vcov(m::IVEstimator) -> Vector{T}

Get residuals for variance-covariance computation (esample rows only).

The result is on the same scale as the stored design matrices: for a weighted
fit, `X_fitted`, `X_original` and `y` are multiplied by `√w`, so the residuals
are too. `m.residuals_esample` holds the unweighted residuals that
`residuals(m)` returns.
"""
function residuals_for_vcov(m::IVEstimator)
    pe = m.postestimation
    isnothing(pe) &&
        throw(ArgumentError("Model does not have post-estimation data stored, so its weights are unknown."))
    pe.weights isa UnitWeights && return m.residuals_esample
    return m.residuals_esample .* sqrt.(pe.weights)
end

##############################################################################
##
## StatsAPI Interface
##
##############################################################################

function StatsAPI.coef(m::IVEstimator)
    # Return 0.0 for collinear coefficients (backward compatibility)
    if !isnothing(m.postestimation) && !isempty(m.postestimation.basis_coef)
        beta = copy(m.coef)
        beta[.!m.postestimation.basis_coef] .= zero(eltype(beta))
        return beta
    else
        return m.coef
    end
end
StatsAPI.coefnames(m::IVEstimator) = m.coefnames
StatsAPI.responsename(m::IVEstimator) = m.responsename

# Variance-covariance (returns precomputed matrix)
StatsAPI.vcov(m::IVEstimator) = m.vcov_matrix

# Standard errors (returns precomputed values)
StatsAPI.stderror(m::IVEstimator) = m.se

StatsAPI.nobs(m::IVEstimator) = m.nobs
StatsAPI.dof(m::IVEstimator) = m.dof
StatsAPI.dof_residual(m::IVEstimator) = m.dof_residual
StatsAPI.r2(m::IVEstimator) = r2(m, :devianceratio)
StatsAPI.deviance(m::IVEstimator) = rss(m)
StatsAPI.nulldeviance(m::IVEstimator) = m.tss
StatsAPI.rss(m::IVEstimator) = m.rss
StatsModels.formula(m::IVEstimator) = m.formula_schema
dof_fes(m::IVEstimator) = m.dof_fes

##############################################################################
##
## CovarianceMatrices.jl Interface for Post-Estimation vcov
##
##############################################################################

"""
    CovarianceMatrices.momentmatrix(m::IVEstimator)

Returns the moment matrix for the model.
- For TSLS: X .* residuals (X contains predicted endogenous)
- For K-class (LIML/Fuller): Adj .* residuals where Adj = W - k*Wres

Required for post-estimation variance-covariance calculations.
"""
function CovarianceMatrices.momentmatrix(m::IVEstimator)
    isnothing(m.postestimation) &&
        throw(ArgumentError("Model does not have post-estimation data stored. Post-estimation vcov not available."))
    !has_residuals_data(m) &&
        throw(ArgumentError("Model does not have residuals stored. Use save=:residuals or save=:all when fitting."))

    # Use Adj if available (K-class: LIML, Fuller), otherwise use X (TSLS)
    pe = m.postestimation
    X_for_vcov = has_kclass_adj(pe) ? pe.Adj : pe.X_fitted
    return X_for_vcov .* residuals_for_vcov(m)
end

##############################################################################
##
## CovarianceMatrices.jl aVar Interface for IVEstimator
##
##############################################################################

# Local alias for CovarianceMatrices (also defined in covariance.jl for OLS)
const _CM = CovarianceMatrices

"""
    CovarianceMatrices.bread(m::IVEstimator)

Compute (X'X)^(-1), the "bread" of the sandwich variance estimator for IV.
Uses the predicted endogenous variables (Xhat) in the design matrix.
"""
_CM.bread(m::IVEstimator) = m.postestimation.invXX

# CovarianceMatrices declares its own `leverage`, distinct from `StatsAPI.leverage`;
# its HC2-HC5 and CR2/CR3 residual adjustments dispatch on that one.
_CM.leverage(m::IVEstimator) = StatsAPI.leverage(m)

# CovarianceMatrices.jl uses numobs, which is distinct from StatsAPI.nobs
_CM.numobs(m::IVEstimator) = m.nobs

function _CM.mask(m::IVEstimator)
    isnothing(m.postestimation) &&
        throw(ArgumentError("Model does not have post-estimation data stored. Post-estimation vcov not available."))
    return m.postestimation.basis_coef
end

"""
    leverage(m::IVEstimator)

Compute leverage values (diagonal of hat matrix) for IV models.

Uses the AER/sandwich formula for IV models:
    h = diag(X * (X̂'X̂)^{-1} * X' * Z * (Z'Z)^{-1} * Z')

Where:
- X = original regressors
- X̂ = fitted values from first stage
- Z = instruments

This matches R's AER::ivreg and sandwich::vcovHC for IV models.
For K-class: Uses Adj matrix (stored in pe.Adj) when X is empty.
"""
function StatsAPI.leverage(m::IVEstimator)
    isnothing(m.postestimation) &&
        throw(ArgumentError("Model does not have post-estimation data stored."))
    pe = m.postestimation

    # For K-class estimators, use the Adj matrix if available
    if has_kclass_adj(pe)
        # K-class uses simple leverage: h = diag(Adj * invXX * Adj')
        Adj = pe.Adj
        invXX = pe.invXX
        return vec(sum(Adj .* (Adj * invXX), dims = 2))
    end

    return _tsls_leverage(pe.X_original, pe.X_fitted, pe.invXX)
end

# TSLS leverage, as R's AER::ivreg: h = diag(X (X̂'X̂)⁻¹ X' P_Z) with P_Z the projection on
# all the instruments. P_Z X is the first-stage fit X̂, so row i is X_i (X̂'X̂)⁻¹ X̂_i'.
# This holds whatever the instruments are, fixed effects used as instruments included,
# and costs O(n·k) rather than the O(n²) of forming P_Z.
_tsls_leverage(X, Xhat, invXhatXhat) = vec(sum((X * invXhatXhat) .* Xhat, dims = 2))

# Residual adjustments for HC/HR estimators come from
# `CovarianceMatrices.residual_adjustment`, which reaches the model through the
# `leverage` and `numobs` protocol methods defined above.
@noinline residualadjustment(k::_CM.HR, m::IVEstimator) = _CM.residual_adjustment(k, m)

# HR1 is the exception: upstream divides by `n - length(coef(m))`, which cannot see
# fixed effects absorbed out of the design, while `dof_residual` counts them.
@noinline residualadjustment(k::_CM.HR1, m::IVEstimator) = sqrt(nobs(m) / dof_residual(m))

# Cluster-robust residual adjustments
@noinline residualadjustment(k::_CM.CR0, m::IVEstimator) = 1.0
@noinline residualadjustment(k::_CM.CR1, m::IVEstimator) = 1.0

# Correlated estimators (HAC, EWC, DriscollKraay, VARHAC, etc.) - no residual adjustment needed
@noinline residualadjustment(k::_CM.Correlated, m::IVEstimator) = 1.0

# The moment matrix of an IV model has the second-stage regressors (`Adj` for the
# K-class estimators, the fitted regressors otherwise) times the residuals.
function _regress_aVar(k, m::IVEstimator; demean = false, prewhite = false, scale = true)
    isnothing(m.postestimation) &&
        throw(ArgumentError("Model does not have post-estimation data stored."))
    !has_residuals_data(m) && throw(ArgumentError("Model does not have residuals stored."))

    # Compute adjusted moment matrix
    # Use Adj if available (K-class: LIML, Fuller), otherwise use X (TSLS)
    pe = m.postestimation
    X_for_vcov = has_kclass_adj(pe) ? pe.Adj : pe.X_fitted

    M = X_for_vcov .* residuals_for_vcov(m)
    # The adjustment can be Float64 on a Float32 model; the moments keep the model's type.
    M = M .* eltype(M).(residualadjustment(k, m))

    # Bandwidth-selection weights come from the regressor matrix, not the moment
    # matrix: they must give the intercept weight 0, and the intercept column of
    # the moment matrix is not constant. `nothing` for non-HAC estimators.
    kw = _CM.bandwidthweights(k, pe.X_original)

    return _moment_aVar(k, M; demean, prewhite, scale, weights = kw)
end

function _regress_aVar(k::_CM.CR, m::IVEstimator; demean = false, prewhite = false,
        scale = true)
    isnothing(m.postestimation) &&
        throw(ArgumentError("Model does not have post-estimation data stored."))
    !has_residuals_data(m) && throw(ArgumentError("Model does not have residuals stored."))

    # Compute adjusted moment matrix
    # Use Adj if available (K-class: LIML, Fuller), otherwise use X (TSLS)
    pe = m.postestimation
    X_for_vcov = has_kclass_adj(pe) ? pe.Adj : pe.X_fitted

    # CR0/CR1 leave the residuals unadjusted; the CR1 correction is a `vcov` scale factor.
    # The CR2/CR3 hat matrix is that of the second-stage regressors, not of
    # `modelmatrix(m)`.
    u = residuals_for_vcov(m)
    if k isa Union{_CM.CR2, _CM.CR3}
        w = pe.weights isa UnitWeights ? similar(u, 0) : pe.weights
        return _leverage_aVar(k, X_for_vcov, X_for_vcov, bread(m), u, w;
            demean, prewhite, scale)
    end
    return _moment_aVar(k, X_for_vcov .* u; demean, prewhite, scale)
end

##############################################################################
##
## Post-Estimation vcov Methods
##
## `vcov`, `stderror` and `aVar` are defined in `utils/covariance.jl`; these are
## the IV methods they call.
##
##############################################################################

function _regress_vcov(ve, m::IVEstimator{T}; dofadjust = true, demean = false,
        prewhite = false) where {T}
    isnothing(m.postestimation) &&
        throw(ArgumentError("Model does not have post-estimation data stored. Post-estimation vcov not available."))
    !has_residuals_data(m) &&
        throw(ArgumentError("Model does not have residuals stored. Use save=:residuals or save=:all when fitting."))

    n = nobs(m)
    B = bread(m)

    A = _regress_aVar(ve, m; demean, prewhite)
    scale = T(n * _finite_sample_factor(ve, m, dofadjust))
    Σ = scale .* B * A * B
    return _wrap_vcov(Symmetric(_expand_collinear(Σ, _CM.mask(m))), ve, A)
end

# With collinear columns, the bread and the moments of an IV model cover only the
# estimable columns: the sandwich `Σ` is computed on those, and the full matrix has
# NaN in the rows and columns of the dropped ones, as for OLS models.
function _expand_collinear(Σ::AbstractMatrix{T}, basis::AbstractVector{Bool}) where {T}
    all(basis) && return Σ
    k = count(basis)
    size(Σ) == (k, k) || throw(DimensionMismatch(
        "the covariance of the estimable coefficients is $(size(Σ, 1))×$(size(Σ, 2)), " *
        "expected $k×$k for $k of $(length(basis)) coefficients"))
    full = fill(T(NaN), length(basis), length(basis))
    idx = findall(basis)
    full[idx, idx] = Σ
    return full
end

# Driscoll-Kraay: see the `_regress_vcov` methods for OLS models.
function _regress_vcov(k::_CM.DriscollKraay, m::IVEstimator; kwargs...)
    isnothing(m.postestimation) &&
        throw(ArgumentError("Model does not have post-estimation data stored. Post-estimation vcov not available."))
    Σ = invoke(
        StatsBase.vcov, Tuple{_CM.DriscollKraay, RegressionModel}, k, m; kwargs...)
    return _wrap_vcov(parent(Σ), k, Σ)
end

function _fe_structure(m::IVEstimator)
    pe = m.postestimation
    return (; names = pe.fe_names, groups = pe.fe_groups, ngroups = pe.ngroups,
        cluster_names = keys(pe.cluster_vars))
end

# The stored cluster vectors for the column names of `k`; see `_resolve_clusters`.
function _cluster_vectors(k, m::IVEstimator)
    return map(_cluster_names(k)) do name
        haskey(m.postestimation.cluster_vars, name) || _cluster_not_found_error(name, m)
        m.postestimation.cluster_vars[name]
    end
end

##############################################################################
##
## Helper Functions
##
##############################################################################

function _cluster_not_found_error(cluster_name::Symbol, m::IVEstimator)
    available = isempty(m.postestimation.cluster_vars) ? "none" :
                join(keys(m.postestimation.cluster_vars), ", :")
    throw(ArgumentError("\n    Cluster variable :$cluster_name not found in model.

    Available cluster variables: :$available

    To use this cluster variable, either:
      1. Re-fit with save_cluster=:$cluster_name
      2. Use data directly: vcov(CR1(df.$cluster_name[model.esample]), model)
    "))
end

##############################################################################
##
## Recompute First-Stage F-Statistics with Different vcov
##
##############################################################################

"""
    recompute_first_stage_fstat(m::IVEstimator, vcov_type)

Recompute robust per-endogenous first-stage F-statistics using a different vcov
estimator. Requires model to have stored first-stage data.
"""
function recompute_first_stage_fstat(m::IVEstimator{T}, vcov_type) where {T}
    pe = m.postestimation
    isnothing(pe) && return T[], T[]
    !has_first_stage_data(pe.first_stage_data) && return T[], T[]

    fsd = pe.first_stage_data
    dof_fes_val = dof_fes(m)

    # A CachedCR's buffers are sized for the second-stage moments, not the first-stage ones.
    resolved_vcov = _uncached(_resolve_clusters(vcov_type, m))

    return compute_per_endogenous_fstats(
        fsd.Xendo_res, fsd.Z_res, fsd.Pi,
        resolved_vcov, dof_fes_val;
        Xendo_orig = fsd.Xendo_orig,
        newZ = fsd.newZ,
        first_stage = _FirstStage(m)
    )
end

"""
    recompute_kp_fstat(m::IVEstimator, k) -> (F_kp, p_kp)

Kleibergen-Paap rk Wald F statistic of `m` and its p-value with the variance estimator
`k`. Both are `NaN` when the statistic does not support `k` (see `_kp_supported`). A
model without stored first-stage data keeps its statistic.
"""
function recompute_kp_fstat(m::IVEstimator{T}, k) where {T}
    pe = m.postestimation
    (isnothing(pe) || !has_first_stage_data(pe.first_stage_data)) && return m.F_kp, m.p_kp
    _kp_supported(k) || return T(NaN), T(NaN)
    fsd = pe.first_stage_data
    k = _resolve_clusters(k, m)
    return compute_first_stage_fstat(
        fsd.Xendo_res, fsd.Z_res, fsd.Pi, k, _kp_factor(k, _FirstStage(m)))
end

# The first-stage regression of `m`: the endogenous regressors on the exogenous
# regressors and the excluded instruments, with the absorbed fixed effects of `m`.
struct _FirstStage{M}
    m::M
end
StatsAPI.nobs(f::_FirstStage) = nobs(f.m)
StatsAPI.dof_residual(f::_FirstStage) = last(first_stage_df(f.m))
function _cluster_dof(k, f::_FirstStage)
    fsd = f.m.postestimation.first_stage_data
    return _cluster_dof(k, f.m) - sum(_CM.mask(f.m)) + fsd.n_exo + size(fsd.Z_res, 2)
end

# The factor `c` of `ranktest`: `vcov`'s finite-sample factor, with n / dof_residual for
# HR1, which `vcov` applies as a residual adjustment, and for Uncorrelated, whose σ² is
# RSS / dof_residual.
_kp_factor(k, f) = _finite_sample_factor(k, f, true)
_kp_factor(k::Union{_CM.HR1, _CM.Uncorrelated}, f) = nobs(f) / dof_residual(f)

"""
    first_stage_df(m::IVEstimator)

Return the numerator and denominator degrees of freedom used by the first-stage
F-tests.
"""
function first_stage_df(m::IVEstimator)
    pe = m.postestimation
    isnothing(pe) && return (0, 0)
    fsd = pe.first_stage_data
    !has_first_stage_data(fsd) && return (0, 0)

    n_excl = size(fsd.Z_res, 2)
    df1 = n_excl
    df2 = nobs(m) - fsd.n_exo - n_excl - dof_fes(m)
    return (df1, df2)
end

##############################################################################
##
## + Operator for IVEstimator + VcovSpec
##
##############################################################################

"""
    Base.:+(m::IVEstimator, v::VcovSpec)

Create a new model with a different variance-covariance estimator.

Returns a new `IVEstimator` with the same underlying data but
precomputed vcov statistics and recomputed first-stage diagnostics
(the robust first-stage F statistics and the Kleibergen-Paap statistic)
using the specified variance estimator. The vcov estimator is
deep-copied to avoid aliasing issues.

# Examples
```julia
model = iv(TSLS(), df, @formula(y ~ x + (endo ~ z)))

# Heteroskedasticity-robust
model_hc3 = model + vcov(HC3())
stderror(model_hc3)  # Uses precomputed HC3 standard errors
first_stage(model_hc3).F_robust  # First-stage F recomputed with HC3

# Cluster-robust
model_cr = iv(TSLS(), df, @formula(y ~ x + (endo ~ z)), save_cluster = :firm)
model_cr1 = model_cr + vcov(CR1(:firm))
```

See also: [`VcovSpec`](@ref)
"""
function Base.:+(m::IVEstimator{T, E, V1, P}, v::VcovSpec{V2}) where {T, E, V1, P, V2}
    # Compute vcov matrix using StatsBase.vcov (which dispatches to IVModel.jl methods)
    vcov_mat = StatsBase.vcov(v.source, m)

    # Use shared helper for stats
    se, t_stats, p_values, F_stat, p_val = _calculate_vcov_stats(m, vcov_mat)

    # Recompute robust first-stage F with this vcov type
    F_first_stage_robust, p_first_stage_robust = recompute_first_stage_fstat(m, v.source)
    F_kp, p_kp = recompute_kp_fstat(m, v.source)

    # Return new IVEstimator with same data but different vcov type
    return IVEstimator{T, E, V2, P, typeof(vcov_mat)}(
        m.estimator, m.coef,
        m.esample, m.residuals_esample, m.has_residuals, m.fe,
        m.postestimation, m.fekeys,
        m.coefnames, m.responsename,
        m.formula, m.formula_schema, m.contrasts,
        m.nobs, m.dof, m.dof_fes, m.dof_residual,
        m.rss, m.tss,
        m.iterations, m.converged, m.r2_within,
        v.source, vcov_mat, se, t_stats, p_values,
        F_stat, p_val,
        m.F_first_stage_nonrobust, m.p_first_stage_nonrobust,
        F_first_stage_robust, p_first_stage_robust,
        F_kp, p_kp
    )
end

##############################################################################
##
## New First-Stage F-Test API
##
##############################################################################

"""
    first_stage_F_iid(m::IVEstimator) -> FirstStageFTest

IID (homoskedastic) first-stage F-test. Equation-by-equation SSR-based F-test.
"""
function first_stage_F_iid(m::IVEstimator{T}) where {T}
    pe = m.postestimation
    F_vec = m.F_first_stage_nonrobust
    p_vec = m.p_first_stage_nonrobust

    df1, df2 = first_stage_df(m)
    endo_names = if !isnothing(pe) && has_first_stage_data(pe.first_stage_data)
        pe.first_stage_data.endogenous_names
    else
        String["endo$i" for i in 1:length(F_vec)]
    end

    if length(F_vec) == 1
        return FirstStageFTest{T, Homoskedastic}(
            F_vec[1], df1, df2, p_vec[1], :iid, Homoskedastic(), endo_names)
    else
        return FirstStageFTest{Vector{T}, Homoskedastic}(
            F_vec, df1, df2, p_vec, :iid, Homoskedastic(), endo_names)
    end
end

"""
    first_stage_F_robust(m::IVEstimator) -> FirstStageFTest

Robust Wald first-stage F-test using the model's current variance estimator.
"""
function first_stage_F_robust(m::IVEstimator{T}) where {T}
    pe = m.postestimation
    F_vec = m.F_first_stage_robust
    p_vec = m.p_first_stage_robust

    df1, df2 = first_stage_df(m)
    endo_names = if !isnothing(pe) && has_first_stage_data(pe.first_stage_data)
        pe.first_stage_data.endogenous_names
    else
        String["endo$i" for i in 1:length(F_vec)]
    end

    vcov_est = m.vcov_estimator

    if length(F_vec) == 1
        return FirstStageFTest{T, typeof(vcov_est)}(
            F_vec[1], df1, df2, p_vec[1], :robust, vcov_est, endo_names)
    else
        return FirstStageFTest{Vector{T}, typeof(vcov_est)}(
            F_vec, df1, df2, p_vec, :robust, vcov_est, endo_names)
    end
end

"""
    first_stage_F_KP(m::IVEstimator) -> FirstStageFTest

Joint Kleibergen-Paap rk Wald F-test for under-identification, computed with the
model's variance estimator, `m.vcov_estimator`.

The statistic is defined for `Uncorrelated`, `HC0`, `HC1`, `CR0`, `CR1` (any number of
cluster dimensions) and the HAC kernels; any other estimator throws an `ArgumentError`.
The finite-sample factor is that of `vcov` applied to the first-stage regression
(`n / dof_residual` for `HC1` and the HAC kernels, `G/(G−1)·(n−1)/(n−K)` for `CR1`, with
K the first-stage coefficients). A HAC kernel with an automatic bandwidth selects it from
the first-stage moments.
"""
function first_stage_F_KP(m::IVEstimator{T}) where {T}
    _check_kp_estimator(m.vcov_estimator)
    pe = m.postestimation
    endo_names = if !isnothing(pe) && has_first_stage_data(pe.first_stage_data)
        pe.first_stage_data.endogenous_names
    else
        String[]
    end

    n_excl = if !isnothing(pe) && has_first_stage_data(pe.first_stage_data)
        size(pe.first_stage_data.Z_res, 2)
    else
        0
    end

    # KP is asymptotically χ²(df)/ℓ, use df2=0 to signal asymptotic
    return FirstStageFTest{T, typeof(m.vcov_estimator)}(
        m.F_kp, n_excl, 0, m.p_kp, :kp, m.vcov_estimator, endo_names)
end

##############################################################################
##
## first_stage() - Return first-stage diagnostics as a struct (DEPRECATED)
##
##############################################################################

"""
    first_stage(m::IVEstimator) -> FirstStageResult

Return first-stage diagnostics for an IV model.
Uses the variance estimator stored in the model for the robust F-statistics.

# Examples
```julia
model = iv(TSLS(), df, @formula(y ~ x + (endo ~ z1 + z2)))
fs = first_stage(model)
fs.F_nonrobust       # Homoskedastic first-stage F-statistics
fs.F_robust          # Robust first-stage F-statistics
```

See also: [`FirstStageResult`](@ref)
"""
function first_stage(m::IVEstimator{T}) where {T}
    isnothing(m.postestimation) &&
        throw(ArgumentError("Model does not have post-estimation data stored. Fit with save=true."))
    !has_first_stage_data(m.postestimation.first_stage_data) &&
        throw(ArgumentError("First-stage data not available."))

    fsd = m.postestimation.first_stage_data

    # Get vcov type name
    vcov_name = string(typeof(m.vcov_estimator).name.name)

    df1, df2 = first_stage_df(m)

    endogenous_names = copy(fsd.endogenous_names[1:length(m.F_first_stage_nonrobust)])

    FirstStageResult{T}(
        endogenous_names,
        m.F_first_stage_nonrobust,
        m.p_first_stage_nonrobust,
        m.F_first_stage_robust,
        m.p_first_stage_robust,
        df1,
        df2,
        length(endogenous_names),
        size(fsd.Z_res, 2),
        vcov_name
    )
end

##############################################################################
##
## Show method for FirstStageResult
##
##############################################################################

function Base.show(io::IO, fs::FirstStageResult{T}) where {T}
    # Calculate dynamic width based on longest endogenous name
    max_name_len = maximum(length, fs.endogenous_names; init = 10)
    totwidth = max(60, max_name_len + 40)

    # Header
    println(io, "First-Stage Diagnostics ($(fs.vcov_type))")
    println_horizontal_line(io, totwidth)

    # Per-endogenous table
    println(io, "Per-Endogenous F-Statistics:")
    println_horizontal_line(io, totwidth)
    @printf(io, "% -30s\n", "Endogenous")
    println_horizontal_line(io, totwidth)

    for (j, name) in enumerate(fs.endogenous_names)
        display_name = length(name) > 28 ? name[1:25] * "..." : name
        @printf(io, "% -30s\n", display_name)
        @printf(io, "  %-22s %14.4f %14.4f\n",
            "F[nonrobust] (P-value)", fs.F_nonrobust[j], fs.p_nonrobust[j])
        @printf(io, "  %-22s %14.4f %14.4f\n",
            "F[$(fs.vcov_type)] (P-value)", fs.F_robust[j], fs.p_robust[j])
    end

    println_horizontal_line(io, totwidth)
    println(io, "\nDoF: ($(fs.df1), $(fs.df2))")
    println(io, "Instruments: $(fs.n_instruments) excluded, $(fs.n_endogenous) endogenous")
    print_horizontal_line(io, totwidth)
end

function Base.show(io::IO, ::MIME"text/html", fs::FirstStageResult{T}) where {T}
    html_table_start(io; class = "regress-table regress-first-stage",
        caption = "First-Stage Diagnostics ($(fs.vcov_type))")

    # Per-endogenous section
    html_thead_start(io; class = "regress-per-endo-header")
    html_row(
        io, ["Endogenous", "F(nonrobust)", "P-value", "F($(fs.vcov_type))", "P-value"];
        is_header = true)
    html_thead_end(io)
    html_tbody_start(io; class = "regress-per-endo-body")
    for (j, name) in enumerate(fs.endogenous_names)
        html_row(io,
            [
                name,
                format_number(fs.F_nonrobust[j]),
                format_pvalue(fs.p_nonrobust[j]),
                format_number(fs.F_robust[j]),
                format_pvalue(fs.p_robust[j])])
    end
    html_tbody_end(io)

    # Footer
    html_tfoot_start(io; class = "regress-footer")
    html_row(io,
        [
            "DoF: ($(fs.df1), $(fs.df2)); Instruments: $(fs.n_instruments) excluded, $(fs.n_endogenous) endogenous",
            "", "", "", ""])
    html_tfoot_end(io)

    html_table_end(io)
end

##############################################################################
##
## Schema
##
##############################################################################
function StatsModels.apply_schema(t::FormulaTerm, schema::StatsModels.Schema,
        Mod::Type{<:IVEstimator}, has_fe_intercept)
    schema = StatsModels.FullRank(schema)
    if has_fe_intercept
        push!(schema.already, InterceptTerm{true}())
    end
    FormulaTerm(apply_schema(t.lhs, schema.schema, StatisticalModel),
        StatsModels.collect_matrix_terms(apply_schema(t.rhs, schema, StatisticalModel)))
end

##############################################################################
##
## Display Result
##
##############################################################################

_estimator_name(m::IVEstimator) = _estimator_name(m.estimator)

function _estimator_name(e::AbstractIVEstimator)
    if e isa TSLS
        return "TSLS"
    elseif e isa LIML
        return "LIML"
    elseif e isa Fuller
        return @sprintf("Fuller(%.1f)", e.a)
    elseif e isa KClass
        return @sprintf("KClass(%.4f)", e.kappa)
    else
        return string(typeof(e).name.name)
    end
end

function top(m::IVEstimator)
    # Use shared summary
    out_common = _summary_table_common(m) # Matrix

    # Add IV specific fields

    # Add Converged status (IV also uses FE solver logic for convergence if FE present)
    # Insert at index 2
    row_converged = ["Converged" m.converged]

    # Split
    part1 = out_common[1:1, :]
    part2 = out_common[2:end, :]
    out = vcat(part1, row_converged, part2)

    # Show kappa for K-class estimators
    if !isnothing(m.postestimation) && !isnan(m.postestimation.kappa)
        out = vcat(out, ["K-class kappa" @sprintf("%.4f", m.postestimation.kappa)])
    end

    # Always show first-stage diagnostics for IV models
    fs_nonrobust = isempty(m.F_first_stage_nonrobust) ? eltype(m.coef)(NaN) :
                   minimum(m.F_first_stage_nonrobust)
    fs_robust = isempty(m.F_first_stage_robust) ? eltype(m.coef)(NaN) :
                minimum(m.F_first_stage_robust)
    out = vcat(out,
        ["F (1st stage, nonrobust)" sprint(show, fs_nonrobust, context = :compact => true);
         "F (1st stage, $(vcov_type_name(m.vcov_estimator)))" sprint(show, fs_robust, context = :compact => true);])

    # Add Iterations if FE
    if has_fe(m)
        out = vcat(out,
            ["Iterations" sprint(show, m.iterations, context = :compact => true);])
    end
    return out
end

import StatsBase: NoQuote, PValue

# Custom type for test statistics (matches OLS)
struct TestStat
    val::Float64
end
Base.show(io::IO, x::TestStat) = isnan(x.val) ? print(io, ".") : @printf(io, "%.4f", x.val)
Base.alignment(io::IO, x::TestStat) = (0, length(sprint(show, x, context = io)))

function Base.show(io::IO, m::IVEstimator)
    ct = coeftable(m)
    cols = ct.cols
    rownms = ct.rownms
    colnms = ct.colnms
    nc = length(cols)
    nr = length(cols[1])
    if length(rownms) == 0
        rownms = [lpad("[$i]", floor(Integer, log10(nr)) + 3) for i in 1:nr]
    end
    mat = [j == 1 ? NoQuote(rownms[i]) :
           j - 1 == ct.pvalcol ? NoQuote(sprint(show, PValue(cols[j - 1][i]))) :
           j - 1 in ct.teststatcol ? TestStat(cols[j - 1][i]) :
           cols[j - 1][i] isa AbstractString ? NoQuote(cols[j - 1][i]) : cols[j - 1][i]
           for i in 1:nr, j in 1:(nc + 1)]
    io = IOContext(io, :compact => true, :limit => false)
    A = Base.alignment(io, mat, 1:size(mat, 1), 1:size(mat, 2),
        typemax(Int), typemax(Int), 3)
    nmswidths = pushfirst!(length.(colnms), 0)
    A = [nmswidths[i] > sum(A[i]) ? (A[i][1] + nmswidths[i] - sum(A[i]), A[i][2]) : A[i]
         for i in 1:length(A)]
    totwidth = compute_table_width(A, colnms)

    # Title: estimator name, right-aligned, yellow
    ctitle = _estimator_name(m)
    if supports_color(io)
        print(io,
            lpad(ANSI_YELLOW * ctitle * ANSI_RESET,
                totwidth - 2 + length(ANSI_YELLOW) + length(ANSI_RESET)))
    else
        print(io, lpad(ctitle, totwidth - 2))
    end
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
        print(io, " "^interwidth)
        if size(ctop, 1) >= 2 * i
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
    vcov_name = vcov_type_name(m.vcov_estimator, vcov(m))
    println(io, "Note: Std. errors computed using $vcov_name variance estimator")
    nothing
end

function Base.show(io::IO, ::MIME"text/html", m::IVEstimator)
    ct = coeftable(m)
    cols = ct.cols
    rownms = ct.rownms
    colnms = ct.colnms

    # Start table with estimator name as caption
    ctitle = _estimator_name(m)
    html_table_start(io; class = "regress-table regress-iv", caption = ctitle)

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
    vcov_name = vcov_type_name(m.vcov_estimator, vcov(m))
    html_tfoot_start(io; class = "regress-footer")
    html_row(io, ["Note: Std. errors computed using $vcov_name variance estimator",
        "", "", "", "", "", ""])
    html_tfoot_end(io)

    html_table_end(io)
end

# Predict and Residuals
# Note: is_cont_fe_int() and has_cont_fe_interaction() are defined in utils/fit_common.jl

function StatsAPI.predict(m::IVEstimator, data)
    Tables.istable(data) ||
        throw(ArgumentError("expected second argument to be a Table, got $(typeof(data))"))

    has_cont_fe_interaction(m.formula) &&
        throw(ArgumentError("Interaction of fixed effect and continuous variable detected in formula; this is currently not supported in `predict`"))

    cdata = Tables.columntable(data)
    cols, nonmissings = StatsModels.missing_omit(cdata, m.formula_schema.rhs)
    Xnew = modelmatrix(m.formula_schema, cols)

    # Type-stable inner function via function barrier
    T = eltype(m.coef)
    return _predict_iv_impl(Xnew, m.coef, nonmissings, m, data, T)
end

# Type-stable inner function for IV predict
function _predict_iv_impl(
        Xnew::AbstractMatrix, coef_vec::AbstractVector{T},
        nonmissings::AbstractVector{Bool}, m::IVEstimator, data, ::Type{T}
) where {T}
    n = length(nonmissings)
    # Always allocate with Union type for consistent return type
    out = Vector{Union{T, Missing}}(missing, n)
    @views out[nonmissings] .= Xnew * coef_vec

    if has_fe(m)
        nrow(fe(m)) > 0 ||
            throw(ArgumentError("Model has no estimated fixed effects. To store estimates of fixed effects, run `iv` with the option save = :fe"))

        df = DataFrame(data; copycols = false)
        fes = leftjoin(select(df, m.fekeys), dropmissing(unique(m.fe)); on = m.fekeys,
            makeunique = true, matchmissing = :equal, order = :left)
        fes = combine(fes, AsTable(Not(m.fekeys)) => sum)

        @views out[nonmissings] .+= fes[nonmissings, 1]
    end

    return out
end

function StatsAPI.residuals(m::IVEstimator, data)
    Tables.istable(data) ||
        throw(ArgumentError("expected second argument to be a Table, got $(typeof(data))"))
    has_fe(m) &&
        throw(ArgumentError("To access residuals for a model with high-dimensional fixed effects,  run `m = iv(..., save = :residuals)` and then access residuals with `residuals(m)`."))
    cdata = Tables.columntable(data)
    cols, nonmissings = StatsModels.missing_omit(cdata, m.formula_schema.rhs)
    Xnew = modelmatrix(m.formula_schema, cols)
    y = response(m.formula_schema, cdata)

    # Type-stable inner function via function barrier
    T = eltype(m.coef)
    return _residuals_iv_impl(y, Xnew, m.coef, nonmissings, T)
end

# Type-stable inner function for IV residuals
function _residuals_iv_impl(
        y::AbstractVector, Xnew::AbstractMatrix, coef_vec::AbstractVector{T},
        nonmissings::AbstractVector{Bool}, ::Type{T}
) where {T}
    n = length(nonmissings)
    # Always allocate with Union type for consistent return type
    out = Vector{Union{T, Missing}}(missing, n)
    @views out[nonmissings] .= y .- Xnew * coef_vec
    return out
end

function StatsAPI.residuals(m::IVEstimator{T}) where {T}
    if !has_residuals_data(m)
        has_fe(m) &&
            throw(ArgumentError("To access residuals in a fixed effect regression, run `iv` with the option save = :residuals, and then access residuals with `residuals()`"))
        !has_fe(m) &&
            throw(ArgumentError("To access residuals, use residuals(m, data) where `m` is an estimated IVEstimator and `data` is a Table"))
    end
    # Reconstruct full-length residuals with missings for non-esample rows
    n = length(m.esample)
    out = Vector{Union{T, Missing}}(missing, n)
    out[m.esample] .= m.residuals_esample
    return out
end

"""
   fe(m::IVEstimator; keepkeys = false)

Return a DataFrame with fixed effects estimates.
"""
function fe(m::IVEstimator; keepkeys = false)
    !has_fe(m) &&
        throw(ArgumentError("fe() is not defined for models without fixed effects"))
    if keepkeys
        m.fe
    else
        m.fe[!, (length(m.fekeys) + 1):end]
    end
end

##############################################################################
##
## IVMatrixEstimator - Matrix-based IV estimation (like OLSMatrixEstimator)
##
##############################################################################

"""
    PostEstimationDataIVMatrix{T}

Stores data needed for post-estimation variance computation in matrix-based IV. The
matrices hold the estimation sample, demeaned on the absorbed fixed effects and scaled by
`√w` for a weighted fit; their columns are the estimated coefficients (collinear columns
dropped), in the order of the columns of `X`.

# Fields
- `X::Matrix{T}`: Regressor matrix
- `Z::Matrix{T}`: Instrument matrix: the exogenous columns of `X`, in order, then the
  excluded instruments
- `X_hat::Matrix{T}`: Projected regressors P_Z * X (the adjustment matrix `Adj` for the
  k-class estimators)
- `y::Vector{T}`: Response vector
- `residuals::Vector{T}`: Residuals e = y - X*β
- `invXhatXhat::Matrix{T}`: (X̂'X̂)⁻¹ for sandwich formula (`inv(A)` for k-class)
- `endo_cols::Vector{Int}`: Columns of `X` that are endogenous
- `weights::Vector{T}`: Observation weights (empty for an unweighted fit)
"""
struct PostEstimationDataIVMatrix{T <: AbstractFloat}
    X::Matrix{T}
    Z::Matrix{T}
    X_hat::Matrix{T}
    y::Vector{T}
    residuals::Vector{T}
    invXhatXhat::Matrix{T}
    endo_cols::Vector{Int}
    weights::Vector{T}
end

# The exogenous columns of `X` and the number of them, which lead the columns of `Z`.
_exo_cols(pe::PostEstimationDataIVMatrix) = setdiff(axes(pe.X, 2), pe.endo_cols)

"""
    IVMatrixEstimator{T, E, V, C} <: AbstractRegressModel

Matrix-based IV estimator for use without formula interface.
Designed for programmatic use (e.g., LocalProjections.jl).

# Type Parameters
- `T`: Element type (Float64 or Float32)
- `E`: IV estimator type (TSLS, LIML, Fuller, KClass)
- `V`: Variance estimator type

# Fields
- `estimator::E`: IV estimator used to fit the model
- `coef::Vector{T}`: Coefficient estimates
- `postestimation::PostEstimationDataIVMatrix{T}`: Data for vcov computation
- `basis_coef::BitVector`: Which coefficients are not collinear
- `nobs::Int`: Number of observations
- `dof::Int`: Number of estimated parameters
- `dof_residual::Int`: Residual degrees of freedom
- `rss::T`: Residual sum of squares
- `tss::T`: Total sum of squares
- `r2::T`: R-squared
- `has_intercept::Bool`: Whether model includes intercept
- `coefnames::Vector{String}`: Coefficient names, one per column of `X`
- `fe_groups::Vector{Vector{Int}}`: Group codes of each absorbed fixed effect (empty
  without fixed effects)
- `fe_ngroups::Vector{Int}`: Number of levels of each absorbed fixed effect
- `esample::BitVector`: Which rows of the input are in the estimation sample
- `vcov_estimator::V`: Variance estimator used
- `vcov_matrix::C`: Precomputed variance-covariance matrix
- `se::Vector{T}`: Standard errors
- `t_stats::Vector{T}`: t-statistics
- `p_values::Vector{T}`: p-values

# Example
```julia
# Z = [exogenous, instruments], X = [exogenous, endogenous]
model = iv(TSLS(), Z, X, y; has_intercept=false, n_endogenous=1)
coef(model)
vcov(HC1(), model)
```
"""
struct IVMatrixEstimator{T <: AbstractFloat, E <: AbstractIVEstimator, V,
    C <: AbstractMatrix{T}} <:
       AbstractRegressModel
    estimator::E
    coef::Vector{T}
    postestimation::PostEstimationDataIVMatrix{T}
    basis_coef::BitVector
    nobs::Int
    dof::Int
    dof_residual::Int
    rss::T
    tss::T
    r2::T
    has_intercept::Bool
    coefnames::Vector{String}
    fe_groups::Vector{Vector{Int}}
    fe_ngroups::Vector{Int}
    esample::BitVector

    # Variance-covariance
    vcov_estimator::V
    vcov_matrix::C
    se::Vector{T}
    t_stats::Vector{T}
    p_values::Vector{T}
end

_estimator_name(m::IVMatrixEstimator) = _estimator_name(m.estimator)

has_iv(::IVMatrixEstimator) = true
has_fe(m::IVMatrixEstimator) = !isempty(m.fe_groups)
dof_fes(m::IVMatrixEstimator) = _fe_dof(m.fe_ngroups, length(m.fe_ngroups))
model_hasintercept(m::IVMatrixEstimator) = m.has_intercept

##############################################################################
## StatsAPI Interface for IVMatrixEstimator
##############################################################################

function StatsAPI.coef(m::IVMatrixEstimator)
    beta = copy(m.coef)
    beta[.!m.basis_coef] .= zero(eltype(beta))
    return beta
end
StatsAPI.coefnames(m::IVMatrixEstimator) = m.coefnames

StatsAPI.nobs(m::IVMatrixEstimator) = m.nobs
StatsAPI.dof(m::IVMatrixEstimator) = m.dof
StatsAPI.dof_residual(m::IVMatrixEstimator) = m.dof_residual
StatsAPI.rss(m::IVMatrixEstimator) = m.rss
StatsAPI.deviance(m::IVMatrixEstimator) = m.rss
StatsAPI.nulldeviance(m::IVMatrixEstimator) = m.tss
StatsAPI.r2(m::IVMatrixEstimator) = m.r2

function StatsAPI.response(m::IVMatrixEstimator)
    return m.postestimation.y
end

function StatsAPI.residuals(m::IVMatrixEstimator)
    pe = m.postestimation
    return isempty(pe.weights) ? pe.residuals : pe.residuals ./ sqrt.(pe.weights)
end

function StatsAPI.fitted(m::IVMatrixEstimator)
    return m.postestimation.y .- m.postestimation.residuals
end

function StatsAPI.modelmatrix(m::IVMatrixEstimator)
    return m.postestimation.X
end

function StatsAPI.vcov(m::IVMatrixEstimator)
    return m.vcov_matrix
end

function StatsAPI.stderror(m::IVMatrixEstimator)
    return m.se
end

##############################################################################
## CovarianceMatrices Interface for IVMatrixEstimator
##############################################################################

"""
    CovarianceMatrices.bread(m::IVMatrixEstimator)

Returns (X̂'X̂)⁻¹ for sandwich variance estimation.
"""
_CM.bread(m::IVMatrixEstimator) = m.postestimation.invXhatXhat

_CM.leverage(m::IVMatrixEstimator) = StatsAPI.leverage(m)

"""
    momentmatrix(m::IVMatrixEstimator)

Returns X̂ * diag(e) for moment-based variance estimation.
For IV, we use the projected regressors X̂ = P_Z * X.
"""
function CovarianceMatrices.momentmatrix(m::IVMatrixEstimator)
    X_hat = m.postestimation.X_hat
    resid = m.postestimation.residuals
    return X_hat .* resid
end

"""
    leverage(m::IVMatrixEstimator)

Diagonal of the IV hat matrix, for HC2/HC3/HC4/HC5.

For k-class estimators the `X_hat`/`invXhatXhat` slots hold the k-class
adjustment matrix and its bread, so `diag(Adj·invA·Adj')` is the leverage. TSLS
is specialized below to the AER formula, which for over-identified models does
not coincide with `diag(X̂(X̂'X̂)⁻¹X̂')`.
"""
function StatsAPI.leverage(m::IVMatrixEstimator)
    X_hat = m.postestimation.X_hat
    invXX = m.postestimation.invXhatXhat
    # h_ii = X̂_i' * (X̂'X̂)⁻¹ * X̂_i
    # Efficient computation: sum((X_hat * invXX) .* X_hat, dims=2)
    return vec(sum((X_hat * invXX) .* X_hat, dims = 2))
end

# TSLS leverage matches R's AER::ivreg / sandwich::vcovHC:
#     h = diag(X · (X̂'X̂)⁻¹ · X' · Z · (Z'Z)⁻¹ · Z')
# This is the same formula the formula-path `leverage(::IVEstimator)` uses. For
# over-identified TSLS it differs from diag(X̂(X̂'X̂)⁻¹X̂'), so HC2/HC3 need it to
# match the formula path.
function StatsAPI.leverage(m::IVMatrixEstimator{T, TSLS}) where {T <: AbstractFloat}
    pe = m.postestimation
    return _tsls_leverage(pe.X, pe.X_hat, pe.invXhatXhat)
end

# CovarianceMatrices.jl uses numobs, which is distinct from StatsAPI.nobs
_CM.numobs(m::IVMatrixEstimator) = m.nobs

_CM.mask(m::IVMatrixEstimator) = m.basis_coef

# Residual adjustments for HC estimators come from
# `CovarianceMatrices.residual_adjustment`, reaching the model through the `leverage`
# and `numobs` protocol methods defined above.
@noinline function residualadjustment(k::_CM.HR, m::IVMatrixEstimator{T}) where {T}
    return _CM.residual_adjustment(k, m)
end

# HR1 is the exception: upstream divides by `n - length(coef(m))`, which cannot see
# fixed effects absorbed out of the design, while `dof_residual` counts them.
@noinline function residualadjustment(k::_CM.HR1, m::IVMatrixEstimator{T}) where {T}
    return sqrt(T(nobs(m)) / T(dof_residual(m)))
end

# Correlated estimators (HAC, EWC, DriscollKraay, VARHAC, etc.) - no adjustment needed
@noinline residualadjustment(k::_CM.Correlated, m::IVMatrixEstimator) = 1.0

# CR (cluster-robust) - no individual residual adjustment needed (handled in aVar)
@noinline residualadjustment(k::_CM.CR0, m::IVMatrixEstimator) = 1.0
@noinline residualadjustment(k::_CM.CR1, m::IVMatrixEstimator) = 1.0

# `X_hat` holds the second-stage regressors (`Adj` for the K-class estimators), as on
# the formula fit, so CR2/CR3 use the same leverage adjustment.
function _regress_aVar(k::_CM.CR, m::IVMatrixEstimator; demean = false, prewhite = false,
        scale = true)
    X_hat = m.postestimation.X_hat
    u = m.postestimation.residuals
    if k isa Union{_CM.CR2, _CM.CR3}
        return _leverage_aVar(k, X_hat, X_hat, bread(m), u, m.postestimation.weights;
            demean, prewhite, scale)
    end
    return _moment_aVar(k, X_hat .* u; demean, prewhite, scale)
end

function _regress_aVar(k, m::IVMatrixEstimator; demean = false, prewhite = false,
        scale = true)
    # Compute adjusted moment matrix: X̂ .* (u .* e)
    # where u is the residual adjustment factor
    X_hat = m.postestimation.X_hat
    resid = m.postestimation.residuals

    M = X_hat .* resid
    # The adjustment can be Float64 on a Float32 model; the moments keep the model's type.
    M = M .* eltype(M).(residualadjustment(k, m))

    # Bandwidth-selection weights come from the regressor matrix, not the moment
    # matrix: they must give the intercept weight 0, and the intercept column of
    # the moment matrix is not constant. `nothing` for non-HAC estimators.
    kw = _CM.bandwidthweights(k, m.postestimation.X)

    return _moment_aVar(k, M; demean, prewhite, scale, weights = kw)
end

function _regress_vcov(ve, m::IVMatrixEstimator{T}; dofadjust = true, demean = false,
        prewhite = false) where {T}
    n = nobs(m)
    B = bread(m)

    A = _regress_aVar(ve, m; demean, prewhite)
    scale = T(n * _finite_sample_factor(ve, m, dofadjust))

    Σ = scale .* B * A * B
    return _wrap_vcov(Symmetric(_expand_collinear(Σ, m.basis_coef)), ve, A)
end

# Driscoll-Kraay: see the `_regress_vcov` methods for OLS models.
function _regress_vcov(k::_CM.DriscollKraay, m::IVMatrixEstimator; kwargs...)
    Σ = invoke(
        StatsBase.vcov, Tuple{_CM.DriscollKraay, RegressionModel}, k, m; kwargs...)
    return _wrap_vcov(parent(Σ), k, Σ)
end

##############################################################################
## VcovSpec + operator for IVMatrixEstimator
##############################################################################

"""
    m::IVMatrixEstimator + v::VcovSpec

Create a new IVMatrixEstimator with updated variance-covariance estimator.
"""
function Base.:+(m::IVMatrixEstimator{T, E, V1}, v::VcovSpec{V2}) where {T, E, V1, V2}
    new_vcov = vcov(v.source, m)
    new_se = sqrt.(diag(new_vcov))

    # Recompute t-stats and p-values
    cc = coef(m)
    new_t = cc ./ new_se
    new_p = 2 .* tdistccdf.(dof_residual(m), abs.(new_t))

    return IVMatrixEstimator{T, E, V2, typeof(new_vcov)}(
        m.estimator,
        m.coef,
        m.postestimation,
        m.basis_coef,
        m.nobs,
        m.dof,
        m.dof_residual,
        m.rss,
        m.tss,
        m.r2,
        m.has_intercept,
        m.coefnames,
        m.fe_groups,
        m.fe_ngroups,
        m.esample,
        v.source,
        new_vcov,
        new_se,
        new_t,
        new_p
    )
end

##############################################################################
## first_stage for IVMatrixEstimator
##############################################################################

# One OLS fit of each column of `Xendo` on `Z`.
function _matrix_first_stage_fits(Z, Xendo, has_intercept, fe, coefnames)
    return [ols(Z, Xendo[:, i]; has_intercept, fe, coefnames) for i in axes(Xendo, 2)]
end

"""
    first_stage(m::IVMatrixEstimator; endogenous_names=nothing, instrument_names=nothing) -> FirstStageIV

Compute first-stage OLS regressions and diagnostics for an IV matrix estimator.

Returns a `FirstStageIV` containing:
- Full OLS models (one per endogenous variable) with `coef`, `stderror`, `fitted`, `residuals`, `r2`
- Non-robust F-statistic per endogenous variable
- Robust F-statistic per endogenous variable using the model's current `vcov`

The first-stage models are named `[exogenous..., instruments...]`, with the exogenous
names taken from `coefnames(m)`.

# Keyword Arguments
- `endogenous_names`: Names for endogenous variables (default: the last `n_endogenous`
  entries of `coefnames(m)`)
- `instrument_names`: Names for excluded instruments (default: `["inst_1", ...]`)

# Example
```julia
m = iv(TSLS(), Z, X, y; n_endogenous=1)
fs = first_stage(m)
coef(fs.models[1])       # first-stage coefficients
stderror(fs.models[1])   # standard errors
fitted(fs.models[1])     # predicted endogenous
fs.F_nonrobust           # non-robust F
fs.F_robust              # robust F
```
"""
function first_stage(m::IVMatrixEstimator{T};
        endogenous_names::Union{Vector{String}, Nothing} = nothing,
        instrument_names::Union{Vector{String}, Nothing} = nothing) where {T}
    pe = m.postestimation
    n_endo = length(pe.endo_cols)
    n_endo > 0 || throw(ArgumentError("Model has no endogenous variables."))

    exo_cols = _exo_cols(pe)
    X = pe.X
    Z = pe.Z
    S = nobs(m)
    k_exo = length(exo_cols)
    K = size(Z, 2) - k_exo  # number of excluded instruments

    Xendo = X[:, pe.endo_cols]
    Z_instr = Z[:, (k_exo + 1):end]

    # Default names
    estimated_names = coefnames(m)[m.basis_coef]
    endo_names = endogenous_names === nothing ?
                 estimated_names[pe.endo_cols] : endogenous_names
    instr_names = instrument_names === nothing ?
                  ["inst_$i" for i in 1:K] : instrument_names
    length(endo_names) == n_endo || throw(DimensionMismatch(
        "endogenous_names has $(length(endo_names)) names but the model has $n_endo endogenous variables"))
    length(instr_names) == K || throw(DimensionMismatch(
        "instrument_names has $(length(instr_names)) names but the model has $K excluded instruments"))
    fs_names = vcat(estimated_names[exo_cols], instr_names)

    # Step 1: Run first-stage OLS regressions (xendo_j on full Z)
    # On a model with absorbed fixed effects, X and Z are already demeaned; passing the
    # groups again makes the first-stage fits count the absorbed levels.
    fs_models = _matrix_first_stage_fits(Z, Xendo, m.has_intercept,
        has_fe(m) ? m.fe_groups : nothing, fs_names)

    # Step 2: Compute residualized data for F-statistics
    # Partial out exogenous regressors from excluded instruments and endogenous
    if k_exo > 0
        Xexo = X[:, exo_cols]
        QR_exo = qr(Xexo)
        Q_exo = Matrix(QR_exo.Q)
        Z_res = Z_instr .- Q_exo * (Q_exo' * Z_instr)
        Xendo_demeaned = Xendo .- Q_exo * (Q_exo' * Xendo)
    else
        Z_res = copy(Z_instr)
        Xendo_demeaned = copy(Xendo)
    end

    # Step 3: Extract Pi (first-stage coefficients for excluded instruments only)
    # Full coef = [coef_exo; coef_instr], we need coef_instr (K × n_endo)
    Pi = Matrix{T}(undef, K, n_endo)
    Xendo_res = Matrix{T}(undef, S, n_endo)
    for i in 1:n_endo
        Pi[:, i] = coef(fs_models[i])[(k_exo + 1):end]
        Xendo_res[:, i] = residuals(fs_models[i])
    end

    # Step 4: Non-robust F per endogenous variable
    # F_nonrobust = S * π̂'π̂ / (K * ω₂₂)
    # where π̂ are reduced-form coefficients on orthogonalized instruments
    QR_Z = qr(Z_res)
    Zs = Matrix(QR_Z.Q)[:, 1:K] .* sqrt(T(S))
    L = k_exo
    dof_omega = S - K - L - dof_fes(m)

    F_nonrobust = Vector{T}(undef, n_endo)
    p_nonrobust = Vector{T}(undef, n_endo)
    for i in 1:n_endo
        pihat = (Zs' * Xendo_demeaned[:, i]) ./ T(S)
        res_i = Xendo_demeaned[:, i] .- Zs * pihat
        omega_22 = dot(res_i, res_i) / dof_omega
        pipi = dot(pihat, pihat)
        F_nonrobust[i] = S * pipi / (K * omega_22)
        p_nonrobust[i] = fdistccdf(K, dof_omega, F_nonrobust[i])
    end

    # Step 5: Robust F per endogenous variable using the model's current vcov
    F_robust,
    p_robust = compute_per_endogenous_fstats(
        Xendo_res, Z_res, Pi,
        m.vcov_estimator, dof_fes(m);
        Xendo_orig = Xendo,
        newZ = Z,
        first_stage = _MatrixFirstStage(m, k_exo + K)
    )

    return FirstStageIV{T, eltype(fs_models)}(
        fs_models, endo_names, instr_names,
        F_nonrobust, p_nonrobust, F_robust, p_robust,
        K, dof_omega, string(typeof(m.vcov_estimator).name.name)
    )
end

##############################################################################
## Show methods for IVMatrixEstimator
##############################################################################

function Base.show(io::IO, m::IVMatrixEstimator)
    print(io, "IVMatrixEstimator(n=$(nobs(m)), k=$(dof(m)))")
end

function Base.show(io::IO, ::MIME"text/plain", m::IVMatrixEstimator{T}) where {T}
    println(io, "IV Matrix Estimator ($(_estimator_name(m)))")
    println(io, "─" ^ 40)
    println(io, "Observations:      $(nobs(m))")
    println(io, "Parameters:        $(dof(m))")
    println(io, "R²:                $(round(r2(m), digits=4))")
    println(io, "Residual DoF:      $(dof_residual(m))")
    println(io, "Endogenous vars:   $(length(m.postestimation.endo_cols))")
    println(io)
    println(io, "Coefficients:")
    cc = coef(m)
    se = m.se
    for (name, c, s) in zip(coefnames(m), cc, se)
        println(io, "  $name = $(round(c, digits=6)) (SE: $(round(s, digits=6)))")
    end
end
