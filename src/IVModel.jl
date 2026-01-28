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
- `F_joint::T`: Joint first-stage F-statistic (Kleibergen-Paap)
- `p_joint::T`: p-value of joint F-statistic
- `endogenous_names::Vector{String}`: Names of endogenous variables
- `F_per_endo::Vector{T}`: F-statistic for each endogenous variable
- `p_per_endo::Vector{T}`: p-value for each endogenous variable
- `n_endogenous::Int`: Number of endogenous variables
- `n_instruments::Int`: Number of excluded instruments
- `vcov_type::String`: Variance estimator used (e.g., "HC1", "HC3")

# Examples
```julia
model = iv(TSLS(), df, @formula(y ~ x + (endo ~ z1 + z2)))
fs = first_stage(model)
fs.F_joint           # Joint F-statistic
fs.F_per_endo        # Per-endogenous F-statistics
```
"""
struct FirstStageResult{T <: AbstractFloat}
    F_joint::T
    p_joint::T
    endogenous_names::Vector{String}
    F_per_endo::Vector{T}
    p_per_endo::Vector{T}
    n_endogenous::Int
    n_instruments::Int
    vcov_type::String
end

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
- `X::Matrix{T}`: Design matrix used for inference (with predicted endogenous for TSLS)
- `Xhat::Matrix{T}`: Original matrix with actual endogenous variables
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
struct PostEstimationDataIV{T, W <: AbstractWeights}
    X::Matrix{T}                # X̂ = fitted values from first stage (confusing name, kept for compatibility)
    Xhat::Matrix{T}             # Original X regressors (confusing name, kept for compatibility)
    crossx::Cholesky{T, Matrix{T}}
    invXX::Symmetric{T, Matrix{T}}  # (X̂'X̂)^{-1}
    weights::W
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
    T, E <: AbstractIVEstimator, V, P <: Union{PostEstimationDataIV{T}, Nothing}} <:
       AbstractRegressModel
    estimator::E  # Which IV estimator was used

    coef::Vector{T}   # Vector of coefficients

    esample::BitVector      # Is the row of the original dataframe part of the estimation sample?
    residuals_esample::Vector{T}  # Residuals for esample rows only (for vcov computation)
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
    vcov_estimator::V                        # Deep copy of the estimator
    vcov_matrix::Symmetric{T, Matrix{T}}    # Precomputed vcov matrix
    se::Vector{T}                            # Standard errors
    t_stats::Vector{T}                       # t-statistics
    p_values::Vector{T}                      # p-values

    # Test statistics (computed with vcov_estimator)
    F::T                    # F-statistic (Wald test)
    p::T                    # P-value of F-stat
    F_kp::T                 # Kleibergen-Paap first-stage F-stat (joint)
    p_kp::T                 # P-value of first-stage F-stat (joint)
    F_kp_per_endo::Vector{T}  # Per-endogenous first-stage F-statistics
    p_kp_per_endo::Vector{T}  # Per-endogenous first-stage p-values
end

has_iv(::IVEstimator) = true
has_fe(m::IVEstimator) = has_fe(m.formula)
r2_within(m::IVEstimator) = m.r2_within
model_hasintercept(m::IVEstimator) = hasintercept(m.formula)

"""
    has_residuals_data(m::IVEstimator) -> Bool

Check if residuals were saved during fitting.
"""
has_residuals_data(m::IVEstimator) = m.has_residuals

"""
    residuals_for_vcov(m::IVEstimator) -> Vector{T}

Get residuals for variance-covariance computation (esample rows only).
This is the type-stable internal accessor for vcov calculations.
"""
residuals_for_vcov(m::IVEstimator) = m.residuals_esample

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
        error("Model does not have post-estimation data stored. Post-estimation vcov not available.")
    !has_residuals_data(m) &&
        error("Model does not have residuals stored. Use save=:residuals or save=:all when fitting.")

    # Use Adj if available (K-class: LIML, Fuller), otherwise use X (TSLS)
    pe = m.postestimation
    X_for_vcov = has_kclass_adj(pe) ? pe.Adj : pe.X
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
    bread(m::IVEstimator)

Compute (X'X)^(-1), the "bread" of the sandwich variance estimator for IV.
Uses the predicted endogenous variables (Xhat) in the design matrix.
"""
bread(m::IVEstimator) = m.postestimation.invXX

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
    isnothing(m.postestimation) && error("Model does not have post-estimation data stored.")
    pe = m.postestimation

    # For K-class estimators, use the Adj matrix if available
    if has_kclass_adj(pe)
        # K-class uses simple leverage: h = diag(Adj * invXX * Adj')
        Adj = pe.Adj
        invXX = pe.invXX
        return vec(sum(Adj .* (Adj * invXX), dims = 2))
    end

    # TSLS uses AER formula: h = diag(X * invXhatXhat * X' * Z * invZZ * Z')
    X_orig = pe.Xhat      # Original regressors (confusing field name)
    invXhatXhat = pe.invXX
    Z = pe.Z
    invZZ = pe.invZZ

    # Compute A = X * invXhatXhat * X' (this is n x n, but we only need for diag computation)
    # Compute P_Z = Z * invZZ * Z' (projection onto instruments)
    # h = diag(A * P_Z)
    #
    # Efficient computation: h_i = (X_i * invXhatXhat * X') * (Z * invZZ * Z'_i)
    # = sum_j (X_i * invXhatXhat * X_j') * (Z_j * invZZ * Z_i')
    #
    # More efficient: compute A_row = X_i * invXhatXhat * X' for each i
    # then h_i = A_row * P_Z[:, i] = A_row * Z * invZZ * Z_i'
    #
    # Best approach: compute full matrices and take diagonal
    # A = X * invXhatXhat * X'
    # B = Z * invZZ * Z'
    # h = diag(A * B)
    #
    # For memory efficiency, compute row by row:
    n = size(X_orig, 1)
    h = Vector{eltype(X_orig)}(undef, n)
    XinvXX = X_orig * invXhatXhat  # n x k
    ZinvZZ = Z * invZZ              # n x q

    @inbounds for i in 1:n
        # A_row_i = X_i * invXhatXhat * X' = XinvXX[i, :] * X'
        # h_i = A_row_i * Z * invZZ * Z_i' = sum_j (XinvXX[i, :] ⋅ X[j, :]) * (ZinvZZ[j, :] ⋅ Z[i, :])
        hi = zero(eltype(X_orig))
        for j in 1:n
            # (X_i * invXhatXhat) ⋅ X_j
            a_ij = zero(eltype(X_orig))
            for k in 1:size(XinvXX, 2)
                a_ij += XinvXX[i, k] * X_orig[j, k]
            end
            # (Z * invZZ)_j ⋅ Z_i
            b_ji = zero(eltype(Z))
            for q in 1:size(ZinvZZ, 2)
                b_ji += ZinvZZ[j, q] * Z[i, q]
            end
            hi += a_ij * b_ji
        end
        h[i] = hi
    end
    return h
end

# Residual adjustments for HC/HR estimators
# Note: HC0 = HR0 and HC1 = HR1 in CovarianceMatrices.jl (type aliases)
@noinline residualadjustment(k::_CM.HR0, m::IVEstimator) = 1.0  # Also handles HC0
@noinline residualadjustment(k::_CM.HR1, m::IVEstimator) = sqrt(nobs(m) / dof_residual(m))  # Also handles HC1
@noinline residualadjustment(k::_CM.HR2, m::IVEstimator) = 1.0 ./ sqrt.(1 .- leverage(m))  # Also handles HC2
@noinline residualadjustment(k::_CM.HR3, m::IVEstimator) = 1.0 ./ (1 .- leverage(m))  # Also handles HC3

@noinline function residualadjustment(k::_CM.HC4, m::IVEstimator)
    n = nobs(m)
    h = leverage(m)
    p = round(Int, sum(h))
    adj = similar(h)
    @inbounds for j in eachindex(h)
        delta = min(4.0, n * h[j] / p)
        adj[j] = 1 / (1 - h[j])^(delta / 2)
    end
    adj
end

@noinline function residualadjustment(k::_CM.HC5, m::IVEstimator)
    n = nobs(m)
    h = leverage(m)
    p = round(Int, sum(h))
    mx = max(n * 0.7 * maximum(h) / p, 4.0)
    adj = similar(h)
    @inbounds for j in eachindex(h)
        alpha = min(n * h[j] / p, mx)
        adj[j] = 1 / (1 - h[j])^(alpha / 4)
    end
    adj
end

# Cluster-robust residual adjustments
@noinline residualadjustment(k::_CM.CR0, m::IVEstimator) = 1.0
@noinline residualadjustment(k::_CM.CR1, m::IVEstimator) = 1.0

# HAC (kernel) estimators - no residual adjustment needed
@noinline residualadjustment(k::_CM.HAC, m::IVEstimator) = 1.0

# CR2 and CR3 for IV - leverage-adjusted cluster-robust
function residualadjustment(k::_CM.CR2, m::IVEstimator)
    @assert length(k.g) == 1 "CR2 for IV currently only supports single-way clustering"
    g = k.g[1]
    X = m.postestimation.X
    resid = residuals_for_vcov(m)
    u = copy(resid)
    XX = bread(m)
    for groups in 1:g.ngroups
        ind = findall(==(groups), g)
        Xg = view(X, ind, :)
        ug = view(u, ind)
        Hgg = Xg * XX * Xg'
        # Apply (I - H_gg)^(-1/2) to residuals
        F = cholesky!(Symmetric(I - Hgg); check = false)
        if issuccess(F)
            ldiv!(ug, F.L, ug)
        end
    end
    return u ./ resid
end

function residualadjustment(k::_CM.CR3, m::IVEstimator)
    @assert length(k.g) == 1 "CR3 for IV currently only supports single-way clustering"
    g = k.g[1]
    X = m.postestimation.X
    resid = residuals_for_vcov(m)
    u = copy(resid)
    XX = bread(m)
    for groups in 1:g.ngroups
        ind = findall(==(groups), g)
        Xg = view(X, ind, :)
        ug = view(u, ind)
        Hgg = Xg * XX * Xg'
        # Apply (I - H_gg)^(-1) to residuals
        F = cholesky!(Symmetric(I - Hgg); check = false)
        if issuccess(F)
            ldiv!(ug, F, ug)
        end
    end
    return u ./ resid
end

"""
    CovarianceMatrices.aVar(k, m::IVEstimator)

Compute the asymptotic variance matrix for IV estimation.
"""
function _CM.aVar(
        k::K,
        m::IVEstimator;
        demean = false,
        prewhite = false,
        scale = true,
        kwargs...
) where {K <: _CM.AbstractAsymptoticVarianceEstimator}
    isnothing(m.postestimation) && error("Model does not have post-estimation data stored.")
    !has_residuals_data(m) && error("Model does not have residuals stored.")

    # Compute adjusted moment matrix
    # Use Adj if available (K-class: LIML, Fuller), otherwise use X (TSLS)
    pe = m.postestimation
    X_for_vcov = has_kclass_adj(pe) ? pe.Adj : pe.X

    u = residualadjustment(k, m)
    M = X_for_vcov .* residuals_for_vcov(m)
    if !(u isa Number && u == 1.0)
        M = M .* u
    end

    # Compute aVar using CovarianceMatrices
    Σ = _CM.aVar(k, M; demean = demean, prewhite = prewhite, scale = scale)
    return Σ
end

# Disambiguating method for cluster-robust estimators
function _CM.aVar(
        k::K,
        m::IVEstimator;
        demean = false,
        prewhite = false,
        scale = true,
        kwargs...
) where {K <: _CM.CR}
    isnothing(m.postestimation) && error("Model does not have post-estimation data stored.")
    !has_residuals_data(m) && error("Model does not have residuals stored.")

    # Compute adjusted moment matrix
    # Use Adj if available (K-class: LIML, Fuller), otherwise use X (TSLS)
    pe = m.postestimation
    X_for_vcov = has_kclass_adj(pe) ? pe.Adj : pe.X

    u = residualadjustment(k, m)
    M = X_for_vcov .* residuals_for_vcov(m)
    if !(u isa Number && u == 1.0)
        M = M .* u
    end

    # Compute aVar using CovarianceMatrices
    Σ = _CM.aVar(k, M; demean = demean, prewhite = prewhite, scale = scale)
    return Σ
end

##############################################################################
##
## Post-Estimation vcov Methods
##
## Primary API (CovarianceMatrices.jl standard):
##   vcov(CR1(cluster_vec), model)
##   stderror(CR1(cluster_vec), model)
##
## Convenience API (for stored cluster variables):
##   vcov(:ClusterVar, :CR1, model)   # looks up cluster from model
##   stderror(:ClusterVar, :CR1, model)
##
##############################################################################

"""
    vcov(ve::CovarianceMatrices.AbstractAsymptoticVarianceEstimator, m::IVEstimator)

Compute variance-covariance matrix using a specified estimator from CovarianceMatrices.jl.

# Supported Estimators
- **Heteroskedasticity-robust**: `HC0`, `HC1`, `HC2`, `HC3`, `HC4`, `HC5`
- **Cluster-robust**: `CR0`, `CR1`, `CR2`, `CR3`
- **HAC**: `Bartlett(bw)`, `Parzen(bw)`, `QuadraticSpectral(bw)`, etc.

# Examples
```julia
model = iv(TSLS(), df, @formula(y ~ x + (endo ~ instrument)))

# Heteroskedasticity-robust
vcov(HC3(), model)

# Cluster-robust (standard CovarianceMatrices.jl API)
vcov(CR1(df.firm_id[model.esample]), model)

# Two-way clustering
vcov(CR1((df.firm_id[model.esample], df.year[model.esample])), model)

# Convenience API (for stored cluster variables - avoids manual subsetting)
model = iv(TSLS(), df, @formula(y ~ x + (endo ~ inst)), save_cluster = :firm_id)
vcov(:firm_id, :CR1, model)
```
"""
function StatsBase.vcov(ve::CovarianceMatrices.AbstractAsymptoticVarianceEstimator, m::IVEstimator{T}) where {T}
    isnothing(m.postestimation) &&
        error("Model does not have post-estimation data stored. Post-estimation vcov not available.")
    !has_residuals_data(m) &&
        error("Model does not have residuals stored. Use save=:residuals or save=:all when fitting.")

    n = nobs(m)
    k = dof(m)
    B = bread(m)
    resid = residuals_for_vcov(m)

    # Check for truly homoskedastic variance estimators (not HC0/HC1)
    # Uncorrelated() assumes i.i.d. errors
    if ve isa CovarianceMatrices.Uncorrelated
        σ² = sum(abs2, resid) / dof_residual(m)
        return Symmetric(σ² * B)
    end

    # Sandwich variance: V = scale * B * A * B where A = aVar(k, m)
    # HC0/HR0, HC1/HR1, HC2, HC3, etc. all use sandwich form
    A = _CM.aVar(ve, m)

    # Scale factor depends on estimator type
    # Note: HC0 = HR0, HC1 = HR1 in CovarianceMatrices.jl
    #
    # The aVar function returns M'M/n where M is the adjusted moment matrix.
    # For HC1 (HR1), residualadjustment = sqrt(n/dof_res), so:
    #   aVar = M'M * (n/dof_res) / n = Meat / dof_res
    # where Meat = X̂'diag(e²)X̂
    #
    # R's sandwich formula for IV with HC1 is:
    #   V = (n/(n-k)) * B * Meat * B
    #
    # Since aVar = Meat / dof_res and dof_res = n - k:
    #   V = (n/(n-k)) * (n-k) * B * aVar * B = n * B * aVar * B
    #
    # So for HC1, scale = n (the DOF adjustment is already in aVar via residualadjustment)
    scale = if ve isa Union{_CM.CR0, _CM.CR1, _CM.CR2, _CM.CR3}
        # Cluster-robust: use fixest-style correction
        _cluster_robust_scale_iv(ve, m, n)
    else
        # HC0/HR0, HC1/HR1, HC2/HR2, HC3/HR3, HC4, HC5: scale by n
        # The DOF adjustment (for HC1) is handled via residualadjustment in aVar
        convert(T, n)
    end

    Σ = scale .* B * A * B
    return Symmetric(Σ)
end

"""
    _cluster_robust_scale_iv(k::_CM.CR, m::IVEstimator, n::Int)

Compute the scale factor for cluster-robust variance estimation for IV models.
Uses fixest-style small sample correction with FE nesting detection.

# Formula
`scale = n * G/(G-1) * (n-1)/(n-K)` where:
- G = number of clusters (for multi-way: minimum cluster count)
- K = k + k_fe_nonnested + k_fe_intercept (only FE NOT nested in cluster are counted)

With K.fixef = "nonnested" (fixest default):
- FE nested in the cluster variable are NOT counted in K
- FE NOT nested in the cluster variable ARE counted in K

This matches R fixest's behavior for IV models with cluster-robust standard errors.
"""
function _cluster_robust_scale_iv(k::_CM.CR, m::IVEstimator, n::Int)
    cluster_groups = k.g
    G = minimum(g.ngroups for g in cluster_groups)

    # G/(G-1) adjustment - only for CR1, CR2, CR3
    G_adj = k isa _CM.CR0 ? 1.0 : G / (G - 1)

    # Compute K for (n-1)/(n-K) adjustment using K.fixef = "nonnested"
    # K = k (non-FE params) + FE DOF for FE not nested in cluster
    k_params = dof(m)
    k_fe_nonnested = _compute_nonnested_fe_dof_iv(m, cluster_groups)

    # Account for intercept absorbed by FE (Stata/fixest behavior)
    # When FE absorbs an intercept, it should be counted in K
    k_fe_intercept = (has_fe(m) && !model_hasintercept(m)) ? 1 : 0

    K = k_params + k_fe_nonnested + k_fe_intercept

    # (n-1)/(n-K) adjustment
    K_adj = (n - 1) / (n - K)

    return convert(Float64, n * G_adj * K_adj)
end

"""
    _compute_nonnested_fe_dof_iv(m::IVEstimator, cluster_groups)

Compute the DOF for fixed effects that are NOT nested in the cluster variable(s).

A fixed effect is "nested" in a cluster if every FE group is contained within
exactly one cluster group (fixest behavior). When FE is nested, it doesn't add
information beyond the clustering and shouldn't be counted in the K adjustment.

Returns 0 if all FE are nested in the clustering, or the sum of ngroups for
non-nested FEs.
"""
function _compute_nonnested_fe_dof_iv(m::IVEstimator, cluster_groups)
    # If no fixed effects, return 0
    dof_fes(m) == 0 && return 0

    # Get FE info from postestimation data
    pe = m.postestimation
    isnothing(pe) && return 0

    fe_names = pe.fe_names
    isempty(fe_names) && return 0

    fe_groups = pe.fe_groups
    ngroups = pe.ngroups

    # Fallback if fe_groups not stored
    if isempty(fe_groups)
        return _compute_nonnested_fe_dof_by_name_iv(m, cluster_groups)
    end

    # Extract cluster refs from the CR estimator's grouping objects
    cluster_refs_list = [g.groups for g in cluster_groups]

    k_fe_nonnested = 0
    for (i, fe_name) in enumerate(fe_names)
        # A FE is nested if it's nested in ANY cluster dimension
        is_nested = any(crefs -> _isnested_groups_iv(fe_groups[i], crefs, ngroups[i]),
            cluster_refs_list)
        if !is_nested
            k_fe_nonnested += ngroups[i]
        end
    end

    return k_fe_nonnested
end

"""
    _isnested_groups_iv(fe_refs::Vector{Int}, cluster_refs::Vector{Int}, n_fe_groups::Int) -> Bool

Check if fixed effect groups are nested within cluster groups for IV models.

A FE is nested in a cluster if every FE group belongs to exactly one cluster group.
This is the fixest definition of nesting.
"""
function _isnested_groups_iv(fe_refs::Vector{Int}, cluster_refs::Vector{Int}, n_fe_groups::Int)
    entries = zeros(Int, n_fe_groups)
    @inbounds for i in eachindex(fe_refs, cluster_refs)
        feref, cref = fe_refs[i], cluster_refs[i]
        if entries[feref] == 0
            entries[feref] = cref
        elseif entries[feref] != cref
            return false
        end
    end
    return true
end

"""
    _compute_nonnested_fe_dof_by_name_iv(m::IVEstimator, cluster_groups)

Fallback nesting detection using name-matching heuristic for IV models.
Used when fe_groups are not available.

FE is considered nested if its name matches one of the cluster names.
"""
function _compute_nonnested_fe_dof_by_name_iv(m::IVEstimator, cluster_groups)
    pe = m.postestimation
    isnothing(pe) && return 0

    fe_names = pe.fe_names
    ngroups = pe.ngroups
    cluster_names = keys(pe.cluster_vars)

    k_fe_nonnested = 0
    for (i, fe_name) in enumerate(fe_names)
        is_nested = fe_name in cluster_names
        if !is_nested
            k_fe_nonnested += ngroups[i]
        end
    end

    return k_fe_nonnested
end

##############################################################################
##
## Symbol-Based Cluster-Robust Variance API
##
## When CR types are constructed with Symbol(s) instead of data vectors,
## these methods look up the cluster data from the model's stored clusters.
##
## Usage:
##   vcov(CR1(:StateID), model)           # single cluster
##   vcov(CR1(:StateID, :YearID), model)  # multi-way clustering
##
##############################################################################

"""
    _lookup_cluster_vecs_iv(cluster_syms::Tuple{Vararg{Symbol}}, m::IVEstimator)

Look up cluster vectors from stored cluster data in the IV model.
Returns a tuple of vectors corresponding to the requested cluster symbols.
"""
function _lookup_cluster_vecs_iv(cluster_syms::Tuple{Vararg{Symbol}}, m::IVEstimator)
    return Tuple(begin
                     haskey(m.postestimation.cluster_vars, name) ||
                         _cluster_not_found_error(name, m)
                     m.postestimation.cluster_vars[name]
                 end
    for name in cluster_syms)
end

"""
    vcov(v::CovarianceMatrices.CR0{T}, m::IVEstimator) where T<:Tuple{Vararg{Symbol}}

Compute cluster-robust variance with CR0 estimator using stored cluster variable(s).

# Examples
```julia
model = iv(TSLS(), df, @formula(y ~ x + (endo ~ inst)), save_cluster = :firm_id)
vcov(CR0(:firm_id), model)
vcov(CR0(:firm_id, :year), model)  # multi-way
```
"""
function CovarianceMatrices.vcov(
        v::CovarianceMatrices.CR0{T}, m::IVEstimator) where {T <: Tuple{Vararg{Symbol}}}
    cluster_vecs = _lookup_cluster_vecs_iv(v.g, m)
    return vcov(CovarianceMatrices.CR0(cluster_vecs), m)
end

"""
    vcov(v::CovarianceMatrices.CR1{T}, m::IVEstimator) where T<:Tuple{Vararg{Symbol}}

Compute cluster-robust variance with CR1 estimator using stored cluster variable(s).

# Examples
```julia
model = iv(TSLS(), df, @formula(y ~ x + (endo ~ inst)), save_cluster = :firm_id)
vcov(CR1(:firm_id), model)
vcov(CR1(:firm_id, :year), model)  # multi-way
```
"""
function CovarianceMatrices.vcov(
        v::CovarianceMatrices.CR1{T}, m::IVEstimator) where {T <: Tuple{Vararg{Symbol}}}
    cluster_vecs = _lookup_cluster_vecs_iv(v.g, m)
    return vcov(CovarianceMatrices.CR1(cluster_vecs), m)
end

"""
    vcov(v::CovarianceMatrices.CR2{T}, m::IVEstimator) where T<:Tuple{Vararg{Symbol}}

Compute cluster-robust variance with CR2 (leverage-adjusted) estimator using stored cluster variable(s).
"""
function CovarianceMatrices.vcov(
        v::CovarianceMatrices.CR2{T}, m::IVEstimator) where {T <: Tuple{Vararg{Symbol}}}
    cluster_vecs = _lookup_cluster_vecs_iv(v.g, m)
    return vcov(CovarianceMatrices.CR2(cluster_vecs), m)
end

"""
    vcov(v::CovarianceMatrices.CR3{T}, m::IVEstimator) where T<:Tuple{Vararg{Symbol}}

Compute cluster-robust variance with CR3 (squared leverage) estimator using stored cluster variable(s).
"""
function CovarianceMatrices.vcov(
        v::CovarianceMatrices.CR3{T}, m::IVEstimator) where {T <: Tuple{Vararg{Symbol}}}
    cluster_vecs = _lookup_cluster_vecs_iv(v.g, m)
    return vcov(CovarianceMatrices.CR3(cluster_vecs), m)
end

"""
    stderror(ve::CovarianceMatrices.AbstractAsymptoticVarianceEstimator, m::IVEstimator)

Compute standard errors using a specified variance estimator.
"""
function StatsBase.stderror(ve::CovarianceMatrices.AbstractAsymptoticVarianceEstimator, m::IVEstimator)
    return sqrt.(diag(vcov(ve, m)))
end

##############################################################################
##
## Helper Functions
##
##############################################################################

function _cluster_not_found_error(cluster_name::Symbol, m::IVEstimator)
    available = isempty(m.postestimation.cluster_vars) ? "none" :
                join(keys(m.postestimation.cluster_vars), ", :")
    error("\n    Cluster variable :$cluster_name not found in model.

    Available cluster variables: :$available

    To use this cluster variable, either:
      1. Re-fit with save_cluster=:$cluster_name
      2. Use data directly: vcov(CR1(df.$cluster_name[model.esample]), model)
    ")
end

##############################################################################
##
## Recompute First-Stage F-Statistics with Different vcov
##
##############################################################################

"""
    _resolve_cr_vcov(vcov_type, m::IVEstimator)

Resolve CR estimator with symbols to CR estimator with actual cluster vectors.
For non-CR or CR with actual data, returns vcov_type unchanged.
"""
function _resolve_cr_vcov(vcov_type::CovarianceMatrices.CR0{T},
        m::IVEstimator) where {T <: Tuple{Vararg{Symbol}}}
    cluster_vecs = _lookup_cluster_vecs_iv(vcov_type.g, m)
    return CovarianceMatrices.CR0(cluster_vecs)
end

function _resolve_cr_vcov(vcov_type::CovarianceMatrices.CR1{T},
        m::IVEstimator) where {T <: Tuple{Vararg{Symbol}}}
    cluster_vecs = _lookup_cluster_vecs_iv(vcov_type.g, m)
    return CovarianceMatrices.CR1(cluster_vecs)
end

function _resolve_cr_vcov(vcov_type::CovarianceMatrices.CR2{T},
        m::IVEstimator) where {T <: Tuple{Vararg{Symbol}}}
    cluster_vecs = _lookup_cluster_vecs_iv(vcov_type.g, m)
    return CovarianceMatrices.CR2(cluster_vecs)
end

function _resolve_cr_vcov(vcov_type::CovarianceMatrices.CR3{T},
        m::IVEstimator) where {T <: Tuple{Vararg{Symbol}}}
    cluster_vecs = _lookup_cluster_vecs_iv(vcov_type.g, m)
    return CovarianceMatrices.CR3(cluster_vecs)
end

# Fallback for non-CR or CR with actual data
_resolve_cr_vcov(vcov_type, m::IVEstimator) = vcov_type

"""
    recompute_first_stage_fstat(m::IVEstimator, vcov_type)

Recompute joint Kleibergen-Paap F-statistic using a different vcov estimator.
Requires model to have stored first-stage data.

# Returns
- `(F_kp, p_kp)`: First-stage F-statistic and p-value with the new vcov
"""
function recompute_first_stage_fstat(m::IVEstimator{T}, vcov_type) where {T}
    pe = m.postestimation
    isnothing(pe) && return T(NaN), T(NaN)
    !has_first_stage_data(pe.first_stage_data) && return T(NaN), T(NaN)

    fsd = pe.first_stage_data
    dof_fes_val = dof_fes(m)

    # Resolve CR symbols to actual cluster vectors
    resolved_vcov = _resolve_cr_vcov(vcov_type, m)

    return compute_first_stage_fstat(
        fsd.Xendo_res, fsd.Z_res, fsd.Pi,
        resolved_vcov, nobs(m), dof(m), dof_fes_val
    )
end

"""
    recompute_per_endogenous_fstats(m::IVEstimator, vcov_type)

Recompute per-endogenous F-statistics using a different vcov estimator.

# Returns
- `(F_stats, p_values)`: Vectors of F-statistics and p-values per endogenous variable
"""
function recompute_per_endogenous_fstats(m::IVEstimator{T}, vcov_type) where {T}
    pe = m.postestimation
    isnothing(pe) && return T[], T[]
    !has_first_stage_data(pe.first_stage_data) && return T[], T[]

    fsd = pe.first_stage_data
    dof_fes_val = dof_fes(m)

    # Resolve CR symbols to actual cluster vectors
    resolved_vcov = _resolve_cr_vcov(vcov_type, m)

    return compute_per_endogenous_fstats(
        fsd.Xendo_res, fsd.Z_res, fsd.Pi,
        resolved_vcov, nobs(m), dof(m), dof_fes_val;
        Xendo_orig = fsd.Xendo_orig,
        newZ = fsd.newZ
    )
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
using the specified variance estimator. The vcov estimator is
deep-copied to avoid aliasing issues.

# Examples
```julia
model = iv(TSLS(), df, @formula(y ~ x + (endo ~ z)))

# Heteroskedasticity-robust
model_hc3 = model + vcov(HC3())
stderror(model_hc3)  # Uses precomputed HC3 standard errors
model_hc3.F_kp       # First-stage F recomputed with HC3

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

    # Recompute first-stage F with this vcov type
    F_kp, p_kp = recompute_first_stage_fstat(m, v.source)

    # Recompute per-endogenous F-stats
    F_kp_per_endo, p_kp_per_endo = recompute_per_endogenous_fstats(m, v.source)

    # Deep copy the vcov estimator to avoid aliasing
    vcov_copy = deepcopy_vcov(v.source)

    # Return new IVEstimator with same data but different vcov type
    return IVEstimator{T, E, V2, P}(
        m.estimator, m.coef,
        m.esample, m.residuals_esample, m.has_residuals, m.fe,
        m.postestimation, m.fekeys,
        m.coefnames, m.responsename,
        m.formula, m.formula_schema, m.contrasts,
        m.nobs, m.dof, m.dof_fes, m.dof_residual,
        m.rss, m.tss,
        m.iterations, m.converged, m.r2_within,
        vcov_copy, Symmetric(vcov_mat), se, t_stats, p_values,
        F_stat, p_val,
        F_kp, p_kp, F_kp_per_endo, p_kp_per_endo
    )
end

##############################################################################
##
## first_stage() - Return first-stage diagnostics as a struct
##
##############################################################################

"""
    first_stage(m::IVEstimator) -> FirstStageResult

Return first-stage diagnostics for an IV model.
Uses the variance estimator stored in the model.

# Returns
A `FirstStageResult` struct containing:
- Joint Kleibergen-Paap F-statistic and p-value
- Per-endogenous F-statistics and p-values
- Metadata (number of instruments, variance estimator type)

# Examples
```julia
model = iv(TSLS(), df, @formula(y ~ x + (endo ~ z1 + z2)))
fs = first_stage(model)
fs.F_joint           # Joint F-statistic
fs.F_per_endo        # Per-endogenous F-stats
```

See also: [`FirstStageResult`](@ref)
"""
function first_stage(m::IVEstimator{T}) where {T}
    isnothing(m.postestimation) &&
        error("Model does not have post-estimation data stored. Fit with save=true.")
    !has_first_stage_data(m.postestimation.first_stage_data) &&
        error("First-stage data not available.")

    fsd = m.postestimation.first_stage_data

    # Get vcov type name
    vcov_name = string(typeof(m.vcov_estimator).name.name)

    FirstStageResult{T}(
        m.F_kp,
        m.p_kp,
        fsd.endogenous_names,
        m.F_kp_per_endo,
        m.p_kp_per_endo,
        length(fsd.endogenous_names),
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
    totwidth = max(60, max_name_len + 35)

    # Header
    println(io, "First-Stage Diagnostics ($(fs.vcov_type))")
    println_horizontal_line(io, totwidth)

    # Joint test
    println(io, "Joint Test (Kleibergen-Paap):")
    @printf(io, "  F-statistic: %10.4f       p-value: %.4f\n", fs.F_joint, fs.p_joint)

    # Per-endogenous table
    println(io, "\nPer-Endogenous F-Statistics:")
    println_horizontal_line(io, totwidth)
    @printf(io, "% -30s %14s %14s\n", "Endogenous", "F-stat", "P-value")
    println_horizontal_line(io, totwidth)

    for (j, name) in enumerate(fs.endogenous_names)
        display_name = length(name) > 28 ? name[1:25] * "..." : name
        @printf(io, "% -30s %14.4f %14.4f\n",
            display_name, fs.F_per_endo[j], fs.p_per_endo[j])
    end

    println_horizontal_line(io, totwidth)
    println(io, "\nInstruments: $(fs.n_instruments) excluded, $(fs.n_endogenous) endogenous")
    print_horizontal_line(io, totwidth)
end

function Base.show(io::IO, ::MIME"text/html", fs::FirstStageResult{T}) where {T}
    html_table_start(io; class = "regress-table regress-first-stage",
        caption = "First-Stage Diagnostics ($(fs.vcov_type))")

    # Joint test section
    html_thead_start(io; class = "regress-joint-test")
    html_row(io, ["Joint Test (Kleibergen-Paap)", "", ""]; is_header = true)
    html_thead_end(io)
    html_tbody_start(io; class = "regress-joint-body")
    html_row(io, ["F-statistic", format_number(fs.F_joint), ""])
    html_row(io, ["P-value", format_pvalue(fs.p_joint), ""])
    html_tbody_end(io)

    # Per-endogenous section
    html_thead_start(io; class = "regress-per-endo-header")
    html_row(io, ["Endogenous", "F-stat", "P-value"]; is_header = true)
    html_thead_end(io)
    html_tbody_start(io; class = "regress-per-endo-body")
    for (j, name) in enumerate(fs.endogenous_names)
        html_row(io, [
            name, format_number(fs.F_per_endo[j]), format_pvalue(fs.p_per_endo[j])])
    end
    html_tbody_end(io)

    # Footer
    html_tfoot_start(io; class = "regress-footer")
    html_row(io, [
        "Instruments: $(fs.n_instruments) excluded, $(fs.n_endogenous) endogenous", "", ""])
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

function _estimator_name(m::IVEstimator)
    e = m.estimator
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

    # Always show first-stage diagnostics for IV models (joint Kleibergen-Paap)
    out = vcat(out,
        ["F (1st stage, joint)" sprint(show, m.F_kp, context = :compact => true);
         "P (1st stage, joint)" @sprintf("%.3f", m.p_kp);])

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
    vcov_name = vcov_type_name(m.vcov_estimator)
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
    vcov_name = vcov_type_name(m.vcov_estimator)
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
        throw("To access residuals for a model with high-dimensional fixed effects,  run `m = iv(..., save = :residuals)` and then access residuals with `residuals(m)`.")
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
            throw("To access residuals in a fixed effect regression, run `iv` with the option save = :residuals, and then access residuals with `residuals()`")
        !has_fe(m) &&
            throw("To access residuals, use residuals(m, data) where `m` is an estimated IVEstimator and `data` is a Table")
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
    !has_fe(m) && throw("fe() is not defined for models without fixed effects")
    if keepkeys
        m.fe
    else
        m.fe[!, (length(m.fekeys) + 1):end]
    end
end

function StatsAPI.coeftable(m::IVEstimator; level = 0.95)
    cc = coef(m)
    se = m.se
    tt = m.t_stats
    pv = m.p_values
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
## IVMatrixEstimator - Matrix-based IV estimation (like OLSMatrixEstimator)
##
##############################################################################

"""
    PostEstimationDataIVMatrix{T}

Stores data needed for post-estimation variance computation in matrix-based IV.

# Fields
- `X::Matrix{T}`: Original regressor matrix [Xexo, Xendo]
- `Z::Matrix{T}`: Instrument matrix [Xexo, Zinstr]
- `X_hat::Matrix{T}`: Projected regressors P_Z * X
- `y::Vector{T}`: Response vector
- `residuals::Vector{T}`: Residuals e = y - X*β
- `invXhatXhat::Matrix{T}`: (X̂'X̂)⁻¹ for sandwich formula
- `n_endogenous::Int`: Number of endogenous variables
"""
struct PostEstimationDataIVMatrix{T <: AbstractFloat}
    X::Matrix{T}
    Z::Matrix{T}
    X_hat::Matrix{T}
    y::Vector{T}
    residuals::Vector{T}
    invXhatXhat::Matrix{T}
    n_endogenous::Int
end

"""
    IVMatrixEstimator{T, V} <: AbstractRegressModel

Matrix-based IV estimator for use without formula interface.
Designed for programmatic use (e.g., LocalProjections.jl).

# Type Parameters
- `T`: Element type (Float64 or Float32)
- `V`: Variance estimator type

# Fields
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
- `vcov_estimator::V`: Variance estimator used
- `vcov_matrix::Symmetric{T, Matrix{T}}`: Precomputed variance-covariance matrix
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
struct IVMatrixEstimator{T <: AbstractFloat, V} <: AbstractRegressModel
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

    # Variance-covariance
    vcov_estimator::V
    vcov_matrix::Symmetric{T, Matrix{T}}
    se::Vector{T}
    t_stats::Vector{T}
    p_values::Vector{T}
end

has_iv(::IVMatrixEstimator) = true
has_fe(::IVMatrixEstimator) = false
dof_fes(::IVMatrixEstimator) = 0
model_hasintercept(m::IVMatrixEstimator) = m.has_intercept

##############################################################################
## StatsAPI Interface for IVMatrixEstimator
##############################################################################

function StatsAPI.coef(m::IVMatrixEstimator)
    beta = copy(m.coef)
    beta[.!m.basis_coef] .= zero(eltype(beta))
    return beta
end

StatsAPI.nobs(m::IVMatrixEstimator) = m.nobs
StatsAPI.dof(m::IVMatrixEstimator) = m.dof
StatsAPI.dof_residual(m::IVMatrixEstimator) = m.dof_residual
StatsAPI.rss(m::IVMatrixEstimator) = m.rss
StatsAPI.r2(m::IVMatrixEstimator) = m.r2

function StatsAPI.response(m::IVMatrixEstimator)
    return m.postestimation.y
end

function StatsAPI.residuals(m::IVMatrixEstimator)
    return m.postestimation.residuals
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
    bread(m::IVMatrixEstimator)

Returns (X̂'X̂)⁻¹ for sandwich variance estimation.
"""
bread(m::IVMatrixEstimator) = m.postestimation.invXhatXhat

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

Returns diagonal of hat matrix H = X̂(X̂'X̂)⁻¹X̂' for HC2/HC3/HC4/HC5.
"""
function StatsAPI.leverage(m::IVMatrixEstimator)
    X_hat = m.postestimation.X_hat
    invXX = m.postestimation.invXhatXhat
    # h_ii = X̂_i' * (X̂'X̂)⁻¹ * X̂_i
    # Efficient computation: sum((X_hat * invXX) .* X_hat, dims=2)
    return vec(sum((X_hat * invXX) .* X_hat, dims = 2))
end

# CovarianceMatrices.jl uses numobs, which is distinct from StatsAPI.nobs
_CM.numobs(m::IVMatrixEstimator) = m.nobs

# Residual adjustments for HC estimators
function residualadjustment(k::_CM.HC0, m::IVMatrixEstimator)
    return ones(eltype(m.postestimation.residuals), nobs(m))
end

function residualadjustment(k::_CM.HC1, m::IVMatrixEstimator{T}) where {T}
    n, k_params = nobs(m), dof(m)
    return fill(sqrt(T(n) / T(n - k_params)), n)
end

function residualadjustment(k::_CM.HC2, m::IVMatrixEstimator{T}) where {T}
    h = leverage(m)
    return T(1) ./ sqrt.(max.(T(1) .- h, eps(T)))
end

function residualadjustment(k::_CM.HC3, m::IVMatrixEstimator{T}) where {T}
    h = leverage(m)
    return T(1) ./ max.(T(1) .- h, eps(T))
end

@noinline function residualadjustment(k::_CM.HC4, m::IVMatrixEstimator{T}) where {T}
    n = nobs(m)
    p = dof(m)
    h = leverage(m)
    δ = @. min(4.0, h * n / p)
    return @. T(1) / (T(1) - h)^δ
end

@noinline function residualadjustment(k::_CM.HC5, m::IVMatrixEstimator{T}) where {T}
    n = nobs(m)
    p = dof(m)
    h = leverage(m)
    hmax = max(n * 0.7 * maximum(h) / p, 4.0)
    δ = @. min(h * n / p, hmax)
    return @. sqrt(T(1) / (T(1) - h)^δ)
end

# HAC (Bartlett, Parzen, etc.) - no adjustment needed
@noinline residualadjustment(k::_CM.HAC, m::IVMatrixEstimator) = 1.0

# CR (cluster-robust) - no individual residual adjustment needed (handled in aVar)
@noinline residualadjustment(k::_CM.CR0, m::IVMatrixEstimator) = 1.0
@noinline residualadjustment(k::_CM.CR1, m::IVMatrixEstimator) = 1.0
@noinline residualadjustment(k::_CM.CR2, m::IVMatrixEstimator) = 1.0
@noinline residualadjustment(k::_CM.CR3, m::IVMatrixEstimator) = 1.0

# aVar for CR (cluster-robust) estimators - disambiguating method
function _CM.aVar(
        k::K,
        m::IVMatrixEstimator{T};
        demean = false,
        prewhite = false,
        scale = true,
        kwargs...
) where {K <: _CM.CR, T}
    # Compute adjusted moment matrix: X̂ .* e
    X_hat = m.postestimation.X_hat
    resid = m.postestimation.residuals
    M = X_hat .* resid

    # Compute aVar using CovarianceMatrices
    Σ = _CM.aVar(k, M; demean = demean, prewhite = prewhite, scale = scale)
    return Σ
end

# aVar for HC-type estimators
function _CM.aVar(
        k::K,
        m::IVMatrixEstimator{T};
        demean = false,
        prewhite = false,
        scale = true,
        kwargs...
) where {K <: _CM.AbstractAsymptoticVarianceEstimator, T}
    # Compute adjusted moment matrix: X̂ .* (u .* e)
    # where u is the residual adjustment factor
    X_hat = m.postestimation.X_hat
    resid = m.postestimation.residuals

    u = residualadjustment(k, m)
    M = X_hat .* resid
    if !(u isa Number && u == 1.0)
        M = M .* u
    end

    # Compute aVar using CovarianceMatrices
    Σ = _CM.aVar(k, M; demean = demean, prewhite = prewhite, scale = scale)
    return Σ
end

"""
    vcov(ve::AbstractAsymptoticVarianceEstimator, m::IVMatrixEstimator)

Compute robust variance-covariance matrix for IV matrix estimator.
"""
function StatsBase.vcov(ve::CovarianceMatrices.AbstractAsymptoticVarianceEstimator,
        m::IVMatrixEstimator{T}) where {T}
    n = nobs(m)
    B = bread(m)
    resid = m.postestimation.residuals

    # Homoskedastic case
    if ve isa CovarianceMatrices.Uncorrelated
        σ² = sum(abs2, resid) / dof_residual(m)
        return Symmetric(σ² * B)
    end

    # Sandwich: V = scale * B * A * B
    A = _CM.aVar(ve, m)
    scale = convert(T, n)

    Σ = scale .* B * A * B
    return Symmetric(Σ)
end

function StatsBase.stderror(ve::CovarianceMatrices.AbstractAsymptoticVarianceEstimator,
        m::IVMatrixEstimator)
    V = vcov(ve, m)
    return sqrt.(diag(V))
end

##############################################################################
## VcovSpec + operator for IVMatrixEstimator
##############################################################################

"""
    m::IVMatrixEstimator + v::VcovSpec

Create a new IVMatrixEstimator with updated variance-covariance estimator.
"""
function Base.:+(m::IVMatrixEstimator{T, V1}, v::VcovSpec{V2}) where {T, V1, V2}
    new_vcov = vcov(v.source, m)
    new_se = sqrt.(diag(new_vcov))

    # Recompute t-stats and p-values
    cc = coef(m)
    new_t = cc ./ new_se
    new_p = 2 .* tdistccdf.(dof_residual(m), abs.(new_t))

    return IVMatrixEstimator{T, V2}(
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
        deepcopy_vcov(v.source),
        new_vcov,
        new_se,
        new_t,
        new_p
    )
end

##############################################################################
## first_stage for IVMatrixEstimator
##############################################################################

"""
    first_stage(m::IVMatrixEstimator)

Compute first-stage F-statistic for IV matrix estimator.
Returns a FirstStageResult with joint F-statistic.
"""
function first_stage(m::IVMatrixEstimator{T}) where {T}
    pe = m.postestimation
    n_endo = pe.n_endogenous

    if n_endo == 0
        return FirstStageResult{T}(
            T(NaN), T(NaN), String[], T[], T[], 0, 0, "N/A"
        )
    end

    # Extract data
    Z = pe.Z
    X = pe.X
    n = size(X, 1)
    k_total = size(X, 2)
    k_exo = k_total - n_endo
    k_z = size(Z, 2) - k_exo  # Number of excluded instruments

    # First stage: regress each endogenous var on Z
    # F-stat tests whether excluded instruments have joint significance

    # Simple approach: compute partial F-stat for excluded instruments
    # For single endogenous: F = (R² / k_z) / ((1 - R²) / (n - k_z - k_exo - 1))

    # Compute projected values
    ZZ = Z' * Z
    ZZ_inv = try
        inv(cholesky(Symmetric(ZZ)))
    catch
        pinv(ZZ)
    end
    P_Z = Z * ZZ_inv * Z'

    # For each endogenous variable, compute first-stage R²
    Xendo = X[:, (k_exo + 1):end]
    Xendo_hat = P_Z * Xendo

    # Joint F-statistic (Cragg-Donald style for simplicity)
    # This is an approximation; full K-P would require more complex computation
    F_values = T[]
    p_values = T[]
    endo_names = ["endo_$i" for i in 1:n_endo]

    for i in 1:n_endo
        x_endo = Xendo[:, i]
        x_hat = Xendo_hat[:, i]

        # Explained SS by instruments
        ess = sum(abs2, x_hat .- mean(x_hat))
        tss_endo = sum(abs2, x_endo .- mean(x_endo))
        rss_endo = tss_endo - ess

        # F-statistic
        df1 = k_z
        df2 = n - k_z - k_exo - (m.has_intercept ? 1 : 0)
        F_val = (ess / df1) / (rss_endo / max(df2, 1))
        p_val = fdistccdf(df1, df2, F_val)

        push!(F_values, F_val)
        push!(p_values, p_val)
    end

    # Joint F is minimum of individual F-stats (conservative)
    F_joint = minimum(F_values)
    p_joint = maximum(p_values)

    return FirstStageResult{T}(
        F_joint,
        p_joint,
        endo_names,
        F_values,
        p_values,
        n_endo,
        k_z,
        string(typeof(m.vcov_estimator))
    )
end

##############################################################################
## Show methods for IVMatrixEstimator
##############################################################################

function Base.show(io::IO, m::IVMatrixEstimator)
    print(io, "IVMatrixEstimator(n=$(nobs(m)), k=$(dof(m)))")
end

function Base.show(io::IO, ::MIME"text/plain", m::IVMatrixEstimator{T}) where {T}
    println(io, "IV Matrix Estimator (TSLS)")
    println(io, "─" ^ 40)
    println(io, "Observations:      $(nobs(m))")
    println(io, "Parameters:        $(dof(m))")
    println(io, "R²:                $(round(r2(m), digits=4))")
    println(io, "Residual DoF:      $(dof_residual(m))")
    println(io, "Endogenous vars:   $(m.postestimation.n_endogenous)")
    println(io)
    println(io, "Coefficients:")
    cc = coef(m)
    se = m.se
    for i in 1:length(cc)
        println(io, "  β[$i] = $(round(cc[i], digits=6)) (SE: $(round(se[i], digits=6)))")
    end
end
