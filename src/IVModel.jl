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
- `first_stage_data::Union{FirstStageData{T}, Nothing}`: First-stage data for F-statistics
- `Adj::Union{Matrix{T}, Nothing}`: K-class adjustment matrix (W - k*Wres) for vcov
- `kappa::Union{T, Nothing}`: K-class parameter (nothing for TSLS, k_LIML for LIML, etc.)
"""
struct PostEstimationDataIV{T, W <: AbstractWeights}
    X::Matrix{T}
    Xhat::Matrix{T}
    crossx::Cholesky{T, Matrix{T}}
    invXX::Symmetric{T, Matrix{T}}
    weights::W
    cluster_vars::NamedTuple
    basis_coef::BitVector
    first_stage_data::Union{FirstStageData{T}, Nothing}
    Adj::Union{Matrix{T}, Nothing}
    kappa::Union{T, Nothing}
end

"""
    IVEstimator <: StatsAPI.RegressionModel

Model type for instrumental variables regression.

Use `iv(estimator, df, formula)` to fit this model type, where `estimator` is
one of `TSLS()`, `LIML()`, etc.

# Type Parameters
- `T`: Float type (Float64 or Float32)
- `V`: Variance-covariance estimator type (HC1, HC3, CR1, etc.)

# Examples
```julia
iv(TSLS(), df, @formula(y ~ x + (endo ~ instrument)))
```
"""
struct IVEstimator{T, V} <: StatsAPI.RegressionModel
    estimator::AbstractIVEstimator  # Which IV estimator was used

    coef::Vector{T}   # Vector of coefficients

    esample::BitVector      # Is the row of the original dataframe part of the estimation sample?
    residuals::Union{AbstractVector, Nothing}
    fe::DataFrame

    # Post-estimation data for CovarianceMatrices.jl
    postestimation::Union{PostEstimationDataIV{T}, Nothing}

    fekeys::Vector{Symbol}

    coefnames::Vector       # Name of coefficients
    responsename::Union{String, Symbol} # Name of dependent variable
    formula::FormulaTerm        # Original formula
    formula_schema::FormulaTerm # Schema for predict
    contrasts::Dict

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
StatsAPI.islinear(m::IVEstimator) = true
StatsAPI.deviance(m::IVEstimator) = rss(m)
StatsAPI.nulldeviance(m::IVEstimator) = m.tss
StatsAPI.rss(m::IVEstimator) = m.rss
StatsAPI.mss(m::IVEstimator) = nulldeviance(m) - rss(m)
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
    isnothing(m.residuals) &&
        error("Model does not have residuals stored. Use save=:residuals or save=:all when fitting.")

    # Use Adj if available (K-class: LIML, Fuller), otherwise use X (TSLS)
    pe = m.postestimation
    X_for_vcov = isnothing(pe.Adj) ? pe.X : pe.Adj
    return X_for_vcov .* m.residuals
end

"""
    CovarianceMatrices.score(m::IVEstimator)

Returns the score matrix (Jacobian of moment conditions) for IV: -X'X/n.
"""
# function CovarianceMatrices.hessian_objective(m::IVEstimator)
#     isnothing(m.X) && error("Model does not have design matrix stored. Post-estimation vcov not available.")
#     return -Symmetric(m.X' * m.X) / m.nobs
# end

"""
    CovarianceMatrices.objective_hessian(m::IVEstimator)

Returns the Hessian of the least squares objective function: X'X/n.
"""
# function CovarianceMatrices.hessian_objective(m::IVEstimator)
#     isnothing(m.X) && error("Model does not have design matrix stored. Post-estimation vcov not available.")
#     return Symmetric(m.X' * m.X) / m.nobs
# end

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
For IV: h_i = X_i * (X'X)^(-1) * X_i' where X contains predicted endogenous.
"""
function leverage(m::IVEstimator)
    isnothing(m.postestimation) && error("Model does not have post-estimation data stored.")
    X = m.postestimation.X
    invXX = m.postestimation.invXX
    # h_i = X_i * invXX * X_i'
    return vec(sum(X .* (X * invXX), dims = 2))
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
    resid = m.residuals
    u = copy(resid)
    XX = bread(m)
    for groups in 1:g.ngroups
        ind = findall(x -> x == groups, g)
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
    resid = m.residuals
    u = copy(resid)
    XX = bread(m)
    for groups in 1:g.ngroups
        ind = findall(g .== groups)
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
    isnothing(m.residuals) && error("Model does not have residuals stored.")

    # Compute adjusted moment matrix
    # Use Adj if available (K-class: LIML, Fuller), otherwise use X (TSLS)
    pe = m.postestimation
    X_for_vcov = isnothing(pe.Adj) ? pe.X : pe.Adj

    u = residualadjustment(k, m)
    M = X_for_vcov .* m.residuals
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
    isnothing(m.residuals) && error("Model does not have residuals stored.")

    # Compute adjusted moment matrix
    # Use Adj if available (K-class: LIML, Fuller), otherwise use X (TSLS)
    pe = m.postestimation
    X_for_vcov = isnothing(pe.Adj) ? pe.X : pe.Adj

    u = residualadjustment(k, m)
    M = X_for_vcov .* m.residuals
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
model = iv(TSLS(), df, @formula(y ~ x + (endo ~ inst)), save_cluster=:firm_id)
vcov(:firm_id, :CR1, model)
```
"""
function StatsBase.vcov(ve::CovarianceMatrices.AbstractAsymptoticVarianceEstimator, m::IVEstimator{T}) where {T}
    isnothing(m.postestimation) &&
        error("Model does not have post-estimation data stored. Post-estimation vcov not available.")
    isnothing(m.residuals) &&
        error("Model does not have residuals stored. Use save=:residuals or save=:all when fitting.")

    n = nobs(m)
    k = dof(m)
    B = bread(m)
    resid = m.residuals

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
Uses fixest-style small sample correction.
"""
function _cluster_robust_scale_iv(k::_CM.CR, m::IVEstimator, n::Int)
    cluster_groups = k.g
    G = minimum(g.ngroups for g in cluster_groups)

    # G/(G-1) adjustment - only for CR1, CR2, CR3
    G_adj = k isa _CM.CR0 ? 1.0 : G / (G - 1)

    # For IV, K = k (number of params) - we don't have FE nesting logic for IV
    # This is a simpler case since IV models typically don't have absorbed FE DOF
    K = dof(m)
    K_adj = (n - 1) / (n - K)

    return convert(Float64, n * G_adj * K_adj)
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
    error("""
    Cluster variable :$cluster_name not found in model.

    Available cluster variables: :$available

    To use this cluster variable, either:
      1. Re-fit with save_cluster=:$cluster_name
      2. Use data directly: vcov(CR1(df.$cluster_name[model.esample]), model)
    """)
end

##############################################################################
##
## Additional Methods
##
##############################################################################

function StatsAPI.loglikelihood(m::IVEstimator)
    n = nobs(m)
    -n/2 * (log(2π * deviance(m) / n) + 1)
end

function StatsAPI.nullloglikelihood(m::IVEstimator)
    n = nobs(m)
    -n/2 * (log(2π * nulldeviance(m) / n) + 1)
end

function nullloglikelihood_within(m::IVEstimator)
    n = nobs(m)
    tss_within = deviance(m) / (1 - m.r2_within)
    -n/2 * (log(2π * tss_within / n) + 1)
end

function StatsAPI.adjr2(model::IVEstimator, variant::Symbol = :devianceratio)
    has_int = hasintercept(formula(model))
    k = dof(model) + dof_fes(model) + has_int
    if variant == :McFadden
        k = k - has_int - has_fe(model)
        ll = loglikelihood(model)
        ll0 = nullloglikelihood(model)
        1 - (ll - k)/ll0
    elseif variant == :devianceratio
        n = nobs(model)
        dev = deviance(model)
        dev0 = nulldeviance(model)
        1 - (dev*(n - (has_int | has_fe(model)))) / (dev0 * max(n - k, 1))
    else
        throw(ArgumentError("variant must be one of :McFadden or :devianceratio"))
    end
end

function StatsAPI.confint(m::IVEstimator; level::Real = 0.95)
    scale = tdistinvcdf(StatsAPI.dof_residual(m), 1 - (1 - level) / 2)
    se = stderror(m)
    hcat(m.coef - scale * se, m.coef + scale * se)
end

##############################################################################
##
## Predict and Residuals
##
##############################################################################

# Note: is_cont_fe_int() and has_cont_fe_interaction() are defined in utils/fit_common.jl

function StatsAPI.predict(m::IVEstimator, data)
    Tables.istable(data) ||
        throw(ArgumentError("expected second argument to be a Table, got $(typeof(data))"))

    has_cont_fe_interaction(m.formula) &&
        throw(ArgumentError("Interaction of fixed effect and continuous variable detected in formula; this is currently not supported in `predict`"))

    cdata = StatsModels.columntable(data)
    cols, nonmissings = StatsModels.missing_omit(cdata, m.formula_schema.rhs)
    Xnew = modelmatrix(m.formula_schema, cols)
    if all(nonmissings)
        out = Xnew * m.coef
    else
        out = Vector{Union{Float64, Missing}}(missing, length(Tables.rows(cdata)))
        out[nonmissings] = Xnew * m.coef
    end

    if has_fe(m)
        nrow(fe(m)) > 0 ||
            throw(ArgumentError("Model has no estimated fixed effects. To store estimates of fixed effects, run `iv` with the option save = :fe"))

        df = DataFrame(data; copycols = false)
        fes = leftjoin(select(df, m.fekeys), dropmissing(unique(m.fe)); on = m.fekeys,
            makeunique = true, matchmissing = :equal, order = :left)
        fes = combine(fes, AsTable(Not(m.fekeys)) => sum)

        if any(ismissing, Matrix(select(df, m.fekeys))) || any(ismissing, Matrix(fes))
            out = allowmissing(out)
        end

        out[nonmissings] .+= fes[nonmissings, 1]

        if any(.!nonmissings)
            out[.!nonmissings] .= missing
        end
    end

    return out
end

function StatsAPI.residuals(m::IVEstimator, data)
    Tables.istable(data) ||
        throw(ArgumentError("expected second argument to be a Table, got $(typeof(data))"))
    has_fe(m) &&
        throw("To access residuals for a model with high-dimensional fixed effects,  run `m = iv(..., save = :residuals)` and then access residuals with `residuals(m)`.")
    cdata = StatsModels.columntable(data)
    cols, nonmissings = StatsModels.missing_omit(cdata, m.formula_schema.rhs)
    Xnew = modelmatrix(m.formula_schema, cols)
    y = response(m.formula_schema, cdata)
    if all(nonmissings)
        out = y - Xnew * m.coef
    else
        out = Vector{Union{Float64, Missing}}(missing, length(Tables.rows(cdata)))
        out[nonmissings] = y - Xnew * m.coef
    end
    return out
end

function StatsAPI.residuals(m::IVEstimator)
    if m.residuals === nothing
        has_fe(m) &&
            throw("To access residuals in a fixed effect regression,  run `iv` with the option save = :residuals, and then access residuals with `residuals()`")
        !has_fe(m) &&
            throw("To access residuals,  use residuals(m, data) where `m` is an estimated IVEstimator and  `data` is a Table")
    end
    m.residuals
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
    out = ["Estimator" _estimator_name(m);
           "Number of obs" sprint(show, nobs(m), context = :compact => true);
           "Converged" m.converged;
           "dof (model)" sprint(show, dof(m), context = :compact => true);
           "dof (residuals)" sprint(show, dof_residual(m), context = :compact => true);
           "R²" @sprintf("%.3f", r2(m));
           "R² adjusted" @sprintf("%.3f", adjr2(m));
           "F-statistic" sprint(show, m.F, context = :compact => true);
           "P-value" @sprintf("%.3f", m.p);]

    # Show kappa for K-class estimators
    if !isnothing(m.postestimation) && !isnothing(m.postestimation.kappa)
        out = vcat(out, ["K-class kappa" @sprintf("%.4f", m.postestimation.kappa)])
    end

    # Always show first-stage diagnostics for IV models (joint Kleibergen-Paap)
    out = vcat(out,
        ["F (1st stage, joint)" sprint(show, m.F_kp, context = :compact => true);
         "P (1st stage, joint)" @sprintf("%.3f", m.p_kp);])
    if has_fe(m)
        out = vcat(out,
            ["R² within" @sprintf("%.3f", m.r2_within);
             "Iterations" sprint(show, m.iterations, context = :compact => true);])
    end
    return out
end

import StatsBase: NoQuote, PValue
function Base.show(io::IO, m::IVEstimator)
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

    ctitle = string(typeof(m))
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

    # Display per-endogenous first-stage F-statistics if available
    _show_per_endogenous_fstats(io, m, totwidth)

    nothing
end

"""
    _show_per_endogenous_fstats(io::IO, m::IVEstimator, totwidth::Int)

Display per-endogenous first-stage F-statistics table.
"""
function _show_per_endogenous_fstats(io::IO, m::IVEstimator, totwidth::Int)
    # Check if we have per-endogenous F-stats
    isempty(m.F_kp_per_endo) && return
    isnothing(m.postestimation) && return
    isnothing(m.postestimation.first_stage_data) && return

    fsd = m.postestimation.first_stage_data
    endo_names = fsd.endogenous_names

    # Print header
    println(io, "\nFirst-Stage F-Statistics (per endogenous variable):")
    println_horizontal_line(io, totwidth)
    @printf(io, "%-30s %14s %14s\n", "Endogenous", "F-stat", "P-value")
    println_horizontal_line(io, totwidth)

    # Print each endogenous variable
    for (j, name) in enumerate(endo_names)
        F_j = m.F_kp_per_endo[j]
        p_j = m.p_kp_per_endo[j]

        # Truncate name if too long
        display_name = length(name) > 28 ? name[1:25] * "..." : name
        @printf(io, "%-30s %14.4f %14.4f\n", display_name, F_j, p_j)
    end

    println_horizontal_line(io, totwidth)
end

function Base.show(io::IO, ::MIME"text/html", m::IVEstimator)
    ct = coeftable(m)
    cols = ct.cols
    rownms = ct.rownms
    colnms = ct.colnms

    # Start table
    html_table_start(io; class="regress-table regress-iv", caption=string(typeof(m)))

    # Summary statistics section
    ctop = top(m)
    html_thead_start(io; class="regress-summary")
    for i in 1:size(ctop, 1)
        html_row(io, [ctop[i, 1], ctop[i, 2]]; class="regress-summary-row")
    end
    html_thead_end(io)

    # Coefficient table header
    html_thead_start(io; class="regress-coef-header")
    html_row(io, vcat([""], colnms); is_header=true)
    html_thead_end(io)

    # Coefficient table body
    html_tbody_start(io; class="regress-coef-body")
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

    # First-stage diagnostics section (if available)
    if !isempty(m.F_kp_per_endo) && !isnothing(m.postestimation) && !isnothing(m.postestimation.first_stage_data)
        fsd = m.postestimation.first_stage_data
        endo_names = fsd.endogenous_names

        html_tfoot_start(io; class="regress-first-stage")
        html_row(io, ["First-Stage F-Statistics", "", "", "", "", "", ""]; class="regress-first-stage-header")
        for (j, name) in enumerate(endo_names)
            F_j = m.F_kp_per_endo[j]
            p_j = m.p_kp_per_endo[j]
            html_row(io, [name, format_number(F_j), format_pvalue(p_j), "", "", "", ""])
        end
        html_tfoot_end(io)
    end

    html_table_end(io)
end

##############################################################################
##
## Schema
##
##############################################################################
function StatsModels.apply_schema(t::FormulaTerm, schema::StatsModels.Schema,
        Mod::Type{IVEstimator}, has_fe_intercept)
    schema = StatsModels.FullRank(schema)
    if has_fe_intercept
        push!(schema.already, InterceptTerm{true}())
    end
    FormulaTerm(apply_schema(t.lhs, schema.schema, StatisticalModel),
        StatsModels.collect_matrix_terms(apply_schema(t.rhs, schema, StatisticalModel)))
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
    isnothing(pe.first_stage_data) && return T(NaN), T(NaN)

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
    isnothing(pe.first_stage_data) && return T[], T[]

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
function Base.:+(m::IVEstimator{T, V1}, v::VcovSpec{V2}) where {T, V1, V2}
    # Compute vcov matrix using StatsBase.vcov (which dispatches to IVModel.jl methods)
    vcov_mat = StatsBase.vcov(v.estimator, m)

    # Compute standard errors
    se = sqrt.(diag(vcov_mat))

    # Compute t-statistics and p-values
    cc = coef(m)
    t_stats = cc ./ se
    p_values = 2 .* tdistccdf.(dof_residual(m), abs.(t_stats))

    # Compute robust F-statistic (Wald test)
    has_int = hasintercept(formula(m))
    F_stat, p_val = compute_robust_fstat(cc, vcov_mat, has_int, dof_residual(m))

    # Recompute first-stage F with this vcov type
    F_kp, p_kp = recompute_first_stage_fstat(m, v.estimator)

    # Recompute per-endogenous F-stats
    F_kp_per_endo, p_kp_per_endo = recompute_per_endogenous_fstats(m, v.estimator)

    # Deep copy the vcov estimator to avoid aliasing
    vcov_copy = deepcopy_vcov(v.estimator)

    # Return new IVEstimator with same data but different vcov type
    return IVEstimator{T, V2}(
        m.estimator, m.coef,
        m.esample, m.residuals, m.fe,
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
function first_stage(m::IVEstimator{T, V}) where {T, V}
    isnothing(m.postestimation) &&
        error("Model does not have post-estimation data stored. Fit with save=true.")
    isnothing(m.postestimation.first_stage_data) && error("First-stage data not available.")

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
    max_name_len = maximum(length, fs.endogenous_names; init=10)
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
    @printf(io, "%-30s %14s %14s\n", "Endogenous", "F-stat", "P-value")
    println_horizontal_line(io, totwidth)

    for (j, name) in enumerate(fs.endogenous_names)
        display_name = length(name) > 28 ? name[1:25] * "..." : name
        @printf(io, "%-30s %14.4f %14.4f\n",
            display_name, fs.F_per_endo[j], fs.p_per_endo[j])
    end

    println_horizontal_line(io, totwidth)
    println(io, "\nInstruments: $(fs.n_instruments) excluded, $(fs.n_endogenous) endogenous")
    print_horizontal_line(io, totwidth)
end

function Base.show(io::IO, ::MIME"text/html", fs::FirstStageResult{T}) where {T}
    html_table_start(io; class="regress-table regress-first-stage",
                     caption="First-Stage Diagnostics ($(fs.vcov_type))")

    # Joint test section
    html_thead_start(io; class="regress-joint-test")
    html_row(io, ["Joint Test (Kleibergen-Paap)", "", ""]; is_header=true)
    html_thead_end(io)
    html_tbody_start(io; class="regress-joint-body")
    html_row(io, ["F-statistic", format_number(fs.F_joint), ""])
    html_row(io, ["P-value", format_pvalue(fs.p_joint), ""])
    html_tbody_end(io)

    # Per-endogenous section
    html_thead_start(io; class="regress-per-endo-header")
    html_row(io, ["Endogenous", "F-stat", "P-value"]; is_header=true)
    html_thead_end(io)
    html_tbody_start(io; class="regress-per-endo-body")
    for (j, name) in enumerate(fs.endogenous_names)
        html_row(io, [name, format_number(fs.F_per_endo[j]), format_pvalue(fs.p_per_endo[j])])
    end
    html_tbody_end(io)

    # Footer
    html_tfoot_start(io; class="regress-footer")
    html_row(io, ["Instruments: $(fs.n_instruments) excluded, $(fs.n_endogenous) endogenous", "", ""])
    html_tfoot_end(io)

    html_table_end(io)
end
