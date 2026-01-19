const CM = CovarianceMatrices
using LinearAlgebra: BLAS

##############################################################################
##
## Moment Matrix Computation (can be overridden by LoopVectorization extension)
##
##############################################################################

"""
    compute_moment_matrix!(M, X, residuals, adjustment) -> M

Compute M = X .* (residuals * adjustment) in-place.
This is the fallback using @simd; RegressLVExt overrides with @turbo.
"""
function compute_moment_matrix!(M::Matrix{T}, X::Matrix{T},
        residuals::Vector{T}, adjustment::T) where {T <: AbstractFloat}
    n, k = size(X)
    @inbounds for j in 1:k
        @simd for i in 1:n
            M[i, j] = X[i, j] * residuals[i] * adjustment
        end
    end
    return M
end

##############################################################################
##
## Helper: Mask vcov matrix with NaN for collinear entries
##
##############################################################################

"""
    mask_vcov_collinear(Σ::AbstractMatrix{T}, basis_coef::BitVector) where {T}

Create a copy of vcov matrix with NaN for collinear entries.
The matrix Σ is expected to be full size (matching length of basis_coef).
Non-collinear entries are preserved; collinear entries are set to NaN.

Uses indexed assignment instead of element-wise loop for better performance.
"""
function mask_vcov_collinear(Σ::AbstractMatrix{T}, basis_coef::BitVector) where {T}
    Σ_out = fill(T(NaN), size(Σ))
    valid_idx = findall(basis_coef)
    Σ_out[valid_idx, valid_idx] = Σ[valid_idx, valid_idx]
    return Σ_out
end

##############################################################################
##
## Direct HC1 Vcov Computation (for use at fit time)
##
##############################################################################

"""
    compute_hc1_vcov_direct(X, residuals, invXX, basis_coef, n, dof_model, dof_fes, dof_residual)

Compute HC1 variance-covariance matrix directly from components.

This function is used at fit time to compute the default vcov without
requiring a fully constructed model object.

# Arguments
- `X::Matrix{T}`: Model matrix (weighted if applicable)
- `residuals::Vector{T}`: Residuals (weighted if applicable)
- `invXX::AbstractMatrix{T}`: Inverse of X'X (from Cholesky or QR)
- `basis_coef::BitVector`: Indicator for non-collinear coefficients
- `n::Int`: Number of observations
- `dof_model::Int`: Degrees of freedom for model parameters
- `dof_fes::Int`: Degrees of freedom absorbed by fixed effects
- `dof_residual::Int`: Residual degrees of freedom

# Returns
- `Symmetric{T, Matrix{T}}`: HC1 variance-covariance matrix
"""
function compute_hc1_vcov_direct(
        X::Matrix{T},
        residuals::Vector{T},
        invXX::AbstractMatrix{T},
        basis_coef::BitVector,
        n::Int,
        dof_model::Int,
        dof_fes::Int,
        dof_residual::Int
) where {T <: AbstractFloat}
    # HR1 residual adjustment: sqrt(n / dof_residual)
    adjustment = sqrt(T(n) / T(dof_residual))

    # Compute moment matrix with adjustment fused: M = X .* (residuals * adjustment)
    # Uses compute_moment_matrix! which can be overridden by LoopVectorization extension
    k = size(X, 2)
    M = Matrix{T}(undef, n, k)
    compute_moment_matrix!(M, X, residuals, adjustment)

    # aVar = M'M / n using BLAS.syrk! for symmetric rank-k update (2x faster)
    aVar_buf = Matrix{T}(undef, k, k)
    BLAS.syrk!('U', 'T', one(T) / T(n), M, zero(T), aVar_buf)
    aVar = Symmetric(aVar_buf, :U)

    # Scale factor for HC1: n * dof_residual / (n - k - k_fe)
    p_total = dof_model + dof_fes
    scale = T(n) * T(dof_residual) / T(n - p_total)

    # Handle collinearity
    if !all(basis_coef)
        k_full = length(basis_coef)
        valid_idx = findall(basis_coef)

        # Extract valid submatrix of aVar
        aVar_valid = aVar[valid_idx, valid_idx]

        # invXX is already reduced size (only non-collinear columns)
        Σ_valid = scale .* invXX * aVar_valid * invXX

        # Expand back to full size with NaN for collinear entries
        Σ = fill(T(NaN), k_full, k_full)
        Σ[valid_idx, valid_idx] = Σ_valid

        return Symmetric(Σ)
    end

    # Standard case: no collinearity
    Σ = scale .* invXX * aVar * invXX
    return Symmetric(Σ)
end

"""
    compute_hc1_vcov_direct_iv(Xhat, residuals, invXX, basis_coef, n, dof_model, dof_fes, dof_residual)

Compute HC1 variance-covariance matrix directly from components for IV models.

This function is used at fit time to compute the default vcov without
requiring a fully constructed model object.

# Arguments
- `Xhat::Matrix{T}`: Model matrix with predicted endogenous (for inference)
- `residuals::Vector{T}`: Residuals
- `invXX::AbstractMatrix{T}`: Inverse of Xhat'Xhat
- `basis_coef::BitVector`: Indicator for non-collinear coefficients
- `n::Int`: Number of observations
- `dof_model::Int`: Degrees of freedom for model parameters
- `dof_fes::Int`: Degrees of freedom absorbed by fixed effects
- `dof_residual::Int`: Residual degrees of freedom

# Returns
- `Symmetric{T, Matrix{T}}`: HC1 variance-covariance matrix
"""
function compute_hc1_vcov_direct_iv(
        Xhat::Matrix{T},
        residuals::Vector{T},
        invXX::AbstractMatrix{T},
        basis_coef::BitVector,
        n::Int,
        dof_model::Int,
        dof_fes::Int,
        dof_residual::Int
) where {T <: AbstractFloat}
    # NOTE: For IV models, Xhat is already reduced (collinear columns removed)
    # So aVar will also be reduced-size: (sum(basis_coef) x sum(basis_coef))
    # invXX is also reduced-size: (sum(basis_coef) x sum(basis_coef))
    # basis_coef is full-size: length = original number of coefficients

    # Compute moment matrix: M = Xhat .* residuals
    M = Xhat .* residuals

    # HR1 residual adjustment: sqrt(n / dof_residual)
    adjustment = sqrt(T(n) / T(dof_residual))
    M .*= adjustment

    # aVar = M'M / n using BLAS.syrk! for symmetric rank-k update (2x faster)
    k = size(M, 2)
    aVar_buf = Matrix{T}(undef, k, k)
    BLAS.syrk!('U', 'T', one(T) / T(n), M, zero(T), aVar_buf)
    aVar = Symmetric(aVar_buf, :U)

    # Scale factor for HC1: n * dof_residual / (n - k - k_fe)
    p_total = dof_model + dof_fes
    scale = T(n) * T(dof_residual) / T(n - p_total)

    # Handle collinearity - expand result back to full size
    if !all(basis_coef)
        k_full = length(basis_coef)
        valid_idx = findall(basis_coef)

        # aVar is already reduced-size (computed from reduced Xhat)
        # invXX is already reduced-size
        # Just compute the sandwich and expand
        Σ_valid = scale .* invXX * aVar * invXX

        # Expand back to full size with NaN for collinear entries
        Σ = fill(T(NaN), k_full, k_full)
        Σ[valid_idx, valid_idx] = Σ_valid

        return Symmetric(Σ)
    end

    # Standard case: no collinearity
    Σ = scale .* invXX * aVar * invXX
    return Symmetric(Σ)
end

##############################################################################
##
## CovarianceMatrices.jl Interface for Post-Estimation vcov
##
##############################################################################

"""
    CovarianceMatrices.momentmatrix(m::OLSEstimator)

Returns the moment matrix for the model (X .* residuals).
Required for post-estimation variance-covariance calculations.
"""
function CovarianceMatrices.momentmatrix(m::OLSEstimator)
    # Get residuals and model matrix from new structure
    resid = residuals(m)
    X = modelmatrix(m)
    return X .* resid
end

function CM.aVar(
        k::K,
        m::OLSEstimator;
        demean = false,
        prewhite = false,
        scale = true,
        kwargs...
) where {K <: CM.AbstractAsymptoticVarianceEstimator}
    CM.setkernelweights!(k, m)
    # Compute moment matrix directly: X .* (y - mu) .* u in single fused broadcast
    # This avoids separate allocation for residuals vector
    # Note: y and mu are already weighted if model has weights
    u = residualadjustment(k, m)
    X = modelmatrix(m)
    y = m.rr.y
    mu = m.rr.mu
    mm = @. X * (y - mu) * u
    basis_coef = m.basis_coef
    Σ = aVar(k, mm; demean = demean, prewhite = prewhite, scale = scale)

    all(basis_coef) && return Σ
    return mask_vcov_collinear(Σ, basis_coef)
end

# Disambiguating method for cluster-robust estimators (CR <: AbstractAsymptoticVarianceEstimator)
# This resolves ambiguity between:
#   - aVar(k::K, m::OLSEstimator) where K <: AbstractAsymptoticVarianceEstimator (above)
#   - aVar(k::CR, m::RegressionModel) from CovarianceMatrices.jl
function CM.aVar(
        k::K,
        m::OLSEstimator;
        demean = false,
        prewhite = false,
        scale = true,
        kwargs...
) where {K <: CM.CR}
    # Compute moment matrix directly: X .* (y - mu) .* u in single fused broadcast
    # This avoids separate allocation for residuals vector
    u = residualadjustment(k, m)
    X = modelmatrix(m)
    y = m.rr.y
    mu = m.rr.mu
    mm = @. X * (y - mu) * u
    basis_coef = m.basis_coef
    Σ = aVar(k, mm; demean = demean, prewhite = prewhite, scale = scale)

    all(basis_coef) && return Σ
    return mask_vcov_collinear(Σ, basis_coef)
end

function CM.setkernelweights!(
        k::CM.HAC{T},
        X::OLSEstimator
) where {T <: Union{CM.NeweyWest, CM.Andrews}}
    CM.setkernelweights!(k, modelmatrix(X))
    k.wlock .= true
end

##############################################################################
##
## Cluster-Robust Variance Estimation
##
## Standard CovarianceMatrices.jl API (with actual data vectors):
##   vcov(CR1(cluster_vec), model)
##   stderror(CR1(cluster_vec), model)
##
## Symbol-based API (looks up stored cluster variables):
##   vcov(CR1(:StateID), model)           # single cluster
##   vcov(CR1(:StateID, :YearID), model)  # multi-way clustering
##
##############################################################################

"""
    stderror(ve::CovarianceMatrices.AbstractAsymptoticVarianceEstimator, m::OLSEstimator)

Compute standard errors using a specified variance estimator.

# Examples
```julia
# Heteroskedasticity-robust
stderror(HC1(), model)

# Cluster-robust (standard CovarianceMatrices.jl API)
stderror(CR1(cluster_vec), model)

# Two-way clustering
stderror(CR1((cluster1, cluster2)), model)
```
"""
function StatsBase.stderror(ve::CovarianceMatrices.AbstractAsymptoticVarianceEstimator, m::OLSEstimator)
    return sqrt.(diag(vcov(ve, m)))
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
    _lookup_cluster_vecs(cluster_syms::Tuple{Vararg{Symbol}}, m::OLSEstimator)

Look up cluster vectors from stored cluster data in the model.
Returns a tuple of vectors corresponding to the requested cluster symbols.
"""
function _lookup_cluster_vecs(cluster_syms::Tuple{Vararg{Symbol}}, m::OLSEstimator)
    return Tuple(begin
                     haskey(m.fes.clusters, name) || _cluster_not_found_error(name, m)
                     m.fes.clusters[name]
                 end
    for name in cluster_syms)
end

"""
    vcov(v::CM.CR0{T}, m::OLSEstimator) where T<:Tuple{Vararg{Symbol}}

Compute cluster-robust variance with CR0 estimator using stored cluster variable(s).

# Examples
```julia
model = ols(df, @formula(y ~ x), save_cluster = :firm_id)
vcov(CR0(:firm_id), model)
vcov(CR0(:firm_id, :year), model)  # multi-way
```
"""
function CM.vcov(v::CM.CR0{T}, m::OLSEstimator) where {T <: Tuple{Vararg{Symbol}}}
    cluster_vecs = _lookup_cluster_vecs(v.g, m)
    return vcov(CM.CR0(cluster_vecs), m)
end

"""
    vcov(v::CM.CR1{T}, m::OLSEstimator) where T<:Tuple{Vararg{Symbol}}

Compute cluster-robust variance with CR1 estimator using stored cluster variable(s).

# Examples
```julia
model = ols(df, @formula(y ~ x), save_cluster = :firm_id)
vcov(CR1(:firm_id), model)
vcov(CR1(:firm_id, :year), model)  # multi-way
```
"""
function CM.vcov(v::CM.CR1{T}, m::OLSEstimator) where {T <: Tuple{Vararg{Symbol}}}
    cluster_vecs = _lookup_cluster_vecs(v.g, m)
    return vcov(CM.CR1(cluster_vecs), m)
end

"""
    vcov(v::CM.CR2{T}, m::OLSEstimator) where T<:Tuple{Vararg{Symbol}}

Compute cluster-robust variance with CR2 (leverage-adjusted) estimator using stored cluster variable(s).
"""
function CM.vcov(v::CM.CR2{T}, m::OLSEstimator) where {T <: Tuple{Vararg{Symbol}}}
    cluster_vecs = _lookup_cluster_vecs(v.g, m)
    return vcov(CM.CR2(cluster_vecs), m)
end

"""
    vcov(v::CM.CR3{T}, m::OLSEstimator) where T<:Tuple{Vararg{Symbol}}

Compute cluster-robust variance with CR3 (squared leverage) estimator using stored cluster variable(s).
"""
function CM.vcov(v::CM.CR3{T}, m::OLSEstimator) where {T <: Tuple{Vararg{Symbol}}}
    cluster_vecs = _lookup_cluster_vecs(v.g, m)
    return vcov(CM.CR3(cluster_vecs), m)
end

##############################################################################
##
## CachedCR Support for Symbol-Based Cluster Lookup
##
## CachedCR provides preallocated buffers for fast repeated variance calculations
## (e.g., wild bootstrap). These methods support the symbol-based API.
##
## Usage:
##   # Create CachedCR from model (resolves symbol to actual cluster data)
##   cached = CachedCR(CR1(:firm_id), model)
##
##   # Use in wild bootstrap
##   for b in 1:1000
##       m_boot = model + vcov(cached)
##   end
##
##############################################################################

"""
    CachedCR(cr::CR, m::OLSEstimator)

Create a CachedCR estimator by resolving symbol-based cluster specification
from the model's stored cluster data.

This is the recommended way to create a CachedCR for use with Regress.jl models,
as it handles symbol-to-vector resolution and determines the correct ncols.

# Examples
```julia
# Fit model with cluster data saved
m = ols(df, @formula(y ~ x1 + x2), save_cluster = :firm_id)

# Create CachedCR from symbol-based CR estimator
cached = CachedCR(CR1(:firm_id), m)

# Use in wild bootstrap loop
for b in 1:1000
    # ... perturb model ...
    V = vcov(cached, m)
end
```

See also: [`CachedCR`](@ref CovarianceMatrices.CachedCR)
"""
function CM.CachedCR(cr::CM.CR0{T}, m::OLSEstimator) where {T <: Tuple{Vararg{Symbol}}}
    cluster_vecs = _lookup_cluster_vecs(cr.g, m)
    ncols = sum(m.basis_coef)  # Number of non-collinear coefficients
    return CM.CachedCR(CM.CR0(cluster_vecs), ncols)
end

function CM.CachedCR(cr::CM.CR1{T}, m::OLSEstimator) where {T <: Tuple{Vararg{Symbol}}}
    cluster_vecs = _lookup_cluster_vecs(cr.g, m)
    ncols = sum(m.basis_coef)
    return CM.CachedCR(CM.CR1(cluster_vecs), ncols)
end

function CM.CachedCR(cr::CM.CR2{T}, m::OLSEstimator) where {T <: Tuple{Vararg{Symbol}}}
    cluster_vecs = _lookup_cluster_vecs(cr.g, m)
    ncols = sum(m.basis_coef)
    return CM.CachedCR(CM.CR2(cluster_vecs), ncols)
end

function CM.CachedCR(cr::CM.CR3{T}, m::OLSEstimator) where {T <: Tuple{Vararg{Symbol}}}
    cluster_vecs = _lookup_cluster_vecs(cr.g, m)
    ncols = sum(m.basis_coef)
    return CM.CachedCR(CM.CR3(cluster_vecs), ncols)
end

# For CR with data vectors, just compute ncols from model
function CM.CachedCR(cr::CM.CR, m::OLSEstimator)
    ncols = sum(m.basis_coef)
    return CM.CachedCR(cr, ncols)
end

"""
    vcov(v::CM.CachedCR, m::OLSEstimator)

Compute cluster-robust variance using CachedCR estimator.

# Examples
```julia
# Create CachedCR from model
cached = CachedCR(CR1(:firm_id), model)

# Compute vcov
V = vcov(cached, model)

# Or use with + operator
model_robust = model + vcov(cached)
```
"""
function CM.vcov(v::CM.CachedCR, m::OLSEstimator)
    # Compute moment matrix: X .* (y - mu)
    # Note: For CachedCR, we use uniform residual adjustment (same as CR0/CR1)
    X = modelmatrix(m)
    y = m.rr.y
    mu = m.rr.mu
    mm = @. X * (y - mu)

    # Get bread matrix and number of observations
    B = bread(m)
    n = nobs(m)
    basis_coef = m.basis_coef

    # Use cached aVar computation
    Σ = aVar(v, mm)

    # Compute scale factor using fixest-style small sample correction
    # This matches the standard vcov(CR, OLSEstimator) method
    scale = _cluster_robust_scale(v.estimator, m, n)

    # Apply sandwich: V = scale * B * Σ * B
    vcov_mat = scale .* B * Σ * B

    all(basis_coef) && return Symmetric(vcov_mat)
    return mask_vcov_collinear(Symmetric(vcov_mat), basis_coef)
end

"""
    aVar(k::CM.CachedCR, m::OLSEstimator; kwargs...)

Compute asymptotic variance using CachedCR estimator.
"""
function CM.aVar(
        k::CM.CachedCR,
        m::OLSEstimator;
        demean = false,
        prewhite = false,
        scale = true,
        kwargs...
)
    # Compute moment matrix directly
    X = modelmatrix(m)
    y = m.rr.y
    mu = m.rr.mu
    mm = @. X * (y - mu)
    basis_coef = m.basis_coef

    Σ = aVar(k, mm; demean = demean, prewhite = prewhite, scale = scale)

    all(basis_coef) && return Σ
    return mask_vcov_collinear(Σ, basis_coef)
end

# Residual adjustment for CachedCR (same as CR0/CR1 - no adjustment)
@noinline residualadjustment(k::CM.CachedCR, r::OLSEstimator) = 1.0

##############################################################################
##
## Helper Functions for Cluster Variable Handling
##
##############################################################################

# Error message for missing cluster variable
function _cluster_not_found_error(cluster_name::Symbol, m::OLSEstimator)
    available = isempty(m.fes.clusters) ? "none" : join(keys(m.fes.clusters), ", :")
    error("""
    Cluster variable :$cluster_name not found in model.

    Available cluster variables: :$available

    To use this cluster variable, either:
      1. Re-fit with save_cluster=:$cluster_name
      2. Use data directly: vcov(CR1(df.$cluster_name[model.esample]), model)
    """)
end

"""
    bread(m::OLSEstimator)

Compute (X'X)^(-1), the "bread" of the sandwich variance estimator.
"""
bread(m::OLSEstimator) = invchol(m.pp)

"""
    leverage(m::OLSEstimator)

Compute leverage values (diagonal of hat matrix H = X(X'X)^(-1)X').
Uses h_i = ||X_i * U^(-1)||^2 where X'X = U'U for efficiency.
"""
function leverage(m::OLSEstimator{T, <:OLSPredictorChol}) where {T}
    # For Cholesky: X'X = U'U, so (X'X)^(-1) = U^(-1) * U^(-T)
    # h_i = X_i * U^(-1) * U^(-T) * X_i' = ||X_i * U^(-1)||^2
    X = modelmatrix(m)
    # Compute X / U and use column-major friendly sum
    # XU = X / U is n×k, we want row-wise sum of squares
    # Transpose to k×n for column-major access: sum(abs2, XU', dims=1)
    XU = X / m.pp.chol.U
    return vec(sum(abs2, XU', dims = 1))
end

function leverage(m::OLSEstimator{T, <:OLSPredictorQR}) where {T}
    # For QR: X = QR, so (X'X)^(-1) = R^(-1) R^(-T)
    # H = X(X'X)^(-1)X' and h_i = X_i (X'X)^(-1) X_i' = ||X_i R^(-1)||^2
    # Using R-factor directly avoids materializing full n×n Q matrix
    X = modelmatrix(m)
    R = m.pp.qr.R
    # Solve X / R using forward substitution: O(nk²) vs O(n²k) for full Q
    XRinv = X / UpperTriangular(R)
    return vec(sum(abs2, XRinv, dims = 2))
end

@noinline residualadjustment(k::CM.HR0, r::OLSEstimator) = 1.0
@noinline residualadjustment(k::CM.HR1, r::OLSEstimator) = sqrt(nobs(r) / dof_residual(r))
@noinline residualadjustment(k::CM.HR2, r::OLSEstimator) = @. 1.0 / sqrt(1.0 - $leverage(r))
@noinline residualadjustment(k::CM.HR3, r::OLSEstimator) = @. 1.0 / (1.0 - $leverage(r))

@noinline function residualadjustment(k::CM.HR4, r::OLSEstimator)
    n = nobs(r)
    h = leverage(r)
    p = round(Int, sum(h))
    @inbounds for j in eachindex(h)
        delta = min(4.0, n * h[j] / p)
        h[j] = 1 / (1 - h[j])^(delta / 2)
    end
    h
end

@noinline function residualadjustment(k::CM.HR4m, r::OLSEstimator)
    n = nobs(r)
    h = leverage(r)
    p = round(Int, sum(h))
    @inbounds for j in eachindex(h)
        delta = min(1, n * h[j] / p) + min(1.5, n * h[j] / p)
        h[j] = 1 / (1 - h[j])^(delta / 2)
    end
    h
end

@noinline function residualadjustment(k::CM.HR5, r::OLSEstimator)
    n = nobs(r)
    h = leverage(r)
    p = round(Int, sum(h))
    mx = max(n * 0.7 * maximum(h) / p, 4.0)
    @inbounds for j in eachindex(h)
        alpha = min(n * h[j] / p, mx)
        h[j] = 1 / (1 - h[j])^(alpha / 4)
    end
    return h
end

# For cluster-robust estimators CR0/CR1, no adjustment to moment matrix needed.
# The clustering is handled by CovarianceMatrices.aVar itself.
@noinline residualadjustment(k::CM.CR0, r::OLSEstimator) = 1.0
@noinline residualadjustment(k::CM.CR1, r::OLSEstimator) = 1.0

# HAC (kernel) estimators - no residual adjustment needed
@noinline residualadjustment(k::CM.HAC, r::OLSEstimator) = 1.0

"""
    _get_group_ranges(g)

Precompute group indices for efficient per-group iteration.
Returns (perm, starts) where perm[starts[i]:(starts[i+1]-1)] gives indices for group i.

This avoids O(n × G) complexity from calling findall for each group.
"""
function _get_group_ranges(g)
    groups = g.groups
    ngroups = g.ngroups
    n = length(groups)

    # Count elements per group
    counts = zeros(Int, ngroups)
    @inbounds for i in 1:n
        counts[groups[i]] += 1
    end

    # Compute starting positions (cumulative sum + 1)
    starts = Vector{Int}(undef, ngroups + 1)
    starts[1] = 1
    @inbounds for i in 1:ngroups
        starts[i + 1] = starts[i] + counts[i]
    end

    # Fill in permutation array
    perm = Vector{Int}(undef, n)
    pos = copy(starts)
    @inbounds for i in 1:n
        gid = groups[i]
        perm[pos[gid]] = i
        pos[gid] += 1
    end

    return perm, starts
end

function residualadjustment(k::CM.CR2, r::OLSEstimator)
    wts = r.rr.wts
    @assert length(k.g) == 1
    g = k.g[1]
    X = modelmatrix(r)
    u_orig = residuals(r)
    u = copy(u_orig)
    !isempty(wts) && @. u *= sqrt(wts)
    XX = bread(r)

    # Precompute group ranges once instead of O(n) findall per group
    perm, starts = _get_group_ranges(g)

    for group_id in 1:g.ngroups
        ind = @view perm[starts[group_id]:(starts[group_id + 1] - 1)]
        Xg = view(X, ind, :)
        ug = view(u, ind, :)
        if isempty(wts)
            Hᵧᵧ = (Xg * XX * Xg')
            ldiv!(ug, cholesky!(Symmetric(I - Hᵧᵧ); check = false).L, ug)
        else
            Hᵧᵧ = (Xg * XX * Xg') .* view(wts, ind)'
            ug .= matrixpowbysvd(I - Hᵧᵧ, -0.5) * ug
        end
    end
    # Return the adjustment factor: adjusted_u / original_u
    # So that M = (X .* u_orig) .* factor = X .* adjusted_u
    return u ./ u_orig
end

function matrixpowbysvd(A, p; tol = eps()^(1 / 1.5))
    s = svd(A)
    V = s.S
    V[V .< tol] .= 0
    return s.V * diagm(0 => V .^ p) * s.Vt
end

function residualadjustment(k::CM.CR3, r::OLSEstimator)
    wts = r.rr.wts
    @assert length(k.g) == 1
    g = k.g[1]
    X = modelmatrix(r)
    u_orig = residuals(r)
    u = copy(u_orig)
    !isempty(wts) && @. u *= sqrt(wts)
    XX = bread(r)

    # Precompute group ranges once instead of O(n) findall per group
    perm, starts = _get_group_ranges(g)

    for group_id in 1:g.ngroups
        ind = @view perm[starts[group_id]:(starts[group_id + 1] - 1)]
        Xg = view(X, ind, :)
        ug = view(u, ind, :)
        if isempty(wts)
            Hᵧᵧ = (Xg * XX * Xg')
            ldiv!(ug, cholesky!(Symmetric(I - Hᵧᵧ); check = false), ug)
        else
            Hᵧᵧ = (Xg * XX * Xg') .* view(wts, ind)'
            ug .= (I - Hᵧᵧ)^(-1) * ug
        end
    end
    # Return the adjustment factor: adjusted_u / original_u
    return u ./ u_orig
end

##############################################################################
##
## Cluster-robust small sample correction (fixest-compatible)
##
##############################################################################

"""
    _cluster_robust_scale(k::CM.CR, m::OLSEstimator, n::Int)

Compute the scale factor for cluster-robust variance estimation using
fixest-style small sample correction.

# Formula
`scale = n * G/(G-1) * (n-1)/(n-K)` where:
- G = number of clusters (for multi-way: minimum cluster count)
- K = k + k_fe_nonnested (only FE NOT nested in cluster are counted)

With K.fixef = "nonnested" (fixest default):
- FE nested in the cluster variable are NOT counted in K
- FE NOT nested in the cluster variable ARE counted in K

This matches R fixest's default behavior for cluster-robust standard errors.
"""
function _cluster_robust_scale(k::CM.CR, m::OLSEstimator, n::Int)
    # Get cluster groupings from the CR estimator
    cluster_groups = k.g

    # G = number of clusters (for multi-way, use minimum)
    G = minimum(g.ngroups for g in cluster_groups)

    # G/(G-1) adjustment (cluster DOF correction) - only for CR1, CR2, CR3
    # CR0 has no small-sample adjustment
    G_adj = k isa CM.CR0 ? 1.0 : G / (G - 1)

    # Compute K for (n-1)/(n-K) adjustment using K.fixef = "nonnested"
    # K = k (non-FE params) + FE DOF for FE not nested in cluster
    # For CR0, we still apply the (n-1)/(n-K) adjustment for consistency
    k_params = dof(m)
    k_fe_nonnested = _compute_nonnested_fe_dof(m, cluster_groups)
    K = k_params + k_fe_nonnested

    # (n-1)/(n-K) adjustment (parameter DOF correction)
    # For CR0, this is the only adjustment applied
    K_adj = (n - 1) / (n - K)

    # Final scale: n * G_adj * K_adj
    # The n factor is because aVar returns (sum)/n, and vcov computes scale * B * A * B
    return convert(Float64, n * G_adj * K_adj)
end

"""
    _compute_nonnested_fe_dof(m::OLSEstimator, cluster_groups)

Compute the DOF for fixed effects that are NOT nested in the cluster variable(s).

A fixed effect is "nested" in a cluster if every FE group is contained within
exactly one cluster group. When FE is nested, it doesn't add information beyond
the clustering and shouldn't be counted in the K adjustment.

Returns 0 if all FE are nested in the clustering, or the full k_fe if none are nested.
"""
function _compute_nonnested_fe_dof(m::OLSEstimator, cluster_groups)
    # If no fixed effects, return 0
    dof_fes(m) == 0 && return 0

    # Get FE names from model
    fe_names = m.fes.fe_names
    isempty(fe_names) && return 0

    # Get cluster variable names from the stored clusters
    cluster_names = keys(m.fes.clusters)

    # Check if each FE is nested in at least one cluster
    # For simplicity, we use a name-matching heuristic:
    # FE is nested if its name matches one of the cluster names
    # This is a reasonable approximation for most use cases
    k_fe_nonnested = 0

    # For now, use a simple approach: if FE name == cluster name, it's nested
    # A more sophisticated approach would check actual nesting of the groupings
    for fe_name in fe_names
        is_nested = fe_name in cluster_names
        if !is_nested
            # This FE is not nested in any cluster - count its DOF
            # We need to compute the DOF for just this FE
            # For simplicity, assume each FE contributes proportionally
            # In practice, this is conservative (may overcount)
            k_fe_nonnested += _fe_dof_for_name(m, fe_name)
        end
    end

    return k_fe_nonnested
end

"""
    _fe_dof_for_name(m::OLSEstimator, fe_name::Symbol)

Get the degrees of freedom absorbed by a specific fixed effect.
For models with a single FE, this is just dof_fes(m).
For multiple FE, we estimate based on the number of levels.
"""
function _fe_dof_for_name(m::OLSEstimator, fe_name::Symbol)
    fe_names = m.fes.fe_names

    # Single FE case - return all FE DOF
    length(fe_names) == 1 && return dof_fes(m)

    # Multiple FE case - we don't have per-FE DOF stored
    # As an approximation, split evenly (this is conservative)
    # A better approach would store per-FE DOF during fitting
    return dof_fes(m) ÷ length(fe_names)
end

function CM.vcov(k::CM.AbstractAsymptoticVarianceEstimator, m::OLSEstimator; dofadjust = true, kwargs...)
    A = aVar(k, m; kwargs...)
    n = nobs(m)
    B = invchol(m.pp)
    basis_coef = m.basis_coef

    # The aVar function returns M'M/n (where M is the adjusted moment matrix).
    # For HR1, residualadjustment = √(n/dof_residual), so the adjusted moment matrix
    # already incorporates part of the DOF adjustment.
    #
    # For HC0/HR0: V = n * B * (M'M/n) * B = B * M'M * B
    # For HC1/HR1: V = n/(n-k-k_fe) * B * M'M * B  (with proper DOF)
    # For cluster-robust: Use fixest-style small sample correction

    scale = if k isa Union{CM.HC1, CM.HR1}
        # HC1: DOF adjustment should account for both k and k_fe
        # residualadjustment for HR1 already applied √(n/dof_residual)
        # So aVar returns M'M * (n/dof_residual) / n = M'M / dof_residual
        # We want final scale = n / (n - k - k_fe)
        # With aVar = M'M / dof_residual and dof_residual ≈ n - k - k_fe - 1,
        # we need: scale * (1/dof_residual) = n / (n - k - k_fe)
        # => scale = n * dof_residual / (n - k - k_fe)
        p_total = dof(m) + dof_fes(m)
        n * dof_residual(m) / (n - p_total)
    elseif k isa Union{CM.CR0, CM.CR1, CM.CR2, CM.CR3}
        # Cluster-robust: Use fixest-style small sample correction
        # Formula: G/(G-1) * (n-1)/(n-K) where:
        #   - G = number of clusters (for multi-way: minimum cluster count)
        #   - K = k + k_fe_nonnested (FE not nested in cluster are counted)
        # With K.fixef = "nonnested" (fixest default):
        #   - FE nested in cluster variable are NOT counted in K
        #   - FE NOT nested in cluster variable ARE counted in K
        _cluster_robust_scale(k, m, n)
    else
        # HC0/HR0: no DOF adjustment, scale = n
        convert(eltype(A), n)
    end

    # Handle dimension mismatch when there is collinearity:
    # - A is k×k (full size, with NaN for collinear entries)
    # - B is k_reduced×k_reduced (from factorization on non-collinear columns)
    # We need to extract the valid submatrix, compute sandwich, then expand back
    if !all(basis_coef)
        k_full = length(basis_coef)
        valid_idx = findall(basis_coef)

        # Extract valid submatrix of A
        A_valid = A[valid_idx, valid_idx]

        # Compute sandwich on reduced dimensions
        Σ_valid = scale .* B * A_valid * B

        # Expand back to full size with NaN for collinear entries
        T = eltype(Σ_valid)
        Σ = fill(T(NaN), k_full, k_full)
        Σ[valid_idx, valid_idx] = Σ_valid

        return Σ
    end

    Σ = scale .* B * A * B

    return Σ
end

##############################################################################
##
## CovarianceMatrices.jl Interface for OLSMatrixEstimator
##
##############################################################################

"""
    CovarianceMatrices.momentmatrix(m::OLSMatrixEstimator)

Returns the moment matrix for the model (X .* residuals).
Required for post-estimation variance-covariance calculations.
"""
function CovarianceMatrices.momentmatrix(m::OLSMatrixEstimator)
    resid = residuals(m)
    X = modelmatrix(m)
    return X .* resid
end

"""
    bread(m::OLSMatrixEstimator)

Compute (X'X)^(-1), the "bread" of the sandwich variance estimator.
"""
bread(m::OLSMatrixEstimator) = invchol(m.pp)

"""
    leverage(m::OLSMatrixEstimator)

Compute leverage values (diagonal of hat matrix H = X(X'X)^(-1)X').
"""
function leverage(m::OLSMatrixEstimator{T, <:OLSPredictorChol}) where {T}
    X = modelmatrix(m)
    # Transpose for column-major access: sum(abs2, XU', dims=1)
    XU = X / m.pp.chol.U
    return vec(sum(abs2, XU', dims = 1))
end

function leverage(m::OLSMatrixEstimator{T, <:OLSPredictorQR}) where {T}
    # For QR: (X'X)^(-1) = R^(-1) R^(-T)
    # h_i = ||X_i R^(-1)||^2, using R-factor directly avoids materializing full Q
    X = modelmatrix(m)
    R = m.pp.qr.R
    XRinv = X / UpperTriangular(R)
    return vec(sum(abs2, XRinv, dims = 2))
end

# Residual adjustment functions for OLSMatrixEstimator
@noinline residualadjustment(k::CM.HR0, r::OLSMatrixEstimator) = 1.0
@noinline residualadjustment(k::CM.HR1, r::OLSMatrixEstimator) = sqrt(nobs(r) /
                                                                      dof_residual(r))
@noinline residualadjustment(k::CM.HR2, r::OLSMatrixEstimator) = @. 1.0 /
                                                                    sqrt(1.0 - $leverage(r))
@noinline residualadjustment(k::CM.HR3, r::OLSMatrixEstimator) = @. 1.0 /
                                                                    (1.0 - $leverage(r))

@noinline function residualadjustment(k::CM.HR4, r::OLSMatrixEstimator)
    n = nobs(r)
    h = leverage(r)
    p = round(Int, sum(h))
    @inbounds for j in eachindex(h)
        delta = min(4.0, n * h[j] / p)
        h[j] = 1 / (1 - h[j])^(delta / 2)
    end
    h
end

@noinline function residualadjustment(k::CM.HR4m, r::OLSMatrixEstimator)
    n = nobs(r)
    h = leverage(r)
    p = round(Int, sum(h))
    @inbounds for j in eachindex(h)
        delta = min(1, n * h[j] / p) + min(1.5, n * h[j] / p)
        h[j] = 1 / (1 - h[j])^(delta / 2)
    end
    h
end

@noinline function residualadjustment(k::CM.HR5, r::OLSMatrixEstimator)
    n = nobs(r)
    h = leverage(r)
    p = round(Int, sum(h))
    mx = max(n * 0.7 * maximum(h) / p, 4.0)
    @inbounds for j in eachindex(h)
        alpha = min(n * h[j] / p, mx)
        h[j] = 1 / (1 - h[j])^(alpha / 4)
    end
    return h
end

# Cluster-robust and HAC estimators - no residual adjustment
@noinline residualadjustment(k::CM.CR0, r::OLSMatrixEstimator) = 1.0
@noinline residualadjustment(k::CM.CR1, r::OLSMatrixEstimator) = 1.0
@noinline residualadjustment(k::CM.HAC, r::OLSMatrixEstimator) = 1.0

function CM.aVar(
        k::K,
        m::OLSMatrixEstimator;
        demean = false,
        prewhite = false,
        scale = true,
        kwargs...
) where {K <: CM.AbstractAsymptoticVarianceEstimator}
    CM.setkernelweights!(k, m)
    # Compute moment matrix directly: X .* (y - mu) .* u in single fused broadcast
    u = residualadjustment(k, m)
    X = modelmatrix(m)
    y = m.rr.y
    mu = m.rr.mu
    mm = @. X * (y - mu) * u
    basis_coef = m.basis_coef
    Σ = aVar(k, mm; demean = demean, prewhite = prewhite, scale = scale)

    all(basis_coef) && return Σ
    return mask_vcov_collinear(Σ, basis_coef)
end

# Disambiguating method for cluster-robust estimators
function CM.aVar(
        k::K,
        m::OLSMatrixEstimator;
        demean = false,
        prewhite = false,
        scale = true,
        kwargs...
) where {K <: CM.CR}
    # Compute moment matrix directly: X .* (y - mu) .* u in single fused broadcast
    u = residualadjustment(k, m)
    X = modelmatrix(m)
    y = m.rr.y
    mu = m.rr.mu
    mm = @. X * (y - mu) * u
    basis_coef = m.basis_coef
    Σ = aVar(k, mm; demean = demean, prewhite = prewhite, scale = scale)

    all(basis_coef) && return Σ
    return mask_vcov_collinear(Σ, basis_coef)
end

function CM.setkernelweights!(
        k::CM.HAC{T},
        X::OLSMatrixEstimator
) where {T <: Union{CM.NeweyWest, CM.Andrews}}
    CM.setkernelweights!(k, modelmatrix(X))
    k.wlock .= true
end

"""
    stderror(ve::CovarianceMatrices.AbstractAsymptoticVarianceEstimator, m::OLSMatrixEstimator)

Compute standard errors using a specified variance estimator.
"""
function StatsBase.stderror(ve::CovarianceMatrices.AbstractAsymptoticVarianceEstimator, m::OLSMatrixEstimator)
    return sqrt.(diag(vcov(ve, m)))
end

function CM.vcov(k::CM.AbstractAsymptoticVarianceEstimator, m::OLSMatrixEstimator; dofadjust = true, kwargs...)
    A = aVar(k, m; kwargs...)
    n = nobs(m)
    B = invchol(m.pp)
    basis_coef = m.basis_coef

    scale = if k isa Union{CM.HC1, CM.HR1}
        # HC1: DOF adjustment
        p_total = dof(m)
        n * dof_residual(m) / (n - p_total)
    elseif k isa Union{CM.CR0, CM.CR1, CM.CR2, CM.CR3}
        # Cluster-robust: simple G/(G-1) * (n-1)/(n-K) adjustment
        # For matrix estimator, no fixed effects so K = k
        G = minimum(g.ngroups for g in k.g)
        G_adj = k isa CM.CR0 ? 1.0 : G / (G - 1)
        K = dof(m)
        K_adj = (n - 1) / (n - K)
        convert(Float64, n * G_adj * K_adj)
    else
        # HC0/HR0: no DOF adjustment, scale = n
        convert(eltype(A), n)
    end

    # Handle collinearity
    if !all(basis_coef)
        k_full = length(basis_coef)
        valid_idx = findall(basis_coef)
        A_valid = A[valid_idx, valid_idx]
        Σ_valid = scale .* B * A_valid * B
        T = eltype(Σ_valid)
        Σ = fill(T(NaN), k_full, k_full)
        Σ[valid_idx, valid_idx] = Σ_valid
        return Σ
    end

    return scale .* B * A * B
end

function StatsAPI.confint(ve::CovarianceMatrices.AbstractAsymptoticVarianceEstimator,
        m::OLSMatrixEstimator; level::Real = 0.95)
    scale = tdistinvcdf(StatsAPI.dof_residual(m), 1 - (1 - level) / 2)
    se = CovarianceMatrices.stderror(ve, m)
    hcat(coef(m) - scale * se, coef(m) + scale * se)
end
