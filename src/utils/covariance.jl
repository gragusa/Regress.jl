const CM = CovarianceMatrices

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
    # Compute moment matrix: M = X .* residuals
    M = X .* residuals

    # HR1 residual adjustment: sqrt(n / dof_residual)
    adjustment = sqrt(T(n) / T(dof_residual))
    M .*= adjustment

    # aVar = M'M / n
    aVar = (M' * M) / T(n)

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

    # aVar = M'M / n (reduced-size since Xhat is reduced)
    aVar = (M' * M) / T(n)

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
    mm = begin
        u = residualadjustment(k, m)
        M = copy(momentmatrix(m))
        @. M = M * u
        M
    end
    basis_coef = m.basis_coef
    Σ = aVar(k, mm; demean = demean, prewhite = prewhite, scale = scale)

    all(basis_coef) && return Σ

    Σ_out = similar(Σ)
    fill!(Σ_out, NaN)
    for j in axes(Σ, 1)
        for i in axes(Σ, 2)
            if basis_coef[j] && basis_coef[i]
                Σ_out[j, i] = Σ[j, i]
            end
        end
    end
    return Σ_out
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
    mm = begin
        u = residualadjustment(k, m)
        M = copy(momentmatrix(m))
        @. M = M * u
        M
    end
    basis_coef = m.basis_coef
    Σ = aVar(k, mm; demean = demean, prewhite = prewhite, scale = scale)

    all(basis_coef) && return Σ

    Σ_out = similar(Σ)
    fill!(Σ_out, NaN)
    for j in axes(Σ, 1)
        for i in axes(Σ, 2)
            if basis_coef[j] && basis_coef[i]
                Σ_out[j, i] = Σ[j, i]
            end
        end
    end
    return Σ_out
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
    return vec(sum(abs2, X / m.pp.chol.U, dims = 2))
end

function leverage(m::OLSEstimator{T, <:OLSPredictorQR}) where {T}
    # For QR: X = QR, so X(X'X)^(-1)X' = QQ'
    # h_i = ||Q_i||^2
    X = modelmatrix(m)
    Q = Matrix(m.pp.qr.Q)[:, 1:size(m.pp.qr.R, 1)]
    return vec(sum(abs2, Q, dims = 2))
end

@noinline residualadjustment(k::CM.HR0, r::OLSEstimator) = 1.0
@noinline residualadjustment(k::CM.HR1, r::OLSEstimator) = √nobs(r) / √dof_residual(r)
@noinline residualadjustment(k::CM.HR2, r::OLSEstimator) = 1.0 ./ (1 .- leverage(r)) .^ 0.5
@noinline residualadjustment(k::CM.HR3, r::OLSEstimator) = 1.0 ./ (1 .- leverage(r))

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

function residualadjustment(k::CM.CR2, r::OLSEstimator)
    wts = r.rr.wts
    @assert length(k.g) == 1
    g = k.g[1]
    X = modelmatrix(r)
    u_orig = residuals(r)
    u = copy(u_orig)
    !isempty(wts) && @. u *= sqrt(wts)
    XX = bread(r)
    for groups in 1:g.ngroups
        ind = findall(x -> x .== groups, g)
        Xg = view(X, ind, :)
        ug = view(u, ind, :)
        if isempty(wts)
            Hᵧᵧ = (Xg * XX * Xg')
            ldiv!(ug, cholesky!(Symmetric(I - Hᵧᵧ); check = false).L, ug)
        else
            Hᵧᵧ = (Xg * XX * Xg') .* view(wts, ind)'
            ug .= matrixpowbysvd(I - Hᵧᵧ, -0.5)*ug
        end
    end
    # Return the adjustment factor: adjusted_u / original_u
    # So that M = (X .* u_orig) .* factor = X .* adjusted_u
    return u ./ u_orig
end

function matrixpowbysvd(A, p; tol = eps()^(1/1.5))
    s = svd(A)
    V = s.S
    V[V .< tol] .= 0
    return s.V*diagm(0=>V .^ p)*s.Vt
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
    for groups in 1:g.ngroups
        ind = findall(g .== groups)
        Xg = view(X, ind, :)
        ug = view(u, ind, :)
        if isempty(wts)
            Hᵧᵧ = (Xg * XX * Xg')
            ldiv!(ug, cholesky!(Symmetric(I - Hᵧᵧ); check = false), ug)
        else
            Hᵧᵧ = (Xg * XX * Xg') .* view(wts, ind)'
            ug .= (I - Hᵧᵧ)^(-1)*ug
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
    return vec(sum(abs2, X / m.pp.chol.U, dims = 2))
end

function leverage(m::OLSMatrixEstimator{T, <:OLSPredictorQR}) where {T}
    X = modelmatrix(m)
    Q = Matrix(m.pp.qr.Q)[:, 1:size(m.pp.qr.R, 1)]
    return vec(sum(abs2, Q, dims = 2))
end

# Residual adjustment functions for OLSMatrixEstimator
@noinline residualadjustment(k::CM.HR0, r::OLSMatrixEstimator) = 1.0
@noinline residualadjustment(k::CM.HR1, r::OLSMatrixEstimator) = √nobs(r) / √dof_residual(r)
@noinline residualadjustment(k::CM.HR2, r::OLSMatrixEstimator) = 1.0 ./ (1 .- leverage(r)) .^ 0.5
@noinline residualadjustment(k::CM.HR3, r::OLSMatrixEstimator) = 1.0 ./ (1 .- leverage(r))

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
    mm = begin
        u = residualadjustment(k, m)
        M = copy(momentmatrix(m))
        @. M = M * u
        M
    end
    basis_coef = m.basis_coef
    Σ = aVar(k, mm; demean = demean, prewhite = prewhite, scale = scale)

    all(basis_coef) && return Σ

    Σ_out = similar(Σ)
    fill!(Σ_out, NaN)
    for j in axes(Σ, 1)
        for i in axes(Σ, 2)
            if basis_coef[j] && basis_coef[i]
                Σ_out[j, i] = Σ[j, i]
            end
        end
    end
    return Σ_out
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
    mm = begin
        u = residualadjustment(k, m)
        M = copy(momentmatrix(m))
        @. M = M * u
        M
    end
    basis_coef = m.basis_coef
    Σ = aVar(k, mm; demean = demean, prewhite = prewhite, scale = scale)

    all(basis_coef) && return Σ

    Σ_out = similar(Σ)
    fill!(Σ_out, NaN)
    for j in axes(Σ, 1)
        for i in axes(Σ, 2)
            if basis_coef[j] && basis_coef[i]
                Σ_out[j, i] = Σ[j, i]
            end
        end
    end
    return Σ_out
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
