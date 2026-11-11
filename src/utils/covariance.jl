const CM = CovarianceMatrices
using LinearAlgebra: BLAS

# Type union for OLS models - allows shared residualadjustment implementations
# Both OLSEstimator and OLSMatrixEstimator use identical residual adjustment logic
const OLSModel = Union{OLSEstimator, OLSMatrixEstimator}

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
## Direct Cluster Aggregation (fused moment + clustering)
##
## This avoids allocating the full n×k moment matrix by computing
## cluster sums directly.
##
##############################################################################

"""
    cluster_aggregate_direct!(X2, X, y, mu, groups) -> X2

Fused moment matrix computation and cluster aggregation.
Computes X2[g, :] = sum over i in cluster g of X[i, :] * (y[i] - mu[i])

This avoids allocating an n×k moment matrix, only allocating G×k.

# Arguments
- `X2::Matrix{T}`: Output buffer of size (ngroups, k), will be zeroed and filled
- `X::Matrix{T}`: Model matrix (n × k)
- `y::Vector{T}`: Response vector (weighted)
- `mu::Vector{T}`: Fitted values (weighted)
- `groups::Vector{Int}`: Cluster assignments (length n; any other length throws a
  `DimensionMismatch`)

# Returns
- `X2`: The filled cluster aggregation matrix
"""
function cluster_aggregate_direct!(X2::Matrix{T}, X::Matrix{T},
        y::AbstractVector{T}, mu::AbstractVector{T},
        groups::Vector{Int}) where {T <: AbstractFloat}
    Base.require_one_based_indexing(X2, X, y, mu, groups)
    n = size(X, 1)
    length(groups) == n || throw(DimensionMismatch(
        "the cluster variable has $(length(groups)) observations but the model has $n"))

    # Zero the output buffer
    fill!(X2, zero(T))

    # Aggregate: X2[g, j] = sum over i in cluster g of X[i,j] * (y[i] - mu[i])
    for j in axes(X, 2)
        for i in eachindex(groups, y, mu)
            g = groups[i]
            r = y[i] - mu[i]
            X2[g, j] += X[i, j] * r
        end
    end

    return X2
end

"""
    cluster_aVar_direct(X, y, mu, clustering; scale=true) -> Matrix

Compute cluster-robust asymptotic variance directly without moment matrix allocation.

Returns X2'X2 / n where X2 is the G×k cluster-aggregated moment matrix.
When scale=true (default), divides by n to match CovarianceMatrices.aVar behavior.
"""
function cluster_aVar_direct(X::Matrix{T}, y::AbstractVector{T}, mu::AbstractVector{T},
        clustering::CM.Clustering; scale::Bool = true) where {T <: AbstractFloat}
    n, k = size(X)
    ngroups = CM.nclusters(clustering)
    groups = CM.groups(clustering)

    # Allocate G×k buffer (much smaller than n×k for typical cluster scenarios)
    X2 = Matrix{T}(undef, ngroups, k)

    # Fused aggregation
    cluster_aggregate_direct!(X2, X, y, mu, groups)

    # Compute X2'X2 using BLAS syrk (symmetric rank-k update)
    aVar_buf = Matrix{T}(undef, k, k)
    BLAS.syrk!('U', 'T', one(T), X2, zero(T), aVar_buf)

    # Scale by n if requested (matches CovarianceMatrices.aVar behavior)
    if scale
        aVar_buf ./= n
    end

    return Symmetric(aVar_buf, :U)
end

##############################################################################
##
## Helper: Mask vcov matrix with NaN for collinear entries
##
##############################################################################

"""
    mask_vcov_collinear(Σ::AbstractMatrix{T}, basis_coef::BitVector) where {T}

Create a copy of a variance matrix with NaN for collinear entries.
The matrix Σ is expected to be full size (matching length of basis_coef).
Non-collinear entries are preserved; collinear entries are set to NaN.

Applies to a completed sandwich, not to an `aVar` result: the `aVar` methods
return the variance of the moment matrix, where coefficient collinearity has not
yet entered. The `vcov` methods that build their own sandwich handle rank
deficiency by subsetting to `findall(basis_coef)` directly.
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

    # Scale factor for HC1 sandwich estimator
    # The adjustment sqrt(n/dof_residual) was applied to residuals above,
    # which gives aVar = (1/dof_residual) * Σ(e_i² * X_i'X_i)
    # The HC1 formula is: Σ = (n/dof_residual) * invXX * Σ(e_i² * X_i'X_i) * invXX
    # So we need: scale * (1/dof_residual) = n/dof_residual, i.e., scale = n
    scale = T(n)

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

        return _wrap_vcov(Symmetric(Σ), CovarianceMatrices.HC1(), nothing)
    end

    # Standard case: no collinearity
    Σ = scale .* invXX * aVar * invXX
    return _wrap_vcov(Symmetric(Σ), CovarianceMatrices.HC1(), nothing)
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

    # Scale factor for HC1 sandwich estimator
    # The adjustment sqrt(n/dof_residual) was applied to residuals above,
    # which gives aVar = (1/dof_residual) * Σ(e_i² * X_i'X_i)
    # The HC1 formula is: Σ = (n/dof_residual) * invXX * Σ(e_i² * X_i'X_i) * invXX
    # So we need: scale * (1/dof_residual) = n/dof_residual, i.e., scale = n
    scale = T(n)

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

        return _wrap_vcov(Symmetric(Σ), CovarianceMatrices.HC1(), nothing)
    end

    # Standard case: no collinearity
    Σ = scale .* invXX * aVar * invXX
    return _wrap_vcov(Symmetric(Σ), CovarianceMatrices.HC1(), nothing)
end

##############################################################################
##
## CovarianceMatrices.jl Interface for Post-Estimation vcov
##
##############################################################################

##############################################################################
##
## CovarianceMatrices.jl Protocol Methods
##
## These implement the CovarianceMatrices.jl protocol for RegressionModel types.
## While Regress.jl uses its own specialized implementations for performance
## (fixest-style DOF corrections, fused broadcasts, direct cluster aggregation),
## these methods document the interface and enable potential future use of
## generic CovarianceMatrices.jl algorithms.
##
##############################################################################

# Protocol method: number of observations
CM.numobs(m::OLSEstimator) = nobs(m)
CM.numobs(m::OLSMatrixEstimator) = nobs(m)

# Protocol method: the residuals that enter the moments. For a weighted model
# `modelmatrix` returns `√w .* X`, and these are the matching `√w .* (y - Xβ)`;
# `residuals(m)` returns the unweighted residuals.
CM.momentresiduals(m::OLSModel) = m.rr.y .- m.rr.mu

# Protocol method: mask for non-collinear coefficients
CM.mask(m::OLSEstimator) = m.basis_coef
CM.mask(m::OLSMatrixEstimator) = m.basis_coef

# Protocol method: weights for the CR2/CR3 leverage blocks.
#
# Empty for both weighted and unweighted models. `modelmatrix` and `momentresiduals`
# already carry the weighting, so `X * bread(m) * X'` is the weighted hat matrix;
# applying `CovarianceMatrices`' additional `.* wts'` on top would square the weights
# and drive `I - H_gg` indefinite.
StatsBase.weights(m::OLSModel) = similar(m.rr.wts, 0)

# Note: CM.bread(), CM.leverage() and StatsAPI.leverage() are defined below in
# their respective sections

# `aVar` on a moment matrix accepts only estimators that need no fitted model: HR0
# and CR0 rather than HR1–HR5 and CR1–CR3. The `aVar` methods for this package's
# models fold the model-based residual adjustments into the moments and apply the
# degrees-of-freedom corrections as `vcov` scale factors, so the matrix computation
# they need is that of HR0, or of CR0 on the same clusters. Callers must have applied
# any CR2/CR3 leverage adjustment to the moments before mapping those to CR0.
_matrix_estimator(k) = k
_matrix_estimator(::CM.HR) = CM.HR0()
_matrix_estimator(k::CM.CR) = CM.CR0(CM.clusters(k))
# The Driscoll-Kraay small-sample correction is a `vcov` factor; the matrix path
# accepts only the uncorrected `:HC0`.
function _matrix_estimator(k::CM.DriscollKraay)
    return CM.DriscollKraay(CM.timekernel(k), CM.timeindex(k), CM.unitindex(k))
end
function _matrix_estimator(k::CM.CachedCR)
    _check_cached_estimator(CM.estimator(k))
    _check_cache(k)
    return CM.CachedCR(_matrix_estimator(CM.estimator(k)), CM.cache(k))
end

"""
    _moment_aVar(k, M; kwargs...) -> CovarianceMatrix

`aVar` of the adjusted moment matrix `M`, computed with `_matrix_estimator(k)` and
recorded as an estimate of `k`.
"""
function _moment_aVar(k, M; prewhite = false, kwargs...)
    # Checked here so that the error names `k` rather than `_matrix_estimator(k)`.
    prewhite && k isa Union{CM.HR, CM.CR} &&
        throw(ArgumentError(
            "`prewhite = true` is not supported by $(nameof(typeof(k))), which does not prewhiten."))
    A = aVar(_matrix_estimator(k), M; prewhite, kwargs...)
    return CovarianceMatrix(parent(A), k, CM.information(A))
end

"""
    _leverage_aVar(k::Union{CR2, CR3}, X, Xh, B, u, w; demean, prewhite, scale)
        -> CovarianceMatrix

`aVar` of the CR2 or CR3 estimator `k` for a model whose design `X` and residuals `u`
are multiplied by `√w`, with `w` empty for an unweighted model. The hat matrix is built
from `Xh`, the columns of `X` with `B = inv(Xh'Xh)`.

With `Z = X ./ √w` the untransformed design, each cluster's moments are
`Z_g' (I - H_gg)^p (w_g .* e_g)`, with `p = -1/2` (CR2) or `p = -1` (CR3) and
`H_gg = Z_g inv(Z'WZ) Z_g' W_g`. This is the convention of R's `sandwich::vcovCL` and of
CovarianceMatrices. With several cluster dimensions, the result is the
inclusion–exclusion sum of the CR0 meats of every intersection of dimensions, each with
the adjustment of its own clusters, as in `vcovCL`.
"""
function _leverage_aVar(k::Union{CM.CR2, CM.CR3}, X, Xh, B, u, w; demean = false,
        prewhite = false, scale = true)
    name = nameof(typeof(k))
    prewhite && throw(ArgumentError(
        "`prewhite = true` is not supported by $name, which does not prewhiten."))
    isempty(w) || all(>(0), w) ||
        throw(ArgumentError(
            "$name needs positive weights; the model has a weight of $(minimum(w))."))
    gs = CM.clusters(k)
    d = length(gs)
    Ω = nothing
    for s in 1:(2 ^ d - 1)
        dims = [i for i in 1:d if isodd(s >> (i - 1))]
        g = length(dims) == 1 ? gs[only(dims)] : CM.Clustering(gs[dims]...)
        ũ = _leverage_adjusted_residuals(k, g, Xh, B, u, w)
        A = parent(aVar(CM.CR0((g,)), X .* ũ; demean, scale))
        A = isodd(length(dims)) ? A : -A
        Ω = isnothing(Ω) ? A : Ω .+ A
    end
    return CovarianceMatrix(Ω, k, (nclusters = CM.nclusters(k),))
end

# The residuals `u` (of the √w-transformed model) with each cluster's block replaced
# by `((I - H̃_gg)^p (w_g .* u_g)) ./ w_g`, where `H̃_gg = X_g B X_g'` is the block of
# the symmetric hat matrix of the transformed design. Because
# `I - H_gg = W_g^(-1/2) (I - H̃_gg) W_g^(1/2)`, multiplying these residuals by `X_g`
# gives the moments of `_leverage_aVar`.
function _leverage_adjusted_residuals(k, g, X, B, u, w)
    Base.require_one_based_indexing(X, u, w)
    name = nameof(typeof(k))
    length(CM.groups(g)) == length(u) || throw(DimensionMismatch(
        "the cluster variable has $(length(CM.groups(g))) observations, the model $(length(u))"))
    rows = [Int[] for _ in 1:CM.nclusters(g)]
    for (i, gi) in pairs(CM.groups(g))
        push!(rows[gi], i)
    end
    p = k isa CM.CR2 ? -1 / 2 : -1
    tol = sqrt(eps(real(float(eltype(X)))))
    adj = float.(u)
    for ind in rows
        isempty(ind) && continue
        Xg = view(X, ind, :)
        λ, Q = eigen(Symmetric(I - Xg * B * Xg'))
        minimum(λ) > tol || throw(ArgumentError(
            "$name needs I - H_gg to be nonsingular for every cluster, but the cluster " *
            "containing row $(first(ind)) has a hat-matrix eigenvalue of 1 (to within " *
            "$tol). This happens when a regressor is nonzero only within one cluster, " *
            "as with a cluster fixed effect."))
        if isempty(w)
            adj[ind] = Q * (λ .^ p .* (Q' * view(u, ind)))
        else
            wg = view(w, ind)
            adj[ind] = (Q * (λ .^ p .* (Q' * (wg .* view(u, ind))))) ./ wg
        end
    end
    return adj
end

"""
    CovarianceMatrices.momentmatrix(m::OLSEstimator)

Returns the moment matrix `modelmatrix(m) .* momentresiduals(m)`, weighted for a
weighted model.
Required for post-estimation variance-covariance calculations.
"""
function CovarianceMatrices.momentmatrix(m::OLSEstimator)
    return modelmatrix(m) .* CM.momentresiduals(m)
end

##############################################################################
##
## Entry points: vcov, stderror and aVar on Regress models
##
## `vcov(k, m)`, `stderror(k, m)` and `aVar(k, m)` on the four model types
## (`OLSEstimator`, `OLSMatrixEstimator`, `IVEstimator`, `IVMatrixEstimator`) are
## defined once, at the end of this file. Each checks the keywords, replaces
## clusters named by column with the model's stored cluster data, and calls
## `_regress_vcov` or `_regress_aVar`, whose methods hold the estimator-specific code.
##
##############################################################################

# The moment matrix of an OLS model is `X .* (y - mu)`; `y` and `mu` are already
# weighted for a weighted model.
function _regress_aVar(k, m::OLSModel; demean = false, prewhite = false, scale = true)
    X = modelmatrix(m)
    y = m.rr.y
    mu = m.rr.mu
    # The adjustment can be Float64 on a Float32 model; the moments keep the model's type.
    u = eltype(X).(residualadjustment(k, m))
    mm = @. X * (y - mu) * u
    # Bandwidth-selection weights come from the model matrix, not the moment
    # matrix: they must give the intercept weight 0, and the intercept column of
    # the moment matrix is not constant. `nothing` for non-HAC estimators.
    kw = CM.bandwidthweights(k, X)
    return _moment_aVar(k, mm; demean, prewhite, scale, weights = kw)
end

function _regress_aVar(k::K, m::OLSModel; demean = false, prewhite = false,
        scale = true) where {K <: CM.CR}
    X = modelmatrix(m)
    y = m.rr.y
    mu = m.rr.mu

    # Single-cluster CR0/CR1 aggregate the scores by cluster directly, without
    # allocating the n×k moment matrix. `demean` and `prewhite` need the moment matrix.
    if length(CM.clusters(k)) == 1 && (K <: Union{CM.CR0, CM.CR1}) && !demean && !prewhite
        clustering = only(CM.clusters(k))
        # Direct aggregation: only allocate G×k instead of n×k
        # Pass scale parameter to match CovarianceMatrices.aVar behavior
        return cluster_aVar_direct(X, y, mu, clustering; scale = scale)
    end

    # The hat matrix of CR2/CR3 uses the estimable columns only.
    if K <: Union{CM.CR2, CM.CR3}
        return _leverage_aVar(k, X, view(X, :, m.basis_coef), invchol(m.pp), y .- mu,
            m.rr.wts; demean, prewhite, scale)
    end

    # Fallback: multi-way CR0/CR1, and `Cluster`
    mm = @. X * (y - mu)
    return _moment_aVar(k, mm; demean, prewhite, scale)
end

##############################################################################
##
## Clusters named by column
##
## `CR1(:firm)`, `Cluster(:firm, :year)` and the like name the clusters by column, and
## `DriscollKraay(K; tis = :year, iis = :firm)` names its time and unit indices. A model
## fitted from a formula stores the columns given to `save_cluster`, and the entry
## points replace the names with that data before any computation.
##
##############################################################################

"""
    _resolve_clusters(k, m)

`k` with clusters, or Driscoll-Kraay time and unit indices, named by column replaced by
the data stored in `m`, as an estimator of the same type. Any other `k` is returned
unchanged.
"""
_resolve_clusters(k, m) = k
for E in (:CR0, :CR1, :CR2, :CR3, :Cluster)
    @eval function _resolve_clusters(k::CM.$E{<:Tuple{Vararg{Symbol}}}, m)
        return CM.$E(_cluster_vectors(k, m))
    end
end
function _resolve_clusters(k::CM.DriscollKraay{<:Any, Symbol, Symbol}, m)
    tis, iis = _cluster_vectors(k, m)
    return CM.DriscollKraay(CM.timekernel(k), tis, iis; correction = CM.correction(k))
end

# A model fitted from matrices takes per-observation data (cluster identifiers,
# Driscoll-Kraay indices) with one entry per row of its input, or one per observation;
# the first is restricted to the estimation sample.
function _sample_rows(v::AbstractVector, m::Union{OLSMatrixEstimator, IVMatrixEstimator},
        what::String)
    length(v) == nobs(m) && return v
    length(v) == length(m.esample) && return esample(m, v)
    throw(DimensionMismatch(
        "the $what has $(length(v)) observations; the model was fitted on " *
        "$(length(m.esample)) rows, of which $(nobs(m)) are in the estimation sample"))
end

for E in (:CR0, :CR1, :CR2, :CR3, :Cluster)
    @eval function _resolve_clusters(
            k::CM.$E{<:Tuple{CM.Clustering, Vararg{CM.Clustering}}},
            m::Union{OLSMatrixEstimator, IVMatrixEstimator})
        return CM.$E(map(c -> _sample_rows(CM.groups(c), m, "cluster variable"),
            CM.clusters(k)))
    end
end
function _resolve_clusters(
        k::CM.DriscollKraay{<:Any, <:AbstractVector, <:AbstractVector},
        m::Union{OLSMatrixEstimator, IVMatrixEstimator})
    return CM.DriscollKraay(CM.timekernel(k),
        _sample_rows(CM.timeindex(k), m, "Driscoll-Kraay time index"),
        _sample_rows(CM.unitindex(k), m, "Driscoll-Kraay unit index");
        correction = CM.correction(k))
end

# The column names that `k` looks up in the stored cluster data.
_cluster_names(k) = CM.clusters(k)
_cluster_names(k::CM.DriscollKraay) = (CM.timeindex(k), CM.unitindex(k))

# The stored cluster vectors for the column names of `k`.
function _cluster_vectors(k, m::OLSEstimator)
    return map(_cluster_names(k)) do name
        haskey(m.fes.clusters, name) || _cluster_not_found_error(name, m)
        m.fes.clusters[name]
    end
end

function _cluster_vectors(k, m::Union{OLSMatrixEstimator, IVMatrixEstimator})
    E = nameof(typeof(k))
    throw(ArgumentError(
        "$(sprint(show, k)) names its clusters by column, but a model fitted from " *
        "matrices stores no cluster data. Pass the group identifiers instead, " *
        "e.g. `$E(groups)`."))
end

function _cluster_vectors(
        k::CM.DriscollKraay, m::Union{OLSMatrixEstimator, IVMatrixEstimator})
    throw(ArgumentError(
        "$(sprint(show, k)) names its time and unit indices by column, but a model " *
        "fitted from matrices stores no data. Pass the indices instead, " *
        "e.g. `DriscollKraay(K; tis = t, iis = id)`."))
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
    CachedCR(cr::CR, m)

Create a `CachedCR` estimator for the Regress model `m` (`OLSEstimator`,
`OLSMatrixEstimator`, `IVEstimator` or `IVMatrixEstimator`), with buffers sized for the
model's moment matrix. For a model fitted from a formula, clusters named by column
(`CR1(:firm_id)`) are looked up in the cluster data saved with the model.

`cr` must be `CR0` or `CR1`; `CR2` and `CR3` throw an `ArgumentError`. `vcov(cached, m)`
equals `vcov(cr, m)`. A `CachedCR` that CovarianceMatrices built from a model of another
type holds that model's adjustments, and `vcov` with it on a Regress model throws an
`ArgumentError`.

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
function CM.CachedCR(cr::CM.CR, m::Union{OLSModel, IVEstimator, IVMatrixEstimator})
    _check_cached_estimator(cr)
    return CM.CachedCR(_resolve_clusters(cr, m), _cr_moment_ncols(m))
end

# Columns of the moment matrix that the model's cluster-robust `aVar` forms. For OLS it
# keeps the collinear columns; for IV it holds only the estimable ones.
_cr_moment_ncols(m::OLSModel) = size(modelmatrix(m), 2)
function _cr_moment_ncols(m::IVEstimator)
    pe = m.postestimation
    isnothing(pe) &&
        throw(ArgumentError("Model does not have post-estimation data stored."))
    return size(has_kclass_adj(pe) ? pe.Adj : pe.X_fitted, 2)
end
_cr_moment_ncols(m::IVMatrixEstimator) = size(m.postestimation.X_hat, 2)

# The estimator whose finite-sample factor `vcov` applies: a `CachedCR` computes the
# estimator it wraps.
_uncached(k) = k
_uncached(k::CM.CachedCR) = CM.estimator(k)

# The cache holds cluster buffers, not the per-cluster leverage adjustments of CR2
# and CR3.
_check_cached_estimator(cr) = nothing
function _check_cached_estimator(cr::Union{CM.CR2, CM.CR3})
    name = nameof(typeof(cr))
    throw(ArgumentError(
        "CachedCR on a Regress model supports CR0 and CR1; $name needs the leverage " *
        "of each cluster, which the cache does not hold. Use vcov($name(...), model)."))
end

# `CachedCR` on a Regress model builds a cache of buffers only. A model cache comes from
# `CachedCR(k, model)` on a model of another type, and holds that model's adjustments.
_check_cache(k::CM.CachedCR) = nothing
function _check_cache(k::CM.CachedCR{<:CM.CR, <:CM.CRModelCache})
    throw(ArgumentError(
        "this CachedCR was built by CovarianceMatrices from a model of another type, " *
        "and its cache holds that model's adjustments. Build it from the Regress model " *
        "with CachedCR(estimator, model)."))
end

##############################################################################
##
## Helper Functions for Cluster Variable Handling
##
##############################################################################

# Error message for missing cluster variable
function _cluster_not_found_error(cluster_name::Symbol, m::OLSEstimator)
    available = isempty(m.fes.clusters) ? "none" : join(keys(m.fes.clusters), ", :")
    throw(ArgumentError("""
    Cluster variable :$cluster_name not found in model.

    Available cluster variables: :$available

    To use this cluster variable, either:
      1. Re-fit with save_cluster=:$cluster_name
      2. Use data directly: vcov(CR1(df.$cluster_name[model.esample]), model)
    """))
end

"""
    CovarianceMatrices.bread(m::OLSEstimator)

Compute (X'X)^(-1), the "bread" of the sandwich variance estimator.
"""
CM.bread(m::OLSEstimator) = invchol(m.pp)

# CovarianceMatrices declares its own `leverage`, distinct from `StatsAPI.leverage`;
# its HC2-HC5 and CR2/CR3 residual adjustments dispatch on that one.
CM.leverage(m::OLSEstimator) = StatsAPI.leverage(m)

"""
    leverage(m::Union{OLSEstimator, OLSMatrixEstimator})

Diagonal of the hat matrix X(X'X)⁻¹X', over the columns of X that collinearity detection
kept (the others have no coefficient).
"""
function StatsAPI.leverage(m::OLSModel)
    has_predictor_data(m.pp) ||
        throw(ArgumentError("Model matrix not stored. Model was fit with save=:minimal."))
    return _leverage(m.pp)
end

# h_i = ‖X_i F⁻¹‖² for upper-triangular F with X'X = F'F. A product with the k×k inverse:
# the right division X / F runs as a triangular solve over all n rows, which BLAS does
# several times more slowly.
_triangular_leverage(X, F) = vec(sum(abs2, X * inv(F), dims = 2))
_leverage(pp::OLSPredictorChol) = _triangular_leverage(pp.X_reduced, pp.chol.U)
_leverage(pp::OLSPredictorQR) = _triangular_leverage(pp.X_reduced, UpperTriangular(pp.qr.R))
function _leverage(pp::OLSPredictorSweep)
    # (X'X)⁻¹ is stored: with (X'X)⁻¹ = LL', h_i = ‖X_i L‖²
    L = cholesky(pp.invXX).L
    return vec(sum(abs2, pp.X_reduced * L, dims = 2))
end

# Residual adjustment for OLS models (both OLSEstimator and OLSMatrixEstimator).
#
# `residualadjustment` returns an elementwise multiplier for the residual vector,
# which the `aVar` methods fold into the moment matrix. The heteroskedasticity-
# consistent adjustments come from `CovarianceMatrices.residual_adjustment`, whose
# HR0/HR2-HR5 methods already return exactly that.
@noinline residualadjustment(k::CM.HR, r::OLSModel) = CM.residual_adjustment(k, r)

# HR1 is the exception: upstream divides by `n - length(coef(r))`, which cannot see
# fixed effects absorbed out of the design. Regress's `dof_residual` counts them, so
# delegating here would understate the correction on any model with `fe(...)` terms.
@noinline residualadjustment(k::CM.HR1, r::OLSModel) = sqrt(nobs(r) / dof_residual(r))

# CR0/CR1 need no per-observation adjustment: the clustering is handled by
# `CovarianceMatrices.aVar`, and the finite-sample correction is applied as a
# scale factor in `vcov` (`_finite_sample_factor`, matching fixest) rather than
# being folded into the residuals as upstream does.
@noinline residualadjustment(k::CM.CR0, r::OLSModel) = 1.0
@noinline residualadjustment(k::CM.CR1, r::OLSModel) = 1.0

# Correlated estimators (HAC, EWC, DriscollKraay, VARHAC, etc.) - no residual adjustment needed
@noinline residualadjustment(k::CM.Correlated, r::OLSModel) = 1.0

# Any other estimator (the smoothers) on any Regress model uses CovarianceMatrices'
# adjustment, which throws a `MethodError` for an estimator it does not know.
@noinline residualadjustment(k, m) = CM.residual_adjustment(k, m)

##############################################################################
##
## Finite-sample factors
##
## Every sandwich `vcov` on a Regress model is `c · n · B · aVar(k, m) · B`, with
## bread `B` and the factor `c = _finite_sample_factor(k, m, dofadjust)`. The factor
## depends on the estimator family only, and is the same on the four model types.
##
##############################################################################

"""
    _finite_sample_factor(k, m, dofadjust) -> Real

Factor `c` in `vcov(k, m) = c · n · B · aVar(k, m) · B`. `n · B · aVar(k, m) · B` is the
raw sandwich for every estimator except `HR1`, whose `n / dof_residual(m)` is part of its
residual adjustment.

- `HR0`–`HR5`: 1.
- `CR0`: (n−1)/(n−K), and `CR1`: G/(G−1)·(n−1)/(n−K), where G is the smallest number of
  clusters over the cluster dimensions and K is `_cluster_dof(k, m)` (fixest's default).
- `CR2`, `CR3`: 1; the leverage adjustment is their small-sample correction.
- `Cluster`: 1.
- Other `Correlated` estimators (HAC, EWC, VARHAC) and the smoothers: n / dof_residual(m),
  or 1 with `dofadjust = false`.
"""
_finite_sample_factor(k, m, dofadjust) = 1
function _finite_sample_factor(k::Union{CM.Correlated, CM.MomentSmoother}, m, dofadjust)
    return dofadjust ? nobs(m) / dof_residual(m) : 1
end
_finite_sample_factor(k::Union{CM.CR2, CM.CR3, CM.Cluster}, m, dofadjust) = 1
function _finite_sample_factor(k::CM.CR0, m, dofadjust)
    n = nobs(m)
    return (n - 1) / (n - _cluster_dof(k, m))
end
function _finite_sample_factor(k::CM.CR1, m, dofadjust)
    n = nobs(m)
    G = minimum(CM.nclusters(k))
    return G / (G - 1) * (n - 1) / (n - _cluster_dof(k, m))
end
function _finite_sample_factor(k::CM.CachedCR, m, dofadjust)
    return _finite_sample_factor(CM.estimator(k), m, dofadjust)
end

"""
    _cluster_dof(k::CR, m) -> Int

The number of parameters K in the (n−1)/(n−K) factor of `CR0` and `CR1`, as fixest
counts it: the estimable coefficients, plus, for each absorbed fixed effect, its number
of levels, or 1 if it is nested in one of the cluster dimensions of `k`, minus one for
each fixed effect after the first (the levels of each additional fixed effect hold one
redundant parameter). A model with absorbed fixed effects has no intercept coefficient.

A fixed effect is nested in a cluster dimension when each of its levels lies within a
single cluster. Without stored fixed-effect groups (a model fitted with
`save = :minimal`), a fixed effect counts as nested when a cluster column saved with the
model has its name.
"""
function _cluster_dof(k,
        m::Union{OLSEstimator, IVEstimator, OLSMatrixEstimator, IVMatrixEstimator})
    K = sum(CM.mask(m))
    (; names, groups, ngroups, cluster_names) = _fe_structure(m)
    isempty(names) && return K
    cluster_refs = map(CM.groups, CM.clusters(k))
    for i in eachindex(names, ngroups)
        nested = if isempty(groups)
            names[i] in cluster_names
        else
            any(refs -> _isnested(groups[i], refs, ngroups[i]), cluster_refs)
        end
        K += nested ? 1 : ngroups[i]
    end
    return K - (length(names) - 1)
end

function _fe_structure(m::OLSEstimator)
    fes = m.fes
    return (; names = fes.fe_names, groups = fes.fe_groups, ngroups = fes.ngroups,
        cluster_names = keys(fes.clusters))
end

# Matrix fits always store their fixed-effect groups, so the names only number them.
function _fe_structure(m::Union{OLSMatrixEstimator, IVMatrixEstimator})
    return (; names = [Symbol(:fe, i) for i in eachindex(m.fe_groups)],
        groups = m.fe_groups, ngroups = m.fe_ngroups, cluster_names = ())
end

"""
    _isnested(fe_refs, cluster_refs, n_fe_groups) -> Bool

Whether every one of the `n_fe_groups` fixed-effect groups in `fe_refs` (codes
`1:n_fe_groups`) lies within a single cluster of `cluster_refs`.
"""
function _isnested(fe_refs::AbstractVector{<:Integer},
        cluster_refs::AbstractVector{<:Integer}, n_fe_groups::Integer)
    entries = zeros(Int, n_fe_groups)
    for i in eachindex(fe_refs, cluster_refs)
        feref, cref = fe_refs[i], cluster_refs[i]
        if entries[feref] == 0
            entries[feref] = cref
        elseif entries[feref] != cref
            return false
        end
    end
    return true
end

function _regress_vcov(k, m::OLSModel; dofadjust = true, demean = false,
        prewhite = false)
    A = _regress_aVar(k, m; demean, prewhite)
    B = invchol(m.pp)
    basis_coef = m.basis_coef
    scale = eltype(B)(nobs(m) * _finite_sample_factor(k, m, dofadjust))

    # With collinear columns, A is k×k while B covers only the estimable columns:
    # compute the sandwich on those, and fill the rest with NaN.
    if !all(basis_coef)
        valid_idx = findall(basis_coef)
        Σ_valid = scale .* B * A[valid_idx, valid_idx] * B
        Σ = fill(eltype(Σ_valid)(NaN), length(basis_coef), length(basis_coef))
        Σ[valid_idx, valid_idx] = Σ_valid
        return _wrap_vcov(Σ, k, A)
    end

    return _wrap_vcov(scale .* B * A * B, k, A)
end

# Driscoll-Kraay variance, computed by CovarianceMatrices' `vcov` for a
# `RegressionModel`. The estimator carries its own time and unit indices and its
# small-sample correction (`DriscollKraay(...; correction = :HC1)`), which
# `dofadjust = false` leaves out. Driscoll-Kraay scales by the number of time
# periods, not by the number of observations, so it does not use the sandwich of the
# other estimators: the two divisors differ by a factor of roughly `T / n`.
for M in (OLSEstimator, OLSMatrixEstimator)
    @eval function _regress_vcov(k::CM.DriscollKraay, m::$M; kwargs...)
        Σ = invoke(CM.vcov, Tuple{CM.DriscollKraay, RegressionModel}, k, m; kwargs...)
        return _wrap_vcov(parent(Σ), k, Σ)
    end
end

##############################################################################
##
## CovarianceMatrices.jl Interface for OLSMatrixEstimator
##
##############################################################################

"""
    CovarianceMatrices.momentmatrix(m::OLSMatrixEstimator)

Returns the moment matrix `modelmatrix(m) .* momentresiduals(m)`, weighted for a
weighted model.
Required for post-estimation variance-covariance calculations.
"""
function CovarianceMatrices.momentmatrix(m::OLSMatrixEstimator)
    return modelmatrix(m) .* CM.momentresiduals(m)
end

"""
    CovarianceMatrices.bread(m::OLSMatrixEstimator)

Compute (X'X)^(-1), the "bread" of the sandwich variance estimator.
"""
CM.bread(m::OLSMatrixEstimator) = invchol(m.pp)

CM.leverage(m::OLSMatrixEstimator) = StatsAPI.leverage(m)

# Note: residualadjustment methods for OLSMatrixEstimator are provided by the
# OLSModel type union (Union{OLSEstimator, OLSMatrixEstimator}) defined earlier
# in this file, eliminating code duplication.

"""
    _wrap_vcov(Σ, estimator, A) -> CovarianceMatrix

Pair a completed sandwich `Σ` with the `estimator` that produced it and the
quantities selected during estimation.

`A` is the `aVar` result the sandwich was built from. When it carries selected
quantities — a HAC bandwidth and kernel weights — they are propagated, so
`bandwidth` and `kernelweights` resolve on the returned matrix. Paths that never
call `aVar`, and the single-cluster CR0/CR1 aggregation that bypasses it, have
nothing to propagate and yield an empty `info`; `bandwidth` then returns
`nothing`, which is the documented result for an estimator that selects nothing.
"""
_wrap_vcov(Σ, estimator, A) = CovarianceMatrix(Σ, estimator, NamedTuple())
function _wrap_vcov(Σ, estimator, A::CovarianceMatrix)
    return CovarianceMatrix(Σ, estimator, CovarianceMatrices.information(A))
end

function StatsAPI.confint(ve::CovarianceMatrices.AbstractAsymptoticVarianceEstimator,
        m::OLSMatrixEstimator; level::Real = 0.95)
    scale = tdistinvcdf(StatsAPI.dof_residual(m), 1 - (1 - level) / 2)
    se = CovarianceMatrices.stderror(ve, m)
    hcat(coef(m) - scale * se, coef(m) + scale * se)
end

##############################################################################
##
## Entry points
##
##############################################################################

const RegressModel = Union{OLSEstimator, OLSMatrixEstimator, IVEstimator, IVMatrixEstimator}

const _VCOV_KEYWORDS = (:dofadjust, :demean, :prewhite)
const _AVAR_KEYWORDS = (:demean, :prewhite, :scale)

function _check_keywords(f, k, kwargs, allowed)
    bad = filter(∉(allowed), collect(keys(kwargs)))
    isempty(bad) || throw(ArgumentError(
        "$f($(nameof(typeof(k)))(...), model) does not accept the keyword(s) " *
        "$(join(bad, ", ")); its keywords are $(join(allowed, ", "))."))
    return nothing
end

# `demean` and `prewhite` act on a moment matrix, which the iid variance does not use.
function _check_iid_keywords(demean, prewhite)
    (demean || prewhite) && throw(ArgumentError(
        "`demean` and `prewhite` do not apply to Uncorrelated(), the iid variance"))
    return nothing
end

## The iid variance of `Uncorrelated()`: `σ² · B` with bread `B` and
## `σ² = RSS / dof_residual(m)`. The residuals are those of the moment matrix, so RSS
## is the weighted sum of squares for a weighted model.

_iid_rss(m::OLSModel) = sum(abs2, CM.momentresiduals(m))
_iid_rss(m::IVEstimator) = sum(abs2, residuals_for_vcov(m))
_iid_rss(m::IVMatrixEstimator) = sum(abs2, m.postestimation.residuals)

# The bread of an OLS model covers only the estimable coefficients; the variance has
# a NaN row and column for each of the others, as in the sandwich `vcov`.
function _iid_expand(V, m::OLSModel)
    basis_coef = m.basis_coef
    all(basis_coef) && return V
    valid_idx = findall(basis_coef)
    Σ = fill(eltype(V)(NaN), length(basis_coef), length(basis_coef))
    Σ[valid_idx, valid_idx] = V
    return Σ
end
_iid_expand(V, m) = V

function _iid_vcov(k::CM.Uncorrelated, m; dofadjust = true, demean = false,
        prewhite = false)
    _check_iid_keywords(demean, prewhite)
    σ² = _iid_rss(m) / (dofadjust ? dof_residual(m) : nobs(m))
    B = CM.bread(m)
    return _wrap_vcov(_iid_expand(Symmetric(σ² * B), m), k, nothing)
end

# `σ_n² · inv(B) / n` with `σ_n² = RSS / n` (without `/ n` for `scale = false`), so that
# `n · B · aVar · B` is the iid variance with `dofadjust = false`. Like `B`, it covers
# the estimable coefficients only.
function _iid_aVar(k::CM.Uncorrelated, m; demean = false, prewhite = false,
        scale = true)
    _check_iid_keywords(demean, prewhite)
    n = nobs(m)
    σ_n² = _iid_rss(m) / n
    A = σ_n² * inv(Symmetric(Matrix(CM.bread(m))))
    return CovarianceMatrix(scale ? A / n : A, k, NamedTuple())
end

"""
    vcov(k::AbstractAsymptoticVarianceEstimator, m; dofadjust = true, demean = false, prewhite = false)
    stderror(k::AbstractAsymptoticVarianceEstimator, m; kwargs...)

Variance-covariance matrix of the coefficients of the Regress model `m` (`OLSEstimator`,
`OLSMatrixEstimator`, `IVEstimator` or `IVMatrixEstimator`), computed with the estimator
`k`, as a `CovarianceMatrix`. `stderror` returns the square roots of its diagonal and
takes the same keywords.

Cluster-robust estimators (`CR0`–`CR3`, `Cluster`) may name their clusters by column,
as in `CR1(:firm)` or `CR1(:firm, :year)`, on a model fitted from a formula with
`save_cluster`; the column data stored with the model is used. A model fitted from
matrices stores no cluster data and needs the group identifiers, as in `CR1(groups)`.
In the same way, `DriscollKraay(K; tis = :year, iis = :firm)` looks up its time and unit
indices in the columns saved with `save_cluster = [:year, :firm]`.

# Finite-sample factors
Each estimator multiplies the raw sandwich `B · (Σ moments) · B` by one factor, the same
on all four model types. `n` is the number of observations, `G` the number of clusters
(with several cluster dimensions, the smallest), and `n - K = dof_residual(m)`.

| Estimator | Factor |
|:--|:--|
| `HC0`, `HC2`–`HC5` | 1 (`HC2`–`HC5` rescale each residual by its leverage) |
| `HC1` | `n/(n-K)` |
| `CR0` | `(n-1)/(n-Kc)` |
| `CR1` | `G/(G-1) · (n-1)/(n-Kc)` |
| `CR2`, `CR3` | 1 (the Bell–McCaffrey leverage adjustment) |
| `Cluster` | 1 |
| HAC kernels, `EWC`, `VARHAC`, smoothers | `n/(n-K)` |
| `DriscollKraay` | its `correction` |
| `Uncorrelated` | `σ² · (X'X)⁻¹` with `σ² = RSS/(n-K)` |

`HC1` and `CR1` equal R fixest's defaults, and `CR0` equals fixest with
`ssc(cluster.adj = FALSE)`. `CR2`, `CR3` and the HAC kernels equal R sandwich
(`vcovCL`, `NeweyWest(adjust = TRUE)`), including weighted `CR2` and `CR3`, and
multiway `CR2` and `CR3` on OLS models. On IV models, multiway `CR2` and `CR3` use the
leverage of the projected regressors in every term, while `vcovCL` uses
`hatvalues(ivreg)` for a term whose clusters are single rows, so the two can differ
slightly. `Kc` counts the parameters as
fixest does: the coefficients, plus, for each absorbed fixed effect, its number of
levels, or 1 if it is nested in a cluster dimension, minus one for each fixed effect
after the first. With several cluster dimensions, R `vcovCL` applies `G_d/(G_d-1)` to
each dimension instead of the uniform `G/(G-1)`.

# Keywords
- `dofadjust`: apply the degrees-of-freedom factor that is not part of the estimator's
  definition: `n/(n-K)` for the long-run estimators (HAC kernels, EWC, VARHAC and the
  smoothers), the `correction` of a `DriscollKraay` estimator, and the `n - K` divisor
  of `Uncorrelated()` (`n` when `false`). The finite-sample factors of `HC1`–`HC5` and
  `CR0`–`CR3` define those estimators, and `dofadjust` does not change them.
- `demean`, `prewhite`: passed to `aVar`.

Any other keyword throws an `ArgumentError`.

# Examples
```julia
model = ols(df, @formula(y ~ x), save_cluster = :firm)
vcov(HC3(), model)
vcov(CR1(:firm), model)
stderror(Bartlett(5), model; prewhite = true)
```
"""
function CM.vcov(k::CM.AbstractAsymptoticVarianceEstimator, m::RegressModel; kwargs...)
    _check_keywords(vcov, k, kwargs, _VCOV_KEYWORDS)
    return _regress_vcov(_resolve_clusters(k, m), m; kwargs...)
end

function CM.vcov(k::CM.Uncorrelated, m::RegressModel; kwargs...)
    _check_keywords(vcov, k, kwargs, _VCOV_KEYWORDS)
    return _iid_vcov(k, m; kwargs...)
end

# `DriscollKraay` has its own method on a `RegressionModel` in CovarianceMatrices, which
# explains that a `type` keyword belongs in the estimator.
function CM.vcov(k::CM.DriscollKraay, m::RegressModel; kwargs...)
    _check_keywords(vcov, k, kwargs, (_VCOV_KEYWORDS..., :type))
    return _regress_vcov(_resolve_clusters(k, m), m; kwargs...)
end

function StatsBase.stderror(k::CM.AbstractAsymptoticVarianceEstimator, m::RegressModel;
        kwargs...)
    _check_keywords(stderror, k, kwargs, _VCOV_KEYWORDS)
    return sqrt.(diag(vcov(k, m; kwargs...)))
end

"""
    aVar(k::AbstractAsymptoticVarianceEstimator, m; demean = false, prewhite = false, scale = true)

Asymptotic variance (the meat of the sandwich) of the moment conditions of the Regress
model `m`, computed with `k`. Clusters named by column are looked up as in `vcov`.
`demean`, `prewhite` and `scale` have the meaning they have for `aVar` on a matrix; any
other keyword throws an `ArgumentError`.

For `Uncorrelated()` the result is `(RSS/n) · inv(B) / n` with bread `B` (without the
`/ n` for `scale = false`), so that `n · B · aVar · B` is
`vcov(Uncorrelated(), m; dofadjust = false)`. It covers the estimable coefficients only,
as `B` does, and `demean` and `prewhite` throw an `ArgumentError`.
"""
function CM.aVar(k::CM.AbstractAsymptoticVarianceEstimator, m::RegressModel; kwargs...)
    _check_keywords(aVar, k, kwargs, _AVAR_KEYWORDS)
    return _regress_aVar(_resolve_clusters(k, m), m; kwargs...)
end

function CM.aVar(k::CM.Uncorrelated, m::RegressModel; kwargs...)
    _check_keywords(aVar, k, kwargs, _AVAR_KEYWORDS)
    return _iid_aVar(k, m; kwargs...)
end

# CovarianceMatrices defines `aVar` methods on a `RegressionModel` for these two.
for K in (CM.CR, CM.DriscollKraay)
    @eval function CM.aVar(k::$K, m::RegressModel; kwargs...)
        _check_keywords(aVar, k, kwargs, _AVAR_KEYWORDS)
        return _regress_aVar(_resolve_clusters(k, m), m; kwargs...)
    end
end
