##############################################################################
##
## Kleibergen-Paap Rank Test for Weak Instruments
##
## Ported from Vcov.jl to work with CovarianceMatrices.jl
## Reference: Kleibergen & Paap (2006), "Generalized Reduced Rank Tests
##            Using the Singular Value Decomposition"
##
##############################################################################

using LinearAlgebra

# Estimators for which the KP statistic is defined. HR2–HR5 and CR2/CR3 have no
# established form for it, and DriscollKraay, EWC, VARHAC and the smoothers are not
# supported.
_kp_supported(k) = false
function _kp_supported(::Union{
        CovarianceMatrices.Uncorrelated, CovarianceMatrices.HR0, CovarianceMatrices.HR1,
        CovarianceMatrices.CR0, CovarianceMatrices.CR1, CovarianceMatrices.HAC})
    true
end
function _kp_supported(k::CovarianceMatrices.CachedCR)
    _kp_supported(CovarianceMatrices.estimator(k))
end

function _check_kp_estimator(k, statistic = "The Kleibergen-Paap statistic")
    _kp_supported(k) && return nothing
    throw(ArgumentError(
        "$statistic is not defined for $(nameof(typeof(k))). It " *
        "supports Uncorrelated, HC0, HC1, CR0, CR1 (any number of cluster dimensions) " *
        "and the HAC kernels (Bartlett, Parzen, QuadraticSpectral, TukeyHanning, " *
        "Truncated). Use, for example, `m + vcov(HC1())`."))
end

"""
    ranktest(Xendo_res, Z_res, Pi, k, c)

Kleibergen-Paap rk statistic for the rank of `Pi`, the l × k coefficients of the
excluded instruments in the first-stage regressions. `Xendo_res` (n × k) holds the
first-stage residuals and `Z_res` (n × l) the excluded instruments with the exogenous
regressors partialled out.

The variance of the first-stage moments `Z_res ⊗ Xendo_res` is
`c · n · aVar(k, moments)`, with `aVar` computed as in `vcov` (HR0 for `HR1`, CR0 on the
same clusters for `CR1`) and `c` the finite-sample factor of the first-stage regression.
For `Uncorrelated` it is the iid variance, `c` times that with σ² = RSS/n.

Returns `NaN` when `Z_res'Z_res`, `Xendo_res'Xendo_res` or the variance of the
statistic is not positive definite.
"""
function ranktest(
        Xendo_res::AbstractMatrix{T},
        Z_res::AbstractMatrix{T},
        Pi::AbstractMatrix{T},
        k,
        c::Real
) where {T <: AbstractFloat}
    Base.require_one_based_indexing(Xendo_res, Z_res, Pi)
    _check_kp_estimator(k)
    nendo = size(Xendo_res, 2)
    l = size(Z_res, 2)
    n = size(Z_res, 1)

    if nendo == 0 || l == 0
        return T(NaN)
    end

    # theta = F * Pi * inv(G), with F and G the Cholesky factors of Z'Z and X'X,
    # has identity covariance under iid errors.
    Fmatrix_chol = cholesky(Symmetric(Z_res' * Z_res); check = false)
    Gmatrix_chol = cholesky(Symmetric(Xendo_res' * Xendo_res); check = false)

    if !issuccess(Fmatrix_chol) || !issuccess(Gmatrix_chol)
        return T(NaN)
    end

    Fmatrix = Fmatrix_chol.U
    Gmatrix = Gmatrix_chol.U

    theta = Fmatrix * (Gmatrix' \ Pi')'

    svddecomp = svd(theta, full = true)
    u = svddecomp.U
    vt = svddecomp.Vt

    # Submatrices for the rank test (Kleibergen-Paap 2006, p. 102)
    u_sub = u[nendo:l, nendo:l]
    vt_sub = vt[nendo, nendo]

    if iszero(u_sub)
        a_qq = u[1:l, nendo:l]
    else
        a_qq = u[1:l, nendo:l] * (u_sub \ sqrt(u_sub * u_sub'))
    end

    if iszero(vt_sub)
        b_qq = vt[1:nendo, nendo]'
    else
        b_qq = sqrt(vt_sub * vt_sub') * (vt_sub' \ vt[1:nendo, nendo]')
    end

    kronv = kron(b_qq, a_qq')
    lambda = kronv * vec(theta)

    if k isa CovarianceMatrices.Uncorrelated
        vlab = Hermitian((T(c) / n) .* (kronv * kronv'))
    else
        meat = _kp_meat(Xendo_res, Z_res, k, c)
        K = kron(Gmatrix, Fmatrix)'
        vhat = K \ (K \ meat)'
        vlab = Hermitian(kronv * vhat * kronv')
    end
    vlab_chol = cholesky(vlab; check = false)
    issuccess(vlab_chol) || return T(NaN)
    r_kp = lambda' * (vlab_chol \ lambda)

    return r_kp[1]
end

# `c · n · aVar(k, M)` for the moments `M`, whose columns are `Z_res .* v_j` for each
# column `v_j` of the first-stage residuals `v`, endogenous variable by endogenous variable.
function _kp_meat(v, Z_res, k, c)
    M = reduce(hcat, (Z_res .* view(v, :, j) for j in axes(v, 2)))
    k = _uncached(k)
    A = _moment_aVar(k, M; demean = false, scale = false,
        weights = CovarianceMatrices.bandwidthweights(k, M))
    return eltype(M)(c) .* parent(A)
end

"""
    compute_first_stage_fstat(Xendo_res, Z_res, Pi, k, c)

Kleibergen-Paap rk Wald F statistic and its p-value, `(F_kp, p_kp)`, from
[`ranktest`](@ref) with the variance estimator `k` and finite-sample factor `c`.
`F_kp = r_kp / l` with `l` the number of excluded instruments, and `p_kp` is the
χ²(l − k + 1) p-value of `r_kp`. Both are `NaN` when the statistic is not defined.
"""
function compute_first_stage_fstat(
        Xendo_res::AbstractMatrix{T},
        Z_res::AbstractMatrix{T},
        Pi::AbstractMatrix{T},
        k,
        c::Real
) where {T <: AbstractFloat}
    nendo = size(Xendo_res, 2)
    l = size(Z_res, 2)

    r_kp = try
        ranktest(Xendo_res, Z_res, Pi, k, c)
    catch e
        _isnumericalfailure(e) || rethrow()
        @info "ranktest failed: $e; first-stage statistics not estimated"
        T(NaN)
    end
    isnan(r_kp) && return T(NaN), T(NaN)

    return T(r_kp / l), T(chisqccdf(l - nendo + 1, r_kp))
end

# Failures that mean a statistic is not defined for the data, rather than a bug.
_isnumericalfailure(e) = e isa Union{PosDefException, SingularException, LAPACKException}

##############################################################################
##
## Per-Endogenous First-Stage F-Statistics
##
##############################################################################

"""
    compute_per_endogenous_fstats(Xendo_res, Z_res, Pi, vcov_type, dof_fes;
                                   Xendo_orig=nothing, newZ=nothing, first_stage=nothing)

Compute F-statistics for each endogenous variable's first-stage regression.

For each endogenous variable j:
- First-stage: X_endo_j = Z * pi_j + e_j
- F-stat tests H0: pi_j = 0 (all excluded instrument coefficients = 0)

# Arguments
- `Xendo_res::Matrix{T}`: Residualized endogenous variables (n × k)
- `Z_res::Matrix{T}`: Residualized instruments (n × l)
- `Pi::Matrix{T}`: First-stage coefficient matrix (l × k), instruments portion only
- `vcov_type`: Variance estimator type (from CovarianceMatrices.jl)
- `dof_fes::Int`: Degrees of freedom absorbed by fixed effects
- `Xendo_orig::Union{Matrix{T}, Nothing}`: Original endogenous (for robust F)
- `newZ::Union{Matrix{T}, Nothing}`: Full first-stage design [Xexo, Z] (for robust F)
- `first_stage`: The first-stage regression whose finite-sample factor the variance uses
  (a `_FirstStage`), or `nothing` for that of the regression on `newZ` alone

# Returns
- `(F_stats, p_values)`: Vectors of F-statistics and p-values, one per endogenous variable
"""
function compute_per_endogenous_fstats(
        Xendo_res::Matrix{T},
        Z_res::Matrix{T},
        Pi::Matrix{T},
        vcov_type,
        dof_fes::Int;
        Xendo_orig::Union{Matrix{T}, Nothing} = nothing,
        newZ::Union{Matrix{T}, Nothing} = nothing,
        first_stage = nothing
) where {T <: AbstractFloat}
    l = size(Z_res, 2)      # Number of excluded instruments

    if !isnothing(Xendo_orig) && !isnothing(newZ)
        return _compute_first_stage_fstats_via_ols(
            newZ, Xendo_orig, l, vcov_type, dof_fes; first_stage
        )
    else
        throw(ArgumentError(
            "compute_per_endogenous_fstats requires Xendo_orig and newZ; " *
            "these must be provided from FirstStageData"))
    end
end

"""
    _compute_first_stage_fstats_via_ols(newZ, Xendo, n_instruments, vcov_type, dof_fes;
                                        first_stage = nothing)

Compute first-stage F-statistics by constructing lightweight OLSMatrixEstimator models
from pre-computed first-stage data and delegating variance computation to CovarianceMatrices.jl.

No refitting is performed — ZZ factorization is computed once and shared across all
endogenous variables. Per-endogenous cost is one matvec (for fitted values) plus the
CovarianceMatrices sandwich formula.

# Arguments
- `newZ::Matrix{T}`: Full first-stage design matrix [Xexo, Z] (n × (k_exo + l))
- `Xendo::Matrix{T}`: Original endogenous variables (n × k)
- `n_instruments::Int`: Number of excluded instruments (l)
- `vcov_type`: Any variance estimator supported by CovarianceMatrices.jl
- `dof_fes::Int`: Degrees of freedom absorbed by fixed effects
- `first_stage`: The first-stage regression whose finite-sample factor the variance uses
  (a `_FirstStage`), or `nothing` for that of the regression on `newZ` alone. The two
  differ for `CR0` and `CR1` when the model absorbs fixed effects, which enter the
  cluster K of `_cluster_dof` but are not columns of `newZ`.

# Returns
- `(F_stats, p_values)`: Vectors of F-statistics and p-values, one per endogenous
"""
function _compute_first_stage_fstats_via_ols(
        newZ::Matrix{T},
        Xendo::Matrix{T},
        n_instruments::Int,
        vcov_type,
        dof_fes::Int;
        first_stage = nothing
) where {T <: AbstractFloat}
    n, k_total = size(newZ)
    k = size(Xendo, 2)
    l = n_instruments
    k_exo = k_total - l

    F_stats = Vector{T}(undef, k)
    p_values = Vector{T}(undef, k)

    if l == 0
        fill!(F_stats, T(NaN))
        fill!(p_values, T(NaN))
        return F_stats, p_values
    end

    # Compute ZZ factorization ONCE for all endogenous
    ZZ = Matrix{T}(undef, k_total, k_total)
    BLAS.syrk!('U', 'T', one(T), newZ, zero(T), ZZ)
    ZZ_sym = Symmetric(ZZ, :U)
    ZZ_chol = cholesky(ZZ_sym; check = false)
    if !issuccess(ZZ_chol)
        fill!(F_stats, T(NaN))
        fill!(p_values, T(NaN))
        return F_stats, p_values
    end

    # Compute all coefficients at once: coef_all is k_total × k
    coef_all = ZZ_chol \ (newZ' * Xendo)

    # DOF for first-stage
    dof_fs = k_total
    dof_residual_fs = max(1, n - k_total - dof_fes)

    # Process each endogenous variable
    for j in 1:k
        y_j = Xendo[:, j]
        beta_j = coef_all[:, j]
        mu_j = newZ * beta_j  # fitted values (one matvec per endogenous)

        # Construct lightweight OLSMatrixEstimator wrapping pre-computed data
        rr = OLSResponse(y_j, mu_j, T[], T[], :first_stage)
        pp = OLSPredictorChol{T}(newZ, newZ, beta_j, ZZ_chol)
        rss_j = sum(abs2, y_j .- mu_j)
        tss_j = sum(abs2, y_j .- mean(y_j))
        basis = trues(k_total)

        vcov_placeholder = Symmetric(Matrix{T}(undef, k_total, k_total))
        fs_model = OLSMatrixEstimator{T, OLSPredictorChol{T}, typeof(vcov_type),
            typeof(vcov_placeholder)}(
            rr, pp, basis,
            n, dof_fs, dof_residual_fs,
            T(rss_j), T(tss_j), T(1 - rss_j / tss_j), true,
            _matrix_coefnames(nothing, k_total), Vector{Int}[], Int[], trues(n),
            vcov_type, vcov_placeholder,
            Vector{T}(undef, k_total), Vector{T}(undef, k_total), Vector{T}(undef, k_total)
        )

        # Use CovarianceMatrices to compute vcov — handles all HC/HAC/CR/EWC types
        vcov_matrix = try
            CovarianceMatrices.vcov(vcov_type, fs_model)
        catch e
            _isnumericalfailure(e) || rethrow()
            F_stats[j] = T(NaN)
            p_values[j] = T(NaN)
            continue
        end
        if !isnothing(first_stage)
            vcov_matrix = vcov_matrix .*
                          T(_finite_sample_factor(vcov_type, first_stage, true) /
                            _finite_sample_factor(vcov_type, fs_model, true))
        end

        # Extract instrument coefficient block (last l × l)
        vcov_pi = Symmetric(vcov_matrix[(k_exo + 1):end, (k_exo + 1):end])
        pi_j = beta_j[(k_exo + 1):end]

        # Wald test: F = (pi' * inv(V_pi) * pi) / l
        vcov_pi_chol = cholesky(vcov_pi; check = false)
        if !issuccess(vcov_pi_chol)
            F_stats[j] = T(NaN)
            p_values[j] = T(NaN)
            continue
        end

        chi2 = pi_j' * (vcov_pi_chol \ pi_j)
        F_j = chi2 / l

        F_stats[j] = F_j
        p_values[j] = fdistccdf(l, _cluster_test_dof(vcov_type, dof_residual_fs), F_j)
    end

    return F_stats, p_values
end
