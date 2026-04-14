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

"""
    ranktest(Xendo_res, Z_res, Pi, vcov_type, nobs, dof_small, dof_fes)

Compute the Kleibergen-Paap rk statistic for testing weak instruments.

# Arguments
- `Xendo_res::Matrix`: Residualized endogenous variables (n × k)
- `Z_res::Matrix`: Residualized instruments (n × l)
- `Pi::Matrix`: First-stage coefficient matrix
- `vcov_type`: Variance estimator type (from CovarianceMatrices.jl)
- `nobs::Int`: Number of observations
- `dof_small::Int`: Degrees of freedom (number of parameters)
- `dof_fes::Int`: Degrees of freedom absorbed by fixed effects

# Returns
- `r_kp`: The Kleibergen-Paap rk statistic
"""
function ranktest(
        Xendo_res::Matrix{T},
        Z_res::Matrix{T},
        Pi::Matrix{T},
        vcov_type,
        nobs::Int,
        dof_small::Int,
        dof_fes::Int
) where {T <: AbstractFloat}
    k = size(Xendo_res, 2)  # Number of endogenous variables
    l = size(Z_res, 2)      # Number of excluded instruments

    # Handle edge cases
    if k == 0 || l == 0
        return T(NaN)
    end

    # Compute theta = F * Pi * G' where F and G are Cholesky factors
    # This transforms the problem to have identity covariance
    ZZ = Symmetric(Z_res' * Z_res)
    XX = Symmetric(Xendo_res' * Xendo_res)

    Fmatrix_chol = cholesky(ZZ; check = false)
    Gmatrix_chol = cholesky(XX; check = false)

    if !issuccess(Fmatrix_chol) || !issuccess(Gmatrix_chol)
        return T(NaN)
    end

    Fmatrix = Fmatrix_chol.U
    Gmatrix = Gmatrix_chol.U

    # theta = F * Pi' * inv(G')
    theta = Fmatrix * (Gmatrix' \ Pi')'

    # Compute SVD decomposition
    svddecomp = svd(theta, full = true)
    u = svddecomp.U
    vt = svddecomp.Vt

    # Extract submatrices for the rank test (see Kleibergen-Paap p.102)
    u_sub = u[k:l, k:l]
    vt_sub = vt[k, k]

    # Compute a_qq and b_qq
    if iszero(u_sub)
        a_qq = u[1:l, k:l]
    else
        a_qq = u[1:l, k:l] * (u_sub \ sqrt(u_sub * u_sub'))
    end

    if iszero(vt_sub)
        b_qq = vt[1:k, k]'
    else
        b_qq = sqrt(vt_sub * vt_sub') * (vt_sub' \ vt[1:k, k]')
    end

    # Kronecker product for the test statistic
    kronv = kron(b_qq, a_qq')
    lambda = kronv * vec(theta)

    # Compute variance depending on vcov type
    # Note: HR0/HR1 (= HC0/HC1) are heteroskedasticity-robust estimators and should
    # use the sandwich form. Only truly homoskedastic estimators (Uncorrelated)
    # should use the simple formula.
    if vcov_type isa CovarianceMatrices.Uncorrelated
        # Truly homoskedastic case (assumes i.i.d. errors)
        vlab_chol = cholesky(Hermitian((kronv * kronv') ./ nobs); check = false)
        if !issuccess(vlab_chol)
            return T(NaN)
        end
        r_kp = lambda' * (vlab_chol \ lambda)
    else
        # Robust case (HC0/HC1/HC2/HC3/cluster-robust) - compute sandwich variance
        K = kron(Gmatrix, Fmatrix)'

        # Compute the "meat" using CovarianceMatrices.jl
        # For robust inference, we need the moment conditions
        # M = Z_res ⊗ (Xendo_res - Z_res * Pi') adjusted for vcov type

        # The meat is E[mm'] where m is the vectorized moment condition
        # For now, use a simplified approach based on the original Vcov.jl

        # Compute residuals from first stage: residuals_fs = Xendo_res - Z_res * Pi
        # Use in-place computation to avoid allocating a new matrix
        residuals_fs = copy(Xendo_res)
        BLAS.gemm!('N', 'N', -one(T), Z_res, Pi, one(T), residuals_fs)

        # Build moment matrix: kron of Z with residuals columns
        # For k=1 (single endogenous), this simplifies to Z_res .* residuals_fs
        # For k>1, we need the full Kronecker structure
        n = nobs
        kl = k * l
        if k == 1
            # Optimized path for single endogenous: avoid Kronecker allocation
            # moment_matrix[:, i] = Z_res[:, i] .* residuals_fs[:, 1]
            moment_matrix = Z_res .* residuals_fs
        else
            moment_matrix = Matrix{T}(undef, n, kl)
            @inbounds for j in 1:k
                res_j = @view residuals_fs[:, j]
                for i in 1:l
                    idx = (j - 1) * l + i
                    Z_i = @view Z_res[:, i]
                    mm_col = @view moment_matrix[:, idx]
                    @simd for row in 1:n
                        mm_col[row] = Z_i[row] * res_j[row]
                    end
                end
            end
        end

        # Compute meat using CovarianceMatrices
        meat = _compute_meat(moment_matrix, vcov_type, nobs, dof_small, dof_fes)

        # Transform variance
        vhat = K \ (K \ meat)'
        vlab_matrix = Hermitian(kronv * vhat * kronv')
        vlab_chol = cholesky(vlab_matrix; check = false)

        if !issuccess(vlab_chol)
            return T(NaN)
        end
        r_kp = lambda' * (vlab_chol \ lambda)
    end

    return r_kp[1]
end

"""
    _compute_meat(moment_matrix, vcov_type, nobs, dof_small, dof_fes)

Compute the meat of the sandwich estimator for the rank test.
"""
function _compute_meat(
        moment_matrix::Matrix{T},
        vcov_type,
        nobs::Int,
        dof_small::Int,
        dof_fes::Int
) where {T <: AbstractFloat}
    n = size(moment_matrix, 1)

    if vcov_type isa CovarianceMatrices.HR0
        # No adjustment
        return moment_matrix' * moment_matrix
    elseif vcov_type isa CovarianceMatrices.HR1
        # HC1 adjustment
        dof_residual = max(1, n - dof_small - dof_fes)
        scale = n / dof_residual
        return scale * (moment_matrix' * moment_matrix)
    elseif vcov_type isa Union{CovarianceMatrices.CR0, CovarianceMatrices.CR1}
        # Cluster-robust - vcov_type.g[1] is a Clustering struct
        clustering = vcov_type.g[1]
        groups = clustering.groups       # Vector{Int}, 1-indexed
        n_clusters = clustering.ngroups
        p = size(moment_matrix, 2)

        # Sum moments within clusters (column-major loop)
        cluster_sums = zeros(T, n_clusters, p)
        @inbounds for j in 1:p
            for i in 1:n
                cluster_sums[groups[i], j] += moment_matrix[i, j]
            end
        end

        meat = cluster_sums' * cluster_sums

        # Apply small-sample correction for CR1
        if vcov_type isa CovarianceMatrices.CR1
            dof_residual = max(1, n - dof_small - dof_fes)
            scale = (n_clusters / (n_clusters - 1)) * (n / dof_residual)
            meat *= scale
        end

        return meat
    else
        # Default: use HC1-like computation
        dof_residual = max(1, n - dof_small - dof_fes)
        scale = n / dof_residual
        return scale * (moment_matrix' * moment_matrix)
    end
end

"""
    compute_first_stage_fstat(Xendo_res, Z_res, Pi, vcov_type, nobs, dof_small, dof_fes)

Compute the Kleibergen-Paap first-stage F-statistic and p-value.

Returns (F_kp, p_kp) where:
- F_kp is the first-stage F-statistic
- p_kp is the p-value from chi-squared distribution
"""
function compute_first_stage_fstat(
        Xendo_res::Matrix{T},
        Z_res::Matrix{T},
        Pi::Matrix{T},
        vcov_type,
        nobs::Int,
        dof_small::Int,
        dof_fes::Int
) where {T <: AbstractFloat}
    k = size(Xendo_res, 2)  # Number of endogenous variables
    l = size(Z_res, 2)      # Number of excluded instruments

    try
        r_kp = ranktest(Xendo_res, Z_res, Pi, vcov_type, nobs, dof_small, dof_fes)

        if isnan(r_kp)
            return T(NaN), T(NaN)
        end

        # Degrees of freedom for chi-squared test
        df = l - k + 1

        # P-value from chi-squared distribution
        p_kp = chisqccdf(df, r_kp)

        # F-statistic: divide by number of instruments
        F_kp = r_kp / l

        return F_kp, p_kp
    catch e
        @info "ranktest failed: $e; first-stage statistics not estimated"
        return T(NaN), T(NaN)
    end
end

##############################################################################
##
## Per-Endogenous First-Stage F-Statistics
##
##############################################################################

"""
    compute_per_endogenous_fstats(Xendo_res, Z_res, Pi, vcov_type, nobs, dof_small, dof_fes;
                                   Xendo_orig=nothing, newZ=nothing)

Compute F-statistics for each endogenous variable's first-stage regression.

For each endogenous variable j:
- First-stage: X_endo_j = Z * pi_j + e_j
- F-stat tests H0: pi_j = 0 (all excluded instrument coefficients = 0)

# Arguments
- `Xendo_res::Matrix{T}`: Residualized endogenous variables (n × k)
- `Z_res::Matrix{T}`: Residualized instruments (n × l)
- `Pi::Matrix{T}`: First-stage coefficient matrix (l × k), instruments portion only
- `vcov_type`: Variance estimator type (from CovarianceMatrices.jl)
- `nobs::Int`: Number of observations
- `dof_small::Int`: Degrees of freedom (number of parameters)
- `dof_fes::Int`: Degrees of freedom absorbed by fixed effects
- `Xendo_orig::Union{Matrix{T}, Nothing}`: Original endogenous (for robust F)
- `newZ::Union{Matrix{T}, Nothing}`: Full first-stage design [Xexo, Z] (for robust F)

# Returns
- `(F_stats, p_values)`: Vectors of F-statistics and p-values, one per endogenous variable
"""
function compute_per_endogenous_fstats(
        Xendo_res::Matrix{T},
        Z_res::Matrix{T},
        Pi::Matrix{T},
        vcov_type,
        nobs::Int,
        dof_small::Int,
        dof_fes::Int;
        Xendo_orig::Union{Matrix{T}, Nothing} = nothing,
        newZ::Union{Matrix{T}, Nothing} = nothing
) where {T <: AbstractFloat}
    l = size(Z_res, 2)      # Number of excluded instruments

    if !isnothing(Xendo_orig) && !isnothing(newZ)
        return _compute_first_stage_fstats_via_ols(
            newZ, Xendo_orig, l, vcov_type, dof_fes
        )
    else
        throw(ArgumentError(
            "compute_per_endogenous_fstats requires Xendo_orig and newZ; " *
            "these must be provided from FirstStageData"))
    end
end

"""
    _compute_first_stage_fstats_via_ols(newZ, Xendo, n_instruments, vcov_type, dof_fes)

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

# Returns
- `(F_stats, p_values)`: Vectors of F-statistics and p-values, one per endogenous
"""
function _compute_first_stage_fstats_via_ols(
        newZ::Matrix{T},
        Xendo::Matrix{T},
        n_instruments::Int,
        vcov_type,
        dof_fes::Int
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

        fs_model = OLSMatrixEstimator{T, OLSPredictorChol{T}, typeof(vcov_type)}(
            rr, pp, basis,
            n, dof_fs, dof_residual_fs,
            T(rss_j), T(tss_j), T(1 - rss_j / tss_j), true,
            vcov_type, Symmetric(Matrix{T}(undef, k_total, k_total)),  # placeholder
            Vector{T}(undef, k_total), Vector{T}(undef, k_total), Vector{T}(undef, k_total)
        )

        # Use CovarianceMatrices to compute vcov — handles all HC/HAC/CR/EWC types
        vcov_matrix = try
            CovarianceMatrices.vcov(vcov_type, fs_model)
        catch
            F_stats[j] = T(NaN)
            p_values[j] = T(NaN)
            continue
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
        p_values[j] = fdistccdf(l, dof_residual_fs, F_j)
    end

    return F_stats, p_values
end
