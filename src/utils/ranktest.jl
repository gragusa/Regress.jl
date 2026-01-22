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
        # Cluster-robust - vcov_type.g contains actual cluster vectors (not symbols)
        clusters = vcov_type.g[1]
        unique_clusters = unique(clusters)
        n_clusters = length(unique_clusters)

        # Create mapping from cluster value to index
        cluster_to_idx = Dict(c => i for (i, c) in enumerate(unique_clusters))

        # Sum moments within clusters
        cluster_sums = zeros(T, n_clusters, size(moment_matrix, 2))
        for (i, c) in enumerate(clusters)
            idx = cluster_to_idx[c]
            cluster_sums[idx, :] .+= moment_matrix[i, :]
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
    k = size(Xendo_res, 2)  # Number of endogenous variables
    l = size(Z_res, 2)      # Number of excluded instruments

    # Use original data for robust F-stats if available
    use_original = !isnothing(Xendo_orig) && !isnothing(newZ) &&
                   !(vcov_type isa CovarianceMatrices.Uncorrelated)

    if use_original
        # Use batched version for better performance - computes ZZ etc. only once
        return _compute_robust_first_stage_fstats_batched(
            newZ, Xendo_orig, l, vcov_type, nobs, dof_small, dof_fes
        )
    else
        # Fall back to per-variable computation for classical F-test
        F_stats = Vector{T}(undef, k)
        p_values = Vector{T}(undef, k)

        for j in 1:k
            residuals_j = Xendo_res[:, j] .- Z_res * Pi[:, j]
            F_j,
            p_j = _compute_single_first_stage_fstat(
                Z_res, Pi[:, j], residuals_j, vcov_type, nobs, dof_small, dof_fes
            )
            F_stats[j] = F_j
            p_values[j] = p_j
        end

        return F_stats, p_values
    end
end

"""
    _compute_single_first_stage_fstat(Z, pi, residuals, vcov_type, nobs, dof_small, dof_fes)

Compute F-statistic for a single first-stage regression.

Tests H0: pi = 0 (all excluded instrument coefficients are zero).

# Arguments
- `Z::Matrix{T}`: Instrument matrix (n × l)
- `pi::Vector{T}`: First-stage coefficient vector (l × 1)
- `residuals::Vector{T}`: First-stage residuals (n × 1)
- `vcov_type`: Variance estimator type
- `nobs::Int`: Number of observations
- `dof_small::Int`: Degrees of freedom
- `dof_fes::Int`: Degrees of freedom absorbed by FE

# Returns
- `(F, p)`: F-statistic and p-value
"""
function _compute_single_first_stage_fstat(
        Z::Matrix{T},
        pi::Vector{T},
        residuals::Vector{T},
        vcov_type,
        nobs::Int,
        dof_small::Int,
        dof_fes::Int
) where {T <: AbstractFloat}
    l = length(pi)  # Number of excluded instruments

    l == 0 && return T(NaN), T(NaN)

    dof_residual = max(1, nobs - dof_small - dof_fes)

    # Compute Z'Z and its Cholesky
    ZZ = Symmetric(Z' * Z)
    ZZ_chol = cholesky(ZZ; check = false)
    !issuccess(ZZ_chol) && return T(NaN), T(NaN)

    # Note: HR0/HR1 (= HC0/HC1) are heteroskedasticity-robust estimators.
    # Only truly homoskedastic estimators (Uncorrelated) should use classical F-test.
    if vcov_type isa CovarianceMatrices.Uncorrelated
        # Classical F-test (homoskedastic): F = (pi' * Z'Z * pi) / (l * s²)
        rss = sum(abs2, residuals)
        s2 = rss / dof_residual

        # Wald test: chi² = pi' * (Z'Z) * pi / s²
        chi2 = (pi' * ZZ * pi) / s2
        F = chi2 / l

        p = fdistccdf(l, dof_residual, F)
        return F, p
    else
        # Robust F-test (HC0/HC1/HC2/HC3/cluster) using sandwich variance
        # Build moment matrix for this first-stage: M = Z .* residuals
        M = Z .* residuals

        # Compute meat of sandwich
        meat = _compute_meat(M, vcov_type, nobs, dof_small, dof_fes)

        # Bread: inv(Z'Z)
        invZZ = inv(ZZ_chol)

        # Sandwich vcov: invZZ * (n * Meat) * invZZ
        # Note: _compute_meat already includes scaling
        vcov_pi = invZZ * meat * invZZ

        # Wald test: chi² = pi' * inv(V) * pi
        vcov_pi_chol = cholesky(Symmetric(vcov_pi); check = false)
        !issuccess(vcov_pi_chol) && return T(NaN), T(NaN)

        chi2 = pi' * (vcov_pi_chol \ pi)
        F = chi2 / l

        p = fdistccdf(l, dof_residual, F)
        return F, p
    end
end

"""
    _compute_robust_first_stage_fstats_batched(newZ, Xendo, n_instruments, vcov_type, nobs, dof_small, dof_fes)

Batched version that computes robust F-statistics for ALL endogenous variables at once.
Avoids redundant computation of ZZ, ZZ_chol, invZZ which are shared across all k endogenous.

# Arguments
- `newZ::Matrix{T}`: Full first-stage design matrix [Xexo, Z] (n × (k_exo + l))
- `Xendo::Matrix{T}`: Original endogenous variables (n × k)
- `n_instruments::Int`: Number of excluded instruments (l)
- `vcov_type`: Variance estimator type (HC0/HC1/etc.)
- `nobs::Int`: Number of observations
- `dof_small::Int`: Total DOF for second-stage
- `dof_fes::Int`: Degrees of freedom absorbed by FE

# Returns
- `(F_stats, p_values)`: Vectors of F-statistics and p-values, one per endogenous
"""
function _compute_robust_first_stage_fstats_batched(
        newZ::Matrix{T},
        Xendo::Matrix{T},
        n_instruments::Int,
        vcov_type,
        nobs::Int,
        dof_small::Int,
        dof_fes::Int
) where {T <: AbstractFloat}
    n, k_total = size(newZ)  # k_total = k_exo + l
    k = size(Xendo, 2)       # Number of endogenous variables
    l = n_instruments
    k_exo = k_total - l

    F_stats = Vector{T}(undef, k)
    p_values = Vector{T}(undef, k)

    if l == 0
        fill!(F_stats, T(NaN))
        fill!(p_values, T(NaN))
        return F_stats, p_values
    end

    # Compute ZZ and its factorization ONCE for all endogenous
    ZZ = Matrix{T}(undef, k_total, k_total)
    BLAS.syrk!('U', 'T', one(T), newZ, zero(T), ZZ)
    ZZ_sym = Symmetric(ZZ, :U)
    ZZ_chol = cholesky(ZZ_sym; check = false)
    if !issuccess(ZZ_chol)
        fill!(F_stats, T(NaN))
        fill!(p_values, T(NaN))
        return F_stats, p_values
    end

    # Compute Zy for ALL endogenous at once: newZ' * Xendo is k_total × k
    Zy_all = newZ' * Xendo  # k_total × k

    # Compute all coefficients at once: coef_full_all is k_total × k
    coef_full_all = ZZ_chol \ Zy_all

    # Compute all residuals at once: Xendo - newZ * coef_full_all is n × k
    # Use in-place update: residuals_all = Xendo - newZ * coef_full_all
    residuals_all = copy(Xendo)
    BLAS.gemm!('N', 'N', -one(T), newZ, coef_full_all, one(T), residuals_all)

    # invZZ needed for sandwich - compute once
    invZZ = inv(ZZ_chol)

    # DOF for first-stage
    dof_residual_fs = max(1, n - k_total - dof_fes)

    # Pre-allocate buffers reused across all endogenous variables
    M = Matrix{T}(undef, n, k_total)
    meat_buf = Matrix{T}(undef, k_total, k_total)
    vcov_full = Matrix{T}(undef, k_total, k_total)
    tmp_buf = Matrix{T}(undef, k_total, k_total)
    pi_tmp = Vector{T}(undef, l)

    # Process each endogenous variable
    for j in 1:k
        residuals_j = @view residuals_all[:, j]

        # Build moment matrix in-place: M = newZ .* residuals_j
        @inbounds for col in 1:k_total
            @simd for row in 1:n
                M[row, col] = newZ[row, col] * residuals_j[row]
            end
        end

        # Compute meat of sandwich using pre-allocated buffer
        _compute_meat_inplace!(meat_buf, M, vcov_type, nobs, k_total, dof_fes)

        # Full sandwich vcov: vcov_full = invZZ * meat * invZZ
        # Use tmp_buf for intermediate result
        mul!(tmp_buf, invZZ, meat_buf)
        mul!(vcov_full, tmp_buf, invZZ)

        # Extract instrument coefficient variances (last l × l block)
        vcov_pi = @view vcov_full[(k_exo + 1):end, (k_exo + 1):end]
        pi_j = @view coef_full_all[(k_exo + 1):end, j]

        # Wald test: chi² = pi' * inv(V_pi) * pi
        vcov_pi_chol = cholesky(Symmetric(vcov_pi); check = false)
        if !issuccess(vcov_pi_chol)
            F_stats[j] = T(NaN)
            p_values[j] = T(NaN)
            continue
        end

        # Copy pi_j to avoid issues with views in ldiv!
        copyto!(pi_tmp, pi_j)
        ldiv!(vcov_pi_chol, pi_tmp)
        chi2 = dot(pi_j, pi_tmp)
        F_j = chi2 / l

        F_stats[j] = F_j
        p_values[j] = fdistccdf(l, dof_residual_fs, F_j)
    end

    return F_stats, p_values
end

"""
    _compute_meat_inplace!(dest, moment_matrix, vcov_type, nobs, dof_small, dof_fes)

In-place version of _compute_meat that writes result to dest.
"""
function _compute_meat_inplace!(
        dest::Matrix{T},
        moment_matrix::Matrix{T},
        vcov_type,
        nobs::Int,
        dof_small::Int,
        dof_fes::Int
) where {T <: AbstractFloat}
    n = size(moment_matrix, 1)

    if vcov_type isa CovarianceMatrices.HR0
        # No adjustment: dest = M' * M
        BLAS.syrk!('U', 'T', one(T), moment_matrix, zero(T), dest)
        # Copy upper to lower
        @inbounds for j in 1:size(dest, 2), i in (j + 1):size(dest, 1)

            dest[i, j] = dest[j, i]
        end
    elseif vcov_type isa CovarianceMatrices.HR1
        # HC1 adjustment
        dof_residual = max(1, n - dof_small - dof_fes)
        scale = T(n) / T(dof_residual)
        BLAS.syrk!('U', 'T', scale, moment_matrix, zero(T), dest)
        @inbounds for j in 1:size(dest, 2), i in (j + 1):size(dest, 1)

            dest[i, j] = dest[j, i]
        end
    elseif vcov_type isa Union{CovarianceMatrices.CR0, CovarianceMatrices.CR1}
        # Cluster-robust - fall back to allocating version for simplicity
        meat = _compute_meat(moment_matrix, vcov_type, nobs, dof_small, dof_fes)
        copyto!(dest, meat)
    else
        # Default: use HC1-like computation
        dof_residual = max(1, n - dof_small - dof_fes)
        scale = T(n) / T(dof_residual)
        BLAS.syrk!('U', 'T', scale, moment_matrix, zero(T), dest)
        @inbounds for j in 1:size(dest, 2), i in (j + 1):size(dest, 1)

            dest[i, j] = dest[j, i]
        end
    end

    return dest
end
