##############################################################################
##
## Montiel-Olea-Pflueger Weak Instrument Test
##
## Implements:
##   - Effective F-statistic (MOP 2013)
##   - Non-robust F-statistic (MOP eq. 14)
##   - Robust F-statistic (Windmeijer 2025)
##   - TSLS, LIML, GMMf estimator coefficients and standard errors
##   - Critical values (Patnaik approximation for TSLS/LIML, noncentral chi-squared for GMMf)
##   - Worst-case Nagar bias bounds (B_TSLS, B_LIML, B_GMMf)
##
## Reference:
##   Montiel Olea, J.L. and C.E. Pflueger (2013), "A robust test for
##   weak instruments", Journal of Business and Economic Statistics, 31:358-369.
##
##   Windmeijer, F. (2025), "The robust F-statistic as a Test for Weak
##   Instruments", Journal of Econometrics.
##
## Restriction: Single endogenous regressor only.
##
##############################################################################

using LinearAlgebra: tr, eigvals, Symmetric, qr, cholesky, diag, dot
using StatsFuns: nchisqcdf

##############################################################################
## Result struct
##############################################################################

"""
    WeakIVTestResult{T}

Result of the Montiel-Olea-Pflueger robust weak instrument test.

Contains test statistics, estimator coefficients and standard errors,
critical values at various bias thresholds, and metadata.

# Fields
- `F_eff::T`: Effective F-statistic (MOP)
- `F_nonrobust::T`: Non-robust F-statistic
- `F_robust::T`: Robust F-statistic (Windmeijer)
- `btsls::T`, `sebtsls::T`: TSLS coefficient and standard error
- `bliml::T`, `sebliml::T`: LIML coefficient and standard error
- `bgmmf::T`, `sebgmmf::T`: GMMf coefficient and standard error
- `kappa::T`: LIML kappa
- `cv_TSLS::NTuple{4,T}`: TSLS critical values at τ ∈ {5%, 10%, 20%, 30%}
- `cv_LIML::NTuple{4,T}`: LIML critical values at τ ∈ {5%, 10%, 20%, 30%}
- `cv_GMMf::NTuple{4,T}`: GMMf critical values at τ ∈ {5%, 10%, 20%, 30%}
- `level::T`: Confidence level alpha
- `K::Int`: Number of excluded instruments
- `N::Int`: Number of observations
"""
struct WeakIVTestResult{T} <: AbstractTest
    # Test statistics
    F_eff::T
    F_nonrobust::T
    F_robust::T

    # Estimator results
    btsls::T
    sebtsls::T
    bliml::T
    sebliml::T
    bgmmf::T
    sebgmmf::T
    kappa::T

    # Critical values (TSLS, LIML, GMMf) at τ ∈ {5%, 10%, 20%, 30%}
    cv_TSLS::NTuple{4, T}
    cv_LIML::NTuple{4, T}
    cv_GMMf::NTuple{4, T}

    # Metadata
    level::T
    K::Int
    N::Int
end

##############################################################################
## Main entry point
##############################################################################

"""
    weakivtest(m::IVEstimator; level=0.05, tol=0.001, benchmark=:nagar) -> WeakIVTestResult

Compute the Montiel-Olea-Pflueger robust weak instrument test as a
post-estimation command on an IV model.

Requires a single endogenous regressor. The model must have been fit with
post-estimation data stored (the default).

The test uses the model's variance estimator, `m.vcov_estimator`, with the
finite-sample factor `vcov` applies to the first-stage regression; for `HC1` and `CR1`
this is Stata `weakivtest`'s W. It supports `Uncorrelated`, `HC0`–`HC5`, `CR0`, `CR1`
(any number of cluster dimensions) and the HAC kernels, and throws an `ArgumentError`
for any other estimator. `HC2`–`HC5` rescale each moment by the leverage `h` of the
first-stage design (the exogenous regressors and the excluded instruments), as `vcov`
rescales the residuals of an OLS fit: by `1/√(1 - h)`, `1/(1 - h)`, or the adaptive
exponents of HC4, HC4m and HC5. With one excluded instrument the robust F is then the
first-stage Wald F under the same estimator. They need that leverage, so they are not
supported on a model with absorbed fixed effects.

# Arguments
- `m::IVEstimator`: A fitted IV model (TSLS, LIML, etc.)
- `level::Real`: Confidence level alpha (default: 0.05)
- `tol::Real`: Convergence tolerance for bias optimization (default: 0.001)
- `benchmark::Symbol`: Bias benchmark — `:nagar` (default, MOP) or `:ols` (Windmeijer OLS)

# Returns
A `WeakIVTestResult` containing effective F, robust F, critical values, etc.

# Examples
```julia
model = iv(TSLS(), df, @formula(y ~ exper + expersq + (educ ~ age + kidslt6 + kidsge6)))
r = weakivtest(model)
r.F_eff           # Effective F-statistic
r.cv_TSLS         # TSLS critical values at 5%, 10%, 20%, 30%
```
"""
function weakivtest(m::IVEstimator{T}; level::Real = 0.05, tol::Real = 0.001,
        benchmark::Symbol = :nagar) where {T}
    # Validate
    pe = m.postestimation
    isnothing(pe) &&
        throw(ArgumentError("Model does not have post-estimation data. Re-fit with save=true."))
    fsd = pe.first_stage_data
    !has_first_stage_data(fsd) &&
        throw(ArgumentError("First-stage data not available. Re-fit with save=true."))

    n_endo = size(fsd.Xendo_res, 2)
    n_endo == 1 ||
        throw(ArgumentError("weakivtest requires exactly one endogenous regressor, got $n_endo."))

    k = _weakiv_estimator(m)

    K = size(fsd.Z_res, 2)  # Number of excluded instruments
    S = nobs(m)              # Number of observations

    # The first-stage design is [Xexo, Z]
    k_exo = fsd.n_exo
    Xexo = fsd.newZ[:, 1:k_exo]
    Xendo_orig = fsd.Xendo_orig[:, 1]
    Z_res = fsd.Z_res

    return _weakivtest_core(
        pe.y, Xexo, Xendo_orig, Z_res,
        k_exo, K, S, _FirstStage(m), k,
        T(level), T(tol), benchmark
    )
end

"""
    weakivtest(m::IVMatrixEstimator; level=0.05, tol=0.001, benchmark=:nagar) -> WeakIVTestResult

Compute the Montiel-Olea-Pflueger robust weak instrument test for a matrix-based
IV model (as used by LocalProjections.jl).

Requires a single endogenous regressor.

# Arguments
- `m::IVMatrixEstimator`: A fitted IV model from matrix-based estimation
- `level::Real`: Confidence level alpha (default: 0.05)
- `tol::Real`: Convergence tolerance for bias optimization (default: 0.001)
- `benchmark::Symbol`: Bias benchmark — `:nagar` (default, MOP) or `:ols` (Windmeijer OLS)

# Returns
A `WeakIVTestResult` containing effective F, robust F, critical values, etc.
"""
function weakivtest(m::IVMatrixEstimator{T}; level::Real = 0.05, tol::Real = 0.001,
        benchmark::Symbol = :nagar) where {T}
    pe = m.postestimation
    n_endo = length(pe.endo_cols)
    n_endo == 1 ||
        throw(ArgumentError("weakivtest requires exactly one endogenous regressor, got $n_endo."))
    k = _weakiv_estimator(m)

    # Z holds the exogenous columns of X, in order, then the excluded instruments.
    exo_cols = _exo_cols(pe)
    k_exo = length(exo_cols)
    Xexo = pe.X[:, exo_cols]
    Xendo_orig = pe.X[:, only(pe.endo_cols)]

    Z_instr = pe.Z[:, (k_exo + 1):end]
    K = size(Z_instr, 2)

    # Residualize excluded instruments from exogenous regressors
    if k_exo > 0
        QR_exo = qr(Xexo)
        Q_exo = Matrix(QR_exo.Q)
        Z_res = Z_instr .- Q_exo * (Q_exo' * Z_instr)
    else
        Z_res = copy(Z_instr)
    end

    S = nobs(m)

    return _weakivtest_core(
        pe.y, Xexo, Xendo_orig, Z_res,
        k_exo, K, S, _MatrixFirstStage(m, k_exo + K), k,
        T(level), T(tol), benchmark
    )
end

# The heteroskedasticity-robust estimators that rescale residuals by their leverage.
const _LeverageHR = Union{CovarianceMatrices.HR2, CovarianceMatrices.HR3,
    CovarianceMatrices.HR4, CovarianceMatrices.HR4m, CovarianceMatrices.HR5}

# The variance estimator of `m`, with column-named clusters resolved, checked to be one
# that the test supports.
function _weakiv_estimator(m)
    k = m.vcov_estimator
    if k isa _LeverageHR
        dof_fes(m) == 0 || throw(ArgumentError(
            "The weak-instrument test with $(nameof(typeof(k))) needs the leverage of the " *
            "first-stage regression, which is not computed for a model with absorbed fixed " *
            "effects. Use HC0 or HC1, or enter the fixed effects as dummy regressors."))
        return k
    end
    _kp_supported(k) || throw(ArgumentError(
        "The weak-instrument test is not defined for $(nameof(typeof(k))). It supports " *
        "Uncorrelated, HC0–HC5, CR0, CR1 (any number of cluster dimensions) and the HAC " *
        "kernels (Bartlett, Parzen, QuadraticSpectral, TukeyHanning, Truncated). Use, for " *
        "example, `m + vcov(HC1())`."))
    return _uncached(_resolve_clusters(k, m))
end

# The leverage `h` of a first-stage design with `ncoef` columns, for
# `CovarianceMatrices.residual_adjustment`.
struct _LeverageDesign{T} <: StatsAPI.RegressionModel
    h::Vector{T}
    ncoef::Int
end
CovarianceMatrices.leverage(d::_LeverageDesign) = d.h
CovarianceMatrices.numobs(d::_LeverageDesign) = length(d.h)
StatsAPI.coef(d::_LeverageDesign) = zeros(d.ncoef)

# The row weights of the moments and the estimator of their variance: for HC2–HC5, the
# residual adjustment of the estimator at the leverage of the design `[Q_exo Zs]`
# (orthonormal columns `Q_exo`, and `Zs` with `Zs'Zs = S·I` and orthogonal to `Q_exo`),
# with HC0 on the rescaled moments; for any other estimator, no weights.
function _moment_weights(k::_LeverageHR, Q_exo, Zs, S)
    h = vec(sum(abs2, Zs; dims = 2)) ./ S
    Q_exo === nothing || (h .+= vec(sum(abs2, Q_exo; dims = 2)))
    ncoef = size(Zs, 2) + (Q_exo === nothing ? 0 : size(Q_exo, 2))
    a = CovarianceMatrices.residual_adjustment(k, _LeverageDesign(h, ncoef))
    return a, CovarianceMatrices.HR0()
end
_moment_weights(k, Q_exo, Zs, S) = (true, k)

# The first-stage regression of a matrix-fitted IV model `m`: `ncoef` counts the
# exogenous regressors and the excluded instruments, and the fixed effects absorbed in
# `m` are absorbed in it too.
struct _MatrixFirstStage{M}
    m::M
    ncoef::Int
end
StatsAPI.nobs(f::_MatrixFirstStage) = nobs(f.m)
StatsAPI.dof_residual(f::_MatrixFirstStage) = nobs(f.m) - f.ncoef - dof_fes(f.m)
_cluster_dof(k, f::_MatrixFirstStage) = _cluster_dof(k, f.m) - sum(_CM.mask(f.m)) + f.ncoef

##############################################################################
## Core computation (shared by IVEstimator and IVMatrixEstimator methods)
##############################################################################

"""
    _weakivtest_core(y, Xexo, Xendo_orig, Z_res, k_exo, K, S, fs,
                     vcov_est, level, eps, benchmark) -> WeakIVTestResult

Internal computational core for the weak IV test. Both `weakivtest(::IVEstimator)`
and `weakivtest(::IVMatrixEstimator)` delegate here after extracting data.

# Arguments
- `y`: Response vector of the fit (demeaned and weighted as its regressors)
- `Xexo`: Exogenous regressors (`k_exo` columns)
- `Xendo_orig`: Original endogenous variable (vector)
- `Z_res`: Excluded instruments residualized from exogenous regressors
- `k_exo`: Number of exogenous regressors (including intercept column if present)
- `K`: Number of excluded instruments
- `S`: Number of observations
- `fs`: The first-stage regression (a `_FirstStage` or `_MatrixFirstStage`), which
  gives the degrees of freedom of Ω and the finite-sample factor of W
- `vcov_est`: Variance estimator, one that `_kp_supported` accepts, with its clusters
  resolved and without a cache
- `level`: Confidence level alpha
- `eps`: Convergence tolerance
- `benchmark`: Bias benchmark (:nagar or :ols)
"""
function _weakivtest_core(
        y_full::Vector{T}, Xexo::Matrix{T},
        Xendo_orig::Vector{T}, Z_res::Matrix{T},
        k_exo::Int, K::Int, S::Int, fs,
        vcov_est, level::T, eps::T, benchmark::Symbol
) where {T}
    # Step 1: Partial out exogenous regressors from y and Xendo
    Q_exo = nothing
    if k_exo > 0
        QR_exo = qr(Xexo)
        Q_exo = Matrix(QR_exo.Q)
        y_res = y_full .- Q_exo * (Q_exo' * y_full)
        Xendo_demeaned = Xendo_orig .- Q_exo * (Q_exo' * Xendo_orig)
    else
        y_res = copy(y_full)
        Xendo_demeaned = copy(Xendo_orig)
    end

    # Step 2: Orthogonalize instruments via QR, matching Stata's `orthog`
    # Stata's `orthog` produces orthogonal columns with sum(Zs_j^2) = n.
    # QR gives Z_res = Q*R with Q orthonormal (||Q_j|| = 1).
    # To match Stata: Zs = Q * sqrt(S), so Zs_j'Zs_j = S.
    # F_eff is NOT basis-invariant, so we must match Stata's scaling.
    QR_Z = qr(Z_res)
    Zs = Matrix(QR_Z.Q)[:, 1:K] .* sqrt(T(S))  # n × K, orthogonal, Zs'Zs = S*I
    # Row weights of every moment matrix below, and the estimator of their variance.
    weights, meat_est = _moment_weights(vcov_est, Q_exo, Zs, S)

    # Step 3: Reduced-form regressions (noconstant, Zs orthogonal with Zs'Zs = S*I)
    pihat = (Zs' * Xendo_demeaned) ./ T(S)   # K × 1
    res_endo = Xendo_demeaned .- Zs * pihat

    dhat = (Zs' * y_res) ./ T(S)              # K × 1
    res_y = y_res .- Zs * dhat

    # Step 4: Compute Omega, the residual covariance of the reduced form, with the
    # degrees of freedom of the first-stage regression
    dof_omega = dof_residual(fs)

    res_matrix = hcat(res_y, res_endo)  # n × 2
    Omega_unscaled = res_matrix' * res_matrix  # 2 × 2
    Omega = Omega_unscaled ./ dof_omega

    # Step 5: Compute W, the variance of the stacked moments [res_y .* Zs, res_endo .* Zs],
    # with the finite-sample factor `vcov` applies to the first-stage regression. For
    # HC1 and CR1 this is Stata weakivtest's W.
    W = if _is_homoskedastic(vcov_est)
        # Ω ⊗ Zs'Zs/S, and Zs'Zs/S = I.
        kron(Omega, Matrix{T}(I, K, K))
    else
        M_stacked = hcat(res_y .* Zs, res_endo .* Zs) .* weights  # n × 2K
        factor = vcov_est isa _LeverageHR ? one(T) : T(_kp_factor(vcov_est, fs))
        factor .* _weakiv_meat(M_stacked, meat_est) ./ S
    end  # 2K × 2K

    omega_22 = Omega[2, 2]

    # Step 6: Extract W submatrices
    W_1 = W[1:K, 1:K]
    W_12 = W[1:K, (K + 1):(2K)]
    W_2 = W[(K + 1):(2K), (K + 1):(2K)]

    # Step 7: Compute LIML kappa
    Omega_raw = Omega_unscaled ./ S
    Omega_raw_mhalf = _matrix_power_sym(Omega_raw, -0.5)
    ww = hcat(y_res, Xendo_demeaned)
    ww_mat = (ww' * ww) ./ S
    kappa_val = minimum(eigvals(Symmetric(Omega_raw_mhalf * ww_mat * Omega_raw_mhalf)))

    # Step 8: Compute estimator coefficients
    pipi = dot(pihat, pihat)
    btsls_val = dot(pihat, dhat) / pipi

    # LIML
    xhatl = Xendo_demeaned .- kappa_val .* res_endo
    limlden = dot(xhatl, Xendo_demeaned) / S
    limlnum = dot(xhatl, y_res) / S
    bliml_val = limlnum / limlden

    # GMMf
    W_2_inv = inv(Symmetric(W_2))
    piW2inv = W_2_inv * pihat
    bgmmf_val = dot(pihat, W_2_inv * dhat) / dot(pihat, piW2inv)

    # Step 9: Compute standard errors
    # TSLS SE
    res_tsls = y_res .- Xendo_demeaned .* btsls_val
    S_1tsls = _weakiv_compute_avar_residuals(res_tsls .* weights, Zs, meat_est, S)
    sebtsls_val = sqrt(
        (1 / pipi) * dot(pihat, S_1tsls * pihat) * (1 / pipi) / S
    )

    # LIML SE
    res_liml = y_res .- Xendo_demeaned .* bliml_val
    is_robust = !_is_homoskedastic(vcov_est)
    if is_robust
        S_1liml = _weakiv_compute_avar_residuals(res_liml .* weights, Zs, meat_est, S)
        sebliml_val = sqrt(
            (1 / limlden) * dot(pihat, S_1liml * pihat) * (1 / limlden) / S
        )
    else
        limlres2 = dot(res_liml, res_liml)
        sebliml_val = sqrt(limlres2 / (S * limlden))
    end

    # GMMf SE
    res_gmmf = y_res .- Xendo_demeaned .* bgmmf_val
    S_1gmmf = _weakiv_compute_avar_residuals(res_gmmf .* weights, Zs, meat_est, S)
    piW2invpi = dot(pihat, piW2inv)
    sebgmmf_val = sqrt(
        (1 / piW2invpi) * dot(piW2inv, S_1gmmf * piW2inv) * (1 / piW2invpi) / S
    )

    # Step 10: Compute F-statistics
    F_nonrobust_val = S * pipi / (K * omega_22)
    F_eff_val = S * pipi / tr(W_2)
    F_robust_val = S * dot(pihat, W_2_inv * pihat) / K

    # Step 11: Compute bias bounds (B_TSLS, B_LIML, B_GMMf). Each is assigned once: the
    # critical-value closures below capture them, and a variable assigned in several
    # branches would be captured as a `Core.Box`.
    B_TSLS = benchmark == :nagar ? _compute_BTSLS_nagar(W_1, W_12, W_2, T(eps)) :
             _compute_BTSLS_ols(Omega, W_1, W_12, W_2, T(eps))
    B_LIML = _compute_BLIML_nagar(W_1, W_12, W_2, Omega, T(eps))

    # B_GMMf computation
    W_2mh = _matrix_power_sym(W_2, -0.5)
    trW_1s = tr(W_2mh * W_1 * W_2mh)
    W_12s = W_2mh * W_12 * W_2mh
    trW_12s = tr(W_12s)
    W_12s_sym = (W_12s .+ W_12s') ./ 2
    eigs_12s = eigvals(Symmetric(W_12s_sym))
    mineig = minimum(eigs_12s)
    maxeig = maximum(eigs_12s)

    B_GMMf = benchmark == :nagar ?
             _compute_BGMMf_nagar(trW_1s, trW_12s, mineig, maxeig, K, bgmmf_val) :
             _compute_BGMMf_ols(Omega, trW_1s, trW_12s, mineig, maxeig, K, bgmmf_val)

    # Step 12: Compute critical values
    tau_x = (T(20), T(10), T(5), T(10 / 3))

    cv_TSLS = ntuple(i -> begin
            x = tau_x[i] * B_TSLS
            _patnaik_critical_value(W_2, T(level), x)
        end, 4)

    cv_LIML = ntuple(i -> begin
            x = tau_x[i] * B_LIML
            _patnaik_critical_value(W_2, T(level), x)
        end, 4)

    cv_GMMf = ntuple(i -> begin
            x_gmmf = tau_x[i] * B_GMMf * K
            _invnchisq(T(K), x_gmmf, T(1) - T(level)) / K
        end, 4)

    return WeakIVTestResult{T}(
        F_eff_val, F_nonrobust_val, F_robust_val,
        btsls_val, sebtsls_val,
        bliml_val, sebliml_val,
        bgmmf_val, sebgmmf_val,
        kappa_val,
        cv_TSLS, cv_LIML, cv_GMMf,
        T(level), K, S
    )
end

##############################################################################
## Helper: check if vcov estimator is homoskedastic
##############################################################################

_is_homoskedastic(v) = v isa CovarianceMatrices.Uncorrelated

# `n · aVar(k, M)` of the moments `M`, as in the Kleibergen-Paap statistic.
function _weakiv_meat(M, k)
    A = _moment_aVar(k, M; demean = false, scale = false,
        weights = CovarianceMatrices.bandwidthweights(k, M))
    return parent(A)
end

##############################################################################
## Helper: compute avar of residuals projected on instruments
##############################################################################

"""
    _weakiv_compute_avar_residuals(resid, Zstar, vcov_est, n)

Compute the avar matrix S_1 of the moments `Zstar .* resid`, `aVar` with no
finite-sample factor (σ² Zstar'Zstar / n with σ² = resid'resid / n under
`Uncorrelated`), for the standard errors of the reduced-form estimators.
Returns the K × K matrix.
"""
function _weakiv_compute_avar_residuals(resid::Vector{T}, Zstar::Matrix{T},
        vcov_est, n::Int) where {T}
    if _is_homoskedastic(vcov_est)
        return (dot(resid, resid) / n) .* (Zstar' * Zstar) ./ n
    end
    return _weakiv_meat(Zstar .* resid, vcov_est) ./ n
end

##############################################################################
## Symmetric matrix power
##############################################################################

function _matrix_power_sym(A::AbstractMatrix{T}, p::Real) where {T}
    F = eigvals(Symmetric(A))
    V = eigvecs(Symmetric(A))
    return V * diagm(0 => F .^ p) * V'
end

# Need eigvecs
using LinearAlgebra: eigvecs, diagm

##############################################################################
## Patnaik critical value (noncentral chi-squared approximation)
##############################################################################

"""
    _patnaik_critical_value(W_2, alpha, x) -> T

Compute the Patnaik-approximated critical value for the effective F test.

Given W_2, computes K_eff and Delta from the eigenvalue distribution,
then returns quantile of noncentral chi-squared(K_eff, Delta) / K_eff.
"""
function _patnaik_critical_value(W_2::AbstractMatrix{T}, alpha::T, x::T) where {T}
    # Normalize eigenvalues
    eigs = real.(eigvals(Symmetric(W_2)))
    s = sum(eigs)
    if s ≤ 0
        return T(NaN)
    end
    W2_norm = eigs ./ s
    sort!(W2_norm)

    # Patnaik moment matching
    variance = 2 * sum(abs2, W2_norm) + 4 * x * maximum(W2_norm)
    K_eff = 2 * (1 + 2 * x) / variance
    Delta = K_eff * x

    # Inverse noncentral chi-squared
    cv = _invnchisq(K_eff, Delta, T(1) - alpha) / K_eff
    return cv
end

##############################################################################
## Inverse noncentral chi-squared via bisection
##############################################################################

"""
    _invnchisq(df, ncp, p) -> T

Compute the p-th quantile of the noncentral chi-squared distribution
with `df` degrees of freedom and noncentrality parameter `ncp`.
Uses bisection on `nchisqcdf`.
"""
function _invnchisq(df::T, ncp::T, p::T) where {T <: Real}
    # Handle edge cases
    ncp < 0 && return T(NaN)
    p ≤ 0 && return T(0)
    p ≥ 1 && return T(Inf)

    # Initial bracket
    # Mean of noncentral chi-squared is df + ncp
    # Variance is 2(df + 2*ncp)
    mu = df + ncp
    sigma = sqrt(2 * (df + 2 * ncp))

    lo = max(T(0), mu - 10 * sigma)
    hi = mu + 10 * sigma

    # Ensure bracket contains the quantile
    while nchisqcdf(df, ncp, hi) < p
        hi *= 2
    end
    while lo > 0 && nchisqcdf(df, ncp, lo) > p
        lo /= 2
    end

    # Bisection
    for _ in 1:200
        mid = (lo + hi) / 2
        if nchisqcdf(df, ncp, mid) < p
            lo = mid
        else
            hi = mid
        end
        if (hi - lo) < 1e-12 * max(1, abs(mid))
            break
        end
    end

    return (lo + hi) / 2
end

# Promote method for mixed types
function _invnchisq(df::Real, ncp::Real, p::Real)
    T = promote_type(typeof(df), typeof(ncp), typeof(p))
    return _invnchisq(T(df), T(ncp), T(p))
end

##############################################################################
## BTSLS computation (Nagar benchmark)
##############################################################################

"""
    _Bmaxfunction_nagar(beta, W_1, W_12, W_2)

Compute the maximal Nagar bias for TSLS at a given beta value.
Returns (B, beta).
"""
function _Bmaxfunction_nagar(beta::T, W_1, W_12, W_2) where {T}
    S_2 = W_2
    S_12 = W_12 .- beta .* W_2
    S_1 = W_1 .- 2 * beta .* W_12 .+ beta^2 .* W_2

    eigs = real.(eigvals(Symmetric((S_12 .+ S_12') ./ 2)))
    mineig = minimum(eigs)
    maxeig = maximum(eigs)

    trS12 = tr(S_12)
    trS2 = tr(S_2)
    trS1 = tr(S_1)
    denom = sqrt(trS2 * trS1)

    if denom ≈ 0
        return T(0)
    end

    B1 = abs(trS12 / denom * (1 - 2 * mineig / trS12))
    B2 = abs(trS12 / denom * (1 - 2 * maxeig / trS12))

    return max(B1, B2)
end

"""
    _compute_BTSLS_nagar(W_1, W_12, W_2, eps) -> T

Compute B_TSLS by grid search + Nelder-Mead optimization.
"""
function _compute_BTSLS_nagar(W_1::AbstractMatrix{T}, W_12, W_2, eps::T) where {T}
    # Find LimitB (limit as beta -> ±∞)
    eigs_W2 = real.(eigvals(Symmetric(W_2)))
    eigmin_W2 = minimum(eigs_W2)
    trW2 = tr(W_2)
    LimitB = 1 - 2 * eigmin_W2 / trW2

    if abs(LimitB) < 1e-15
        return T(0)
    end

    # Find betastart range
    betastart = _find_betastart(
        beta -> _Bmaxfunction_nagar(beta, W_1, W_12, W_2), LimitB, eps)

    if betastart == 0
        return _Bmaxfunction_nagar(T(0), W_1, W_12, W_2)
    end

    # Grid search
    best_beta = _grid_search(
        beta -> _Bmaxfunction_nagar(beta, W_1, W_12, W_2), betastart)

    # Nelder-Mead refinement
    best_beta = _nelder_mead_1d(
        beta -> -_Bmaxfunction_nagar(beta, W_1, W_12, W_2), best_beta)

    return _Bmaxfunction_nagar(best_beta, W_1, W_12, W_2)
end

##############################################################################
## BTSLS computation (OLS benchmark - gfweakivtestols)
##############################################################################

function _Bmaxfunction_ols(beta::T, Omega, W_1, W_12, W_2) where {T}
    S_12 = W_12 .- beta .* W_2
    S_1 = W_1 .- 2 * beta .* W_12 .+ beta^2 .* W_2

    eigs = real.(eigvals(Symmetric((S_12 .+ S_12') ./ 2)))
    mineig = minimum(eigs)
    maxeig = maximum(eigs)

    trS12 = tr(S_12)
    trW2 = tr(W_2)

    # OLS benchmark: BM = sqrt((Omega[1,1] - 2*beta*Omega[1,2] + beta^2*Omega[2,2]) / Omega[2,2])
    BM = sqrt(max(T(0), (Omega[1, 1] - 2 * beta * Omega[1, 2] + beta^2 * Omega[2, 2]) /
                        Omega[2, 2]))

    if BM ≈ 0 || trW2 ≈ 0
        return T(0)
    end

    B1 = abs((trS12 - 2 * mineig) / trW2) / BM
    B2 = abs((trS12 - 2 * maxeig) / trW2) / BM

    return max(B1, B2)
end

function _compute_BTSLS_ols(Omega, W_1::AbstractMatrix{T}, W_12, W_2, eps::T) where {T}
    eigs_W2 = real.(eigvals(Symmetric(W_2)))
    eigmin_W2 = minimum(eigs_W2)
    trW2 = tr(W_2)
    LimitB = 1 - 2 * eigmin_W2 / trW2

    if abs(LimitB) < 1e-15
        return T(0)
    end

    betastart = _find_betastart(
        beta -> _Bmaxfunction_ols(beta, Omega, W_1, W_12, W_2), LimitB, eps)

    if betastart == 0
        return _Bmaxfunction_ols(T(0), Omega, W_1, W_12, W_2)
    end

    best_beta = _grid_search(
        beta -> _Bmaxfunction_ols(beta, Omega, W_1, W_12, W_2), betastart)

    best_beta = _nelder_mead_1d(
        beta -> -_Bmaxfunction_ols(beta, Omega, W_1, W_12, W_2), best_beta)

    return _Bmaxfunction_ols(best_beta, Omega, W_1, W_12, W_2)
end

##############################################################################
## BLIML computation
##############################################################################

function _BmaxLIML(beta::T, W_1, W_12, W_2, Omega) where {T}
    S_2 = W_2
    S_12 = W_12 .- beta .* W_2
    S_1 = W_1 .- 2 * beta .* W_12 .+ beta^2 .* W_2

    om_1 = Omega[1, 1]
    om_12 = Omega[1, 2]
    om_2 = Omega[2, 2]
    sig_12 = om_12 - beta * om_2
    sig_1 = om_1 - 2 * beta * om_12 + beta^2 * om_2

    if abs(sig_1) < 1e-15
        return T(0)
    end

    Matrix_val = 2 .* S_12 .- (sig_12 / sig_1) .* S_1
    Matrix_val = (Matrix_val .+ Matrix_val') ./ 2
    eigs = real.(eigvals(Symmetric(Matrix_val)))
    mineig = minimum(eigs)
    maxeig = maximum(eigs)

    trS2 = tr(S_2)
    trS1 = tr(S_1)
    trS12 = tr(S_12)
    denom = sqrt(trS2 * trS1)

    if denom ≈ 0
        return T(0)
    end

    base = trS12 - (sig_12 / sig_1) * trS1
    B1 = abs((base - mineig) / denom)
    B2 = abs((base - maxeig) / denom)

    return max(B1, B2)
end

function _compute_BLIML_nagar(W_1::AbstractMatrix{T}, W_12, W_2, Omega, eps::T) where {T}
    eigs_W2 = real.(eigvals(Symmetric(W_2)))
    eigmax_W2 = maximum(eigs_W2)
    trW2 = tr(W_2)
    LimitB = eigmax_W2 / trW2

    if abs(LimitB) < 1e-15
        return T(0)
    end

    betastart = _find_betastart(
        beta -> _BmaxLIML(beta, W_1, W_12, W_2, Omega), LimitB, eps)

    if betastart == 0
        return _BmaxLIML(T(0), W_1, W_12, W_2, Omega)
    end

    best_beta = _grid_search(
        beta -> _BmaxLIML(beta, W_1, W_12, W_2, Omega), betastart)

    best_beta = _nelder_mead_1d(
        beta -> -_BmaxLIML(beta, W_1, W_12, W_2, Omega), best_beta)

    return _BmaxLIML(best_beta, W_1, W_12, W_2, Omega)
end

##############################################################################
## BGMMf computation (Nagar benchmark)
##############################################################################

function _BGMMf_objective_nagar(beta::T, trW_1s, trW_12s, eig, K) where {T}
    num = trW_12s - 2 * eig - (K - 2) * beta
    den = sqrt(K) * sqrt(max(T(0), trW_1s - 2 * beta * trW_12s + K * beta^2))
    return abs(den) < 1e-15 ? T(0) : num / den
end

function _compute_BGMMf_nagar(trW_1s::T, trW_12s::T, mineig::T, maxeig::T,
        K::Int, bgmmf_start::T) where {T}
    results = T[]

    for (eig, sense) in [(mineig, :max), (maxeig, :max), (mineig, :min), (maxeig, :min)]
        try
            if sense == :max
                beta_opt = _nelder_mead_1d(
                    b -> -_BGMMf_objective_nagar(b, trW_1s, trW_12s, eig, K), bgmmf_start)
                push!(results, abs(_BGMMf_objective_nagar(
                    beta_opt, trW_1s, trW_12s, eig, K)))
            else
                beta_opt = _nelder_mead_1d(
                    b -> _BGMMf_objective_nagar(b, trW_1s, trW_12s, eig, K), bgmmf_start)
                push!(results, abs(_BGMMf_objective_nagar(
                    beta_opt, trW_1s, trW_12s, eig, K)))
            end
        catch e
            # A combination whose optimization fails numerically does not enter the bound.
            _isnumericalfailure(e) || rethrow()
        end
    end

    return isempty(results) ? T(0) : maximum(results)
end

##############################################################################
## BGMMf computation (OLS benchmark)
##############################################################################

function _BGMMf_objective_ols(beta::T, Omega, trW_1s, trW_12s, eig, K) where {T}
    num = (trW_12s - 2 * eig - (K - 2) * beta) / K
    den = sqrt(max(T(0), (Omega[1, 1] - 2 * beta * Omega[1, 2] + beta^2 * Omega[2, 2]) /
                         Omega[2, 2]))
    return abs(den) < 1e-15 ? T(0) : num / den
end

function _compute_BGMMf_ols(Omega, trW_1s::T, trW_12s::T, mineig::T, maxeig::T,
        K::Int, bgmmf_start::T) where {T}
    results = T[]

    for (eig, sense) in [(mineig, :max), (maxeig, :max), (mineig, :min), (maxeig, :min)]
        try
            if sense == :max
                beta_opt = _nelder_mead_1d(
                    b -> -_BGMMf_objective_ols(b, Omega, trW_1s, trW_12s, eig, K), bgmmf_start)
                push!(results, abs(_BGMMf_objective_ols(
                    beta_opt, Omega, trW_1s, trW_12s, eig, K)))
            else
                beta_opt = _nelder_mead_1d(
                    b -> _BGMMf_objective_ols(b, Omega, trW_1s, trW_12s, eig, K), bgmmf_start)
                push!(results, abs(_BGMMf_objective_ols(
                    beta_opt, Omega, trW_1s, trW_12s, eig, K)))
            end
        catch e
            # A combination whose optimization fails numerically does not enter the bound.
            _isnumericalfailure(e) || rethrow()
        end
    end

    return isempty(results) ? T(0) : maximum(results)
end

##############################################################################
## Optimization helpers: grid search + 1D Nelder-Mead
##############################################################################

"""
    _find_betastart(f, LimitB, eps) -> T

Find a range [-betastart, betastart] such that f(±betastart) is within
fraction eps of LimitB. Matches Stata's BTSLS_start / BLIML_start logic.
"""
function _find_betastart(f, LimitB, eps)
    if abs(LimitB) < 1e-15
        return 0
    end

    betastart = 0
    val = max(abs(abs(f(betastart)) / LimitB - 1),
        abs(abs(f(-betastart)) / LimitB - 1))

    if val ≤ eps
        return 0
    end

    for _ in 1:1000
        val = max(abs(abs(f(betastart)) / LimitB - 1),
            abs(abs(f(-betastart)) / LimitB - 1))
        if val ≤ eps
            break
        end
        betastart += 1
    end

    return betastart
end

"""
    _grid_search(f, betastart; npoints=10000) -> T

Grid search over [-betastart, betastart] to find the beta that maximizes f.
"""
function _grid_search(f, betastart; npoints = 10000)
    if betastart == 0
        return 0.0
    end

    best_val = -Inf
    best_beta = -betastart
    step = 2 * betastart / npoints

    t = -betastart
    while t ≤ betastart
        val = f(t)
        if val > best_val
            best_val = val
            best_beta = t
        end
        t += step
    end

    return best_beta
end

"""
    _nelder_mead_1d(f, x0; delta=0.5, maxiter=1000, tol=1e-10) -> T

Simple 1D Nelder-Mead (simplex) minimization.
Returns the minimizer of f.
"""
function _nelder_mead_1d(f, x0; delta = 0.5, maxiter = 1000, tol = 1e-10)
    # 1D simplex: two points
    x1 = x0
    x2 = x0 + delta

    f1 = f(x1)
    f2 = f(x2)

    for _ in 1:maxiter
        # Sort
        if f2 < f1
            x1, x2 = x2, x1
            f1, f2 = f2, f1
        end

        # Centroid (just x1 in 1D)
        xc = x1

        # Reflection
        xr = 2 * xc - x2
        fr = f(xr)

        if fr < f1
            # Expansion
            xe = xc + 2 * (xr - xc)
            fe = f(xe)
            if fe < fr
                x2 = xe
                f2 = fe
            else
                x2 = xr
                f2 = fr
            end
        elseif fr < f2
            x2 = xr
            f2 = fr
        else
            # Contraction
            xk = xc + 0.5 * (x2 - xc)
            fk = f(xk)
            if fk < f2
                x2 = xk
                f2 = fk
            else
                # Shrink
                x2 = x1 + 0.5 * (x2 - x1)
                f2 = f(x2)
            end
        end

        # Check convergence
        if abs(x2 - x1) < tol * max(1, abs(x1))
            break
        end
    end

    return f1 < f2 ? x1 : x2
end

##############################################################################
## Show method
##############################################################################

function Base.show(io::IO, r::WeakIVTestResult{T}) where {T}
    level_pct = round(Int, r.level * 100)
    same_tsls_liml = r.cv_TSLS == r.cv_LIML
    same_all = same_tsls_liml && r.cv_TSLS == r.cv_GMMf
    colw = 12

    println(io)
    println(io, "Montiel-Pflueger robust weak instrument test")
    println(io, "─" ^ 54)
    @printf(io, "TSLS coefficient (SE):        %8.4f (%8.4f)\n", r.btsls, r.sebtsls)
    @printf(io, "LIML coefficient (SE):        %8.4f (%8.4f)\n", r.bliml, r.sebliml)
    @printf(io, "GMMf coefficient (SE):        %8.4f (%8.4f)\n", r.bgmmf, r.sebgmmf)
    @printf(io, "kappa:                        %12.4f\n", r.kappa)
    @printf(io, "F[nonrobust]:                 %12.3f\n", r.F_nonrobust)
    @printf(io, "F[effective]:                 %12.3f\n", r.F_eff)
    @printf(io, "F[robust]:                    %12.3f\n", r.F_robust)
    @printf(io, "Confidence level alpha:       %9d%%\n", level_pct)
    println(io, "─" ^ 54)
    println(io)
    println(io, "Use:")
    if same_all
        println(io, "  Compare F[effective] or F[robust] to the common critical values below.")
    else
        println(io, "  Compare F[effective] to TSLS/LIML critical values.")
        println(io, "  Compare F[robust] to GMMf critical values.")
    end
    println(io, "  If F exceeds the critical value at tau, weak-IV bias is below that tau bound.")
    println(io)
    println(io, "─" ^ 54)
    if same_all
        @printf(io, "%-10s %*s\n", "tau", colw, "Common")
        println(io, "─" ^ 54)
        @printf(io, "%-10s %*.3f\n", "tau=5%", colw, r.cv_TSLS[1])
        @printf(io, "%-10s %*.3f\n", "tau=10%", colw, r.cv_TSLS[2])
        @printf(io, "%-10s %*.3f\n", "tau=20%", colw, r.cv_TSLS[3])
        @printf(io, "%-10s %*.3f\n", "tau=30%", colw, r.cv_TSLS[4])
    elseif same_tsls_liml
        @printf(io, "%-10s %*s %*s\n", "tau", colw, "TSLS/LIML", colw, "GMMf")
        println(io, "─" ^ 54)
        @printf(io, "%-10s %*.3f %*.3f\n", "tau=5%", colw, r.cv_TSLS[1], colw, r.cv_GMMf[1])
        @printf(io, "%-10s %*.3f %*.3f\n", "tau=10%", colw, r.cv_TSLS[2], colw,
            r.cv_GMMf[2])
        @printf(io, "%-10s %*.3f %*.3f\n", "tau=20%", colw, r.cv_TSLS[3], colw,
            r.cv_GMMf[3])
        @printf(io, "%-10s %*.3f %*.3f\n", "tau=30%", colw, r.cv_TSLS[4], colw,
            r.cv_GMMf[4])
    else
        @printf(io, "%-10s %*s %*s %*s\n", "tau", colw, "TSLS", colw, "LIML", colw, "GMMf")
        println(io, "─" ^ 54)
        @printf(io,
            "%-10s %*.3f %*.3f %*.3f\n", "tau=5%", colw, r.cv_TSLS[1], colw, r.cv_LIML[1], colw,
            r.cv_GMMf[1])
        @printf(io,
            "%-10s %*.3f %*.3f %*.3f\n", "tau=10%", colw, r.cv_TSLS[2], colw, r.cv_LIML[2], colw,
            r.cv_GMMf[2])
        @printf(io,
            "%-10s %*.3f %*.3f %*.3f\n", "tau=20%", colw, r.cv_TSLS[3], colw, r.cv_LIML[3], colw,
            r.cv_GMMf[3])
        @printf(io,
            "%-10s %*.3f %*.3f %*.3f\n", "tau=30%", colw, r.cv_TSLS[4], colw, r.cv_LIML[4], colw,
            r.cv_GMMf[4])
    end
    println(io, "─" ^ 54)
end
