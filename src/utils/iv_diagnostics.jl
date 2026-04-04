##############################################################################
##
## IV Diagnostic Tests: First-Stage F, Wu-Hausman, Sargan
##
##############################################################################

using LinearAlgebra: dot

"""
    FirstStageFResult{T}

Result of the IID (SSR-based) first-stage F-test for each endogenous variable.

# Fields
- `F_per_endo::Vector{T}`: F-statistic for each endogenous variable
- `p_per_endo::Vector{T}`: p-value for each endogenous variable
- `df1::Int`: Numerator degrees of freedom (number of excluded instruments)
- `df2::Int`: Denominator degrees of freedom (first-stage residual df)
- `endogenous_names::Vector{String}`: Names of endogenous variables
"""
struct FirstStageFResult{T <: AbstractFloat}
    F_per_endo::Vector{T}
    p_per_endo::Vector{T}
    df1::Int
    df2::Int
    endogenous_names::Vector{String}
end

function Base.show(io::IO, r::FirstStageFResult{T}) where {T}
    println(io, "First-Stage F-test (IID)")
    println(io, "─" ^ 50)
    for (j, name) in enumerate(r.endogenous_names)
        @printf(io, "  %-15s  F = %8.3f  (p = %.4e)  df = (%d, %d)\n",
            name, r.F_per_endo[j], r.p_per_endo[j], r.df1, r.df2)
    end
end

function _compute_first_stage_f_iid(
        fsd::FirstStageData{T},
        n::Int,
        dof_fes::Int
) where {T}
    k_endo = size(fsd.Xendo_orig, 2)
    n_excl = size(fsd.Z_res, 2)
    n_exo = fsd.n_exo

    df1 = n_excl
    df2 = n - n_exo - n_excl - dof_fes

    F_per_endo = Vector{T}(undef, k_endo)
    p_per_endo = Vector{T}(undef, k_endo)

    Xexo = view(fsd.newZ, :, 1:n_exo)
    qr_exo = qr(Xexo)

    for j in 1:k_endo
        xendo_j = fsd.Xendo_orig[:, j]
        ssr_full = sum(abs2, fsd.Xendo_res[:, j])
        resid_restricted = xendo_j - Xexo * (qr_exo \ xendo_j)
        ssr_restricted = sum(abs2, resid_restricted)

        F_per_endo[j] = ((ssr_restricted - ssr_full) / df1) / (ssr_full / df2)
        p_per_endo[j] = fdistccdf(df1, df2, F_per_endo[j])
    end

    return F_per_endo, p_per_endo, df1, df2
end

"""
    WuHausmanResult{T}

Result of the Wu-Hausman endogeneity test.

Tests H₀: instrumented variables are exogenous.

# Fields
- `stat::T`: F-statistic
- `p::T`: p-value
- `df1::Int`: Numerator df (number of endogenous variables)
- `df2::Int`: Denominator df
"""
struct WuHausmanResult{T <: AbstractFloat}
    stat::T
    p::T
    df1::Int
    df2::Int
end

function Base.show(io::IO, r::WuHausmanResult)
    @printf(io, "Wu-Hausman: stat = %8.5f, p = %.4e, on %d and %d DoF.\n",
        r.stat, r.p, r.df1, r.df2)
end

"""
    SarganResult{T}

Result of the Sargan overidentification test.

Tests H₀: instruments are valid (uncorrelated with structural error).
Only available for overidentified models.

# Fields
- `stat::T`: χ² test statistic
- `p::T`: p-value
- `df::Int`: Degrees of freedom (n_instruments - n_endogenous)
"""
struct SarganResult{T <: AbstractFloat}
    stat::T
    p::T
    df::Int
end

function Base.show(io::IO, r::SarganResult)
    @printf(io, "Sargan: stat = %8.5f, p = %.6f, on %d DoF.\n",
        r.stat, r.p, r.df)
end

##############################################################################
## first_stage_f — IID SSR-based first-stage F-test
##############################################################################

"""
    first_stage_f(m::IVEstimator) -> FirstStageFResult

Compute the standard (IID) first-stage F-test for each endogenous variable.

This is the classical F-test for joint significance of excluded instruments
in each first-stage regression, using the SSR-comparison formula:

    F = ((SSR_restricted - SSR_full) / df1) / (SSR_full / df2)

where `SSR_restricted` comes from regressing the endogenous variable on only
the exogenous regressors (no instruments), and `SSR_full` includes the instruments.

This matches fixest's `fitstat(m, "ivf1")`.
"""
function first_stage_f(m::IVEstimator{T}) where {T}
    pe = m.postestimation
    isnothing(pe) && error("No post-estimation data. Fit with save != :minimal.")
    fsd = pe.first_stage_data
    has_first_stage_data(fsd) || error("First-stage data not available.")

    F_per_endo, p_per_endo, df1, df2 = _compute_first_stage_f_iid(fsd, m.nobs, m.dof_fes)

    k_endo = length(F_per_endo)
    endogenous_names = copy(fsd.endogenous_names[1:k_endo])
    return FirstStageFResult{T}(F_per_endo, p_per_endo, df1, df2, endogenous_names)
end

##############################################################################
## wu_hausman — Durbin-Wu-Hausman endogeneity test
##############################################################################

"""
    wu_hausman(m::IVEstimator) -> WuHausmanResult

Compute the Wu-Hausman F-test for endogeneity.

Tests H₀: instrumented variables are exogenous (OLS is consistent).

The test augments the second-stage regression with first-stage residuals
and tests their joint significance:

    y = X_original * β + v̂ * δ + ε

where v̂ are first-stage residuals. Under H₀, δ = 0.

This matches fixest's Wu-Hausman statistic.
"""
function wu_hausman(m::IVEstimator{T}) where {T}
    pe = m.postestimation
    isnothing(pe) && error("No post-estimation data. Fit with save != :minimal.")
    fsd = pe.first_stage_data
    has_first_stage_data(fsd) || error("First-stage data not available.")

    k_endo = length(fsd.endogenous_names)
    y = pe.y
    X = pe.X_original  # [Xexo, Xendo] in demeaned+weighted space
    v_hat = fsd.Xendo_res  # first-stage residuals

    # Restricted model: y = X * β (standard OLS)
    qr_X = qr(X)
    resid_r = y - X * (qr_X \ y)
    ssr_r = sum(abs2, resid_r)

    # Unrestricted model: y = [X, v_hat] * β_aug
    W = hcat(X, v_hat)
    qr_W = qr(W)
    resid_u = y - W * (qr_W \ y)
    ssr_u = sum(abs2, resid_u)

    df1 = k_endo
    df2 = m.dof_residual - k_endo

    F = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
    p = fdistccdf(df1, df2, F)

    return WuHausmanResult{T}(F, p, df1, df2)
end

##############################################################################
## sargan — Sargan overidentification test
##############################################################################

"""
    sargan(m::IVEstimator) -> SarganResult

Compute the Sargan test for overidentifying restrictions.

Tests H₀: instruments are valid (uncorrelated with structural error).
Only available for overidentified models (n_instruments > n_endogenous).

The test regresses second-stage residuals on all instruments and exogenous
variables, and computes:

    S = n * R² ~ χ²(n_instruments - n_endogenous)

This matches fixest's Sargan statistic.
"""
function sargan(m::IVEstimator{T}) where {T}
    pe = m.postestimation
    isnothing(pe) && error("No post-estimation data. Fit with save != :minimal.")
    fsd = pe.first_stage_data
    has_first_stage_data(fsd) || error("First-stage data not available.")

    k_endo = length(fsd.endogenous_names)
    n_excl = size(fsd.Z_res, 2)

    n_excl > k_endo || error(
        "Sargan test requires overidentification (n_instruments=$n_excl > n_endogenous=$k_endo).")

    # Second-stage residuals using original X
    coef_full = copy(m.coef)
    coef_full[.!pe.basis_coef] .= zero(T)
    e = pe.y - pe.X_original * coef_full[pe.basis_coef .| .!pe.basis_coef]

    # Full instrument matrix Z = [Xexo, Z_excluded]
    Z_full = pe.Z

    # Auxiliary regression: e on Z_full using pre-computed (Z'Z)^{-1}
    beta_aux = pe.invZZ * (Z_full' * e)
    e_hat = Z_full * beta_aux

    rss_aux = sum(abs2, e - e_hat)
    tss_e = sum(abs2, e)

    n = m.nobs
    stat = n * (one(T) - rss_aux / tss_e)
    df = n_excl - k_endo
    p = chisqccdf(df, stat)

    return SarganResult{T}(stat, p, df)
end
