@testitem "IV HC variance vs R" tags = [:iv, :validation] begin
    using Regress
    using DataFrames
    using CSV
    using StatsBase: coef, vcov, stderror, nobs
    using CovarianceMatrices: HC0, HC1, HC2, HC3

    # The file basic_validation_df.csv contains a drw from this model
    # Model: y ~ x | z1 + z2
    # n = 500, k = 2 (intercept + x)
    # dof_residual should be 500 - 2 = 498

    csv_path = joinpath(@__DIR__, "data", "basic_validation_df.csv")
    df = CSV.read(csv_path, DataFrame)

    ## Reference value from R (AER package ivreg)
    coefs_tsls = [0.9592032, 2.0587160]

    V_TSLS_HC1 = [0.0017817895920835959 0.00020032733726142889;
                  0.00020032733726142886 0.0057013792157889157]
    V_TSLS_HC2 = [0.0017827669083746174 0.00020078616154375079;
                  0.00020078616154375079 0.0057318506193747027]
    V_TSLS_HC3 = [0.0017909605285917555 0.00020204613244595676;
                  0.00020204613244595679 0.0057860720244417617]
    V_TSLS_HC4 = [0.0017934706407452825 0.00020270513567441562;
                  0.0002027051356744156 0.0058512766899867013]

    coef_liml = [0.9591984, 2.058297]

    V_LIML_HC1 = [0.00020058 0.0017824;
                  0.0017824 0.00571862]

    m_tsls = iv(TSLS(), df, @formula(y~(x ~ z1 + z2)))

    # Test that HC standard errors are computed correctly with proper DOF
    v_tsls_hc0 = vcov(HC0(), m_tsls)
    v_tsls_hc1 = vcov(HC1(), m_tsls)  # Explicit HC1 call (not default)
    v_tsls_hc2 = vcov(HC2(), m_tsls)
    v_tsls_hc3 = vcov(HC3(), m_tsls)
    v_tsls_hc4 = vcov(HC4(), m_tsls)

    # v_tsls_hac1 = vcov(Bartlett{NeweyWest}(), m_tsls)
    # v_tsls_hac2 = vcov(Bartlett{Andrews}(), m_tsls)

    @test V_TSLS_HC1≈v_tsls_hc1 rtol=1e-6
    @test V_TSLS_HC2≈v_tsls_hc2 rtol=1e-6
    @test V_TSLS_HC3≈v_tsls_hc3 rtol=1e-6
    @test V_TSLS_HC4≈v_tsls_hc4 rtol=1e-6
    # @test V_TSLS_HAC1≈v_tsls_hac1 rtol=1e-6
    # @test V_TSLS_HAC2≈v_tsls_hac2 rtol=1e-6
end

@testitem "OLS standard errors vs GLM.jl" tags = [:ols, :validation] begin
    using Regress
    using DataFrames
    using CSV
    using StatsBase: coef, vcov, stderror, nobs
    using CovarianceMatrices: HC0, HC1, HC2, HC3
    using GLM

    ## The file basic_validation_df.csv is now in the test/data directory.
    csv_path = joinpath(@__DIR__, "data", "basic_validation_df.csv")
    df = CSV.read(csv_path, DataFrame)

    m_glm = lm(@formula(y~x), df)
    m_ols = ols(df, @formula(y~x))

    # DOF should match - validates the intercept double-counting fix for OLS
    @test dof_residual(m_ols) == dof_residual(m_glm)
    @test length(coef(m_ols)) == length(coef(m_glm))

    # Coefficients should match
    @test coef(m_ols)≈coef(m_glm) rtol=1e-10

    se_glm_hc0 = stderror(HC0(), m_glm)
    se_ols_hc0 = stderror(HC0(), m_ols)
    @test se_ols_hc0≈se_glm_hc0 rtol=1e-6

    # HC1 standard errors should match
    se_glm_hc1 = stderror(HC1(), m_glm)
    se_ols_hc1 = stderror(HC1(), m_ols)
    @test se_ols_hc1≈se_glm_hc1 rtol=1e-6

    # HC2 standard errors should match
    se_glm_hc2 = stderror(HC2(), m_glm)
    se_ols_hc2 = stderror(HC2(), m_ols)
    @test se_ols_hc2≈se_glm_hc2 rtol=1e-6

    # HC3 standard errors should match
    se_glm_hc3 = stderror(HC3(), m_glm)
    se_ols_hc3 = stderror(HC3(), m_ols)
    @test se_ols_hc3≈se_glm_hc3 rtol=1e-6

    se_glm_hac1 = stderror(Bartlett{NeweyWest}(), m_glm)
    se_ols_hac1 = stderror(Bartlett{NeweyWest}(), m_ols)
    @test se_ols_hac1≈se_glm_hac1 rtol=1e-6

    se_glm_hac2 = stderror(Bartlett{Andrews}(), m_glm)
    se_ols_hac2 = stderror(Bartlett{Andrews}(), m_ols)
    @test se_ols_hac2≈se_glm_hac2 rtol=1e-6
end
