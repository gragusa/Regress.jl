@testitem "LIML basic" tags = [:iv, :kclass, :smoke] begin
    using Regress
    using Regress.StableRNGs: StableRNG
    using DataFrames
    using StatsBase: coef, vcov

    # Helper to create test data
    n = 500
    rng = StableRNG(42)
    z1 = randn(rng, n)
    z2 = randn(rng, n)
    e = randn(rng, n)
    u = 0.5 .* e .+ randn(rng, n)
    x = 0.5 .* z1 .+ 0.3 .* z2 .+ u
    y = 2.0 .* x .+ 1.0 .+ e
    df = DataFrame(y = y, x = x, z1 = z1, z2 = z2)

    m = Regress.iv(Regress.LIML(), df, @formula(y ~ (x ~ z1 + z2)))

    @test length(coef(m)) == 2
    @test m.postestimation.kappa !== nothing
    @test m.postestimation.kappa >= 1.0  # LIML kappa should be >= 1
    @test m.estimator isa Regress.LIML
end

@testitem "Fuller basic" tags = [:iv, :kclass] begin
    using Regress
    using Regress.StableRNGs: StableRNG
    using DataFrames
    using StatsBase: coef

    # Helper to create test data
    n = 500
    rng = StableRNG(42)
    z1 = randn(rng, n)
    z2 = randn(rng, n)
    e = randn(rng, n)
    u = 0.5 .* e .+ randn(rng, n)
    x = 0.5 .* z1 .+ 0.3 .* z2 .+ u
    y = 2.0 .* x .+ 1.0 .+ e
    df = DataFrame(y = y, x = x, z1 = z1, z2 = z2)

    # Test Fuller with default a=1
    m1 = Regress.iv(Regress.Fuller(), df, @formula(y ~ (x ~ z1 + z2)))
    @test m1.estimator isa Regress.Fuller
    @test m1.estimator.a == 1.0

    # Test Fuller with custom a
    m4 = Regress.iv(Regress.Fuller(4.0), df, @formula(y ~ (x ~ z1 + z2)))
    @test m4.estimator isa Regress.Fuller
    @test m4.estimator.a == 4.0

    # Fuller kappa should be less than LIML kappa
    m_liml = Regress.iv(Regress.LIML(), df, @formula(y ~ (x ~ z1 + z2)))
    @test m1.postestimation.kappa < m_liml.postestimation.kappa
end

@testitem "KClass(1.0) equals TSLS" tags = [:iv, :kclass, :tsls] begin
    using Regress
    using Regress.StableRNGs: StableRNG
    using DataFrames
    using StatsBase: coef

    # Helper to create test data
    n = 500
    rng = StableRNG(42)
    z1 = randn(rng, n)
    z2 = randn(rng, n)
    e = randn(rng, n)
    u = 0.5 .* e .+ randn(rng, n)
    x = 0.5 .* z1 .+ 0.3 .* z2 .+ u
    y = 2.0 .* x .+ 1.0 .+ e
    df = DataFrame(y = y, x = x, z1 = z1, z2 = z2)

    m_tsls = Regress.iv(Regress.TSLS(), df, @formula(y ~ (x ~ z1 + z2)))
    m_k1 = Regress.iv(Regress.KClass(1.0), df, @formula(y ~ (x ~ z1 + z2)))

    # Coefficients should match (note: different algorithms so tolerance needed)
    @test coef(m_tsls) ≈ coef(m_k1) atol=1e-6
end

@testitem "K-class vcov" tags = [:iv, :kclass, :vcov] begin
    using Regress
    using Regress.StableRNGs: StableRNG
    using DataFrames
    using StatsBase: coef, vcov, stderror
    using CovarianceMatrices: HC3

    # Helper to create test data
    n = 500
    rng = StableRNG(42)
    z1 = randn(rng, n)
    z2 = randn(rng, n)
    e = randn(rng, n)
    u = 0.5 .* e .+ randn(rng, n)
    x = 0.5 .* z1 .+ 0.3 .* z2 .+ u
    y = 2.0 .* x .+ 1.0 .+ e
    df = DataFrame(y = y, x = x, z1 = z1, z2 = z2)

    m = Regress.iv(Regress.LIML(), df, @formula(y ~ (x ~ z1 + z2)))

    # Test default vcov (HC1)
    V_default = vcov(m)
    @test size(V_default) == (2, 2)
    @test all(isfinite, V_default)

    # Test HC3 vcov
    V_hc3 = vcov(HC3(), m)
    @test size(V_hc3) == (2, 2)
    @test all(isfinite, V_hc3)

    # Test model + vcov() operator
    m_hc3 = m + Regress.vcov(HC3())
    @test all(isfinite, m_hc3.se)
end

@testitem "K-class cluster" tags = [:iv, :kclass, :cluster] begin
    using Regress
    using Regress.StableRNGs: StableRNG
    using DataFrames
    using StatsBase: vcov, stderror
    using CovarianceMatrices: CR1

    rng = StableRNG(42)
    n = 500
    z1 = randn(rng, n)
    z2 = randn(rng, n)
    cluster = repeat(1:50, inner = 10)
    e = randn(rng, n)
    u = 0.5 .* e .+ randn(rng, n)
    x = 0.5 .* z1 .+ 0.3 .* z2 .+ u
    y = 2.0 .* x .+ 1.0 .+ e

    df = DataFrame(y = y, x = x, z1 = z1, z2 = z2, cluster = cluster)

    # Fit with saved cluster variable
    m = Regress.iv(Regress.LIML(), df, @formula(y ~ (x ~ z1 + z2)), save_cluster = :cluster)

    # Test cluster-robust vcov using symbol
    V_cr = vcov(CR1(:cluster), m)
    @test size(V_cr) == (2, 2)
    @test all(isfinite, V_cr)

    # Test model + vcov(CR1(:cluster))
    m_cr = m + Regress.vcov(CR1(:cluster))
    @test all(isfinite, m_cr.se)
end

@testitem "K-class multiple endogenous" tags = [:iv, :kclass] begin
    using Regress
    using Regress.StableRNGs: StableRNG
    using DataFrames
    using StatsBase: coef

    rng = StableRNG(42)
    n = 500
    z1 = randn(rng, n)
    z2 = randn(rng, n)
    z3 = randn(rng, n)
    e = randn(rng, n)
    u1 = 0.3 .* e .+ randn(rng, n)
    u2 = 0.3 .* e .+ randn(rng, n)
    x1 = 0.4 .* z1 .+ 0.3 .* z2 .+ u1
    x2 = 0.3 .* z2 .+ 0.4 .* z3 .+ u2
    y = 1.5 .* x1 .+ 0.5 .* x2 .+ 1.0 .+ e

    df = DataFrame(y = y, x1 = x1, x2 = x2, z1 = z1, z2 = z2, z3 = z3)

    # Test LIML with 2 endogenous variables
    m = Regress.iv(Regress.LIML(), df, @formula(y ~ (x1 + x2 ~ z1 + z2 + z3)))

    @test length(coef(m)) == 3  # 2 endogenous + intercept
    @test m.postestimation.kappa !== nothing
    @test m.estimator isa Regress.LIML
end

@testitem "Matrix k-class matches formula path" tags = [:iv, :kclass, :vcov] begin
    using Regress
    using Regress.StableRNGs: StableRNG
    using DataFrames
    using StatsBase: coef, vcov, stderror, leverage
    using CovarianceMatrices: HC0, HC1, HC2, HC3

    rng = StableRNG(42)
    n = 500
    z1 = randn(rng, n)
    z2 = randn(rng, n)
    w = randn(rng, n)                      # exogenous regressor
    e = randn(rng, n)
    u = 0.5 .* e .+ randn(rng, n)
    x = 0.5 .* z1 .+ 0.3 .* z2 .+ 0.2 .* w .+ u   # endogenous
    y = 2.0 .* x .+ 1.5 .* w .+ 1.0 .+ e
    df = DataFrame(y = y, x = x, w = w, z1 = z1, z2 = z2)

    f = @formula(y ~ w + (x ~ z1 + z2))
    # Combined layout: X = [intercept, w, x]; Z = [intercept, w, z1, z2]
    X = hcat(ones(n), w, x)
    Z = hcat(ones(n), w, z1, z2)

    for est in (Regress.LIML(), Regress.Fuller(1.0), Regress.KClass(1.2))
        mf = Regress.iv(est, df, f)
        mm = Regress.iv(est, Z, X, y; has_intercept = true, n_endogenous = 1)

        @test mm.estimator === est
        @test coef(mm) ≈ coef(mf) atol = 1e-8
        @test stderror(mm) ≈ stderror(mf) atol = 1e-8
        @test vcov(mm) ≈ vcov(mf) atol = 1e-8

        for hc in (HC0(), HC1(), HC2(), HC3())
            @test vcov(hc, mm) ≈ vcov(hc, mf) atol = 1e-8
        end

        # The + operator carries the estimator and reproduces robust SEs
        mm_r = mm + Regress.vcov(HC3())
        @test mm_r.estimator === est
        @test stderror(mm_r) ≈ stderror(mf + Regress.vcov(HC3())) atol = 1e-8
    end

    # Over-identified TSLS: matrix and formula paths agree on coefficients and all
    # HC variants, including the leverage-based HC2/HC3.
    mf_tsls = Regress.iv(Regress.TSLS(), df, f)
    mm_tsls = Regress.iv(Regress.TSLS(), Z, X, y; has_intercept = true, n_endogenous = 1)
    @test mm_tsls.estimator === Regress.TSLS()
    @test coef(mm_tsls) ≈ coef(mf_tsls) atol = 1e-8
    @test leverage(mm_tsls) ≈ leverage(mf_tsls) atol = 1e-8
    for hc in (HC0(), HC1(), HC2(), HC3())
        @test vcov(hc, mm_tsls) ≈ vcov(hc, mf_tsls) atol = 1e-8
    end
end

@testitem "Matrix k-class with multiple endogenous matches formula path" tags = [
    :iv, :kclass] begin
    using Regress
    using Regress.StableRNGs: StableRNG
    using DataFrames
    using StatsBase: coef, stderror

    rng = StableRNG(7)
    n = 600
    z1 = randn(rng, n)
    z2 = randn(rng, n)
    z3 = randn(rng, n)
    e = randn(rng, n)
    u1 = 0.3 .* e .+ randn(rng, n)
    u2 = 0.3 .* e .+ randn(rng, n)
    x1 = 0.4 .* z1 .+ 0.3 .* z2 .+ u1
    x2 = 0.3 .* z2 .+ 0.4 .* z3 .+ u2
    y = 1.5 .* x1 .+ 0.5 .* x2 .+ 1.0 .+ e
    df = DataFrame(y = y, x1 = x1, x2 = x2, z1 = z1, z2 = z2, z3 = z3)

    f = @formula(y ~ (x1 + x2 ~ z1 + z2 + z3))
    # X = [intercept, x1, x2]; Z = [intercept, z1, z2, z3]
    X = hcat(ones(n), x1, x2)
    Z = hcat(ones(n), z1, z2, z3)

    for est in (Regress.LIML(), Regress.Fuller(1.0))
        mf = Regress.iv(est, df, f)
        mm = Regress.iv(est, Z, X, y; has_intercept = true, n_endogenous = 2)
        @test coef(mm) ≈ coef(mf) atol = 1e-8
        @test stderror(mm) ≈ stderror(mf) atol = 1e-8
    end
end

@testitem "Matrix IV diagnostics match formula path" tags = [:iv, :diagnostics] begin
    using Regress
    using Regress: first_stage_F_iid, first_stage_F_robust, first_stage_F_KP,
                   wu_hausman, sargan, pvalue
    using Regress.StableRNGs: StableRNG
    using DataFrames
    using StatsBase: coef
    using CovarianceMatrices: HC3, CR1

    rng = StableRNG(42)
    n = 400
    z1 = randn(rng, n)
    z2 = randn(rng, n)
    z3 = randn(rng, n)
    w = randn(rng, n)                      # exogenous regressor
    e = randn(rng, n)
    u = 0.5 .* e .+ randn(rng, n)
    x = 0.5 .* z1 .+ 0.3 .* z2 .+ 0.2 .* z3 .+ 0.2 .* w .+ u   # endogenous
    y = 2.0 .* x .+ 1.5 .* w .+ 1.0 .+ e
    df = DataFrame(y = y, x = x, w = w, z1 = z1, z2 = z2, z3 = z3)

    f = @formula(y ~ w + (x ~ z1 + z2 + z3))
    # X = [intercept, w, x]; Z = [intercept, w, z1, z2, z3] — overidentified (3 > 1)
    X = hcat(ones(n), w, x)
    Z = hcat(ones(n), w, z1, z2, z3)

    mf = Regress.iv(Regress.TSLS(), df, f)
    mm = Regress.iv(Regress.TSLS(), Z, X, y; has_intercept = true, n_endogenous = 1)

    # First-stage F family: IID, robust (HC1 default), and Kleibergen-Paap.
    fi_f, fi_m = first_stage_F_iid(mf), first_stage_F_iid(mm)
    @test fi_m.stat ≈ fi_f.stat atol = 1e-8
    @test fi_m.p ≈ fi_f.p atol = 1e-10
    @test (fi_m.df1, fi_m.df2) == (fi_f.df1, fi_f.df2)

    fr_f, fr_m = first_stage_F_robust(mf), first_stage_F_robust(mm)
    @test fr_m.stat ≈ fr_f.stat atol = 1e-8
    @test fr_m.p ≈ fr_f.p atol = 1e-10
    @test (fr_m.df1, fr_m.df2) == (fr_f.df1, fr_f.df2)

    kp_f, kp_m = first_stage_F_KP(mf), first_stage_F_KP(mm)
    @test kp_m.stat ≈ kp_f.stat atol = 1e-6
    @test kp_m.p ≈ kp_f.p atol = 1e-8
    @test kp_m.df1 == kp_f.df1

    # Endogeneity and overidentification tests.
    wh_f, wh_m = wu_hausman(mf), wu_hausman(mm)
    @test wh_m.stat ≈ wh_f.stat atol = 1e-7
    @test wh_m.p ≈ wh_f.p atol = 1e-9
    @test (wh_m.df1, wh_m.df2) == (wh_f.df1, wh_f.df2)

    sg_f, sg_m = sargan(mf), sargan(mm)
    @test sg_m.stat ≈ sg_f.stat atol = 1e-7
    @test sg_m.p ≈ sg_f.p atol = 1e-9
    @test sg_m.df == sg_f.df

    # pvalue() on the returned test types (AbstractTest interface).
    @test pvalue(fr_m) == fr_m.p
    @test pvalue(wh_m) == wh_m.p
    @test pvalue(sg_m) == sg_m.p

    # Robust first-stage F tracks the model's variance estimator (HC3, then CR1).
    mf3, mm3 = mf + Regress.vcov(HC3()), mm + Regress.vcov(HC3())
    @test first_stage_F_robust(mm3).stat ≈ first_stage_F_robust(mf3).stat atol = 1e-8

    cl = rand(rng, 1:15, n)
    dfc = copy(df)
    dfc.firm = cl
    mfc = Regress.iv(Regress.TSLS(), dfc, f, save_cluster = :firm) +
          Regress.vcov(CR1(:firm))
    mmc = mm + Regress.vcov(CR1(cl))
    @test first_stage_F_robust(mmc).stat ≈ first_stage_F_robust(mfc).stat atol = 1e-8
    # KP is fixed at HR1 and therefore invariant to the model's vcov.
    @test first_stage_F_KP(mmc).stat ≈ first_stage_F_KP(mf).stat atol = 1e-6

    # No exogenous regressors, no intercept: X = [x], Z = [z1, z2, z3].
    Xne = reshape(x, n, 1)
    Zne = hcat(z1, z2, z3)
    mne = Regress.iv(Regress.TSLS(), Zne, Xne, y; has_intercept = false, n_endogenous = 1)
    mfne = Regress.iv(Regress.TSLS(), df, @formula(y ~ 0 + (x ~ 0 + z1 + z2 + z3)))
    @test first_stage_F_iid(mne).stat ≈ first_stage_F_iid(mfne).stat atol = 1e-8
    @test first_stage_F_robust(mne).stat ≈ first_stage_F_robust(mfne).stat atol = 1e-8
    @test wu_hausman(mne).stat ≈ wu_hausman(mfne).stat atol = 1e-7
    @test sargan(mne).stat ≈ sargan(mfne).stat atol = 1e-7

    # Just-identified: Sargan is undefined and must throw on both paths.
    Xji = hcat(ones(n), w, x)
    Zji = hcat(ones(n), w, z1)
    mji = Regress.iv(Regress.TSLS(), Zji, Xji, y; has_intercept = true, n_endogenous = 1)
    @test_throws ArgumentError sargan(mji)
    @test_throws "overidentification" sargan(mji)
end

@testitem "Matrix IV diagnostics with multiple endogenous match formula path" tags = [
    :iv, :diagnostics] begin
    using Regress
    using Regress: first_stage_F_iid, first_stage_F_robust, first_stage_F_KP,
                   wu_hausman, sargan
    using Regress.StableRNGs: StableRNG
    using DataFrames

    rng = StableRNG(7)
    n = 600
    z1 = randn(rng, n)
    z2 = randn(rng, n)
    z3 = randn(rng, n)
    e = randn(rng, n)
    u1 = 0.3 .* e .+ randn(rng, n)
    u2 = 0.3 .* e .+ randn(rng, n)
    x1 = 0.4 .* z1 .+ 0.3 .* z2 .+ u1
    x2 = 0.3 .* z2 .+ 0.4 .* z3 .+ u2
    y = 1.5 .* x1 .+ 0.5 .* x2 .+ 1.0 .+ e
    df = DataFrame(y = y, x1 = x1, x2 = x2, z1 = z1, z2 = z2, z3 = z3)

    f = @formula(y ~ (x1 + x2 ~ z1 + z2 + z3))
    X = hcat(ones(n), x1, x2)
    Z = hcat(ones(n), z1, z2, z3)

    mf = Regress.iv(Regress.TSLS(), df, f)
    mm = Regress.iv(Regress.TSLS(), Z, X, y; has_intercept = true, n_endogenous = 2)

    @test first_stage_F_iid(mm).stat ≈ first_stage_F_iid(mf).stat atol = 1e-8
    @test first_stage_F_robust(mm).stat ≈ first_stage_F_robust(mf).stat atol = 1e-8
    @test first_stage_F_KP(mm).stat ≈ first_stage_F_KP(mf).stat atol = 1e-6
    @test wu_hausman(mm).stat ≈ wu_hausman(mf).stat atol = 1e-7
    @test sargan(mm).stat ≈ sargan(mf).stat atol = 1e-7
end

@testitem "LIML with FE" tags = [:iv, :kclass, :fe] begin
    using Regress
    using Regress: fe
    using Regress.StableRNGs: StableRNG
    using DataFrames
    using StatsBase: coef

    rng = StableRNG(42)
    n = 500
    z1 = randn(rng, n)
    z2 = randn(rng, n)
    fe_id = repeat(1:50, inner = 10)
    fe_effect = randn(rng, 50)[fe_id]
    e = randn(rng, n)
    u = 0.5 .* e .+ randn(rng, n)
    x = 0.5 .* z1 .+ 0.3 .* z2 .+ u
    y = 2.0 .* x .+ fe_effect .+ e

    df = DataFrame(y = y, x = x, z1 = z1, z2 = z2, fe_id = fe_id)

    # Test LIML with fixed effects
    m = Regress.iv(Regress.LIML(), df, @formula(y ~ (x ~ z1 + z2) + fe(fe_id)))

    @test length(coef(m)) == 1  # Only x coefficient (no intercept with FE)
    @test m.postestimation.kappa !== nothing
end

@testitem "LIML Stata validation" tags = [:iv, :kclass, :validation] begin
    using Regress
    using DataFrames
    using CSV
    using CategoricalArrays: categorical
    using StatsBase: coef, stderror, r2
    using CovarianceMatrices: CR1

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../test/data/iv_nested.csv")))
    df.state_id = categorical(df.state_id)

    # LIML with state as categorical (dummy variables, not absorbed FE)
    m = Regress.iv(Regress.LIML(), df, @formula(y ~ x1 + x2 + state_id + (endo ~ z)))

    # Coefficients from Stata ivreg2:
    # . ivreg2 y (endo=z) x1 x2 i.state_id, liml
    @test coef(m)[1] ≈ -0.1403594 atol=0.01  # intercept
    @test coef(m)[2] ≈ 0.4484096 atol=0.01   # x1
    @test coef(m)[3] ≈ 0.2649239 atol=0.01   # x2
    # state_id dummies (reference = state 1)
    @test coef(m)[4] ≈ 2.700246 atol=0.01    # state_id: 2
    @test coef(m)[5] ≈ 2.866612 atol=0.01    # state_id: 3
    @test coef(m)[6] ≈ 0.2536304 atol=0.01   # state_id: 4
    @test coef(m)[7] ≈ 2.053986 atol=0.01    # state_id: 5
    @test coef(m)[8] ≈ 2.033306 atol=0.01    # endo

    # R-squared
    @test r2(m) ≈ 0.9142 atol=0.01

    # LIML without FE, with cluster-robust SE
    # . ivreg2 y (endo=z) x1 x2 , liml cluster(state_id)
    m2 = Regress.iv(Regress.LIML(), df, @formula(y ~ x1 + x2 + (endo ~ z)), save_cluster = :state_id)

    # Coefficients
    @test coef(m2)[1] ≈ 1.448071 atol=0.01   # intercept
    @test coef(m2)[2] ≈ 0.4741162 atol=0.01  # x1
    @test coef(m2)[3] ≈ 0.3173924 atol=0.01  # x2
    @test coef(m2)[4] ≈ 2.001556 atol=0.01   # endo

    # Cluster-robust SE (CR1)
    # Note: Small differences from Stata expected due to finite-sample corrections
    se_cr1 = stderror(CR1(:state_id), m2)
    @test se_cr1[1] ≈ 0.5667004 atol=0.10   # intercept SE (wider tolerance for cluster SE)
    @test se_cr1[2] ≈ 0.0803681 atol=0.02   # x1 SE
    @test se_cr1[3] ≈ 0.0642344 atol=0.02   # x2 SE
    @test se_cr1[4] ≈ 0.0673868 atol=0.02   # endo SE

    # R-squared
    @test r2(m2) ≈ 0.8554 atol=0.01
end

@testitem "Nested FE with cluster" tags = [:fe, :cluster] begin
    using Regress
    using Regress: fe
    using DataFrames
    using CSV
    using StatsBase: stderror
    using CovarianceMatrices: CR1

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../test/data/iv_nested.csv")))

    # County FE is nested in state_id cluster
    # When FE is nested in cluster, DOF adjustment should account for nesting
    m = Regress.ols(df, @formula(y ~ x1 + x2 + endo + fe(county_id)), save_cluster = :state_id)

    se_cr1 = stderror(CR1(:state_id), m)
    @test all(isfinite, se_cr1)

    # Compare with state FE (non-nested - state is the cluster level)
    m2 = Regress.ols(df, @formula(y ~ x1 + x2 + endo + fe(state_id)), save_cluster = :state_id)
    se_cr1_state = stderror(CR1(:state_id), m2)
    @test all(isfinite, se_cr1_state)
end

@testitem "LIML with nested FE and cluster" tags = [:iv, :kclass, :fe, :cluster] begin
    using Regress
    using Regress: fe
    using DataFrames
    using CSV
    using StatsBase: stderror
    using CovarianceMatrices: CR1

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../test/data/iv_nested.csv")))

    # LIML with state FE (absorbed), cluster at state level
    m = Regress.iv(Regress.LIML(), df, @formula(y ~ x1 + x2 + (endo ~ z) + fe(state_id)),
        save_cluster = :state_id)

    se_cr1 = stderror(CR1(:state_id), m)
    @test all(isfinite, se_cr1)
    # LIML kappa should be >= 1, allowing for floating point tolerance
    @test m.postestimation.kappa >= 1.0 - 1e-10
end
