@testitem "weakivtest: Nagar benchmark matches Stata gfweakivtest" tags = [:iv, :weakiv] begin
    using Regress
    using CSV
    using DataFrames

    # Load mroz dataset (used in Stata validation)
    mroz_path = joinpath(@__DIR__, "data", "mroz.csv")
    if !isfile(mroz_path)
        @warn "mroz.csv not found at $mroz_path, skipping weakivtest validation"
        return
    end
    df = CSV.read(mroz_path, DataFrame)

    # Fit TSLS model: lwage ~ exper + expersq + (educ ~ age + kidslt6 + kidsge6), robust
    m = Regress.iv(Regress.TSLS(), df, @formula(lwage ~
                                                exper + expersq +
                                                (educ ~ age + kidslt6 + kidsge6)))

    r = Regress.weakivtest(m)

    # Stata reference values from: ivreg2 lwage exper expersq (educ=age kidslt6 kidsge6), rob
    #                              gfweakivtest

    # Coefficients and SEs
    @test r.btsls ≈ 0.0964 atol = 0.0001
    @test r.sebtsls ≈ 0.0865 atol = 0.0001
    @test r.bliml ≈ 0.0958 atol = 0.0001
    @test r.sebliml ≈ 0.0913 atol = 0.0001
    @test r.kappa ≈ 1.0016 atol = 0.0001
    @test r.bgmmf ≈ 0.0948 atol = 0.0001
    @test r.sebgmmf ≈ 0.0868 atol = 0.0001

    # F-statistics
    @test r.F_nonrobust ≈ 4.342 atol = 0.001
    @test r.F_eff ≈ 4.552 atol = 0.001
    @test r.F_robust ≈ 5.021 atol = 0.001

    # TSLS critical values (Nagar benchmark)
    @test r.cv_TSLS[1] ≈ 15.711 atol = 0.01  # tau=5%
    @test r.cv_TSLS[2] ≈ 9.957 atol = 0.01   # tau=10%
    @test r.cv_TSLS[3] ≈ 6.749 atol = 0.01   # tau=20%
    @test r.cv_TSLS[4] ≈ 5.560 atol = 0.01   # tau=30%

    # LIML critical values
    @test r.cv_LIML[1] ≈ 15.406 atol = 0.01
    @test r.cv_LIML[2] ≈ 9.789 atol = 0.01
    @test r.cv_LIML[3] ≈ 6.654 atol = 0.01
    @test r.cv_LIML[4] ≈ 5.491 atol = 0.01

    # GMMf critical values
    @test r.cv_GMMf[1] ≈ 13.651 atol = 0.01
    @test r.cv_GMMf[2] ≈ 8.745 atol = 0.01
    @test r.cv_GMMf[3] ≈ 6.021 atol = 0.01
    @test r.cv_GMMf[4] ≈ 5.018 atol = 0.02  # slightly looser for NM optimization

    # Metadata
    @test r.K == 3
    @test r.N == 428
    @test r.level ≈ 0.05
end

@testitem "weakivtest: OLS benchmark matches Stata gfweakivtestols" tags = [:iv, :weakiv] begin
    using Regress
    using CSV
    using DataFrames

    mroz_path = joinpath(@__DIR__, "data", "mroz.csv")
    if !isfile(mroz_path)
        @warn "mroz.csv not found, skipping"
        return
    end
    df = CSV.read(mroz_path, DataFrame)

    m = Regress.iv(Regress.TSLS(), df, @formula(lwage ~
                                                exper + expersq +
                                                (educ ~ age + kidslt6 + kidsge6)))
    r = Regress.weakivtest(m; benchmark = :ols)

    # Stata reference: gfweakivtestols
    # F-stats and coefficients are the same as Nagar
    @test r.F_eff ≈ 4.552 atol = 0.001
    @test r.btsls ≈ 0.0964 atol = 0.0001

    # OLS benchmark gives different TSLS critical values
    @test r.cv_TSLS[1] ≈ 15.900 atol = 0.01  # tau=5%
    @test r.cv_TSLS[2] ≈ 10.062 atol = 0.01  # tau=10%
    @test r.cv_TSLS[3] ≈ 6.808 atol = 0.01   # tau=20%
    @test r.cv_TSLS[4] ≈ 5.602 atol = 0.02   # tau=30%

    # LIML critical values are the SAME for OLS and Nagar benchmark
    @test r.cv_LIML[1] ≈ 15.406 atol = 0.01

    # GMMf critical values differ
    @test r.cv_GMMf[1] ≈ 13.901 atol = 0.01
    @test r.cv_GMMf[2] ≈ 8.882 atol = 0.01
    @test r.cv_GMMf[3] ≈ 6.098 atol = 0.01
    @test r.cv_GMMf[4] ≈ 5.073 atol = 0.02
end

@testitem "weakivtest: validation checks" tags = [:iv, :weakiv] begin
    using Regress
    using DataFrames
    using StableRNGs

    rng = StableRNG(42)
    n = 200
    z1 = randn(rng, n)
    z2 = randn(rng, n)
    z3 = randn(rng, n)
    x = 0.5 * z1 + 0.3 * z2 + 0.2 * z3 + randn(rng, n)
    y = 1.0 .+ 2.0 .* x .+ randn(rng, n)

    df = DataFrame(y = y, x = x, z1 = z1, z2 = z2, z3 = z3)

    m = Regress.iv(Regress.TSLS(), df, @formula(y ~ (x ~ z1 + z2 + z3)))
    r = Regress.weakivtest(m)

    # Basic structural checks
    @test r.K == 3
    @test r.N == n
    @test r.level ≈ 0.05
    @test r.F_eff > 0
    @test r.F_robust > 0
    @test r.F_nonrobust > 0
    @test all(cv -> cv > 0, r.cv_TSLS)
    @test all(cv -> cv > 0, r.cv_LIML)
    @test all(cv -> cv > 0, r.cv_GMMf)

    # Critical values should be decreasing with tau
    @test r.cv_TSLS[1] > r.cv_TSLS[2] > r.cv_TSLS[3] > r.cv_TSLS[4]
    @test r.cv_LIML[1] > r.cv_LIML[2] > r.cv_LIML[3] > r.cv_LIML[4]
    @test r.cv_GMMf[1] > r.cv_GMMf[2] > r.cv_GMMf[3] > r.cv_GMMf[4]

    # With strong instruments, F_eff should be large
    @test r.F_eff > 10
end

@testitem "weakivtest: requires single endogenous" tags = [:iv, :weakiv] begin
    using Regress
    using DataFrames
    using StableRNGs

    rng = StableRNG(123)
    n = 100
    z1 = randn(rng, n)
    z2 = randn(rng, n)
    z3 = randn(rng, n)
    x1 = z1 + randn(rng, n)
    x2 = z2 + randn(rng, n)
    y = 1.0 .+ x1 .+ x2 .+ randn(rng, n)

    df = DataFrame(y = y, x1 = x1, x2 = x2, z1 = z1, z2 = z2, z3 = z3)

    m = Regress.iv(Regress.TSLS(), df, @formula(y ~ (x1 + x2 ~ z1 + z2 + z3)))
    @test_throws ArgumentError Regress.weakivtest(m)
    @test_throws "weakivtest requires exactly one endogenous regressor, got 2" Regress.weakivtest(m)
end

@testitem "weakivtest: tol keyword" tags = [:iv, :weakiv] begin
    using Regress
    using DataFrames
    using StableRNGs

    rng = StableRNG(7)
    n = 200
    z1 = randn(rng, n)
    z2 = randn(rng, n)
    z3 = randn(rng, n)
    x = 0.5 * z1 + 0.3 * z2 + 0.2 * z3 + randn(rng, n)
    y = 1.0 .+ 2.0 .* x .+ randn(rng, n)
    df = DataFrame(y = y, x = x, z1 = z1, z2 = z2, z3 = z3)

    # Formula path: passing the default explicitly matches the default.
    m = Regress.iv(Regress.TSLS(), df, @formula(y ~ (x ~ z1 + z2 + z3)))
    r_default = Regress.weakivtest(m)
    r_tol = Regress.weakivtest(m; tol = 0.001)
    @test r_tol.cv_TSLS == r_default.cv_TSLS
    @test r_tol.cv_LIML == r_default.cv_LIML
    @test r_tol.cv_GMMf == r_default.cv_GMMf

    # A different tolerance is accepted and yields a valid result.
    r_loose = Regress.weakivtest(m; tol = 0.01)
    @test all(cv -> cv > 0, r_loose.cv_TSLS)

    # Matrix path also accepts tol.
    X = hcat(ones(n), x)
    Z = hcat(ones(n), z1, z2, z3)
    m_matrix = Regress.iv(Regress.TSLS(), Z, X, y; has_intercept = false, n_endogenous = 1)
    r_matrix = Regress.weakivtest(m_matrix; tol = 0.001)
    @test all(cv -> cv > 0, r_matrix.cv_TSLS)
end

@testitem "weakivtest: IVMatrixEstimator matches IVEstimator" tags = [:iv, :weakiv] begin
    using Regress
    using CSV
    using DataFrames

    mroz_path = joinpath(@__DIR__, "data", "mroz.csv")
    if !isfile(mroz_path)
        @warn "mroz.csv not found, skipping"
        return
    end
    df = CSV.read(mroz_path, DataFrame)

    # Fit via formula API (IVEstimator)
    m_formula = Regress.iv(
        Regress.TSLS(), df, @formula(lwage ~
                                     exper + expersq +
                                     (educ ~ age + kidslt6 + kidsge6)))
    r_formula = Regress.weakivtest(m_formula)

    # Fit via matrix API (IVMatrixEstimator) with the same data
    # Build matrices manually: X = [intercept, exper, expersq, educ], Z = [intercept, exper, expersq, age, kidslt6, kidsge6]
    df_complete = dropmissing(df, [
        :lwage, :exper, :expersq, :educ, :age, :kidslt6, :kidsge6])
    n = nrow(df_complete)
    intercept = ones(n)
    X = hcat(intercept, Float64.(df_complete.exper), Float64.(df_complete.expersq),
        Float64.(df_complete.educ))
    Z = hcat(intercept, Float64.(df_complete.exper), Float64.(df_complete.expersq),
        Float64.(df_complete.age), Float64.(df_complete.kidslt6),
        Float64.(df_complete.kidsge6))
    y = Float64.(df_complete.lwage)

    m_matrix = Regress.iv(Regress.TSLS(), Z, X, y; has_intercept = false, n_endogenous = 1)
    r_matrix = Regress.weakivtest(m_matrix)

    # Results should match
    @test r_matrix.F_eff ≈ r_formula.F_eff atol = 1e-8
    @test r_matrix.F_robust ≈ r_formula.F_robust atol = 1e-8
    @test r_matrix.F_nonrobust ≈ r_formula.F_nonrobust atol = 1e-8
    @test r_matrix.btsls ≈ r_formula.btsls atol = 1e-8
    @test r_matrix.sebtsls ≈ r_formula.sebtsls atol = 1e-8
    @test r_matrix.bliml ≈ r_formula.bliml atol = 1e-8
    @test r_matrix.sebliml ≈ r_formula.sebliml atol = 1e-8
    @test r_matrix.bgmmf ≈ r_formula.bgmmf atol = 1e-8
    @test r_matrix.sebgmmf ≈ r_formula.sebgmmf atol = 1e-8
    @test r_matrix.kappa ≈ r_formula.kappa atol = 1e-8
    @test all(r_matrix.cv_TSLS .≈ r_formula.cv_TSLS)
    @test all(r_matrix.cv_LIML .≈ r_formula.cv_LIML)
    @test all(r_matrix.cv_GMMf .≈ r_formula.cv_GMMf)
    @test r_matrix.K == r_formula.K
    @test r_matrix.N == r_formula.N
end

@testitem "weakivtest: just-identified (K=1)" tags = [:iv, :weakiv] begin
    using Regress
    using DataFrames
    using StableRNGs

    rng = StableRNG(99)
    n = 300
    z = randn(rng, n)
    x = 0.8 * z + randn(rng, n)
    y = 1.0 .+ 1.5 .* x .+ randn(rng, n)

    df = DataFrame(y = y, x = x, z = z)

    m = Regress.iv(Regress.TSLS(), df, @formula(y ~ (x ~ z)))
    r = Regress.weakivtest(m)

    @test r.K == 1
    # For K=1: F_eff and F_robust should be related
    # (not necessarily equal due to different formulations)
    @test r.F_eff > 0
    @test r.F_robust > 0
    @test isfinite(r.cv_TSLS[1])
    @test isfinite(r.cv_GMMf[1])
end

@testitem "weakivtest follows the variance estimator" tags = [:iv, :weakiv, :validation] begin
    using Regress
    using DataFrames, CSV
    using Regress: fe

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    wt(m, k) = weakivtest(m + vcov(k))

    # With one endogenous regressor the robust F is the first-stage Wald F of the excluded
    # instruments. References: R fixest 0.14.1
    # `fitstat(feols(Sales ~ NDI | Price ~ Pimin + Pop, df, vcov = v), ~ivwald1)`.
    m = Regress.iv(Regress.TSLS(), df, @formula(Sales ~ NDI + (Price ~ Pimin + Pop)),
        save_cluster = [:State, :Year])
    for (k, ref) in ((Uncorrelated(), 2622.461860257), (HC1(), 1464.27486829565),
        (CR1(:State), 280.848595798773), (CR1(:State, :Year), 280.440867252914))
        @test wt(m, k).F_robust ≈ ref rtol = 1e-10
    end
    # Absorbed fixed effect: fixest `feols(Sales ~ NDI | State | Price ~ Pimin + Pop, ...)`.
    mf = Regress.iv(
        Regress.TSLS(), df, @formula(Sales ~ NDI + (Price ~ Pimin + Pop) + fe(State)),
        save_cluster = [:State, :Year])
    for (k, ref) in ((Uncorrelated(), 2776.23219867604), (HC1(), 1435.11064681269),
        (CR1(:State), 332.781459426826), (CR1(:Year), 1146.43018399374))
        @test wt(mf, k).F_robust ≈ ref rtol = 1e-10
    end
    for k in (HC0(), CR0(:Year), Bartlett(3))
        @test wt(m, k).F_robust ≈ only(first_stage_F_robust(m + vcov(k)).stat) rtol = 1e-10
    end

    # Under iid errors W = Ω ⊗ I, so the three F-statistics coincide.
    r = wt(m, Uncorrelated())
    @test r.F_eff ≈ r.F_nonrobust rtol = 1e-12
    @test r.F_robust ≈ r.F_nonrobust rtol = 1e-12

    # Clusters named by column equal the cluster vector, and every dimension is used.
    r1 = wt(m, CR1(:State))
    @test r1.F_eff ≈ wt(m, CR1(df.State)).F_eff rtol = 1e-12
    @test r1.F_eff ≈ wt(m, CachedCR(CR1(:State), m)).F_eff rtol = 1e-12
    @test !isapprox(wt(m, CR1(:State, :Year)).F_eff, r1.F_eff; rtol = 1e-6)
    @test wt(m, HC0()).F_eff != wt(m, HC1()).F_eff

    for k in (CR2(:State), CR3(:State), EWC(4))
        @test_throws "The weak-instrument test is not defined for" wt(m, k)
        @test_throws "It supports Uncorrelated, HC0–HC5, CR0, CR1" wt(m, k)
    end

    # HC2–HC5 rescale the moments by the leverage of the first-stage design, so the robust
    # F is the first-stage Wald F under the same estimator, with one excluded instrument
    # or two, and with one instrument the effective F equals it.
    m1 = Regress.iv(Regress.TSLS(), df, @formula(Sales ~ NDI + (Price ~ Pimin)))
    for k in (HC2(), HC3(), HC4(), HC4m(), HC5())
        @test wt(m, k).F_robust ≈ only(first_stage_F_robust(m + vcov(k)).stat) rtol = 1e-10
        local r = wt(m1, k)
        @test r.F_robust ≈ only(first_stage_F_robust(m1 + vcov(k)).stat) rtol = 1e-10
        @test r.F_eff ≈ r.F_robust rtol = 1e-10
        @test r.F_eff != wt(m1, HC0()).F_eff
        @test isfinite(r.sebtsls) && isfinite(r.sebliml) && isfinite(r.sebgmmf)
    end
    # The matrix fit gives the formula fit's numbers.
    n1 = nrow(df)
    mm1 = Regress.iv(Regress.TSLS(), hcat(ones(n1), df.NDI, df.Pimin),
        hcat(ones(n1), df.NDI, df.Price), Float64.(df.Sales); has_intercept = true,
        n_endogenous = 1)
    @test weakivtest(mm1 + vcov(HC3())).F_eff ≈ wt(m1, HC3()).F_eff rtol = 1e-10
    @test weakivtest(mm1 + vcov(HC3())).sebtsls ≈ wt(m1, HC3()).sebtsls rtol = 1e-10
    # They need the leverage, which absorbed fixed effects would change.
    @test_throws "needs the leverage of the first-stage regression" wt(mf, HC3())

    # A matrix-fitted model gives the formula fit's statistics under a cluster estimator.
    n = nrow(df)
    X = hcat(ones(n), df.NDI, df.Price)
    Z = hcat(ones(n), df.NDI, df.Pimin, df.Pop)
    mm = Regress.iv(Regress.TSLS(), Z, X, Float64.(df.Sales); has_intercept = true,
        n_endogenous = 1)
    rm = weakivtest(mm + vcov(CR1(df.State)))
    @test rm.F_eff ≈ r1.F_eff rtol = 1e-10
    @test rm.F_robust ≈ r1.F_robust rtol = 1e-10
end
