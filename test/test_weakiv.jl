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
    @test_throws ErrorException Regress.weakivtest(m)
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
