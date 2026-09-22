@testitem "Over-identified TSLS leverage matches the formula path" tags = [:iv, :vcov] begin
    using Regress
    using Regress.StableRNGs: StableRNG
    using DataFrames
    using StatsBase: coef, leverage, stderror, vcov
    using CovarianceMatrices: HC0, HC1, HC2, HC3

    rng = StableRNG(42)
    n = 200

    # Three excluded instruments for one endogenous regressor: over-identified,
    # which is where the projection and AER leverage formulas diverge.
    z1 = randn(rng, n)
    z2 = randn(rng, n)
    z3 = randn(rng, n)
    x_exo = randn(rng, n)
    x_endo = 0.5z1 + 0.4z2 + 0.3z3 + randn(rng, n)
    y = 1.0 .+ 2.0x_endo .+ 0.5x_exo .+ randn(rng, n)

    df = DataFrame(; y, x_endo, x_exo, z1, z2, z3)
    mf = Regress.iv(Regress.TSLS(), df, @formula(y~x_exo + (x_endo ~ z1 + z2 + z3)))

    X = hcat(ones(n), x_exo, x_endo)
    Z = hcat(ones(n), x_exo, z1, z2, z3)
    mm = Regress.iv(Regress.TSLS(), Z, X, y; has_intercept = true, n_endogenous = 1)

    @test mm.estimator === Regress.TSLS()
    @test coef(mm) ≈ coef(mf) atol = 1e-8
    @test leverage(mm) ≈ leverage(mf) atol = 1e-8

    # HC2/HC3 are leverage-based, so they are the variants that disagree when the
    # matrix path uses diag(X̂(X̂'X̂)⁻¹X̂') instead of the AER formula.
    for hc in (HC0(), HC1(), HC2(), HC3())
        @test vcov(hc, mm) ≈ vcov(hc, mf) atol = 1e-8
        @test stderror(hc, mm) ≈ stderror(hc, mf) atol = 1e-8
    end
end

@testitem "Just-identified TSLS leverage is unchanged" tags = [:iv, :vcov] begin
    using Regress
    using Regress.StableRNGs: StableRNG
    using DataFrames
    using StatsBase: coef, leverage, vcov
    using CovarianceMatrices: HC2, HC3

    rng = StableRNG(7)
    n = 150

    # One instrument per endogenous regressor: the two leverage formulas coincide
    # here, so this pins that the specialization did not disturb the exact case.
    z1 = randn(rng, n)
    x_exo = randn(rng, n)
    x_endo = 0.8z1 + randn(rng, n)
    y = 1.0 .+ 2.0x_endo .+ 0.5x_exo .+ randn(rng, n)

    df = DataFrame(; y, x_endo, x_exo, z1)
    mf = Regress.iv(Regress.TSLS(), df, @formula(y~x_exo + (x_endo ~ z1)))

    X = hcat(ones(n), x_exo, x_endo)
    Z = hcat(ones(n), x_exo, z1)
    mm = Regress.iv(Regress.TSLS(), Z, X, y; has_intercept = true, n_endogenous = 1)

    @test coef(mm) ≈ coef(mf) atol = 1e-8
    @test leverage(mm) ≈ leverage(mf) atol = 1e-8
    for hc in (HC2(), HC3())
        @test vcov(hc, mm) ≈ vcov(hc, mf) atol = 1e-8
    end
end
