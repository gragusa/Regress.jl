@testitem "Automatic HAC bandwidth excludes the intercept moment on IV models" tags = [
    :iv, :vcov, :hac] begin
    using Regress
    using Regress.StableRNGs: StableRNG
    using StatsBase: coef
    using Regress: Bartlett, NeweyWest, Andrews
    using CovarianceMatrices: bandwidth

    rng = StableRNG(11)
    n = 300
    z = randn(rng, n)
    x = 0.9z + randn(rng, n)
    y = 1.0 .+ 2.0x + randn(rng, n)
    X = hcat(ones(n), x)

    # Instrumenting X with itself makes the IV moment condition identical to the
    # OLS one, so the two models must select the same bandwidth. They differ only
    # if one of them lets the intercept moment into the selection.
    m_ols = Regress.ols(X, y)
    m_iv = Regress.iv(Regress.TSLS(), X, X, y; has_intercept = true, n_endogenous = 1)
    @test coef(m_ols) ≈ coef(m_iv) atol = 1e-10

    for K in (Bartlett{NeweyWest}, Bartlett{Andrews})
        b_ols = bandwidth(Regress.vcov(K(), m_ols))
        b_iv = bandwidth(Regress.vcov(K(), m_iv))
        @test b_ols ≈ b_iv rtol = 1e-8
    end
end

@testitem "Automatic HAC bandwidth agrees across IV model paths" tags = [:iv, :vcov, :hac] begin
    using Regress
    using Regress.StableRNGs: StableRNG
    using DataFrames
    using Regress: Bartlett, NeweyWest
    using CovarianceMatrices: bandwidth

    rng = StableRNG(11)
    n = 300
    z = randn(rng, n)
    z2 = randn(rng, n)
    x = 0.6z + 0.4z2 + randn(rng, n)
    y = 1.0 .+ 2.0x + randn(rng, n)

    df = DataFrame(; y, x, z, z2)
    mf = Regress.iv(Regress.TSLS(), df, @formula(y~(x~z + z2)))
    X = hcat(ones(n), x)
    Z = hcat(ones(n), z, z2)
    mm = Regress.iv(Regress.TSLS(), Z, X, y; has_intercept = true, n_endogenous = 1)

    Vf = Regress.vcov(Bartlett{NeweyWest}(), mf)
    Vm = Regress.vcov(Bartlett{NeweyWest}(), mm)
    @test bandwidth(Vf) ≈ bandwidth(Vm) rtol = 1e-8
    @test Vf ≈ Vm rtol = 1e-8
end

@testitem "weakivtest accepts an automatic-bandwidth HAC estimator" tags = [
    :iv, :vcov, :hac] begin
    using Regress
    using Regress.StableRNGs: StableRNG
    using DataFrames
    using Regress: Bartlett, NeweyWest

    rng = StableRNG(11)
    n = 300
    z = randn(rng, n)
    z2 = randn(rng, n)
    x = 0.6z + 0.4z2 + randn(rng, n)
    y = 1.0 .+ 2.0x + randn(rng, n)

    df = DataFrame(; y, x, z, z2)
    mf = Regress.iv(Regress.TSLS(), df, @formula(y~(x~z + z2)))
    X = hcat(ones(n), x)
    Z = hcat(ones(n), z, z2)
    mm = Regress.iv(Regress.TSLS(), Z, X, y; has_intercept = true, n_endogenous = 1)

    # `weakivtest` applies the estimator to a stacked moment matrix whose column
    # count differs from the model's, so a variance estimator that carried state
    # sized to one of them could not serve both.
    rf = weakivtest(mf + Regress.vcov(Bartlett{NeweyWest}()))
    rm = weakivtest(mm + Regress.vcov(Bartlett{NeweyWest}()))
    @test rf isa WeakIVTestResult
    @test rm isa WeakIVTestResult

    # A fixed bandwidth takes a different path through the same code.
    @test weakivtest(mf + Regress.vcov(Bartlett(4))) isa WeakIVTestResult
end
