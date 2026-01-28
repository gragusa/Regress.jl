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

    m = iv(LIML(), df, @formula(y ~ (x ~ z1 + z2)))

    @test length(coef(m)) == 2
    @test m.postestimation.kappa !== nothing
    @test m.postestimation.kappa >= 1.0  # LIML kappa should be >= 1
    @test m.estimator isa LIML
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
    m1 = iv(Fuller(), df, @formula(y ~ (x ~ z1 + z2)))
    @test m1.estimator isa Fuller
    @test m1.estimator.a == 1.0

    # Test Fuller with custom a
    m4 = iv(Fuller(4.0), df, @formula(y ~ (x ~ z1 + z2)))
    @test m4.estimator isa Fuller
    @test m4.estimator.a == 4.0

    # Fuller kappa should be less than LIML kappa
    m_liml = iv(LIML(), df, @formula(y ~ (x ~ z1 + z2)))
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

    m_tsls = iv(TSLS(), df, @formula(y ~ (x ~ z1 + z2)))
    m_k1 = iv(KClass(1.0), df, @formula(y ~ (x ~ z1 + z2)))

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

    m = iv(LIML(), df, @formula(y ~ (x ~ z1 + z2)))

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
    m = iv(LIML(), df, @formula(y ~ (x ~ z1 + z2)), save_cluster = :cluster)

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
    m = iv(LIML(), df, @formula(y ~ (x1 + x2 ~ z1 + z2 + z3)))

    @test length(coef(m)) == 3  # 2 endogenous + intercept
    @test m.postestimation.kappa !== nothing
    @test m.estimator isa LIML
end

@testitem "LIML with FE" tags = [:iv, :kclass, :fe] begin
    using Regress
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
    m = iv(LIML(), df, @formula(y ~ (x ~ z1 + z2) + fe(fe_id)))

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
    m = iv(LIML(), df, @formula(y ~ x1 + x2 + state_id + (endo ~ z)))

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
    m2 = iv(LIML(), df, @formula(y ~ x1 + x2 + (endo ~ z)), save_cluster = :state_id)

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
    using DataFrames
    using CSV
    using StatsBase: stderror
    using CovarianceMatrices: CR1

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../test/data/iv_nested.csv")))

    # County FE is nested in state_id cluster
    # When FE is nested in cluster, DOF adjustment should account for nesting
    m = ols(df, @formula(y ~ x1 + x2 + endo + fe(county_id)), save_cluster = :state_id)

    se_cr1 = stderror(CR1(:state_id), m)
    @test all(isfinite, se_cr1)

    # Compare with state FE (non-nested - state is the cluster level)
    m2 = ols(df, @formula(y ~ x1 + x2 + endo + fe(state_id)), save_cluster = :state_id)
    se_cr1_state = stderror(CR1(:state_id), m2)
    @test all(isfinite, se_cr1_state)
end

@testitem "LIML with nested FE and cluster" tags = [:iv, :kclass, :fe, :cluster] begin
    using Regress
    using DataFrames
    using CSV
    using StatsBase: stderror
    using CovarianceMatrices: CR1

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../test/data/iv_nested.csv")))

    # LIML with state FE (absorbed), cluster at state level
    m = iv(LIML(), df, @formula(y ~ x1 + x2 + (endo ~ z) + fe(state_id)), save_cluster = :state_id)

    se_cr1 = stderror(CR1(:state_id), m)
    @test all(isfinite, se_cr1)
    # LIML kappa should be >= 1, allowing for floating point tolerance
    @test m.postestimation.kappa >= 1.0 - 1e-10
end
