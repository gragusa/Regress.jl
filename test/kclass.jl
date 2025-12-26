##############################################################################
##
## K-Class Estimators Tests (LIML, Fuller, KClass)
##
##############################################################################

using Test
using Regress
using Regress.StableRNGs: StableRNG
using DataFrames
using StatsBase: coef, vcov
using CovarianceMatrices: HC1, HC3, CR1

# Helper to create test data
function create_test_data(n = 500; rng = StableRNG(42))
    z1 = randn(rng, n)
    z2 = randn(rng, n)
    e = randn(rng, n)
    u = 0.5 .* e .+ randn(rng, n)
    x = 0.5 .* z1 .+ 0.3 .* z2 .+ u
    y = 2.0 .* x .+ 1.0 .+ e
    DataFrame(y = y, x = x, z1 = z1, z2 = z2)
end

@testset "LIML basic functionality" begin
    df = create_test_data()

    m = iv(LIML(), df, @formula(y ~ (x ~ z1 + z2)))

    @test length(coef(m)) == 2
    @test m.postestimation.kappa !== nothing
    @test m.postestimation.kappa >= 1.0  # LIML kappa should be >= 1
    @test m.estimator isa LIML
end

@testset "Fuller basic functionality" begin
    df = create_test_data()

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

@testset "KClass(1.0) equals TSLS" begin
    df = create_test_data()

    m_tsls = iv(TSLS(), df, @formula(y ~ (x ~ z1 + z2)))
    m_k1 = iv(KClass(1.0), df, @formula(y ~ (x ~ z1 + z2)))

    # Coefficients should match (note: different algorithms so tolerance needed)
    @test coef(m_tsls) ≈ coef(m_k1) atol=1e-6
end

@testset "K-class vcov works" begin
    df = create_test_data()

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

@testset "Cluster-robust vcov for K-class" begin
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

@testset "Multiple endogenous variables" begin
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

@testset "LIML with fixed effects" begin
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
