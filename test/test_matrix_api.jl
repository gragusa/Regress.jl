@testitem "matrix OLS - accessors and factorizations" tags = [:ols, :matrix] begin
    using Regress, StatsBase

    n = 200
    t = range(0, 1; length = n)
    X = hcat(ones(n), cos.(2π .* t), sin.(2π .* t))
    beta = [1.0, 2.0, -1.5]
    noise = 0.05 .* cos.(7π .* t)   # deterministic, not collinear with X
    y = X * beta .+ noise

    ref = Regress.ols(X, y)

    # The matrix API accepts :auto/:chol/:qr (not :sweep, which is formula-only).
    for fac in (:auto, :chol, :qr)
        m = Regress.ols(X, y; factorization = fac)
        @test coef(m) ≈ coef(ref)
        @test coef(m) ≈ beta atol = 0.02
        @test length(residuals(m)) == n
        @test length(fitted(m)) == n
        @test fitted(m) .+ residuals(m) ≈ y
        @test size(modelmatrix(m)) == (n, 3)
        @test nobs(m) == n
        @test dof(m) == 3
        @test dof_residual(m) == n - 3
        @test response(m) === y
        @test 0.0 < r2(m) < 1.0
        @test isfinite(adjr2(m))
        @test all(isfinite, vcov(m))
        @test vcov(m) ≈ vcov(ref) rtol = 1e-6
    end
end

@testitem "matrix OLS - vcov re-specification via + operator" tags = [:ols, :matrix] begin
    using Regress, StatsBase
    using CovarianceMatrices

    n = 200
    t = range(0, 1; length = n)
    X = hcat(ones(n), cos.(2π .* t), sin.(2π .* t))
    y = X * [0.5, 1.0, -0.3] .+ 0.1 .* (t .- 0.5)
    m = Regress.ols(X, y)

    m3 = m + vcov(HC3())
    @test stderror(m3) ≈ stderror(HC3(), m)
    @test stderror(m3) != stderror(m)          # HC1 default differs from HC3
    @test coef(m3) == coef(m)                   # coefficients unchanged
    @test size(confint(m3)) == (3, 2)
end

@testitem "matrix OLS - confint variants" tags = [:ols, :matrix] begin
    using Regress, StatsBase
    using CovarianceMatrices

    n = 200
    t = range(0, 1; length = n)
    X = hcat(ones(n), cos.(2π .* t), sin.(2π .* t))
    y = X * [0.5, 1.0, -0.3] .+ 0.1 .* (t .- 0.5)
    m = Regress.ols(X, y)

    ci_default = confint(m)
    ci_hc3 = confint(HC3(), m)
    ci_90 = confint(m; level = 0.90)

    @test size(ci_default) == (3, 2)
    @test size(ci_hc3) == (3, 2)
    @test size(ci_90) == (3, 2)
    # A 90% interval is narrower than the default 95% interval.
    @test all((ci_90[:, 2] .- ci_90[:, 1]) .< (ci_default[:, 2] .- ci_default[:, 1]))
end

@testitem "matrix OLS - collinear columns" tags = [:ols, :matrix] begin
    using Regress, StatsBase

    n = 200
    t = range(0, 1; length = n)
    x1 = cos.(2π .* t)
    x2 = sin.(2π .* t)
    X = hcat(ones(n), x1, x2, x2)   # last column duplicates x2
    y = X[:, 1:3] * [1.0, 2.0, -0.5]

    m = Regress.ols(X, y)
    c = coef(m)
    @test any(==(0.0), c)             # collinear column coefficient set to zero
    @test any(isnan, stderror(m))     # its standard error is NaN
end

@testitem "OLS save=:minimal clears data" tags = [:ols] begin
    using Regress, DataFrames, CSV, StatsBase

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))

    m = Regress.ols(df, @formula(Sales ~ NDI + CPI); save = :minimal)
    @test !Regress.has_matrices(m)
    # Coefficients and standard errors survive :minimal.
    @test all(isfinite, coef(m))
    @test all(isfinite, stderror(m))
    # Data-dependent accessors report the data is gone.
    @test_throws "not stored" response(m)
    @test_throws "not stored" fitted(m)
    @test_throws "not stored" modelmatrix(m)
end

@testitem "deepcopy_vcov - cluster estimators are independent" tags = [:vcov] begin
    using Regress
    using CovarianceMatrices

    # Symbol-based CR estimators are immutable: same instance is fine.
    crs = CR1(:g)
    @test Regress.deepcopy_vcov(crs) === crs

    # HR estimators are stateless singletons.
    @test Regress.deepcopy_vcov(HC2()) === HC2()

    # Data-vector CR estimators must copy their cluster groups so mutating the
    # copy cannot alter the original.
    ids = repeat(1:25, inner = 4)
    cr = CR1(ids)
    crc = Regress.deepcopy_vcov(cr)
    @test typeof(crc) == typeof(cr)
    @test crc.g[1].groups !== cr.g[1].groups
    @test crc.g[1].groups == cr.g[1].groups
end
