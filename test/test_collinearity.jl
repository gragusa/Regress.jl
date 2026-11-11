@testitem "collinearity with fixed effects" tags = [:ols, :fe, :vcov] begin
    using DataFrames, CSV, Regress, Random, Statistics
    using Regress: fe
    using StatsBase: coef, stderror
    using CovarianceMatrices: CR1

    # read the data
    csvfile = CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv"))
    df = DataFrame(csvfile)

    # create a bigger dataset
    sort!(df, [:State, :Year])
    nstates = maximum(df.State)
    dflarge = copy(df)
    for i in 1:100
        dfnew = copy(df)
        dfnew.State .+= i .* nstates
        append!(dflarge, dfnew)
    end

    # create a dummy variable that is collinear with the State-FE
    dflarge.highstate = dflarge.State .< median(dflarge.State)

    # create a second 'high-dimensional' categorical variable
    Random.seed!(1234)
    dflarge.catvar = rand(1:200, nrow(dflarge))

    rr = Regress.ols(
        dflarge, @formula(Price ~ highstate + Pop + fe(Year) + fe(catvar) + fe(State)),
        save_cluster = :State, fe_tol = 1e-6)

    # test that the collinear coefficient is zero and the standard error is NaN
    @test coef(rr)[1] ≈ 0.0
    @test isnan(stderror(CR1(:State), rr)[1])
end

@testitem "collinearity - small data" tags = [:ols] begin
    using DataFrames, Regress
    using StatsBase: stderror

    df = DataFrame(
        [36.9302 44.5105;
         39.4935 44.5044;
         38.946 44.5072;
         37.8005 44.5098;
         37.2613 44.5103;
         35.3885 44.5109;],
        :auto
    )
    rr = Regress.ols(df, @formula(x1 ~ x2))
    @test all(!isnan, stderror(rr))
end

@testitem "sweep collinearity detection drops later columns" tags = [:ols] begin
    using Regress, LinearAlgebra, StableRNGs

    rng = StableRNG(3)
    n = 200
    x1, x2, x3 = randn(rng, n), 1e3 .* randn(rng, n), randn(rng, n)
    X = [ones(n) x1 x2 2 .* x2 x3 x1 .+ x2]

    # The column that the earlier ones explain is dropped, as in sweep_solve!.
    for has_intercept in (true, false)
        basis, _ = Regress.detect_collinearity(X; method = :sweep, has_intercept)
        @test basis == [true, true, true, false, true, false]
        aug = Matrix(X' * X)
        b, _, _ = Regress.sweep_solve!(aug, X' * x3, has_intercept; tol = 1e-8)
        @test b == basis
    end

    # On a full-rank matrix the sweep leaves -inv(A) in the upper triangle.
    A = Matrix(X[:, [1, 2, 3, 5]]' * X[:, [1, 2, 3, 5]])
    S = copy(A)
    basis, rank = Regress.sweep_collinear!(S, 1e-8)
    @test all(basis) && rank == 4
    @test Symmetric(S, :U) ≈ -inv(A) rtol = 1e-10

    # Both detection methods find the same rank; :sweep keeps formula order.
    y = x1 .+ randn(rng, n)
    m = Regress.ols(X, y)
    @test isnan(stderror(m)[4]) && isnan(stderror(m)[6])
    @test sum(Regress.ols(X, y; collinearity = :qr).basis_coef) == 4
end

@testitem "leverage-based variances with collinear columns" tags = [:ols, :vcov] begin
    using Regress, DataFrames, StableRNGs, StatsBase
    using CovarianceMatrices

    rng = StableRNG(6)
    n = 200
    df = DataFrame(x1 = randn(rng, n), x2 = randn(rng, n), g = rand(rng, 1:20, n))
    df.x3 = df.x1 .+ df.x2
    df.y = df.x1 .+ randn(rng, n)
    X = [ones(n) df.x1 df.x2 df.x3]

    # The leverage covers the estimable columns, for every factorization and both paths.
    for fac in (:chol, :qr, :sweep)
        mf = Regress.ols(df, @formula(y ~ x1 + x2 + x3); factorization = fac)
        mm = Regress.ols(X, df.y; factorization = fac)
        for m in (mf, mm), K in (HC2(), HC3())

            se = stderror(K, m)
            @test se[1:3] ≈ stderror(K, Regress.ols(X[:, 1:3], df.y)) rtol = 1e-10
            @test isnan(se[4])
        end
        @test stderror(CR2(df.g), mm)[1:3] ≈
              stderror(CR2(df.g), Regress.ols(X[:, 1:3], df.y)) rtol = 1e-10
        @test leverage(mm) ≈ leverage(Regress.ols(X[:, 1:3], df.y)) rtol = 1e-10
    end
end

@testitem "IV covariances with collinear exogenous columns" tags = [:iv, :vcov] begin
    using DataFrames, Regress, StableRNGs
    using CovarianceMatrices: HC1, HC3, Bartlett, NeweyWest, EWC
    using StatsBase: coef, stderror, vcov

    rng = StableRNG(11)
    n = 200
    z, w = randn(rng, n), randn(rng, n)
    x = 0.5 .* z .+ 0.3 .* w .+ randn(rng, n)
    y = 0.3 .* x .+ w .+ randn(rng, n)
    df = DataFrame(; y, x, z, w, w2 = 2 .* w)
    Zm, Xm = [ones(n) w 2 .* w z], [ones(n) w 2 .* w x]

    # A post-estimation covariance has NaN for the dropped column and equals the
    # covariance of the model without it elsewhere, in the formula and matrix APIs.
    for est in (Regress.TSLS(), Regress.Fuller(1.0)),
        k in (HC1(), HC3(), Bartlett{NeweyWest}(), EWC(8))

        dup = Regress.iv(est, df, @formula(y ~ w + w2 + (x ~ z)); save = :all) + vcov(k)
        clean = Regress.iv(est, df, @formula(y ~ w + (x ~ z)); save = :all) + vcov(k)
        mat = Regress.iv(est, Zm, Xm, y; n_endogenous = 1) + vcov(k)
        keep = [1, 2, 4]
        for m in (dup, mat)
            V = vcov(m)
            @test size(V) == (4, 4)
            @test all(isnan, V[3, :]) && all(isnan, V[:, 3])
            @test V[keep, keep] ≈ vcov(clean) rtol = 1e-8
            @test stderror(m)[keep] ≈ stderror(clean) rtol = 1e-8
        end
        @test Regress.first_stage(mat).F_robust ≈ Regress.first_stage(clean).F_robust rtol = 1e-8
        k isa EWC && continue
        @test Regress.weakivtest(mat).F_eff ≈ Regress.weakivtest(clean).F_eff rtol = 1e-8
    end
end
