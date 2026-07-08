@testitem "generic axes: matrix API" tags = [:generic_axes] begin
    using Regress
    using OffsetArrays
    using StableRNGs

    rng = StableRNGs.StableRNG(42)
    n = 40

    # OLS design + response, and an IV setup sharing the exogenous columns.
    X = hcat(ones(n), randn(rng, n, 2))
    y = X * [1.0, 2.0, -1.5] .+ 0.1 .* randn(rng, n)

    Z = hcat(ones(n), randn(rng, n, 2))          # 2 excluded instruments
    Xiv = hcat(ones(n), randn(rng, n))           # 1 exogenous + 1 endogenous
    yiv = Z[:, 1] .+ Xiv[:, 2] .+ 0.1 .* randn(rng, n)

    @testset "1-based lazy views match plain inputs" begin
        # `view` preserves 1-based axes; the matrix API must accept it and return
        # results identical to the materialized inputs.
        ref_ols = Regress.ols(X, y)
        v_ols = Regress.ols(view(X, :, :), view(y, :))
        @test coef(v_ols) == coef(ref_ols)
        @test stderror(v_ols) == stderror(ref_ols)

        for est in (TSLS(), LIML(), Fuller(1.0), KClass(0.5))
            ref_iv = Regress.iv(est, Z, Xiv, yiv)
            v_iv = Regress.iv(est, view(Z, :, :), view(Xiv, :, :), view(yiv, :))
            @test coef(v_iv) == coef(ref_iv)
            @test stderror(v_iv) == stderror(ref_iv)
        end
    end

    @testset "offset inputs are rejected with a clear message" begin
        # The matrix API materializes inputs into 1-based `Matrix`/`Vector` and
        # indexes positionally, so it declares `require_one_based_indexing`. An
        # offset argument — including a mix of offset and plain arguments whose
        # lengths still agree — must error immediately, not silently misalign.
        Xo = OffsetArray(X, -1, 0)
        yo = OffsetArray(y, -1)
        Zo = OffsetArray(Z, -1, 0)
        Xivo = OffsetArray(Xiv, -1, 0)
        yivo = OffsetArray(yiv, -1)

        msg = "offset arrays are not supported"

        @test_throws msg Regress.ols(Xo, yo)
        @test_throws msg Regress.ols(Xo, y)      # mixed: lengths agree, axes do not
        @test_throws msg Regress.ols(X, yo)

        for est in (TSLS(), LIML(), Fuller(1.0), KClass(0.5))
            @test_throws msg Regress.iv(est, Zo, Xivo, yivo)
            @test_throws msg Regress.iv(est, Z, Xivo, yiv)   # mixed
            @test_throws msg Regress.iv(est, Zo, Xiv, yiv)   # mixed
        end
    end
end
