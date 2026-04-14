@testitem "parse_fixedeffect - basic" tags = [:formula, :fe, :smoke] begin
    using CSV, DataFrames
    using Regress
    using Regress: fe, parse_fixedeffect, _parse_fixedeffect, _multiply
    using FixedEffects

    fe_eq(x::FixedEffect, y::FixedEffect) = x.refs == y.refs &&
                                            x.interaction == y.interaction && x.n == y.n
    fe_eq(a::Tuple, b::Tuple) = length(a) == length(b) &&
                                all(fe_eq(ai, bi) for (ai, bi) in zip(a, b))
    fe_eq(a::Vector{<:FixedEffect},
        b::Vector{<:FixedEffect}) = length(a) == length(b) &&
                                    all(fe_eq(ai, bi) for (ai, bi) in zip(a, b))
    tup_eq(a::Tuple, b::Tuple) = length(a) == length(b) &&
                                 all(_eq(ai, bi) for (ai, bi) in zip(a, b))
    _eq(a, b) = a == b
    _eq(a::FixedEffect, b::FixedEffect) = fe_eq(a, b)
    _eq(a::Vector{<:FixedEffect}, b::Vector{<:FixedEffect}) = fe_eq(a, b)
    _eq(a::Tuple, b::Tuple) = length(a) == length(b) &&
                              all(_eq(ai, bi) for (ai, bi) in zip(a, b))

    csvfile = CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv"))
    df = DataFrame(csvfile)

    # Any table type supporting the Tables.jl interface should work
    for data in [df, csvfile]
        @test _parse_fixedeffect(data, term(:Price)) === nothing
        @test _parse_fixedeffect(data, ConstantTerm(1)) === nothing
        @test _eq(_parse_fixedeffect(data, fe(:State)),
            (FixedEffect(data.State), :fe_State, [:State]))

        @test parse_fixedeffect(data, ()) == (FixedEffect[], Symbol[], Symbol[])

        f = @formula(y ~ 1 + Price)
        ts1 = f.rhs
        ts2 = term(1) + term(:Price)
        @test parse_fixedeffect(data, f) == (FixedEffect[], Symbol[], Symbol[])
        @test parse_fixedeffect(data, ts1) == (FixedEffect[], Symbol[], Symbol[])
        @test _eq(parse_fixedeffect(data, ts2), parse_fixedeffect(data, ts1))

        f = @formula(y ~ 1 + Price + fe(State))
        ts1 = f.rhs
        ts2 = term(1) + term(:Price) + fe(:State)
        @test _eq(parse_fixedeffect(data, f),
            ([FixedEffect(data.State)], [:fe_State], [:State]))
        @test _eq(parse_fixedeffect(data, ts1),
            ([FixedEffect(data.State)], [:fe_State], [:State]))
        @test _eq(parse_fixedeffect(data, ts2), parse_fixedeffect(data, ts1))

        f = @formula(y ~ Price + fe(State) + fe(Year))
        ts1 = f.rhs
        ts2 = term(:Price) + fe(:State) + fe(:Year)
        @test _eq(parse_fixedeffect(data, f),
            ([FixedEffect(data.State), FixedEffect(data.Year)],
                [:fe_State, :fe_Year], [:State, :Year]))
        @test _eq(parse_fixedeffect(data, ts1),
            ([FixedEffect(data.State), FixedEffect(data.Year)],
                [:fe_State, :fe_Year], [:State, :Year]))
        @test _eq(parse_fixedeffect(data, ts2), parse_fixedeffect(data, ts1))
    end
end

@testitem "parse_fixedeffect - interactions" tags = [:formula, :fe] begin
    using CSV, DataFrames
    using Regress
    using Regress: fe, parse_fixedeffect, _parse_fixedeffect, _multiply
    using FixedEffects

    fe_eq(x::FixedEffect, y::FixedEffect) = x.refs == y.refs &&
                                            x.interaction == y.interaction && x.n == y.n
    _eq(a, b) = a == b
    _eq(a::FixedEffect, b::FixedEffect) = fe_eq(a, b)
    _eq(a::Vector{<:FixedEffect},
        b::Vector{<:FixedEffect}) = length(a) == length(b) &&
                                    all(fe_eq(ai, bi) for (ai, bi) in zip(a, b))
    _eq(a::Tuple, b::Tuple) = length(a) == length(b) &&
                              all(_eq(ai, bi) for (ai, bi) in zip(a, b))

    csvfile = CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv"))
    df = DataFrame(csvfile)

    # Any table type supporting the Tables.jl interface should work
    for data in [df, csvfile]
        @test _eq(_parse_fixedeffect(data, fe(:State)&term(:Year)),
            (FixedEffect(data.State, interaction = _multiply(data, [:Year])),
                Symbol("fe_State&Year"), [:State]))
        @test _eq(_parse_fixedeffect(data, fe(:State)&fe(:Year)),
            (
                FixedEffect(data.State, data.Year), Symbol("fe_State&fe_Year"), [
                    :State, :Year]))

        f = @formula(y ~ Price + fe(State)&Year)
        ts1 = f.rhs
        ts2 = term(:Price) + fe(:State)&term(:Year)
        @test _eq(parse_fixedeffect(data, f),
            ([FixedEffect(data.State, interaction = _multiply(data, [:Year]))],
                [Symbol("fe_State&Year")], [:State]))
        @test _eq(parse_fixedeffect(data, ts1),
            ([FixedEffect(data.State, interaction = _multiply(data, [:Year]))],
                [Symbol("fe_State&Year")], [:State]))
        @test _eq(parse_fixedeffect(data, ts2), parse_fixedeffect(data, ts1))

        f = @formula(y ~ Price + fe(State)*fe(Year))
        ts1 = f.rhs
        ts2 = term(:Price) + fe(:State) + fe(:Year) + fe(:State)&fe(:Year)
        @test _eq(parse_fixedeffect(data, f),
            (
                [
                    FixedEffect(data.State), FixedEffect(data.Year), FixedEffect(data.State, data.Year)],
                [:fe_State, :fe_Year, Symbol("fe_State&fe_Year")],
                [:State, :Year]))
        @test _eq(parse_fixedeffect(data, ts1),
            (
                [
                    FixedEffect(data.State), FixedEffect(data.Year), FixedEffect(data.State, data.Year)],
                [:fe_State, :fe_Year, Symbol("fe_State&fe_Year")],
                [:State, :Year]))
        @test _eq(parse_fixedeffect(data, ts2), parse_fixedeffect(data, ts1))
    end
end

@testitem "parse_fixedeffect - Tables.jl" tags = [:formula] begin
    using CSV, DataFrames
    using Regress
    using Regress: fe, parse_fixedeffect
    using FixedEffects

    fe_eq(x::FixedEffect, y::FixedEffect) = x.refs == y.refs &&
                                            x.interaction == y.interaction && x.n == y.n
    _eq(a, b) = a == b
    _eq(a::FixedEffect, b::FixedEffect) = fe_eq(a, b)
    _eq(a::Vector{<:FixedEffect},
        b::Vector{<:FixedEffect}) = length(a) == length(b) &&
                                    all(fe_eq(ai, bi) for (ai, bi) in zip(a, b))
    _eq(a::Tuple, b::Tuple) = length(a) == length(b) &&
                              all(_eq(ai, bi) for (ai, bi) in zip(a, b))

    csvfile = CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv"))
    df = DataFrame(csvfile)

    # Test that both CSV.File and DataFrame produce the same results
    f = @formula(y ~ Price + fe(State) + fe(Year))
    result_df = parse_fixedeffect(df, f)
    result_csv = parse_fixedeffect(csvfile, f)

    @test _eq(result_df, result_csv)
end

@testitem "lags - numerical correctness" tags = [:formula, :lags] begin
    using DataFrames, LinearAlgebra
    using Regress
    using StableRNGs

    rng = StableRNG(42)
    n = 100
    x = randn(rng, n)
    y = randn(rng, n)
    df = DataFrame(y = y, x = x)

    # --- Test 1: lags(x, 3) matches manual X\y ---
    nlags = 3
    # Valid rows: nlags+1 to n (rows where all lags are available)
    valid = (nlags + 1):n
    X_manual = hcat(ones(length(valid)),
        x[valid .- 1],
        x[valid .- 2],
        x[valid .- 3])
    beta_manual = X_manual \ y[valid]

    m = Regress.ols(df, @formula(y ~ lags(x, 3)))
    @test coef(m) ≈ beta_manual
    @test nobs(m) == length(valid)

    # --- Test 2: lags(log(y), 3) — transform then lag ---
    ypos = exp.(randn(rng, n))  # positive values for log
    w = randn(rng, n)
    df2 = DataFrame(w = w, ypos = ypos)

    logy = log.(ypos)
    X_manual2 = hcat(ones(length(valid)),
        logy[valid .- 1],
        logy[valid .- 2],
        logy[valid .- 3])
    beta_manual2 = X_manual2 \ w[valid]

    m2 = Regress.ols(df2, @formula(w ~ lags(log(ypos), 3)))
    @test coef(m2) ≈ beta_manual2
    @test nobs(m2) == length(valid)

    # --- Test 3: coefficient names ---
    cn = coefnames(m)
    @test cn[1] == "(Intercept)"
    @test cn[2] == "x_lag1"
    @test cn[3] == "x_lag2"
    @test cn[4] == "x_lag3"

    # --- Test 4: combined with other terms ---
    z = randn(rng, n)
    df3 = DataFrame(y = y, x = x, z = z)
    m3 = Regress.ols(df3, @formula(y ~ z + lags(x, 2)))
    @test length(coef(m3)) == 4  # intercept + z + 2 lags

    # Verify against manual
    valid2 = 3:n
    X_manual3 = hcat(ones(length(valid2)),
        z[valid2],
        x[valid2 .- 1],
        x[valid2 .- 2])
    beta_manual3 = X_manual3 \ y[valid2]
    @test coef(m3) ≈ beta_manual3

    # --- Test 5: lags with nested transforms combined with other function terms ---
    # Regression test: lags(log(abs(x)), 3) + log(x) must not error
    xpos = abs.(x) .+ 0.1
    df5 = DataFrame(y = y, xpos = xpos)
    m5 = Regress.ols(df5, @formula(y ~ lags(log(abs(xpos)), 3) + log(xpos)))
    @test length(coef(m5)) == 5  # intercept + 3 lags + log(xpos)

    logabsx = log.(abs.(xpos))
    valid5 = (nlags + 1):n
    X_manual5 = hcat(ones(length(valid5)),
        logabsx[valid5 .- 1],
        logabsx[valid5 .- 2],
        logabsx[valid5 .- 3],
        log.(xpos[valid5]))
    beta_manual5 = X_manual5 \ y[valid5]
    @test coef(m5) ≈ beta_manual5

    # Verify coefficient names for nested transforms
    cn5 = coefnames(m5)
    @test cn5[2] == "log(abs(xpos))_lag1"
    @test cn5[3] == "log(abs(xpos))_lag2"
    @test cn5[4] == "log(abs(xpos))_lag3"
end

@testitem "lags - esample correctness" tags = [:formula, :lags] begin
    using DataFrames
    using Regress
    using StableRNGs

    rng = StableRNG(99)
    n = 20

    # --- Test 1: basic esample with lags ---
    df1 = DataFrame(x = randn(rng, n), y = randn(rng, n))
    m1 = Regress.ols(df1, @formula(y ~ lags(x, 2)))
    @test sum(m1.esample) == nobs(m1) == n - 2
    # First 2 rows should be false (NaN from lags), rest true
    @test m1.esample[1:2] == [false, false]
    @test all(m1.esample[3:n])

    # --- Test 2: missing at start + NaN from lags ---
    # Row 1 removed by completecases. Subsetted data is rows 2:n (19 rows).
    # lags(x,2) → NaN in first 2 rows of subsetted data → original rows 2,3.
    y2 = Vector{Union{Float64, Missing}}(randn(rng, n))
    y2[1] = missing  # row 1: missing
    df2 = DataFrame(x = randn(rng, n), y = y2)
    m2 = Regress.ols(df2, @formula(y ~ lags(x, 2)))
    @test m2.esample[1] == false   # missing
    @test m2.esample[2] == false   # NaN from lag (1st row of subsetted data)
    @test m2.esample[3] == false   # NaN from lag (2nd row of subsetted data)
    @test all(m2.esample[4:n])
    @test sum(m2.esample) == nobs(m2) == n - 3

    # --- Test 3: missing in y and x in the middle ---
    y3 = Vector{Union{Float64, Missing}}(randn(rng, n))
    x3 = Vector{Union{Float64, Missing}}(randn(rng, n))
    y3[10] = missing   # missing y at row 10
    x3[12] = missing   # missing x at row 12
    df3 = DataFrame(x = x3, y = y3)
    m3 = Regress.ols(df3, @formula(y ~ lags(x, 2)))
    @test m3.esample[1:2] == [false, false]  # NaN from lags
    @test m3.esample[10] == false             # missing y
    @test m3.esample[12] == false             # missing x
    @test sum(m3.esample) == nobs(m3) == n - 4

    # --- Test 4: missing at end ---
    y4 = Vector{Union{Float64, Missing}}(randn(rng, n))
    y4[n] = missing
    df4 = DataFrame(x = randn(rng, n), y = y4)
    m4 = Regress.ols(df4, @formula(y ~ lags(x, 2)))
    @test m4.esample[1:2] == [false, false]   # NaN from lags
    @test m4.esample[n] == false               # missing
    @test all(m4.esample[3:(n - 1)])
    @test sum(m4.esample) == nobs(m4) == n - 3

    # --- Test 5: NaN in y at start and end (not missing) ---
    df5 = DataFrame(x = randn(rng, n), y = [NaN; NaN; NaN; randn(rng, n - 4); NaN])
    m5 = Regress.ols(df5, @formula(y ~ x + lags(x, 2)))
    @test m5.esample[1:3] == [false, false, false]  # NaN y + NaN lags
    @test m5.esample[n] == false                      # NaN y at end
    @test sum(m5.esample) == nobs(m5) == n - 4

    # --- Test 6: residuals alignment via esample ---
    df6 = DataFrame(x = randn(rng, n), y = randn(rng, n))
    m6 = Regress.ols(df6, @formula(y ~ lags(x, 3)))
    res_full = fill(NaN, n)
    res_full[m6.esample] .= residuals(m6)
    @test all(!isnan, res_full[m6.esample])
    @test all(isnan, res_full[.!m6.esample])
    @test length(residuals(m6)) == sum(m6.esample)
end
