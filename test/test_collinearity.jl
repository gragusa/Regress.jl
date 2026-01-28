@testitem "collinearity with fixed effects" tags = [:ols, :fe, :vcov] begin
    using DataFrames, CSV, Regress, Random, Statistics
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

    # run the regression with the default setting (tol = 1e-6)
    rr = ols(
        dflarge, @formula(Price ~ highstate + Pop + fe(Year) + fe(catvar) + fe(State)),
        save_cluster = :State, tol = 1e-6)

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
    rr = ols(df, @formula(x1 ~ x2))
    @test all(!isnan, stderror(rr))
end
