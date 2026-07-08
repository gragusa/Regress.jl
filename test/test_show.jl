@testitem "OLS show - text and HTML" tags = [:ols, :show] begin
    using Regress, DataFrames, CSV, StatsBase

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    m = Regress.ols(df, @formula(Sales ~ NDI + CPI))

    txt = sprint(show, m)
    @test occursin("OLS", txt)
    @test occursin("Number of obs", txt)
    @test occursin("R²", txt)
    @test occursin("NDI", txt)
    @test occursin("(Intercept)", txt)
    @test occursin("Std. errors computed using HC1", txt)

    html = sprint(show, MIME("text/html"), m)
    @test occursin("<table", html)
    @test occursin("</table>", html)
    @test occursin("<caption>OLS</caption>", html)
    @test occursin("<thead", html)
    @test occursin("<tbody", html)
    @test occursin("<tfoot", html)
    @test occursin("NDI", html)
    # p-value column escapes ">" in the header
    @test occursin("Pr(&gt;|t|)", html)

    # Color output path: title in yellow, horizontal rules in green
    cbuf = IOBuffer()
    show(IOContext(cbuf, :color => true), m)
    colored = String(take!(cbuf))
    @test occursin("\e[33m", colored)  # yellow title
    @test occursin("\e[32m", colored)  # green rule
end

@testitem "OLS show - FE model reports iterations" tags = [:ols, :fe, :show] begin
    using Regress, DataFrames, CSV, StatsBase
    using Regress: fe

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    m = Regress.ols(df, @formula(Sales ~ NDI + fe(State)))

    txt = sprint(show, m)
    @test occursin("Iterations", txt)
    @test occursin("R² within", txt)
end

@testitem "OLS show - vcov type labels" tags = [:ols, :show] begin
    using Regress, DataFrames, CSV, StatsBase
    using CovarianceMatrices

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    df.StateInt = Int.(df.State)
    m = Regress.ols(df, @formula(Sales ~ NDI + CPI); save_cluster = :StateInt)

    @test occursin("HC1", sprint(show, m + vcov(HC1())))
    @test occursin("HC3", sprint(show, m + vcov(HC3())))
    @test occursin("CR1", sprint(show, m + vcov(CR1(:StateInt))))
    @test occursin("Bartlett(5)", sprint(show, m + vcov(Bartlett(5))))
end

@testitem "IV show - text and HTML" tags = [:iv, :show] begin
    using Regress, DataFrames, CSV, StatsBase

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    m = Regress.iv(TSLS(), df, @formula(Sales ~ CPI + (Price ~ Pimin)))

    txt = sprint(show, m)
    @test occursin("TSLS", txt)
    @test occursin("Number of obs", txt)
    @test occursin("1st stage", txt)
    @test occursin("Price", txt)
    @test occursin("(Intercept)", txt)

    html = sprint(show, MIME("text/html"), m)
    @test occursin("<table", html)
    @test occursin("<caption>TSLS</caption>", html)
    @test occursin("Price", html)
    @test occursin("1st stage", html)
end

@testitem "FirstStageResult show - text and HTML" tags = [:iv, :show] begin
    using Regress, DataFrames, CSV, StatsBase

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    m = Regress.iv(TSLS(), df, @formula(Sales ~ CPI + (Price ~ Pimin)))
    fs = first_stage(m)

    txt = sprint(show, fs)
    @test occursin("First-Stage Diagnostics", txt)
    @test occursin("Per-Endogenous F-Statistics", txt)
    @test occursin("Price", txt)
    @test occursin("F[nonrobust]", txt)
    @test occursin("DoF:", txt)
    @test occursin("Instruments:", txt)

    html = sprint(show, MIME("text/html"), fs)
    @test occursin("<table", html)
    @test occursin("First-Stage Diagnostics", html)
    @test occursin("Endogenous", html)
    @test occursin("Price", html)
end

@testitem "WeakIVTestResult show" tags = [:iv, :weakiv, :show] begin
    using Regress, DataFrames, CSV, StatsBase

    df = DataFrame(CSV.File(joinpath(dirname(pathof(Regress)), "../dataset/Cigar.csv")))
    m = Regress.iv(TSLS(), df, @formula(Sales ~ CPI + (Price ~ Pimin)))
    r = weakivtest(m)

    txt = sprint(show, r)
    @test occursin("Montiel-Pflueger robust weak instrument test", txt)
    @test occursin("TSLS coefficient", txt)
    @test occursin("F[effective]", txt)
    @test occursin("F[robust]", txt)
    @test occursin("tau=5%", txt)
    @test occursin("tau=30%", txt)
end

@testitem "IV estimator type show" tags = [:iv, :show] begin
    using Regress

    @test sprint(show, TSLS()) == "TSLS()"
    @test sprint(show, LIML()) == "LIML()"
    @test sprint(show, Fuller()) == "Fuller()"
    @test sprint(show, Fuller(4.0)) == "Fuller(4.0)"
    @test sprint(show, KClass(0.5)) == "KClass(0.5)"
end
