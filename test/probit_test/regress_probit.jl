# Equivalent Julia/Regress.jl probit benchmark for:
# visit_dummy ~ age + hhninc + hhkids + educ + married + id FE + year FE

using CSV
using DataFrames
using Regress
using Regress: fe, probit
using Statistics
using StatsAPI: coef, loglikelihood, nobs

const RWM_NAMES = [
    :id, :female, :year, :age, :hsat, :handdum, :handper,
    :hhninc, :hhkids, :educ, :married, :haupts, :reals,
    :fachhs, :abitur, :univ, :working, :bluec, :whitec,
    :self, :beamt, :docvis, :hospvis, :public, :addon
]

const PROBIT_FORMULA = @formula(
    visit_dummy ~ age + hhninc + hhkids + educ + married + fe(id) + fe(year)
)

const PROBIT_BETA0 = zeros(5)

function load_rwm_data()
    repo_root = normpath(joinpath(@__DIR__, "..", ".."))
    data_path = joinpath(repo_root, "test", "data", "rwm.data")
    df = CSV.read(
        data_path,
        DataFrame;
        header = false,
        delim = ' ',
        ignorerepeated = true
    )
    rename!(df, RWM_NAMES)
    df[!, :visit_dummy] = ifelse.(df.docvis .> 0, 1, 0)
    return df
end

function fit_regress_probit(df)
    return probit(
        df,
        PROBIT_FORMULA;
        beta0 = copy(PROBIT_BETA0),
        max_iter = 1000,
        tolerance = 1e-6
    )
end

function time_regress_probit(df; samples::Integer = 5)
    # Warmup: avoid measuring compilation and first-call specialization.
    model = fit_regress_probit(df)

    elapsed = Float64[]
    for _ in 1:samples
        GC.gc()
        push!(elapsed, @elapsed fit_regress_probit(df))
    end

    return (
        model = model,
        median_seconds = median(elapsed),
        minimum_seconds = minimum(elapsed),
        samples = samples
    )
end

function regress_probit_results(; samples::Integer = 5)
    df = load_rwm_data()
    bench = time_regress_probit(df; samples = samples)
    model = bench.model

    names = ["age", "hhninc", "hhkids", "educ", "married"]
    values = Float64.(coef(model))

    return DataFrame(
        kind = vcat(
            fill("coef", length(names)),
            ["time", "time", "meta", "meta", "meta"]
        ),
        name = vcat(
            names,
            ["median_seconds", "minimum_seconds", "nobs", "loglikelihood", "samples"]
        ),
        value = vcat(
            values,
            [
                bench.median_seconds,
                bench.minimum_seconds,
                Float64(nobs(model)),
                Float64(loglikelihood(model)),
                Float64(bench.samples)
            ]
        )
    )
end

function write_regress_probit_results(output_csv::AbstractString; samples::Integer = 5)
    results = regress_probit_results(; samples = samples)
    CSV.write(output_csv, results)
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    output_csv = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "regress_probit_results.csv")
    samples = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 5
    write_regress_probit_results(output_csv; samples = samples)
    println("Regress.jl probit results written to: ", output_csv)
end
