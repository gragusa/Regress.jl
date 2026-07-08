using CSV
using DataFrames
using Test

include("regress_probit.jl")

const PROBIT_TEST_DIR = @__DIR__
const FIXEST_SCRIPT = joinpath(PROBIT_TEST_DIR, "fixest_probit.R")
const FIXEST_RESULTS = joinpath(PROBIT_TEST_DIR, "fixest_probit_results.csv")
const REGRESS_RESULTS = joinpath(PROBIT_TEST_DIR, "regress_probit_results.csv")

function rows_by_name(df::DataFrame, kind::AbstractString)
    out = Dict{String, Float64}()
    for row in eachrow(df[df.kind .== kind, :])
        out[String(row.name)] = Float64(row.value)
    end
    return out
end

function read_results(path::AbstractString)
    return DataFrame(CSV.File(path))
end

function run_fixest_probit(; samples::Integer = 5, nthreads::Integer = 1)
    cmd = `Rscript $FIXEST_SCRIPT $FIXEST_RESULTS $samples $nthreads`
    try
        run(cmd)
    catch err
        @warn "Skipping fixest comparison: Rscript/fixest is not available or failed" exception = err
        return nothing
    end
    return read_results(FIXEST_RESULTS)
end

@testset "probit: Regress.jl vs R fixest" begin
    samples = parse(Int, get(ENV, "PROBIT_BENCH_SAMPLES", "5"))
    nthreads = parse(Int, get(ENV, "PROBIT_FIXEST_THREADS", "1"))

    fixest = run_fixest_probit(; samples = samples, nthreads = nthreads)
    if fixest === nothing
        @test_skip "Rscript/fixest is not available"
    else
        regress = write_regress_probit_results(REGRESS_RESULTS; samples = samples)

        fixest_coef = rows_by_name(fixest, "coef")
        regress_coef = rows_by_name(regress, "coef")
        fixest_time = rows_by_name(fixest, "time")
        regress_time = rows_by_name(regress, "time")

        coef_names = ["age", "hhninc", "hhkids", "educ", "married"]
        expected_coef_names = Set(coef_names)
        @test Set(keys(regress_coef)) == expected_coef_names
        @test issubset(Set(keys(fixest_coef)), expected_coef_names)

        common_coef_names = [name
                             for name in coef_names
                             if haskey(fixest_coef, name) && haskey(regress_coef, name)]
        dropped_by_fixest = setdiff(coef_names, common_coef_names)
        @info "fixest coefficient set" common = common_coef_names dropped_by_fixest

        @test !isempty(common_coef_names)

        fixest_vector = [fixest_coef[name] for name in common_coef_names]
        regress_vector = [regress_coef[name] for name in common_coef_names]

        @test regress_vector≈fixest_vector rtol=5e-5 atol=1e-6

        @test isfinite(fixest_time["median_seconds"])
        @test isfinite(regress_time["median_seconds"])
        @test fixest_time["median_seconds"] > 0
        @test regress_time["median_seconds"] > 0

        ratio = regress_time["median_seconds"] / fixest_time["median_seconds"]
        @info "probit timing comparison" fixest_seconds=fixest_time["median_seconds"] regress_seconds=regress_time["median_seconds"] regress_over_fixest=ratio
    end
end
