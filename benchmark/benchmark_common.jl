#
# Common benchmark utilities shared across all benchmark configurations
#

using DataFrames, Random
using BenchmarkTools
using CSV
using Printf, Dates

# Import packages with qualified names to avoid fe() conflict
import FixedEffectModels as FEM
import Regress as REG
using CovarianceMatrices: CR1, vcov

# Configure BenchmarkTools
BenchmarkTools.DEFAULT_PARAMETERS.samples = 5
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 120

# ------------------------------------------------------------------------------
# Data Structures
# ------------------------------------------------------------------------------

struct BenchmarkResult
    scenario::String
    n::Int
    fem_time::Float64
    regress_time::Float64
    fixest_time::Float64
end

# For CUDA benchmarks
struct CudaBenchmarkResult
    scenario::String
    n::Int
    regress_cuda_time::Float64
    fixest_threaded_time::Float64
end

# ------------------------------------------------------------------------------
# Formatting Helpers
# ------------------------------------------------------------------------------

fmt(x::Float64) = isnan(x) ? "N/A" : @sprintf("%.3f", x)
function format_n(n::Int)
    n >= 1_000_000 ? "$(div(n, 1_000_000))M" : n >= 1_000 ? "$(div(n, 1_000))K" : string(n)
end

# ------------------------------------------------------------------------------
# Markdown Generation
# ------------------------------------------------------------------------------

function generate_markdown_header(title::String, config::Dict{String, Any})
    lines = String[]
    push!(lines, "# $title")
    push!(lines, "")
    push!(lines, "## Configuration")
    push!(lines, "")
    for (key, value) in sort(collect(config), by = first)
        push!(lines, "- **$key:** $value")
    end
    push!(lines, "- **Date:** $(Dates.now())")
    push!(lines, "")
    return join(lines, "\n")
end

function generate_standard_table(results::Vector{BenchmarkResult})
    lines = String[]
    push!(lines, "## Results")
    push!(lines, "")
    push!(lines, "| Scenario | N | FEM (s) | Regress (s) | fixest (s) | Regress/FEM | fixest/FEM |")
    push!(lines, "|----------|---|---------|-------------|------------|-------------|------------|")

    for r in results
        ratio_regress = r.fem_time > 0 ? r.regress_time / r.fem_time : NaN
        ratio_fixest = r.fem_time > 0 ? r.fixest_time / r.fem_time : NaN
        scenario_escaped = replace(r.scenario, "|" => "\\|")
        line = "| $(scenario_escaped) | $(format_n(r.n)) | $(fmt(r.fem_time)) | $(fmt(r.regress_time)) | $(fmt(r.fixest_time)) | $(fmt(ratio_regress)) | $(fmt(ratio_fixest)) |"
        push!(lines, line)
    end

    push!(lines, "")
    push!(lines, "**Legend:**")
    push!(lines, "- FEM = FixedEffectModels.jl")
    push!(lines, "- Regress = Regress.jl")
    push!(lines, "- fixest = R fixest package")
    push!(lines, "- Ratio < 1 means faster than FEM")
    push!(lines, "")
    return join(lines, "\n")
end

function generate_cuda_table(results::Vector{CudaBenchmarkResult})
    lines = String[]
    push!(lines, "## Results")
    push!(lines, "")
    push!(lines, "| Scenario | N | Regress CUDA (s) | fixest threaded (s) | CUDA/fixest |")
    push!(lines, "|----------|---|------------------|---------------------|-------------|")

    for r in results
        ratio = r.fixest_threaded_time > 0 ? r.regress_cuda_time / r.fixest_threaded_time :
                NaN
        scenario_escaped = replace(r.scenario, "|" => "\\|")
        line = "| $(scenario_escaped) | $(format_n(r.n)) | $(fmt(r.regress_cuda_time)) | $(fmt(r.fixest_threaded_time)) | $(fmt(ratio)) |"
        push!(lines, line)
    end

    push!(lines, "")
    push!(lines, "**Legend:**")
    push!(lines, "- Regress CUDA = Regress.jl with CUDA acceleration")
    push!(lines, "- fixest threaded = R fixest with OpenMP threading")
    push!(lines, "- Ratio < 1 means CUDA is faster")
    push!(lines, "")
    return join(lines, "\n")
end

# ------------------------------------------------------------------------------
# R fixest benchmarks runner
# ------------------------------------------------------------------------------

function run_fixest_benchmarks(nthreads::Int; script_dir::String = @__DIR__)
    println("\n" * "=" ^ 70)
    println("Running fixest (R) benchmarks via shell...")
    println("Requested threads: $nthreads")
    println("=" ^ 70)

    script_path = joinpath(dirname(script_dir), "benchmark_fixest.R")
    results_path = joinpath(script_dir, "fixest_results.csv")

    cmd = `Rscript $script_path $nthreads $results_path`
    println("Command: $cmd\n")
    run(cmd)

    fixest_df = CSV.read(results_path, DataFrame)
    fixest_times = Dict{String, Float64}()
    for row in eachrow(fixest_df)
        fixest_times[row.scenario] = row.time_seconds
    end
    return fixest_times
end

# ------------------------------------------------------------------------------
# Dataset Generation
# ------------------------------------------------------------------------------

function generate_dataset1(; seed::Int = 42)
    Random.seed!(seed)
    N = 10_000_000
    K = 100
    id1 = rand(1:div(N, K), N)
    id2 = rand(1:K, N)
    x1 = 5 .* cos.(id1) .+ 5 .* sin.(id2) .+ randn(N)
    x2 = cos.(id1) .+ sin.(id2) .+ randn(N)
    y = 3 .* x1 .+ 5 .* x2 .+ cos.(id1) .+ cos.(id2) .^ 2 .+ randn(N)
    return DataFrame(id1 = id1, id2 = id2, x1 = x1, x2 = x2, y = y), N
end

function generate_dataset2(; seed::Int = 42)
    Random.seed!(seed)
    N = 800_000
    M = 40_000  # workers
    O = 5_000   # firms
    id1 = rand(1:M, N)
    id2 = [rand(max(1, div(x, 8) - 10):min(O, div(x, 8) + 10)) for x in id1]
    x1 = 5 .* cos.(id1) .+ 5 .* sin.(id2) .+ randn(N)
    x2 = cos.(id1) .+ sin.(id2) .+ randn(N)
    y = 3 .* x1 .+ 5 .* x2 .+ cos.(id1) .+ cos.(id2) .^ 2 .+ randn(N)
    return DataFrame(id1 = id1, id2 = id2, x1 = x1, x2 = x2, y = y), N
end

function generate_dataset3(; seed::Int = 42)
    Random.seed!(seed)
    n = 10_000_000
    nb_dum = [div(n, 20), floor(Int, sqrt(n)), floor(Int, n^0.33)]
    id1 = rand(1:nb_dum[1], n)
    id2 = rand(1:nb_dum[2], n)
    id3 = rand(1:nb_dum[3], n)
    X1 = rand(n)
    ln_y = 3 .* X1 .+ rand(n)
    return DataFrame(X1 = X1, ln_y = ln_y, id1 = id1, id2 = id2, id3 = id3), n
end

# ------------------------------------------------------------------------------
# Benchmark Scenarios (Julia only)
# ------------------------------------------------------------------------------

function run_julia_benchmarks(fixest_times::Dict{String, Float64};
        method::Symbol = :cpu,
        nthreads::Int = 1)
    results = BenchmarkResult[]

    # Dataset 1: 10M observations
    println("\n" * "=" ^ 70)
    println("Julia Benchmarks: Dataset 1 (10M observations)")
    println("=" ^ 70)

    df1, N1 = generate_dataset1()

    # Scenario 1: Simple OLS
    scenario = "OLS: y ~ x1 + x2"
    println("\n  $scenario")

    formula_fem = FEM.@formula(y ~ x1 + x2)
    formula_reg = REG.@formula(y ~ x1 + x2)

    FEM.reg(df1, formula_fem)
    t_fem = @benchmark FEM.reg($df1, $formula_fem) samples=5 evals=1
    fem_t = median(t_fem).time / 1e9
    println("    FEM: $(fmt(fem_t))s")

    REG.ols(df1, formula_reg; method = method, nthreads = nthreads)
    t_reg = @benchmark REG.ols($df1, $formula_reg; method = $method, nthreads = $nthreads) samples=5 evals=1
    reg_t = median(t_reg).time / 1e9
    println("    Regress: $(fmt(reg_t))s")

    fix_t = get(fixest_times, scenario, NaN)
    println("    fixest: $(fmt(fix_t))s")
    push!(results, BenchmarkResult(scenario, N1, fem_t, reg_t, fix_t))

    # Scenario 2: OLS + cluster
    scenario = "OLS + cluster(id2)"
    println("\n  $scenario")

    FEM.reg(df1, formula_fem, FEM.Vcov.cluster(:id2))
    t_fem = @benchmark FEM.reg($df1, $formula_fem, FEM.Vcov.cluster(:id2)) samples=5 evals=1
    fem_t = median(t_fem).time / 1e9
    println("    FEM: $(fmt(fem_t))s")

    cluster_vec = df1.id2
    m = REG.ols(df1, formula_reg; method = method, nthreads = nthreads)
    vcov(CR1(cluster_vec), m)
    t_reg = @benchmark begin
        m=REG.ols($df1, $formula_reg; method = $method, nthreads = $nthreads)
        vcov(CR1($cluster_vec), m)
    end samples=5 evals=1
    reg_t = median(t_reg).time / 1e9
    println("    Regress: $(fmt(reg_t))s")

    fix_t = get(fixest_times, scenario, NaN)
    println("    fixest: $(fmt(fix_t))s")
    push!(results, BenchmarkResult(scenario, N1, fem_t, reg_t, fix_t))

    # Scenario 3: One FE
    scenario = "FE: y ~ x1 + x2 | id1"
    println("\n  $scenario")

    formula_fem_fe = FEM.@formula(y ~ x1 + x2 + FEM.fe(id1))
    formula_reg_fe = REG.@formula(y ~ x1 + x2 + REG.fe(id1))

    FEM.reg(df1, formula_fem_fe)
    t_fem = @benchmark FEM.reg($df1, $formula_fem_fe) samples=5 evals=1
    fem_t = median(t_fem).time / 1e9
    println("    FEM: $(fmt(fem_t))s")

    REG.ols(df1, formula_reg_fe; method = method, nthreads = nthreads)
    t_reg = @benchmark REG.ols($df1, $formula_reg_fe; method = $method, nthreads = $nthreads) samples=5 evals=1
    reg_t = median(t_reg).time / 1e9
    println("    Regress: $(fmt(reg_t))s")

    fix_t = get(fixest_times, scenario, NaN)
    println("    fixest: $(fmt(fix_t))s")
    push!(results, BenchmarkResult(scenario, N1, fem_t, reg_t, fix_t))

    # Scenario 4: One FE + cluster
    scenario = "FE + cluster: | id1"
    println("\n  $scenario")

    FEM.reg(df1, formula_fem_fe, FEM.Vcov.cluster(:id1))
    t_fem = @benchmark FEM.reg($df1, $formula_fem_fe, FEM.Vcov.cluster(:id1)) samples=5 evals=1
    fem_t = median(t_fem).time / 1e9
    println("    FEM: $(fmt(fem_t))s")

    m = REG.ols(
        df1, formula_reg_fe, save_cluster = :id1; method = method, nthreads = nthreads)
    vcov(CR1(:id1), m)
    t_reg = @benchmark begin
        m=REG.ols($df1, $formula_reg_fe, save_cluster = :id1;
            method = $method, nthreads = $nthreads)
        vcov(CR1(:id1), m)
    end samples=5 evals=1
    reg_t = median(t_reg).time / 1e9
    println("    Regress: $(fmt(reg_t))s")

    fix_t = get(fixest_times, scenario, NaN)
    println("    fixest: $(fmt(fix_t))s")
    push!(results, BenchmarkResult(scenario, N1, fem_t, reg_t, fix_t))

    # Scenario 5: Two FEs
    scenario = "FE: y ~ x1 + x2 | id1 + id2"
    println("\n  $scenario")

    formula_fem_2fe = FEM.@formula(y ~ x1 + x2 + FEM.fe(id1) + FEM.fe(id2))
    formula_reg_2fe = REG.@formula(y ~ x1 + x2 + REG.fe(id1) + REG.fe(id2))

    FEM.reg(df1, formula_fem_2fe)
    t_fem = @benchmark FEM.reg($df1, $formula_fem_2fe) samples=5 evals=1
    fem_t = median(t_fem).time / 1e9
    println("    FEM: $(fmt(fem_t))s")

    REG.ols(df1, formula_reg_2fe; method = method, nthreads = nthreads)
    t_reg = @benchmark REG.ols($df1, $formula_reg_2fe; method = $method, nthreads = $nthreads) samples=5 evals=1
    reg_t = median(t_reg).time / 1e9
    println("    Regress: $(fmt(reg_t))s")

    fix_t = get(fixest_times, scenario, NaN)
    println("    fixest: $(fmt(fix_t))s")
    push!(results, BenchmarkResult(scenario, N1, fem_t, reg_t, fix_t))

    df1 = nothing
    GC.gc()

    # Dataset 2: Worker-Firm (800K)
    println("\n" * "=" ^ 70)
    println("Julia Benchmarks: Dataset 2 - Worker-Firm (800K observations)")
    println("=" ^ 70)

    df2, N2 = generate_dataset2()

    scenario = "Worker-Firm FE"
    println("\n  $scenario")

    formula_fem_2fe = FEM.@formula(y ~ x1 + x2 + FEM.fe(id1) + FEM.fe(id2))
    formula_reg_2fe = REG.@formula(y ~ x1 + x2 + REG.fe(id1) + REG.fe(id2))

    FEM.reg(df2, formula_fem_2fe)
    t_fem = @benchmark FEM.reg($df2, $formula_fem_2fe) samples=5 evals=1
    fem_t = median(t_fem).time / 1e9
    println("    FEM: $(fmt(fem_t))s")

    REG.ols(df2, formula_reg_2fe; method = method, nthreads = nthreads)
    t_reg = @benchmark REG.ols($df2, $formula_reg_2fe; method = $method, nthreads = $nthreads) samples=5 evals=1
    reg_t = median(t_reg).time / 1e9
    println("    Regress: $(fmt(reg_t))s")

    fix_t = get(fixest_times, scenario, NaN)
    println("    fixest: $(fmt(fix_t))s")
    push!(results, BenchmarkResult(scenario, N2, fem_t, reg_t, fix_t))

    df2 = nothing
    GC.gc()

    # Dataset 3: fixest-style (10M, 3 FEs)
    println("\n" * "=" ^ 70)
    println("Julia Benchmarks: Dataset 3 - fixest-style (10M observations)")
    println("=" ^ 70)

    df3, N3 = generate_dataset3()

    # 1 FE + cluster
    scenario = "1 FE + cluster"
    println("\n  $scenario")

    formula_fem_1fe = FEM.@formula(ln_y ~ X1 + FEM.fe(id1))
    formula_reg_1fe = REG.@formula(ln_y ~ X1 + REG.fe(id1))

    FEM.reg(df3, formula_fem_1fe, FEM.Vcov.cluster(:id1))
    t_fem = @benchmark FEM.reg($df3, $formula_fem_1fe, FEM.Vcov.cluster(:id1)) samples=5 evals=1
    fem_t = median(t_fem).time / 1e9
    println("    FEM: $(fmt(fem_t))s")

    m = REG.ols(
        df3, formula_reg_1fe, save_cluster = :id1; method = method, nthreads = nthreads)
    vcov(CR1(:id1), m)
    t_reg = @benchmark begin
        m=REG.ols($df3, $formula_reg_1fe, save_cluster = :id1;
            method = $method, nthreads = $nthreads)
        vcov(CR1(:id1), m)
    end samples=5 evals=1
    reg_t = median(t_reg).time / 1e9
    println("    Regress: $(fmt(reg_t))s")

    fix_t = get(fixest_times, scenario, NaN)
    println("    fixest: $(fmt(fix_t))s")
    push!(results, BenchmarkResult(scenario, N3, fem_t, reg_t, fix_t))

    # 2 FE + cluster
    scenario = "2 FE + cluster"
    println("\n  $scenario")

    formula_fem_2fe = FEM.@formula(ln_y ~ X1 + FEM.fe(id1) + FEM.fe(id2))
    formula_reg_2fe = REG.@formula(ln_y ~ X1 + REG.fe(id1) + REG.fe(id2))

    FEM.reg(df3, formula_fem_2fe, FEM.Vcov.cluster(:id1))
    t_fem = @benchmark FEM.reg($df3, $formula_fem_2fe, FEM.Vcov.cluster(:id1)) samples=5 evals=1
    fem_t = median(t_fem).time / 1e9
    println("    FEM: $(fmt(fem_t))s")

    m = REG.ols(
        df3, formula_reg_2fe, save_cluster = :id1; method = method, nthreads = nthreads)
    vcov(CR1(:id1), m)
    t_reg = @benchmark begin
        m=REG.ols($df3, $formula_reg_2fe, save_cluster = :id1;
            method = $method, nthreads = $nthreads)
        vcov(CR1(:id1), m)
    end samples=5 evals=1
    reg_t = median(t_reg).time / 1e9
    println("    Regress: $(fmt(reg_t))s")

    fix_t = get(fixest_times, scenario, NaN)
    println("    fixest: $(fmt(fix_t))s")
    push!(results, BenchmarkResult(scenario, N3, fem_t, reg_t, fix_t))

    # 3 FE + cluster
    scenario = "3 FE + cluster"
    println("\n  $scenario")

    formula_fem_3fe = FEM.@formula(ln_y ~ X1 + FEM.fe(id1) + FEM.fe(id2) + FEM.fe(id3))
    formula_reg_3fe = REG.@formula(ln_y ~ X1 + REG.fe(id1) + REG.fe(id2) + REG.fe(id3))

    FEM.reg(df3, formula_fem_3fe, FEM.Vcov.cluster(:id1))
    t_fem = @benchmark FEM.reg($df3, $formula_fem_3fe, FEM.Vcov.cluster(:id1)) samples=5 evals=1
    fem_t = median(t_fem).time / 1e9
    println("    FEM: $(fmt(fem_t))s")

    m = REG.ols(
        df3, formula_reg_3fe, save_cluster = :id1; method = method, nthreads = nthreads)
    vcov(CR1(:id1), m)
    t_reg = @benchmark begin
        m=REG.ols($df3, $formula_reg_3fe, save_cluster = :id1;
            method = $method, nthreads = $nthreads)
        vcov(CR1(:id1), m)
    end samples=5 evals=1
    reg_t = median(t_reg).time / 1e9
    println("    Regress: $(fmt(reg_t))s")

    fix_t = get(fixest_times, scenario, NaN)
    println("    fixest: $(fmt(fix_t))s")
    push!(results, BenchmarkResult(scenario, N3, fem_t, reg_t, fix_t))

    return results
end
