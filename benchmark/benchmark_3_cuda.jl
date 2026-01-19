#!/usr/bin/env julia
#
# Benchmark 3: CUDA vs Threaded fixest
#
# Compares:
# - Regress.jl with CUDA acceleration
# - R fixest with OpenMP threading (maximum threads)
#
# Prerequisites:
#   - NVIDIA GPU with CUDA support
#   - CUDA.jl installed and functional
#   - R fixest compiled with OpenMP support
#
# Run: julia --project=.. -t auto benchmark_3_cuda.jl
#

include("benchmark_common.jl")

using LinearAlgebra

# Try to load CUDA
const HAS_CUDA = try
    using CUDA
    CUDA.functional()
catch e
    false
end

const BENCHMARK_NAME = "CUDA vs Threaded fixest Benchmark"
const OUTPUT_FILE = "benchmark_results_3_cuda.md"

function run_cuda_benchmarks(fixest_times::Dict{String, Float64})
    results = CudaBenchmarkResult[]

    if !HAS_CUDA
        @error "CUDA is not available. Cannot run CUDA benchmarks."
        return results
    end

    # Print GPU info
    println("\nCUDA Device: $(CUDA.name(CUDA.device()))")
    println("CUDA Memory: $(round(CUDA.total_memory() / 1024^3, digits=2)) GB")

    # Dataset 1: 10M observations
    println("\n" * "=" ^ 70)
    println("CUDA Benchmarks: Dataset 1 (10M observations)")
    println("=" ^ 70)

    df1, N1 = generate_dataset1()

    # Scenario 1: Simple OLS
    scenario = "OLS: y ~ x1 + x2"
    println("\n  $scenario")

    formula_reg = REG.@formula(y ~ x1 + x2)

    REG.ols(df1, formula_reg; method=:CUDA)
    CUDA.synchronize()
    t_reg = @benchmark begin
        REG.ols($df1, $formula_reg; method=:CUDA)
        CUDA.synchronize()
    end samples=5 evals=1
    cuda_t = median(t_reg).time / 1e9
    println("    Regress CUDA: $(fmt(cuda_t))s")

    fix_t = get(fixest_times, scenario, NaN)
    println("    fixest threaded: $(fmt(fix_t))s")
    push!(results, CudaBenchmarkResult(scenario, N1, cuda_t, fix_t))

    # Scenario 2: OLS + cluster (CUDA computes model, vcov on CPU)
    scenario = "OLS + cluster(id2)"
    println("\n  $scenario")

    cluster_vec = df1.id2
    m = REG.ols(df1, formula_reg; method=:CUDA)
    CUDA.synchronize()
    vcov(CR1(cluster_vec), m)
    t_reg = @benchmark begin
        m = REG.ols($df1, $formula_reg; method=:CUDA)
        CUDA.synchronize()
        vcov(CR1($cluster_vec), m)
    end samples=5 evals=1
    cuda_t = median(t_reg).time / 1e9
    println("    Regress CUDA: $(fmt(cuda_t))s")

    fix_t = get(fixest_times, scenario, NaN)
    println("    fixest threaded: $(fmt(fix_t))s")
    push!(results, CudaBenchmarkResult(scenario, N1, cuda_t, fix_t))

    # Scenario 3: One FE
    scenario = "FE: y ~ x1 + x2 | id1"
    println("\n  $scenario")

    formula_reg_fe = REG.@formula(y ~ x1 + x2 + REG.fe(id1))

    REG.ols(df1, formula_reg_fe; method=:CUDA)
    CUDA.synchronize()
    t_reg = @benchmark begin
        REG.ols($df1, $formula_reg_fe; method=:CUDA)
        CUDA.synchronize()
    end samples=5 evals=1
    cuda_t = median(t_reg).time / 1e9
    println("    Regress CUDA: $(fmt(cuda_t))s")

    fix_t = get(fixest_times, scenario, NaN)
    println("    fixest threaded: $(fmt(fix_t))s")
    push!(results, CudaBenchmarkResult(scenario, N1, cuda_t, fix_t))

    # Scenario 4: One FE + cluster
    scenario = "FE + cluster: | id1"
    println("\n  $scenario")

    m = REG.ols(df1, formula_reg_fe, save_cluster=:id1; method=:CUDA)
    CUDA.synchronize()
    vcov(CR1(:id1), m)
    t_reg = @benchmark begin
        m = REG.ols($df1, $formula_reg_fe, save_cluster=:id1; method=:CUDA)
        CUDA.synchronize()
        vcov(CR1(:id1), m)
    end samples=5 evals=1
    cuda_t = median(t_reg).time / 1e9
    println("    Regress CUDA: $(fmt(cuda_t))s")

    fix_t = get(fixest_times, scenario, NaN)
    println("    fixest threaded: $(fmt(fix_t))s")
    push!(results, CudaBenchmarkResult(scenario, N1, cuda_t, fix_t))

    # Scenario 5: Two FEs
    scenario = "FE: y ~ x1 + x2 | id1 + id2"
    println("\n  $scenario")

    formula_reg_2fe = REG.@formula(y ~ x1 + x2 + REG.fe(id1) + REG.fe(id2))

    REG.ols(df1, formula_reg_2fe; method=:CUDA)
    CUDA.synchronize()
    t_reg = @benchmark begin
        REG.ols($df1, $formula_reg_2fe; method=:CUDA)
        CUDA.synchronize()
    end samples=5 evals=1
    cuda_t = median(t_reg).time / 1e9
    println("    Regress CUDA: $(fmt(cuda_t))s")

    fix_t = get(fixest_times, scenario, NaN)
    println("    fixest threaded: $(fmt(fix_t))s")
    push!(results, CudaBenchmarkResult(scenario, N1, cuda_t, fix_t))

    df1 = nothing
    GC.gc()
    CUDA.reclaim()

    # Dataset 2: Worker-Firm
    println("\n" * "=" ^ 70)
    println("CUDA Benchmarks: Dataset 2 - Worker-Firm (800K observations)")
    println("=" ^ 70)

    df2, N2 = generate_dataset2()

    scenario = "Worker-Firm FE"
    println("\n  $scenario")

    formula_reg_2fe = REG.@formula(y ~ x1 + x2 + REG.fe(id1) + REG.fe(id2))

    REG.ols(df2, formula_reg_2fe; method=:CUDA)
    CUDA.synchronize()
    t_reg = @benchmark begin
        REG.ols($df2, $formula_reg_2fe; method=:CUDA)
        CUDA.synchronize()
    end samples=5 evals=1
    cuda_t = median(t_reg).time / 1e9
    println("    Regress CUDA: $(fmt(cuda_t))s")

    fix_t = get(fixest_times, scenario, NaN)
    println("    fixest threaded: $(fmt(fix_t))s")
    push!(results, CudaBenchmarkResult(scenario, N2, cuda_t, fix_t))

    df2 = nothing
    GC.gc()
    CUDA.reclaim()

    # Dataset 3: fixest-style
    println("\n" * "=" ^ 70)
    println("CUDA Benchmarks: Dataset 3 - fixest-style (10M observations)")
    println("=" ^ 70)

    df3, N3 = generate_dataset3()

    # 1 FE + cluster
    scenario = "1 FE + cluster"
    println("\n  $scenario")

    formula_reg_1fe = REG.@formula(ln_y ~ X1 + REG.fe(id1))

    m = REG.ols(df3, formula_reg_1fe, save_cluster=:id1; method=:CUDA)
    CUDA.synchronize()
    vcov(CR1(:id1), m)
    t_reg = @benchmark begin
        m = REG.ols($df3, $formula_reg_1fe, save_cluster=:id1; method=:CUDA)
        CUDA.synchronize()
        vcov(CR1(:id1), m)
    end samples=5 evals=1
    cuda_t = median(t_reg).time / 1e9
    println("    Regress CUDA: $(fmt(cuda_t))s")

    fix_t = get(fixest_times, scenario, NaN)
    println("    fixest threaded: $(fmt(fix_t))s")
    push!(results, CudaBenchmarkResult(scenario, N3, cuda_t, fix_t))

    # 2 FE + cluster
    scenario = "2 FE + cluster"
    println("\n  $scenario")

    formula_reg_2fe = REG.@formula(ln_y ~ X1 + REG.fe(id1) + REG.fe(id2))

    m = REG.ols(df3, formula_reg_2fe, save_cluster=:id1; method=:CUDA)
    CUDA.synchronize()
    vcov(CR1(:id1), m)
    t_reg = @benchmark begin
        m = REG.ols($df3, $formula_reg_2fe, save_cluster=:id1; method=:CUDA)
        CUDA.synchronize()
        vcov(CR1(:id1), m)
    end samples=5 evals=1
    cuda_t = median(t_reg).time / 1e9
    println("    Regress CUDA: $(fmt(cuda_t))s")

    fix_t = get(fixest_times, scenario, NaN)
    println("    fixest threaded: $(fmt(fix_t))s")
    push!(results, CudaBenchmarkResult(scenario, N3, cuda_t, fix_t))

    # 3 FE + cluster
    scenario = "3 FE + cluster"
    println("\n  $scenario")

    formula_reg_3fe = REG.@formula(ln_y ~ X1 + REG.fe(id1) + REG.fe(id2) + REG.fe(id3))

    m = REG.ols(df3, formula_reg_3fe, save_cluster=:id1; method=:CUDA)
    CUDA.synchronize()
    vcov(CR1(:id1), m)
    t_reg = @benchmark begin
        m = REG.ols($df3, $formula_reg_3fe, save_cluster=:id1; method=:CUDA)
        CUDA.synchronize()
        vcov(CR1(:id1), m)
    end samples=5 evals=1
    cuda_t = median(t_reg).time / 1e9
    println("    Regress CUDA: $(fmt(cuda_t))s")

    fix_t = get(fixest_times, scenario, NaN)
    println("    fixest threaded: $(fmt(fix_t))s")
    push!(results, CudaBenchmarkResult(scenario, N3, cuda_t, fix_t))

    return results
end

function main()
    nthreads = Threads.nthreads()
    BLAS.set_num_threads(nthreads)

    println("=" ^ 70)
    println(BENCHMARK_NAME)
    println("=" ^ 70)
    println()
    println("Configuration:")
    println("  Julia threads: $nthreads")
    println("  CUDA available: $HAS_CUDA")
    if HAS_CUDA
        println("  CUDA device: $(CUDA.name(CUDA.device()))")
        println("  CUDA memory: $(round(CUDA.total_memory() / 1024^3, digits=2)) GB")
    end
    println("  R fixest threads: $nthreads (for comparison)")
    println()

    if !HAS_CUDA
        @error """
        CUDA is not available or not functional.

        To use CUDA:
        1. Ensure you have an NVIDIA GPU
        2. Install CUDA toolkit
        3. Add CUDA.jl: ] add CUDA

        Exiting benchmark.
        """
        return
    end

    # Run fixest with max threads for fair comparison
    fixest_times = run_fixest_benchmarks(nthreads; script_dir=@__DIR__)

    # Run CUDA benchmarks
    results = run_cuda_benchmarks(fixest_times)

    if isempty(results)
        @error "No CUDA benchmark results. Exiting."
        return
    end

    # Generate report
    config = Dict{String, Any}(
        "Benchmark Type" => "CUDA GPU vs CPU (threaded fixest)",
        "Julia Version" => string(VERSION),
        "CUDA Device" => CUDA.name(CUDA.device()),
        "CUDA Memory" => "$(round(CUDA.total_memory() / 1024^3, digits=2)) GB",
        "CUDA Driver" => string(CUDA.version()),
        "R fixest Threads" => nthreads,
        "Description" => "Comparing Regress.jl CUDA acceleration against multi-threaded R fixest"
    )

    md_content = generate_markdown_header(BENCHMARK_NAME, config)
    md_content *= generate_cuda_table(results)

    md_content *= """
## Notes on CUDA Benchmarks

### What's being measured

- **Regress CUDA**: Full regression including data transfer to GPU, computation, and result transfer back
- **fixest threaded**: R fixest using OpenMP with $nthreads threads

### GPU Memory Considerations

- Large datasets may exceed GPU memory
- Fixed effects demeaning is performed on GPU when using `method=:CUDA`
- Variance-covariance computation (for clustered SE) is done on CPU after model fitting

### When CUDA Helps Most

- Very large datasets (millions of observations)
- Models with many fixed effects (demeaning is parallelized on GPU)
- Simple OLS on large data (matrix operations are GPU-accelerated)

### When CPU May Be Faster

- Small datasets (GPU transfer overhead dominates)
- Models with very few observations per fixed effect group
- When GPU memory is insufficient and paging occurs
"""

    open(OUTPUT_FILE, "w") do f
        write(f, md_content)
    end

    println("\n" * "=" ^ 70)
    println("Results saved to $OUTPUT_FILE")
    println("=" ^ 70)
    println()
    println(md_content)
end

main()
