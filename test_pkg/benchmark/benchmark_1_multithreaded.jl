#!/usr/bin/env julia
#
# Benchmark 1: Multi-Threaded Comparison
#
# All packages use the same number of threads:
# - Julia: N threads (from -t argument)
# - BLAS: N threads
# - R fixest: N threads (via OpenMP)
#
# Run: julia --project=.. -t auto benchmark_1_multithreaded.jl
#   or: julia --project=.. -t 8 benchmark_1_multithreaded.jl
#

include("benchmark_common.jl")

using LinearAlgebra

const BENCHMARK_NAME = "Multi-Threaded Benchmark"
const OUTPUT_FILE = "benchmark_results_1_multithreaded.md"

function main()
    nthreads = Threads.nthreads()

    # Set BLAS threads to match Julia threads
    BLAS.set_num_threads(nthreads)

    println("=" ^ 70)
    println(BENCHMARK_NAME)
    println("=" ^ 70)
    println()
    println("Configuration:")
    println("  Julia threads: $nthreads")
    println("  BLAS threads: $(BLAS.get_num_threads())")
    println("  R fixest threads: $nthreads")
    println()

    if nthreads == 1
        @warn "Running with only 1 thread. Use -t auto or -t N for multi-threaded benchmark"
    end

    # Run fixest with same number of threads
    fixest_times = run_fixest_benchmarks(nthreads; script_dir=@__DIR__)

    # Run Julia benchmarks with matching threads
    results = run_julia_benchmarks(fixest_times; method=:cpu, nthreads=nthreads)

    # Generate report
    config = Dict{String, Any}(
        "Benchmark Type" => "Multi-Threaded (CPU)",
        "Julia Version" => string(VERSION),
        "Julia Threads" => nthreads,
        "BLAS Library" => BLAS.get_config().loaded_libs[1].libname,
        "BLAS Threads" => BLAS.get_num_threads(),
        "R fixest Threads" => nthreads,
        "Description" => "All packages using $nthreads threads for parallel comparison"
    )

    md_content = generate_markdown_header(BENCHMARK_NAME, config)
    md_content *= generate_standard_table(results)

    # Add note about OpenMP
    md_content *= """
## Notes

- R fixest requires OpenMP support for multi-threading. If OpenMP is not available,
  fixest will run single-threaded regardless of the thread setting.
- Julia's FixedEffectModels.jl and Regress.jl use Julia's native threading.
- BLAS operations use the configured BLAS library's threading.
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
