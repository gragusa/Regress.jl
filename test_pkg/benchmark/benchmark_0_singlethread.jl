#!/usr/bin/env julia
#
# Benchmark 0: Single-Threaded Comparison
#
# All packages run single-threaded:
# - Julia: 1 thread, BLAS single-threaded
# - R fixest: 1 thread (no OpenMP)
#
# Run: julia --project=.. -t 1 benchmark_0_singlethread.jl
#

include("benchmark_common.jl")

using LinearAlgebra
BLAS.set_num_threads(1)

const BENCHMARK_NAME = "Single-Threaded Benchmark"
const OUTPUT_FILE = "benchmark_results_0_singlethread.md"

function main()
    println("=" ^ 70)
    println(BENCHMARK_NAME)
    println("=" ^ 70)
    println()
    println("Configuration:")
    println("  Julia threads: $(Threads.nthreads())")
    println("  BLAS threads: $(BLAS.get_num_threads())")
    println("  R fixest threads: 1")
    println()

    if Threads.nthreads() != 1
        @warn "This benchmark should be run with -t 1 for accurate single-threaded results"
    end

    # Run fixest with 1 thread
    fixest_times = run_fixest_benchmarks(1; script_dir = @__DIR__)

    # Run Julia benchmarks single-threaded
    results = run_julia_benchmarks(fixest_times; method = :cpu, nthreads = 1)

    # Generate report
    config = Dict{String, Any}(
        "Benchmark Type" => "Single-Threaded (Baseline)",
        "Julia Version" => string(VERSION),
        "Julia Threads" => Threads.nthreads(),
        "BLAS Library" => BLAS.get_config().loaded_libs[1].libname,
        "BLAS Threads" => BLAS.get_num_threads(),
        "R fixest Threads" => 1,
        "Description" => "All packages running single-threaded for baseline comparison"
    )

    md_content = generate_markdown_header(BENCHMARK_NAME, config)
    md_content *= generate_standard_table(results)

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
