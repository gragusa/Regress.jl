#!/usr/bin/env julia
#
# Benchmark 2: MKL BLAS Benchmark
#
# Uses Intel MKL for BLAS operations (requires MKL.jl to be loaded BEFORE LinearAlgebra)
#
# Prerequisites:
#   1. Add MKL.jl to the project: ] add MKL
#   2. MKL must be loaded before any LinearAlgebra operations
#
# Run: julia --project=.. -t auto benchmark_2_mkl.jl
#
# Note: This script attempts to use MKL if available, otherwise falls back to default BLAS
#

# Try to load MKL before anything else
const HAS_MKL = try
    using MKL
    true
catch e
    @warn "MKL.jl not available. Install with: ] add MKL" exception=e
    false
end

include("benchmark_common.jl")

using LinearAlgebra

const BENCHMARK_NAME = "MKL BLAS Benchmark"
const OUTPUT_FILE = "benchmark_results_2_mkl.md"

function main()
    nthreads = Threads.nthreads()

    # Set BLAS threads
    BLAS.set_num_threads(nthreads)

    # Get BLAS info
    blas_config = BLAS.get_config()
    blas_lib = blas_config.loaded_libs[1].libname
    is_mkl = occursin("mkl", lowercase(blas_lib))

    println("=" ^ 70)
    println(BENCHMARK_NAME)
    println("=" ^ 70)
    println()
    println("Configuration:")
    println("  Julia threads: $nthreads")
    println("  BLAS library: $blas_lib")
    println("  BLAS threads: $(BLAS.get_num_threads())")
    println("  MKL detected: $is_mkl")
    println("  R fixest threads: $nthreads")
    println()

    if !is_mkl
        @warn """
        MKL is NOT the active BLAS library.
        To use MKL:
        1. Add MKL.jl: ] add MKL
        2. Ensure MKL is loaded before LinearAlgebra (this script does this)
        3. You may need to restart Julia after installing MKL

        Current BLAS: $blas_lib
        """
    end

    # Run fixest with matching threads
    fixest_times = run_fixest_benchmarks(nthreads; script_dir = @__DIR__)

    # Run Julia benchmarks
    results = run_julia_benchmarks(fixest_times; method = :cpu, nthreads = nthreads)

    # Generate report
    config = Dict{String, Any}(
        "Benchmark Type" => is_mkl ? "MKL BLAS" : "Default BLAS (MKL not available)",
        "Julia Version" => string(VERSION),
        "Julia Threads" => nthreads,
        "BLAS Library" => blas_lib,
        "BLAS Threads" => BLAS.get_num_threads(),
        "MKL Active" => is_mkl,
        "R fixest Threads" => nthreads,
        "Description" => is_mkl ?
                         "Using Intel MKL for optimized BLAS operations" :
                         "MKL not available - using default BLAS"
    )

    md_content = generate_markdown_header(BENCHMARK_NAME, config)
    md_content *= generate_standard_table(results)

    # Add MKL-specific notes
    md_content *= """
## Notes on MKL

$(is_mkl ? "**MKL is active** - BLAS operations are using Intel MKL." : "**MKL is NOT active** - Using default BLAS library.")

### Installing MKL

To use MKL in Julia:

```julia
using Pkg
Pkg.add("MKL")
```

Then ensure `using MKL` appears before `using LinearAlgebra` in your code.

### MKL Benefits

- Optimized for Intel processors
- Efficient multi-threaded BLAS operations
- Can significantly improve performance for matrix operations
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
