# Benchmark Suite

This directory contains benchmark scripts comparing **FixedEffectModels.jl**, **Regress.jl**, and **fixest (R)** across different hardware configurations.

## Benchmark Configurations

| Benchmark | File | Description |
|-----------|------|-------------|
| **0. Single-Threaded** | `benchmark_0_singlethread.jl` | Baseline: all packages single-threaded |
| **1. Multi-Threaded** | `benchmark_1_multithreaded.jl` | All packages use same number of threads |
| **2. MKL BLAS** | `benchmark_2_mkl.jl` | Uses Intel MKL for BLAS operations |
| **3. CUDA vs fixest** | `benchmark_3_cuda.jl` | Regress.jl CUDA vs threaded fixest |

## Running the Benchmarks

### Prerequisites

```bash
cd test_pkg
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Benchmark 0: Single-Threaded (Baseline)

```bash
julia --project=.. -t 1 benchmark_0_singlethread.jl
```

Output: `benchmark_results_0_singlethread.md`

### Benchmark 1: Multi-Threaded

```bash
# Use all available threads
julia --project=.. -t auto benchmark_1_multithreaded.jl

# Or specify thread count
julia --project=.. -t 8 benchmark_1_multithreaded.jl
```

Output: `benchmark_results_1_multithreaded.md`

### Benchmark 2: MKL BLAS

First, install MKL:
```julia
using Pkg
Pkg.add("MKL")
```

Then run:
```bash
julia --project=.. -t auto benchmark_2_mkl.jl
```

Output: `benchmark_results_2_mkl.md`

### Benchmark 3: CUDA vs Threaded fixest

Requires NVIDIA GPU with CUDA support:
```julia
using Pkg
Pkg.add("CUDA")
```

Then run:
```bash
julia --project=.. -t auto benchmark_3_cuda.jl
```

Output: `benchmark_results_3_cuda.md`

## Output Files

Each benchmark produces a markdown file with:

1. **Configuration details**: Julia version, thread counts, BLAS library, etc.
2. **Results table**: Timing comparisons across all scenarios
3. **Notes**: Benchmark-specific observations and recommendations

## Scenarios Tested

All benchmarks run the same set of scenarios:

| Dataset | Scenario | N | Description |
|---------|----------|---|-------------|
| 1 | OLS: y ~ x1 + x2 | 10M | Simple OLS |
| 1 | OLS + cluster(id2) | 10M | OLS with cluster-robust SE |
| 1 | FE: y ~ x1 + x2 \| id1 | 10M | One fixed effect |
| 1 | FE + cluster: \| id1 | 10M | One FE + cluster SE |
| 1 | FE: y ~ x1 + x2 \| id1 + id2 | 10M | Two fixed effects |
| 2 | Worker-Firm FE | 800K | Complex FE structure |
| 3 | 1 FE + cluster | 10M | fixest-style benchmark |
| 3 | 2 FE + cluster | 10M | fixest-style benchmark |
| 3 | 3 FE + cluster | 10M | fixest-style benchmark |

## R fixest Threading

R fixest requires OpenMP for multi-threading. On macOS, the default R installation may not have OpenMP support. Check with:

```r
library(fixest)
setFixest_nthreads(4)  # Will warn if OpenMP not available
```

To get OpenMP support on macOS:
- Install R via Homebrew: `brew install r`
- Or use R from CRAN with OpenMP-enabled compilers
