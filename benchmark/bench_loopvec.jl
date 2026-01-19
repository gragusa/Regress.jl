#=
Benchmark: LoopVectorization impact on Regress.jl

Compares OLS performance with and without LoopVectorization extension.
Tests 10M observations with 4 regressors.

Run from test_pkg directory (which has LoopVectorization as a dependency):
    cd test_pkg && julia --project=. ../benchmark/bench_loopvec.jl
=#

using BenchmarkTools
using DataFrames
using Printf
using Random
using Statistics

println("=" ^ 70)
println("Regress.jl LoopVectorization Benchmark")
println("=" ^ 70)
println()

# Setup
const N = 10_000_000
const K = 4

println("Generating test data: N = $N, K = $K")
Random.seed!(42)

df = DataFrame(
    y = randn(N),
    x1 = randn(N),
    x2 = randn(N),
    x3 = randn(N),
    x4 = randn(N)
)
df.y .= 1.0 .+ 2.0 .* df.x1 .+ 0.5 .* df.x2 .- 1.0 .* df.x3 .+ 0.3 .* df.x4 .+
        0.1 .* randn(N)
println("Data generated.")
println()

y_vec = Float64.(df.y)

#------------------------------------------------------------------------------
# Benchmark WITHOUT LoopVectorization first
#------------------------------------------------------------------------------
println("-" ^ 70)
println("Loading Regress WITHOUT LoopVectorization")
println("-" ^ 70)

using Regress
using StatsModels
using StatsBase: uweights
using StatsAPI: fitted

lv_ext = Base.get_extension(Regress, :RegressLVExt)
println("RegressLVExt loaded: $(lv_ext !== nothing)")

formula = @formula(y ~ x1 + x2 + x3 + x4)

# Warmup
_ = ols(df, formula)
GC.gc()

println("\n1. Full OLS fit (@simd):")
bench_full_simd = @benchmark ols($df, $formula) samples=5 evals=1
display(bench_full_simd)
time_full_simd = median(bench_full_simd).time / 1e9

m = ols(df, formula)
mu_vec = copy(fitted(m))

println("\n2. compute_rss (@simd):")
bench_rss_simd = @benchmark Regress.compute_rss($y_vec, $mu_vec) samples=50
display(bench_rss_simd)
time_rss_simd = median(bench_rss_simd).time / 1e6

wts = uweights(N)
m_val = mean(y_vec)
println("\n3. _tss_centered (@simd):")
bench_tss_simd = @benchmark Regress._tss_centered($y_vec, $m_val, $wts) samples=50
display(bench_tss_simd)
time_tss_simd = median(bench_tss_simd).time / 1e6

#------------------------------------------------------------------------------
# Now load LoopVectorization and re-benchmark
#------------------------------------------------------------------------------
println()
println("-" ^ 70)
println("Loading LoopVectorization (triggers extension loading)...")
println("-" ^ 70)

using LoopVectorization

lv_ext = Base.get_extension(Regress, :RegressLVExt)
println("RegressLVExt loaded: $(lv_ext !== nothing)")

# Warmup with turbo
_ = ols(df, formula)
GC.gc()

println("\n1. Full OLS fit (@turbo):")
bench_full_turbo = @benchmark ols($df, $formula) samples=5 evals=1
display(bench_full_turbo)
time_full_turbo = median(bench_full_turbo).time / 1e9

println("\n2. compute_rss (@turbo):")
bench_rss_turbo = @benchmark Regress.compute_rss($y_vec, $mu_vec) samples=50
display(bench_rss_turbo)
time_rss_turbo = median(bench_rss_turbo).time / 1e6

println("\n3. _tss_centered (@turbo):")
bench_tss_turbo = @benchmark Regress._tss_centered($y_vec, $m_val, $wts) samples=50
display(bench_tss_turbo)
time_tss_turbo = median(bench_tss_turbo).time / 1e6

#------------------------------------------------------------------------------
# Summary
#------------------------------------------------------------------------------
println()
println("=" ^ 70)
println("SUMMARY: @simd vs @turbo (N = $N, K = $K)")
println("=" ^ 70)
println()
println("┌─────────────────────────────┬───────────┬───────────┬─────────┐")
println("│ Function                    │   @simd   │  @turbo   │ Speedup │")
println("├─────────────────────────────┼───────────┼───────────┼─────────┤")
@printf("│ Full OLS fit                │ %7.3f s │ %7.3f s │  %5.2fx │\n",
    time_full_simd, time_full_turbo, time_full_simd / time_full_turbo)
@printf("│ compute_rss                 │ %7.2f ms│ %7.2f ms│  %5.2fx │\n",
    time_rss_simd, time_rss_turbo, time_rss_simd / time_rss_turbo)
@printf("│ _tss_centered               │ %7.2f ms│ %7.2f ms│  %5.2fx │\n",
    time_tss_simd, time_tss_turbo, time_tss_simd / time_tss_turbo)
println("└─────────────────────────────┴───────────┴───────────┴─────────┘")
println()
println("Note: Full OLS includes StatsModels overhead (~66%) not optimizable by LV.")
println()
println("Benchmark complete.")
