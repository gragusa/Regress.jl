using DataFrames, Random, BenchmarkTools, Printf

import FixedEffectModels as FEM
import Regress as REG
using CovarianceMatrices: CR1, vcov
using LinearAlgebra
BLAS.set_num_threads(1)

BenchmarkTools.DEFAULT_PARAMETERS.samples = 5
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 120

function generate_dataset1()
    rng = Random.default_rng()
    Random.seed!(rng, 42)
    N = 10_000_000
    K = 100
    id1 = rand(rng, 1:div(N, K), N)
    id2 = rand(rng, 1:K, N)
    x1 = 5 .* cos.(id1) .+ 5 .* sin.(id2) .+ randn(rng, N)
    x2 = cos.(id1) .+ sin.(id2) .+ randn(rng, N)
    y = 3 .* x1 .+ 5 .* x2 .+ cos.(id1) .+ cos.(id2) .^ 2 .+ randn(rng, N)
    return DataFrame(id1 = id1, id2 = id2, x1 = x1, x2 = x2, y = y), N
end

fmt(x) = @sprintf("%.3f", x)

println("=" ^ 70)
println("Benchmark with collinearity = :sweep (matching FEM)")
println("=" ^ 70)
println()

df1, N1 = generate_dataset1()
results = []

# Scenario 1: Simple OLS
scenario = "OLS: y ~ x1 + x2"
println("\n  $scenario")

formula_fem = FEM.@formula(y ~ x1 + x2)
formula_reg = REG.@formula(y ~ x1 + x2)

FEM.reg(df1, formula_fem)
t_fem = @benchmark FEM.reg($df1, $formula_fem) samples=5 evals=1
fem_t = median(t_fem).time / 1e9
println("    FEM: $(fmt(fem_t))s")

REG.ols(df1, formula_reg, collinearity = :sweep)
t_reg = @benchmark REG.ols($df1, $formula_reg, collinearity = :sweep) samples=5 evals=1
reg_t = median(t_reg).time / 1e9
println("    Regress: $(fmt(reg_t))s")
println("    Ratio: $(fmt(reg_t/fem_t))")
push!(results, (scenario, fem_t, reg_t))

# Scenario 2: OLS + cluster
scenario = "OLS + cluster(id2)"
println("\n  $scenario")

FEM.reg(df1, formula_fem, FEM.Vcov.cluster(:id2))
t_fem = @benchmark FEM.reg($df1, $formula_fem, FEM.Vcov.cluster(:id2)) samples=5 evals=1
fem_t = median(t_fem).time / 1e9
println("    FEM: $(fmt(fem_t))s")

cluster_vec = df1.id2
m = REG.ols(df1, formula_reg, collinearity = :sweep)
vcov(CR1(cluster_vec), m)
t_reg = @benchmark begin
    m=REG.ols($df1, $formula_reg, collinearity = :sweep)
    vcov(CR1($cluster_vec), m)
end samples=5 evals=1
reg_t = median(t_reg).time / 1e9
println("    Regress: $(fmt(reg_t))s")
println("    Ratio: $(fmt(reg_t/fem_t))")
push!(results, (scenario, fem_t, reg_t))

# Scenario 3: One FE
scenario = "FE: y ~ x1 + x2 | id1"
println("\n  $scenario")

formula_fem_fe = FEM.@formula(y ~ x1 + x2 + FEM.fe(id1))
formula_reg_fe = REG.@formula(y ~ x1 + x2 + REG.fe(id1))

FEM.reg(df1, formula_fem_fe)
t_fem = @benchmark FEM.reg($df1, $formula_fem_fe) samples=5 evals=1
fem_t = median(t_fem).time / 1e9
println("    FEM: $(fmt(fem_t))s")

REG.ols(df1, formula_reg_fe, collinearity = :sweep)
t_reg = @benchmark REG.ols($df1, $formula_reg_fe, collinearity = :sweep) samples=5 evals=1
reg_t = median(t_reg).time / 1e9
println("    Regress: $(fmt(reg_t))s")
println("    Ratio: $(fmt(reg_t/fem_t))")
push!(results, (scenario, fem_t, reg_t))

# Scenario 4: One FE + cluster
scenario = "FE + cluster: | id1"
println("\n  $scenario")

FEM.reg(df1, formula_fem_fe, FEM.Vcov.cluster(:id1))
t_fem = @benchmark FEM.reg($df1, $formula_fem_fe, FEM.Vcov.cluster(:id1)) samples=5 evals=1
fem_t = median(t_fem).time / 1e9
println("    FEM: $(fmt(fem_t))s")

m = REG.ols(df1, formula_reg_fe, save_cluster = :id1, collinearity = :sweep)
vcov(CR1(:id1), m)
t_reg = @benchmark begin
    m=REG.ols($df1, $formula_reg_fe, save_cluster = :id1, collinearity = :sweep)
    vcov(CR1(:id1), m)
end samples=5 evals=1
reg_t = median(t_reg).time / 1e9
println("    Regress: $(fmt(reg_t))s")
println("    Ratio: $(fmt(reg_t/fem_t))")
push!(results, (scenario, fem_t, reg_t))

println()
println("=" ^ 70)
println("Summary (with collinearity = :sweep)")
println("=" ^ 70)
println()
println("| Scenario | FEM (s) | Regress (s) | Ratio |")
println("|----------|---------|-------------|-------|")
for (s, f, r) in results
    println("| $s | $(fmt(f)) | $(fmt(r)) | $(fmt(r/f)) |")
end
