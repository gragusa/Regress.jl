using DataFrames, Random, Chairmarks, Printf, Statistics

import FixedEffectModels as FEM
import Regress as REG
using LinearAlgebra
BLAS.set_num_threads(1)

function generate_iv_dataset()
    rng = Random.default_rng()
    Random.seed!(rng, 42)
    N = 1_000_000  # 1M observations for IV
    K = 100

    # Generate IVs
    z1 = randn(rng, N)
    z2 = randn(rng, N)

    # Generate endogenous variable correlated with instruments
    endog = 0.5 .* z1 .+ 0.3 .* z2 .+ randn(rng, N)

    # Exogenous variables
    x1 = randn(rng, N)
    x2 = randn(rng, N)

    # Fixed effects
    id1 = rand(rng, 1:div(N, K), N)
    id2 = rand(rng, 1:K, N)

    # Outcome with endogeneity
    y = 2.0 .* x1 .+ 3.0 .* x2 .+ 1.5 .* endog .+ 0.5 .* randn(rng, N)

    return DataFrame(
        id1 = id1, id2 = id2,
        x1 = x1, x2 = x2,
        endog = endog,
        z1 = z1, z2 = z2,
        y = y
    ), N
end

fmt(x) = @sprintf("%.3f", x)

println("=" ^ 70)
println("IV Performance Benchmark: Regress.jl vs FixedEffectModels.jl")
println("Using Chairmarks for reliable timing")
println("=" ^ 70)
println()
println("Configuration:")
println("  Julia threads: $(Threads.nthreads())")
println("  BLAS threads: $(BLAS.get_num_threads())")
println()

df, N = generate_iv_dataset()
println("Dataset: $(N) observations")
println("-" ^ 40)

results = []

# Scenario 1: Simple IV
scenario = "IV: y ~ x1 + x2 + (endog ~ z1 + z2)"
println("\n  $scenario")

formula_fem = FEM.@formula(y ~ x1 + x2 + (endog ~ z1 + z2))
formula_reg = REG.@formula(y ~ x1 + x2 + (endog ~ z1 + z2))

# Warmup
FEM.reg(df, formula_fem)
REG.iv(REG.TSLS(), df, formula_reg)

t_fem = @be FEM.reg($df, $formula_fem) seconds=30
fem_t = median(t_fem).time
println("    FEM: $(fmt(fem_t))s ($(length(t_fem.samples)) samples)")

t_reg = @be REG.iv(REG.TSLS(), $df, $formula_reg) seconds=30
reg_t = median(t_reg).time
println("    Regress: $(fmt(reg_t))s ($(length(t_reg.samples)) samples)")
println("    Ratio: $(fmt(reg_t/fem_t))")
push!(results, (scenario, fem_t, reg_t))

# Scenario 2: IV with one FE
scenario = "IV + FE: | id1"
println("\n  $scenario")

formula_fem_fe = FEM.@formula(y ~ x1 + x2 + (endog ~ z1 + z2) + FEM.fe(id1))
formula_reg_fe = REG.@formula(y ~ x1 + x2 + (endog ~ z1 + z2) + REG.fe(id1))

FEM.reg(df, formula_fem_fe)
REG.iv(REG.TSLS(), df, formula_reg_fe)

t_fem = @be FEM.reg($df, $formula_fem_fe) seconds=30
fem_t = median(t_fem).time
println("    FEM: $(fmt(fem_t))s ($(length(t_fem.samples)) samples)")

t_reg = @be REG.iv(REG.TSLS(), $df, $formula_reg_fe) seconds=30
reg_t = median(t_reg).time
println("    Regress: $(fmt(reg_t))s ($(length(t_reg.samples)) samples)")
println("    Ratio: $(fmt(reg_t/fem_t))")
push!(results, (scenario, fem_t, reg_t))

# Scenario 3: IV with two FEs
scenario = "IV + 2FE: | id1 + id2"
println("\n  $scenario")

formula_fem_2fe = FEM.@formula(y ~ x1 + x2 + (endog ~ z1 + z2) + FEM.fe(id1) + FEM.fe(id2))
formula_reg_2fe = REG.@formula(y ~ x1 + x2 + (endog ~ z1 + z2) + REG.fe(id1) + REG.fe(id2))

FEM.reg(df, formula_fem_2fe)
REG.iv(REG.TSLS(), df, formula_reg_2fe)

t_fem = @be FEM.reg($df, $formula_fem_2fe) seconds=60
fem_t = median(t_fem).time
println("    FEM: $(fmt(fem_t))s ($(length(t_fem.samples)) samples)")

t_reg = @be REG.iv(REG.TSLS(), $df, $formula_reg_2fe) seconds=60
reg_t = median(t_reg).time
println("    Regress: $(fmt(reg_t))s ($(length(t_reg.samples)) samples)")
println("    Ratio: $(fmt(reg_t/fem_t))")
push!(results, (scenario, fem_t, reg_t))

println()
println("=" ^ 70)
println("Summary")
println("=" ^ 70)
println()
println("| Scenario | FEM (s) | Regress (s) | Ratio |")
println("|----------|---------|-------------|-------|")
for (s, f, r) in results
    println("| $s | $(fmt(f)) | $(fmt(r)) | $(fmt(r/f)) |")
end
