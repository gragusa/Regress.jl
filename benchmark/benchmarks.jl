"""
Benchmarks for Regress.jl

Benchmark suite measuring:
1. OLS estimation (formula and matrix interfaces)
2. OLS with fixed effects
3. IV estimation (TSLS, LIML, Fuller)
4. Post-estimation robust variance computation
"""

using BenchmarkTools
using DataFrames
using Random
using StableRNGs

using Regress
using CovarianceMatrices

# ============================================================================
# Data Generation
# ============================================================================

const DEFAULT_SEED = 20240612

function generate_ols_data(rng::AbstractRNG, n::Int, k::Int)
    X = randn(rng, n, k)
    y = X * ones(k) + randn(rng, n)
    df = DataFrame(X, [Symbol("x$i") for i in 1:k])
    df.y = y
    return df, X, y
end

function generate_fe_data(rng::AbstractRNG, n::Int, k::Int, n_groups::Int)
    df, X, y = generate_ols_data(rng, n, k)
    df.group1 = rand(rng, 1:n_groups, n)
    df.group2 = rand(rng, 1:(n_groups ÷ 2), n)
    return df
end

function generate_iv_data(rng::AbstractRNG, n::Int, n_endo::Int, n_inst::Int)
    # Generate instruments
    Z = randn(rng, n, n_inst)

    # Generate endogenous variables (correlated with error)
    u = randn(rng, n)
    X_endo = Z[:, 1:n_endo] * 0.5 .+ 0.5 * u .+ randn(rng, n, n_endo)

    # Generate exogenous controls
    X_exog = randn(rng, n, 2)

    # Generate outcome
    y = X_endo * ones(n_endo) + X_exog * ones(2) + u

    # Build DataFrame
    df = DataFrame(y = y)
    for i in 1:n_endo
        df[!, Symbol("endo$i")] = X_endo[:, i]
    end
    for i in 1:n_inst
        df[!, Symbol("z$i")] = Z[:, i]
    end
    for i in 1:2
        df[!, Symbol("x$i")] = X_exog[:, i]
    end

    return df
end

# ============================================================================
# Benchmark Suite
# ============================================================================

const SUITE = BenchmarkGroup()

# ----------------------------------------------------------------------------
# OLS Estimation Benchmarks
# ----------------------------------------------------------------------------

SUITE["ols"] = BenchmarkGroup()

# Small dataset
let rng = StableRNG(DEFAULT_SEED)
    df_small, X_small, y_small = generate_ols_data(rng, 1000, 5)

    SUITE["ols"]["formula_n1000_k5"] = @benchmarkable ols($df_small, @formula(y ~
                                                                              x1 + x2 + x3 +
                                                                              x4 + x5))
    SUITE["ols"]["matrix_n1000_k5"] = @benchmarkable ols($X_small, $y_small)
end

# Medium dataset
let rng = StableRNG(DEFAULT_SEED + 1)
    df_medium, X_medium, y_medium = generate_ols_data(rng, 10_000, 10)

    SUITE["ols"]["formula_n10000_k10"] = @benchmarkable ols(
        $df_medium, @formula(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10))
    SUITE["ols"]["matrix_n10000_k10"] = @benchmarkable ols($X_medium, $y_medium)
end

# Large dataset
let rng = StableRNG(DEFAULT_SEED + 2)
    df_large, X_large, y_large = generate_ols_data(rng, 100_000, 20)

    SUITE["ols"]["matrix_n100000_k20"] = @benchmarkable ols($X_large, $y_large)
end

# ----------------------------------------------------------------------------
# Fixed Effects Benchmarks
# ----------------------------------------------------------------------------

SUITE["fe"] = BenchmarkGroup()

let rng = StableRNG(DEFAULT_SEED + 10)
    df_fe = generate_fe_data(rng, 10_000, 5, 100)

    SUITE["fe"]["one_fe_n10000"] = @benchmarkable ols($df_fe, @formula(y ~
                                                                       x1 + x2 + x3 + x4 +
                                                                       x5 + fe(group1)))
    SUITE["fe"]["two_fe_n10000"] = @benchmarkable ols(
        $df_fe, @formula(y ~ x1 + x2 + x3 + x4 + x5 + fe(group1) + fe(group2)))
end

let rng = StableRNG(DEFAULT_SEED + 11)
    df_fe_large = generate_fe_data(rng, 50_000, 5, 500)

    SUITE["fe"]["one_fe_n50000"] = @benchmarkable ols(
        $df_fe_large, @formula(y ~ x1 + x2 + x3 + x4 + x5 + fe(group1)))
end

# ----------------------------------------------------------------------------
# IV Estimation Benchmarks
# ----------------------------------------------------------------------------

SUITE["iv"] = BenchmarkGroup()

let rng = StableRNG(DEFAULT_SEED + 20)
    df_iv = generate_iv_data(rng, 5000, 1, 3)

    SUITE["iv"]["tsls_n5000_1endo"] = @benchmarkable iv(
        TSLS(), $df_iv, @formula(y ~ x1 + x2 + (endo1 ~ z1 + z2 + z3)))
    SUITE["iv"]["liml_n5000_1endo"] = @benchmarkable iv(
        LIML(), $df_iv, @formula(y ~ x1 + x2 + (endo1 ~ z1 + z2 + z3)))
    SUITE["iv"]["fuller_n5000_1endo"] = @benchmarkable iv(
        Fuller(1.0), $df_iv, @formula(y ~ x1 + x2 + (endo1 ~ z1 + z2 + z3)))
end

let rng = StableRNG(DEFAULT_SEED + 21)
    df_iv_large = generate_iv_data(rng, 20_000, 2, 5)

    SUITE["iv"]["tsls_n20000_2endo"] = @benchmarkable iv(
        TSLS(), $df_iv_large, @formula(y ~
                                       x1 + x2 + (endo1 + endo2 ~ z1 + z2 + z3 + z4 + z5)))
end

# ----------------------------------------------------------------------------
# Robust Variance Benchmarks
# ----------------------------------------------------------------------------

SUITE["vcov"] = BenchmarkGroup()

let rng = StableRNG(DEFAULT_SEED + 30)
    df_vcov, _, _ = generate_ols_data(rng, 10_000, 10)
    model = ols(df_vcov, @formula(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10))

    SUITE["vcov"]["hc0"] = @benchmarkable $model + vcov(HC0())
    SUITE["vcov"]["hc1"] = @benchmarkable $model + vcov(HC1())
    SUITE["vcov"]["hc2"] = @benchmarkable $model + vcov(HC2())
    SUITE["vcov"]["hc3"] = @benchmarkable $model + vcov(HC3())
end

let rng = StableRNG(DEFAULT_SEED + 31)
    df_cluster = generate_fe_data(rng, 10_000, 5, 100)
    model_cluster = ols(df_cluster, @formula(y ~ x1 + x2 + x3 + x4 + x5))

    SUITE["vcov"]["cr0"] = @benchmarkable $model_cluster + vcov(CR0($df_cluster.group1))
    SUITE["vcov"]["cr1"] = @benchmarkable $model_cluster + vcov(CR1($df_cluster.group1))
end
