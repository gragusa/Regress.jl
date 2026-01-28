# Generate validation data for comparison with R fixest package
# This script creates a dataset with endogenous variables, instruments, and fixed effects

using StableRNGs
using DataFrames
using CSV

"""
    generate_fixest_validation_data()

Generate a DataFrame for validating Regress.jl against R fixest.

DGP (true parameters):
- beta_endo = 2.0
- beta_x1 = 0.5
- beta_x2 = 0.3

The endogenous variable `endo` is correlated with the error term through a shared
first-stage error component.
"""
function generate_fixest_validation_data()
    rng = StableRNG(12345)
    n = 1000

    # Fixed effect groupings
    fe1 = repeat(1:20, inner = 50)
    fe2 = repeat(1:10, outer = 100)

    # FE effects (random draws)
    fe1_effects = randn(rng, 20)
    fe2_effects = randn(rng, 10)
    fe1_effect = fe1_effects[fe1]
    fe2_effect = fe2_effects[fe2]

    # Exogenous controls
    x1 = randn(rng, n)
    x2 = randn(rng, n)

    # Instruments
    Z_continuous = randn(rng, n) .+ 0.3 .* x1  # correlated with x1 but not with error
    Z_cat = rand(rng, 1:5, n)
    z_cat_effects = [0.0, 0.5, -0.3, 0.8, -0.2]
    z_cat_effect = z_cat_effects[Z_cat]

    # First stage: endo depends on instruments + exogenous + FE + first-stage error
    first_stage_error = randn(rng, n)
    endo = 0.5 .+ 0.6 .* Z_continuous .+ z_cat_effect .+ 0.2 .* x1 .+
           0.5 .* fe1_effect .+ first_stage_error

    # Structural equation error (correlated with first-stage error -> endogeneity)
    u = 0.5 .* first_stage_error .+ randn(rng, n)

    # Outcome: y = 1.0 + 2.0*endo + 0.5*x1 + 0.3*x2 + fe1 + fe2 + u
    y = 1.0 .+ 2.0 .* endo .+ 0.5 .* x1 .+ 0.3 .* x2 .+
        fe1_effect .+ fe2_effect .+ u

    return DataFrame(
        y = y,
        endo = endo,
        x1 = x1,
        x2 = x2,
        Z_continuous = Z_continuous,
        Z_cat = Z_cat,
        fe1 = fe1,
        fe2 = fe2
    )
end

# Generate and save data
if abspath(PROGRAM_FILE) == @__FILE__
    df = generate_fixest_validation_data()
    output_path = joinpath(@__DIR__, "fixest_validation_data.csv")
    CSV.write(output_path, df)
    println("Generated validation data: $output_path")
    println("  n = $(nrow(df))")
    println("  Columns: $(names(df))")
end
