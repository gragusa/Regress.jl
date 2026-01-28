# Generate reproducible IV test data with nested FE structure
# Run this script to regenerate iv_nested.csv

using CSV, DataFrames, Random
using StableRNGs

# Use StableRNG for reproducibility across Julia versions
rng = StableRNG(42)

# Parameters
n = 500
n_states = 5
n_counties_per_state = 10  # County is nested in state

# Generate nested geographic structure
state_ids = repeat(1:n_states, inner = n ÷ n_states)
# Counties are nested: each state has unique county IDs
county_ids = zeros(Int, n)
for i in 1:n
    state = state_ids[i]
    local_county = mod1((i - 1) ÷ (n ÷ (n_states * n_counties_per_state)) + 1, n_counties_per_state)
    # Make county globally unique but nested in state
    county_ids[i] = (state - 1) * n_counties_per_state + local_county
end

# Generate exogenous controls
x1 = randn(rng, n)
x2 = randn(rng, n)

# Generate instrument (correlated with endogenous but not with error)
z = randn(rng, n) .+ 0.5 .* x1

# Generate endogenous variable (correlated with instrument and error)
# First stage: endo = z + x1 + x2 + state effects + first_stage_error
state_fe_values = randn(rng, n_states)
state_fe = [state_fe_values[s] for s in state_ids]
first_stage_error = randn(rng, n)
endo = 0.8 .* z .+ 0.3 .* x1 .+ 0.2 .* x2 .+ state_fe .+ first_stage_error

# Generate structural error (correlated with first_stage_error for endogeneity)
u = 0.5 .* first_stage_error .+ randn(rng, n)

# Generate outcome variable
# True model: y = 1.0 + 2.0*endo + 0.5*x1 + 0.3*x2 + state_fe + county_fe + u
county_fe_values = randn(rng, n_states * n_counties_per_state)
county_fe = [county_fe_values[c] for c in county_ids]
y = 1.0 .+ 2.0 .* endo .+ 0.5 .* x1 .+ 0.3 .* x2 .+ state_fe .+ county_fe .+ u

# Create DataFrame
df = DataFrame(
    y = y,
    endo = endo,
    x1 = x1,
    x2 = x2,
    z = z,
    state_id = state_ids,
    county_id = county_ids
)

# Write to CSV
CSV.write(joinpath(@__DIR__, "iv_nested.csv"), df)
println("Generated iv_nested.csv with $(nrow(df)) observations")
println("  States: $(n_states)")
println("  Counties: $(n_states * n_counties_per_state) (nested in states)")
