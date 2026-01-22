"""
Fixed effects component for OLS estimation.

This type stores fixed effect estimates, cluster variables, and metadata
specific to econometric models with high-dimensional fixed effects.
"""
struct OLSFixedEffects{T <: AbstractFloat}
    # Fixed effect solutions
    fe::DataFrame                   # Solved fixed effects (can be empty)
    fe_names::Vector{Symbol}        # Names of FE variables

    # Cluster variables for robust vcov
    clusters::NamedTuple           # Cluster identifiers (subsetted to esample)

    # Fixed effects grouping vectors for nesting detection
    fe_groups::Vector{Vector{Int}} # One ref vector per FE dimension (empty for :minimal)

    # Fixed effects metadata
    dof_fes::Int                   # Degrees of freedom absorbed by FEs
    ngroups::Vector{Int}           # Number of groups per FE dimension

    # Solver metadata
    iterations::Int
    converged::Bool
    method::Symbol                 # :cpu, :cuda, :metal, :none
end

"""
    has_fes(fes::OLSFixedEffects) -> Bool

Check if model has fixed effects.
"""
has_fes(fes::OLSFixedEffects) = fes.dof_fes > 0

"""
    has_clusters(fes::OLSFixedEffects) -> Bool

Check if model has cluster variables.
"""
has_clusters(fes::OLSFixedEffects) = length(fes.clusters) > 0

"""
    build_empty_fes(T::Type) -> OLSFixedEffects{T}

Construct an empty fixed effects component for models without FEs.
"""
function build_empty_fes(::Type{T}) where {T <: AbstractFloat}
    return OLSFixedEffects{T}(
        DataFrame(),
        Symbol[],
        NamedTuple(),
        Vector{Int}[],  # Empty fe_groups
        0,
        Int[],
        0,
        true,
        :none
    )
end
