"""
Response component for OLS estimation.

Similar to GLM.jl's LmResp, this type stores the response vector,
fitted values, and weights for a linear regression model.

When `save=:minimal` is used, `y` and `mu` are empty vectors to save memory.
Use `has_response_data(rr)` to check if response data is available.
"""
mutable struct OLSResponse{T <: AbstractFloat}
    y::Vector{T}                    # Original response vector (empty if save=:minimal)
    mu::Vector{T}                   # Fitted values ŷ = X*β (empty if save=:minimal)
    wts::Vector{T}                  # Weights (empty = unweighted)
    offset::Vector{T}               # Offset (empty = no offset, for GLM compatibility)

    # Response metadata
    response_name::Symbol
end

"""
    has_response_data(rr::OLSResponse) -> Bool

Check if response object has stored data (y and mu).
Returns false when model was fit with save=:minimal.
"""
has_response_data(rr::OLSResponse) = !isempty(rr.y)

"""
    clear_response_data!(rr::OLSResponse{T}) where {T}

Clear response data (y and mu) to save memory. Used for save=:minimal mode.
"""
function clear_response_data!(rr::OLSResponse{T}) where {T}
    empty!(rr.y)
    empty!(rr.mu)
    return rr
end

"""
    build_response(y, wts, response_name) -> OLSResponse

Construct an OLSResponse object from response vector, weights, and name.

# Arguments
- `y::Vector{T}`: Response vector
- `wts::AbstractWeights`: Weights (UnitWeights for unweighted)
- `response_name::Symbol`: Name of the response variable

# Returns
- `OLSResponse{T}`: Response object with uninitialized fitted values
"""
function build_response(y::Vector{T}, wts::AbstractWeights,
        response_name::Symbol) where {T <: AbstractFloat}
    # Convert weights to vector (or empty for unweighted)
    wts_vec = wts isa UnitWeights ? T[] : convert(Vector{T}, wts.values)

    # Initialize fitted values (will be updated after solve)
    mu = similar(y)

    # Empty offset for now (future: support offsets)
    offset = T[]

    return OLSResponse(y, mu, wts_vec, offset, response_name)
end

"""
    isweighted(rr::OLSResponse) -> Bool

Check if response object has weights.
"""
isweighted(rr::OLSResponse) = !isempty(rr.wts)

"""
    hasoffset(rr::OLSResponse) -> Bool

Check if response object has an offset.
"""
hasoffset(rr::OLSResponse) = !isempty(rr.offset)
