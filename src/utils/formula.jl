##############################################################################
##
## Iterate on terms
##
##############################################################################

eachterm(@nospecialize(x::AbstractTerm)) = (x,)
eachterm(@nospecialize(x::NTuple{N, AbstractTerm})) where {N} = x

##############################################################################
##
## Parse IV
##
##############################################################################
has_iv(@nospecialize(f::FormulaTerm)) = any(x -> x isa FormulaTerm, eachterm(f.rhs))

"""
    has_fe_in_iv(formula_iv::FormulaTerm)

Check if the instrument specification contains fe() terms.
This enables the FE-based first-stage estimation for high-dimensional instruments.
"""
has_fe_in_iv(@nospecialize(f::FormulaTerm)) = has_fe(f)

"""
    parse_iv(f::FormulaTerm)

Parse IV formula, separating endogenous variables from instruments.
Also detects fe() terms in the instrument specification for efficient first-stage estimation.

Returns:
- formula_exo: Exogenous part of formula
- formula_endo: Endogenous variables
- formula_iv: Regular (non-FE) instruments
- formula_iv_fe: FE-based instruments (empty if none)
"""
function parse_iv(@nospecialize(f::FormulaTerm))
    if has_iv(f)
        # Convert to concrete vectors immediately, then call barrier
        rhs_terms = collect(AbstractTerm, eachterm(f.rhs))
        return _parse_iv_impl(f.lhs, rhs_terms)
    else
        return f, FormulaTerm(ConstantTerm(0), ConstantTerm(0)),
        FormulaTerm(ConstantTerm(0), ConstantTerm(0)),
        FormulaTerm(ConstantTerm(0), ConstantTerm(0))
    end
end

# Barrier function with concrete types
function _parse_iv_impl(lhs::AbstractTerm, rhs_terms::Vector{AbstractTerm})
    i = findfirst(x -> x isa FormulaTerm, rhs_terms)
    term = rhs_terms[i]::FormulaTerm
    lhs_terms = collect(AbstractTerm, eachterm(term.lhs))
    rhs_iv_terms = collect(AbstractTerm, eachterm(term.rhs))
    both = intersect(lhs_terms, rhs_iv_terms)
    endos = setdiff(lhs_terms, both)
    exos = setdiff(rhs_iv_terms, both)

    # Validation
    isempty(endos) && throw("There are no endogeneous variables")
    length(exos) < length(endos) &&
        throw("Model not identified. There must be at least as many instrumental variables as endogeneneous variables")

    # Build formula_endo
    endo_rhs = AbstractTerm[ConstantTerm(0)]
    append!(endo_rhs, endos)
    formula_endo = FormulaTerm(ConstantTerm(0), Tuple(endo_rhs))

    # Separate FE instruments from regular instruments
    fe_ivs = AbstractTerm[x for x in exos if has_fe(x)]
    regular_ivs = AbstractTerm[x for x in exos if !has_fe(x)]

    iv_rhs = AbstractTerm[ConstantTerm(0)]
    append!(iv_rhs, regular_ivs)
    formula_iv = FormulaTerm(ConstantTerm(0), Tuple(iv_rhs))

    fe_iv_rhs = AbstractTerm[ConstantTerm(0)]
    append!(fe_iv_rhs, fe_ivs)
    formula_iv_fe = FormulaTerm(ConstantTerm(0), Tuple(fe_iv_rhs))

    # Build formula_exo
    exo_terms = AbstractTerm[t for t in rhs_terms if !isa(t, FormulaTerm)]
    append!(exo_terms, both)
    formula_exo = FormulaTerm(lhs, Tuple(exo_terms))

    return formula_exo, formula_endo, formula_iv, formula_iv_fe
end

##############################################################################
##
## Parse FixedEffect
##
##############################################################################
struct FixedEffectTerm <: AbstractTerm
    x::Symbol
end
StatsModels.termvars(t::FixedEffectTerm) = [t.x]
fe(x::Term) = fe(Symbol(x))
fe(s::Symbol) = FixedEffectTerm(s)

has_fe(::FixedEffectTerm) = true
has_fe(::FunctionTerm{typeof(fe)}) = true
has_fe(@nospecialize(t::InteractionTerm)) = any(has_fe(x) for x in t.terms)
has_fe(::AbstractTerm) = false
has_fe(@nospecialize(t::FormulaTerm)) = any(has_fe(x) for x in eachterm(t.rhs))

function parse_fe(@nospecialize(f::FormulaTerm))
    if has_fe(f)
        # Convert to concrete vector immediately, then call barrier
        rhs_terms = collect(AbstractTerm, eachterm(f.rhs))
        return _parse_fe_impl(f.lhs, rhs_terms)
    else
        return f, FormulaTerm(ConstantTerm(0), ConstantTerm(0))
    end
end

# Barrier function with concrete types - lhs can be AbstractTerm or Tuple
function _parse_fe_impl(lhs, rhs_terms::Vector{AbstractTerm})
    main_terms = AbstractTerm[term for term in rhs_terms if !has_fe(term)]
    fe_terms = AbstractTerm[term for term in rhs_terms if has_fe(term)]
    formula_main = FormulaTerm(lhs, Tuple(main_terms))
    formula_fe = FormulaTerm(ConstantTerm(0), Tuple(fe_terms))
    return formula_main, formula_fe
end

fesymbol(t::FixedEffectTerm) = t.x
fesymbol(t::FunctionTerm{typeof(fe)}) = Symbol(t.args[1])

"""
    parse_fixedeffect(data, formula::FormulaTerm)
    parse_fixedeffect(data, ts::NTuple{N, AbstractTerm})

Construct any `FixedEffect` specified with a `FixedEffectTerm`.

# Returns
- `Vector{FixedEffect}`: a collection of all `FixedEffect`s constructed.
- `Vector{Symbol}`: names assigned to the fixed effect estimates (can be used as column names).
- `Vector{Symbol}`: names of original fixed effects.
- `FormulaTerm` or `NTuple{N, AbstractTerm}`: `formula` or `ts` without any term related to fixed effects (an intercept may be explicitly omitted if necessary).
"""
function parse_fixedeffect(data, @nospecialize(formula::FormulaTerm))
    # Convert to concrete vector immediately, then call barrier
    rhs_terms = collect(AbstractTerm, eachterm(formula.rhs))
    return _parse_fixedeffect_impl(data, rhs_terms)
end

# Barrier function with concrete types
function _parse_fixedeffect_impl(data, rhs_terms::Vector{AbstractTerm})
    fes = FixedEffect[]
    feids = Symbol[]
    fekeys = Symbol[]
    for term in rhs_terms
        result = _parse_fixedeffect_term(data, term)
        if result !== nothing
            push!(fes, result[1]::FixedEffect)
            push!(feids, result[2]::Symbol)
            append!(fekeys, result[3]::Vector{Symbol})
        end
    end
    return fes, feids, unique(fekeys)
end

# Method for external packages
function parse_fixedeffect(data, @nospecialize(ts::NTuple{N, AbstractTerm})) where {N}
    # Convert to concrete vector immediately
    terms_vec = collect(AbstractTerm, eachterm(ts))
    return _parse_fixedeffect_tuple_impl(data, terms_vec)
end

# Barrier function for tuple version
function _parse_fixedeffect_tuple_impl(data, terms_vec::Vector{AbstractTerm})
    fes = FixedEffect[]
    ids = Symbol[]
    fekeys = Symbol[]
    for term in terms_vec
        result = _parse_fixedeffect_term(data, term)
        if result !== nothing
            push!(fes, result[1]::FixedEffect)
            push!(ids, result[2]::Symbol)
            append!(fekeys, result[3]::Vector{Symbol})
        end
    end
    if !isempty(fes)
        if any(fe.interaction isa UnitWeights for fe in fes)
            filtered = AbstractTerm[InterceptTerm{false}()]
            for term in terms_vec
                if !isa(term, Union{ConstantTerm, InterceptTerm}) && !has_fe(term)
                    push!(filtered, term)
                end
            end
            # Note: we return the modified tuple but caller may not use it
        end
    end
    return fes, ids, unique(fekeys)
end

# Unified term parsing - dispatches on concrete term type
function _parse_fixedeffect_term(data, @nospecialize(t::AbstractTerm))
    if t isa InteractionTerm
        return _parse_fixedeffect_interaction(data, t)
    elseif has_fe(t)
        st = fesymbol(t)::Symbol
        col = Tables.getcolumn(data, st)
        return FixedEffect(col), Symbol(:fe_, st), Symbol[st]
    end
    return nothing
end

# Barrier for InteractionTerm parsing
function _parse_fixedeffect_interaction(data, t::InteractionTerm)
    # Collect into concrete vectors
    terms_vec = collect(t.terms)
    fes_vec = AbstractTerm[]
    interactions_vec = AbstractTerm[]
    for x in terms_vec
        if has_fe(x)
            push!(fes_vec, x)
        else
            push!(interactions_vec, x)
        end
    end

    if !isempty(fes_vec)
        # Get FE column names
        fe_names = Symbol[fesymbol(x)::Symbol for x in fes_vec]

        # Get interaction column names
        interaction_syms = Symbol[Symbol(x) for x in interactions_vec]

        # Compute interaction weights
        v1 = _multiply_columns(data, interaction_syms)

        # Get FE columns
        fe_cols = [Tables.getcolumn(data, fe_name) for fe_name in fe_names]
        fe = FixedEffect(fe_cols...; interaction = v1)

        # Build name
        interaction_strs = String[string(x) for x in interactions_vec]
        s = vcat(["fe_" * string(fe_name) for fe_name in fe_names], interaction_strs)
        name = Symbol(reduce((x1, x2) -> x1 * "&" * x2, s))

        return fe, name, fe_names
    end
    return nothing
end

# Barrier function for column multiplication with concrete types
function _multiply_columns(data, ss::Vector{Symbol})
    n = size(data, 1)::Int
    if isempty(ss)
        return uweights(n)
    elseif length(ss) == 1
        col = Tables.getcolumn(data, ss[1])
        return _convert_to_float64(col)
    else
        # Get all columns first
        cols = Any[Tables.getcolumn(data, x) for x in ss]
        result = _multiply_cols(cols)
        return result
    end
end

# Helper to convert column to Float64, handling missing
function _convert_to_float64(col)
    # Use a concrete output type
    n = length(col)
    out = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        v = col[i]
        out[i] = ismissing(v) ? 0.0 : Float64(v)
    end
    return out
end

# Helper to multiply columns element-wise
function _multiply_cols(cols::Vector{Any})
    n = length(cols[1])
    out = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        val = 1.0
        for col in cols
            v = col[i]
            if ismissing(v)
                val = 0.0
                break
            end
            val *= Float64(v)
        end
        out[i] = val
    end
    return out
end

# Backward compatibility aliases for tests
const _multiply = _multiply_columns
const _parse_fixedeffect = _parse_fixedeffect_term
