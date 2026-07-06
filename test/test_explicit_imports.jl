@testitem "ExplicitImports" tags = [:explicit_imports] begin
    using ExplicitImports
    using Regress

    # Deliberate reliance on non-public names:
    # - PValue, NoQuote (StatsBase): coefficient-table formatting, same usage as GLM.jl
    # - Schema, FullRank, collect_matrix_terms, missing_omit, needs_schema,
    #   termvars (StatsModels): the standard downstream apply_schema integration
    # - istable, getcolumn, ColumnTable (Tables): documented interface, not
    #   declared `public` by Tables.jl
    # - CR, Clustering, _residuals, mask, numobs, setkernelweights!
    #   (CovarianceMatrices): protocol methods Regress extends
    # - alignment, print_matrix_row (Base): custom coeftable show
    # - QRCompactWY (LinearAlgebra): concrete type of the stored QR factorization
    test_explicit_imports(Regress;
        ignore = (:PValue, :NoQuote,
            :Schema, :FullRank, :collect_matrix_terms, :missing_omit,
            :needs_schema, :termvars,
            :istable, :getcolumn, :ColumnTable,
            :CR, :Clustering, :_residuals, :mask, :numobs, :setkernelweights!,
            :alignment, :print_matrix_row,
            :QRCompactWY))
end
