##############################################################################
##
## Display Formatting Utilities
##
## Shared utilities for text and HTML output of model results.
##
##############################################################################

using StatsBase: NoQuote, PValue

# ANSI color codes for terminal output
const ANSI_GREEN = "\e[32m"
const ANSI_YELLOW = "\e[33m"
const ANSI_RESET = "\e[0m"

# Box-drawing character for horizontal lines
const BOX_HORIZONTAL = '─'  # U+2500

"""
    supports_color(io::IO) -> Bool

Check if the IO context supports colored output.
Uses `:color` context property, defaulting to `false`.
"""
supports_color(io::IO) = get(io, :color, false)::Bool

"""
    print_horizontal_line(io::IO, width::Int; char::Char=BOX_HORIZONTAL)

Print a horizontal line of specified width, with optional green coloring
if the IO context supports color.
"""
function print_horizontal_line(io::IO, width::Int; char::Char = BOX_HORIZONTAL)
    if supports_color(io)
        print(io, ANSI_GREEN, repeat(char, width), ANSI_RESET)
    else
        print(io, repeat(char, width))
    end
end

"""
    println_horizontal_line(io::IO, width::Int; char::Char=BOX_HORIZONTAL)

Print a horizontal line followed by newline.
"""
function println_horizontal_line(io::IO, width::Int; char::Char = BOX_HORIZONTAL)
    print_horizontal_line(io, width; char = char)
    println(io)
end

"""
    compute_table_width(A::Vector, colnms::Vector{String}) -> Int

Compute the total table width based on column alignments and column names.
This matches the actual printed width when using `Base.print_matrix_row` with
a 2-space separator.

# Arguments
- `A`: Vector of alignment tuples from `Base.alignment()`
- `colnms`: Vector of column names (excluding row names)

# Returns
Total width in characters that the coefficient table will occupy.
"""
function compute_table_width(A::Vector, colnms::Vector{String})
    # A[1] is row names, A[2:end] are data columns
    # Header prints: spaces for A[1], then for each column: "  " + lpad(name, sum(A[j+1]))
    # Data rows use Base.print_matrix_row with "  " separator

    # Row names column
    width = sum(A[1])

    # Data columns: each has 2-space prefix + column width
    for j in 1:length(colnms)
        width += 2 + sum(A[j + 1])
    end

    return width
end

##############################################################################
##
## HTML Output Utilities
##
##############################################################################

"""
    html_escape(s::AbstractString) -> String

Escape HTML special characters in a string.
"""
function html_escape(s::AbstractString)
    s = replace(s, "&" => "&amp;")
    s = replace(s, "<" => "&lt;")
    s = replace(s, ">" => "&gt;")
    s = replace(s, "\"" => "&quot;")
    s = replace(s, "'" => "&#39;")
    return s
end

"""
    html_table_start(io::IO; class="regress-table", caption="")

Write opening HTML table tag with class and optional caption.
"""
function html_table_start(io::IO; class::String = "regress-table", caption::String = "")
    println(io, """<table class="$(class)">""")
    if !isempty(caption)
        println(io, "  <caption>$(html_escape(caption))</caption>")
    end
end

"""
    html_table_end(io::IO)

Write closing HTML table tag.
"""
html_table_end(io::IO) = println(io, "</table>")

"""
    html_thead_start(io::IO; class="")

Write opening thead tag with optional class.
"""
function html_thead_start(io::IO; class::String = "")
    class_attr = isempty(class) ? "" : """ class="$(class)" """
    println(io, "  <thead$(class_attr)>")
end

"""
    html_thead_end(io::IO)

Write closing thead tag.
"""
html_thead_end(io::IO) = println(io, "  </thead>")

"""
    html_tbody_start(io::IO; class="")

Write opening tbody tag with optional class.
"""
function html_tbody_start(io::IO; class::String = "")
    class_attr = isempty(class) ? "" : """ class="$(class)" """
    println(io, "  <tbody$(class_attr)>")
end

"""
    html_tbody_end(io::IO)

Write closing tbody tag.
"""
html_tbody_end(io::IO) = println(io, "  </tbody>")

"""
    html_tfoot_start(io::IO; class="")

Write opening tfoot tag with optional class.
"""
function html_tfoot_start(io::IO; class::String = "")
    class_attr = isempty(class) ? "" : """ class="$(class)" """
    println(io, "  <tfoot$(class_attr)>")
end

"""
    html_tfoot_end(io::IO)

Write closing tfoot tag.
"""
html_tfoot_end(io::IO) = println(io, "  </tfoot>")

"""
    html_row(io::IO, cells::Vector; class="", is_header=false)

Write an HTML table row with cells.

# Arguments
- `io`: Output IO stream
- `cells`: Vector of cell values (will be converted to strings and escaped)
- `class`: Optional CSS class for the row
- `is_header`: If true, use `<th>` tags instead of `<td>`
"""
function html_row(io::IO, cells::Vector; class::String = "", is_header::Bool = false)
    tag = is_header ? "th" : "td"
    row_attr = isempty(class) ? "" : """ class="$(class)" """
    print(io, "    <tr$(row_attr)>")
    for cell in cells
        print(io, "<$(tag)>$(html_escape(string(cell)))</$(tag)>")
    end
    println(io, "</tr>")
end

"""
    format_pvalue(p::Real) -> String

Format a p-value for HTML display.
"""
function format_pvalue(p::Real)
    if p < 0.0001
        return "<0.0001"
    else
        return @sprintf("%.4f", p)
    end
end

"""
    format_number(x::Real; digits=4) -> String

Format a number with specified decimal digits for HTML display.
"""
function format_number(x::Real; digits::Int = 4)
    return @sprintf("%.*f", digits, x)
end

##############################################################################
##
## Variance-Covariance Type Name Formatting
##
##############################################################################

"""
    vcov_type_name(v) -> String

Return a readable name for the variance-covariance estimator type.
Used in display output to indicate what type of standard errors are shown.

# Examples
```julia
vcov_type_name(HC1())       # "HC1"
vcov_type_name(HC3())       # "HC3"
vcov_type_name(CR1(:firm))  # "CR1"
vcov_type_name(Bartlett(5)) # "Bartlett(5)"
```
"""
function vcov_type_name(v)
    # Default: use the type name
    return string(typeof(v).name.name)
end

# HC/HR estimators - show as "HC" (more common notation)
# In CovarianceMatrices.jl, HC0-HC3 are type aliases for HR0-HR3
vcov_type_name(::CovarianceMatrices.HR0) = "HC0"
vcov_type_name(::CovarianceMatrices.HR1) = "HC1"
vcov_type_name(::CovarianceMatrices.HR2) = "HC2"
vcov_type_name(::CovarianceMatrices.HR3) = "HC3"

# HAC estimators with bandwidth
# Fixed bandwidth: Bartlett(5) -> "Bartlett(5)"
# Auto bandwidth: Bartlett(NeweyWest) -> "Bartlett(auto)"

# Helper to get clean kernel name (BartlettKernel -> Bartlett)
function _hac_kernel_name(v::CovarianceMatrices.HAC)
    typename = string(typeof(v).name.name)
    return replace(typename, "Kernel" => "")
end

function vcov_type_name(v::CovarianceMatrices.HAC{<:CovarianceMatrices.Fixed})
    typename = _hac_kernel_name(v)
    bw_val = CovarianceMatrices.bandwidth(v)
    # Show as integer if it's a whole number
    if bw_val == floor(bw_val)
        return @sprintf("%s(%d)", typename, Int(bw_val))
    else
        return @sprintf("%s(%.2f)", typename, bw_val)
    end
end

# A data-driven kernel selects its bandwidth from the data, so the number is a
# property of the estimate, not of the estimator; `vcov_type_name(v, V)` reports it.
vcov_type_name(v::CovarianceMatrices.HAC) = @sprintf("%s(auto)", _hac_kernel_name(v))

"""
    vcov_type_name(estimator, V) -> String

Name the variance estimator, reporting the bandwidth `V` selected when the
estimator does not fix one itself.

`V` is the computed variance matrix, as returned by [`vcov`](@ref). A
data-driven HAC kernel chooses its bandwidth from the data, so the number
exists only on the estimate; `CovarianceMatrices.bandwidth` returns `nothing`
for estimators that select nothing, and the name is then unchanged.
"""
vcov_type_name(v, V) = vcov_type_name(v)

function vcov_type_name(v::CovarianceMatrices.HAC, V)
    bw_val = CovarianceMatrices.bandwidth(V)
    bw_val === nothing && return vcov_type_name(v)
    return @sprintf("%s(auto: %.2f)", _hac_kernel_name(v), bw_val)
end

# A fixed bandwidth is already in the estimator; reading it back off the
# estimate would report the same number.
vcov_type_name(v::CovarianceMatrices.HAC{<:CovarianceMatrices.Fixed}, V) = vcov_type_name(v)

##############################################################################
##
## Regression table for text output (OLS and IV estimators)
##
##############################################################################

"""
    coeftable_layout(io::IO, ct::CoefTable) -> (mat, A, width)

Cells of `ct` (row names first), their alignments as returned by
`Base.alignment`, and the width in characters of the printed table.
"""
function coeftable_layout(io::IO, ct::CoefTable)
    cols, colnms = ct.cols, ct.colnms
    nc = length(cols)
    nr = length(cols[1])
    rownms = isempty(ct.rownms) ?
             [lpad("[$i]", floor(Integer, log10(nr)) + 3) for i in 1:nr] : ct.rownms
    mat = [j == 1 ? NoQuote(rownms[i]) :
           j - 1 == ct.pvalcol ? NoQuote(sprint(show, PValue(cols[j - 1][i]))) :
           j - 1 in ct.teststatcol ? TestStat(cols[j - 1][i]) :
           cols[j - 1][i] isa AbstractString ? NoQuote(cols[j - 1][i]) : cols[j - 1][i]
           for i in 1:nr, j in 1:(nc + 1)]
    A = Base.alignment(io, mat, 1:size(mat, 1), 1:size(mat, 2),
        typemax(Int), typemax(Int), 3)
    nmswidths = pushfirst!(length.(colnms), 0)
    A = [nmswidths[i] > sum(A[i]) ? (A[i][1] + nmswidths[i] - sum(A[i]), A[i][2]) : A[i]
         for i in 1:length(A)]
    return mat, A, compute_table_width(A, colnms)
end

"""
    without_confint(ct::CoefTable) -> CoefTable

`ct` without its confidence-interval columns ("Lower …", "Upper …").
"""
function without_confint(ct::CoefTable)
    keep = findall(n -> !(startswith(n, "Lower") || startswith(n, "Upper")), ct.colnms)
    remap(c) = c == 0 ? 0 : something(findfirst(==(c), keep), 0)
    return CoefTable(reduce(hcat, ct.cols[keep]), ct.colnms[keep], ct.rownms,
        remap(ct.pvalcol), remap(ct.teststatcol))
end

"""
    print_regression(io::IO, title, ctop::Matrix, ct::CoefTable, note)

Print a fitted regression as text: the title, the summary statistics `ctop`
(label, value), the coefficient table `ct`, and `note`.

The output fits `displaysize(io)[2]` when it can: if the coefficient table is
wider, its confidence-interval columns are left out (and the note says so),
and summary statistics that do not fit two to a line are printed one per line.
"""
function print_regression(io::IO, title::AbstractString, ctop::AbstractMatrix,
        ct::CoefTable, note::AbstractString)
    available = displaysize(io)[2]
    io = IOContext(io, :compact => true, :limit => false)
    mat, A, totwidth = coeftable_layout(io, ct)
    omitted = false
    if totwidth > available && any(n -> startswith(n, "Lower"), ct.colnms)
        ct = without_confint(ct)
        mat, A, totwidth = coeftable_layout(io, ct)
        omitted = true
    end
    colnms = ct.colnms

    if supports_color(io)
        print(io,
            lpad(ANSI_YELLOW * title * ANSI_RESET,
                totwidth - 2 + length(ANSI_YELLOW) + length(ANSI_RESET)))
    else
        print(io, lpad(title, totwidth - 2))
    end
    println(io)
    println_horizontal_line(io, totwidth)

    labels = [string(ctop[i, 1], ":") for i in 1:size(ctop, 1)]
    values = [string(ctop[i, 2]) for i in 1:size(ctop, 1)]
    halfwidth = div(totwidth, 2) - 1
    interwidth = 2 + mod(totwidth, 2)
    if all(i -> length(labels[i]) + 1 + length(values[i]) <= halfwidth, eachindex(labels))
        for i in 1:(div(length(labels) - 1, 2) + 1)
            print(io, labels[2i - 1], lpad(values[2i - 1], halfwidth -
                                                           length(labels[2i - 1])))
            if length(labels) >= 2i
                print(io, " "^interwidth)
                print(io, labels[2i], lpad(values[2i], halfwidth - length(labels[2i])))
            end
            println(io)
        end
    else
        for i in eachindex(labels)
            println(io, labels[i], lpad(values[i], totwidth - length(labels[i])))
        end
    end

    println_horizontal_line(io, totwidth)
    print(io, repeat(' ', sum(A[1])))
    for j in 1:length(colnms)
        print(io, "  ", lpad(colnms[j], sum(A[j + 1])))
    end
    println(io)
    println_horizontal_line(io, totwidth)
    for i in 1:size(mat, 1)
        Base.print_matrix_row(io, mat, A, i, 1:size(mat, 2), "  ")
        i != size(mat, 1) && println(io)
    end
    println(io)
    println_horizontal_line(io, totwidth)
    println(io, note)
    omitted && println(io, "Confidence intervals omitted to fit the display width.")
    nothing
end
