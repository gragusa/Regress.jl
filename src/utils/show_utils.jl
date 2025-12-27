##############################################################################
##
## Display Formatting Utilities
##
## Shared utilities for text and HTML output of model results.
##
##############################################################################

# ANSI color codes for terminal output
const ANSI_GREEN = "\e[32m"
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
function print_horizontal_line(io::IO, width::Int; char::Char=BOX_HORIZONTAL)
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
function println_horizontal_line(io::IO, width::Int; char::Char=BOX_HORIZONTAL)
    print_horizontal_line(io, width; char=char)
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
function html_table_start(io::IO; class::String="regress-table", caption::String="")
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
function html_thead_start(io::IO; class::String="")
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
function html_tbody_start(io::IO; class::String="")
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
function html_tfoot_start(io::IO; class::String="")
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
function html_row(io::IO, cells::Vector; class::String="", is_header::Bool=false)
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
function format_number(x::Real; digits::Int=4)
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
function vcov_type_name(v::CovarianceMatrices.HAC)
    typename = string(typeof(v).name.name)
    bw = v.bw
    if bw isa CovarianceMatrices.Fixed
        return @sprintf("%s(%d)", typename, Int(bw.bw))
    else
        return @sprintf("%s(auto)", typename)
    end
end
