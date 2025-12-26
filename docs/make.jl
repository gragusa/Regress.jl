using Documenter
using Regress

makedocs(
    sitename = "Regress.jl",
    modules = [Regress],
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "User Guide" => [
            "OLS Estimation" => "ols.md",
            "IV Estimation" => "iv.md",
            "Fixed Effects" => "fixed_effects.md",
            "Variance Estimation" => "variance.md",
        ],
        "API Reference" => "api.md",
    ],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://gragusa.github.io/Regress.jl",
    ),
    warnonly = [:missing_docs],
)

deploydocs(
    repo = "github.com/gragusa/Regress.jl.git",
    devbranch = "master",
    push_preview = true,
)
