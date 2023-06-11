# Use
#
#     DOCUMENTER_DEBUG=true julia --color=yes make.jl local [nonstrict] [fixdoctests]
#
# for local builds.

using Documenter
using MonotonicSplines

# Doctest setup
DocMeta.setdocmeta!(
    MonotonicSplines,
    :DocTestSetup,
    :(using MonotonicSplines);
    recursive=true,
)

makedocs(
    sitename = "MonotonicSplines",
    modules = [MonotonicSplines],
    format = Documenter.HTML(
        prettyurls = !("local" in ARGS),
        canonical = "https://bat.github.io/MonotonicSplines.jl/stable/"
    ),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
        "LICENSE" => "LICENSE.md",
    ],
    doctest = ("fixdoctests" in ARGS) ? :fix : true,
    linkcheck = !("nonstrict" in ARGS),
    strict = !("nonstrict" in ARGS),
)

deploydocs(
    repo = "github.com/bat/MonotonicSplines.jl.git",
    forcepush = true,
    push_preview = true,
)
