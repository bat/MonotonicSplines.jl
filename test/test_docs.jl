# This file is a part of MonotonicSplines.jl, licensed under the MIT License (MIT).

using Test
using MonotonicSplines
import Documenter

Documenter.DocMeta.setdocmeta!(
    MonotonicSplines,
    :DocTestSetup,
    :(using MonotonicSplines);
    recursive=true,
)
Documenter.doctest(MonotonicSplines)
