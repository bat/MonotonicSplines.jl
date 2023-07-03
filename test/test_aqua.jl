# This file is a part of MonotonicSplines.jl, licensed under the MIT License (MIT).

import Test
import Aqua
import MonotonicSplines

Test.@testset "Aqua tests" begin
    Aqua.test_all(
        MonotonicSplines,
        ambiguities = false,
        project_toml_formatting = VERSION≥v"1.7"
    )
end # testset
