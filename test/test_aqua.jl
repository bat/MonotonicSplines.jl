# This file is a part of MonotonicSplines.jl, licensed under the MIT License (MIT).

import Test
import Aqua
import MonotonicSplines

Test.@testset "Aqua tests" begin
    Aqua.test_all(
        MonotonicSplines,
        ambiguities = true,
        piracies=false # we are currently overloading some KernelAbstractions.jl functions, so we don't want to test for piracy
    )
end # testset
