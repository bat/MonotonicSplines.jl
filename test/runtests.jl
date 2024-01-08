# This file is a part of MonotonicSplines.jl, licensed under the MIT License (MIT).

import Test

Test.@testset "Package MonotonicSplines" begin
    include("test_aqua.jl")
    include("test_utils.jl")
    include("test_rqspline.jl")
    include("test_rqspline_pullbacks.jl")
    include("test_docs.jl")
    isempty(Test.detect_ambiguities(MonotonicSplines))
end # testset
