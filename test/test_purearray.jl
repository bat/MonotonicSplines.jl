# This file is a part of MonotonicSplines.jl, licensed under the MIT License (MIT).

using MonotonicSplines
using MonotonicSplines: rqs_forward, rqs_inverse, rqs_apply_purearray, RQSForward, RQSInverse
using Test
using DelimitedFiles

@testset "rqs_apply_purearray" begin
    local nn_out = readdlm("test_outputs/test_nn_output.txt")
    local pX, pY, dYdX = MonotonicSplines.rqs_params_from_nn(vcat(nn_out, reverse(nn_out, dims=1)), 2)
    local x = let x1 = readdlm("test_outputs/x_test.txt")
        vcat(x1, reverse(x1, dims=2))
    end
    local x_wide = 4 .* x

    for (trafo, ref_fun) in ((RQSForward(), rqs_forward), (RQSInverse(), rqs_inverse))
        @test all(isapprox.(rqs_apply_purearray(trafo, x, pX, pY, dYdX), ref_fun(x, pX, pY, dYdX)))
        @test all(isapprox.(rqs_apply_purearray(trafo, x_wide, pX, pY, dYdX), ref_fun(x_wide, pX, pY, dYdX)))
    end
end
