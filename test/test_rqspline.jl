# This file is a part of MonotonicSplines.jl, licensed under the MIT License (MIT).

using MonotonicSplines
using Test


@testset "monotonic_spline" begin

    n_dims_to_transform = 5
    n_smpls = 40

    x_test = ones(n_dims_to_transform, n_smpls)

    RQS_test = RQSpline(test_params_processed...)


    
end
