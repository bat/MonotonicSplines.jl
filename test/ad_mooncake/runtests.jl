# This file is a part of MonotonicSplines.jl, licensed under the MIT License (MIT).

import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(Pkg.PackageSpec(path = joinpath(@__DIR__, "..", "..")))
Pkg.instantiate()

using Test
import Mooncake

include(joinpath(@__DIR__, "..", "ad_common.jl"))

@testset "mooncake" begin
    @test Base.get_extension(MonotonicSplines, :MonotonicSplinesMooncakeExt) isa Module

    for (f_loss, grad_ref) in ((f_loss_forward, grad_forward_ref), (f_loss_inverse, grad_inverse_ref))
        local cache = Mooncake.prepare_gradient_cache(f_loss, ad_x, ad_pX, ad_pY, ad_dYdX)
        local val, grads = Mooncake.value_and_gradient!!(cache, f_loss, ad_x, ad_pX, ad_pY, ad_dYdX)
        @test val ≈ f_loss(ad_x, ad_pX, ad_pY, ad_dYdX)
        @test all(isapprox.(grads[2:5], grad_ref))
    end
end
