# This file is a part of MonotonicSplines.jl, licensed under the MIT License (MIT).

import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(Pkg.PackageSpec(path = joinpath(@__DIR__, "..", "..")))
Pkg.instantiate()

using Test
import Enzyme

include(joinpath(@__DIR__, "..", "ad_common.jl"))

@testset "enzyme" begin
    for (f_loss, grad_ref) in ((f_loss_forward, grad_forward_ref), (f_loss_inverse, grad_inverse_ref))
        local δx, δpX, δpY, δdYdX = zero(ad_x), zero(ad_pX), zero(ad_pY), zero(ad_dYdX)
        Enzyme.autodiff(Enzyme.Reverse, f_loss,
            Enzyme.Duplicated(ad_x, δx), Enzyme.Duplicated(ad_pX, δpX),
            Enzyme.Duplicated(ad_pY, δpY), Enzyme.Duplicated(ad_dYdX, δdYdX)
        )
        @test all(isapprox.((δx, δpX, δpY, δdYdX), grad_ref))
    end
end
