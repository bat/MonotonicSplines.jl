# This file is a part of MonotonicSplines.jl, licensed under the MIT License (MIT).

import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(Pkg.PackageSpec(path = joinpath(@__DIR__, "..", "..")))
Pkg.instantiate()

using Test
import Enzyme
import Reactant
using ChangesOfVariables: with_logabsdet_jacobian

include(joinpath(@__DIR__, "..", "ad_common.jl"))

@testset "reactant" begin
    @test Base.get_extension(MonotonicSplines, :MonotonicSplinesReactantExt) isa Module

    Reactant.set_default_backend("cpu")

    local xr = Reactant.to_rarray(ad_x)
    local pXr = Reactant.to_rarray(ad_pX)
    local pYr = Reactant.to_rarray(ad_pY)
    local dYdXr = Reactant.to_rarray(ad_dYdX)

    spline_ladj(x, pX, pY, dYdX) = with_logabsdet_jacobian(RQSpline(pX, pY, dYdX), x)
    local compiled = Reactant.@compile spline_ladj(xr, pXr, pYr, dYdXr)
    local yr, ljr = compiled(xr, pXr, pYr, dYdXr)
    local y_ref, lj_ref = rqs_forward(ad_x, ad_pX, ad_pY, ad_dYdX)
    @test Array(yr) ≈ y_ref
    @test Array(ljr) ≈ lj_ref

    for (f_loss, grad_ref) in ((f_loss_forward, grad_forward_ref), (f_loss_inverse, grad_inverse_ref))
        local grad_compiled = Reactant.@compile Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(f_loss), xr, pXr, pYr, dYdXr)
        local grads = grad_compiled(Enzyme.Reverse, Enzyme.Const(f_loss), xr, pXr, pYr, dYdXr)
        @test all(isapprox.(Array.(grads), grad_ref))
    end
end
