# This file is a part of MonotonicSplines.jl, licensed under the MIT License (MIT).

using MonotonicSplines
using MonotonicSplines: rqs_forward, rqs_inverse, rqs_apply_purearray, RQSForward, RQSInverse
using Test
using DelimitedFiles
using ChangesOfVariables: with_logabsdet_jacobian

import Zygote
import Mooncake
import Enzyme
import Reactant

ad_nn_out = readdlm("test_outputs/test_nn_output.txt")
ad_pX, ad_pY, ad_dYdX = MonotonicSplines.rqs_params_from_nn(vcat(ad_nn_out, reverse(ad_nn_out, dims=1)), 2)
ad_x = let x1 = readdlm("test_outputs/x_test.txt")
    vcat(x1, reverse(x1, dims=2))
end

f_loss_forward(x, pX, pY, dYdX) = sum(sum.(rqs_forward(x, pX, pY, dYdX)))
f_loss_inverse(x, pX, pY, dYdX) = sum(sum.(rqs_inverse(x, pX, pY, dYdX)))

grad_forward_ref = Zygote.gradient(f_loss_forward, ad_x, ad_pX, ad_pY, ad_dYdX)
grad_inverse_ref = Zygote.gradient(f_loss_inverse, ad_x, ad_pX, ad_pY, ad_dYdX)

@testset "rqs_apply_purearray" begin
    ad_x_wide = 4 .* ad_x
    for (trafo, ref_fun) in ((RQSForward(), rqs_forward), (RQSInverse(), rqs_inverse))
        @test all(isapprox.(rqs_apply_purearray(trafo, ad_x, ad_pX, ad_pY, ad_dYdX), ref_fun(ad_x, ad_pX, ad_pY, ad_dYdX)))
        @test all(isapprox.(rqs_apply_purearray(trafo, ad_x_wide, ad_pX, ad_pY, ad_dYdX), ref_fun(ad_x_wide, ad_pX, ad_pY, ad_dYdX)))
    end
end

@testset "mooncake" begin
    @test Base.get_extension(MonotonicSplines, :MonotonicSplinesMooncakeExt) isa Module

    for (f_loss, grad_ref) in ((f_loss_forward, grad_forward_ref), (f_loss_inverse, grad_inverse_ref))
        local cache = Mooncake.prepare_gradient_cache(f_loss, ad_x, ad_pX, ad_pY, ad_dYdX)
        local val, grads = Mooncake.value_and_gradient!!(cache, f_loss, ad_x, ad_pX, ad_pY, ad_dYdX)
        @test val ≈ f_loss(ad_x, ad_pX, ad_pY, ad_dYdX)
        @test all(isapprox.(grads[2:5], grad_ref))
    end
end

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
