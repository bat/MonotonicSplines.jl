# This file is a part of MonotonicSplines.jl, licensed under the MIT License (MIT).

# Shared test data and Zygote reference gradients for the AD backend test
# suites in test/ad_*/.

using MonotonicSplines
using MonotonicSplines: rqs_forward, rqs_inverse
using DelimitedFiles
import Zygote

ad_nn_out = readdlm(joinpath(@__DIR__, "test_outputs", "test_nn_output.txt"))
ad_pX, ad_pY, ad_dYdX = MonotonicSplines.rqs_params_from_nn(vcat(ad_nn_out, reverse(ad_nn_out, dims=1)), 2)
ad_x = let x1 = readdlm(joinpath(@__DIR__, "test_outputs", "x_test.txt"))
    vcat(x1, reverse(x1, dims=2))
end

f_loss_forward(x, pX, pY, dYdX) = sum(sum.(rqs_forward(x, pX, pY, dYdX)))
f_loss_inverse(x, pX, pY, dYdX) = sum(sum.(rqs_inverse(x, pX, pY, dYdX)))

grad_forward_ref = Zygote.gradient(f_loss_forward, ad_x, ad_pX, ad_pY, ad_dYdX)
grad_inverse_ref = Zygote.gradient(f_loss_inverse, ad_x, ad_pX, ad_pY, ad_dYdX)
