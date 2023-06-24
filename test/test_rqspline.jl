# This file is a part of MonotonicSplines.jl, licensed under the MIT License (MIT).

using ChainRulesCore
using ChangesOfVariables
using DelimitedFiles
using KernelAbstractions
using MonotonicSplines
using Test

test_params_processed_unshaped = readdlm("test_outputs/test_params_processed.txt")
test_params_processed = Tuple([reshape(test_params_processed_unshaped[i,:], 11,1,10) for i in 1:3])

w = test_params_processed[1]
h = test_params_processed[2]
d = test_params_processed[3]

x_test = readdlm("test_outputs/x_test.txt")
y_test = readdlm("test_outputs/y_test.txt")

ladj_forward_test = readdlm("test_outputs/ladj_forward_test.txt")
ladj_backward_test = readdlm("test_outputs/ladj_backward_test.txt")

RQS_test = RQSpline(test_params_processed...)
RQS_inv_test = RQSplineInv(test_params_processed...)

dydw_test = reshape(readdlm("test_outputs/dydw.txt"), 11,1,10)
dydh_test = reshape(readdlm("test_outputs/dydh.txt"), 11,1,10)
dydd_test = reshape(readdlm("test_outputs/dydd.txt"), 11,1,10)

dljdw_test = reshape(readdlm("test_outputs/dljdw.txt"), 11,1,10)
dljdh_test = reshape(readdlm("test_outputs/dljdh.txt"), 11,1,10)
dljdd_test = reshape(readdlm("test_outputs/dljdd.txt"), 11,1,10)

t1_test = readdlm("test_outputs/t1.txt")
t2_test = readdlm("test_outputs/t2.txt")

@testset "rqs_structs" begin
    @test RQS_test isa RQSpline && isapprox(RQS_test.widths, w) && isapprox(RQS_test.heights, h) && isapprox(RQS_test.derivatives, d)
    @test RQS_inv_test isa RQSplineInv && isapprox(RQS_inv_test.widths, w) && isapprox(RQS_inv_test.heights, h) && isapprox(RQS_inv_test.derivatives, d)
end

@testset "rqs_high_lvl_applications" begin
    @test isapprox(RQS_test(x_test), y_test)
    @test isapprox(RQS_inv_test(y_test), x_test)
end

@testset "with_logabsdet_jacobian" begin
    @test all(isapprox.(ChangesOfVariables.with_logabsdet_jacobian(RQS_test,x_test), (y_test, ladj_forward_test)))
    @test all(isapprox.(ChangesOfVariables.with_logabsdet_jacobian(RQS_inv_test,y_test), (x_test, ladj_backward_test)))
end

@testset "rqs_low_lvl_applications" begin
    @test all(isapprox.(MonotonicSplines.spline_forward(RQS_test, x_test), (y_test, ladj_forward_test)))
    @test all(isapprox.(MonotonicSplines.spline_backward(RQS_inv_test, y_test), (x_test, ladj_backward_test)))
    @test all(isapprox.(MonotonicSplines.rqs_forward(x_test, w, h, d, w, h, d), (y_test, ladj_forward_test)))
    @test all(isapprox.(MonotonicSplines.rqs_backward(y_test, w, h, d), (x_test, ladj_backward_test)))
end

@testset "rqs_forward_pullback" begin
    @test MonotonicSplines.rqs_forward_pullback(x_test, w,h,d, w,h,d, zeros(size(x_test)...), zeros(1,size(x_test,2))) isa Tuple{ChainRulesCore.NoTangent, ChainRulesCore.Thunk{MonotonicSplines.var"#5#6"{Matrix{Float64}}}, Array{Float64, 3}, Array{Float64, 3}, Array{Float64, 3}, Array{Float64, 3}, Array{Float64, 3}, Array{Float64, 3}}
end

@testset "rqs_kernels" begin
    forward_kernel_test = MonotonicSplines.rqs_forward_kernel!(CPU(), 1)
    y_kernel_test = zeros(size(x_test)...)
    ladj_forward_kernel_test = zeros(size(x_test)...)
    forward_kernel_test(x_test, y_kernel_test, ladj_forward_kernel_test, w,h,d, ndrange=size(x_test))
    @test isapprox(y_kernel_test, y_test)
    @test isapprox(sum(ladj_forward_kernel_test, dims = 1), ladj_forward_test)

    backward_kernel_test = MonotonicSplines.rqs_backward_kernel!(CPU(), 1)
    x_kernel_test = zeros(size(x_test)...)
    ladj_backward_kernel_test = zeros(size(x_test)...)
    backward_kernel_test(y_test, x_kernel_test, ladj_backward_kernel_test, w,h,d, ndrange=size(x_test))

    @test isapprox(x_kernel_test, x_test)
    @test isapprox(sum(ladj_backward_kernel_test, dims = 1), ladj_backward_test)
end

@testset "forward_pullback_kernel" begin
    y = zeros(size(x_test)...)
    logjac = zeros(size(x_test)...)
    dydw = ones(size(w)...)
    dydh = ones(size(w)...)
    dydd = ones(size(w)...)
    dljdw = ones(size(w)...)
    dljdh = ones(size(w)...)
    dljdd = ones(size(w)...)
    t1 = ones(size(x_test)...)
    t2 = ones(size(x_test)...)

    forward_pbk_test = MonotonicSplines.rqs_forward_pullback_kernel!(CPU(),1)
    forward_pbk_test(x_test, y, logjac, w, h, d, dydw, dydh, dydd, dljdw, dljdh, dljdd, t1, t2, ndrange=size(x_test))

    @test isapprox(y, y_test) 
    @test isapprox(logjac, ladj_forward_test)

    @test isapprox(dydw, dydw_test)    
    @test isapprox(dydh, dydh_test)
    @test isapprox(dydd, dydd_test)    

    @test isapprox(dljdw, dljdw_test)
    @test isapprox(dljdh, dljdh_test)    
    @test isapprox(dljdd, dljdd_test)   

    @test isapprox(t1, t1_test) 
    @test isapprox(t2, t2_test)
end

@testset "eval_rqs_params" begin
    @test all(isapprox.(MonotonicSplines.eval_forward_rqs_params(w[1,1,1], w[2,1,1], h[1,1,1], h[2,1,1], h[1,1,1], h[2,1,1], x_test[1,1]), (-4.905420651500841, -7.711169845398972)))
    @test all(isapprox.(MonotonicSplines.eval_backward_rqs_params(w[1,1,1], w[2,1,1], h[1,1,1], h[2,1,1], h[1,1,1], h[2,1,1], y_test[1,1]), (-4.981580571322357, -7.878864476215551)))

    @test  all(isapprox.(MonotonicSplines.eval_forward_rqs_params_with_grad(w[1,1,1], w[2,1,1], h[1,1,1], h[2,1,1], h[1,1,1], h[2,1,1], y_test[1,1])[1:2], (-4.905034319051312, -7.137238177160876)))
    @test  all(isapprox.(MonotonicSplines.eval_forward_rqs_params_with_grad(w[1,1,1], w[2,1,1], h[1,1,1], h[2,1,1], h[1,1,1], h[2,1,1], y_test[1,1])[3], (-0.0008667347881545143, 0.0016616793516759222)))
    @test  all(isapprox.(MonotonicSplines.eval_forward_rqs_params_with_grad(w[1,1,1], w[2,1,1], h[1,1,1], h[2,1,1], h[1,1,1], h[2,1,1], y_test[1,1])[4], (0.4767412102473232, 0.5232587897526768)))
    @test  all(isapprox.(MonotonicSplines.eval_forward_rqs_params_with_grad(w[1,1,1], w[2,1,1], h[1,1,1], h[2,1,1], h[1,1,1], h[2,1,1], y_test[1,1])[5], (-0.007999949292501972, 0.008554077697712834)))
    @test  all(isapprox.(MonotonicSplines.eval_forward_rqs_params_with_grad(w[1,1,1], w[2,1,1], h[1,1,1], h[2,1,1], h[1,1,1], h[2,1,1], y_test[1,1])[6], (0.028636334897900184, 0.9994232883327587)))
    @test  all(isapprox.(MonotonicSplines.eval_forward_rqs_params_with_grad(w[1,1,1], w[2,1,1], h[1,1,1], h[2,1,1], h[1,1,1], h[2,1,1], y_test[1,1])[7], (-10.242083538900392, 10.242083538900392)))
    @test  all(isapprox.(MonotonicSplines.eval_forward_rqs_params_with_grad(w[1,1,1], w[2,1,1], h[1,1,1], h[2,1,1], h[1,1,1], h[2,1,1], y_test[1,1])[8], (0.10285870789209196, 0.0764067679810318)))
end
