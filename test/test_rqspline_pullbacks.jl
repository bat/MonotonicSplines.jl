# This file is a part of MonotonicSplines.jl, licensed under the MIT License (MIT).

using Adapt
using ChainRulesCore
using ChangesOfVariables
using DelimitedFiles
using HeterogeneousComputing
using KernelAbstractions
using MonotonicSplines
using Test

compute_units = isdefined(Main, :CUDA) ? [AbstractComputeUnit(CUDA.device()), CPUnit()] : [CPUnit()]

for compute_unit in compute_units

    local compute_unit_type = compute_unit isa AbstractGPUnit ? "GPU" : "CPU"

    local rqs_forward_pullback_test = Tuple([reshape(readdlm("test_outputs/rqs_forward_pullback_test.txt")[i,:], 11, 1, 10) for i in 1:3])
  
    local tangent_x_test = adapt(compute_unit, readdlm("test_outputs/t1.txt"))
    local tangent_LogJac_test = adapt(compute_unit, readdlm("test_outputs/t2.txt"))

    local ∂y∂w_test = adapt(compute_unit, reshape(readdlm("test_outputs/dydw.txt"), 11,1,10))
    local ∂y∂h_test = adapt(compute_unit, reshape(readdlm("test_outputs/dydh.txt"), 11,1,10))
    local ∂y∂w_test = adapt(compute_unit, reshape(readdlm("test_outputs/dydd.txt"), 11,1,10))

    local ∂LogJac∂w_test = adapt(compute_unit, reshape(readdlm("test_outputs/dljdw.txt"), 11,1,10))
    local ∂LogJac∂h_test = adapt(compute_unit, reshape(readdlm("test_outputs/dljdh.txt"), 11,1,10))
    local ∂LogJac∂d_test = adapt(compute_unit, reshape(readdlm("test_outputs/dljdd.txt"), 11,1,10))

    @testset "rqs_forward_pullback_$compute_unit_type" begin
        @test all(isapprox.(MonotonicSplines.rqs_forward_pullback(x_test, w,h,d, ones(size(x_test)...), ones(1,size(x_test,2))), rqs_forward_pullback_test))
    end

    @testset "forward_pullback_kernel_$compute_unit_type" begin
        y = zeros(size(x_test)...)
        logjac = zeros(size(x_test)...)
        ∂y∂w = ones(size(w)...)
        ∂y∂h = ones(size(w)...)
        ∂y∂d = ones(size(w)...)
        ∂LogJac∂w = ones(size(w)...)
        ∂LogJac∂h = ones(size(w)...)
        ∂LogJac∂d = ones(size(w)...)
        tangent_x = ones(size(x_test)...)
        tangent_LogJac = ones(size(x_test)...)

        forward_pbk_test = MonotonicSplines.rqs_forward_pullback_kernel!(CPU(),4)
        forward_pbk_test(x_test, y, logjac, w, h, d, ∂y∂w, ∂y∂h, ∂y∂d, ∂LogJac∂w, ∂LogJac∂h, ∂LogJac∂d, tangent_x, tangent_LogJac, ndrange=size(x_test))

        @test isapprox(y, y_test) 
        @test isapprox(logjac, ladj_forward_test)

        @test isapprox(∂y∂w, ∂y∂w_test)    
        @test isapprox(∂y∂h, ∂y∂h_test)
        @test isapprox(∂y∂d, ∂y∂w_test)    

        @test isapprox(∂LogJac∂w, ∂LogJac∂w_test)
        @test isapprox(∂LogJac∂h, ∂LogJac∂h_test)    
        @test isapprox(∂LogJac∂d, ∂LogJac∂d_test)   

        @test isapprox(tangent_x, tangent_x_test) 
        @test isapprox(tangent_LogJac, tangent_LogJac_test)
    end

    @testset "eval_rqs_params_$compute_unit_type" begin
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


end