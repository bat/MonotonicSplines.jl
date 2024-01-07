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

    local test_params_processed_unshaped = adapt(compute_unit, readdlm("test_outputs/test_params_processed.txt"))
    local test_params_processed = adapt(compute_unit, Tuple([reshape(test_params_processed_unshaped[i,:], 11,1,10) for i in 1:3]))

    local w = test_params_processed[1]
    local h = test_params_processed[2]
    local d = test_params_processed[3]

    local x_test = adapt(compute_unit, readdlm("test_outputs/x_test.txt"))
    local y_test = adapt(compute_unit, readdlm("test_outputs/y_test.txt"))

    local ladj_forward_test = adapt(compute_unit, readdlm("test_outputs/ladj_forward_test.txt"))
    local ladj_backward_test = adapt(compute_unit, readdlm("test_outputs/ladj_backward_test.txt"))

    local rqs_forward_pullback_test = Tuple([reshape(readdlm("test_outputs/rqs_forward_pullback_test.txt")[i,:], 11, 1, 10) for i in 1:3])
  
    local tangent_x_forw_test = adapt(compute_unit, readdlm("test_outputs/t1_forw.txt"))
    local tangent_LogJac_forw_test = adapt(compute_unit, readdlm("test_outputs/t2_forw.txt"))

    local ∂y∂w_forw_test = adapt(compute_unit, reshape(readdlm("test_outputs/dydw_forw.txt"), 11,1,10))
    local ∂y∂h_forw_test = adapt(compute_unit, reshape(readdlm("test_outputs/dydh_forw.txt"), 11,1,10))
    local ∂y∂d_forw_test = adapt(compute_unit, reshape(readdlm("test_outputs/dydd_forw.txt"), 11,1,10))

    local ∂LogJac∂w_forw_test = adapt(compute_unit, reshape(readdlm("test_outputs/dljdw_forw.txt"), 11,1,10))
    local ∂LogJac∂h_forw_test = adapt(compute_unit, reshape(readdlm("test_outputs/dljdh_forw.txt"), 11,1,10))
    local ∂LogJac∂d_forw_test = adapt(compute_unit, reshape(readdlm("test_outputs/dljdd_forw.txt"), 11,1,10))

    local tangent_x_backw_test = adapt(compute_unit, readdlm("test_outputs/t1_backw.txt"))
    local tangent_LogJac_backw_test = adapt(compute_unit, readdlm("test_outputs/t2_backw.txt"))

    local ∂y∂w_backw_test = adapt(compute_unit, reshape(readdlm("test_outputs/dydw_backw.txt"), 11,1,10))
    local ∂y∂h_backw_test = adapt(compute_unit, reshape(readdlm("test_outputs/dydh_backw.txt"), 11,1,10))
    local ∂y∂d_backw_test = adapt(compute_unit, reshape(readdlm("test_outputs/dydd_backw.txt"), 11,1,10))

    local ∂LogJac∂w_backw_test = adapt(compute_unit, reshape(readdlm("test_outputs/dljdw_backw.txt"), 11,1,10))
    local ∂LogJac∂h_backw_test = adapt(compute_unit, reshape(readdlm("test_outputs/dljdh_backw.txt"), 11,1,10))
    local ∂LogJac∂d_backw_test = adapt(compute_unit, reshape(readdlm("test_outputs/dljdd_backw.txt"), 11,1,10))

    @testset "rqs_forward_pullback_$compute_unit_type" begin
        @test all(isapprox.(MonotonicSplines.rqs_pullback(MonotonicSplines.eval_forward_rqs_params_with_grad, x_test, w,h,d, ones(size(x_test)...), ones(1,size(x_test,2))), rqs_forward_pullback_test))
    end

    @testset "forward_pullback_kernel_$compute_unit_type" begin
        y = zeros(size(x_test)...)
        logjac = zeros(size(x_test)...)
        ∂y∂w_forw = ones(size(w)...)
        ∂y∂h_forw = ones(size(w)...)
        ∂y∂d_forw = ones(size(w)...)
        ∂LogJac∂w_forw = ones(size(w)...)
        ∂LogJac∂h_forw = ones(size(w)...)
        ∂LogJac∂d_forw = ones(size(w)...)
        tangent_x_forw = ones(size(x_test)...)
        tangent_LogJac_forw = ones(size(x_test)...)

        forward_pbk_test = MonotonicSplines.rqs_pullback_kernel!(CPU(),4)
        forward_pbk_test(MonotonicSplines.eval_forward_rqs_params_with_grad, x_test, y, logjac, w, h, d, ∂y∂w_forw, ∂y∂h_forw, ∂y∂d_forw, ∂LogJac∂w_forw, ∂LogJac∂h_forw, ∂LogJac∂d_forw, tangent_x_forw, tangent_LogJac_forw, ndrange=size(x_test))

        @test isapprox(y, y_test) 
        @test isapprox(logjac, ladj_forward_test)

        @test isapprox(∂y∂w_forw, ∂y∂w_forw_test)    
        @test isapprox(∂y∂h_forw, ∂y∂h_forw_test)
        @test isapprox(∂y∂d_forw, ∂y∂d_forw_test)    

        @test isapprox(∂LogJac∂w_forw, ∂LogJac∂w_forw_test)
        @test isapprox(∂LogJac∂h_forw, ∂LogJac∂h_forw_test)    
        @test isapprox(∂LogJac∂d_forw, ∂LogJac∂d_forw_test)   

        @test isapprox(tangent_x_forw, tangent_x_forw_test) 
        @test isapprox(tangent_LogJac_forw, tangent_LogJac_forw_test)
    end

    @testset "eval_forward_rqs_params_$compute_unit_type" begin
        @test all(isapprox.(MonotonicSplines.eval_forward_rqs_params(w[1,1,1], w[2,1,1], h[1,1,1], h[2,1,1], h[1,1,1], h[2,1,1], x_test[1,1]), (-4.905420651500841, -7.711169845398972)))

        @test  all(isapprox.(MonotonicSplines.eval_forward_rqs_params_with_grad(w[1,1,1], w[2,1,1], h[1,1,1], h[2,1,1], h[1,1,1], h[2,1,1], y_test[1,1])[1:2], (-4.905034319051312, -7.137238177160876)))
        @test  all(isapprox.(MonotonicSplines.eval_forward_rqs_params_with_grad(w[1,1,1], w[2,1,1], h[1,1,1], h[2,1,1], h[1,1,1], h[2,1,1], y_test[1,1])[3], (-0.0008667347881545143, 0.0016616793516759222)))
        @test  all(isapprox.(MonotonicSplines.eval_forward_rqs_params_with_grad(w[1,1,1], w[2,1,1], h[1,1,1], h[2,1,1], h[1,1,1], h[2,1,1], y_test[1,1])[4], (0.4767412102473232, 0.5232587897526768)))
        @test  all(isapprox.(MonotonicSplines.eval_forward_rqs_params_with_grad(w[1,1,1], w[2,1,1], h[1,1,1], h[2,1,1], h[1,1,1], h[2,1,1], y_test[1,1])[5], (-0.007999949292501972, 0.008554077697712834)))
        @test  all(isapprox.(MonotonicSplines.eval_forward_rqs_params_with_grad(w[1,1,1], w[2,1,1], h[1,1,1], h[2,1,1], h[1,1,1], h[2,1,1], y_test[1,1])[6], (0.028636334897900184, 0.9994232883327587)))
        @test  all(isapprox.(MonotonicSplines.eval_forward_rqs_params_with_grad(w[1,1,1], w[2,1,1], h[1,1,1], h[2,1,1], h[1,1,1], h[2,1,1], y_test[1,1])[7], (-10.242083538900392, 10.242083538900392)))
        @test  all(isapprox.(MonotonicSplines.eval_forward_rqs_params_with_grad(w[1,1,1], w[2,1,1], h[1,1,1], h[2,1,1], h[1,1,1], h[2,1,1], y_test[1,1])[8], (0.10285870789209196, 0.0764067679810318)))
    end

    @testset "backward_pullback_kernel_$compute_unit_type" begin
        y = zeros(size(x_test)...)
        logjac = zeros(size(x_test)...)
        ∂y∂w_backw = ones(size(w)...)
        ∂y∂h_backw = ones(size(w)...)
        ∂y∂d_backw = ones(size(w)...)
        ∂LogJac∂w_backw = ones(size(w)...)
        ∂LogJac∂h_backw = ones(size(w)...)
        ∂LogJac∂d_backw = ones(size(w)...)
        tangent_x_backw = ones(size(x_test)...)
        tangent_LogJac_backw = ones(size(x_test)...)

        backward_pbk_test = MonotonicSplines.rqs_pullback_kernel!(CPU(),4)
        backward_pbk_test(MonotonicSplines.eval_backward_rqs_params_with_grad, y_test, y, logjac, w, h, d, ∂y∂w_backw, ∂y∂h_backw, ∂y∂d_backw, ∂LogJac∂w_backw, ∂LogJac∂h_backw, ∂LogJac∂d_backw, tangent_x_backw, tangent_LogJac_backw, ndrange=size(x_test))

        @test isapprox(y, x_test) 
        @test isapprox(logjac, ladj_backward_test)

        @test isapprox(∂y∂w_backw, ∂y∂w_backw_test)    
        @test isapprox(∂y∂h_backw, ∂y∂h_backw_test)
        @test isapprox(∂y∂d_backw, ∂y∂d_backw_test)    

        @test isapprox(∂LogJac∂w_backw, ∂LogJac∂w_backw_test)
        @test isapprox(∂LogJac∂h_backw, ∂LogJac∂h_backw_test)    
        @test isapprox(∂LogJac∂d_backw, ∂LogJac∂d_backw_test)   

        @test isapprox(tangent_x_backw, tangent_x_backw_test) 
        @test isapprox(tangent_LogJac_backw, tangent_LogJac_backw_test)
    end

    @testset "eval_backward_rqs_params_$compute_unit_type" begin
        @test all(isapprox.(MonotonicSplines.eval_backward_rqs_params(w[1,1,1], w[2,1,1], h[1,1,1], h[2,1,1], h[1,1,1], h[2,1,1], y_test[1,1]), (-4.981580571322357, -7.878864476215551)))

        @test  all(isapprox.(MonotonicSplines.eval_backward_rqs_params_with_grad(w[1,1,1], w[2,1,1], h[1,1,1], h[2,1,1], h[1,1,1], h[2,1,1], y_test[1,1])[1:2], (-4.981580571322357, -7.878864476215551)))
        @test  all(isapprox.(MonotonicSplines.eval_backward_rqs_params_with_grad(w[1,1,1], w[2,1,1], h[1,1,1], h[2,1,1], h[1,1,1], h[2,1,1], y_test[1,1])[3], (0.9964451502079126, 0.0035548497920874222)))
        @test  all(isapprox.(MonotonicSplines.eval_backward_rqs_params_with_grad(w[1,1,1], w[2,1,1], h[1,1,1], h[2,1,1], h[1,1,1], h[2,1,1], y_test[1,1])[4], (-0.09860802071607391, 0.09898668350943046)))
        @test  all(isapprox.(MonotonicSplines.eval_backward_rqs_params_with_grad(w[1,1,1], w[2,1,1], h[1,1,1], h[2,1,1], h[1,1,1], h[2,1,1], y_test[1,1])[5], (0.001692608637565046, 0.001853942717018299)))
        @test  all(isapprox.(MonotonicSplines.eval_backward_rqs_params_with_grad(w[1,1,1], w[2,1,1], h[1,1,1], h[2,1,1], h[1,1,1], h[2,1,1], y_test[1,1])[6], (-0.4170286379079101, 0.4170286379079101)))
        @test  all(isapprox.(MonotonicSplines.eval_backward_rqs_params_with_grad(w[1,1,1], w[2,1,1], h[1,1,1], h[2,1,1], h[1,1,1], h[2,1,1], y_test[1,1])[7], (-9.709051434667053, 10.696997710661178)))
        @test  all(isapprox.(MonotonicSplines.eval_backward_rqs_params_with_grad(w[1,1,1], w[2,1,1], h[1,1,1], h[2,1,1], h[1,1,1], h[2,1,1], y_test[1,1])[8], (-0.01798457442421056, 0.2012775903613969)))
    end
end