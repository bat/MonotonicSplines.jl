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

function _fd_grad(f, x::AbstractArray)
    g = zero(float.(x))
    for i in eachindex(x)
        h = 1e-5 * max(one(abs(x[i])), abs(x[i]))
        xp = copy(x); xp[i] += h
        xm = copy(x); xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2*h)
    end
    return g
end

for compute_unit in compute_units

    local compute_unit_type = compute_unit isa AbstractGPUnit ? "GPU" : "CPU"

    local test_params_processed_unshaped = adapt(compute_unit, readdlm("test_outputs/test_params_processed.txt"))
    local test_params_processed = adapt(compute_unit, Tuple([reshape(test_params_processed_unshaped[i,:], 11,1,10) for i in 1:3]))

    local pX = test_params_processed[1]
    local pY = test_params_processed[2]
    local dYdX = test_params_processed[3]

    local x_test = adapt(compute_unit, readdlm("test_outputs/x_test.txt"))
    local y_test = adapt(compute_unit, readdlm("test_outputs/y_test.txt"))

    local ladj_forward_test = adapt(compute_unit, readdlm("test_outputs/ladj_forward_test.txt"))
    local ladj_inverse_test = adapt(compute_unit, readdlm("test_outputs/ladj_inverse_test.txt"))

    local rqs_forward_pullback_test = Tuple([reshape(readdlm("test_outputs/rqs_forward_pullback_test.txt")[i,:], 11, 1, 10) for i in 1:3])
  
    local tangent_x_forw_test = adapt(compute_unit, readdlm("test_outputs/t1_forw.txt"))
    local tangent_LogJac_forw_test = adapt(compute_unit, readdlm("test_outputs/t2_forw.txt"))

    local ∂y∂pX_forw_test = adapt(compute_unit, reshape(readdlm("test_outputs/dydw_forw.txt"), 11,1,10))
    local ∂y∂pY_forw_test = adapt(compute_unit, reshape(readdlm("test_outputs/dydh_forw.txt"), 11,1,10))
    local ∂y∂dYdX_forw_test = adapt(compute_unit, reshape(readdlm("test_outputs/dydd_forw.txt"), 11,1,10))

    local ∂LogJac∂pX_forw_test = adapt(compute_unit, reshape(readdlm("test_outputs/dljdw_forw.txt"), 11,1,10))
    local ∂LogJac∂pY_forw_test = adapt(compute_unit, reshape(readdlm("test_outputs/dljdh_forw.txt"), 11,1,10))
    local ∂LogJac∂dYdX_forw_test = adapt(compute_unit, reshape(readdlm("test_outputs/dljdd_forw.txt"), 11,1,10))

    local tangent_x_backw_test = adapt(compute_unit, readdlm("test_outputs/t1_backw.txt"))
    local tangent_LogJac_backw_test = adapt(compute_unit, readdlm("test_outputs/t2_backw.txt"))

    local ∂y∂pX_backw_test = adapt(compute_unit, reshape(readdlm("test_outputs/dydw_backw.txt"), 11,1,10))
    local ∂y∂pY_backw_test = adapt(compute_unit, reshape(readdlm("test_outputs/dydh_backw.txt"), 11,1,10))
    local ∂y∂dYdX_backw_test = adapt(compute_unit, reshape(readdlm("test_outputs/dydd_backw.txt"), 11,1,10))

    local ∂LogJac∂pX_backw_test = adapt(compute_unit, reshape(readdlm("test_outputs/dljdw_backw.txt"), 11,1,10))
    local ∂LogJac∂pY_backw_test = adapt(compute_unit, reshape(readdlm("test_outputs/dljdh_backw.txt"), 11,1,10))
    local ∂LogJac∂dYdX_backw_test = adapt(compute_unit, reshape(readdlm("test_outputs/dljdd_backw.txt"), 11,1,10))

    @testset "rqs_forward_pullback_$compute_unit_type" begin
        local δx, δpX, δpY, δdYdX = MonotonicSplines.rqs_pullback(MonotonicSplines.RQSForward(), x_test, pX,pY,dYdX, ones(size(x_test)...), ones(1,size(x_test,2)))
        @test all(isapprox.((δpX, δpY, δdYdX), rqs_forward_pullback_test))
        @test size(δx) == size(x_test)
    end

    @testset "forward_pullback_kernel_$compute_unit_type" begin
        y = zeros(size(x_test)...)
        logjac = zeros(size(x_test)...)
        ∂y∂pX_forw = ones(size(pX)...)
        ∂y∂pY_forw = ones(size(pX)...)
        ∂y∂dYdX_forw = ones(size(pX)...)
        ∂LogJac∂pX_forw = ones(size(pX)...)
        ∂LogJac∂pY_forw = ones(size(pX)...)
        ∂LogJac∂dYdX_forw = ones(size(pX)...)
        tangent_x_forw = ones(size(x_test)...)
        tangent_LogJac_forw = ones(size(x_test)...)
        δx_forw = zeros(size(x_test)...)

        forward_pbk_test = MonotonicSplines.rqs_pullback_kernel!(CPU(),4)
        forward_pbk_test(MonotonicSplines.RQSForward(), x_test, y, logjac, δx_forw, pX, pY, dYdX, ∂y∂pX_forw, ∂y∂pY_forw, ∂y∂dYdX_forw, ∂LogJac∂pX_forw, ∂LogJac∂pY_forw, ∂LogJac∂dYdX_forw, tangent_x_forw, tangent_LogJac_forw, ndrange=size(x_test))

        @test isapprox(y, y_test) 
        @test isapprox(logjac, ladj_forward_test)

        @test isapprox(∂y∂pX_forw, ∂y∂pX_forw_test)    
        @test isapprox(∂y∂pY_forw, ∂y∂pY_forw_test)
        @test isapprox(∂y∂dYdX_forw, ∂y∂dYdX_forw_test)    

        @test isapprox(∂LogJac∂pX_forw, ∂LogJac∂pX_forw_test)
        @test isapprox(∂LogJac∂pY_forw, ∂LogJac∂pY_forw_test)    
        @test isapprox(∂LogJac∂dYdX_forw, ∂LogJac∂dYdX_forw_test)   

        @test isapprox(tangent_x_forw, tangent_x_forw_test) 
        @test isapprox(tangent_LogJac_forw, tangent_LogJac_forw_test)
    end

    @testset "eval_forward_rqs_params_$compute_unit_type" begin
        @test all(isapprox.(MonotonicSplines.eval_forward_rqs_params(pX[1,1,1], pX[2,1,1], pY[1,1,1], pY[2,1,1], pY[1,1,1], pY[2,1,1], x_test[1,1]), (-4.905420651500841, -7.711169845398972)))

        @test  all(isapprox.(MonotonicSplines.eval_forward_rqs_params_with_grad(pX[1,1,1], pX[2,1,1], pY[1,1,1], pY[2,1,1], pY[1,1,1], pY[2,1,1], y_test[1,1])[1:2], (-4.905034319051312, -7.137238177160876)))
        @test  all(isapprox.(MonotonicSplines.eval_forward_rqs_params_with_grad(pX[1,1,1], pX[2,1,1], pY[1,1,1], pY[2,1,1], pY[1,1,1], pY[2,1,1], y_test[1,1])[3], (-0.0008667347881545143, 0.0016616793516759222)))
        @test  all(isapprox.(MonotonicSplines.eval_forward_rqs_params_with_grad(pX[1,1,1], pX[2,1,1], pY[1,1,1], pY[2,1,1], pY[1,1,1], pY[2,1,1], y_test[1,1])[4], (0.4767412102473232, 0.5232587897526768)))
        @test  all(isapprox.(MonotonicSplines.eval_forward_rqs_params_with_grad(pX[1,1,1], pX[2,1,1], pY[1,1,1], pY[2,1,1], pY[1,1,1], pY[2,1,1], y_test[1,1])[5], (-0.007999949292501972, 0.008554077697712834)))
        @test  all(isapprox.(MonotonicSplines.eval_forward_rqs_params_with_grad(pX[1,1,1], pX[2,1,1], pY[1,1,1], pY[2,1,1], pY[1,1,1], pY[2,1,1], y_test[1,1])[6], (0.028636334897900184, 0.9994232883327587)))
        @test  all(isapprox.(MonotonicSplines.eval_forward_rqs_params_with_grad(pX[1,1,1], pX[2,1,1], pY[1,1,1], pY[2,1,1], pY[1,1,1], pY[2,1,1], y_test[1,1])[7], (-10.242083538900392, 10.242083538900392)))
        @test  all(isapprox.(MonotonicSplines.eval_forward_rqs_params_with_grad(pX[1,1,1], pX[2,1,1], pY[1,1,1], pY[2,1,1], pY[1,1,1], pY[2,1,1], y_test[1,1])[8], (0.10285870789209196, 0.0764067679810318)))
    end

    @testset "inverse_pullback_kernel_$compute_unit_type" begin
        y = zeros(size(x_test)...)
        logjac = zeros(size(x_test)...)
        ∂y∂pX_backw = ones(size(pX)...)
        ∂y∂pY_backw = ones(size(pX)...)
        ∂y∂dYdX_backw = ones(size(pX)...)
        ∂LogJac∂pX_backw = ones(size(pX)...)
        ∂LogJac∂pY_backw = ones(size(pX)...)
        ∂LogJac∂dYdX_backw = ones(size(pX)...)
        tangent_x_backw = ones(size(x_test)...)
        tangent_LogJac_backw = ones(size(x_test)...)
        δx_backw = zeros(size(x_test)...)

        inverse_pbk_test = MonotonicSplines.rqs_pullback_kernel!(CPU(),4)
        inverse_pbk_test(MonotonicSplines.RQSInverse(), y_test, y, logjac, δx_backw, pX, pY, dYdX, ∂y∂pX_backw, ∂y∂pY_backw, ∂y∂dYdX_backw, ∂LogJac∂pX_backw, ∂LogJac∂pY_backw, ∂LogJac∂dYdX_backw, tangent_x_backw, tangent_LogJac_backw, ndrange=size(x_test))

        @test isapprox(y, x_test) 
        @test isapprox(logjac, ladj_inverse_test)

        @test isapprox(∂y∂pX_backw, ∂y∂pX_backw_test)    
        @test isapprox(∂y∂pY_backw, ∂y∂pY_backw_test)
        @test isapprox(∂y∂dYdX_backw, ∂y∂dYdX_backw_test)    

        @test isapprox(∂LogJac∂pX_backw, ∂LogJac∂pX_backw_test)
        @test isapprox(∂LogJac∂pY_backw, ∂LogJac∂pY_backw_test)    
        @test isapprox(∂LogJac∂dYdX_backw, ∂LogJac∂dYdX_backw_test)   

        @test isapprox(tangent_x_backw, tangent_x_backw_test) 
        @test isapprox(tangent_LogJac_backw, tangent_LogJac_backw_test)
    end

    @testset "eval_inverse_rqs_params_$compute_unit_type" begin
        @test all(isapprox.(MonotonicSplines.eval_inverse_rqs_params(pX[1,1,1], pX[2,1,1], pY[1,1,1], pY[2,1,1], pY[1,1,1], pY[2,1,1], y_test[1,1]), (-4.981580571322357, -7.878864476215551)))

        @test  all(isapprox.(MonotonicSplines.eval_inverse_rqs_params_with_grad(pX[1,1,1], pX[2,1,1], pY[1,1,1], pY[2,1,1], pY[1,1,1], pY[2,1,1], y_test[1,1])[1:2], (-4.981580571322357, -7.878864476215551)))
        @test  all(isapprox.(MonotonicSplines.eval_inverse_rqs_params_with_grad(pX[1,1,1], pX[2,1,1], pY[1,1,1], pY[2,1,1], pY[1,1,1], pY[2,1,1], y_test[1,1])[3], (0.9964451502079126, 0.0035548497920874222)))
        @test  all(isapprox.(MonotonicSplines.eval_inverse_rqs_params_with_grad(pX[1,1,1], pX[2,1,1], pY[1,1,1], pY[2,1,1], pY[1,1,1], pY[2,1,1], y_test[1,1])[4], (-0.09860802071607391, 0.09898668350943046)))
        @test  all(isapprox.(MonotonicSplines.eval_inverse_rqs_params_with_grad(pX[1,1,1], pX[2,1,1], pY[1,1,1], pY[2,1,1], pY[1,1,1], pY[2,1,1], y_test[1,1])[5], (0.001692608637565046, 0.001853942717018299)))
        @test  all(isapprox.(MonotonicSplines.eval_inverse_rqs_params_with_grad(pX[1,1,1], pX[2,1,1], pY[1,1,1], pY[2,1,1], pY[1,1,1], pY[2,1,1], y_test[1,1])[6], (-0.4170286379079101, 0.4170286379079101)))
        @test  all(isapprox.(MonotonicSplines.eval_inverse_rqs_params_with_grad(pX[1,1,1], pX[2,1,1], pY[1,1,1], pY[2,1,1], pY[1,1,1], pY[2,1,1], y_test[1,1])[7], (-9.709051434667053, 10.696997710661178)))
        @test  all(isapprox.(MonotonicSplines.eval_inverse_rqs_params_with_grad(pX[1,1,1], pX[2,1,1], pY[1,1,1], pY[2,1,1], pY[1,1,1], pY[2,1,1], y_test[1,1])[8], (-0.01798457442421056, 0.2012775903613969)))
    end

    if compute_unit isa CPUnit
        @testset "pullback_vs_finite_differences" begin
            local nn_out = readdlm("test_outputs/test_nn_output.txt")
            local pX2, pY2, dYdX2 = MonotonicSplines.rqs_params_from_nn(vcat(nn_out, reverse(nn_out, dims=1)), 2)
            local x2 = vcat(x_test, reverse(x_test, dims=2))

            for (trafo, rqs_fun) in (
                (MonotonicSplines.RQSForward(), MonotonicSplines.rqs_forward),
                (MonotonicSplines.RQSInverse(), MonotonicSplines.rqs_inverse)
            )
                local δY = ones(size(x2))
                local δlogJac = ones(1, size(x2,2))
                local δx, δpX, δpY, δdYdX = MonotonicSplines.rqs_pullback(trafo, x2, pX2, pY2, dYdX2, δY, δlogJac)

                local f_sum(x, pX, pY, dYdX) = sum(sum.(rqs_fun(x, pX, pY, dYdX)))
                @test isapprox(δx, _fd_grad(x -> f_sum(x, pX2, pY2, dYdX2), x2), rtol = 1e-6)
                @test isapprox(δpX, _fd_grad(p -> f_sum(x2, p, pY2, dYdX2), pX2), rtol = 1e-6, atol = 1e-8)
                @test isapprox(δpY, _fd_grad(p -> f_sum(x2, pX2, p, dYdX2), pY2), rtol = 1e-6, atol = 1e-8)
                @test isapprox(δdYdX, _fd_grad(p -> f_sum(x2, pX2, pY2, p), dYdX2), rtol = 1e-6, atol = 1e-8)

                # tangent wrt x with only one of the two outputs used
                local δx_y = MonotonicSplines.rqs_pullback(trafo, x2, pX2, pY2, dYdX2, δY, zero(δlogJac))[1]
                @test isapprox(δx_y, _fd_grad(x -> sum(rqs_fun(x, pX2, pY2, dYdX2)[1]), x2), rtol = 1e-6)
                local δx_lj = MonotonicSplines.rqs_pullback(trafo, x2, pX2, pY2, dYdX2, zero(δY), δlogJac)[1]
                @test isapprox(δx_lj, _fd_grad(x -> sum(rqs_fun(x, pX2, pY2, dYdX2)[2]), x2), rtol = 1e-6)
            end

            # outside the spline range the tangent wrt x is the identity's
            local x_out = [-7.0 7.0; 6.0 -6.0]
            local pX3, pY3, dYdX3 = (p[:,:,1:2] for p in (pX2, pY2, dYdX2))
            local δx_out = MonotonicSplines.rqs_pullback(MonotonicSplines.RQSForward(), x_out, pX3, pY3, dYdX3, fill(2.0, size(x_out)), ones(1,2))[1]
            @test δx_out ≈ fill(2.0, size(x_out))
        end
    end
end