# This file is a part of MonotonicSplines.jl, licensed under the MIT License (MIT).

using Adapt
using ChainRulesCore
using ChangesOfVariables
using DelimitedFiles
using HeterogeneousComputing
using KernelAbstractions
using MonotonicSplines
using Test

import InverseFunctions
using Functors: functor

compute_units = isdefined(Main, :CUDA) ? [AbstractComputeUnit(CUDA.device()), CPUnit()] : [CPUnit()]

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

    local RQS_test = RQSpline(test_params_processed...)
    local RQS_inv_test = InvRQSpline(test_params_processed...)

    @testset "rqs_structs_$compute_unit_type" begin
        @test RQS_test isa RQSpline && isapprox(RQS_test.pX, pX) && isapprox(RQS_test.pY, pY) && isapprox(RQS_test.dYdX, dYdX)
        @test RQS_inv_test isa InvRQSpline && isapprox(RQS_inv_test.pX, pX) && isapprox(RQS_inv_test.pY, pY) && isapprox(RQS_inv_test.dYdX, dYdX)
    end

    @testset "rqs_high_lvl_applications_$compute_unit_type" begin
        @test isapprox(RQS_test(x_test), y_test)
        @test isapprox(RQS_inv_test(y_test), x_test)
    end

    @testset "inverse_$compute_unit_type" begin
        InverseFunctions.test_inverse(RQS_test, x_test)
        InverseFunctions.test_inverse(RQS_inv_test, y_test)
    end

    @testset "with_logabsdet_jacobian_$compute_unit_type" begin
        @test all(isapprox.(ChangesOfVariables.with_logabsdet_jacobian(RQS_test,x_test), (y_test, ladj_forward_test)))
        @test all(isapprox.(ChangesOfVariables.with_logabsdet_jacobian(RQS_inv_test,y_test), (x_test, ladj_inverse_test)))
    end

    @testset "functor$compute_unit_type" begin
        @test functor(RQS_test)[1] isa NamedTuple
        @test functor(RQS_inv_test)[1] isa NamedTuple
    end

    @testset "rqs_low_lvl_applications_$compute_unit_type" begin
        @test all(isapprox.(MonotonicSplines.rqs_forward(x_test, pX, pY, dYdX), (y_test, ladj_forward_test)))
        @test all(isapprox.(MonotonicSplines.rqs_inverse(y_test, pX, pY, dYdX), (x_test, ladj_inverse_test)))
    end

    @testset "rqs_kernels_$compute_unit_type" begin
        forward_kernel_test = MonotonicSplines.rqs_forward_kernel!(CPU(), 4)
        y_kernel_test = zeros(size(x_test)...)
        ladj_forward_kernel_test = zeros(size(x_test)...)
        forward_kernel_test(x_test, y_kernel_test, ladj_forward_kernel_test, pX,pY,dYdX, ndrange=size(x_test))
        @test isapprox(y_kernel_test, y_test)
        @test isapprox(sum(ladj_forward_kernel_test, dims = 1), ladj_forward_test)

        inverse_kernel_test = MonotonicSplines.rqs_inverse_kernel!(CPU(), 4)
        x_kernel_test = zeros(size(x_test)...)
        ladj_inverse_kernel_test = zeros(size(x_test)...)
        inverse_kernel_test(y_test, x_kernel_test, ladj_inverse_kernel_test, pX,pY,dYdX, ndrange=size(x_test))

        @test isapprox(x_kernel_test, x_test)
        @test isapprox(sum(ladj_inverse_kernel_test, dims = 1), ladj_inverse_test)
    end
end
