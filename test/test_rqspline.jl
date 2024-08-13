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

    local RQS_test = RQSpline(test_params_processed...)
    local RQS_inv_test = InvRQSpline(test_params_processed...)

    @testset "rqs_structs_$compute_unit_type" begin
        @test RQS_test isa RQSpline && isapprox(RQS_test.widths, w) && isapprox(RQS_test.heights, h) && isapprox(RQS_test.derivatives, d)
        @test RQS_inv_test isa InvRQSpline && isapprox(RQS_inv_test.widths, w) && isapprox(RQS_inv_test.heights, h) && isapprox(RQS_inv_test.derivatives, d)
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
        @test all(isapprox.(ChangesOfVariables.with_logabsdet_jacobian(RQS_inv_test,y_test), (x_test, ladj_backward_test)))
    end

    @testset "rqs_low_lvl_applications_$compute_unit_type" begin
        @test all(isapprox.(MonotonicSplines.spline_forward(RQS_test, x_test), (y_test, ladj_forward_test)))
        @test all(isapprox.(MonotonicSplines.spline_backward(RQS_inv_test, y_test), (x_test, ladj_backward_test)))
        @test all(isapprox.(MonotonicSplines.rqs_forward(x_test, w, h, d), (y_test, ladj_forward_test)))
        @test all(isapprox.(MonotonicSplines.rqs_backward(y_test, w, h, d), (x_test, ladj_backward_test)))
    end

    @testset "rqs_kernels_$compute_unit_type" begin
        forward_kernel_test = MonotonicSplines.rqs_forward_kernel!(CPU(), 4)
        y_kernel_test = zeros(size(x_test)...)
        ladj_forward_kernel_test = zeros(size(x_test)...)
        forward_kernel_test(x_test, y_kernel_test, ladj_forward_kernel_test, w,h,d, ndrange=size(x_test))
        @test isapprox(y_kernel_test, y_test)
        @test isapprox(sum(ladj_forward_kernel_test, dims = 1), ladj_forward_test)

        backward_kernel_test = MonotonicSplines.rqs_backward_kernel!(CPU(), 4)
        x_kernel_test = zeros(size(x_test)...)
        ladj_backward_kernel_test = zeros(size(x_test)...)
        backward_kernel_test(y_test, x_kernel_test, ladj_backward_kernel_test, w,h,d, ndrange=size(x_test))

        @test isapprox(x_kernel_test, x_test)
        @test isapprox(sum(ladj_backward_kernel_test, dims = 1), ladj_backward_test)
    end
end
