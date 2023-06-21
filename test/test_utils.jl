# This file is a part of MonotonicSplines.jl, licensed under the MIT License (MIT).

using MonotonicSplines
using Test
using FileIO

@testset "parameter_processing_functions" begin
    test_outputs = load("utils_test_outputs.jld2")

    test_params_processed = test_outputs["test_params_processed"]

    softmax_tri_test_output = test_outputs["softmax_tri_test_output"]
    softmax_matrix_test_output = test_outputs["softmax_matrix_test_output"]
    softmax_vector_test_output = test_outputs["softmax_vector_test_output"]

    cumsum_tri_test_output = test_outputs["cumsum_tri_test_output"]
    cumsum_matrix_test_output = test_outputs["cumsum_matrix_test_output"]
    cumsum_vector_test_output = test_outputs["cumsum_vector_test_output"]

    softplus_tri_test_output = test_outputs["softplus_tri_test_output"]
    softplus_matrix_test_output = test_outputs["softplus_matrix_test_output"]
    softplus_vector_test_output = test_outputs["softplus_vector_test_output"]
    
    K = 10
    n_dims_to_transform = 5
    n_smpls = 40

    test_nn_output = ones((3 * K - 1) * n_dims_to_transform, n_smpls)
    test_params_raw = reshape(test_nn_output, :, n_dims_to_transform, n_smpls)[1:K,:,:]
    test_params_matrix = ones(5,10)
    test_params_vector = ones(10)

    @test get_params(test_nn_output, n_dims_to_transform) == test_params_processed

    @test MonotonicSplines._softmax_tri(test_params_raw) == softmax_tri_test_output
    @test MonotonicSplines._softmax(test_params_matrix) == softmax_matrix_test_output
    @test MonotonicSplines._softmax(test_params_vector) == softmax_vector_test_output

    @test MonotonicSplines._cumsum_tri(test_params_raw) == cumsum_tri_test_output
    @test MonotonicSplines._cumsum(test_params_matrix) == cumsum_matrix_test_output
    @test MonotonicSplines._cumsum(test_params_vector) == cumsum_vector_test_output

    @test MonotonicSplines._softplus_tri(test_params_raw) == softplus_tri_test_output
    @test MonotonicSplines._softplus(test_params_vector) == softplus_vector_test_output
    @test MonotonicSplines._softplus(test_params_matrix) == softplus_matrix_test_output
end

@testset "output_sorting_and_bin_search" begin
    test_y1 = hcat(fill(0.1,10), fill(0.4,10), fill(0.6,10))'
    test_y2 = hcat(fill(1,10), fill(2,10), fill(3,10), fill(4,10), fill(5, 10), fill(6,10))'
     
    output_sorted = hcat(fill(0.1,10), fill(2,10), fill(3,10), fill(0.4,10), fill(5, 10), fill(0.6,10))'
    
    test_array_2_search = collect(1:10)

    @test MonotonicSplines._sort_dimensions(test_y1, test_y2, [true, false, false, true, false, true]) == output_sorted

    @test MonotonicSplines.searchsortedfirst_impl(test_array_2_search, 4.5) == 5
end
