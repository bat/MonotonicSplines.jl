# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT)
"""
    get_params(θ_raw::AbstractArray, n_dims_trafo::Integer, B::Real = 5.)

Process the raw output parameters of a neural net to be usable as the widths, heights and derivatives to characterize a set of rational quadratic spline functions.
Outputs a tripel `w,h,d` containing the widths, heights, and derivative parameters, either on the CPU or GPU, depending on the backend of the input `θ_raw`.

The `θ_raw` should be a matrix, with the columns being the raw parameters for a sample.
`n_dims_trafo` is the number of spline functions for which parameters are supposed to be produced.
The output parameters are stored in an `K+1 x n_spline_functions_per_sample x n_samples` array.
"""
function get_params(θ_raw::AbstractArray, n_dims_trafo::Integer, B::Real = 5.)
    N = size(θ_raw, 2)
    K = Int((size(θ_raw,1)/n_dims_trafo+1)/3)
    θ = reshape(θ_raw, :, n_dims_trafo, N)

    device = KernelAbstractions.get_backend(θ_raw)

    w = device isa GPU ? cat(cu(repeat([-B], 1, n_dims_trafo, N)), _cumsum_tri(_softmax_tri(θ[1:K,:,:])); dims = 1) : cat(repeat([-B], 1, n_dims_trafo, N), _cumsum_tri(_softmax_tri(θ[1:K,:,:])), dims = 1)
    h = device isa GPU ? cat(cu(repeat([-B], 1, n_dims_trafo, N)), _cumsum_tri(_softmax_tri(θ[K+1:2K,:,:])); dims = 1) : cat(repeat([-B], 1, n_dims_trafo, N), _cumsum_tri(_softmax_tri(θ[K+1:2K,:,:])), dims = 1)
    d = device isa GPU ? cat(cu(repeat([1], 1, n_dims_trafo, N)), _softplus_tri(θ[2K+1:end,:,:]), cu(repeat([1], 1, n_dims_trafo, N)); dims = 1) : cat(repeat([1], 1, n_dims_trafo, N), _softplus_tri(θ[2K+1:end,:,:]), repeat([1], 1, n_dims_trafo, N), dims = 1)

    return w, h, d
end

export get_params

"""
    _sort_dimensions(y₁::AbstractArray, y₂::AbstractArray, mask::AbstractVector)

Output an array identical to `y₂`, but with the rows specified by `mask` overwritten with the corresponding rows in `y₁`.
"""
function _sort_dimensions(y₁::AbstractArray, y₂::AbstractArray, mask::AbstractVector)
    
    if mask[1]
        res = reshape(y₁[1,:],1,size(y₁,2))
        c=2
    else
        res = reshape(y₂[1,:],1,size(y₁,2))
        c=1
    end

    for (i,b) in enumerate(mask[2:end])
        if b
            res = vcat(res, reshape(y₁[c,:],1,size(y₁,2)))
            c+=1
        else
            res = vcat(res, reshape(y₂[i+1,:],1,size(y₂,2)))
        end
    end

    return res
end

function _softmax(x::AbstractVector)

    exp_x = exp.(x)
    sum_exp_x = sum(exp_x)

    return exp_x ./ sum_exp_x 
end

function _softmax(x::AbstractMatrix)

    val = cat([_softmax(i) for i in eachrow(x)]..., dims=2)'

    return val 
end

function _softmax_tri(x::AbstractArray)
    exp_x = exp.(x)
    inv_sum_exp_x = inv.(sum(exp_x, dims = 1))

    return inv_sum_exp_x .* exp_x
end

function _cumsum(x::AbstractVector; B = 5)
    return 2 .* B .* cumsum(x) .- B 
end

function _cumsum(x::AbstractMatrix)

    return cat([_cumsum(i) for i in eachrow(x)]..., dims=2)'
end

function _cumsum_tri(x::AbstractArray, B::Real = 5.)
    
    return 2 .* B .* cumsum(x, dims = 1) .- B 
end

function _softplus(x::AbstractVector)

    return log.(exp.(x) .+ 1) 
end

function _softplus(x::AbstractMatrix)

    val = cat([_softplus(i) for i in eachrow(x)]..., dims=2)'

    return val
end

function _softplus_tri(x::AbstractArray)
    return log.(exp.(x) .+ 1) 
end

midpoint(lo::T, hi::T) where T<:Integer = lo + ((hi - lo) >>> 0x01)
binary_log(x::T) where {T<:Integer} = 8 * sizeof(T) - leading_zeros(x - 1)

function searchsortedfirst_impl(
        v::AbstractVector, 
        x::Real
    )
    
    u = one(Integer)
    lo = one(Integer) - u
    hi = length(v) + u
    
    n = binary_log(length(v))+1
    m = one(Integer)
    
    @inbounds for i in 1:n
        m_1 = midpoint(lo, hi)
        m = Base.ifelse(lo < hi - u, m_1, m)
        lo = Base.ifelse(v[m] < x, m, lo)
        hi = Base.ifelse(v[m] < x, hi, m)
    end
    return hi
end
