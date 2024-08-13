# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT)

"""
    get_params(θ_raw::AbstractArray, n_dims_trafo::Integer, B::Real = 5.)

Process the raw output parameters of a neural network to generate parameters for a set of rational quadratic spline functions.

# Arguments
- `θ_raw`: A matrix where each column represents the raw parameters for a sample.
- `n_dims_trafo`: The number of spline functions for which parameters are to be produced.
- `B`: Sets the rage of the splines.

# Returns
- A tuple `pX, pY, dYdX` containing the positions of and derivatives at the spline knots.
  The parameters are stored in a `K+1 x n_spline_functions_per_sample x n_samples` array.
"""
function get_params(θ_raw::AbstractArray, n_dims_trafo::Integer, B::Real = 5.)
    N = size(θ_raw, 2)
    K = Int((size(θ_raw,1)/n_dims_trafo+1)/3)
    θ = reshape(θ_raw, :, n_dims_trafo, N)

    compute_unit = get_compute_unit(θ_raw)

    pX =  cat(adapt(compute_unit, repeat([-B], 1, n_dims_trafo, N)), _cumsum_tri(_softmax_tri(θ[1:K,:,:])); dims = 1)
    pY =  cat(adapt(compute_unit, repeat([-B], 1, n_dims_trafo, N)), _cumsum_tri(_softmax_tri(θ[K+1:2K,:,:])); dims = 1)
    dYdX =  cat(adapt(compute_unit, repeat([1], 1, n_dims_trafo, N)), _softplus_tri(θ[2K+1:end,:,:]), adapt(compute_unit, repeat([1], 1, n_dims_trafo, N)); dims = 1)

    return pX, pY, dYdX
end

export get_params

"""
    _sort_dimensions(y₁::AbstractArray, y₂::AbstractArray, mask::AbstractVector)

Create a new array by selectively replacing rows from `y₂` with corresponding rows from `y₁` based on a boolean mask, `mask`.

# Arguments
- `y₁`: An array from which rows are taken. It should have the same number of columns as `y₂`.
- `y₂`: An array that serves as the base for the output. Rows specified by `mask` are replaced with corresponding rows from `y₁`.
- `mask`: A boolean vector of the same length as the number of rows in `y₁` and `y₂`. If the i-th element of `mask` is true, the i-th row of `y₂` is replaced with the i-th row of `y₁` in the output.

# Returns
- `res`: An array of the same shape as `y₂`, but with rows specified by `mask` replaced with corresponding rows from `y₁`.
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


_ka_threads(::KernelAbstractions.CPU) = (Base.Threads.nthreads(),)
_ka_threads(::KernelAbstractions.Backend) = ()
