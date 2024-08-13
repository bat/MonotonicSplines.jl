# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).

"""
    RQSpline(widths::AbstractArray{<:Real}, heights::AbstractArray{<:Real}, derivatives::AbstractArray{<:Real})

A struct that holds the parameters to characterize multiple rational quadratic spline functions. 
These functions are used to transform each component of multiple samples, providing a high-performance 
solution for batch transformations.

The `RQSpline` struct is based on the scheme first defined by Gregory and Delbourgo 
(https://doi.org/10.1093/imanum/2.2.123) and is designed to handle `n_dims x n_samples` spline functions For a batch of 
`n_samples` `n_dims`-dimensional samples.

# Fields
- `widths`: A `K+1 x n_dims x n_samples` array that holds the width parameters for each spline function. 
  The first dimension corresponds to the number of segments in a single spline function, the second dimension 
  corresponds to the spline functions for a single sample, and the third dimension corresponds to the sets of 
  splines for different samples.

- `heights`: A `K+1 x n_dims x n_samples` array that holds the height parameters for each spline function. 
  The dimensions are organized in the same way as the `widths` array.

- `derivatives`: A `K+1 x n_dims x n_samples` array that holds the derivative parameters for each spline function. 
  The dimensions are organized in the same way as the `widths` and `heights` arrays.

# Usage
An instance of `RQSpline` can be used like a function to apply the characterized spline transformations 
to the components of the samples.

# Performance
The struct is designed to leverage parallel computing and automatic differentiation for high performance. 
It uses `KernelAbstractions.jl` for parallel execution on either a CPU or a GPU, and different backends for automatic 
differentiation, providing precise and efficient gradient computations.

"""
struct RQSpline{
    T<:Real, N,
    TX<:AbstractArray{T,N}, TY<:AbstractArray{T,N}, TD<:AbstractArray{T,N}
} <: Function
    widths::TX
    heights::TY
    derivatives::TD
end

export RQSpline

(f::RQSpline)(x::AbstractMatrix{<:Real}) = spline_forward(f, x)[1]

function ChangesOfVariables.with_logabsdet_jacobian(
    f::RQSpline,
    x::AbstractMatrix{<:Real}
)
    return spline_forward(f, x)
end

"""
    InvRQSpline(widths::AbstractArray{<:Real}, heights::AbstractArray{<:Real}, derivatives::AbstractArray{<:Real})

A struct that holds the parameters to characterize multiple inverse rational quadratic spline functions. 
These functions are used to transform each component of multiple samples, providing a high-performance 
solution for batch transformations.

The `InvRQSpline` struct is based on the scheme first defined by Gregory and Delbourgo 
(https://doi.org/10.1093/imanum/2.2.123) and is designed to handle `n_dims x n_samples` inverse spline functions.

# Fields
- `widths`: A `K+1 x n_dims x n_samples` array that holds the width parameters for each inverse spline function. 
  The first dimension corresponds to the number of segments in a single spline function, the second dimension 
  corresponds to the inverse spline functions for a single sample, and the third dimension corresponds to the sets of 
  inverse splines for different samples.

- `heights`: A `K+1 x n_dims x n_samples` array that holds the height parameters for each inverse spline function. 
  The dimensions are organized in the same way as the `widths` array.

- `derivatives`: A `K+1 x n_dims x n_samples` array that holds the derivative parameters for each inverse spline function. 
  The dimensions are organized in the same way as the `widths` and `heights` arrays.

# Usage
An instance of `InvRQSpline` can be used like a function to apply the characterized inverse spline transformations 
to the components of the samples.

# Performance
The struct is designed to leverage parallel computing and automatic differentiation for high performance. 
It uses `KernelAbstractions.jl` for parallel execution on either a CPU or a GPU, and `Zygote.jl` for automatic 
differentiation, providing precise and efficient gradient computations.

# Note
The same parameters are used to characterize both the forward (`RQSpline`) and inverse (`InvRQSpline`) spline functions. 
The struct used to store them determines the equation they are evaluated in.

"""
struct InvRQSpline{
    T<:Real, N,
    TX<:AbstractArray{T,N}, TY<:AbstractArray{T,N}, TD<:AbstractArray{T,N}
} <: Function
    widths::TX
    heights::TY
    derivatives::TD
end

export InvRQSpline

(f::InvRQSpline)(x::AbstractMatrix{<:Real}) = spline_backward(f, x)[1]

function ChangesOfVariables.with_logabsdet_jacobian(
    f::InvRQSpline,
    x::AbstractMatrix{<:Real}
)
    return spline_backward(f, x)
end

# Transformation forward:
"""
    spline_forward(trafo::RQSpline, x::AbstractMatrix{<:Real})

Apply the rational quadratic spline functions, characterized by the parameters stored in `trafo`, to the matrix `x`.

This function provides a high-performance solution for batch transformations, applying multiple spline functions 
to a matrix of samples simultaneously. One sample is stored in a column of the input matrix `x`.

# Arguments
- `trafo`: An instance of `RQSpline` that holds the parameters of the spline functions. These parameters are
  `widths`, `heights`, and `derivatives`, each of which is a 3D array with dimensions `K+1 x n_dims x n_samples`.

- `x`: A matrix of real numbers to which the spline functions are applied. The `[i,j]`-th element of `x` is transformed 
  by the spline function characterized by the parameters in the `[:,i,j]` entries in the parameter arrays stored in `trafo`.

# Returns
Two objects are returned:
- `y`: A matrix of the same shape as `x` that holds the transformed values.
- `logJac`: A `1 x size(x,2)` matrix that holds the sums of the values of the logarithm of the absolute values of the determinant 
   of the Jacobians of the spline functions applied to a column of `x`.
"""
function spline_forward(trafo::RQSpline, x::AbstractMatrix{<:Real})
    return rqs_forward(x, trafo.widths, trafo.heights, trafo.derivatives)
end

"""
    rqs_forward(x::AbstractArray{<:Real}, w::AbstractArray{<:Real}, h::AbstractArray{<:Real}, d::AbstractArray{<:Real})

Apply the rational quadratic spline functions, characterized by the parameters `w` (widths), `h` (heights), and `d` (derivatives), 
to the array `x`.

One sample is stored in a column of the input `x`.

# Arguments
- `x`: An array of real numbers to which the spline functions are applied. The `[i,j]`-th element of `x` is transformed 
  by the spline function characterized by the parameters in the `[:,i,j]` entries in `w`, `h`, and `d`.

- `w`, `h`, `d`: Arrays that hold the width, height, and derivative parameters of the spline functions, respectively. 
  Each of these is a 3D array with dimensions `K+1 x n_dims x n_samples`.

# Returns
Two objects are returned:
- `y`: A matrix of the same shape as `x` that holds the transformed values.
- `logJac`: A `1 x size(x,2)` matrix that holds the sums of the values of the logarithm of the absolute values of the determinant 
  of the Jacobians of the spline functions applied to a column of `x`.
"""
function rqs_forward(
        x::AbstractArray{<:Real},
        w::AbstractArray{<:Real},
        h::AbstractArray{<:Real},
        d::AbstractArray{<:Real}
    ) 

    compute_unit = get_compute_unit(x)
    n = compute_unit isa AbstractGPUnit ? 256 : Threads.nthreads()
    kernel! = rqs_forward_kernel!(compute_unit, n)

    y = similar(x)
    logJac = similar(x)

    kernel!(x, y, logJac, w, h, d, ndrange=size(x))

    logJac = sum(logJac, dims=1)

    return y, logJac
end

"""
    rqs_forward_kernel!(x::AbstractArray, y::AbstractArray, logJac::AbstractArray, w::AbstractArray, h::AbstractArray, d::AbstractArray)

This kernel function applies the rational quadratic spline functions characterized by the parameters `w`, `h`, and `d` to `x`. 

# Arguments
- `x`: An array of real numbers to which the spline functions are applied.
- `w`, `h`, `d`: Arrays that hold the width, height, and derivative parameters of the spline functions, respectively.
- `y`: An array where the transformed values are stored.
- `logJac`: An array where the sums of the values of the logarithm of the absolute values of the determinant of the Jacobians of the spline 
functions applied to a column of `x` are stored.

# Description
The function works by applying the spline function characterized by the parameters in the `[:,i,j]` entries in `w`, `h`, and `d` to the 
`[i,j]`-th element of `x`. The transformed values are stored in `y` and the sums of the values of the logarithm of the absolute values of 
the determinant of the Jacobians of the spline functions applied to a column of `x` are stored in `logJac`.

# Note
To find the bin `k` in which the respective x value for a spline falls in, the corresponding column of `w` is searched.
"""
@kernel function rqs_forward_kernel!(
    x::AbstractArray{<:Real},
    y::AbstractArray{<:Real},
    logJac::AbstractArray{<:Real},
    w::AbstractArray{<:Real},
    h::AbstractArray{<:Real},
    d::AbstractArray{<:Real}
)

    i, j = @index(Global, NTuple)

    K = size(w, 1) - 1

    # Find the bin index
    k1 = searchsortedfirst_impl(view(w, :, i, j), x[i,j]) - 1
    k2 = one(typeof(k1))

    # Is inside of range
    isinside = (1 <= k1 <= K)
    k = Base.ifelse(isinside, k1, k2)

    x_tmp = Base.ifelse(isinside, x[i,j], w[k,i,j]) # Simplifies calculations
    (yᵢⱼ, LogJacᵢⱼ) = eval_forward_rqs_params(w[k,i,j], w[k+1,i,j], h[k,i,j], h[k+1,i,j], d[k,i,j], d[k+1,i,j], x_tmp)

    y[i,j] = Base.ifelse(isinside, yᵢⱼ, x[i,j]) 
    logJac[i, j] = Base.ifelse(isinside, LogJacᵢⱼ, zero(typeof(LogJacᵢⱼ)))
end

"""
    eval_forward_rqs_params(wₖ::Real, wₖ₊₁::Real, hₖ::Real, hₖ₊₁::Real, dₖ::Real, dₖ₊₁::Real, x::Real)

Apply a rational quadratic spline segment to a number `x`, and calculate the logarithm of the absolute value of this function's Jacobian.

# Arguments
- `wₖ`, `wₖ₊₁`: The width parameters of the spline segment at the edges of the `k`-th interval.
- `hₖ`, `hₖ₊₁`: The height parameters of the spline segment.
- `dₖ`, `dₖ₊₁`: The derivative parameters of the spline segment.
- `x`: The value at which the spline function is evaluated.

# Returns
- `y`: The transformed value after applying the rational quadratic spline segment to `x`.
- `logJac`: The logarithm of the absolute value of the derivative of the segment at `x`.
"""
function eval_forward_rqs_params(
    wₖ::Real, wₖ₊₁::Real, 
    hₖ::Real, hₖ₊₁::Real, 
    dₖ::Real, dₖ₊₁::Real, 
    x::Real) 
      
    Δy = hₖ₊₁ - hₖ
    Δx = wₖ₊₁ - wₖ
    sk = Δy / Δx
    ξ = (x - wₖ) / Δx

    denom = (sk + (dₖ₊₁ + dₖ - 2*sk)*ξ*(1-ξ))
    nom_1 =  sk*ξ*ξ + dₖ*ξ*(1-ξ)
    nom_2 = Δy * nom_1
    nom_3 = dₖ₊₁*ξ*ξ + 2*sk*ξ*(1-ξ) + dₖ*(1-ξ)^2
    nom_4 = sk*sk*nom_3

    y = hₖ + nom_2/denom

    # LogJacobian
    logJac = log(abs(nom_4))-2*log(abs(denom))

    return y, logJac
end

# Transformation backward: 
"""
    spline_backward(trafo::InvRQSpline, x::AbstractMatrix{<:Real})

Apply the inverse rational quadratic spline functions, characterized by the parameters stored in `trafo`, to the matrix `x`.

This function provides a high-performance solution for batch transformations, applying multiple inverse spline functions 
to a matrix of samples simultaneously. One sample is stored in a column of the input matrix `x`.

# Arguments
- `trafo`: An instance of `InvRQSpline` that holds the parameters of the inverse spline functions. These parameters are
  `widths`, `heights`, and `derivatives`, each of which is a 3D array with dimensions `K+1 x n_dims x n_samples`.

- `x`: A matrix of real numbers to which the inverse spline functions are applied. The `[i,j]`-th element of `x` is transformed 
  by the inverse spline function characterized by the parameters in the `[:,i,j]` entries in the parameter arrays stored in `trafo`.

# Returns
Two objects are returned:
- `y`: A matrix of the same shape as `x` that holds the transformed values.
- `logJac`: A `1 x size(x,2)` matrix that holds the sums of the values of the logarithm of the absolute values of the determinant 
    of the Jacobians of the spline functions applied to a column of `x`.
"""
function spline_backward(trafo::InvRQSpline, x::AbstractMatrix{<:Real})
    return rqs_backward(x, trafo.widths, trafo.heights, trafo.derivatives)
end

"""
    rqs_backward(x::AbstractArray{<:Real}, w::AbstractArray{<:Real}, h::AbstractArray{<:Real}, d::AbstractArray{<:Real})

Apply the inverse rational quadratic spline functions, characterized by the parameters `w` (widths), `h` (heights), and `d` (derivatives), 
to the array `x`.

One sample is stored in a column of the input `x`.

# Arguments
- `x`: An array of real numbers to which the inverse spline functions are applied. The `[i,j]`-th element of `x` is transformed 
  by the inverse spline function characterized by the parameters in the `[:,i,j]` entries in `w`, `h`, and `d`.

- `w`, `h`, `d`: Arrays that hold the width, height, and derivative parameters of the inverse spline functions, respectively. 
  Each of these is a 3D array with dimensions `K+1 x n_dims x n_samples`.

# Returns
Two objects are returned:
- `y`: A matrix of the same shape as `x` that holds the transformed values.
- `logJac`: A `1 x size(x,2)` matrix that holds the sums of the values of the logarithm of the absolute values of the determinant 
  of the Jacobians of the inverse spline functions applied to a column of `x`.
"""
function rqs_backward(
        x::AbstractArray{<:Real},
        w::AbstractArray{<:Real},
        h::AbstractArray{<:Real},
        d::AbstractArray{<:Real}
    )

    compute_unit = get_compute_unit(x)
    n = compute_unit isa AbstractGPUnit ? 256 : Threads.nthreads()
    kernel! = rqs_backward_kernel!(compute_unit, n)

    y = similar(x)
    logJac = similar(x) 
    kernel!(x, y, logJac, w, h, d, ndrange=size(x))
    logJac = sum(logJac, dims=1)

    return y, logJac
end

"""
    rqs_backward_kernel!(x::AbstractArray, y::AbstractArray, logJac::AbstractArray, w::AbstractArray, h::AbstractArray, d::AbstractArray)

This kernel function applies the inverse rational quadratic spline functions characterized by the parameters `w`, `h`, and `d` to `x`. 

# Arguments
- `x`: An array of real numbers to which the inverse spline functions are applied.
- `w`, `h`, `d`: Arrays that hold the width, height, and derivative parameters of the inverse spline functions, respectively.
- `y`: An array where the transformed values will be stored.
- `logJac`: An array where the sums of the values of the logarithm of the absolute values of the determinant of the Jacobians of the inverse spline 
functions applied to a column of `x` will be stored.
"""
@kernel function rqs_backward_kernel!(
        x::AbstractArray{<:Real},
        y::AbstractArray{<:Real},
        logJac::AbstractArray{<:Real},
        w::AbstractArray{<:Real},
        h::AbstractArray{<:Real},
        d::AbstractArray{<:Real}
    ) 

    i, j = @index(Global, NTuple)
    
    K = size(w, 1) - 1

    # Find the bin index
    k1 = searchsortedfirst_impl(view(h, :, i, j), x[i,j]) - 1
    k2 = one(typeof(k1))

    # Is inside of range
    isinside = (k1 < K) && (k1 > 0)
    k = Base.ifelse(isinside, k1, k2)

    x_tmp = Base.ifelse(isinside, x[i,j], w[k,i,j])  # Simplifies unnecessary calculations
    (yᵢⱼ, LogJacᵢⱼ) = eval_backward_rqs_params(w[k,i,j], w[k+1,i,j], h[k,i,j], h[k+1,i,j], d[k,i,j], d[k+1,i,j], x_tmp)

    y[i,j] = Base.ifelse(isinside, yᵢⱼ, x[i,j]) 
    logJac[i, j] = Base.ifelse(isinside, LogJacᵢⱼ, zero(typeof(LogJacᵢⱼ)))
end

"""
    eval_forward_rqs_params(wₖ::Real, wₖ₊₁::Real, hₖ::Real, hₖ₊₁::Real, dₖ::Real, dₖ₊₁::Real, x::Real)

Apply a rational quadratic spline segment to a number `x`, and calculate the logarithm of the absolute value of this function's derivative.

# Arguments
- `wₖ`, `wₖ₊₁`: The width parameters of the spline segment at the edges of the `k`-th interval.
- `hₖ`, `hₖ₊₁`: The height parameters of the spline segment.
- `dₖ`, `dₖ₊₁`: The derivative parameters of the spline segment.
- `x`: The value at which the spline function is evaluated.

# Returns
- `y`: The transformed value after applying the rational quadratic spline segment to `x`.
- `logJac`: The logarithm of the absolute value of the derivative of the segment at `x`.
"""
function eval_backward_rqs_params(
    wₖ::M0, wₖ₊₁::M0, 
    hₖ::M1, hₖ₊₁::M1, 
    dₖ::M2, dₖ₊₁::M2, 
    x::M3) where {M0<:Real,M1<:Real, M2<:Real, M3<:Real}

    Δy = hₖ₊₁ - hₖ
    Δy2 = x - hₖ # use y instead of X, because of inverse
    Δx = wₖ₊₁ - wₖ
    sk = Δy / Δx

    a = Δy * (sk - dₖ) + Δy2 * (dₖ₊₁ + dₖ - 2*sk)
    b = Δy * dₖ - Δy2 * (dₖ₊₁ + dₖ - 2*sk)
    c = - sk * Δy2

    denom = -b - sqrt(b*b - 4*a*c)

    y = (2 * c / denom) * Δx + wₖ

    # Gradient computation:
    da =  (dₖ₊₁ + dₖ - 2*sk)
    db = -(dₖ₊₁ + dₖ - 2*sk)
    dc = - sk

    temp2 = 1 / (2*sqrt(b*b - 4*a*c))

    grad = 2 * dc * denom - 2 * c * (-db - temp2 * (2 * b * db - 4 * a * dc - 4 * c * da))
    LogJac = log(abs(Δx * grad)) - 2*log(abs(denom))

    return y, LogJac
end

# For debugging 
export eval_backward_rqs_params
