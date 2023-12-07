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
struct RQSpline <: Function
    widths::AbstractArray{<:Real}
    heights::AbstractArray{<:Real}
    derivatives::AbstractArray{<:Real}
end

export RQSpline
@functor RQSpline

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
struct InvRQSpline <: Function
    widths::AbstractArray{<:Real}
    heights::AbstractArray{<:Real}
    derivatives::AbstractArray{<:Real}
end

@functor InvRQSpline
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

# Usage
The function is used to apply the forward transformation characterized by an `RQSpline` instance to a matrix of samples.

# Performance
The function leverages parallel computing and automatic differentiation for high performance. 
It uses `KernelAbstractions.jl` for parallel execution on either a CPU or a GPU, and different backends for automatic 
differentiation, providing precise and efficient gradient computations.
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

# Note
The function executes in a kernel, on the same backend as `x` is stored on (CPU or GPU), and the output is also returned on this same backend.
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
    rqs_forward_pullback(x::AbstractArray{<:Real}, w::AbstractArray{<:Real}, h::AbstractArray{<:Real}, d::AbstractArray{<:Real}, tangent_1::AbstractArray, tangent_2::AbstractArray)

Compute and return the gradients of the rational quadratic spline functions characterized by `w`, `h`, and `d`, evaluated at the values in `x` with respect to `w`, `h`, and `d`. 

This function is designed to make the transformation using Rational Quadratic Splines in this package automatically differentiable. 

# Arguments
- `x`: An array of real numbers at which the spline functions are evaluated.
- `w`, `h`, `d`: Arrays that hold the width, height, and derivative parameters of the spline functions, respectively.
- `tangent_1`, `tangent_2`: Arrays that hold the tangent vectors for the transformed output and the log abs det jacobians respectively.

# Returns
Three values are returned:
- `∂y∂w + ∂LogJac∂w`: The gradient of the spline functions with respect to `w`.
- `∂y∂h + ∂LogJac∂h`: The gradient of the spline functions with respect to `h`.
- `∂y∂d + ∂LogJac∂d`: The gradient of the spline functions with respect to `d`.

# Usage
The function is used to compute the gradients of the spline functions, making the transformation using Rational Quadratic Splines in this package automatically differentiable.

# Performance
The function leverages parallel computing for high performance. It uses `KernelAbstractions.jl` for parallel execution on either a CPU or a GPU.

# Note
The function executes in a kernel, on the same backend as `x` is stored (CPU or GPU), and the output is also returned on the same backend.
"""
function rqs_forward_pullback(
        x::AbstractArray{<:Real},
        w::AbstractArray{<:Real},
        h::AbstractArray{<:Real},
        d::AbstractArray{<:Real},
        tangent_1::AbstractArray{<:Real},
        tangent_2::AbstractArray{<:Real};
    ) 

    compute_unit = get_compute_unit(x)
    n = compute_unit isa AbstractGPUnit ? 256 : Threads.nthreads()

    y = similar(x)
    logJac = similar(x)

    ∂y∂w = fill!(similar(w), zero(eltype(w)))
    ∂y∂d = fill!(similar(h), zero(eltype(h)))
    ∂y∂h = fill!(similar(d), zero(eltype(d)))

    ∂LogJac∂w = fill!(similar(w), zero(eltype(w)))
    ∂LogJac∂h = fill!(similar(h), zero(eltype(h)))
    ∂LogJac∂d = fill!(similar(d), zero(eltype(d)))

    kernel! = rqs_forward_pullback_kernel!(compute_unit, n)

    kernel!(
        x, y, logJac, 
        w, h, d,
        ∂y∂w, ∂y∂h, ∂y∂d,
        ∂LogJac∂w, ∂LogJac∂h, ∂LogJac∂d, 
        tangent_1,
        tangent_2,
        ndrange=size(x)
        )

    logJac = sum(logJac, dims=1)

    return ∂y∂w + ∂LogJac∂w, ∂y∂h + ∂LogJac∂h, ∂y∂d + ∂LogJac∂d
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

To find the bin `k` in which the respective x value for a spline falls in, the corresponding column of `w` is searched.

# Note
This function is a kernel function and is used within the `rqs_forward` function to perform the transformation, and is not intended to be called directly by the user.
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
    rqs_forward_pullback_kernel!(
        x::AbstractArray,
        y::AbstractArray,
        logJac::AbstractArray,
        w::AbstractArray,
        h::AbstractArray,
        d::AbstractArray,
        ∂y∂w_tangent::AbstractArray,
        ∂y∂h_tangent::AbstractArray,
        ∂y∂d_tangent::AbstractArray,
        ∂LogJac∂w_tangent::AbstractArray,
        ∂LogJac∂h_tangent::AbstractArray,
        ∂LogJac∂d_tangent::AbstractArray,
        tangent_1::AbstractArray,
        tangent_2::AbstractArray,
    )

This kernel function calculates the gradients of the rational quadratic spline functions characterized by `w`, `h`, and `d`, evaluated at the values in `x` and of `logJac`.

# Arguments
- `x`: An array of real numbers to which the spline functions are applied.
- `w`, `h`, `d`: Arrays that hold the width, height, and derivative parameters of the spline functions, respectively.
- `y`: An array where the transformed values are stored.
- `logJac`: An array where the sums of the values of the logarithm of the absolute values of the determinant of the Jacobians of the spline functions applied to a column of `x` are stored.
- `∂y∂w_tangent`, `∂y∂h_tangent`, `∂y∂d_tangent`: Arrays where the gradients of the spline functions with respect to `w`, `h`, and `d` are stored, respectively.
- `∂LogJac∂w_tangent`, `∂LogJac∂h_tangent`, `∂LogJac∂d_tangent`: Arrays where the gradients of `logJac` with respect to `w`, `h`, and `d` are stored, respectively.
- `tangent_1`, `tangent_2`: Arrays that hold the tangent vectors for the forward pass.

# Description
The transformed values are stored in `y` and the sums of the values of the logarithm of the absolute values of the determinant of the Jacobians of the spline functions applied to a column of `x` are stored in `logJac`. 
The gradients of the spline functions and `logJac` with respect to `w`, `h`, and `d` are calculated and stored in the respective arrays. Meaning the corresponding input arrays are overwritten with the computed gradients.

# Note
This function is a kernel function and is used within the `rqs_forward_pullback` function to calculate the gradients of the spline functions and `logJac`. It is not intended to be called directly by the user.
"""
@kernel function rqs_forward_pullback_kernel!(
        x::AbstractArray{<:Real},
        y::AbstractArray{<:Real},
        logJac::AbstractArray{<:Real},
        w::AbstractArray{<:Real},
        h::AbstractArray{<:Real},
        d::AbstractArray{<:Real},
        ∂y∂w_tangent::AbstractArray{<:Real},
        ∂y∂h_tangent::AbstractArray{<:Real},
        ∂y∂d_tangent::AbstractArray{<:Real},
        ∂LogJac∂w_tangent::AbstractArray{<:Real},
        ∂LogJac∂h_tangent::AbstractArray{<:Real},
        ∂LogJac∂d_tangent::AbstractArray{<:Real},
        tangent_1::AbstractArray{<:Real},
        tangent_2::AbstractArray{<:Real},
    )

    i, j = @index(Global, NTuple)

    # minus one is to account for left pad
    K = size(w, 1) - 1

    # Find the bin index
    k1 = searchsortedfirst_impl(view(w, :, i, j), x[i,j]) - 1
    k2 = one(typeof(k1))

    # Is inside of range
    isinside = (1 <= k1 <= K)
    k = Base.ifelse(isinside, k1, k2)

    x_tmp = Base.ifelse(isinside, x[i,j], w[k,i,j]) # Simplifies calculations
    (yᵢⱼ, LogJacᵢⱼ, ∂y∂w, ∂y∂h, ∂y∂d, ∂LogJac∂w, ∂LogJac∂h, ∂LogJac∂d) = eval_forward_rqs_params_with_grad(w[k,i,j], w[k+1,i,j], h[k,i,j], h[k+1,i,j], d[k,i,j], d[k+1,i,j], x_tmp)

    y[i,j] = Base.ifelse(isinside, yᵢⱼ, x[i,j]) 
    logJac[i,j] = Base.ifelse(isinside, LogJacᵢⱼ, zero(typeof(LogJacᵢⱼ)))

    ∂y∂w_tangent[k, i, j]      = tangent_1[i,j] * Base.ifelse(isinside, ∂y∂w[1], zero(eltype(∂y∂w)))
    ∂y∂h_tangent[k, i, j]      = tangent_1[i,j] * Base.ifelse(isinside, ∂y∂h[1], zero(eltype(∂y∂h)))
    ∂y∂d_tangent[k, i, j]      = tangent_1[i,j] * Base.ifelse(isinside, ∂y∂d[1], zero(eltype(∂y∂d)))
    ∂LogJac∂w_tangent[k, i, j] = tangent_2[1,j] * Base.ifelse(isinside, ∂LogJac∂w[1], zero(eltype(∂LogJac∂w)))
    ∂LogJac∂h_tangent[k, i, j] = tangent_2[1,j] * Base.ifelse(isinside, ∂LogJac∂h[1], zero(eltype(∂LogJac∂h)))
    ∂LogJac∂d_tangent[k, i, j] = tangent_2[1,j] * Base.ifelse(isinside, ∂LogJac∂d[1], zero(eltype(∂LogJac∂d)))

    ∂y∂w_tangent[k+1, i, j]       = tangent_1[i,j] * Base.ifelse(isinside, ∂y∂w[2], zero(eltype(∂y∂w)))
    ∂y∂h_tangent[k+1, i, j]       = tangent_1[i,j] * Base.ifelse(isinside, ∂y∂h[2], zero(eltype(∂y∂h)))
    ∂y∂d_tangent[k+1, i, j]       = tangent_1[i,j] * Base.ifelse(isinside, ∂y∂d[2], zero(eltype(∂y∂d)))
    ∂LogJac∂w_tangent[k+1, i, j]  = tangent_2[1,j] * Base.ifelse(isinside, ∂LogJac∂w[2], zero(eltype(∂LogJac∂w)))
    ∂LogJac∂h_tangent[k+1, i, j]  = tangent_2[1,j] * Base.ifelse(isinside, ∂LogJac∂h[2], zero(eltype(∂LogJac∂h)))
    ∂LogJac∂d_tangent[k+1, i, j]  = tangent_2[1,j] * Base.ifelse(isinside, ∂LogJac∂d[2], zero(eltype(∂LogJac∂d)))
end

function ChainRulesCore.rrule(
    ::typeof(rqs_forward),
    x::AbstractArray{<:Real},
    w::AbstractArray{<:Real},
    h::AbstractArray{<:Real},
    d::AbstractArray{<:Real}
)

    y, logJac = rqs_forward(x, w, h, d)
    compute_unit = get_compute_unit(x)

    pullback(tangent) = (NoTangent(), @thunk(tangent[1] .* exp.(logJac)), rqs_forward_pullback(x, w, h, d, adapt(compute_unit, tangent[1]), adapt(compute_unit, tangent[2]))...)

    return (y, logJac), pullback
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

"""
    eval_forward_rqs_params_with_grad(wₖ::M0, wₖ₊₁::M0, hₖ::M1, hₖ₊₁::M1, dₖ::M2, dₖ₊₁::M2, x::M3) where {M0<:Real,M1<:Real, M2<:Real, M3<:Real}

Apply a rational quadratic spline segment to `x`, calculate the logarithm of the absolute value of the derivative of the segment at `x`, 
and compute the gradient of that segment with respect to the spline parameters.

# Arguments
- `wₖ`, `wₖ₊₁`: The width parameters of the spline segment at the edges of the `k`-th interval.
- `hₖ`, `hₖ₊₁`: The height parameters of the spline segment.
- `dₖ`, `dₖ₊₁`: The derivative parameters of the spline segment.
- `x`: The value at which the spline function is evaluated.

# Returns
- `y`: The transformed value after applying the rational quadratic spline segment to `x`.
- `logJac`: The logarithm of the absolute value of the derivative of the segment at `x`.
- `∂y∂w`, `∂y∂h`, `∂y∂d`: The gradients of `y` with respect to the width, height, and derivative parameters, respectively.
- `∂LogJac∂w`, `∂LogJac∂h`, `∂LogJac∂d`: The gradients of `logJac` with respect to the width, height, and derivative parameters, respectively.
"""
function eval_forward_rqs_params_with_grad(
    wₖ::M0, wₖ₊₁::M0, 
    hₖ::M1, hₖ₊₁::M1, 
    dₖ::M2, dₖ₊₁::M2, 
    x::M3) where {M0<:Real,M1<:Real, M2<:Real, M3<:Real}

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

    # Gradient of parameters:

    # dy / dw_k
    ∂s∂wₖ = Δy/Δx^2
    ∂ξ∂wₖ = (-Δx + x - wₖ)/Δx^2
    ∂y∂wₖ = (Δy / denom^2) * ((∂s∂wₖ*ξ^2 + 2*sk*ξ*∂ξ∂wₖ + dₖ*(∂ξ∂wₖ -
                2*ξ*∂ξ∂wₖ))*denom - nom_1*(∂s∂wₖ - 2*∂s∂wₖ*ξ*(1-ξ) + (dₖ₊₁ + dₖ - 2*sk)*(∂ξ∂wₖ - 2*ξ*∂ξ∂wₖ)) )
    ∂LogJac∂wₖ = (1/nom_4)*(2*sk*∂s∂wₖ*nom_3 + sk*sk*(2*dₖ₊₁*ξ*∂ξ∂wₖ + 2*∂s∂wₖ*ξ*(1-ξ)+2*sk*(∂ξ∂wₖ - 2*ξ*∂ξ∂wₖ)-dₖ*2*(1-ξ)*∂ξ∂wₖ)) - (2/denom)*(∂s∂wₖ - 2*∂s∂wₖ*ξ*(1-ξ) + (dₖ₊₁ + dₖ - 2*sk)*(∂ξ∂wₖ - 2*ξ*∂ξ∂wₖ))

    # dy / dw_k+1
    ∂s∂wₖ₊₁ = -Δy/Δx^2
    ∂ξ∂wₖ₊₁ = -(x - wₖ) / Δx^2
    ∂y∂wₖ₊₁ = (Δy / denom^2) * ((∂s∂wₖ₊₁*ξ^2 + 2*sk*ξ*∂ξ∂wₖ₊₁ + dₖ*(∂ξ∂wₖ₊₁ -
                2*ξ*∂ξ∂wₖ₊₁))*denom - nom_1*(∂s∂wₖ₊₁ - 2*∂s∂wₖ₊₁*ξ*(1-ξ) + (dₖ₊₁ + dₖ - 2*sk)*(∂ξ∂wₖ₊₁ - 2*ξ*∂ξ∂wₖ₊₁)) )
    ∂LogJac∂wₖ₊₁ = (1/nom_4)*(2*sk*∂s∂wₖ₊₁*nom_3 + sk*sk*(2*dₖ₊₁*ξ*∂ξ∂wₖ₊₁ + 2*∂s∂wₖ₊₁*ξ*(1-ξ)+2*sk*(∂ξ∂wₖ₊₁ - 2*ξ*∂ξ∂wₖ₊₁)-dₖ*2*(1-ξ)*∂ξ∂wₖ₊₁)) - (2/denom)*(∂s∂wₖ₊₁ - 2*∂s∂wₖ₊₁*ξ*(1-ξ) + (dₖ₊₁ + dₖ - 2*sk)*(∂ξ∂wₖ₊₁ - 2*ξ*∂ξ∂wₖ₊₁))

    # dy / dh_k
    ∂s∂hₖ = -1/Δx
    ∂y∂hₖ = 1 + (1/denom^2)*((-nom_1+Δy*ξ*ξ*∂s∂hₖ)*denom - nom_2 * (∂s∂hₖ - 2*∂s∂hₖ*ξ*(1-ξ)) )
    ∂LogJac∂hₖ = (1/nom_4)*(2*sk*∂s∂hₖ*nom_3 + sk*sk*2*∂s∂hₖ*ξ*(1-ξ)) - (2/denom)*(∂s∂hₖ - 2*∂s∂hₖ*ξ*(1-ξ))

    # dy / dh_k+1
    ∂s∂hₖ₊₁ = 1/Δx
    ∂y∂hₖ₊₁ = (1/denom^2)*((nom_1+Δy*ξ*ξ*∂s∂hₖ₊₁)*denom - nom_2 * (∂s∂hₖ₊₁ - 2*∂s∂hₖ₊₁*ξ*(1-ξ)) )
    ∂LogJac∂hₖ₊₁ = (1/nom_4)*(2*sk*∂s∂hₖ₊₁*nom_3 + sk*sk*2*∂s∂hₖ₊₁*ξ*(1-ξ)) - (2/denom)*(∂s∂hₖ₊₁ - 2*∂s∂hₖ₊₁*ξ*(1-ξ))

    # dy / dd_k
    ∂y∂dₖ = (1/denom^2) * ((Δy*ξ*(1-ξ))*denom - nom_2*ξ*(1-ξ) )
    ∂LogJac∂dₖ = (1/nom_4)*sk^2*(1-ξ)^2 - (2/denom)*ξ*(1-ξ)

    # dy / dδ_k+1
    ∂y∂dₖ₊₁ = -(nom_2/denom^2) * ξ*(1-ξ)
    ∂LogJac∂dₖ₊₁ = (1/nom_4)*sk^2*ξ^2 - (2/denom)*ξ*(1-ξ)

    ∂y∂w = (∂y∂wₖ, ∂y∂wₖ₊₁)
    ∂y∂h = (∂y∂hₖ, ∂y∂hₖ₊₁)
    ∂y∂d = (∂y∂dₖ, ∂y∂dₖ₊₁)

    ∂LogJac∂w = (∂LogJac∂wₖ, ∂LogJac∂wₖ₊₁)
    ∂LogJac∂h = (∂LogJac∂hₖ, ∂LogJac∂hₖ₊₁)
    ∂LogJac∂d = (∂LogJac∂dₖ, ∂LogJac∂dₖ₊₁)

    return y, logJac, ∂y∂w, ∂y∂h, ∂y∂d, ∂LogJac∂w, ∂LogJac∂h, ∂LogJac∂d
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

# Usage
The function is used to apply the backward transformation characterized by an `InvRQSpline` instance to a matrix of samples.

# Performance
The function leverages parallel computing and automatic differentiation for high performance. 
It uses `KernelAbstractions.jl` for parallel execution on either a CPU or a GPU, and different backends for automatic 
differentiation, providing precise and efficient gradient computations.
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

# Note
The function executes in a kernel, on the same backend as `x` is stored on (CPU or GPU), and the output is also returned on this same backend.
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

# Description
The function works by applying the inverse spline function characterized by the parameters in the `[:,i,j]` entries in `w`, `h`, and `d` to the 
`[i,j]`-th element of `x`. The transformed values are stored in `y` and the sums of the values of the logarithm of the absolute values of 
the determinant of the Jacobians of the inverse spline functions applied to a column of `x` are stored in `logJac`.

To find the bin `k` in which the respective x value for a spline falls in, the corresponding column of `h` is searched.

# Note
This function is a kernel function and is used within the `rqs_backward` function to perform the transformation, and is not intended to be called directly by the user.
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
