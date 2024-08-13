# This file is a part of MonotonicSplines.jl, licensed under the MIT License (MIT).

"""
    rqs_pullback(param_eval_function::Function, x::AbstractArray{<:Real}, w::AbstractArray{<:Real}, h::AbstractArray{<:Real}, d::AbstractArray{<:Real}, tangent_1::AbstractArray, tangent_2::AbstractArray)

Compute the gradients of the rational quadratic spline functions characterized by `w`, `h`, and `d`, evaluated at the values in `x` with respect to `w`, `h`, and `d`. 

This function is designed to make the transformation using Rational Quadratic Splines in this package automatically differentiable.
Whether the gradients of the forward or backward spline functions are calculated is determined by the `param_eval_function` argument.

# Arguments
- `param_eval_function`: The function used to evaluate a spline segment. Different functions are used for the forward and backward spline functions.
- `x`: An array of real numbers at which the spline functions are evaluated.
- `w`, `h`, `d`: Arrays that hold the width, height, and derivative parameters of the spline functions, respectively.
- `tangent_1`, `tangent_2`: Arrays that hold the tangent vectors for the transformed output and the log abs det jacobians respectively.

# Returns
Three values are returned:
- `∂y∂w + ∂LogJac∂w`: An array with the same shape as `w`, with the `[:,j,k]`-th element holding the gradient of the `[j,k]`-th element of `y` with respect to the width parameters
                      plus the gradient of the logarithm of the absolute value of the derivative of this `[j,k]`-th element of `y` with respect to the `[j,k]`-th element of `x`, with respect to the width parameters. 
                      For Example, the `[i,j,k]` element of this array is `∂yⱼₖ/∂wᵢⱼₖ + ∂(log(abs(∂yⱼₖ/∂xⱼₖ)))/∂wᵢⱼₖ`.
- `∂y∂h + ∂LogJac∂h`: An array with the same shape as `h`, holding the same gradients as described above, but with respect to the height parameters.
- `∂y∂d + ∂LogJac∂d`: An array with the same shape as `d`, holding the same gradients as described above, but with respect to the derivative parameters.

# Note
Since only one segment of each spline is evaluated for one element of `x`, the returned gradients are filled with zeros, except for the parameters of the segment that is evaluated for that element of `x`.
For example, if the `[j,k]` -th element of `x` falls in the `l`-th bin of the interval mask, the `[:,j,k]` entries in `∂y∂w + ∂LogJac∂w` are all zero, except the `[l,j,k]` and `[l+1,j,k]` elements, which hold 
the (generally non-zero) values `∂yⱼₖ/∂wₗⱼₖ + ∂(log(abs(∂yⱼₖ/∂xⱼₖ)))/∂wₗⱼₖ` and `∂yⱼₖ/∂wₗ₊₁ⱼₖ + ∂(log(abs(∂yⱼₖ/∂xⱼₖ)))/∂wₗ₊₁ⱼₖ` respectively.

The function executes in a kernel, on the same backend as `x` is stored (CPU or GPU), and the output is also returned on the same backend.
"""
function rqs_pullback(
    param_eval_function::Function,
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

    kernel! = rqs_pullback_kernel!(compute_unit, n)

    kernel!(
        param_eval_function,
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
    rqs_pullback_kernel(
        param_eval_function::Function,
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
- `param_eval_function` The function used to evaluate a spline segment. Different functions are used for the forward and backward passes.
- `x`: An array of real numbers to which the spline functions are applied.
- `w`, `h`, `d`: Arrays that hold the width, height, and derivative parameters of the spline functions, respectively.
- `y`: An array where the transformed values are stored.
- `logJac`: An array where the sums of the values of the logarithm of the absolute values of the determinant of the Jacobians of the spline functions applied to a column of `x` are stored.
- `∂y∂w_tangent`, `∂y∂h_tangent`, `∂y∂d_tangent`: Arrays that will contain the gradients of the spline functions with respect to `w`, `h`, and `d`, respectively.
- `∂LogJac∂w_tangent`, `∂LogJac∂h_tangent`, `∂LogJac∂d_tangent`: Arrays that will contain the gradients of `logJac` with respect to `w`, `h`, and `d`, respectively.
- `tangent_1`, `tangent_2`: Arrays that hold the tangent vectors for the forward pass.

For an explanation of the shape and contents of the gradient arrays, see the documentation of the [`rqs_pullback()`](@ref) function.

# Note
This function is a kernel function and is used within the `rqs_forward_pullback` function to calculate the gradients of the spline functions and `logJac`. It is not intended to be called directly by the user.
"""
@kernel function rqs_pullback_kernel!(
        param_eval_function::Function,
        x::AbstractArray{<:Real},
        y::AbstractArray{<:Real},
        LogJac::AbstractArray{<:Real},
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

    # minus one to account for left pad
    K = size(w, 1) - 1

    # Find the bin index
    array_to_search = Base.ifelse(param_eval_function == eval_forward_rqs_params_with_grad, w, h)

    k1 = searchsortedfirst_impl(view(array_to_search, :, i, j), x[i,j]) - 1
    k2 = one(typeof(k1))

    # Is inside of range
    isinside = (1 <= k1 <= K)
    k = Base.ifelse(isinside, k1, k2)

    x_tmp = Base.ifelse(isinside, x[i,j], w[k,i,j]) # Simplifies calculations

    (yᵢⱼ, LogJacᵢⱼ, ∂y∂w, ∂y∂h, ∂y∂d, ∂LogJac∂w, ∂LogJac∂h, ∂LogJac∂d) = param_eval_function(w[k,i,j], w[k+1,i,j], h[k,i,j], h[k+1,i,j], d[k,i,j], d[k+1,i,j], x_tmp)

    y[i,j] = Base.ifelse(isinside, yᵢⱼ, x[i,j]) 
    LogJac[i,j] = Base.ifelse(isinside, LogJacᵢⱼ, zero(typeof(LogJacᵢⱼ)))

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


"""
    eval_forward_rqs_params_with_grad(wₖ::M0, wₖ₊₁::M0, hₖ::M1, hₖ₊₁::M1, dₖ::M2, dₖ₊₁::M2, x::M3) where {M0<:Real,M1<:Real, M2<:Real, M3<:Real}

Apply a rational quadratic spline segment to `x`, calculate the logarithm of the absolute value of the derivative ("LogJac") of the segment at `x`, 
and compute the gradient of that segment and the LogJac with respect to the spline parameters.

# Arguments
- `wₖ`, `wₖ₊₁`: The width parameters of the spline segment at the edges of the `k`-th interval.
- `hₖ`, `hₖ₊₁`: The height parameters of the spline segment.
- `dₖ`, `dₖ₊₁`: The derivative parameters of the spline segment.
- `x`: The value at which the spline function is evaluated.

# Returns
- `y`: The transformed value after applying the rational quadratic spline segment to `x`.
- `logJac`: The logarithm of the absolute value of the derivative of the segment at `x`.
- `∂y∂w`, `∂y∂h`, `∂y∂d`: The gradients of `y` with respect to the two width, height, and derivative parameters, respectively.
- `∂LogJac∂w`, `∂LogJac∂h`, `∂LogJac∂d`: The gradients of `logJac` with respect to the two width, height, and derivative parameters, respectively.
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

    # dy / dd_k+1
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


"""
    eval_backward_rqs_params_with_grad(wₖ::M0, wₖ₊₁::M0, 
                                       hₖ::M1, hₖ₊₁::M1, 
                                       dₖ::M2, dₖ₊₁::M2, 
                                       x::M3            ) where {M0<:Real,M1<:Real, M2<:Real, M3<:Real}

Apply an inverse rational quadratic spline segment to `x`, calculate the logarithm of the absolute value of the derivative ("LogJac") of the segment at `x`, 
and compute the gradient of that segment and the LogJac with respect to the spline parameters.

# Arguments
- `wₖ`, `wₖ₊₁`: The width parameters of the spline segment at the edges of the `k`-th interval.
- `hₖ`, `hₖ₊₁`: The height parameters of the spline segment.
- `dₖ`, `dₖ₊₁`: The derivative parameters of the spline segment.
- `x`: The value at which the spline function is evaluated.

# Returns
- `y`: The transformed value after applying the inverse rational quadratic spline segment to `x`.
- `logJac`: The logarithm of the absolute value of the derivative of the segment at `x`.
- `∂y∂w`, `∂y∂h`, `∂y∂d`: The gradients of `y` with respect to the two width, height, and derivative parameters, respectively.
- `∂LogJac∂w`, `∂LogJac∂h`, `∂LogJac∂d`: The gradients of `LogJac` with respect to the two width, height, and derivative parameters, respectively.
"""
function eval_backward_rqs_params_with_grad(
    wₖ::M0, wₖ₊₁::M0, 
    hₖ::M1, hₖ₊₁::M1, 
    dₖ::M2, dₖ₊₁::M2, 
    x::M3) where {M0<:Real,M1<:Real, M2<:Real, M3<:Real}

    Δy = hₖ₊₁ - hₖ
    Δy2 = x - hₖ
    Δx = wₖ₊₁ - wₖ
    sk = Δy / Δx

    κ = Δy * (sk - dₖ) + Δy2 * (dₖ₊₁ + dₖ - 2*sk)
    β = Δy * dₖ - Δy2 * (dₖ₊₁ + dₖ - 2*sk)
    ζ = -sk * Δy2
    θ = sqrt(β*β - 4*κ*ζ)

    # Partial derivatives with respect to x
    ∂κ∂x = dₖ₊₁ + dₖ - 2*sk
    ∂β∂x = -∂κ∂x
    ∂ζ∂x = -sk
    ∂θ∂x = (1/θ) * (β * ∂β∂x - 2 * (∂κ∂x * ζ + κ * ∂ζ∂x))

    # Enumerator of -∂y∂x
    μ = 2 * Δx * (∂ζ∂x * (β + θ) - ζ * (∂β∂x + ∂θ∂x))

    # Partial derivatives with regard to wₖ, wₖ₊₁, hₖ, hₖ₊₁, dₖ, dₖ₊₁
    ∂κ∂wₖ   = (Δy^2 - 2Δy*Δy2) / Δx^2
    ∂κ∂hₖ   = 2 * Δy2 / Δx - dₖ₊₁
    ∂κ∂hₖ₊₁ = 2 * (Δy - Δy2) / Δx - dₖ
    ∂κ∂dₖ   = x - hₖ₊₁
    ∂κ∂dₖ₊₁ = Δy2
    
    ∂β∂wₖ   = 2 * Δy2 * Δy / Δx^2
    ∂β∂hₖ   = dₖ₊₁ - 2 * (Δy + Δy2) / Δx
    ∂β∂hₖ₊₁ = dₖ + 2 * Δy2 / Δx
    ∂β∂dₖ   = hₖ₊₁ - x
    ∂β∂dₖ₊₁ = -Δy2

    ∂ζ∂wₖ   = -Δy2 * Δy / Δx^2
    ∂ζ∂hₖ   = (Δy2 + Δy) / Δx
    ∂ζ∂hₖ₊₁ = -Δy2 / Δx

    ∂θ∂wₖ   = (1/θ) * (β * ∂β∂wₖ   - 2 * (∂κ∂wₖ   * ζ + κ * ∂ζ∂wₖ  ))
    ∂θ∂hₖ   = (1/θ) * (β * ∂β∂hₖ   - 2 * (∂κ∂hₖ   * ζ + κ * ∂ζ∂hₖ  ))
    ∂θ∂hₖ₊₁ = (1/θ) * (β * ∂β∂hₖ₊₁ - 2 * (∂κ∂hₖ₊₁ * ζ + κ * ∂ζ∂hₖ₊₁))
    ∂θ∂dₖ   = (1/θ) * (β * ∂β∂dₖ   - 2 *  ∂κ∂dₖ   * ζ)
    ∂θ∂dₖ₊₁ = (1/θ) * (β * ∂β∂dₖ₊₁ - 2 *  ∂κ∂dₖ₊₁ * ζ)

    ∂κ∂x∂wₖ   = -2 * Δy / Δx^2
    ∂κ∂x∂hₖ   =  2 / Δx
    ∂κ∂x∂hₖ₊₁ = -∂κ∂x∂hₖ

    ∂β∂x∂wₖ   = 2 * Δy / Δx^2
    ∂β∂x∂hₖ   = -2 / Δx
    ∂β∂x∂hₖ₊₁ =  -∂β∂x∂hₖ

    ∂ζ∂x∂wₖ   = -Δy / Δx^2
    ∂ζ∂x∂hₖ   =  1 / Δx
    ∂ζ∂x∂hₖ₊₁ = -∂ζ∂x∂hₖ

    ∂θ∂x∂wₖ   = (1 / θ) * (((∂β∂wₖ   * ∂β∂x + β * ∂β∂x∂wₖ  ) - 2 * (∂κ∂x∂wₖ   * ζ + ∂κ∂x * ∂ζ∂wₖ   + ∂κ∂wₖ   * ∂ζ∂x + κ * ∂ζ∂x∂wₖ  )) - ∂θ∂wₖ   * ∂θ∂x)
    ∂θ∂x∂hₖ   = (1 / θ) * (((∂β∂hₖ   * ∂β∂x + β * ∂β∂x∂hₖ  ) - 2 * (∂κ∂x∂hₖ   * ζ + ∂κ∂x * ∂ζ∂hₖ   + ∂κ∂hₖ   * ∂ζ∂x + κ * ∂ζ∂x∂hₖ  )) - ∂θ∂hₖ   * ∂θ∂x)
    ∂θ∂x∂hₖ₊₁ = (1 / θ) * (((∂β∂hₖ₊₁ * ∂β∂x + β * ∂β∂x∂hₖ₊₁) - 2 * (∂κ∂x∂hₖ₊₁ * ζ + ∂κ∂x * ∂ζ∂hₖ₊₁ + ∂κ∂hₖ₊₁ * ∂ζ∂x + κ * ∂ζ∂x∂hₖ₊₁)) - ∂θ∂hₖ₊₁ * ∂θ∂x)
    ∂θ∂x∂dₖ   = (1 / θ) * (((∂β∂dₖ   * ∂β∂x - β) - 2 * (ζ + ∂κ∂dₖ   * ∂ζ∂x)) - ∂θ∂dₖ   * ∂θ∂x)
    ∂θ∂x∂dₖ₊₁ = (1 / θ) * (((∂β∂dₖ₊₁ * ∂β∂x - β) - 2 * (ζ + ∂κ∂dₖ₊₁ * ∂ζ∂x)) - ∂θ∂dₖ₊₁ * ∂θ∂x)

    ∂μ∂wₖ   = 2 * (Δx * (∂ζ∂x∂wₖ   * (β + θ) + ∂ζ∂x * (∂β∂wₖ   + ∂θ∂wₖ)   - ∂ζ∂wₖ   * (∂β∂x + ∂θ∂x) - ζ * (∂β∂x∂wₖ   + ∂θ∂x∂wₖ)) - μ / (2*Δx))
    ∂μ∂hₖ   = 2 * (Δx * (∂ζ∂x∂hₖ   * (β + θ) + ∂ζ∂x * (∂β∂hₖ   + ∂θ∂hₖ)   - ∂ζ∂hₖ   * (∂β∂x + ∂θ∂x) - ζ * (∂β∂x∂hₖ   + ∂θ∂x∂hₖ)))
    ∂μ∂hₖ₊₁ = 2 * (Δx * (∂ζ∂x∂hₖ₊₁ * (β + θ) + ∂ζ∂x * (∂β∂hₖ₊₁ + ∂θ∂hₖ₊₁) - ∂ζ∂hₖ₊₁ * (∂β∂x + ∂θ∂x) - ζ * (∂β∂x∂hₖ₊₁ + ∂θ∂x∂hₖ₊₁)))
    ∂μ∂dₖ   = 2 * (Δx * (∂ζ∂x * (∂β∂dₖ   + ∂θ∂dₖ)   - ζ * (∂θ∂x∂dₖ   - 1)))
    ∂μ∂dₖ₊₁ = 2 * (Δx * (∂ζ∂x * (∂β∂dₖ₊₁ + ∂θ∂dₖ₊₁) - ζ * (∂θ∂x∂dₖ₊₁ - 1)))

    ∂y∂wₖ   = 2 * (Δx * ((ζ * (∂β∂wₖ   + ∂θ∂wₖ  )) / (β + θ)^2 - ∂ζ∂wₖ   / (β + θ)) + ζ/(β + θ)) + 1
    ∂y∂wₖ₊₁ = -∂y∂wₖ + 1
    ∂y∂hₖ   = 2 * (Δx * ((ζ * (∂β∂hₖ   + ∂θ∂hₖ  )) / (β + θ)^2 - ∂ζ∂hₖ   / (β + θ)))
    ∂y∂hₖ₊₁ = 2 * (Δx * ((ζ * (∂β∂hₖ₊₁ + ∂θ∂hₖ₊₁)) / (β + θ)^2 - ∂ζ∂hₖ₊₁ / (β + θ)))
    ∂y∂dₖ   = 2 *  Δx *  (ζ * (∂β∂dₖ   + ∂θ∂dₖ  )) / (β + θ)^2 
    ∂y∂dₖ₊₁ = 2 *  Δx *  (ζ * (∂β∂dₖ₊₁ + ∂θ∂dₖ₊₁)) / (β + θ)^2 

    ∂LogJac∂wₖ   = (1 / μ) * ∂μ∂wₖ   - (2 / (β + θ)) * (∂β∂wₖ   + ∂θ∂wₖ)
    ∂LogJac∂wₖ₊₁ = -∂LogJac∂wₖ
    ∂LogJac∂hₖ   = (1 / μ) * ∂μ∂hₖ   - (2 / (β + θ)) * (∂β∂hₖ   + ∂θ∂hₖ)
    ∂LogJac∂hₖ₊₁ = (1 / μ) * ∂μ∂hₖ₊₁ - (2 / (β + θ)) * (∂β∂hₖ₊₁ + ∂θ∂hₖ₊₁)
    ∂LogJac∂dₖ   = (1 / μ) * ∂μ∂dₖ   - (2 / (β + θ)) * (∂β∂dₖ   + ∂θ∂dₖ)
    ∂LogJac∂dₖ₊₁ = (1 / μ) * ∂μ∂dₖ₊₁ - (2 / (β + θ)) * (∂β∂dₖ₊₁ + ∂θ∂dₖ₊₁)

    # Transformed output
    y =  wₖ - 2 * (ζ / (β + θ)) * Δx

    # LogJacobian, logaritm of the absolute value of ∂y/∂x
    LogJac = log(abs(μ)) - 2*log(abs(β + θ))

    ∂y∂w = (∂y∂wₖ, ∂y∂wₖ₊₁)
    ∂y∂h = (∂y∂hₖ, ∂y∂hₖ₊₁)
    ∂y∂d = (∂y∂dₖ, ∂y∂dₖ₊₁)

    ∂LogJac∂w = (∂LogJac∂wₖ, ∂LogJac∂wₖ₊₁)
    ∂LogJac∂h = (∂LogJac∂hₖ, ∂LogJac∂hₖ₊₁)
    ∂LogJac∂d = (∂LogJac∂dₖ, ∂LogJac∂dₖ₊₁)

    return y, LogJac, ∂y∂w, ∂y∂h, ∂y∂d, ∂LogJac∂w, ∂LogJac∂h, ∂LogJac∂d
end
