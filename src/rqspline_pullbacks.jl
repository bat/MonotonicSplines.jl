# This file is a part of MonotonicSplines.jl, licensed under the MIT License (MIT).

# Non-public:
#=
    rqs_pullback(param_eval_function::Function, x::AbstractArray{<:Real}, pX::AbstractArray{<:Real}, pY::AbstractArray{<:Real}, dYdX::AbstractArray{<:Real}, tangent_1::AbstractArray, tangent_2::AbstractArray)

Compute the gradients of the rational quadratic spline functions characterized by `pX`, `pY`, and `dYdX`, evaluated at the values in `x` with respect to `pX`, `pY`, and `dYdX`. 

This function is designed to make the transformation using Rational Quadratic Splines in this package automatically differentiable.
Whether the gradients of the forward or inverse spline functions are calculated is determined by the `param_eval_function` argument.

# Arguments
- `param_eval_function`: The function used to evaluate a spline segment. Different functions are used for the forward and inverse spline functions.
- `x`: An array of real numbers at which the spline functions are evaluated.
- `pY`, `pY`, `dYdX`: Arrays that hold the width, height, and derivative parameters of the spline functions, respectively.
- `tangent_1`, `tangent_2`: Arrays that hold the tangent vectors for the transformed output and the log abs det jacobians respectively.

# Returns
Three values are returned:
- `∂y∂pX + ∂LogJac∂pX`: An array with the same shape as `pY`, with the `[:,j,k]`-th element holding the gradient of the `[j,k]`-th element of `y` with respect to the width parameters
                      plus the gradient of the logarithm of the absolute value of the derivative of this `[j,k]`-th element of `y` with respect to the `[j,k]`-th element of `x`, with respect to the width parameters. 
                      For Example, the `[i,j,k]` element of this array is `∂yⱼₖ/∂pXᵢⱼₖ + ∂(log(abs(∂yⱼₖ/∂xⱼₖ)))/∂pXᵢⱼₖ`.
- `∂y∂pY + ∂LogJac∂pY`: An array with the same shape as `pY`, holding the same gradients as described above, but with respect to the height parameters.
- `∂y∂dYdX + ∂LogJac∂dYdX`: An array with the same shape as `dYdX`, holding the same gradients as described above, but with respect to the derivative parameters.

# Note
Since only one segment of each spline is evaluated for one element of `x`, the returned gradients are filled with zeros, except for the parameters of the segment that is evaluated for that element of `x`.
For example, if the `[j,k]` -th element of `x` falls in the `l`-th bin of the interval mask, the `[:,j,k]` entries in `∂y∂pX + ∂LogJac∂pX` are all zero, except the `[l,j,k]` and `[l+1,j,k]` elements, which hold 
the (generally non-zero) values `∂yⱼₖ/∂pXₗⱼₖ + ∂(log(abs(∂yⱼₖ/∂xⱼₖ)))/∂pXₗⱼₖ` and `∂yⱼₖ/∂pXₗ₊₁ⱼₖ + ∂(log(abs(∂yⱼₖ/∂xⱼₖ)))/∂pXₗ₊₁ⱼₖ` respectively.

The function executes in a kernel, on the same backend as `x` is stored (CPU or GPU), and the output is also returned on the same backend.
=#
function rqs_pullback(
    param_eval_function::Function,
    x::AbstractArray{<:Real},
    pX::AbstractArray{<:Real},
    pY::AbstractArray{<:Real},
    dYdX::AbstractArray{<:Real},
    tangent_1::AbstractArray{<:Real},
    tangent_2::AbstractArray{<:Real};
) 
    compute_unit = get_compute_unit(x)
    backend = ka_backend(compute_unit)
    kernel! = rqs_pullback_kernel!(backend, _ka_threads(backend)...)

    y = similar(x)
    logJac = similar(x)

    ∂y∂pX = fill!(similar(pX), zero(eltype(pX)))
    ∂y∂dYdX = fill!(similar(pY), zero(eltype(pY)))
    ∂y∂pY = fill!(similar(dYdX), zero(eltype(dYdX)))

    ∂LogJac∂pX = fill!(similar(pX), zero(eltype(pX)))
    ∂LogJac∂pY = fill!(similar(pY), zero(eltype(pY)))
    ∂LogJac∂dYdX = fill!(similar(dYdX), zero(eltype(dYdX)))

    kernel!(
        param_eval_function,
        x, y, logJac, 
        pX, pY, dYdX,
        ∂y∂pX, ∂y∂pY, ∂y∂dYdX,
        ∂LogJac∂pX, ∂LogJac∂pY, ∂LogJac∂dYdX, 
        tangent_1,
        tangent_2,
        ndrange=size(x)
        )

    logJac = sum(logJac, dims=1)

    return ∂y∂pX + ∂LogJac∂pX, ∂y∂pY + ∂LogJac∂pY, ∂y∂dYdX + ∂LogJac∂dYdX
end


# Non-public:
#=
    rqs_pullback_kernel(
        param_eval_function::Function,
        x::AbstractArray,
        y::AbstractArray,
        logJac::AbstractArray,
        pXw::AbstractArray,
        pY::AbstractArray,
        dYdX::AbstractArray,
        ∂y∂pX_tangent::AbstractArray,
        ∂y∂pY_tangent::AbstractArray,
        ∂y∂dYdX_tangent::AbstractArray,
        ∂LogJac∂pX_tangent::AbstractArray,
        ∂LogJac∂pY_tangent::AbstractArray,
        ∂LogJac∂dYdX_tangent::AbstractArray,
        tangent_1::AbstractArray,
        tangent_2::AbstractArray,
    )

This kernel function calculates the gradients of the rational quadratic spline functions characterized by `pX`, `pY`, and `dYdX`, evaluated at the values in `x` and of `logJac`.

# Arguments
- `param_eval_function` The function used to evaluate a spline segment. Different functions are used for the forward and inverse passes.
- `x`: An array of real numbers to which the spline functions are applied.
- `pX`, `pY`, `dYdX`: Arrays that hold the width, height, and derivative parameters of the spline functions, respectively.
- `y`: An array where the transformed values are stored.
- `logJac`: An array where the sums of the values of the logarithm of the absolute values of the determinant of the Jacobians of the spline functions applied to a column of `x` are stored.
- `∂y∂pX_tangent`, `∂y∂pY_tangent`, `∂y∂dYdX_tangent`: Arrays that will contain the gradients of the spline functions with respect to `pX`, `pY`, and `dYdX`, respectively.
- `∂LogJac∂pX_tangent`, `∂LogJac∂pY_tangent`, `∂LogJac∂dYdX_tangent`: Arrays that will contain the gradients of `logJac` with respect to `pX`, `pY`, and `dYdX`, respectively.
- `tangent_1`, `tangent_2`: Arrays that hold the tangent vectors for the forward pass.

For an explanation of the shape and contents of the gradient arrays, see the documentation of the [`rqs_pullback()`](@ref) function.

# Note
This function is a kernel function and is used within the `rqs_forward_pullback` function to calculate the gradients of the spline functions and `logJac`. It is not intended to be called directly by the user.
=#
@kernel function rqs_pullback_kernel!(
        param_eval_function::Function,
        x::AbstractArray{<:Real},
        y::AbstractArray{<:Real},
        LogJac::AbstractArray{<:Real},
        pX::AbstractArray{<:Real},
        pY::AbstractArray{<:Real},
        dYdX::AbstractArray{<:Real},
        ∂y∂pX_tangent::AbstractArray{<:Real},
        ∂y∂pY_tangent::AbstractArray{<:Real},
        ∂y∂dYdX_tangent::AbstractArray{<:Real},
        ∂LogJac∂pX_tangent::AbstractArray{<:Real},
        ∂LogJac∂pY_tangent::AbstractArray{<:Real},
        ∂LogJac∂dYdX_tangent::AbstractArray{<:Real},
        tangent_1::AbstractArray{<:Real},
        tangent_2::AbstractArray{<:Real},
    )

    i, j = @index(Global, NTuple)

    # minus one to account for left pad
    K = size(pX, 1) - 1

    # Find the bin index
    array_to_search = Base.ifelse(param_eval_function == eval_forward_rqs_params_with_grad, pX, pY)

    k1 = searchsortedfirst_impl(view(array_to_search, :, i, j), x[i,j]) - 1
    k2 = one(typeof(k1))

    # Is inside of range
    isinside = (1 <= k1 <= K)
    k = Base.ifelse(isinside, k1, k2)

    x_tmp = Base.ifelse(isinside, x[i,j], pX[k,i,j]) # Simplifies calculations

    (yᵢⱼ, LogJacᵢⱼ, ∂y∂pX, ∂y∂pY, ∂y∂dYdX, ∂LogJac∂pX, ∂LogJac∂pY, ∂LogJac∂dYdX) = param_eval_function(pX[k,i,j], pX[k+1,i,j], pY[k,i,j], pY[k+1,i,j], dYdX[k,i,j], dYdX[k+1,i,j], x_tmp)

    y[i,j] = Base.ifelse(isinside, yᵢⱼ, x[i,j]) 
    LogJac[i,j] = Base.ifelse(isinside, LogJacᵢⱼ, zero(typeof(LogJacᵢⱼ)))

    ∂y∂pX_tangent[k, i, j]      = tangent_1[i,j] * Base.ifelse(isinside, ∂y∂pX[1], zero(eltype(∂y∂pX)))
    ∂y∂pY_tangent[k, i, j]      = tangent_1[i,j] * Base.ifelse(isinside, ∂y∂pY[1], zero(eltype(∂y∂pY)))
    ∂y∂dYdX_tangent[k, i, j]      = tangent_1[i,j] * Base.ifelse(isinside, ∂y∂dYdX[1], zero(eltype(∂y∂dYdX)))
    ∂LogJac∂pX_tangent[k, i, j] = tangent_2[1,j] * Base.ifelse(isinside, ∂LogJac∂pX[1], zero(eltype(∂LogJac∂pX)))
    ∂LogJac∂pY_tangent[k, i, j] = tangent_2[1,j] * Base.ifelse(isinside, ∂LogJac∂pY[1], zero(eltype(∂LogJac∂pY)))
    ∂LogJac∂dYdX_tangent[k, i, j] = tangent_2[1,j] * Base.ifelse(isinside, ∂LogJac∂dYdX[1], zero(eltype(∂LogJac∂dYdX)))

    ∂y∂pX_tangent[k+1, i, j]       = tangent_1[i,j] * Base.ifelse(isinside, ∂y∂pX[2], zero(eltype(∂y∂pX)))
    ∂y∂pY_tangent[k+1, i, j]       = tangent_1[i,j] * Base.ifelse(isinside, ∂y∂pY[2], zero(eltype(∂y∂pY)))
    ∂y∂dYdX_tangent[k+1, i, j]       = tangent_1[i,j] * Base.ifelse(isinside, ∂y∂dYdX[2], zero(eltype(∂y∂dYdX)))
    ∂LogJac∂pX_tangent[k+1, i, j]  = tangent_2[1,j] * Base.ifelse(isinside, ∂LogJac∂pX[2], zero(eltype(∂LogJac∂pX)))
    ∂LogJac∂pY_tangent[k+1, i, j]  = tangent_2[1,j] * Base.ifelse(isinside, ∂LogJac∂pY[2], zero(eltype(∂LogJac∂pY)))
    ∂LogJac∂dYdX_tangent[k+1, i, j]  = tangent_2[1,j] * Base.ifelse(isinside, ∂LogJac∂dYdX[2], zero(eltype(∂LogJac∂dYdX)))
end


# Non-public:
#=
    eval_forward_rqs_params_with_grad(pXₖ::M0, pXₖ₊₁::M0, pYₖ::M1, pYₖ₊₁::M1, dYdXₖ::M2, dYdXₖ₊₁::M2, x::M3) where {M0<:Real,M1<:Real, M2<:Real, M3<:Real}

Apply a rational quadratic spline segment to `x`, calculate the logarithm of the absolute value of the derivative ("LogJac") of the segment at `x`, 
and compute the gradient of that segment and the LogJac with respect to the spline parameters.

# Arguments
- `pXₖ`, `pXₖ₊₁`: The width parameters of the spline segment at the edges of the `k`-th interval.
- `pYₖ`, `pYₖ₊₁`: The height parameters of the spline segment.
- `dYdXₖ`, `dYdXₖ₊₁`: The derivative parameters of the spline segment.
- `x`: The value at which the spline function is evaluated.

# Returns
- `y`: The transformed value after applying the rational quadratic spline segment to `x`.
- `logJac`: The logarithm of the absolute value of the derivative of the segment at `x`.
- `∂y∂pX`, `∂y∂pY`, `∂y∂dYdX`: The gradients of `y` with respect to the two width, height, and derivative parameters, respectively.
- `∂LogJac∂pX`, `∂LogJac∂pY`, `∂LogJac∂dYdX`: The gradients of `logJac` with respect to the two width, height, and derivative parameters, respectively.
=#
function eval_forward_rqs_params_with_grad(
    pXₖ::M0, pXₖ₊₁::M0, 
    pYₖ::M1, pYₖ₊₁::M1, 
    dYdXₖ::M2, dYdXₖ₊₁::M2, 
    x::M3) where {M0<:Real,M1<:Real, M2<:Real, M3<:Real}

    Δy = pYₖ₊₁ - pYₖ
    Δx = pXₖ₊₁ - pXₖ
    sk = Δy / Δx
    ξ = (x - pXₖ) / Δx

    denom = (sk + (dYdXₖ₊₁ + dYdXₖ - 2*sk)*ξ*(1-ξ))
    nom_1 =  sk*ξ*ξ + dYdXₖ*ξ*(1-ξ)
    nom_2 = Δy * nom_1
    nom_3 = dYdXₖ₊₁*ξ*ξ + 2*sk*ξ*(1-ξ) + dYdXₖ*(1-ξ)^2
    nom_4 = sk*sk*nom_3

    y = pYₖ + nom_2/denom

    # LogJacobian
    logJac = log(abs(nom_4))-2*log(abs(denom))

    # Gradient of parameters:

    # dy / dw_k
    ∂s∂pXₖ = Δy/Δx^2
    ∂ξ∂pXₖ = (-Δx + x - pXₖ)/Δx^2
    ∂y∂pXₖ = (Δy / denom^2) * ((∂s∂pXₖ*ξ^2 + 2*sk*ξ*∂ξ∂pXₖ + dYdXₖ*(∂ξ∂pXₖ -
                2*ξ*∂ξ∂pXₖ))*denom - nom_1*(∂s∂pXₖ - 2*∂s∂pXₖ*ξ*(1-ξ) + (dYdXₖ₊₁ + dYdXₖ - 2*sk)*(∂ξ∂pXₖ - 2*ξ*∂ξ∂pXₖ)) )
    ∂LogJac∂pXₖ = (1/nom_4)*(2*sk*∂s∂pXₖ*nom_3 + sk*sk*(2*dYdXₖ₊₁*ξ*∂ξ∂pXₖ + 2*∂s∂pXₖ*ξ*(1-ξ)+2*sk*(∂ξ∂pXₖ - 2*ξ*∂ξ∂pXₖ)-dYdXₖ*2*(1-ξ)*∂ξ∂pXₖ)) - (2/denom)*(∂s∂pXₖ - 2*∂s∂pXₖ*ξ*(1-ξ) + (dYdXₖ₊₁ + dYdXₖ - 2*sk)*(∂ξ∂pXₖ - 2*ξ*∂ξ∂pXₖ))

    # dy / dw_k+1
    ∂s∂pXₖ₊₁ = -Δy/Δx^2
    ∂ξ∂pXₖ₊₁ = -(x - pXₖ) / Δx^2
    ∂y∂pXₖ₊₁ = (Δy / denom^2) * ((∂s∂pXₖ₊₁*ξ^2 + 2*sk*ξ*∂ξ∂pXₖ₊₁ + dYdXₖ*(∂ξ∂pXₖ₊₁ -
                2*ξ*∂ξ∂pXₖ₊₁))*denom - nom_1*(∂s∂pXₖ₊₁ - 2*∂s∂pXₖ₊₁*ξ*(1-ξ) + (dYdXₖ₊₁ + dYdXₖ - 2*sk)*(∂ξ∂pXₖ₊₁ - 2*ξ*∂ξ∂pXₖ₊₁)) )
    ∂LogJac∂pXₖ₊₁ = (1/nom_4)*(2*sk*∂s∂pXₖ₊₁*nom_3 + sk*sk*(2*dYdXₖ₊₁*ξ*∂ξ∂pXₖ₊₁ + 2*∂s∂pXₖ₊₁*ξ*(1-ξ)+2*sk*(∂ξ∂pXₖ₊₁ - 2*ξ*∂ξ∂pXₖ₊₁)-dYdXₖ*2*(1-ξ)*∂ξ∂pXₖ₊₁)) - (2/denom)*(∂s∂pXₖ₊₁ - 2*∂s∂pXₖ₊₁*ξ*(1-ξ) + (dYdXₖ₊₁ + dYdXₖ - 2*sk)*(∂ξ∂pXₖ₊₁ - 2*ξ*∂ξ∂pXₖ₊₁))

    # dy / dh_k
    ∂s∂pYₖ = -1/Δx
    ∂y∂pYₖ = 1 + (1/denom^2)*((-nom_1+Δy*ξ*ξ*∂s∂pYₖ)*denom - nom_2 * (∂s∂pYₖ - 2*∂s∂pYₖ*ξ*(1-ξ)) )
    ∂LogJac∂pYₖ = (1/nom_4)*(2*sk*∂s∂pYₖ*nom_3 + sk*sk*2*∂s∂pYₖ*ξ*(1-ξ)) - (2/denom)*(∂s∂pYₖ - 2*∂s∂pYₖ*ξ*(1-ξ))

    # dy / dh_k+1
    ∂s∂pYₖ₊₁ = 1/Δx
    ∂y∂pYₖ₊₁ = (1/denom^2)*((nom_1+Δy*ξ*ξ*∂s∂pYₖ₊₁)*denom - nom_2 * (∂s∂pYₖ₊₁ - 2*∂s∂pYₖ₊₁*ξ*(1-ξ)) )
    ∂LogJac∂pYₖ₊₁ = (1/nom_4)*(2*sk*∂s∂pYₖ₊₁*nom_3 + sk*sk*2*∂s∂pYₖ₊₁*ξ*(1-ξ)) - (2/denom)*(∂s∂pYₖ₊₁ - 2*∂s∂pYₖ₊₁*ξ*(1-ξ))

    # dy / dd_k
    ∂y∂dYdXₖ = (1/denom^2) * ((Δy*ξ*(1-ξ))*denom - nom_2*ξ*(1-ξ) )
    ∂LogJac∂dYdXₖ = (1/nom_4)*sk^2*(1-ξ)^2 - (2/denom)*ξ*(1-ξ)

    # dy / dd_k+1
    ∂y∂dYdXₖ₊₁ = -(nom_2/denom^2) * ξ*(1-ξ)
    ∂LogJac∂dYdXₖ₊₁ = (1/nom_4)*sk^2*ξ^2 - (2/denom)*ξ*(1-ξ)

    ∂y∂pX = (∂y∂pXₖ, ∂y∂pXₖ₊₁)
    ∂y∂pY = (∂y∂pYₖ, ∂y∂pYₖ₊₁)
    ∂y∂dYdX = (∂y∂dYdXₖ, ∂y∂dYdXₖ₊₁)

    ∂LogJac∂pX = (∂LogJac∂pXₖ, ∂LogJac∂pXₖ₊₁)
    ∂LogJac∂pY = (∂LogJac∂pYₖ, ∂LogJac∂pYₖ₊₁)
    ∂LogJac∂dYdX = (∂LogJac∂dYdXₖ, ∂LogJac∂dYdXₖ₊₁)

    return y, logJac, ∂y∂pX, ∂y∂pY, ∂y∂dYdX, ∂LogJac∂pX, ∂LogJac∂pY, ∂LogJac∂dYdX
end


# Non-public:
#=
    eval_inverse_rqs_params_with_grad(pXₖ::M0, pXₖ₊₁::M0, 
                                       pYₖ::M1, pYₖ₊₁::M1, 
                                       dYdXₖ::M2, dYdXₖ₊₁::M2, 
                                       x::M3            ) where {M0<:Real,M1<:Real, M2<:Real, M3<:Real}

Apply an inverse rational quadratic spline segment to `x`, calculate the logarithm of the absolute value of the derivative ("LogJac") of the segment at `x`, 
and compute the gradient of that segment and the LogJac with respect to the spline parameters.

# Arguments
- `pXₖ`, `pXₖ₊₁`: The width parameters of the spline segment at the edges of the `k`-th interval.
- `pYₖ`, `pYₖ₊₁`: The height parameters of the spline segment.
- `dYdXₖ`, `dYdXₖ₊₁`: The derivative parameters of the spline segment.
- `x`: The value at which the spline function is evaluated.

# Returns
- `y`: The transformed value after applying the inverse rational quadratic spline segment to `x`.
- `logJac`: The logarithm of the absolute value of the derivative of the segment at `x`.
- `∂y∂pX`, `∂y∂pY`, `∂y∂dYdX`: The gradients of `y` with respect to the two width, height, and derivative parameters, respectively.
- `∂LogJac∂pX`, `∂LogJac∂pY`, `∂LogJac∂dYdX`: The gradients of `LogJac` with respect to the two width, height, and derivative parameters, respectively.
=#
function eval_inverse_rqs_params_with_grad(
    pXₖ::M0, pXₖ₊₁::M0, 
    pYₖ::M1, pYₖ₊₁::M1, 
    dYdXₖ::M2, dYdXₖ₊₁::M2, 
    x::M3) where {M0<:Real,M1<:Real, M2<:Real, M3<:Real}

    Δy = pYₖ₊₁ - pYₖ
    Δy2 = x - pYₖ
    Δx = pXₖ₊₁ - pXₖ
    sk = Δy / Δx

    κ = Δy * (sk - dYdXₖ) + Δy2 * (dYdXₖ₊₁ + dYdXₖ - 2*sk)
    β = Δy * dYdXₖ - Δy2 * (dYdXₖ₊₁ + dYdXₖ - 2*sk)
    ζ = -sk * Δy2
    θ = sqrt(β*β - 4*κ*ζ)

    # Partial derivatives with respect to x
    ∂κ∂x = dYdXₖ₊₁ + dYdXₖ - 2*sk
    ∂β∂x = -∂κ∂x
    ∂ζ∂x = -sk
    ∂θ∂x = (1/θ) * (β * ∂β∂x - 2 * (∂κ∂x * ζ + κ * ∂ζ∂x))

    # Enumerator of -∂y∂x
    μ = 2 * Δx * (∂ζ∂x * (β + θ) - ζ * (∂β∂x + ∂θ∂x))

    # Partial derivatives with regard to pXₖ, pXₖ₊₁, pYₖ, pYₖ₊₁, dYdXₖ, dYdXₖ₊₁
    ∂κ∂pXₖ   = (Δy^2 - 2Δy*Δy2) / Δx^2
    ∂κ∂pYₖ   = 2 * Δy2 / Δx - dYdXₖ₊₁
    ∂κ∂pYₖ₊₁ = 2 * (Δy - Δy2) / Δx - dYdXₖ
    ∂κ∂dYdXₖ   = x - pYₖ₊₁
    ∂κ∂dYdXₖ₊₁ = Δy2
    
    ∂β∂pXₖ   = 2 * Δy2 * Δy / Δx^2
    ∂β∂pYₖ   = dYdXₖ₊₁ - 2 * (Δy + Δy2) / Δx
    ∂β∂pYₖ₊₁ = dYdXₖ + 2 * Δy2 / Δx
    ∂β∂dYdXₖ   = pYₖ₊₁ - x
    ∂β∂dYdXₖ₊₁ = -Δy2

    ∂ζ∂pXₖ   = -Δy2 * Δy / Δx^2
    ∂ζ∂pYₖ   = (Δy2 + Δy) / Δx
    ∂ζ∂pYₖ₊₁ = -Δy2 / Δx

    ∂θ∂pXₖ   = (1/θ) * (β * ∂β∂pXₖ   - 2 * (∂κ∂pXₖ   * ζ + κ * ∂ζ∂pXₖ  ))
    ∂θ∂pYₖ   = (1/θ) * (β * ∂β∂pYₖ   - 2 * (∂κ∂pYₖ   * ζ + κ * ∂ζ∂pYₖ  ))
    ∂θ∂pYₖ₊₁ = (1/θ) * (β * ∂β∂pYₖ₊₁ - 2 * (∂κ∂pYₖ₊₁ * ζ + κ * ∂ζ∂pYₖ₊₁))
    ∂θ∂dYdXₖ   = (1/θ) * (β * ∂β∂dYdXₖ   - 2 *  ∂κ∂dYdXₖ   * ζ)
    ∂θ∂dYdXₖ₊₁ = (1/θ) * (β * ∂β∂dYdXₖ₊₁ - 2 *  ∂κ∂dYdXₖ₊₁ * ζ)

    ∂κ∂x∂pXₖ   = -2 * Δy / Δx^2
    ∂κ∂x∂pYₖ   =  2 / Δx
    ∂κ∂x∂pYₖ₊₁ = -∂κ∂x∂pYₖ

    ∂β∂x∂pXₖ   = 2 * Δy / Δx^2
    ∂β∂x∂pYₖ   = -2 / Δx
    ∂β∂x∂pYₖ₊₁ =  -∂β∂x∂pYₖ

    ∂ζ∂x∂pXₖ   = -Δy / Δx^2
    ∂ζ∂x∂pYₖ   =  1 / Δx
    ∂ζ∂x∂pYₖ₊₁ = -∂ζ∂x∂pYₖ

    ∂θ∂x∂pXₖ   = (1 / θ) * (((∂β∂pXₖ   * ∂β∂x + β * ∂β∂x∂pXₖ  ) - 2 * (∂κ∂x∂pXₖ   * ζ + ∂κ∂x * ∂ζ∂pXₖ   + ∂κ∂pXₖ   * ∂ζ∂x + κ * ∂ζ∂x∂pXₖ  )) - ∂θ∂pXₖ   * ∂θ∂x)
    ∂θ∂x∂pYₖ   = (1 / θ) * (((∂β∂pYₖ   * ∂β∂x + β * ∂β∂x∂pYₖ  ) - 2 * (∂κ∂x∂pYₖ   * ζ + ∂κ∂x * ∂ζ∂pYₖ   + ∂κ∂pYₖ   * ∂ζ∂x + κ * ∂ζ∂x∂pYₖ  )) - ∂θ∂pYₖ   * ∂θ∂x)
    ∂θ∂x∂pYₖ₊₁ = (1 / θ) * (((∂β∂pYₖ₊₁ * ∂β∂x + β * ∂β∂x∂pYₖ₊₁) - 2 * (∂κ∂x∂pYₖ₊₁ * ζ + ∂κ∂x * ∂ζ∂pYₖ₊₁ + ∂κ∂pYₖ₊₁ * ∂ζ∂x + κ * ∂ζ∂x∂pYₖ₊₁)) - ∂θ∂pYₖ₊₁ * ∂θ∂x)
    ∂θ∂x∂dYdXₖ   = (1 / θ) * (((∂β∂dYdXₖ   * ∂β∂x - β) - 2 * (ζ + ∂κ∂dYdXₖ   * ∂ζ∂x)) - ∂θ∂dYdXₖ   * ∂θ∂x)
    ∂θ∂x∂dYdXₖ₊₁ = (1 / θ) * (((∂β∂dYdXₖ₊₁ * ∂β∂x - β) - 2 * (ζ + ∂κ∂dYdXₖ₊₁ * ∂ζ∂x)) - ∂θ∂dYdXₖ₊₁ * ∂θ∂x)

    ∂μ∂pXₖ   = 2 * (Δx * (∂ζ∂x∂pXₖ   * (β + θ) + ∂ζ∂x * (∂β∂pXₖ   + ∂θ∂pXₖ)   - ∂ζ∂pXₖ   * (∂β∂x + ∂θ∂x) - ζ * (∂β∂x∂pXₖ   + ∂θ∂x∂pXₖ)) - μ / (2*Δx))
    ∂μ∂pYₖ   = 2 * (Δx * (∂ζ∂x∂pYₖ   * (β + θ) + ∂ζ∂x * (∂β∂pYₖ   + ∂θ∂pYₖ)   - ∂ζ∂pYₖ   * (∂β∂x + ∂θ∂x) - ζ * (∂β∂x∂pYₖ   + ∂θ∂x∂pYₖ)))
    ∂μ∂pYₖ₊₁ = 2 * (Δx * (∂ζ∂x∂pYₖ₊₁ * (β + θ) + ∂ζ∂x * (∂β∂pYₖ₊₁ + ∂θ∂pYₖ₊₁) - ∂ζ∂pYₖ₊₁ * (∂β∂x + ∂θ∂x) - ζ * (∂β∂x∂pYₖ₊₁ + ∂θ∂x∂pYₖ₊₁)))
    ∂μ∂dYdXₖ   = 2 * (Δx * (∂ζ∂x * (∂β∂dYdXₖ   + ∂θ∂dYdXₖ)   - ζ * (∂θ∂x∂dYdXₖ   - 1)))
    ∂μ∂dYdXₖ₊₁ = 2 * (Δx * (∂ζ∂x * (∂β∂dYdXₖ₊₁ + ∂θ∂dYdXₖ₊₁) - ζ * (∂θ∂x∂dYdXₖ₊₁ - 1)))

    ∂y∂pXₖ   = 2 * (Δx * ((ζ * (∂β∂pXₖ   + ∂θ∂pXₖ  )) / (β + θ)^2 - ∂ζ∂pXₖ   / (β + θ)) + ζ/(β + θ)) + 1
    ∂y∂pXₖ₊₁ = -∂y∂pXₖ + 1
    ∂y∂pYₖ   = 2 * (Δx * ((ζ * (∂β∂pYₖ   + ∂θ∂pYₖ  )) / (β + θ)^2 - ∂ζ∂pYₖ   / (β + θ)))
    ∂y∂pYₖ₊₁ = 2 * (Δx * ((ζ * (∂β∂pYₖ₊₁ + ∂θ∂pYₖ₊₁)) / (β + θ)^2 - ∂ζ∂pYₖ₊₁ / (β + θ)))
    ∂y∂dYdXₖ   = 2 *  Δx *  (ζ * (∂β∂dYdXₖ   + ∂θ∂dYdXₖ  )) / (β + θ)^2 
    ∂y∂dYdXₖ₊₁ = 2 *  Δx *  (ζ * (∂β∂dYdXₖ₊₁ + ∂θ∂dYdXₖ₊₁)) / (β + θ)^2 

    ∂LogJac∂pXₖ   = (1 / μ) * ∂μ∂pXₖ   - (2 / (β + θ)) * (∂β∂pXₖ   + ∂θ∂pXₖ)
    ∂LogJac∂pXₖ₊₁ = -∂LogJac∂pXₖ
    ∂LogJac∂pYₖ   = (1 / μ) * ∂μ∂pYₖ   - (2 / (β + θ)) * (∂β∂pYₖ   + ∂θ∂pYₖ)
    ∂LogJac∂pYₖ₊₁ = (1 / μ) * ∂μ∂pYₖ₊₁ - (2 / (β + θ)) * (∂β∂pYₖ₊₁ + ∂θ∂pYₖ₊₁)
    ∂LogJac∂dYdXₖ   = (1 / μ) * ∂μ∂dYdXₖ   - (2 / (β + θ)) * (∂β∂dYdXₖ   + ∂θ∂dYdXₖ)
    ∂LogJac∂dYdXₖ₊₁ = (1 / μ) * ∂μ∂dYdXₖ₊₁ - (2 / (β + θ)) * (∂β∂dYdXₖ₊₁ + ∂θ∂dYdXₖ₊₁)

    # Transformed output
    y =  pXₖ - 2 * (ζ / (β + θ)) * Δx

    # LogJacobian, logaritm of the absolute value of ∂y/∂x
    LogJac = log(abs(μ)) - 2*log(abs(β + θ))

    ∂y∂pX = (∂y∂pXₖ, ∂y∂pXₖ₊₁)
    ∂y∂pY = (∂y∂pYₖ, ∂y∂pYₖ₊₁)
    ∂y∂dYdX = (∂y∂dYdXₖ, ∂y∂dYdXₖ₊₁)

    ∂LogJac∂pX = (∂LogJac∂pXₖ, ∂LogJac∂pXₖ₊₁)
    ∂LogJac∂pY = (∂LogJac∂pYₖ, ∂LogJac∂pYₖ₊₁)
    ∂LogJac∂dYdX = (∂LogJac∂dYdXₖ, ∂LogJac∂dYdXₖ₊₁)

    return y, LogJac, ∂y∂pX, ∂y∂pY, ∂y∂dYdX, ∂LogJac∂pX, ∂LogJac∂pY, ∂LogJac∂dYdX
end
