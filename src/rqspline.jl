# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).
"""
    RQSpline(widths::AbstractArray{<:Real}, heights::AbstractArray{<:Real}, derivatives::AbstractArray{<:Real})

Object holding the parameters to characterize several rational quadratic spline functions after the scheme 
first defined by Gregory and Delbourgo in https://doi.org/10.1093/imanum/2.2.123.
`RQSpline` holds the parameters to characterize `n_dims x n_samples` spline functions to transform 
each component of `n_samples` samples.

`widths`, `heights`, and `derivatives` are `K+1 x n_dims x n_samples` arrays, 
with the parameters to characterize a single spline function with `K` segments in the first dimension. 
Along the second dimension, the spline functions for a single sample are stored, and along the third dimension the sets 
splines for differen samples.
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
    RQSplineInv(widths::AbstractArray{<:Real}, heights::AbstractArray{<:Real}, derivatives::AbstractArray{<:Real})

Object holding the parameters to characterize several inverse rational quadratic spline functions analogous to `RQSplineInv`.

The same parameters are used to characterize the forward and inverse spline functions, the struct used to store them decides 
the equation they are evaluated in. 
"""
struct RQSplineInv <: Function
    widths::AbstractArray{<:Real}
    heights::AbstractArray{<:Real}
    derivatives::AbstractArray{<:Real}
end

@functor RQSplineInv
export RQSplineInv

(f::RQSplineInv)(x::AbstractMatrix{<:Real}) = spline_backward(f, x)[1]

function ChangesOfVariables.with_logabsdet_jacobian(
    f::RQSplineInv,
    x::AbstractMatrix{<:Real}
)
    return spline_backward(f, x)
end

# Transformation forward: 
"""
    spline_forward(trafo::RQSpline, x::AbstractMatrix{<:Real})

Apply the rational quadratic spline functions characterized by the parameters stored in `trafo` to the matrix `x`.
The spline function characterized by the parameters in the `[:,i,j]` entries in `trafo` is applied to the `[i,j]`-th element of `x`.
"""
function spline_forward(trafo::RQSpline, x::AbstractMatrix{<:Real})
    return rqs_forward(x, trafo.widths, trafo.heights, trafo.derivatives, trafo.widths, trafo.heights, trafo.derivatives)
end

"""
    rqs_forward(x::AbstractArray{<:Real}, w::AbstractArray{<:Real}, h::AbstractArray{<:Real}, d::AbstractArray{<:Real}, w_logJac::AbstractArray{<:Real}, h_logJac::AbstractArray{<:Real}, d_logJac::AbstractArray{<:Real})

Apply the rational quadratic spline functions characterized by `w`, `h`, and `d` to `x`. 
The spline function characterized by the parameters in the `[:,i,j]` entries in `trafo` is applied to the `[i,j]`-th element of `x`.

Return the transformed values in a matrix `y` of the same shape as `x`, and return a `1 x size(x,2)` -matrix holding the sums of the values 
of the logarithm of the absolute values of the determinant of the jacobians of the spline functions applied to a column of `x`.

The function executes in a kernel, on the same backend as `x` is stored (CPU or GPU), the output will also be returned on the same backend.
"""
function rqs_forward(
        x::AbstractArray{M0},
        w::AbstractArray{M1},
        h::AbstractArray{M2},
        d::AbstractArray{M3},
        w_logJac::AbstractArray{M4},
        h_logJac::AbstractArray{M5},
        d_logJac::AbstractArray{M6} 
    ) where {M0<:Real,M1<:Real, M2<:Real, M3<:Real, M4<:Real, M5<:Real, M6<:Real}

    T = promote_type(M0, M1, M2, M3, M4, M5, M6)
    ndims, nsmpls = size(x)

    device = KernelAbstractions.get_backend(x)
    n = device isa GPU ? 256 : Threads.nthreads()
    kernel! = rqs_forward_kernel!(device, n)

    y = device isa GPU ? gpu(zeros(T, ndims, nsmpls)) : zeros(T, ndims, nsmpls)
    logJac = device isa GPU ? gpu(zeros(T, ndims, nsmpls)) : zeros(T, ndims, nsmpls)

    kernel!(x, y, logJac, w, h, d, ndrange=size(x))

    logJac = sum(logJac, dims=1)

    return y, logJac
end

"""
    rqs_forward_pullback(x::AbstractArray{<:Real}, w::AbstractArray{<:Real}, h::AbstractArray{<:Real}, d::AbstractArray{<:Real}, w_logJac::AbstractArray{<:Real}, h_logJac::AbstractArray{<:Real}, d_logJac::AbstractArray{<:Real}, tangent_1::AbstractArray, tangent_2::AbstractArray)

Return the gradients of the spline functions characterized by `w`, `h`, and `d`, evaluated at the values in `x`.
The output will be on the same backend as `x` and `w`, `h`, and `d` (CPU or GPU).
"""
function rqs_forward_pullback(
        x::AbstractArray{M0},
        w::AbstractArray{M1},
        h::AbstractArray{M2},
        d::AbstractArray{M3},
        w_logJac::AbstractArray{M4},
        h_logJac::AbstractArray{M5},
        d_logJac::AbstractArray{M6},
        tangent_1::AbstractArray,
        tangent_2::AbstractArray;
    ) where {M0<:Real,M1<:Real, M2<:Real, M3<:Real, M4<:Real, M5<:Real, M6<:Real}

    T = promote_type(M0, M1, M2, M3, M4, M5, M6)

    ndims = size(x, 1)
    nsmpls = size(x, 2)
    nparams = size(w, 1)

    device = KernelAbstractions.get_backend(x)
    n = device isa GPU ? 256 : Threads.nthreads()

    y = device isa GPU ? gpu(zeros(T, ndims, nsmpls)) : zeros(T, ndims, nsmpls)
    logJac = device isa GPU ? gpu(zeros(T, ndims, nsmpls)) : zeros(T, ndims, nsmpls)

    ∂y∂w = device isa GPU ? gpu(zeros(T, nparams, ndims, nsmpls)) : zeros(T, nparams, ndims, nsmpls)
    ∂y∂h = device isa GPU ? gpu(zeros(T, nparams, ndims, nsmpls)) : zeros(T, nparams, ndims, nsmpls)
    ∂y∂d = device isa GPU ? gpu(zeros(T, nparams, ndims, nsmpls)) : zeros(T, nparams, ndims, nsmpls)

    ∂LogJac∂w = device isa GPU ? gpu(zeros(T, nparams, ndims, nsmpls)) : zeros(T, nparams, ndims, nsmpls)
    ∂LogJac∂h = device isa GPU ? gpu(zeros(T, nparams, ndims, nsmpls)) : zeros(T, nparams, ndims, nsmpls)
    ∂LogJac∂d = device isa GPU ? gpu(zeros(T, nparams, ndims, nsmpls)) : zeros(T, nparams, ndims, nsmpls)

    kernel! = rqs_forward_pullback_kernel!(device, n)

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

    return NoTangent(), @thunk(tangent_1 .* exp.(logJac)), ∂y∂w, ∂y∂h, ∂y∂d, ∂LogJac∂w, ∂LogJac∂h, ∂LogJac∂d
end

""" 
    rqs_forward_kernel!(x::AbstractArray, y::AbstractArray, logJac::AbstractArray, w::AbstractArray, h::AbstractArray, d::AbstractArray)

Apply the rational quadratic spline functions characterized by `w`, `h`, and `d` to `x`. 
The spline function characterized by the parameters in the `[:,i,j]` entries in `trafo` is applied to the `[i,j]`-th element of `x`.

The transformed values are stored in `y` the sums of the values of the logarithm of the absolute values of the determinant 
of the jacobians of the spline functions applied to a column of `x` are stored in the corresponding column of `logJac`.

To find the bin `k` in which the respective x value for a spline falls in, a the corresponding column of `w` is searched.j
"""
@kernel function rqs_forward_kernel!(
    x::AbstractArray,
    y::AbstractArray,
    logJac::AbstractArray,
    w::AbstractArray,
    h::AbstractArray,
    d::AbstractArray
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
Return the gradients of the rational quadratic spline functions characterized by `w`, `h`, and `d`, evaluated at the values in `x` and of `logJac`.
The output will be on the same backend as `x` and `w`, `h`, and `d` (CPU or GPU).
"""
@kernel function rqs_forward_pullback_kernel!(
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
    ∂LogJac∂d_tangent[k+1, i, j]  = tangent_2[1,j] * Base.ifelse(isinside, ∂LogJac∂d[2], zero(eltype(∂LogJac∂d))) # account for right pad in d?
end

function ChainRulesCore.rrule(
    ::typeof(rqs_forward),
    x::AbstractArray{M0},
    w::AbstractArray{M1},
    h::AbstractArray{M2},
    d::AbstractArray{M3},
    w_logJac::AbstractArray{M4},
    h_logJac::AbstractArray{M5},
    d_logJac::AbstractArray{M6};
) where {M0<:Real,M1<:Real, M2<:Real, M3<:Real, M4<:Real, M5<:Real, M6<:Real}

    y, logJac = rqs_forward(x, w, h, d, w_logJac, h_logJac, d_logJac)
    device = KernelAbstractions.get_backend(x)
    pullback(tangent) = device isa GPU ? rqs_forward_pullback(x, w, h, d, w_logJac, h_logJac, d_logJac, gpu(tangent[1]), gpu(tangent[2])) : rqs_forward_pullback(x, w, h, d, w_logJac, h_logJac, d_logJac, tangent[1], tangent[2])
    return (y, logJac), pullback
end

"""
    eval_forward_rqs_params(wₖ::Real, wₖ₊₁::Real, hₖ::Real, hₖ₊₁::Real, dₖ::Real, dₖ₊₁::Real, x::Real)

Apply a rational quadratic spline segment to `x`, and calculate the logarithm of the absolute value of this function's jacobian.
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
    eval_forward_rqs_params_with_grad(wₖ::Real, wₖ₊₁::Real, hₖ::Real, hₖ₊₁::Real, dₖ::Real, dₖ₊₁::Real, x::Real)

Apply a rational quadratic spline segment to `x`, and calculate the logarithm of the absolute value of this function's jacobian.
And calculate the gradient of that function depending on the spline parameters.
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
    rqs_backward(trafo::RQSplineInv, x::AbstractMatrix{<:Real})

Apply the inverse rational quadratic spline functions characterized by the parameters stored in `trafo` to the matrix `x`.
The rational quadratic spline function characterized by the parameters in the `[:,i,j]` entries in `trafo` is applied to the `[i,j]`-th element of `x`.
"""
function spline_backward(trafo::RQSplineInv, x::AbstractMatrix{<:Real})
    return rqs_backward(x, trafo.widths, trafo.heights, trafo.derivatives)

end

"""
    rqs_backward(x::AbstractArray{<:Real}, w::AbstractArray{<:Real}, h::AbstractArray{<:Real}, d::AbstractArray{<:Real}, w_logJac::AbstractArray{<:Real}, h_logJac::AbstractArray{<:Real}, d_logJac::AbstractArray{<:Real})

Apply the inverse rational quadratic spline functions characterized by `w`, `h`, and `d` to `x`. 
The spline function characterized by the parameters in the `[:,i,j]` entries in `trafo` is applied to the `[i,j]`-th element of `x`.

Return the transformed values in a matrix `y` of the same shape as `x`, and return a `1 x size(x,2)` -matrix holding the sums of the values 
of the logarithm of the absolute values of the determinant of the jacobians of the spline functions applied to a column of `x`.

The function executes in a kernel, on the same backend as `x` is stored (CPU or GPU), the output will also be returned on the same backend.
"""
function rqs_backward(
        x::AbstractArray{M0},
        w::AbstractArray{M1},
        h::AbstractArray{M2},
        d::AbstractArray{M3}
    ) where {M0<:Real,M1<:Real, M2<:Real, M3<:Real}
    
    T = promote_type(M0, M1, M2, M3)
    ndims, nsmpls = size(x)

    device = KernelAbstractions.get_backend(x)
    n = device isa GPU ? 256 : Threads.nthreads()
    kernel! = rqs_backward_kernel!(device, n)

    y = device isa GPU ? gpu(zeros(T, ndims, nsmpls)) : zeros(T, ndims, nsmpls)
    logJac = device isa GPU ? gpu(zeros(T, ndims, nsmpls)) : zeros(T, ndims, nsmpls)

    kernel!(x, y, logJac, w, h, d, ndrange=size(x))

    logJac = sum(logJac, dims=1)

    return y, logJac
end

@kernel function rqs_backward_kernel!(
        x::AbstractArray,
        y::AbstractArray,
        logJac::AbstractArray,
        w::AbstractArray,
        h::AbstractArray,
        d::AbstractArray
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
    logJac[i, j] += Base.ifelse(isinside, LogJacᵢⱼ, zero(typeof(LogJacᵢⱼ)))
end

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
