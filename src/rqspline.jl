# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).
# The algorithm implemented here is described in https://arxiv.org/abs/1906.04032 

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

function spline_forward(trafo::RQSpline, x::AbstractMatrix{<:Real})
    return spline_forward(x, trafo.widths, trafo.heights, trafo.derivatives, trafo.widths, trafo.heights, trafo.derivatives)
end

function spline_forward(
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

    device = KernelAbstractions.get_device(x)
    n = device isa GPU ? 256 : Threads.nthreads()
    kernel! = spline_forward_kernel!(device, n)

    y = device isa GPU ? gpu(zeros(T, ndims, nsmpls)) : zeros(T, ndims, nsmpls)
    logJac = device isa GPU ? gpu(zeros(T, ndims, nsmpls)) : zeros(T, ndims, nsmpls)

    ev = kernel!(x, y, logJac, w, h, d, ndrange=size(x))

    wait(ev)
    logJac = sum(logJac, dims=1)

    return y, logJac
end


function spline_forward_pullback(
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

    device = KernelAbstractions.get_device(x)
    n = device isa GPU ? 256 : Threads.nthreads()

    y = device isa GPU ? gpu(zeros(T, ndims, nsmpls)) : zeros(T, ndims, nsmpls)
    logJac = device isa GPU ? gpu(zeros(T, ndims, nsmpls)) : zeros(T, ndims, nsmpls)

    ∂y∂w = device isa GPU ? gpu(zeros(T, nparams, ndims, nsmpls)) : zeros(T, nparams, ndims, nsmpls)
    ∂y∂h = device isa GPU ? gpu(zeros(T, nparams, ndims, nsmpls)) : zeros(T, nparams, ndims, nsmpls)
    ∂y∂d = device isa GPU ? gpu(zeros(T, nparams, ndims, nsmpls)) : zeros(T, nparams, ndims, nsmpls)

    ∂LogJac∂w = device isa GPU ? gpu(zeros(T, nparams, ndims, nsmpls)) : zeros(T, nparams, ndims, nsmpls)
    ∂LogJac∂h = device isa GPU ? gpu(zeros(T, nparams, ndims, nsmpls)) : zeros(T, nparams, ndims, nsmpls)
    ∂LogJac∂d = device isa GPU ? gpu(zeros(T, nparams, ndims, nsmpls)) : zeros(T, nparams, ndims, nsmpls)

    kernel! = spline_forward_pullback_kernel!(device, n)

    ev = kernel!(
        x, y, logJac, 
        w, h, d,
        ∂y∂w, ∂y∂h, ∂y∂d,
        ∂LogJac∂w, ∂LogJac∂h, ∂LogJac∂d, 
        tangent_1,
        tangent_2,
        ndrange=size(x)
        )

    wait(ev)

    logJac = sum(logJac, dims=1)

    return NoTangent(), @thunk(tangent_1 .* exp.(logJac)), ∂y∂w, ∂y∂h, ∂y∂d, ∂LogJac∂w, ∂LogJac∂h, ∂LogJac∂d
end

@kernel function spline_forward_kernel!(
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
    (yᵢⱼ, LogJacᵢⱼ) = eval_forward_spline_params(w[k,i,j], w[k+1,i,j], h[k,i,j], h[k+1,i,j], d[k,i,j], d[k+1,i,j], x_tmp)

    y[i,j] = Base.ifelse(isinside, yᵢⱼ, x[i,j]) 
    logJac[i, j] = Base.ifelse(isinside, LogJacᵢⱼ, zero(typeof(LogJacᵢⱼ)))
end


@kernel function spline_forward_pullback_kernel!(
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
    (yᵢⱼ, LogJacᵢⱼ, ∂y∂w, ∂y∂h, ∂y∂d, ∂LogJac∂w, ∂LogJac∂h, ∂LogJac∂d) = eval_forward_spline_params_with_grad(w[k,i,j], w[k+1,i,j], h[k,i,j], h[k+1,i,j], d[k,i,j], d[k+1,i,j], x_tmp)

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
    ::typeof(spline_forward),
    x::AbstractArray{M0},
    w::AbstractArray{M1},
    h::AbstractArray{M2},
    d::AbstractArray{M3},
    w_logJac::AbstractArray{M4},
    h_logJac::AbstractArray{M5},
    d_logJac::AbstractArray{M6};
) where {M0<:Real,M1<:Real, M2<:Real, M3<:Real, M4<:Real, M5<:Real, M6<:Real}

    y, logJac = spline_forward(x, w, h, d, w_logJac, h_logJac, d_logJac)
    device = KernelAbstractions.get_device(x)
    pullback(tangent) = device isa GPU ? spline_forward_pullback(x, w, h, d, w_logJac, h_logJac, d_logJac, gpu(tangent[1]), gpu(tangent[2])) : spline_forward_pullback(x, w, h, d, w_logJac, h_logJac, d_logJac, tangent[1], tangent[2])
    return (y, logJac), pullback
end

function eval_forward_spline_params(
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

function eval_forward_spline_params_with_grad(
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


function spline_backward(trafo::RQSplineInv, x::AbstractMatrix{<:Real})
    return spline_backward(x, trafo.widths, trafo.heights, trafo.derivatives)

end


function spline_backward(
        x::AbstractArray{M0},
        w::AbstractArray{M1},
        h::AbstractArray{M2},
        d::AbstractArray{M3}
    ) where {M0<:Real,M1<:Real, M2<:Real, M3<:Real}
    
    T = promote_type(M0, M1, M2, M3)
    ndims, nsmpls = size(x)

    device = KernelAbstractions.get_device(x)
    n = device isa GPU ? 256 : Threads.nthreads()
    kernel! = spline_backward_kernel!(device, n)

    y = device isa GPU ? gpu(zeros(T, ndims, nsmpls)) : zeros(T, ndims, nsmpls)
    logJac = device isa GPU ? gpu(zeros(T, ndims, nsmpls)) : zeros(T, ndims, nsmpls)

    ev = kernel!(x, y, logJac, w, h, d, ndrange=size(x))

    wait(ev)
    logJac = sum(logJac, dims=1)

    return y, logJac
end

@kernel function spline_backward_kernel!(
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
    (yᵢⱼ, LogJacᵢⱼ) = eval_backward_spline_params(w[k,i,j], w[k+1,i,j], h[k,i,j], h[k+1,i,j], d[k,i,j], d[k+1,i,j], x_tmp)

    y[i,j] = Base.ifelse(isinside, yᵢⱼ, x[i,j]) 
    logJac[i, j] += Base.ifelse(isinside, LogJacᵢⱼ, zero(typeof(LogJacᵢⱼ)))
end

function eval_backward_spline_params(
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
