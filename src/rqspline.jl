# This file is a part of MonotonicSplines.jl, licensed under the MIT License (MIT).

# The spline implemented here is described in https://arxiv.org/abs/1906.04032 .

struct TrainableRQSpline <: Function
    widths::AbstractMatrix{<:Real}
    heights::AbstractMatrix{<:Real}
    derivatives::AbstractMatrix{<:Real}
end

export TrainableRQSpline
@functor TrainableRQSpline

struct RQSpline <: Function
    widths::AbstractMatrix{<:Real}
    heights::AbstractMatrix{<:Real}
    derivatives::AbstractMatrix{<:Real}
end

export RQSpline
@functor RQSpline

struct TrainableRQSplineInv <: Function
    widths::AbstractMatrix{<:Real}
    heights::AbstractMatrix{<:Real}
    derivatives::AbstractMatrix{<:Real}
end

@functor TrainableRQSplineInv
export TrainableRQSplineInv

struct RQSplineInv <: Function
    widths::AbstractMatrix{<:Real}
    heights::AbstractMatrix{<:Real}
    derivatives::AbstractMatrix{<:Real}
end

@functor RQSplineInv
export RQSplineInv


Base.:(==)(a::TrainableRQSpline, b::TrainableRQSpline) = a.widths == b.widths && a.heights == b.heights && a.derivatives == b.derivatives

Base.isequal(a::TrainableRQSpline, b::TrainableRQSpline) = isequal(a.widths, b.widths) && isequal(a.heights, b.heights) && isequal(a.derivatives, b.derivatives)

Base.hash(x::TrainableRQSpline, h::UInt) = hash(x.widths, hash(x.heights, hash(x.derivatives, hash(:TrainableRQSpline, hash(:EuclidianNormalizingFlows, h)))))

(f::TrainableRQSpline)(x::AbstractMatrix{<:Real}) = spline_forward(f, x)[1]

function ChangesOfVariables.with_logabsdet_jacobian(
    f::TrainableRQSpline,
    x::AbstractMatrix{<:Real}
)
    return spline_forward(f, x)
end

function InverseFunctions.inverse(f::TrainableRQSpline)
    return TrainableRQSplineInv(f.widths, f.heights, f.derivatives)
end

Base.:(==)(a::TrainableRQSplineInv, b::TrainableRQSplineInv) = a.widths == b.widths && a.heights == b.heights && a.derivatives == b.derivatives

Base.isequal(a::TrainableRQSplineInv, b::TrainableRQSplineInv) = isequal(a.widths, b.widths) && isequal(a.heights, b.heights) && isequal(a.derivatives, b.derivatives)

Base.hash(x::TrainableRQSplineInv, h::UInt) = hash(x.widths, hash(x.heights, hash(x.derivatives, hash(:TrainableRQSplineInv, hash(:EuclidianNormalizingFlows, h)))))

(f::TrainableRQSplineInv)(x::AbstractMatrix{<:Real}) = spline_backward(f, x)[1]

function ChangesOfVariables.with_logabsdet_jacobian(
    f::TrainableRQSplineInv,
    x::AbstractMatrix{<:Real}
)
    return spline_backward(f, x)
end

function InverseFunctions.inverse(f::TrainableRQSplineInv)
    return TrainableRQSpline(f.widths, f.heights, f.derivatives)
end

# Transformation forward: 

function spline_forward(trafo::TrainableRQSpline, x::AbstractMatrix{<:Real}; B=5.)

    @assert size(trafo.widths, 1) == size(trafo.heights, 1) == size(trafo.derivatives, 1) == size(x, 1) >= 1
    @assert size(trafo.widths, 2) == size(trafo.heights, 2) == (size(trafo.derivatives, 2) + 1) >= 2

    ndims = size(x, 1)

    w = _cumsum(_softmax(trafo.widths))
    h = _cumsum(_softmax(trafo.heights))
    d = _softplus(trafo.derivatives)

    w = hcat(repeat([-B,], ndims,1), w)
    h = hcat(repeat([-B,], ndims,1), h)
    d = hcat(repeat([1,], ndims,1), d)
    d = hcat(d, repeat([1,], ndims,1))

    return spline_forward(RQSpline(w,h,d), x)
end

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

    ndims = size(x, 1)
    nsmpls = size(x, 2)

    y = zeros(T, ndims, nsmpls)
    logJac = zeros(T, ndims, nsmpls)

    device = KernelAbstractions.get_device(x)
    n = device isa GPU ? 256 : 4
    kernel! = spline_forward_kernel!(device, n)

    ev = kernel!(x, y, logJac, w, h, d, ndrange=size(x))

    wait(ev)

    return y, sum(logJac, dims=1)
end


function spline_forward_pullback(
        x::AbstractArray{M0},
        w::AbstractArray{M1},
        h::AbstractArray{M2},
        d::AbstractArray{M3},
        w_logJac::AbstractArray{M4},
        h_logJac::AbstractArray{M5},
        d_logJac::AbstractArray{M6},
        tangent::ChainRulesCore.Tangent;
    ) where {M0<:Real,M1<:Real, M2<:Real, M3<:Real, M4<:Real, M5<:Real, M6<:Real}

    T = promote_type(M0, M1, M2, M3, M4, M5, M6)

    ndims = size(x, 1)
    nsmpls = size(x, 2)
    nparams = size(w, 2) 

    y = zeros(T, ndims, nsmpls)
    logJac = zeros(T, ndims, nsmpls)

    ∂y∂w = zeros(T, ndims, nparams)
    ∂y∂h = zeros(T, ndims, nparams)
    ∂y∂d = zeros(T, ndims, nparams+1)

    ∂LogJac∂w = zeros(T, ndims, nparams)
    ∂LogJac∂h = zeros(T, ndims, nparams)
    ∂LogJac∂d = zeros(T, ndims, nparams+1)

    device = KernelAbstractions.get_device(x)
    n = device isa GPU ? 256 : 4
    kernel! = spline_forward_pullback_kernel!(device, n)

    ev = kernel!(
        x, y, logJac, 
        w, h, d,
        ∂y∂w, ∂y∂h, ∂y∂d,
        ∂LogJac∂w, ∂LogJac∂h, ∂LogJac∂d, 
        tangent,
        ndrange=size(x)
        )

    wait(ev)
    logJac = sum(logJac, dims=1)

    return NoTangent(), @thunk(tangent[1] .* exp.(logJac)), ∂y∂w, ∂y∂h, ∂y∂d, ∂LogJac∂w, ∂LogJac∂h, ∂LogJac∂d
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

    K = size(w, 2)

    # Find the bin index
    k1 = searchsortedfirst_impl(w[i,:], x[i,j]) - 1
    k2 = one(typeof(k1))

    # Is inside of range
    isinside = (k1 < K) && (k1 > 0)
    k = Base.ifelse(isinside, k1, k2)

    x_tmp = Base.ifelse(isinside, x[i,j], w[i,k]) # Simplifies calculations
    (yᵢⱼ, LogJacᵢⱼ) = eval_forward_spline_params(w[i,k], w[i,k+1], h[i,k], h[i,k+1], d[i,k], d[i,k+1], x_tmp)

    y[i,j] = Base.ifelse(isinside, yᵢⱼ, x[i,j]) 
    logJac[i, j] += Base.ifelse(isinside, LogJacᵢⱼ, zero(typeof(LogJacᵢⱼ)))
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
        tangent::ChainRulesCore.Tangent
    )

    i, j = @index(Global, NTuple)

    K = size(w, 2)

    # Find the bin index
    k1 = searchsortedfirst_impl(w[i,:], x[i,j]) - 1
    k2 = one(typeof(k1))

    # Is inside of range
    isinside = (k1 < K) && (k1 > 0)
    k = Base.ifelse(isinside, k1, k2)

    x_tmp = Base.ifelse(isinside, x[i,j], w[i,k]) # Simplifies calculations
    (yᵢⱼ, LogJacᵢⱼ, ∂y∂wₖ, ∂y∂hₖ, ∂y∂dₖ, ∂LogJac∂wₖ, ∂LogJac∂hₖ, ∂LogJac∂dₖ) = eval_forward_spline_params_with_grad(w[i,k], w[i,k+1], h[i,k], h[i,k+1], d[i,k], d[i,k+1], x_tmp)

    y[i,j] = Base.ifelse(isinside, yᵢⱼ, x[i,j]) 
    logJac[i, j] += Base.ifelse(isinside, LogJacᵢⱼ, zero(typeof(LogJacᵢⱼ)))

    left_edge_istrue = (1 < k < K)
    left_edge_ind = Base.ifelse(left_edge_istrue, k-1, one(typeof(k)))

    @atomic ∂y∂w_tangent[i, left_edge_ind+1]      += tangent[1][i,j] * Base.ifelse(isinside * left_edge_istrue, ∂y∂wₖ[1], zero(eltype(∂y∂wₖ)))
    @atomic ∂y∂h_tangent[i, left_edge_ind+1]      += tangent[1][i,j] * Base.ifelse(isinside * left_edge_istrue, ∂y∂hₖ[1], zero(eltype(∂y∂hₖ)))
    @atomic ∂y∂d_tangent[i, left_edge_ind+1]      += tangent[1][i,j] * Base.ifelse(isinside * left_edge_istrue, ∂y∂dₖ[1], zero(eltype(∂y∂dₖ)))
    @atomic ∂LogJac∂w_tangent[i, left_edge_ind+1] += tangent[2][1,j] * Base.ifelse(isinside * left_edge_istrue, ∂LogJac∂wₖ[1], zero(eltype(∂LogJac∂wₖ)))
    @atomic ∂LogJac∂h_tangent[i, left_edge_ind+1] += tangent[2][1,j] * Base.ifelse(isinside * left_edge_istrue, ∂LogJac∂hₖ[1], zero(eltype(∂LogJac∂hₖ)))
    @atomic ∂LogJac∂d_tangent[i, left_edge_ind+1] += tangent[2][1,j] * Base.ifelse(isinside * left_edge_istrue, ∂LogJac∂dₖ[1], zero(eltype(∂LogJac∂dₖ)))
 
    right_edge_istrue = (k < K - 1)
    right_edge_ind = Base.ifelse(right_edge_istrue, k, one(typeof(k)))

    @atomic ∂y∂w_tangent[i, right_edge_ind+1]       += tangent[1][i,j] * Base.ifelse(isinside * right_edge_istrue, ∂y∂wₖ[2], zero(eltype(∂y∂wₖ)))
    @atomic ∂y∂h_tangent[i, right_edge_ind+1]       += tangent[1][i,j] * Base.ifelse(isinside * right_edge_istrue, ∂y∂hₖ[2], zero(eltype(∂y∂hₖ)))
    @atomic ∂y∂d_tangent[i, right_edge_ind+1]       += tangent[1][i,j] * Base.ifelse(isinside * right_edge_istrue, ∂y∂dₖ[2], zero(eltype(∂y∂dₖ)))
    @atomic ∂LogJac∂w_tangent[i, right_edge_ind+1]  += tangent[2][1,j] * Base.ifelse(isinside * right_edge_istrue, ∂LogJac∂wₖ[2], zero(eltype(∂LogJac∂wₖ)))
    @atomic ∂LogJac∂h_tangent[i, right_edge_ind+1]  += tangent[2][1,j] * Base.ifelse(isinside * right_edge_istrue, ∂LogJac∂hₖ[2], zero(eltype(∂LogJac∂hₖ)))
    @atomic ∂LogJac∂d_tangent[i, right_edge_ind+1]  += tangent[2][1,j] * Base.ifelse(isinside * right_edge_istrue, ∂LogJac∂dₖ[2], zero(eltype(∂LogJac∂dₖ)))

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

    # To do: Rewrite to avoid repeating calculation. 
    y, logJac = spline_forward(x, w, h, d, w_logJac, h_logJac, d_logJac)
    pullback(tangent) = spline_forward_pullback(x, w, h, d, w_logJac, h_logJac, d_logJac, tangent)
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

function spline_backward(trafo::TrainableRQSplineInv, x::AbstractMatrix{<:Real};   B = 5.)

    @assert size(trafo.widths, 1) == size(trafo.heights, 1) == size(trafo.derivatives, 1) == size(x, 1)  >= 1
    @assert size(trafo.widths, 2) == size(trafo.heights, 2) == (size(trafo.derivatives, 2) + 1)  >= 2

    ndims = size(x, 1)

    w = _cumsum(_softmax(trafo.widths))
    h = _cumsum(_softmax(trafo.heights))
    d = _softplus(trafo.derivatives)

    w = hcat(repeat([-B,], ndims,1), w)
    h = hcat(repeat([-B,], ndims,1), h)
    d = hcat(repeat([1,], ndims,1), d)
    d = hcat(d, repeat([1,], ndims,1))

    return spline_backward(RQSplineInv(w, h, d), x)
end

function spline_backward(trafo::RQSplineInv, x::AbstractMatrix{<:Real})
    return spline_backward(x, trafo.widths, trafo.heights, trafo.derivatives)
end


function spline_backward(
        x::AbstractArray{M0},
        w::AbstractArray{M1},
        h::AbstractArray{M2},
        d::AbstractArray{M3},
    ) where {M0<:Real,M1<:Real, M2<:Real, M3<:Real}

    T = promote_type(M0, M1, M2, M3)

    ndims = size(x, 1)
    nsmpls = size(x, 2)

    y = zeros(T, ndims, nsmpls)
    logJac = zeros(T, ndims, nsmpls)

    device = KernelAbstractions.get_device(x)
    n = device isa GPU ? 256 : 4
    kernel! = spline_backward_kernel!(device, n)

    ev = kernel!(x, y, logJac, w, h, d, ndrange=size(x))

    wait(ev)

    return y, sum(logJac, dims=1)
end

@kernel function spline_backward_kernel!(
        x::AbstractMatrix{M0},
        y::AbstractMatrix{M1},
        logJac::AbstractMatrix{M2},
        w::AbstractMatrix{M3},
        h::AbstractMatrix{M4},
        d::AbstractMatrix{M5}
    ) where {M0<:Real, M1<:Real, M2<:Real, M3<:Real, M4<:Real, M5<:Real,}

    i, j = @index(Global, NTuple)
    
    K = size(w, 2)

    # Find the bin index
    k1 = searchsortedfirst_impl(h[i,:], x[i,j]) - 1
    k2 = one(typeof(k1))

   # Is inside of range
   isinside = (k1 < K) && (k1 > 0)
   k = Base.ifelse(isinside, k1, k2)

    x_tmp = Base.ifelse(isinside, x[i,j], h[i,k]) # Simplifies unnecessary calculations
    (yᵢⱼ, LogJacᵢⱼ) = eval_backward_spline_params(w[i,k], w[i,k+1], h[i,k], h[i,k+1], d[i,k], d[i,k+1], x_tmp)

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

# Utils: 

function _softmax(x::AbstractVector)

    exp_x = exp.(x)
    sum_exp_x = sum(exp_x)

    return exp_x ./ sum_exp_x 
end

function _softmax(x::AbstractMatrix)

    val = cat([_softmax(i) for i in eachrow(x)]..., dims=2)'

    return val 
end

function _cumsum(x::AbstractVector; B = 5)
    return 2 .* B .* cumsum(x) .- B 
end

function _cumsum(x::AbstractMatrix)

    return cat([_cumsum(i) for i in eachrow(x)]..., dims=2)'
end

function _softplus(x::AbstractVector)

    return log.(exp.(x) .+ 1) 
end

function _softplus(x::AbstractMatrix)

    val = cat([_softplus(i) for i in eachrow(x)]..., dims=2)'

    return val
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
