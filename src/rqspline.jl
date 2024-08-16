# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).

"""
    struct RQSpline{T<:Real,N,...} <: Function

Represents a rational quadratic spline function or a set of such functions for
multiple dimensions and samples.

Implements the splines originally proposed by Gregory and Delbourgo 
(https://doi.org/10.1093/imanum/2.2.123)

Constructors:

```julia
RQSpline(pX::AbstractVector{<:Real}, pY::AbstractVector{<:Real}, dYdX::AbstractVector{<:Real})
RQSpline(pX::AbstractArray{<:Real,3}, pY::AbstractArray{<:Real,3}, dYdX::AbstractArray{<:Real,3})
```

Fields:

- `pX`: An array that holds the x-position of the spline knots.

- `pY`: An array that holds the y-position of the spline knots.

- `dYdX`: An array that holds the derivative at the spline knots.

`pX`, `pY` and `dYdX` may be

- vectors of length `K+1`, representing a one-dimensional spline with `K` segments, or

- three-dimensional arrays of the shape `K+1 x n_dims x n_samples`, representing a set of splines for different
  dimensions and input samples.

The spline is continued with a linear function of slope one beyond the first
and last knot, so the first and last entry of dYdX should be one if the spline
is indended to be used at this extended range.

`RQSpline` supports the InverseFunctions, ChangesOfVariables, and Functors
APIs.

Example:

```
using MonotonicSplines

f = RQSpline(posX, posX, dY_dX)
Y = f(X)

using InverseFunctions: inverse
X ≈ inverse(f)(Y)

using ChangesOfVariables: with_logabsdet_jacobian
Y, LADJ = with_logabsdet_jacobian(f, X)
```

When instantiated as a set of multi-dimension/samples splines, `RQSpline` uses
the package KernelAbstractions for parallel CPU or GPU processing. Custom
`ChainRulesCore` rules are provided for effecient automatic differentation.

Random spline generation is supported and RQSpline comes with specialized
support for Plots:

```julia
using MonotonicSplines, Plots, InverseFunctions
f = rand(RQSpline)
plot(f); plot!(inverse(f))
plot(f, xlims = (-6, 6)); plot!(inverse(f), xlims = (-6, 6))
```
"""
struct RQSpline{
    T<:Real, N,
    TX<:AbstractArray{T,N}, TY<:AbstractArray{T,N}, TD<:AbstractArray{T,N}
} <: Function
    pX::TX
    pY::TY
    dYdX::TD
end

export RQSpline

(f::RQSpline{<:Any,1})(x::Real) = rqs_forward(x, f.pX, f.pY, f.dYdX)[1]
(f::RQSpline{<:Any,3})(x::AbstractMatrix{<:Real}) = rqs_forward(x, f.pX, f.pY, f.dYdX)[1]

function ChangesOfVariables.with_logabsdet_jacobian(f::RQSpline{<:Any,1}, x::Real)
    return rqs_forward(x, f.pX, f.pY, f.dYdX)
end

function ChangesOfVariables.with_logabsdet_jacobian(f::RQSpline{<:Any,3}, x::AbstractMatrix{<:Real} )
    return rqs_forward(x, f.pX, f.pY, f.dYdX)
end


function rand_rqspline(rng::AbstractRNG, ::Type{T}) where T
    n = 7
    range = 4
    pX = pushfirst!(cumsum(rand(rng, T, n-1)), 0)
    pX .= pX ./ last(pX) .* 2*range .- range
    pY = pushfirst!(cumsum(rand(rng, T, n-1)), 0)
    pY .= pY ./ last(pY) .* 2*range .- range
    est_dYdX = vcat(1, estimate_dYdX(pX, pY)[begin+1:end-1], 1)
    mod_dYdX = vcat(1, exp.(randn(rng, T, n-2) .* 1//5), 1)
    dYdX = est_dYdX .* mod_dYdX
    return RQSpline(pX, pY, dYdX)
end

function Random.rand(r::AbstractRNG, ::Random.SamplerType{RQSpline{T}}) where T
    return rand_rqspline(r, T)
end

function Random.rand(r::AbstractRNG, ::Random.SamplerType{RQSpline})
    return rand_rqspline(r, Float64)
end



"""
  struct InvRQSpline{T<:Real,N,...} <: Function

Represents the inverse of [`RQSpline`](@ref).

`InvRQSpline` holds the same spline knot parameters as the corresponding
`RQSpline`.

Users should not instantiate `InvRQSpline` directly, use
`InverseFunctions.inverse(RQSpline(...))` instead.
"""
struct InvRQSpline{
    T<:Real, N,
    TX<:AbstractArray{T,N}, TY<:AbstractArray{T,N}, TD<:AbstractArray{T,N}
} <: Function
    pX::TX
    pY::TY
    dYdX::TD
end

export InvRQSpline

(f::InvRQSpline{<:Any,1})(x::Real) = rqs_inverse(x, f.pX, f.pY, f.dYdX)[1]
(f::InvRQSpline{<:Any,3})(x::AbstractMatrix{<:Real}) = rqs_inverse(x, f.pX, f.pY, f.dYdX)[1]

function ChangesOfVariables.with_logabsdet_jacobian(f::InvRQSpline{<:Any,1}, x::Real)
    return rqs_inverse(x, f.pX, f.pY, f.dYdX)
end

function ChangesOfVariables.with_logabsdet_jacobian(f::InvRQSpline{<:Any,3}, x::AbstractMatrix{<:Real} )
    return rqs_inverse(x, f.pX, f.pY, f.dYdX)
end



"""
    MonotonicSplines.rqs_forward(x::Real, pX::AbstractVector{<:Real}, pY::AbstractVector{<:Real}, dYdX::AbstractVector{<:Real})
    MonotonicSplines.rqs_forward(X::AbstractArray{<:Real,2}, pX::AbstractArray{<:Real,3}, pY::AbstractArray{<:Real,3}, dYdX::AbstractArray{<:Real,3})

Apply the rational quadratic spline function(s) defined by the parameters `pX`
(pX), `pY` (pY), and `dYdX` (dYdX), to the input(s) `X`.

See [`RQSpline`](@ref) for more details.
"""
function rqs_forward end


function rqs_forward(
    x::Real,
    pX::AbstractVector{<:Real},
    pY::AbstractVector{<:Real},
    dYdX::AbstractVector{<:Real}
)
    K = size(pX, 1) - 1

    # Find the bin index
    k1 = searchsortedfirst_impl(view(pX, :), x) - 1
    k2 = one(typeof(k1))

    # Is inside of range
    isinside = (1 <= k1 <= K)
    k = Base.ifelse(isinside, k1, k2)

    x_tmp = Base.ifelse(isinside, x, pX[k]) # Simplifies calculations
    (y_tmp, logJac_tmp) = eval_forward_rqs_params(pX[k], pX[k+1], pY[k], pY[k+1], dYdX[k], dYdX[k+1], x_tmp)

    y = Base.ifelse(isinside, y_tmp, x) 
    logJac = Base.ifelse(isinside, logJac_tmp, zero(typeof(logJac_tmp)))

    return y, logJac
end


function rqs_forward(
    x::AbstractArray{<:Real,2},
    pX::AbstractArray{<:Real,3},
    pY::AbstractArray{<:Real,3},
    dYdX::AbstractArray{<:Real,3}
)
    compute_unit = get_compute_unit(x)
    backend = ka_backend(compute_unit)
    kernel! = rqs_forward_kernel!(backend, _ka_threads(backend)...)

    y = similar(x)
    logJac = similar(x)

    kernel!(x, y, logJac, pX, pY, dYdX, ndrange=size(x))

    logJac = sum(logJac, dims=1)

    return y, logJac
end


@kernel function rqs_forward_kernel!(
    x::AbstractArray{<:Real},
    y::AbstractArray{<:Real},
    logJac::AbstractArray{<:Real},
    pX::AbstractArray{<:Real},
    pY::AbstractArray{<:Real},
    dYdX::AbstractArray{<:Real}
)

    i, j = @index(Global, NTuple)

    K = size(pX, 1) - 1

    # Find the bin index
    k1 = searchsortedfirst_impl(view(pX, :, i, j), x[i,j]) - 1
    k2 = one(typeof(k1))

    # Is inside of range
    isinside = (1 <= k1 <= K)
    k = Base.ifelse(isinside, k1, k2)

    x_tmp = Base.ifelse(isinside, x[i,j], pX[k,i,j]) # Simplifies calculations
    (yᵢⱼ, LogJacᵢⱼ) = eval_forward_rqs_params(pX[k,i,j], pX[k+1,i,j], pY[k,i,j], pY[k+1,i,j], dYdX[k,i,j], dYdX[k+1,i,j], x_tmp)

    y[i,j] = Base.ifelse(isinside, yᵢⱼ, x[i,j]) 
    logJac[i, j] = Base.ifelse(isinside, LogJacᵢⱼ, zero(typeof(LogJacᵢⱼ)))
end


# Non-public:
#=
    MonotonicSplines.eval_forward_rqs_params(pXₖ::Real, pXₖ₊₁::Real, pYₖ::Real, pYₖ₊₁::Real, dYdXₖ::Real, dYdXₖ₊₁::Real, x::Real)

Apply a rational quadratic spline segment to a number `x`, and calculate the logarithm of the absolute value of this function's Jacobian.

# Arguments
- `pXₖ`, `pXₖ₊₁`: The width parameters of the spline segment at the edges of the `k`-th interval.
- `pYₖ`, `pYₖ₊₁`: The height parameters of the spline segment.
- `dYdXₖ`, `dYdXₖ₊₁`: The derivative parameters of the spline segment.
- `x`: The value at which the spline function is evaluated.

# Returns
- `y`: The transformed value after applying the rational quadratic spline segment to `x`.
- `logJac`: The logarithm of the absolute value of the derivative of the segment at `x`.
=#
function eval_forward_rqs_params(
    pXₖ::Real, pXₖ₊₁::Real, 
    pYₖ::Real, pYₖ₊₁::Real, 
    dYdXₖ::Real, dYdXₖ₊₁::Real, 
    x::Real) 
      
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

    return y, logJac
end


"""
    MonotonicSplines.rqs_inverse(x::Real, pX::AbstractVector{<:Real}, pY::AbstractVector{<:Real}, dYdX::AbstractVector{<:Real})
    MonotonicSplines.rqs_inverse(X::AbstractArray{<:Real,2}, pX::AbstractArray{<:Real,3}, pY::AbstractArray{<:Real,3}, dYdX::AbstractArray{<:Real,3})

Apply the inverse of the rational quadratic spline function(s) defined by the
parameters `pX` (pX), `pY` (pY), and `dYdX` (dYdX), to the input(s)
`X`.

See [`InvRQSpline`](@ref) for more details.
"""
function rqs_inverse end

function rqs_inverse(
    x::Real,
    pX::AbstractVector{<:Real},
    pY::AbstractVector{<:Real},
    dYdX::AbstractVector{<:Real}
)
    K = size(pX, 1) - 1

    # Find the bin index
    k1 = searchsortedfirst_impl(view(pY, :), x) - 1
    k2 = one(typeof(k1))

    # Is inside of range
    isinside = (k1 < K) && (k1 > 0)
    k = Base.ifelse(isinside, k1, k2)

    x_tmp = Base.ifelse(isinside, x, pX[k])  # Simplifies unnecessary calculations
    (y_tmp, logJac_tmp) = eval_inverse_rqs_params(pX[k], pX[k+1], pY[k], pY[k+1], dYdX[k], dYdX[k+1], x_tmp)

    y = Base.ifelse(isinside, y_tmp, x) 
    logJac = Base.ifelse(isinside, logJac_tmp, zero(typeof(logJac_tmp)))

    return y, logJac
end

function rqs_inverse(
        x::AbstractArray{<:Real,2},
        pX::AbstractArray{<:Real,3},
        pY::AbstractArray{<:Real,3},
        dYdX::AbstractArray{<:Real,3}
    )

    compute_unit = get_compute_unit(x)
    backend = ka_backend(compute_unit)
    kernel! = rqs_inverse_kernel!(backend, _ka_threads(backend)...)

    y = similar(x)
    logJac = similar(x) 
    kernel!(x, y, logJac, pX, pY, dYdX, ndrange=size(x))
    logJac = sum(logJac, dims=1)

    return y, logJac
end


@kernel function rqs_inverse_kernel!(
        x::AbstractArray{<:Real},
        y::AbstractArray{<:Real},
        logJac::AbstractArray{<:Real},
        pX::AbstractArray{<:Real},
        pY::AbstractArray{<:Real},
        dYdX::AbstractArray{<:Real}
    )
    i, j = @index(Global, NTuple)
    
    K = size(pX, 1) - 1

    # Find the bin index
    k1 = searchsortedfirst_impl(view(pY, :, i, j), x[i,j]) - 1
    k2 = one(typeof(k1))

    # Is inside of range
    isinside = (k1 < K) && (k1 > 0)
    k = Base.ifelse(isinside, k1, k2)

    x_tmp = Base.ifelse(isinside, x[i,j], pX[k,i,j])  # Simplifies unnecessary calculations
    (yᵢⱼ, LogJacᵢⱼ) = eval_inverse_rqs_params(pX[k,i,j], pX[k+1,i,j], pY[k,i,j], pY[k+1,i,j], dYdX[k,i,j], dYdX[k+1,i,j], x_tmp)

    y[i,j] = Base.ifelse(isinside, yᵢⱼ, x[i,j]) 
    logJac[i, j] = Base.ifelse(isinside, LogJacᵢⱼ, zero(typeof(LogJacᵢⱼ)))
end


#=
    MonotonicSplines.eval_inverse_rqs_params(pXₖ::Real, pXₖ₊₁::Real, pYₖ::Real, pYₖ₊₁::Real, dYdXₖ::Real, dYdXₖ₊₁::Real, x::Real)

Apply a rational quadratic spline segment to a number `x`, and calculate the logarithm of the absolute value of this function's derivative.

# Arguments
- `pXₖ`, `pXₖ₊₁`: The width parameters of the spline segment at the edges of the `k`-th interval.
- `pYₖ`, `pYₖ₊₁`: The height parameters of the spline segment.
- `dYdXₖ`, `dYdXₖ₊₁`: The derivative parameters of the spline segment.
- `x`: The value at which the spline function is evaluated.

# Returns
- `y`: The transformed value after applying the rational quadratic spline segment to `x`.
- `logJac`: The logarithm of the absolute value of the derivative of the segment at `x`.
=#
function eval_inverse_rqs_params(
    pXₖ::M0, pXₖ₊₁::M0, 
    pYₖ::M1, pYₖ₊₁::M1, 
    dYdXₖ::M2, dYdXₖ₊₁::M2, 
    x::M3) where {M0<:Real,M1<:Real, M2<:Real, M3<:Real}

    Δy = pYₖ₊₁ - pYₖ
    Δy2 = x - pYₖ # use y instead of X, because of inverse
    Δx = pXₖ₊₁ - pXₖ
    sk = Δy / Δx

    a = Δy * (sk - dYdXₖ) + Δy2 * (dYdXₖ₊₁ + dYdXₖ - 2*sk)
    b = Δy * dYdXₖ - Δy2 * (dYdXₖ₊₁ + dYdXₖ - 2*sk)
    c = - sk * Δy2

    denom = -b - sqrt(b*b - 4*a*c)

    y = (2 * c / denom) * Δx + pXₖ

    # Gradient computation:
    da =  (dYdXₖ₊₁ + dYdXₖ - 2*sk)
    db = -(dYdXₖ₊₁ + dYdXₖ - 2*sk)
    dc = - sk

    temp2 = 1 / (2*sqrt(b*b - 4*a*c))

    grad = 2 * dc * denom - 2 * c * (-db - temp2 * (2 * b * db - 4 * a * dc - 4 * c * da))
    LogJac = log(abs(Δx * grad)) - 2*log(abs(denom))

    return y, LogJac
end
