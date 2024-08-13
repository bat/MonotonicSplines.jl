# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).

"""
    struct RQSpline{T<:Real,N,...} <: Function

Represents a rational quadratic spline function or a set of such functions for
multiple dimensions and samples.

Implements the splines originally proposed by Gregory and Delbourgo 
(https://doi.org/10.1093/imanum/2.2.123)

Constructors:

```julia
RQSpline(widths::AbstractVector{<:Real}, heights::AbstractVector{<:Real}, derivatives::AbstractVector{<:Real})
RQSpline(widths::AbstractArray{<:Real,3}, heights::AbstractArray{<:Real,3}, derivatives::AbstractArray{<:Real,3})
```

Fields:

- `widths`: An array that holds the x-position of the spline knots.

- `heights`: An array that holds the y-position of the spline knots.

- `derivatives`: An array that holds the derivative at the spline knots.

`widths`, `heights` and `derivatives` may be

- vectors of length `K+1`, representing a one-dimensional spline with `K` segments, or

- three-dimensional arrays of the shape `K+1 x n_dims x n_samples`, representing a set of splines for different
  dimensions and input samples.

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

(f::RQSpline{<:Any,1})(x::Real) = rqs_forward(x, f.widths, f.heights, f.derivatives)[1]
(f::RQSpline{<:Any,3})(x::AbstractMatrix{<:Real}) = rqs_forward(x, f.widths, f.heights, f.derivatives)[1]

function ChangesOfVariables.with_logabsdet_jacobian(f::RQSpline{<:Any,1}, x::Real)
    return rqs_forward(x, f.widths, f.heights, f.derivatives)
end

function ChangesOfVariables.with_logabsdet_jacobian(f::RQSpline{<:Any,3}, x::AbstractMatrix{<:Real} )
    return rqs_forward(x, f.widths, f.heights, f.derivatives)
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
    widths::TX
    heights::TY
    derivatives::TD
end

export InvRQSpline

(f::InvRQSpline{<:Any,1})(x::Real) = rqs_backward(x, f.widths, f.heights, f.derivatives)[1]
(f::InvRQSpline{<:Any,3})(x::AbstractMatrix{<:Real}) = rqs_backward(x, f.widths, f.heights, f.derivatives)[1]

function ChangesOfVariables.with_logabsdet_jacobian(f::InvRQSpline{<:Any,1}, x::Real)
    return rqs_backward(x, f.widths, f.heights, f.derivatives)
end

function ChangesOfVariables.with_logabsdet_jacobian(f::InvRQSpline{<:Any,3}, x::AbstractMatrix{<:Real} )
    return rqs_backward(x, f.widths, f.heights, f.derivatives)
end


Base.@deprecate spline_forward(f::RQSpline, x::AbstractMatrix{<:Real}) rqs_forward(x, f.widths, f.heights, f.derivatives)



"""
    MonotonicSplines.rqs_forward(x::Real, w::AbstractVector{<:Real}, h::AbstractVector{<:Real}, d::AbstractVector{<:Real})
    MonotonicSplines.rqs_forward(X::AbstractArray{<:Real,2}, w::AbstractArray{<:Real,3}, h::AbstractArray{<:Real,3}, d::AbstractArray{<:Real,3})

Apply the rational quadratic spline function(s) defined by the parameters `w`
(widths), `h` (heights), and `d` (derivatives), to the input(s) `X`.

See [`RQSpline`](@ref) for more details.
"""
function rqs_forward end


function rqs_forward(
    x::Real,
    w::AbstractVector{<:Real},
    h::AbstractVector{<:Real},
    d::AbstractVector{<:Real}
)
    K = size(w, 1) - 1

    # Find the bin index
    k1 = searchsortedfirst_impl(view(w, :), x) - 1
    k2 = one(typeof(k1))

    # Is inside of range
    isinside = (1 <= k1 <= K)
    k = Base.ifelse(isinside, k1, k2)

    x_tmp = Base.ifelse(isinside, x, w[k]) # Simplifies calculations
    (y_tmp, logJac_tmp) = eval_forward_rqs_params(w[k], w[k+1], h[k], h[k+1], d[k], d[k+1], x_tmp)

    y = Base.ifelse(isinside, y_tmp, x) 
    logJac = Base.ifelse(isinside, logJac_tmp, zero(typeof(logJac_tmp)))

    return y, logJac
end


function rqs_forward(
    x::AbstractArray{<:Real,2},
    w::AbstractArray{<:Real,3},
    h::AbstractArray{<:Real,3},
    d::AbstractArray{<:Real,3}
)
    compute_unit = get_compute_unit(x)
    backend = ka_backend(compute_unit)
    kernel! = rqs_forward_kernel!(backend, _ka_threads(backend)...)

    y = similar(x)
    logJac = similar(x)

    kernel!(x, y, logJac, w, h, d, ndrange=size(x))

    logJac = sum(logJac, dims=1)

    return y, logJac
end


"""
    KernelAbstractions.@kernel rqs_forward_kernel!(x::AbstractArray, y::AbstractArray, logJac::AbstractArray, w::AbstractArray, h::AbstractArray, d::AbstractArray)

KernelAbstractions kernel that implements [`rqs_forward`](@ref).
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
    MonotonicSplines.eval_forward_rqs_params(wₖ::Real, wₖ₊₁::Real, hₖ::Real, hₖ₊₁::Real, dₖ::Real, dₖ₊₁::Real, x::Real)

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


Base.@deprecate spline_backward(f::InvRQSpline, x::AbstractMatrix{<:Real}) rqs_backward(x, f.widths, f.heights, f.derivatives)


"""
    MonotonicSplines.rqs_backward(x::Real, w::AbstractVector{<:Real}, h::AbstractVector{<:Real}, d::AbstractVector{<:Real})
    MonotonicSplines.rqs_backward(X::AbstractArray{<:Real,2}, w::AbstractArray{<:Real,3}, h::AbstractArray{<:Real,3}, d::AbstractArray{<:Real,3})

Apply the inverse of the rational quadratic spline function(s) defined by the
parameters `w` (widths), `h` (heights), and `d` (derivatives), to the input(s)
`X`.

See [`InvRQSpline`](@ref) for more details.
"""
function rqs_backward end

function rqs_backward(
    x::Real,
    w::AbstractVector{<:Real},
    h::AbstractVector{<:Real},
    d::AbstractVector{<:Real}
)
    K = size(w, 1) - 1

    # Find the bin index
    k1 = searchsortedfirst_impl(view(h, :), x) - 1
    k2 = one(typeof(k1))

    # Is inside of range
    isinside = (k1 < K) && (k1 > 0)
    k = Base.ifelse(isinside, k1, k2)

    x_tmp = Base.ifelse(isinside, x, w[k])  # Simplifies unnecessary calculations
    (y_tmp, logJac_tmp) = eval_backward_rqs_params(w[k], w[k+1], h[k], h[k+1], d[k], d[k+1], x_tmp)

    y = Base.ifelse(isinside, y_tmp, x) 
    logJac = Base.ifelse(isinside, logJac_tmp, zero(typeof(logJac_tmp)))

    return y, logJac
end

function rqs_backward(
        x::AbstractArray{<:Real,2},
        w::AbstractArray{<:Real,3},
        h::AbstractArray{<:Real,3},
        d::AbstractArray{<:Real,3}
    )

    compute_unit = get_compute_unit(x)
    backend = ka_backend(compute_unit)
    kernel! = rqs_inverse_kernel!(backend, _ka_threads(backend)...)

    y = similar(x)
    logJac = similar(x) 
    kernel!(x, y, logJac, w, h, d, ndrange=size(x))
    logJac = sum(logJac, dims=1)

    return y, logJac
end


"""
    KernelAbstractions.@kernel rqs_backward_kernel!(x::AbstractArray, y::AbstractArray, logJac::AbstractArray, w::AbstractArray, h::AbstractArray, d::AbstractArray)

KernelAbstractions kernel that implements [`rqs_backward`](@ref).
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
    MonotonicSplines.eval_backward_rqs_params(wₖ::Real, wₖ₊₁::Real, hₖ::Real, hₖ₊₁::Real, dₖ::Real, dₖ₊₁::Real, x::Real)

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
