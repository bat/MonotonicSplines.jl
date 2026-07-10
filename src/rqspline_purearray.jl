# This file is a part of MonotonicSplines.jl, licensed under the MIT License (MIT).

"""
    MonotonicSplines.rqs_apply_purearray(trafo::Union{RQSForward,RQSInverse}, x::AbstractMatrix, pX::AbstractArray{<:Any,3}, pY::AbstractArray{<:Any,3}, dYdX::AbstractArray{<:Any,3})

Apply rational quadratic spline functions using only broadcasts and
reductions, without scalar indexing.

Equivalent to the kernel-based `rqs_apply`, but compatible with
tracing-based frameworks like Reactant. The spline segment parameters are
selected via one-hot contractions instead of a bin search.
"""
function rqs_apply_purearray(
    trafo::Union{RQSForward,RQSInverse},
    x::AbstractMatrix,
    pX::AbstractArray{<:Any,3},
    pY::AbstractArray{<:Any,3},
    dYdX::AbstractArray{<:Any,3}
)
    K = size(pX, 1) - 1

    knots = _rqs_search_knots(trafo, pX, pY)
    k1 = dropdims(sum(knots .< reshape(x, 1, size(x)...); dims = 1); dims = 1)
    isinside = (k1 .>= 1) .& (k1 .<= K)
    k = clamp.(k1, 1, K)

    ks = cumsum(fill!(similar(x, Int, K + 1, 1, 1), 1); dims = 1)
    onehot_k  = ks .== reshape(k, 1, size(k)...)
    onehot_k1 = ks .== reshape(k .+ 1, 1, size(k)...)
    _gather(P, onehot) = dropdims(sum(P .* onehot; dims = 1); dims = 1)

    pXₖ, pXₖ₊₁ = _gather(pX, onehot_k), _gather(pX, onehot_k1)
    pYₖ, pYₖ₊₁ = _gather(pY, onehot_k), _gather(pY, onehot_k1)
    dYdXₖ, dYdXₖ₊₁ = _gather(dYdX, onehot_k), _gather(dYdX, onehot_k1)

    x_tmp = ifelse.(isinside, x, _rqs_search_knots(trafo, pXₖ, pYₖ))
    y_ins, logJac_ins = _rqs_segment_purearray(trafo, pXₖ, pXₖ₊₁, pYₖ, pYₖ₊₁, dYdXₖ, dYdXₖ₊₁, x_tmp)

    y = ifelse.(isinside, y_ins, x)
    logJac = sum(ifelse.(isinside, logJac_ins, zero(eltype(logJac_ins))); dims = 1)

    return y, logJac
end


# Broadcasted counterpart of eval_forward_rqs_params:
function _rqs_segment_purearray(
    ::RQSForward,
    pXₖ, pXₖ₊₁, pYₖ, pYₖ₊₁, dYdXₖ, dYdXₖ₊₁, x
)
    Δy = pYₖ₊₁ .- pYₖ
    Δx = pXₖ₊₁ .- pXₖ
    sk = Δy ./ Δx
    ξ = (x .- pXₖ) ./ Δx

    denom = sk .+ (dYdXₖ₊₁ .+ dYdXₖ .- 2 .* sk) .* ξ .* (1 .- ξ)
    nom_1 = sk .* ξ .* ξ .+ dYdXₖ .* ξ .* (1 .- ξ)
    nom_3 = dYdXₖ₊₁ .* ξ .* ξ .+ 2 .* sk .* ξ .* (1 .- ξ) .+ dYdXₖ .* (1 .- ξ) .^ 2

    y = pYₖ .+ Δy .* nom_1 ./ denom
    logJac = log.(abs.(sk .* sk .* nom_3)) .- 2 .* log.(abs.(denom))

    return y, logJac
end


# Broadcasted counterpart of eval_inverse_rqs_params:
function _rqs_segment_purearray(
    ::RQSInverse,
    pXₖ, pXₖ₊₁, pYₖ, pYₖ₊₁, dYdXₖ, dYdXₖ₊₁, x
)
    Δy = pYₖ₊₁ .- pYₖ
    Δy2 = x .- pYₖ
    Δx = pXₖ₊₁ .- pXₖ
    sk = Δy ./ Δx
    dsum = dYdXₖ₊₁ .+ dYdXₖ .- 2 .* sk

    a = Δy .* (sk .- dYdXₖ) .+ Δy2 .* dsum
    b = Δy .* dYdXₖ .- Δy2 .* dsum
    c = .- sk .* Δy2

    θ = sqrt.(b .* b .- 4 .* a .* c)
    denom = .- b .- θ

    y = (2 .* c ./ denom) .* Δx .+ pXₖ

    da = dsum
    db = .- dsum
    dc = .- sk
    temp2 = 1 ./ (2 .* θ)
    grad = 2 .* dc .* denom .- 2 .* c .* (.- db .- temp2 .* (2 .* b .* db .- 4 .* a .* dc .- 4 .* c .* da))
    logJac = log.(abs.(Δx .* grad)) .- 2 .* log.(abs.(denom))

    return y, logJac
end
