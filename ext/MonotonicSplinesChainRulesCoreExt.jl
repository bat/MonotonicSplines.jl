# This file is a part of MonotonicSplines.jl, licensed under the MIT License (MIT).

module MonotonicSplinesChainRulesCoreExt

using MonotonicSplines
using MonotonicSplines: rqs_forward, rqs_backward

import ChainRulesCore
using ChainRulesCore: @thunk, NoTangent


function ChainRulesCore.rrule(
    ::typeof(rqs_forward),
    x::AbstractArray{<:Real},
    w::AbstractArray{<:Real},
    h::AbstractArray{<:Real},
    d::AbstractArray{<:Real}
)

    y, logJac = rqs_forward(x, w, h, d)
    compute_unit = get_compute_unit(x)

    pullback(tangent) = (
        NoTangent(),
        @thunk(tangent[1] .* exp.(logJac)),
        rqs_pullback(eval_forward_rqs_params_with_grad, x, w, h, d, adapt(compute_unit, tangent[1]), adapt(compute_unit, tangent[2]))...
    )

    return (y, logJac), pullback
end


function ChainRulesCore.rrule(
    ::typeof(rqs_backward),
    x::AbstractArray{<:Real},
    w::AbstractArray{<:Real},
    h::AbstractArray{<:Real},
    d::AbstractArray{<:Real}
)

    y, logJac = rqs_backward(x, w, h, d)
    compute_unit = get_compute_unit(x)

    pullback(tangent) = (
        NoTangent(),
        @thunk(tangent[1] .* exp.(logJac)),
        rqs_pullback(eval_backward_rqs_params_with_grad, x, w, h, d, adapt(compute_unit, tangent[1]), adapt(compute_unit, tangent[2]))...
    )

    return (y, logJac), pullback
end


end # module MonotonicSplinesChainRulesCoreExt
