# This file is a part of MonotonicSplines.jl, licensed under the MIT License (MIT).

module MonotonicSplinesChainRulesCoreExt

using MonotonicSplines
using MonotonicSplines: rqs_forward, rqs_inverse

import ChainRulesCore
using ChainRulesCore: @thunk, NoTangent


function ChainRulesCore.rrule(
    ::typeof(rqs_forward),
    x::AbstractArray{<:Real,2},
    pX::AbstractArray{<:Real,3},
    pY::AbstractArray{<:Real,3},
    dYdX::AbstractArray{<:Real,3}
)

    y, logJac = rqs_forward(x, pX, pY, dYdX)
    compute_unit = get_compute_unit(x)

    pullback(tangent) = (
        NoTangent(),
        @thunk(tangent[1] .* exp.(logJac)),
        rqs_pullback(eval_forward_rqs_params_with_grad, x, pX, pY, dYdX, adapt(compute_unit, tangent[1]), adapt(compute_unit, tangent[2]))...
    )

    return (y, logJac), pullback
end


function ChainRulesCore.rrule(
    ::typeof(rqs_inverse),
    x::AbstractArray{<:Real,2},
    pX::AbstractArray{<:Real,3},
    pY::AbstractArray{<:Real,3},
    dYdX::AbstractArray{<:Real,3}
)

    y, logJac = rqs_inverse(x, pX, pY, dYdX)
    compute_unit = get_compute_unit(x)

    pullback(tangent) = (
        NoTangent(),
        @thunk(tangent[1] .* exp.(logJac)),
        rqs_pullback(eval_inverse_rqs_params_with_grad, x, pX, pY, dYdX, adapt(compute_unit, tangent[1]), adapt(compute_unit, tangent[2]))...
    )

    return (y, logJac), pullback
end


end # module MonotonicSplinesChainRulesCoreExt
