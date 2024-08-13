# This file is a part of MonotonicSplines.jl, licensed under the MIT License (MIT).

module MonotonicSplinesChainRulesCoreExt

using MonotonicSplines
using MonotonicSplines: rqs_forward, rqs_inverse, rqs_pullback
using MonotonicSplines: eval_forward_rqs_params_with_grad, eval_inverse_rqs_params_with_grad

import ChainRulesCore
using ChainRulesCore: @thunk, AbstractZero, NoTangent, unthunk

using Adapt: adapt
using HeterogeneousComputing: get_compute_unit


function ChainRulesCore.rrule(
    ::typeof(rqs_forward),
    x::AbstractArray{<:Real,2},
    pX::AbstractArray{<:Real,3},
    pY::AbstractArray{<:Real,3},
    dYdX::AbstractArray{<:Real,3}
)
    y, logJac = rqs_forward(x, pX, pY, dYdX)
    compute_unit = get_compute_unit(x)

    function rqs_forward_pullback(tangent)
        maybe_δY, maybe_δlogJac = map(unthunk, tangent)
        δY = adapt(compute_unit, maybe_δY isa AbstractZero ? zero(y) : maybe_δY)
        δlogJac = adapt(compute_unit, maybe_δlogJac isa AbstractZero ? zero(logJac) : maybe_δlogJac)

        (
            NoTangent(),
            @thunk(tangent[1] .* exp.(logJac)),
            rqs_pullback(eval_forward_rqs_params_with_grad, x, pX, pY, dYdX, δY, δlogJac)...
        )
    end

    return (y, logJac), rqs_forward_pullback
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

    function rqs_inverse_pullback(tangent)
        maybe_δY, maybe_δlogJac = map(unthunk, tangent)
        δY = adapt(compute_unit, maybe_δY isa AbstractZero ? zero(y) : maybe_δY)
        δlogJac = adapt(compute_unit, maybe_δlogJac isa AbstractZero ? zero(logJac) : maybe_δlogJac)

        (
            NoTangent(),
            @thunk(tangent[1] .* exp.(logJac)),
            rqs_pullback(eval_inverse_rqs_params_with_grad, x, pX, pY, dYdX, δY, δlogJac)...
        )
    end

    return (y, logJac), rqs_inverse_pullback
end


end # module MonotonicSplinesChainRulesCoreExt
