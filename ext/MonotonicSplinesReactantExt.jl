# This file is a part of MonotonicSplines.jl, licensed under the MIT License (MIT).

module MonotonicSplinesReactantExt

using Reactant: AnyTracedRArray
using MonotonicSplines: RQSForward, RQSInverse, rqs_apply_purearray
import MonotonicSplines

MonotonicSplines.rqs_apply(
    trafo::Union{RQSForward,RQSInverse},
    x::AnyTracedRArray{<:Any,2},
    pX::AnyTracedRArray{<:Any,3},
    pY::AnyTracedRArray{<:Any,3},
    dYdX::AnyTracedRArray{<:Any,3}
) = rqs_apply_purearray(trafo, x, pX, pY, dYdX)

end # module MonotonicSplinesReactantExt
