# This file is a part of MonotonicSplines.jl, licensed under the MIT License (MIT).

module MonotonicSplinesMooncakeExt

import Mooncake
using MonotonicSplines: rqs_forward, rqs_inverse

Mooncake.@from_rrule(
    Mooncake.DefaultCtx,
    Tuple{typeof(rqs_forward), AbstractArray{<:Real,2}, AbstractArray{<:Real,3}, AbstractArray{<:Real,3}, AbstractArray{<:Real,3}},
)

Mooncake.@from_rrule(
    Mooncake.DefaultCtx,
    Tuple{typeof(rqs_inverse), AbstractArray{<:Real,2}, AbstractArray{<:Real,3}, AbstractArray{<:Real,3}, AbstractArray{<:Real,3}},
)

end # module MonotonicSplinesMooncakeExt
