# This file is a part of MonotonicSplines.jl, licensed under the MIT License (MIT).

module MonotonicSplinesFunctorsExt

using MonotonicSplines
using Functors: @functor

@functor RQSpline
@functor InvRQSpline

end # module MonotonicSplinesFunctorsExt
