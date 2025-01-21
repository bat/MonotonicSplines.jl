# This file is a part of MonotonicSplines.jl, licensed under the MIT License (MIT).

module MonotonicSplinesFunctorsExt

using MonotonicSplines
import Functors
using Functors: @functor

@static if !isdefined(Base, :pkgversion) || pkgversion(Functors) < v"0.5"

@functor RQSpline
@functor InvRQSpline

end # Functors < v"0.5"

end # module MonotonicSplinesFunctorsExt
