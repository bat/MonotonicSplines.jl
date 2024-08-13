# This file is a part of MonotonicSplines.jl, licensed under the MIT License (MIT).

module MonotonicSplinesInverseFunctionsExt

using MonotonicSplines
import InverseFunctions

InverseFunctions.inverse(f::RQSpline) = InvRQSpline(f.pX, f.pY, f.dYdX)
InverseFunctions.inverse(f::InvRQSpline) = RQSpline(f.pX, f.pY, f.dYdX)

end # module MonotonicSplinesInverseFunctionsExt
