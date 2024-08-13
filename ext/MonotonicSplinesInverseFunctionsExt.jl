# This file is a part of MonotonicSplines.jl, licensed under the MIT License (MIT).

module MonotonicSplinesInverseFunctionsExt

using MonotonicSplines
import InverseFunctions

InverseFunctions.inverse(f::RQSpline) = InvRQSpline(f.widths, f.heights, f.derivatives)
InverseFunctions.inverse(f::InvRQSpline) = RQSpline(f.widths, f.heights, f.derivatives)

end # module MonotonicSplinesInverseFunctionsExt
