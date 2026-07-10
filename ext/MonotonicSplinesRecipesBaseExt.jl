# This file is a part of MonotonicSplines.jl, licensed under the MIT License (MIT).

module MonotonicSplinesRecipesBaseExt

import MonotonicSplines: RQSpline, InvRQSpline
import RecipesBase: @recipe, @series

@recipe function f(@nospecialize(f::Union{RQSpline{<:Any,1}, InvRQSpline{<:Any,1}}))
    isinverse = f isa InvRQSpline
    label --> (!isinverse ? "RQSpline" : "inverse RQSpline")

    knotsX, knotsY = !isinverse ? (f.pX, f.pY) : (f.pY, f.pX)
    from, until = map(float, get(plotattributes, :xlims, (minimum(knotsX), maximum(knotsX))))

    @series begin
        f, from + eps(from), until - eps(until)
    end

    @series begin
        primary := false
        seriestype := :scatter
        knotsX, knotsY
    end

    nothing
end

end # module MonotonicSplinesRecipesBaseExt
