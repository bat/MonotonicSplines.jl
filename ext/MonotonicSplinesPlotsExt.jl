# This file is a part of MonotonicSplines.jl, licensed under the MIT License (MIT).

module MonotonicSplinesPlotsExt

import MonotonicSplines: RQSpline, InvRQSpline
import RecipesBase: @recipe, @series
using Plots: palette
import MonotonicSplines

function _get_next_default_plotcolor(plotattributes::Dict{Symbol}, color_attr::Symbol = :seriescolor)
    if haskey(plotattributes, color_attr)
        return plotattributes[color_attr]
    elseif haskey(plotattributes, :seriescolor)
        return plotattributes[:seriescolor]
    else
        serlist = plotattributes[:plot_object].series_list
        pl = collect(palette(:auto))
        if !isempty(serlist)
            last_color = last(serlist)[color_attr]
            found_color = false
            selected_color = first(pl)
            for c in pl
                if found_color
                    # choose next color
                    selected_color = c
                    break
                end
                if oftype(c, last_color) ≈ c
                    found_color = true
                end
            end
            return selected_color
        else
            return first(pl)
        end
    end
end

@recipe function f(@nospecialize(f::Union{RQSpline{<:Any,1}, InvRQSpline{<:Any,1}}))
    isinverse = f isa InvRQSpline
    label = !isinverse ? "RQSpline" : "inverse RQSpline"
    pX, pY = f.pX, f.pY

    color = _get_next_default_plotcolor(plotattributes, :linecolor)
    plotattributes[:seriescolor] = color

    knotsX, knotsY = !isinverse ? (pX, pY) : (pY, pX)
    from, until = map(float, get(plotattributes, :xlims, (minimum(knotsX), maximum(knotsX))))

    @series begin
        label := nothing
        seriestype := :scatter
        knotsX, knotsY
    end

    @series begin
        label --> label
        f, from + eps(from), until - eps(until)
    end

    nothing
end

end # module MonotonicSplinesPlotsExt
