# This file is a part of MonotonicSplines.jl, licensed under the MIT License (MIT).

module MonotonicSplinesMakieExt

import Makie
import MonotonicSplines: RQSpline, InvRQSpline

const _SingleRQSpline = Union{RQSpline{<:Any,1}, InvRQSpline{<:Any,1}}

Makie.@recipe RQSplinePlot (spline,) begin
    color = @inherit linecolor
    npoints = 200
    cycle = [:color]
end

Makie.plottype(::_SingleRQSpline) = RQSplinePlot

_knots(f::RQSpline{<:Any,1}) = (f.pX, f.pY)
_knots(f::InvRQSpline{<:Any,1}) = (f.pY, f.pX)

function Makie.plot!(p::RQSplinePlot{<:Tuple{_SingleRQSpline}})
    curve = Makie.lift(p.spline, p.npoints) do f, n
        knotsX, _ = _knots(f)
        from, until = float.(extrema(knotsX))
        xs = range(from + eps(from), until - eps(until), length = n)
        Makie.Point2.(xs, map(f, xs))
    end
    knots = Makie.lift(p.spline) do f
        knotsX, knotsY = _knots(f)
        Makie.Point2.(knotsX, knotsY)
    end
    Makie.lines!(p, curve, color = p.color)
    Makie.scatter!(p, knots, color = p.color)
    p
end

end # module MonotonicSplinesMakieExt
