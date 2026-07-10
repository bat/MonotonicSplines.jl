# This file is a part of MonotonicSplines.jl, licensed under the MIT License (MIT).

using MonotonicSplines
using Test

import Makie
using InverseFunctions: inverse

@testset "makie" begin
    @test Base.get_extension(MonotonicSplines, :MonotonicSplinesMakieExt) isa Module

    f = rand(RQSpline)

    fap = Makie.plot(f)
    @test fap isa Makie.FigureAxisPlot
    plt = fap.plot
    @test length(plt.plots) == 2
    @test plt.plots[1] isa Makie.Lines
    @test plt.plots[2] isa Makie.Scatter
    @test plt.plots[1].color[] == plt.plots[2].color[]

    knotsX, knotsY = f.pX, f.pY
    @test [Makie.Point2(x, y) for (x, y) in zip(knotsX, knotsY)] == plt.plots[2][1][]

    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1])
    p1 = Makie.plot!(ax, f)
    p2 = Makie.plot!(ax, inverse(f))
    @test p1.color[] != p2.color[]
end
