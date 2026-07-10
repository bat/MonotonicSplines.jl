# MonotonicSplines.jl

This package provides a high-performance, GPU- and
[AD](https://en.wikipedia.org/wiki/Automatic_differentiation)-friendly implementation of monotonic spline functions in Julia.

The intended use case is as a building block in [Normalizing Flows](https://en.wikipedia.org/wiki/Flow-based_generative_model), resp. parameter transformations in general.

This package currently includes the *monotonic rational quadratic splines* as defined in ["Neural Spline Flows, Durkan et al. 2019"](https://arxiv.org/abs/1906.04032).

The splines defined here support the [ChangesOfVariables](https://github.com/JuliaMath/ChangesOfVariables.jl), [InverseFunctions](https://github.com/JuliaMath/InverseFunctions.jl) and [Functors](https://github.com/FluxML/Functors.jl) APIs.

Automatic differentiation is supported via custom [ChainRulesCore](https://github.com/JuliaDiff/ChainRulesCore.jl) `rrule` methods (used by e.g. [Zygote](https://github.com/FluxML/Zygote.jl)) and via [Mooncake](https://github.com/chalk-lab/Mooncake.jl). [Enzyme](https://github.com/EnzymeAD/Enzyme.jl) can differentiate the spline functions directly, and splines can be traced, compiled and differentiated with [Reactant](https://github.com/EnzymeAD/Reactant.jl).

Plotting single splines is supported via both [RecipesBase](https://github.com/JuliaPlots/RecipesBase.jl) (so `plot(spline)` works with [Plots](https://github.com/JuliaPlots/Plots.jl)) and [Makie](https://github.com/MakieOrg/Makie.jl).

The package uses [KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl) to provide multi-threaded CPU and multi-vendor GPU support.


## Quickstart

```julia
using MonotonicSplines, Plots, InverseFunctions, ChangesOfVariables

f = rand(RQSpline)
f.pX, f.pY, f.dYdX

plot(f, xlims = (-6, 6)); plot!(inverse(f), xlims = (-6, 6))

x = 1.2
y = f(x)
with_logabsdet_jacobian(f, x)
inverse(f)(y)
with_logabsdet_jacobian(inverse(f), y)
```

## Table of contents
---

```@contents
Pages = ["introduction.md", "api.md"]
Depth = 3
```
