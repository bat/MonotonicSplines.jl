# MonotonicSplines.jl

This package provides a high-performance, GPU- and
[AD](https://en.wikipedia.org/wiki/Automatic_differentiation)-friendly implementation of monotonic spline functions in Julia.

The intended use cases is as a building block in [Normalizing FLows](https://en.wikipedia.org/wiki/Flow-based_generative_model), resp. parameter transformations in general.

MonotonicSplines currently implements rational quadratic splines as decribed in ["Neural Spline Flows, Durkan et al. 2019"](https://arxiv.org/abs/1906.04032).

The splines defined here support the [ChangesOfVariables](https://github.com/JuliaMath/ChangesOfVariables.jl), [InverseFunctions](https://github.com/JuliaMath/InverseFunctions.jl) and [Functors](https://github.com/FluxML/Functors.jl) APIs. The splines also come with some custom [ChainRulesCore](https://github.com/JuliaDiff/ChainRulesCore.jl) `rrule` methods to speed up automatic differentiation.

The package uses [KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl) to provide both GPU and multi-vendor GPU support.


## Manual Outline
---

```@contents
Pages = ["introduction.md","guide.md"]
Depth = 3
```
