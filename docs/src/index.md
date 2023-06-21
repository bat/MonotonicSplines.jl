# MonotonicSplines.jl

A package implementing monotonoic spline functions for applications in spline-based approaches to Normalizing FLows [(Wikipedia)](https://en.wikipedia.org/wiki/Flow-based_generative_model).

This package currently includes the *monotonous rational quadratic splines* as defined in ["Neural Spline Flows, Durkan et al. 2019"](https://arxiv.org/abs/1906.04032).

## Package Features
---

This package was created with high performance as the overall goal and features automatic code differentiation, powered by [Zygote.jl](https://fluxml.ai/Zygote.jl/latest/), and GPU compatibility, powered by [KernelAbstractions.jl](https://juliagpu.github.io/KernelAbstractions.jl/stable/).

## Manual Outline
---

```@contents
Pages = ["introduction.md","implementation.md"]
Depth = 3
```
