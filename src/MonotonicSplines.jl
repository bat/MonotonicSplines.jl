# This file is a part of MonotonicSplines.jl, licensed under the MIT License (MIT).

"""
    MonotonicSplines

High-performance monotonic splines in Julia.
"""
module MonotonicSplines

import Adapt
using Adapt: adapt

using LinearAlgebra
import Random
using Random: AbstractRNG

import ChangesOfVariables

using HeterogeneousComputing
using HeterogeneousComputing: ka_backend

using KernelAbstractions

include("rqspline.jl")
include("rqspline_pullbacks.jl")
include("utils.jl")

end # module
