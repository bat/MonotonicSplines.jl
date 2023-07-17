# This file is a part of MonotonicSplines.jl, licensed under the MIT License (MIT).

"""
    MonotonicSplines

High-performance monotonic splines in Julia.
"""
module MonotonicSplines

import Adapt
using Adapt: adapt

using LinearAlgebra

import ChainRulesCore
using ChainRulesCore: rrule, @thunk, NoTangent

import ChangesOfVariables

using HeterogeneousComputing

import InverseFunctions

import Functors
using Functors: @functor

using KernelAbstractions

include("rqspline.jl")
include("utils.jl")

end # module
