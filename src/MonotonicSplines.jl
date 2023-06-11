# This file is a part of MonotonicSplines.jl, licensed under the MIT License (MIT).

"""
    MonotonicSplines

High-performance monotonic splines in Julia.
"""
module MonotonicSplines

using LinearAlgebra

import ChainRulesCore
using ChainRulesCore: @thunk

import ChangesOfVariables
import InverseFunctions

import Functors
using Functors: @functor

import KernelAbstractions
using KernelAbstractions: @kernel, @index, @atomic

include("rqspline.jl")

end # module
