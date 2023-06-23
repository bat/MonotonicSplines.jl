# This file is a part of MonotonicSplines.jl, licensed under the MIT License (MIT).

"""
    MonotonicSplines

High-performance monotonic splines in Julia.
"""
module MonotonicSplines

using LinearAlgebra

import ChainRulesCore
using ChainRulesCore

import ChangesOfVariables
import InverseFunctions

import Functors
using Functors: @functor

using KernelAbstractions

using DelimitedFiles

include("rqspline.jl")
include("utils.jl")

end # module
