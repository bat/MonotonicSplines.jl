# Implementation
---
In this package, batches of samples with multiple components may be transformed via spline functions. 
The samples are stored in `n_dims` ``\times`` `n_samples` matrices. So a column corresponds to a sample with `n_dims` components, and the matrix holds `n_samples` samples. 

The `RQSpline` and `RQSplineInv` structs hold the `widths`, `heights`, and `derivatives` parameters that are used to characterize the `n_dims` ``\times`` `n_samples` spline functions used to transform the target input. 

`widths`, `heights`, and `derivatives` are `K+1` ``\times`` `n_dims` ``\times`` `n_samples` arrays, with the parameters to characterize a single spline function with `K` segments in the first dimension. 
Along the second dimension, the parameters for `n_dims` spline functions for a single sample are stored, and along the third dimension the sets splines for differen samples.

The spline function characterized by the parameters in the `[:,i,j]` entries in the parameter arrays is applied to the `[i,j]`-th element of a target input matrix `x`.


The application of the spline functions to a target matrix is performed via kernels defined with [`KernelAbstractions.jl`](https://juliagpu.github.io/KernelAbstractions.jl/stable/). This allows for the execution of this code on either a CPU or a GPU, speeding up the transformation process.

Thus, several entries in a target matrix may be transformed in paralell, using multithreading. 

The code is also automatically differentiable (via [Zygote.jl](https://fluxml.ai/Zygote.jl/latest/)), allowing for the calculation of exact gradients. 