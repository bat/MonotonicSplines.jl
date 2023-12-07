# Implementation

This package is designed to transform batches of samples of a $D$-dimensional target distribution via spline functions. The samples are stored in matrices, where each column corresponds to a sample with multiple components.

In the case of rational quadratic spline functions, this package uses the structs `RQSpline` and `InvRQSpline` to hold the parameters (widths, heights, and derivatives) that characterize the spline functions used for transformation. These parameters are stored in 3D arrays, allowing the package to handle multiple spline functions for multiple samples simultaneously.

The package leverages the power of parallel computing through the use of [`KernelAbstractions.jl`](https://juliagpu.github.io/KernelAbstractions.jl/stable/). This allows the execution of the code in paralell on either a CPU or a GPU, significantly speeding up the transformation process. Thus multiple entries in a target matrix can be transformed in parallel, utilizing multithreading.

Furthermore, this package uses automatic code differentiation (AD), enabling the calculation of exact gradients. The use may specify a desired AD backend via the  [`HeterogeneousComputing.jl`](https://github.com/oschulz/HeterogeneousComputing.jl) API.
This feature enhances the performance of the package by providing precise and efficient gradient computations, which are crucial for optimization tasks in machine learning.

---
In this package, batches of samples with multiple components may be transformed via spline functions. 
The samples are stored in `n_dims` ``\times`` `n_samples` matrices. So a column corresponds to a sample with `n_dims` components, and the matrix holds `n_samples` samples. 

The `RQSpline` and `RQSplineInv` structs hold the `widths`, `heights`, and `derivatives` parameters that are used to characterize the `n_dims` ``\times`` `n_samples` spline functions used to transform the target input. 

`widths`, `heights`, and `derivatives` are `K+1` ``\times`` `n_dims` ``\times`` `n_samples` arrays, with the parameters to characterize a single spline function with `K` segments in the first dimension. 
Along the second dimension, the parameters for `n_dims` spline functions for a single sample are stored, and along the third dimension the sets splines for differen samples.

The spline function characterized by the parameters in the `[:,i,j]` entries in the parameter arrays is applied to the `[i,j]`-th element of a target input matrix `x`.