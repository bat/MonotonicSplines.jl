# MonotonicSplines.jl

[![Documentation for stable version](https://img.shields.io/badge/docs-stable-blue.svg)](https://bat.github.io/MonotonicSplines.jl/stable)
[![Documentation for development version](https://img.shields.io/badge/docs-dev-blue.svg)](https://bat.github.io/MonotonicSplines.jl/dev)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
[![Build Status](https://github.com/bat/MonotonicSplines.jl/workflows/CI/badge.svg?branch=main)](https://github.com/bat/MonotonicSplines.jl/actions?query=workflow%3ACI)
[![Codecov](https://codecov.io/gh/bat/MonotonicSplines.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/bat/MonotonicSplines.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)


This package provides high-performance, GPU- and
[AD](https://en.wikipedia.org/wiki/Automatic_differentiation)-friendly
monotonic spline functions in Julia for use in
[Normalizing Flows](https://en.wikipedia.org/wiki/Flow-based_generative_model),
resp. parameter transformations in general.

This package currently includes the *monotonic rational quadratic splines* defined in ["Neural Spline Flows, Durkan et al. 2019"](https://arxiv.org/abs/1906.04032).

Please see the Documentation linked below for details.

## Documentation

* [Documentation for stable version](https://bat.github.io/MonotonicSplines.jl/stable)
* [Documentation for development version](https://bat.github.io/MonotonicSplines.jl/dev)

## Installation

To install `MonotonicSplines.jl`, start Julia and run 

```Julia
julia> using Pkg
julia> pkg.add("MonotonicSplines.jl")
```
and 
```Julia
julia> using MonotonicSplines
```
to use the functions the package provides.

## Example usage of rational quadratic spline functions for use in Normalizing Flows

Given a set $`\{ \mathbf{x}_i\}`$ ($`i = 1,..., N_{\text{samples}}`$) of $`D`$ dimensional samples, a (partial) Normalizing Flow $`\mathbf{f}`$ using Splines transforms a number of $`D-d~`$  ($`1 \leq d \leq D`$) components of each sample $`\mathbf{x}_i`$ to obtain a set of partially transformed samples $`\{ \mathbf{y}_i\}`$:

```math
\begin{align*}
\mathbf{f} : \mathbb{R}^D \rightarrow \mathbb{R}^D, ~~ \mathbf{x} = 
\begin{pmatrix}
x_{1}                   \\
x_{2}                   \\
\vdots                  \\
x_{d-1}                 \\
x_{d}                   \\
\vdots                  \\
x_{D}                   \\
\end{pmatrix} 
\mapsto
\begin{pmatrix}
x_{1}                   \\
x_{2}                   \\
\vdots                  \\
x_{d-1}                 \\
f_{\theta_d}(x_d)       \\
\vdots                  \\
f_{\theta_{D}}(x_{D})   \\
\end{pmatrix} 
= \mathbf{y}
\end{align*}
```
Here $`f_{\theta_j} : \mathbb{R} \rightarrow \mathbb{R} ~~ (j = d,...,D)~`$  denotes a single spline function, characterized by the parameters $`~\theta_{j} = (\text{pX}_j~, ~\text{heigths}_j~,~\text{dYdX}_j)`$ in the case of the rational quadratic spline functions defined in ["Neural Spline Flows, Durkan et al. 2019"](https://arxiv.org/abs/1906.04032). 

Consider a single sample $`\mathbf{x}_i \in \mathbb{R}^D`$ from our sample set. 

In the context of Normalizing Flows, the set of parameters $`\{\theta_j\}`$ that characterize the spline functions $`\{f_{\theta_j}\}`$ to transform the $`d`$ -th to $`D`$ -th components of $`\mathbf{x}_i`$ are obtained by processing the output of a neural net $`NN`$. 

This neural net takes the first $`d`$ components $`\{ x_{i,1},..., x_{i,d}\}`$ of $`\mathbf{x}_i`$ that are *not* transformed by the Flow $`\mathbf{f}`$ as input. The output then is a vector of $`(D-d) \cdot (3K-1)`$ components, where $`K`$ is the number of segments in a spline function. 

These "raw" spline parameters are then processed as described in ["Neural Spline Flows, Durkan et al. 2019"](https://arxiv.org/abs/1906.04032) to obtain $`\{ \theta_j \}`$. 

`MonotonicSplines.jl` is designed with parallelism in mind, and this implementation allows for the simultaneous transformation of batches of samples using spline functions.

To this end, the parameters for characterizing sets of several spline functions are stored in the same struct.

Now consider the task of transforming the entire set of samples $`\{\mathbf{x}_i\}`$ via the Normalizing Flow $`f`$. 

To achieve this, each of the $`(D-d) \cdot N_{\text{samples}}`$ components to be transformed obtains an individual spline function $`f_{\theta_j}^{(i)}`$.

So the spline function $`f_{\theta_j}^{(i)}`$ is applied to the $`j`$ -th component of the $`i`$ -th sample.

Given the output `params_raw` of the neural net $`NN`$, we can obtain the `pX`, `pY`, and `dYdX` to characterize the desired spline function as follows:
```Julia
julia> pX, pY, dYdX = rqs_params_from_nn(params_raw, n_dims_to_transform)
```
Here, `params_raw` is a `3(K-1) * n_dims_to_transform x n_samples` -matrix. `K` again is the number of spline segments and `n_dims_to_transform` $`~= D-d~`$ is the number of components to transform per sample. 

The `i` -th column of this matrix `params_raw` is the output of the neural net $`NN`$ with the first $`d`$ components $`\{ x_{i,1},..., x_{i,d}\}`$ of the `i` -th sample $`\mathbf{x}_i`$ as the input.

`pX`, `pY`, and `dYdX` each are `K x n_dims_to_transform x n_samples` -arrays. The `[:,j,i]` entries hold the parameters to characterize $`f_{\theta_j}^{(i)}`$, the spline function to transform the `j` -th component of the `i` -th sample from the sample set.

We then define the set of spline functions by:

```Julia
julia> rqs_splines = RQSpline(pX, pY, dYdX)
```
An object holding the parameters to characterize `n_dims_to_transform x n_samples` spline functions.

To apply the spline functions characterized by the parameters stored in `rqs_splines`, we first isolate the components of the sample set that are supposed to be transformed and then do: 

```Julia
julia> Y_partial = rqs_splines(X_partial)
```
`X_partial` is a `n_dims_to_transform x n_samples` -matrix, holding the components of the sample set $`\{\mathbf{x}_i\}`$ that are to be transformed. So the `i` -th column in `X_partial` holds the `d` -th to `D` -th elements of the `i` -th sample in $`\{\mathbf{x}_i\}`$. 

`Y_partial` is a `n_dims_to_transform x n_samples` matrix, where the `i,j` -th component is the transformed value of the `i,j` -th entry in `X_partial`. 

For further details on the implementation, see the [Documentation for stable version](https://bat.github.io/MonotonicSplines.jl/stable).
