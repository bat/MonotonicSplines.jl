# Introduction
---

This page provides a summarized explanation of the implementation of the rational quadratic spline functions as defined in [^1]. For a comprehensive derivation and explanation of the spline function's usage, see [^1] and [^2].

Rational quadratic functions are defined as the quotient of two quadratic polynomials and offer superior flexibility to other function families [^1]. They are easily differentiable, and since they are constructed to be monotonous, they also are analytically invertible. 

Durkan et al. construct their rational quadratic splines $f$ from $K$ different rational quadratic functions. The parameterization of the splines follows the method of Gregory and Delbourgo [^3]. They characterize a spline by the coordinates $\{(x_k,y_k)\},  k = 1, ..., K+1$, of the $K+1$ points called \textit{knots} in which two adjacent spline segments are joined together. The parameterization also incorporates the $K-1$ derivatives of the spline at the interior knots. 

limit the number of parameters needed to characterize a spline, they are defined on a finite interval
$$
\begin{align*}
    f : [-B, B] \rightarrow [-B, B] , ~ \text{with}~ B \in \mathbb{R}.
\end{align*}
$$
Here we chose $B = 5.0$ by default, but this can be changed arbitrarily. 

Gregory and Delbourgo split this $[-B, B]$ input domain into $K$ bins and define each spline segment on one bin. The knots lie on the bin edges on the $x$-axis and at the desired $y$ value through which the spline is supposed to pass. 

In the $k$-th bin ($1\leq k \leq K$), the respective spline segment $f_k$ is defined as an expression of the coordinates $\{(x_k,y_k)\}$ and $\{(x_{k+1},y_{k+1})\}$ of the enclosing $k$ -th and $k+1$ -st knots as well as the spline's derivatives $\delta_{k}$ and $\delta_{k+1}$ at the knots:
$$
\begin{align}
    \label{eq:rqs}
    f_k : [x_k, x_{k+1}) \rightarrow [-B,B], ~x \mapsto y_k + \frac{(y_{k+1}-y_{k})[s_k\xi^2+\delta_k\xi(1-\xi)]}{s_k+[\delta_{k+1}+\delta_k-2s_k]\xi(1-\xi)},
\end{align}
$$
$$
\begin{align*}
    \text{where} \quad s_k = s_k(x)= \frac{y_{k+1} - y_k}{x_{k+1}-x} \quad \text{and} \quad \xi = \xi(x) = \frac{x - x_k}{x_{k+1}-x_k}
\end{align*}
$$
The spline function $f$ is then defined piece-wise from the segment functions within each bin. The segments of the inverse spline function $f^{-1}$ are given by:
$$
\begin{flalign*}
    \label{eq:rqs_inv}
    & f_k^{-1} : [y_k, y_{k+1}) \rightarrow [-B,B], ~y \mapsto x_k + \frac{(x_{k+1}-x_{k})[s_k'(\xi')^2+\delta_k\xi'(1-\xi')]}{s_k'+[\delta_{k+1}+\delta_k-2s_k']\xi'(1-\xi')}~,\quad 
\end{flalign*}
$$
$$
\begin{flalign*}
    & \text{where}\\
    & s_k' = s_k'(y)= \frac{x_{k+1} - y_k}{y_{k+1}-y}\\
    & \xi' = \xi'(y) = \frac{2c}{-b-\sqrt{b^2-4ac}}\\
    & a = a(y) = (y_{k+1} - y_k)(s_k' - \delta_k) + (y-y_k)(\delta_{k+1}+\delta_k-2s_k')\\[7pt]
    & b = b(y) = (y_{k+1}-y_k)\delta_k-(y-y_k)(\delta_{k+1}+\delta_k-2s_k')\\[7pt]
    & c = c(y) = -s_k'(y-y_k)
\end{flalign*}
$$
The derivative for these functions can also be calculated analytically and are given by:
$$
\begin{flalign*}
& \frac{\text{d}}{\text{d}x} f_k(x) = \frac{s_k^2(\delta_{k+1}\xi^2+2s_k\xi(1-\xi)+\delta_k(1-\xi)^2)}{(s_k+(\delta_{k+1}+\delta_k-2s_k)\xi(1-\xi))^2} \quad \text{and}\\
& \frac{\text{d}}{\text{d}y} f_k^{-1}(y) = \frac{1}{\frac{\text{d}}{\text{d}x}f_k(x)} 
\end{flalign*}
$$

Inside the $[-B, B] \times [-B, B]$ interval mask, the spline is characterized by the knots it passes through, as described above. Outside of this mask, $f$ is set to be the identity function $id : \mathbb{R} \rightarrow \mathbb{R},~ x \mapsto x$, to allow $f$ to act on boundless input while retaining a finite number of parameters. 

The outermost knots are set to be at the interval edges $(x_1,y_1)\equiv(-B,-B), ~ (x_K,y_K)\equiv(B,B)$ and the corresponding derivatives are set to be $1$ to match the derivative of the identity function. Durkan et al. found this to improve the numerical stability of the Flow during training \cite{neural_spline_flows}. 

With this construction, $3(K - 1)$ parameters are needed to fully characterize a spline function $f$ : $K + 1$ sets of $x_k$ and $y_k$ coordinates for the knots, plus $K + 1$ derivatives $\delta_k$ minus the two fixed sets of coordinates and derivatives for the boundary knots.

[^1]: Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios *Neural Spline Flows*. [arXiv:1906.04032](https://arxiv.org/abs/1906.04032)

[^2]: Ivan Kobyzev, Simon J.D. Prince, Marcus A. Brubaker. *
Normalizing Flows: An Introduction and Review of Current Methods*. [arXiv:1908.09257](https://arxiv.org/abs/1908.09257)

[^3]:  J. A. GREGORY, R. DELBOURGO, *Piecewise Rational Quadratic Interpolation to Monotonic Data*. [DOI:10.1093](https://doi.org/10.1093/imanum/2.2.123)
