# 26. Numerical Analysis for Compiler and Kernel Engineers

Optimizations change arithmetic order, precision, and sometimes the approximation itself. A frontier
systems engineer needs to explain the resulting error, not just increase a tolerance until tests pass.
This chapter supplies the numerical reasoning behind the precision experiments in Chapter 24.

## 26.1 Conditioning versus implementation error

Conditioning describes how sensitive the mathematical problem is to input perturbations. Stability
describes the error introduced by an algorithm. A perfectly implemented low-precision operation can
still produce a large output error for an ill-conditioned problem.

For a sum, compare `abs(sum(x))` with `sum(abs(x))`. When positive and negative terms nearly cancel,
a tiny relative perturbation of inputs can cause a large relative change in the result. Consequently,
relative error alone is a poor diagnostic. Report maximum absolute error, relative error away from
zero, and a norm-based error suited to the operation.

**Lab:** sum `[1e8, 1, -1e8]` in f32 in different orders, then compare naive, pairwise, and compensated
summation on random and cancellation-heavy arrays. Use a higher-precision reference and state its
limitations. Separate input conversion error from accumulation error.

## 26.2 Dot products and error budgets

Under the usual model `fl(a op b)=(a op b)*(1+delta)` with `abs(delta)<=u` for normal finite results,
a sequential sum has an error bound involving `gamma_n=n*u/(1-n*u)` when `n*u<1`. A dot-product
bound scales with `sum(abs(a_i*b_i))`; severe cancellation can make relative output error large.
Overflow, underflow, subnormals, and exceptional values require separate treatment.

Fused multiply-add rounds a multiply-plus-add once rather than rounding the product separately.
It can improve accuracy while changing bitwise results. Tensor-core products and framework reference
kernels may have different input conversion and accumulation policies. “Both f32” is not a complete
numerical specification.

**Lab:** sweep reduction length and input scale for each supported precision path. Predict where
error should grow, compare against measurement, and identify saturation or underflow. Write an error
budget for GEMM plus bias plus activation instead of choosing an unexplained global tolerance.

## 26.3 Transcendentals, softmax, and normalization

Approximations for exp, log, reciprocal, and reciprocal square root have domains and error profiles.
Range reduction maps the argument to a smaller interval, a polynomial or table approximates the
function there, and reconstruction maps back. Exceptional inputs and boundary continuity still matter.

Stable softmax subtracts the row maximum before exponentiation. This prevents many overflows but
does not make every masking configuration meaningful. Normalization statistics can lose precision
when computed as `mean(x*x)-mean(x)^2` for nearly constant large values; a centered or online method
can be more stable. Define population versus sample variance explicitly.

**Lab:** implement an online mean/variance algorithm and a merge rule for two partial groups. Compare
with the naive expression on near-constant inputs. Audit a fused normalization kernel's variance,
epsilon placement, accumulator precision, and backward formula.

## 26.4 Gradient and training validation

Finite differences test a local derivative but have truncation and cancellation errors. Sweep the
step size rather than trusting one value. Directional derivatives reduce cost: compare a finite
difference along v with the inner product of the analytic gradient and v. A vector-Jacobian product
and Jacobian-vector product satisfy an adjoint identity that can test implementations without
constructing the full Jacobian.

Training validation also requires optimizer equations, reduction conventions, gradient accumulation,
clipping, random-number behavior, and checkpoint state. Two correct nondeterministic kernels can
produce diverging trajectories; compare distributions and short matched steps before attributing
all differences to a bug or all differences to randomness.

**Exit gate:** a numerical test report for GEMM, softmax, normalization, and gradients; explicit
precision policies; and a short training comparison that explains observed deviations.

## 26.5 Worked lab: mergeable variance and quantization error

A running variance state stores `(n, mean, M2)`, where M2 is the sum of squared deviations from the
current mean. After receiving x, set `delta=x-mean`, update the mean by `delta/(n+1)`, and add
`delta*(x-new_mean)` to M2. This avoids subtracting two nearly equal large second moments.

For two groups A and B, let `delta=mean_B-mean_A` and `n=n_A+n_B`. The merged M2 is
`M2_A+M2_B+delta^2*n_A*n_B/n`. The extra term accounts for the groups having different centers.
An empty group is the identity. This merge rule makes a parallel reduction possible, with the
usual floating-point dependence on reduction order.

```bash
.venv/bin/python -m labs.numerics
```

@include labs/numerics.py

The lab evaluates near-constant data around 1e8 using double precision. Changing it to f32 before
the calculation can erase small variations at input conversion; no variance algorithm can recover
information already rounded away. Separate storage error from algorithmic error.

For per-channel int8 quantization, the chosen scale makes the largest magnitude map to 127. With
nearest rounding and no additional clipping of finite in-range values, reconstruction error is at
most half a scale step, apart from floating-point rounding in the quantizer. Zero channels use scale
1 and quantized value 0, avoiding division by zero. NumPy's rounding specifies ties to even here.

**Experiment:** check the half-step error bound channel by channel, add a single extreme outlier,
and compare per-channel with a global scale. Then evaluate the perturbed weights in a model: a small
weight error bound does not alone bound validation loss. This is a numerical reference lab, not an
integer hardware GEMM or a full quantized inference runtime.
