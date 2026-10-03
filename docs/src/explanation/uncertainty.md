# Uncertainty quantification

Set `uncertainty = true` in [`PIVParameters`](@ref) to estimate the random
error of each particle image velocimetry (PIV) vector. The result's
`uncertainty_u` and `uncertainty_v` fields contain one-standard-deviation
estimates in pixels, calculated with the correlation-statistics method of
[Wieneke2015](@citet).

## How it works

After image deformation has converged, the two deformed windows should show
the same particle pattern; any residual asymmetry between them is caused by
noise, out-of-plane loss, and local gradients. These effects
perturb the correlation peak. The method measures, pixel by pixel, the
asymmetry statistics of the correlation-difference terms and propagates
them to the displacement estimate, yielding a per-window standard
deviation for `u` and `v` separately.

## What it needs

- **A converged multi-pass schedule.** The derivation assumes the
  correlation peak sits at nearly zero residual displacement, so the
  estimate is computed on the *final pass only* and is meaningful only
  after the predictor–corrector iteration has settled. Repeat the final
  window size, e.g. `multipass_parameters([32, 16, 16]; uncertainty = true,
  ...)`, or iterate the final pass to convergence with
  `final = (max_iterations = 3,)` (see
  [iterative passes](multipass.md#Convergence-sweeps-and-iterative-passes)) —
  an iterating final pass estimates the uncertainty once, from the deformed
  windows of its last sweep.
- **Moderate noise.** The linearization is intended for uncertainties up to
  about 0.3 px. A finite estimate above that range needs caution; it is not
  automatically clipped or rejected. The estimator returns `NaN` when the
  correlation statistics do not give a valid estimate.

## What the numbers mean

The estimate describes the **random error of the correlation measurement at
that window**:

- Systematic errors (peak locking, image-deformation interpolation error
  with small particle images, calibration bias) are invisible to it;
  diagnose those with [`peak_locking`](@ref) and, for ground-truthed cases,
  [`error_statistics`](@ref).
- The estimate is *not updated* when validation replaces or substitutes a
  vector: it describes the original correlation, not the replacement.
- Some low-quality windows report very large σ. When comparing uncertainty
  against a reference error, report the valid-vector selection and inspect
  the distribution as well as its median. A few large estimates can dominate
  a mean.

Each σ is itself an estimate from the handful of particles in one window,
so it scatters by roughly 30–40% from one image pair to the next. Read a
single vector's error divided by its σ like a t-statistic with a few degrees
of freedom: on noisy synthetic images about 52–62% of errors fall within 1σ
and 89–93% within 2σ, and σ averaged over repeated pairs at the same
window covers the random error at 60–65% / 91–95%, close to the Gaussian
68% / 95%. Use about
2.5σ (95–98% coverage in the same tests) when you need a 95% interval for
an individual vector.

The variance of the correlation asymmetry is never taken below its zero-lag
term ``S_{0,0}``, the independent-pixel limit of [Wieneke2015](@citet). A
window whose deformed images differ at all therefore always reports σ > 0;
σ = 0 means the two deformed windows match exactly.

[How accurate are the measurements?](validation_results.md) gives the
synthetic coverage measurements, including the systematic error that
dominates the total error on clean images with small particles.

## Use uncertainty alongside validation and sensitivity checks

A validation flag marks a rejected vector. A replacement supplies a value
from neighboring measurements. An uncertainty estimate describes the
original correlation measurement. Check the flags before interpreting the
uncertainty of a displayed or exported value.

Repeat an analysis with another reasonable final window size and compare
profiles at the same locations. Differences can reveal spatial averaging
or sensitivity to processing even when the reported random uncertainty is
small. This comparison does not by itself identify which result is closer
to the true field; it helps determine whether the feature you need is
stable under those choices.

For a velocity component ``U = s d / \Delta t``, `physical` scales the
displacement uncertainty by ``s / \Delta t``. It does not add uncertainty in
the spatial calibration ``s`` or pair delay ``\Delta t``. Under a first-order,
independent-input approximation, their contributions combine as

```math
\sigma_U^2 \approx
\left(\frac{s}{\Delta t}\right)^2 \sigma_d^2 +
\left(\frac{d}{\Delta t}\right)^2 \sigma_s^2 +
\left(\frac{s d}{\Delta t^2}\right)^2 \sigma_{\Delta t}^2.
```

Correlated inputs require covariance terms. Tracer response, unresolved flow
structure, and calibration-model errors also need separate assessment;
the correlation estimator does not provide a complete uncertainty budget
[Sciacchitano2019](@cite). Start with the
[image-quality guide](../howto/image_quality.md) and the worked
[window-size comparison](../tutorials/real_data.md).

## Ensemble pooling

In [`run_piv_ensemble`](@ref), the per-window statistics are summed across
all pairs. The reported uncertainty describes the displacement estimated
from the combined correlation plane. Adding pairs can reduce random
uncertainty, but a decrease at every sample count is not guaranteed.

The uncertainty model assumes a common displacement. If the flow fluctuates,
the combined peak can broaden or become asymmetric, and its position need
not equal the arithmetic mean of separately measured vectors. This
uncertainty estimate does not describe the flow's fluctuation amplitude.
Use [`field_statistics`](@ref) over single-pair results to quantify that
variation; the [sequence tutorial](../tutorials/sequence_statistics.md)
compares the two calculations.

## Execution precision on GPU backends

The KernelAbstractions (KA) family of backends computes the same additive
statistics on its execution device and always accumulates them in Float64,
including for Float32 images.
An iterative pass runs one uncertainty-quantification sweep over the final
device-resident warped
windows; an ensemble keeps the pooled statistics on the device until final
analysis. GPUs with weak Float64 throughput may spend more time on UQ than on correlation;
see [Run PIV on a GPU](../howto/gpu.md) for benchmarking guidance.

## Stereo propagation

[`run_piv_stereo`](@ref) propagates the two cameras' per-window estimates
through the same least-squares operator used for the (u, v, w)
reconstruction, assuming independent per-camera errors, into
`uncertainty_u` / `uncertainty_v` / `uncertainty_w` in world units.

## Cheap proxies

Every PIV result also carries two correlation diagnostics: `peak_ratio`
compares the primary and secondary peaks, while `correlation_moment`
describes peak spread. A higher ratio or narrower peak can make a vector
easier to interpret, but neither is a calibrated uncertainty or a guarantee
that the displacement is correct. Inspect them with the vector field and
validation flags.
