# Uncertainty quantification

With `uncertainty = true` in [`PIVParameters`](@ref), Hammerhead estimates a
per-vector measurement uncertainty — one standard deviation, in pixels —
into the `uncertainty_u` / `uncertainty_v` fields of the result. The
estimator is the *correlation statistics* method of
[Wieneke2015](@citet), implemented from the paper.

## How it works

After image deformation has converged, the two deformed windows should show
the same particle pattern; any residual asymmetry between them is caused by
noise, out-of-plane loss, and local gradients — exactly the effects that
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
- **Moderate noise.** The method is accurate for uncertainties up to about
  0.3 px; beyond that (or when the window has no usable correlation
  signal) the fields hold `NaN`.

## What the numbers mean

The estimate describes the **random error of the correlation measurement at
that window** — nothing more:

- Systematic errors (peak locking, calibration bias) are invisible to it;
  diagnose those with [`peak_locking`](@ref) and, for ground-truthed cases,
  [`error_statistics`](@ref).
- The estimate is *not updated* when validation replaces or substitutes a
  vector: it describes the original correlation, not the replacement.
- Windows that are nearly outliers legitimately report very large σ. When
  comparing uncertainty against a reference error, use medians over
  non-outlier vectors rather than means — a handful of near-outlier
  windows otherwise dominates.

On synthetic noise sweeps, the median estimate tracks the measured RMS
error within about ±25% up to 20% image noise.

## Ensemble pooling

In [`run_piv_ensemble`](@ref), the per-window statistics are summed across
all pairs — the ensemble correlation plane is itself such a sum — so the
reported uncertainty describes the *ensemble-mean* vector and shrinks as
pairs are added. Like ensemble correlation itself, this assumes the same
displacement in every pair; genuine pair-to-pair flow fluctuation is not
captured. Quantify fluctuation with [`field_statistics`](@ref) over
single-pair results instead.

## Execution precision on GPU backends

The KA-family backends compute the same additive statistics on their execution
device and always accumulate them in Float64, including for Float32 images.
An iterative pass runs one UQ sweep over the final device-resident warped
windows; an ensemble keeps the pooled statistics on the device until final
analysis. The CPU finalizer remains the correctness reference. Consumer GPUs
with weak Float64 throughput may spend more time on UQ than on correlation;
see [Run PIV on a GPU](../howto/gpu.md) for benchmarking guidance.

## Stereo propagation

[`run_piv_stereo`](@ref) propagates the two cameras' per-window estimates
through the same least-squares operator used for the (u, v, w)
reconstruction, assuming independent per-camera errors, into
`uncertainty_u` / `uncertainty_v` / `uncertainty_w` in world units.

## Cheap proxies

Every result also carries two always-on quality indicators: `peak_ratio`
(primary-to-secondary correlation peak ratio; higher is more reliable) and
`correlation_moment` (peak second moment; lower is sharper). They rank
vectors well but are not calibrated uncertainties.
