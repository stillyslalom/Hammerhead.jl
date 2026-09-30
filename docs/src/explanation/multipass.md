# Multi-pass interrogation and image deformation

In particle image velocimetry (PIV), small windows sample more local flow
but lose more particle pairs at their boundaries for a given displacement.
Large windows retain more particle information while averaging over a wider
area. Multi-pass interrogation starts with large windows and refines the
measurement with smaller ones. Image deformation reduces the remaining shift
seen by those smaller windows [Scarano2002](@cite).

For an initial pass with equal-sized windows, displacement below about one
quarter of the window width is a useful starting guideline, not a hard
algorithmic limit. Particle density, velocity gradients, and out-of-plane
particle loss also affect whether a window yields a usable measurement.
Inspect these with the [image-quality guide](../howto/image_quality.md).

Window overlap sets the distance between vector locations. For example,
32 px windows with 16 px overlap give 16 px vector spacing; the measurement
still uses a 32 px interrogation footprint. Increasing overlap alone does
not resolve smaller structures. Compare final window sizes and the features
they recover, as in the [tip-vortex tutorial](../tutorials/real_data.md).

## The predictor–corrector loop

`run_piv(imgA, imgB, passes)` with a vector of [`PIVParameters`](@ref) —
conveniently built with [`multipass_parameters`](@ref) — runs one
correlation pass per entry:

1. The first pass measures the field at the coarsest window size.
2. Each later pass takes the previous pass's *validated* field as a
   predictor: the field is smoothed (a 3×3 binomial kernel, controlled by
   `predictor_smoothing`), interpolated to pixel resolution, and used to
   deform the images.
3. The pass then correlates the deformed images, measuring only the small
   *residual* displacement, and adds the predictor back.

Because each pass only needs to measure the residual, window sizes can
shrink across passes, as in `multipass_parameters([64, 32, 16])`, even when
the total displacement would be too large for a useful single-pass
measurement at the final size.

## Symmetric (central-difference) deformation

Hammerhead deforms *both* images symmetrically: image A is resampled shifted
by −d/2 and image B by +d/2, where d is the predictor displacement at each
pixel (cubic B-spline resampling). Content displaced by exactly d is then
aligned in both outputs. The resulting vector is attributed to the
trajectory midpoint. For smooth flow, this symmetric placement removes the
leading-order position bias that would arise from assigning a displacement
to only one endpoint.

## Convergence sweeps and iterative passes

A pass with `max_iterations > 1` repeats correlation at that window size,
using its latest validated field as the next deformation predictor. It stops
when the field converges or reaches `max_iterations`:

```julia
multipass_parameters([64, 32, 16]; final = (max_iterations = 3,))
```

This iterates the 16-px stage. With early exit disabled,
`multipass_parameters([64, 32, 16, 16, 16])` gives the same number of sweeps.
Convergence means the 95th percentile of the
per-vector displacement change between successive sweeps drops below
`convergence_tol` (0.05 px by default). This percentile lets the pass stop
when most vectors have settled, even if a few low-signal windows alternate
between peaks. Inspect or reject those windows through validation. Setting
`convergence_tol = 0` disables the early exit, making the pass run exactly
`max_iterations` sweeps, equivalent to repeating the pass that many
times in the schedule.

Within an iterating pass, a flagged vector's local-median replacement seeds
the next deformation, and the window is measured again. This gives that
window another chance to produce a valid measurement before the next,
smaller-window pass uses the field as its predictor.

Each extra sweep deforms both images and correlates every window, so budget
roughly the time of another pass for each sweep.

Convergence matters for two reasons:

- Per-vector [uncertainty quantification](uncertainty.md) assumes the
  correlation peak of the deformed windows sits at nearly zero residual;
  it runs on the final pass only.
- A small residual makes the final correlation easier to fit. It does not
  establish the accuracy of the result: image quality, particle loss, and
  unresolved velocity gradients still affect the measurement.

## Validation between passes

Every pass validates its field (see [`PIVParameters`](@ref)) and
intermediate passes *always* replace invalid vectors regardless of the
`replace_outliers` setting — a spike in the predictor would otherwise
corrupt the deformation for every window it touches. The sweeps of an
iterating pass replace internally for the same reason, but the *returned*
field still honors `replace_outliers`: with replacement off, cells that are
still flagged after the last sweep hold their measured displacement. Masked
regions are filled from valid neighbors before smoothing for the same reason
(see [the masking model](masking.md)).
