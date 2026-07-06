# Multi-pass interrogation and image deformation

A single correlation pass faces a hard trade-off: small windows resolve
fine flow structure but can only measure small displacements reliably (the
quarter-window rule), while large windows measure large displacements but
average over structure. Multi-pass interrogation with window refinement
gets both, and image deformation extends it to strongly sheared flows
[Scarano2002](@cite).

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
shrink across passes — `multipass_parameters([64, 32, 16])` — without
violating the quarter-window limit, even when the total displacement is
large.

## Symmetric (central-difference) deformation

Hammerhead deforms *both* images symmetrically: image A is resampled shifted
by −d/2 and image B by +d/2, where d is the predictor displacement at each
pixel (cubic B-spline resampling). Content displaced by exactly d is then
aligned in both outputs. Compared to deforming only one image, the
symmetric scheme is second-order accurate: the measurement is centered at
the midpoint of the particle trajectory, which cancels the leading-order
bias in curved or sheared flow.

## Convergence sweeps

Repeating the final window size — `multipass_parameters([64, 32, 16, 16])`
— adds convergence sweeps: extra passes at constant resolution that let the
predictor–corrector iteration settle. Residual displacements shrink toward
zero and the measurement approaches the deformation-limited accuracy.

Two features **require** a converged schedule:

- Per-vector [uncertainty quantification](uncertainty.md) assumes the
  correlation peak of the deformed windows sits at nearly zero residual;
  it runs on the final pass only.
- The ~0.03 px RMS accuracy figure of the
  [padded + apodized configuration](correlation.md) is only reached once
  the residual is small.

## Validation between passes

Every pass validates its field (see [`PIVParameters`](@ref)) and
intermediate passes *always* replace invalid vectors regardless of the
`replace_outliers` setting — a spike in the predictor would otherwise
corrupt the deformation for every window it touches. Masked regions are
filled from valid neighbors before smoothing for the same reason (see
[the masking model](masking.md)).
