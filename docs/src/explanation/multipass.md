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

## Convergence sweeps and iterative passes

A pass with `max_iterations > 1` adds convergence sweeps at that window size:
it feeds its own validated field back as the deformation predictor and
re-correlates, sweep after sweep, until the field converges or the budget is
spent. Residual displacements shrink toward zero and the measurement
approaches the deformation-limited accuracy —

```julia
multipass_parameters([64, 32, 16]; final = (max_iterations = 3,))
```

This iterates the 16-px stage and stops as soon as continuing would not change
the answer. Repeating the final window size explicitly —
`multipass_parameters([64, 32, 16, 16, 16])` — is the older equivalent form
when the early exit is disabled. Convergence means the 95th percentile of the
per-vector displacement change between successive sweeps drops below
`convergence_tol` (0.05 px by default). The criterion is a
percentile rather than a maximum deliberately: a few bistable low-signal
windows flicker between correlation peaks for arbitrarily many sweeps (on
synthetic test scenes the maximum change stays near a pixel while the median
falls below 10⁻³ px), and those windows are validation's problem — a
max-norm would never converge and the early exit would be dead. Setting
`convergence_tol = 0` disables the early exit, making the pass run exactly
`max_iterations` sweeps — exactly equivalent to repeating the pass that many
times in the schedule.

Iteration also stops validation failures from cascading: within an
iterating pass a flagged vector's local-median replacement seeds the next
sweep's deformation, and the window is then *re-measured* — so replacement
artifacts relax toward measured data within the stage instead of leaking
into the next (smaller-window) pass's predictor, where a corrupted
predictor would spoil every window it touches.

Each extra sweep costs one full-image re-deformation plus one re-correlation
of every window (roughly the cost of an extra pass; the deformation is
O(image) no matter how few windows still change, which is why Hammerhead
re-correlates the whole field rather than tracking per-vector convergence).

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
corrupt the deformation for every window it touches. The sweeps of an
iterating pass replace internally for the same reason, but the *returned*
field still honors `replace_outliers`: with replacement off, cells that are
still flagged after the last sweep hold their measured displacement. Masked
regions are filled from valid neighbors before smoothing for the same reason
(see [the masking model](masking.md)).
