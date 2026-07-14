# Stereo geometry and self-calibration

Planar particle image velocimetry (PIV) measures two displacement components
projected onto the image
plane. With two cameras viewing the light sheet from different angles,
the projections differ, and the difference encodes the out-of-plane
component. This page explains Hammerhead's stereo chain: calibration →
dewarping → reconstruction → self-calibration.

## Camera calibration

A camera model maps world points `(X, Y, Z)` to pixel coordinates.
Hammerhead provides two, both fit with [`calibrate_camera`](@ref) from
(pixel, world) point pairs — typically produced by imaging a dot-grid
calibration plate at several Z positions and running
[`detect_calibration_grid`](@ref):

- [`PinholeCamera`](@ref) — a projective model fitted by direct linear
  transformation (DLT). Physically grounded
  and exactly invertible, but it cannot represent lens distortion or
  refraction. Needs points on ≥ 2 Z planes.
- [`SoloffCamera`](@ref) — the 19-term polynomial of
  [Soloff1997](@citet), cubic in X/Y and quadratic in Z. It absorbs
  distortion and refraction empirically, at the cost of a Newton-iterated
  inverse. Needs ≥ 3 Z planes. This is the default and the standard choice
  for real rigs.

Check any fit with [`calibration_quality`](@ref); see the
[stereo-rig how-to](../howto/stereo_rig.md) for what residuals to expect
from real plates.

## Dewarping to a common plane

Rather than correlating raw images and reconciling geometry afterwards,
Hammerhead resamples both cameras' images onto one regular grid of world
coordinates in the measurement plane (a [`DewarpGrid`](@ref) shared by the
rig, one [`ImageDewarper`](@ref) per camera). After dewarping, the same
world point sits at the same pixel in both cameras' images, so the standard
2D engine can run per camera with identical parameters, and the two vector
grids match point for point. Nodes outside a camera's view are recorded in
a validity mask; the analysis is restricted to the stereo overlap (see
[the masking model](masking.md)).

## Three-component reconstruction

At each grid point, camera *i*'s viewing ray drifts in-plane by
`(tXᵢ, tYᵢ)` world units per unit Z (evaluated from the calibration by
central differences). A world displacement `(dx, dy, dz)` therefore appears
in camera *i*'s dewarped two-component (2C) field as

```
uᵢ = dx − dz·tXᵢ,    vᵢ = dy − dz·tYᵢ.
```

Two cameras give four equations for three unknowns;
[`run_piv_stereo`](@ref) solves them by least squares per vector. The
farther apart the viewing angles, the better conditioned `dz` is; with
parallel rays (identical cameras) the system is degenerate and the vector
comes out `NaN`. Per-camera uncertainties propagate through the same
operator (see [uncertainty quantification](uncertainty.md)).

The result is a [`StereoPIVResult`](@ref): world-coordinate grid,
`(u, v, w)` in world units per frame interval, union outlier/mask flags,
and both per-camera 2C results retained for diagnostics. To turn the
displacements into velocities, attach a [`PhysicalScale`](@ref) with `dt`
only and call [`physical`](@ref) — see the
[scaling how-to](../howto/scaling.md).

## Self-calibration

Calibration assumed the plate sat exactly in the light sheet. In practice
it never does — millimeter offsets and small tilts are routine — and the
error contaminates `w` directly. The fix is *disparity self-calibration*
[Wieneke2005](@citet), implemented as [`self_calibrate`](@ref):

1. **Measure the disparity.** Dewarp both cameras' images of the *same
   instant* and cross-correlate them (ensemble sum-of-correlation over
   several instants, one pass with large windows). If the sheet were
   exactly at the assumed plane, the two views would coincide; any
   systematic disparity field measures the misregistration.
2. **Triangulate.** Each disparity vector, attributed symmetrically to the
   two viewing rays, is triangulated to a world point on the *true* sheet.
   Vectors with large triangulation residuals are rejected as false
   correlations.
3. **Fit and correct.** A plane is fitted through the triangulated points,
   and both camera models are rigidly transformed so the fitted plane
   becomes the measurement plane. A `PinholeCamera` absorbs the transform
   exactly into its projection matrix; other models are wrapped in a
   [`TransformedCamera`](@ref).
4. **Iterate.** The measurement-correction loop repeats (the symmetric
   disparity attribution is second-order exact, and iteration absorbs the
   rest), then a final measurement verifies convergence.

The returned dewarpers are drop-in replacements for
[`run_piv_stereo`](@ref), and the [`SelfCalibrationReport`](@ref) records
per-pass disparity statistics, fitted planes, and the cumulative world
transform. Well-corrected setups converge to a residual disparity
root-mean-square (RMS) below
0.1 px on real recordings [Wieneke2005](@cite); the synthetic test fixtures
reach 0.01 px.

One caveat: self-calibration moves the world frame's Z origin onto the
actual light sheet, anchored so that camera 1's view barely moves. If your
downstream analysis depends on the original plate-defined frame, the
cumulative transform in the report (`R`, `t`) maps corrected-frame
coordinates back to it.
