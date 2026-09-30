# Stereo geometry and self-calibration

Planar particle image velocimetry (PIV) measures two displacement components
in an image plane. Two cameras viewing the same light sheet from different
angles provide distinct projections; their difference helps determine the
out-of-plane component. Stereo analysis calibrates each camera, dewarps both
views onto a common plane, corrects sheet misalignment, and reconstructs
three displacement components.

## Camera calibration

A camera model maps world points `(X, Y, Z)` to pixel coordinates.
Hammerhead provides two, both fit with [`calibrate_camera`](@ref) from
(pixel, world) point pairs — typically produced by imaging a dot-grid
calibration plate at several Z positions and running
[`detect_calibration_grid`](@ref):

- [`PinholeCamera`](@ref) is a projective model fitted by direct linear
  transformation (DLT). Pixel-to-world coordinates at a specified Z plane
  can be solved directly, but the model does not represent lens distortion
  or refraction. It needs points on at least two Z planes.
- [`SoloffCamera`](@ref) uses the 19-term polynomial of
  [Soloff1997](@citet), cubic in X/Y and quadratic in Z. It can accommodate
  distortion and refraction empirically, at the cost of a Newton-iterated
  inverse at a specified Z plane. It needs at least three Z planes and is
  the default model.

Check any fit with [`calibration_quality`](@ref); see the
[stereo-rig how-to](../howto/stereo_rig.md) for what residuals to expect
from real plates.

## Dewarping to a common plane

Hammerhead resamples both cameras' images onto one regular grid of world
coordinates in the measurement plane (a [`DewarpGrid`](@ref) shared by the
rig, one [`ImageDewarper`](@ref) per camera). After dewarping, a world-grid
location has the same image index in both views, so the standard
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
smaller the difference between their viewing directions, the less stable the
`dz` estimate; with parallel rays the system is degenerate and the vector is
`NaN`. Camera angles also affect how much particle pattern both cameras can see.
Per-camera uncertainties propagate through the same
operator (see [uncertainty quantification](uncertainty.md)).

The result is a [`StereoPIVResult`](@ref): world-coordinate grid,
`(u, v, w)` in world units per frame interval, union outlier/mask flags,
and both per-camera 2C results retained for diagnostics. To turn the
displacements into velocities, attach a [`PhysicalScale`](@ref) with `dt`
only and call [`physical`](@ref) — see the
[scaling how-to](../howto/scaling.md).

## Self-calibration

Calibration uses the plate position as the measurement plane. If the light
sheet is offset or tilted relative to the plate, the reconstruction can be
biased. [`self_calibrate`](@ref) corrects this misalignment using disparity
self-calibration [Wieneke2005](@cite):

1. **Measure the disparity.** Dewarp both cameras' images of the *same
   instant* and cross-correlate them (ensemble sum-of-correlation over
   several instants, one pass with large windows). If the sheet were
   exactly at the assumed plane and the camera models were accurate, the
   two views would align. A systematic disparity can indicate sheet
   misregistration; camera-model error can contribute too.
2. **Triangulate.** Each disparity vector, attributed symmetrically to the
   two viewing rays, is triangulated to a world point on the *true* sheet.
   Vectors with large triangulation residuals are rejected as false
   correlations.
3. **Fit and correct.** A plane is fitted through the triangulated points,
   and both camera models are rigidly transformed so the fitted plane
   becomes the measurement plane. A `PinholeCamera` absorbs the transform
   exactly into its projection matrix; other models are wrapped in a
   [`TransformedCamera`](@ref).
4. **Iterate.** The measurement-correction loop repeats until the residual
   meets the stopping criterion or the correction limit is reached. A final
   measurement records the remaining disparity.

Use the returned dewarpers with [`run_piv_stereo`](@ref). The
[`SelfCalibrationReport`](@ref) records
per-pass disparity statistics, fitted planes, and the cumulative world
transform. Inspect the remaining disparity alongside calibration residuals
and image overlap; a thick sheet or decorrelation between views can keep it
above the requested tolerance.

Self-calibration places the corrected frame's measurement plane on the
fitted light sheet and anchors its orientation to the original frame. If your
downstream analysis depends on the original plate-defined frame, the
cumulative transform in the report (`R`, `t`) maps corrected-frame
coordinates back to it.
