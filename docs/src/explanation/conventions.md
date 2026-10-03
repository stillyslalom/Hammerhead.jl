# Coordinates, signs, and units

Hammerhead uses the coordinate and unit conventions below for its results.

## Image coordinates

Julia stores images as matrices indexed `img[row, col]`. Hammerhead names
the two directions:

- **x** runs along **columns** (image-right is +x),
- **y** runs along **rows** (image-*down* is +y, because row numbers grow
  downward).

Pixel positions are written `(x, y)` in that order. The interrogation grid
of a [`PIVResult`](@ref) uses the same frame: `result.x` holds
window-center column coordinates and `result.y` window-center row
coordinates, and field matrices such as `result.u` are indexed
`[iy, ix]` — row index first, matching the image.

## Displacement sign convention

A particle at `(row, col)` in the first image that is found at
`(row + dv, col + du)` in the second image has displacement `(du, dv)`:

- `u` is the x-displacement (along columns; positive = image-right),
- `v` is the y-displacement (along rows; positive = image-down).

In the usual display orientation, +v points *down*. The Makie plotting extension
([`plot_vector_field`](@ref)) reverses the y-axis so vector plots match the
image orientation.

Planar PIV and PTV store displacements in **pixels per frame interval**.
Stereo reconstruction stores displacements in the calibration grid's world
units per frame interval. Physical calibration is metadata: attach a
[`PhysicalScale`](@ref) (pixel size, frame interval `dt`,
and display unit labels) with the `scale` keyword of any driver or with
[`with_scale`](@ref). Attaching it leaves the arrays unchanged; convert with
[`physical`](@ref), which returns a same-type result whose positions are
lengths and whose displacements (and uncertainties) are velocities
(`pixel_size / dt`). The units are whatever you put in: millimeters and
seconds in, mm/s out. Load Unitful for quantity-based construction
(`PhysicalScale(20.0u"µm", 0.5u"ms")`).

Convert **last**: validators, [`peak_locking`](@ref), and the `epsilon` floor
used by universal outlier detection (UOD) work with pixel displacements.
Correlation diagnostics (`peak_ratio`, `correlation_moment`) remain unchanged
by `physical`. Run pixel-based checks on the raw result. A
converted result carries an identity scale with the same unit labels, so
`physical` is idempotent and plots label their axes correctly either way.
See the [scaling how-to](../howto/scaling.md).

Temporal spectra need a separate **sampling interval**: pass `dt` explicitly
to [`result_spectrum`](@ref). `PhysicalScale.dt` describes the delay between
the two images used to measure a displacement, which need not equal the time
between successive velocity samples. For example, frames acquired every
0.01 s and grouped as `(1, 2), (3, 4), …` produce velocity samples every
0.02 s. Use that 0.02 s sampling interval for spectra, whether the results
are raw or converted with `physical`.

## World coordinates (stereo)

Stereo processing works in a physical world frame `(X, Y, Z)` (units set by
the calibration target, typically mm), with `X`/`Y` in the calibration-plate
plane and `Z` out of plane. When cameras are mounted roughly upright,
[`detect_calibration_grid`](@ref) anchors +X to the lattice direction
nearest image-right and +Y to the one nearest image-*up*, making the frame
right-handed with +Z toward the cameras.

Dewarped images ([`DewarpGrid`](@ref), [`ImageDewarper`](@ref)) are indexed
in the order the grid ranges are given: `out[r, c]` shows the world point
`(grid.x[c], grid.y[r], grid.z)`. With an ascending `y` range, world +Y
therefore points *down* the dewarped image; pass a descending `y` range if
you want a display-oriented (+Y up) image. Either way the bookkeeping is
consistent: displacements measured on dewarped images convert to world
units as `du * step(grid.x)` and `dv * step(grid.y)`, signs included, and
that is exactly what [`run_piv_stereo`](@ref) does internally.

A [`StereoPIVResult`](@ref) reports `x`/`y` in world units on the dewarp
grid and `(u, v, w)` in world units per frame interval, with `w` along
world +Z. Spatial scaling is therefore already done; to get velocities,
attach a [`PhysicalScale`](@ref) with `dt` and the unit labels only
(`pixel_size` stays 1) and call [`physical`](@ref) — the per-camera `cam1`/
`cam2` results always stay in dewarped pixels.

## Particle tracking velocimetry (PTV)

Particle tracking velocimetry measures displacement per *particle* instead of
per *window*. Use it for sparse seeding or when you need individual paths,
including near walls where a PIV window would span a velocity gradient. PTV
uses the same x/y directions and pixels-per-frame units described above. Attach
a [`PhysicalScale`](@ref) and call [`physical`](@ref) to convert its results.
A [`TrackingResult`](@ref) stores positions, with velocities calculated from
successive positions. Its converted scale retains `dt`; pass that scale to
[`trajectory_velocities`](@ref) to obtain physical velocities.

With explicit `sample_times`, [`track_particles`](@ref) returns a
[`TimedTrackingResult`](@ref). Prediction and validation then use actual elapsed
intervals. Its velocity method applies spatial `pixel_size` and divides by
recorded time differences; `PhysicalScale.dt` does not divide these values again.
Endpoint velocities span adjacent observations; interior velocities span the
two outer observations. On irregular samples, those secants are interval-average
slopes, not instantaneous derivatives at the central observation. See
[actual-time tracking](../howto/tracking_timing.md) for units and persistence.

**Frame-A attribution.** A [`PTVResult`](@ref) reports `x`/`y` as the
*frame-A* particle positions and `u`/`v` as the displacement to frame B. This
differs from multipass PIV with symmetric image deformation, which attributes
each vector to the *midpoint* of the trajectory. The synthetic generator uses
forward-Euler motion, so a PTV displacement can be compared directly with the reference
velocity at the frame-A position multiplied by the frame interval.

**Flag, don't replace.** A tracked displacement is a measurement of one
specific particle. The scattered outlier test [Duncan2010](@cite) only
*flags* suspicious vectors (in
`result.outliers`) and leaves `u`/`v` untouched — unlike PIV, which replaces
flagged windows with a local median. In the multi-frame tracker, a flagged link
is excluded from the trajectory so it does not affect later predictions.

**Hybrid by default.** With sparse seeding the true displacement can exceed the
particle spacing, and pure nearest-neighbor matching then links the wrong
particles. [`run_ptv`](@ref) therefore runs a coarse [`run_piv`](@ref)
internally to predict where each particle goes, and matches against that
prediction [Keane1995](@cite). Pass an existing `PIVResult`, a displacement
NamedTuple, or `nothing` (pure nearest neighbor) to override.

## Grid layout

Interrogation windows use a stride of `window_size - overlap`;
`result.x`/`result.y` are their window
*centers*. All per-vector fields (`u`, `v`, `peak_ratio`, `outliers`, …)
share the `(length(y), length(x))` grid shape.

The mask stored in a [`PIVResult`](@ref) uses `true` for excluded windows.
[`extract_region`](@ref) returns an `included` grid with `true` for windows
selected and valid within the requested region. Its older `mask` field is an
alias for `included`, so it also uses `true` for included windows.

For area-form [`circulation`](@ref), a masked or nonfinite cell can leave part
of the requested region uncovered. The default call raises an error in that
case. Use `coverage=:report` to inspect the valid and requested areas and
the `complete` flag before using a partial integral.

With the default `search_area_size == window_size`, tiling begins at the
top-left corner. A larger centered search area moves the outer centers inward
so its full footprint stays inside both images, without changing the stride.
