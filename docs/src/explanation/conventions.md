# Coordinates, signs, and units

Every quantity Hammerhead produces follows one consistent coordinate
convention. This page states it once, so the other pages don't have to.

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
`(row + dv, col + du)` in the second image yields a **positive**
displacement `(du, dv)`:

- `u` is the x-displacement (along columns; positive = image-right),
- `v` is the y-displacement (along rows; positive = image-down).

This is the natural convention for image data, but note that +v points
*down* in the usual display orientation. The Makie plotting extension
([`plot_vector_field`](@ref)) reverses the y-axis so vector plots match the
image orientation.

Displacements are **measured in pixels per frame interval**, and the stored
arrays of every result keep those measured units. Physical calibration is
metadata: attach a [`PhysicalScale`](@ref) (pixel size, frame interval `dt`,
and display unit labels) with the `scale` keyword of any driver or with
[`with_scale`](@ref) — the arrays don't change — and convert explicitly with
[`physical`](@ref), which returns a same-type result whose positions are
lengths and whose displacements (and uncertainties) are velocities
(`pixel_size / dt`). The units are whatever you put in: millimeters and
seconds in, mm/s out. Load Unitful for quantity-based construction
(`PhysicalScale(20.0u"µm", 0.5u"ms")`).

Convert **last**: validators, [`peak_locking`](@ref), the `epsilon` floor used
by universal outlier detection (UOD),
and the correlation diagnostics (`peak_ratio`, `correlation_moment`) are
pixel-native and are never converted — run them on the raw result. A
converted result carries an identity scale with the same unit labels, so
`physical` is idempotent and plots label their axes correctly either way.
See the [scaling how-to](../howto/scaling.md).

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

Particle tracking velocimetry measures the same displacement as particle
image velocimetry (PIV) but per *particle*
instead of per *window*, and it is the better tool wherever the seeding is
sparse or Lagrangian information matters: low-density flows, the near-wall
region where large windows straddle a velocity gradient, or any case where you
want individual particle paths rather than an Eulerian field. Everything on
this page still holds — `u` along x, `v` along y, pixels per frame interval,
physical units via an attached [`PhysicalScale`](@ref) and
[`physical`](@ref) — with three PTV-specific conventions. (One tracking
nuance: a [`TrackingResult`](@ref) stores only positions, velocities being
derived by differencing, so its converted scale keeps `dt` — pass it to
[`trajectory_velocities`](@ref) for physical velocities.)

**Frame-A attribution.** A [`PTVResult`](@ref) reports `x`/`y` as the
*frame-A* particle positions and `u`/`v` as the displacement to frame B. This
differs from PIV's symmetric image deformation, which attributes each vector to
the *midpoint* of the trajectory. Frame-A attribution matches the
forward-Euler contract of the synthetic generator exactly (each particle's true
displacement is the velocity at its launch point), so ground-truth comparisons
are direct — no midpoint correction, no interpolation error.

**Flag, don't replace.** A tracked displacement is a measurement of one
specific particle; there is nothing meaningful to substitute for it. So the
scattered outlier test [Duncan2010](@cite) only *flags* suspicious vectors (in
`result.outliers`) and leaves `u`/`v` untouched — unlike PIV, which replaces
flagged windows with a local median. In the multi-frame tracker, a flagged link
is simply not made (it would poison the constant-velocity predictor
downstream).

**Hybrid by default.** With sparse seeding the true displacement can exceed the
particle spacing, and pure nearest-neighbor matching then links the wrong
particles. [`run_ptv`](@ref) therefore runs a coarse [`run_piv`](@ref)
internally to predict where each particle goes, and matches against that
prediction [Keane1995](@cite) — so it works out of the box at realistic
displacements. Pass an existing `PIVResult`, a displacement NamedTuple, or
`nothing` (pure nearest neighbor) to override.

## Grid layout

Interrogation windows use a stride of `window_size - overlap`;
`result.x`/`result.y` are their window
*centers*. All per-vector fields (`u`, `v`, `peak_ratio`, `outliers`, …)
share the `(length(y), length(x))` grid shape.

With the default `search_area_size == window_size`, tiling begins at the
top-left corner. A larger centered search area moves the outer centers inward
so its full footprint stays inside both images, without changing the stride.
