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

Displacements are reported in **pixels per frame interval**. Hammerhead does
not apply time or magnification scaling: converting to physical velocity
(multiplying by pixel size and dividing by the inter-frame time) is left to
the caller.

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
world +Z. As in 2D, no time scaling is applied.

## Grid layout

Interrogation windows tile the image starting at the top-left corner with a
stride of `window_size - overlap`; `result.x`/`result.y` are the window
*centers*. All per-vector fields (`u`, `v`, `peak_ratio`, `outliers`, …)
share the `(length(y), length(x))` grid shape.
