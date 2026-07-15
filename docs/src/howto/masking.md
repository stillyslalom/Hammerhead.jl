# Mask reflections and geometry

**Goal:** exclude a region — model geometry, a wall, a persistent
reflection — from the analysis so it produces no vectors and cannot
contaminate its neighbors. Background on the semantics is in
[the masking model](../explanation/masking.md).

## Draw a polygon mask

[`polygon_mask`](@ref) rasterizes a polygon given as `(x, y)` corners in
pixel coordinates (`x` along columns):

```julia
using Hammerhead

mask = polygon_mask(size(imgA), [(120, 40), (480, 90), (450, 300), (100, 260)])
result = run_piv(imgA, imgB, passes; mask)
```

Combine several exclusion regions with element-wise OR:

```julia
blade  = polygon_mask(size(imgA), blade_vertices)
glare  = polygon_mask(size(imgA), glare_vertices)
result = run_piv(imgA, imgB, passes; mask = blade .| glare)
```

## Load a mask image

Many datasets ship a mask as an image file (e.g. the PIV Challenge case 1C
impeller mask). [`load_mask`](@ref) thresholds it into a Bool matrix:

```julia
mask = load_mask("mask.png")                 # bright pixels = excluded
mask = load_mask("mask.png"; invert = true)  # dark pixels = excluded
```

Any image-sized `Bool`/`BitMatrix` works as a mask — build one from raw
pixel logic if that's easier, e.g. `mask = background .> 0.8` to exclude
persistently bright (reflective) pixels of a
[`compute_background`](@ref) image.

For common bright reflections, dark bodies, low-texture regions, or sharp
geometry edges, [`automatic_mask`](@ref) provides intensity-, local-contrast-,
and edge-derived masks. Its two-frame method applies the detector to both
frames and returns their union, the safe pair-mask convention for a moving
boundary:

```julia
mask = automatic_mask(imgA, imgB; method = :intensity,
                      threshold = 0.9, side = :high, grow = 3)
mask = grow_mask(mask, 2)       # add another safety margin
mask = shrink_mask(mask, 1)     # trim an over-conservative mask
```

## Control when a window is dropped

Windows whose masked-pixel fraction reaches `mask_threshold` (default 0.5)
are dropped entirely; windows below it correlate over their valid pixels
only. Lower the threshold to be more conservative near edges:

```julia
result = run_piv(imgA, imgB, passes; mask, mask_threshold = 0.25)
```

## Read the results correctly

```julia
result.mask       # true = window dropped (NaN fields, no measurement)
result.outliers   # true = measured but failed validation (replaced if enabled)
valid = .!(result.mask .| result.outliers)
```

Masked cells are *never* counted as outliers, and `NaN`s in `u`/`v` at
masked cells are intentional — filter with `valid` before computing your
own statistics (the built-in [`field_statistics`](@ref) and
[`error_statistics`](@ref) already do).

## Batch and stereo

The same `mask` keyword flows through [`run_piv_sequence`](@ref) and
[`run_ptv_sequence`](@ref). It may be one static mask, a per-pair sequence, a
per-frame sequence (the two exposure masks are unioned), or an
`(i, frameA, frameB) -> mask` callback. Ensemble correlation uses one static
mask for all pairs. For stereo, pass a *grid-sized* mask to [`run_piv_stereo`](@ref) /
[`self_calibrate`](@ref); it is combined with the dewarpers' out-of-view
masks automatically.
