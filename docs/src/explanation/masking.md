# The masking model

Geometry, walls, reflections, or areas outside a camera's view can obscure
the particle pattern. Mark these regions before analysis so they do not
contribute misleading vectors.

## Masks are image-sized and `true` = excluded

An analysis mask is a `Bool` matrix the size of the images, with `true`
marking pixels to *exclude*. Build one with [`polygon_mask`](@ref), load one
from an image file with [`load_mask`](@ref), or use any Bool array; combine
regions with `.|`. Pass it as `run_piv(imgA, imgB, passes; mask)`.

A mask passed to `run_piv` describes **static image geometry** and is not
warped between passes. Sequence drivers also accept one mask per pair, one
mask per frame (the two exposure masks are unioned), or an
`(i, frameA, frameB) -> mask` callback. The union marks pixels excluded in
either exposure; `mask_threshold` then determines which windows are dropped.

## Masked ≠ outlier

[`PIVResult`](@ref) keeps two per-vector flags with different meanings:

- `result.mask` — windows **dropped** when enough of an interrogation or
  search footprint is masked. They
  carry *no measurement*: `u`, `v`, `peak_ratio`, and `correlation_moment`
  are `NaN` there. A masked window is not "bad data"; it is no data.
- `result.outliers` — windows that produced a measurement which then
  **failed validation**. When replacement is active, `u`/`v` at these
  positions hold the local-median replacement.

A window is never both: masked cells are excluded from validation entirely.

## How masked pixels enter the correlation

A node is dropped when the masked-pixel fraction reaches `mask_threshold`
(default 0.5) in either its frame-A interrogation footprint or frame-B search
footprint. Footprints *below* the threshold are still correlated, over their
valid pixels only: masked pixels are loaded at the mean of the window's
valid pixels, which is zero after the correlator's mean subtraction. This
avoids introducing an artificial intensity step at the mask boundary. A
window partly covered by a mask still has less particle information, so
inspect its correlation and validation diagnostics.

## Masked regions downstream

The exclusion propagates through every stage that looks at neighbors:

- **Validation** — universal outlier detection never flags a masked cell
  and never includes one in a neighbor median (`NaN`s cannot poison the
  test).
- **Replacement** — local-median replacement neither fills masked cells nor
  draws donor vectors from them.
- **Multi-pass predictor** — before smoothing and deformation, masked cells
  are filled from valid neighbors so the interpolated predictor stays
  finite everywhere; the filled values are only used to condition the
  deformation, and the cells are re-dropped in the pass output.
- **Statistics** — [`field_statistics`](@ref), [`error_statistics`](@ref),
  and friends skip masked cells via their valid-sample logic.

## Stereo masks

Dewarping produces a validity mask per camera (`dw.mask`, grid-sized,
`true` = that camera cannot see the node). [`run_piv_stereo`](@ref) and
[`self_calibrate`](@ref) combine both cameras' masks with any user mask
(`dw1.mask .| dw2.mask .| user`). Both cameras use the same grid and
exclusion map, with window drops determined by `mask_threshold`.
