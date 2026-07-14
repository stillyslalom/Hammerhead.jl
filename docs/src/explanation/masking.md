# The masking model

Real recordings contain regions that must not produce vectors: model
geometry, walls, reflections, or (in stereo) parts of the plane one camera
cannot see. Hammerhead handles these with a single, consistent masking
model.

## Masks are static, image-sized, and `true` = excluded

An analysis mask is a `Bool` matrix the size of the images, with `true`
marking pixels to *exclude*. Build one with [`polygon_mask`](@ref), load one
from an image file with [`load_mask`](@ref), or use any Bool array; combine
regions with `.|`. Pass it as `run_piv(imgA, imgB, passes; mask)`.

The mask describes **static lab-frame geometry**. It is deliberately *not*
warped between passes of a multi-pass run: a wall stays where it is no
matter what the flow does. (Per-frame dynamic masks are not currently
supported.)

## Masked ≠ outlier

Hammerhead keeps two distinct per-vector flags in a [`PIVResult`](@ref),
and conflating them is the most common masking mistake in particle image
velocimetry (PIV) software:

- `result.mask` — windows **dropped** because they overlap the mask. They
  carry *no measurement*: `u`, `v`, `peak_ratio`, and `correlation_moment`
  are `NaN` there. A masked window is not "bad data"; it is no data.
- `result.outliers` — windows that produced a measurement which then
  **failed validation**. When replacement is active, `u`/`v` at these
  positions hold the local-median replacement.

A window is never both: masked cells are excluded from validation entirely.

## How masked pixels enter the correlation

A window whose masked-pixel fraction reaches `mask_threshold` (default 0.5)
is dropped. Windows *below* the threshold are still correlated, over their
valid pixels only: masked pixels are loaded at the mean of the window's
valid pixels, which is zero after the correlator's mean subtraction. This
choice means the mask edge introduces **no intensity step** — a hard zero
would act like a bright edge and bias the correlation peak toward zero
displacement.

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
(`dw1.mask .| dw2.mask .| user`), so both cameras are analyzed on identical
grids restricted to the stereo overlap region.
